//! _hashlib module — the OpenSSL-backed digest surface `hashlib.py` probes.
//!
//! Digest state is object-owned, matching PyPy/CPython's per-HASH context.
//! The algorithm implementations live in `pyre-native` (outside LLBC
//! extraction); `_HashState` embeds their fixed-size opaque state buffer in
//! the GC payload, so there is no TLS, global registry, or semantic side
//! table. `HASH` and its `HASHXOF` subclass are native immutable types whose
//! methods operate directly on that incremental state.

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

const HASH_STATE_WORDS: usize = 64;

#[repr(C)]
struct HashStateStorage {
    words: [usize; HASH_STATE_WORDS + 1],
}

impl HashStateStorage {
    fn as_ptr(&self) -> *const usize {
        let address = self.words.as_ptr() as usize;
        ((address + 15) & !15) as *const usize
    }

    fn as_mut_ptr(&mut self) -> *mut usize {
        self.as_ptr() as *mut usize
    }
}

/// The fixed-size native digest context embedded directly in its Python
/// owner. CPython stores an `EVP_MD_CTX *` on `EVPobject`; this inline opaque
/// buffer is pyre-native's allocation-free equivalent.
// CPython 3.14 Modules/_hashopenssl.c creates HASH from a heap type spec;
// HASH_spec carries IMMUTABLETYPE.
#[crate::pyre_class("_hashlib.HASH", cpython_heaptype)]
pub struct W_HashState {
    name: PyObjectRef,
    digest_size: i64,
    storage: HashStateStorage,
}

impl W_HashState {
    fn new(name: &'static str, data: &[u8]) -> Result<PyObjectRef, crate::PyError> {
        assert_eq!(
            HASH_STATE_WORDS,
            pyre_native::hash::HASH_STATE_STORAGE_WORDS
        );
        assert_eq!(16, pyre_native::hash::HASH_STATE_STORAGE_ALIGN);
        let _roots = gc_roots::push_roots();
        let name_slot = gc_roots::shadow_stack_len();
        gc_roots::pin_root(w_str_new(name));
        let state = W_HashState::allocate_stable(W_HashState {
            ob: PyObject::default(),
            name: gc_roots::shadow_stack_get(name_slot),
            digest_size: pyre_native::hash::digest_output_size(name).unwrap_or(0) as i64,
            storage: HashStateStorage {
                words: [0; HASH_STATE_WORDS + 1],
            },
        });
        let this = W_HashState::from_obj(state).expect("fresh _HashState");
        assert!(unsafe {
            pyre_native::hash::state_init(this.storage.as_mut_ptr(), HASH_STATE_WORDS, name)
        });
        if !data.is_empty() {
            unsafe {
                pyre_native::hash::state_update(this.storage.as_mut_ptr(), HASH_STATE_WORDS, data);
            }
        }
        if matches!(name, "shake_128" | "shake_256") {
            unsafe { (*state).w_class = hash_xof_type() };
        }
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_blake2(
        name: &'static str,
        data: &[u8],
        digest_size: usize,
        key: &[u8],
        salt: &[u8],
        person: &[u8],
        fanout: u8,
        depth: u8,
        leaf_size: u32,
        node_offset: u64,
        node_depth: u8,
        inner_size: usize,
        last_node: bool,
    ) -> Result<PyObjectRef, crate::PyError> {
        let _roots = gc_roots::push_roots();
        let name_slot = gc_roots::shadow_stack_len();
        gc_roots::pin_root(w_str_new(name));
        let state = W_HashState::allocate_stable(W_HashState {
            ob: PyObject::default(),
            name: gc_roots::shadow_stack_get(name_slot),
            digest_size: digest_size as i64,
            storage: HashStateStorage {
                words: [0; HASH_STATE_WORDS + 1],
            },
        });
        let this = W_HashState::from_obj(state).expect("fresh BLAKE2 state");
        assert!(unsafe {
            pyre_native::hash::state_init_blake2(
                this.storage.as_mut_ptr(),
                HASH_STATE_WORDS,
                name,
                digest_size,
                key,
                salt,
                person,
                fanout,
                depth,
                leaf_size,
                node_offset,
                node_depth,
                inner_size,
                last_node,
            )
        });
        if !data.is_empty() {
            unsafe {
                pyre_native::hash::state_update(this.storage.as_mut_ptr(), HASH_STATE_WORDS, data);
            }
        }
        Ok(state)
    }

    fn canonical_name(&self) -> &str {
        unsafe { w_str_get_value(self.name) }
    }

    fn digest_bytes(&self, method: &str, args: &[PyObjectRef]) -> Result<Vec<u8>, crate::PyError> {
        let name = self.canonical_name();
        let length = if matches!(name, "shake_128" | "shake_256") {
            if args.len() != 1 {
                return Err(crate::PyError::type_error(format!(
                    "{method}() missing required argument 'length' (pos 1)"
                )));
            }
            let length = crate::baseobjspace::int_w(crate::baseobjspace::space_index(args[0])?)?;
            usize::try_from(length)
                .map_err(|_| crate::PyError::value_error("length must be non-negative"))?
        } else {
            if !args.is_empty() {
                return Err(crate::PyError::type_error(format!(
                    "{method}() takes no arguments ({} given)",
                    args.len()
                )));
            }
            0
        };
        Ok(unsafe {
            pyre_native::hash::state_digest(self.storage.as_ptr(), HASH_STATE_WORDS, length)
        })
    }
}

/// Sweep-time cleanup for the placement-initialized native digest context.
/// The Python name is a traced raw reference and has no Rust drop glue.
pub unsafe fn w_hash_state_dealloc(obj: PyObjectRef) {
    let this = unsafe { &mut *(obj as *mut W_HashState) };
    unsafe {
        pyre_native::hash::state_drop(this.storage.as_mut_ptr(), HASH_STATE_WORDS);
    }
}

mod hash_state_class {
    use super::*;

    #[crate::pyre_methods]
    impl W_HashState {
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            _args: &[PyObjectRef],
        ) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::PyError::type_error(
                "cannot create '_hashlib.HASH' instances",
            ))
        }

        #[getter]
        fn name(&self) -> PyObjectRef {
            self.name
        }

        #[getter]
        fn digest_size(&self) -> i64 {
            self.digest_size
        }

        #[getter]
        fn block_size(&self) -> i64 {
            match self.canonical_name() {
                "md5" | "sha1" | "sha224" | "sha256" | "blake2s" => 64,
                "sha384" | "sha512" | "blake2b" => 128,
                "sha3_224" => 144,
                "sha3_256" | "shake_256" => 136,
                "sha3_384" => 104,
                "sha3_512" => 72,
                "shake_128" => 168,
                _ => 0,
            }
        }

        fn update(&mut self, data: PyObjectRef) -> Result<(), crate::PyError> {
            let data = read_hash_buffer(data)?;
            unsafe {
                pyre_native::hash::state_update(self.storage.as_mut_ptr(), HASH_STATE_WORDS, &data);
            }
            Ok(())
        }

        fn digest(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            Ok(w_bytes_from_bytes(
                &self.digest_bytes("digest", &args[1..])?,
            ))
        }

        fn hexdigest(&self, args: &[PyObjectRef]) -> Result<String, crate::PyError> {
            let digest = self.digest_bytes("hexdigest", &args[1..])?;
            let mut out = String::with_capacity(digest.len() * 2);
            for byte in digest {
                use std::fmt::Write;
                let _ = write!(out, "{byte:02x}");
            }
            Ok(out)
        }

        fn copy(&self) -> Result<PyObjectRef, crate::PyError> {
            let _roots = gc_roots::push_roots();
            let name_slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(self.name);
            let clone = W_HashState::allocate_stable(W_HashState {
                ob: PyObject::default(),
                name: gc_roots::shadow_stack_get(name_slot),
                digest_size: self.digest_size,
                storage: HashStateStorage {
                    words: [0; HASH_STATE_WORDS + 1],
                },
            });
            let clone = W_HashState::from_obj(clone).expect("fresh _HashState");
            unsafe {
                pyre_native::hash::state_copy(
                    self.storage.as_ptr(),
                    clone.storage.as_mut_ptr(),
                    HASH_STATE_WORDS,
                );
                // PyPy `_hashlib.HASHXOF(HASH)` and CPython EVP_copy preserve
                // the concrete HASH/HASHXOF class. Both share the same native
                // payload layout, so carry the live instance's class tag.
                clone.ob.w_class = self.ob.w_class;
            }
            Ok(clone as *mut W_HashState as PyObjectRef)
        }

        fn __repr__(&self) -> String {
            let class_name = if matches!(self.canonical_name(), "shake_128" | "shake_256") {
                "HASHXOF"
            } else {
                "HASH"
            };
            format!(
                "<{} _hashlib.{class_name} object @ {:#x}>",
                self.canonical_name(),
                self as *const Self as usize
            )
        }
    }
}

/// PyPy `lib_pypy/_hashlib/__init__.py`: `class HASHXOF(HASH): pass`.
///
/// The subclass introduces no fields, hence it deliberately reuses HASH's
/// `W_HashState` layout instead of inventing a parallel payload or side table.
#[majit_macros::dont_look_inside]
fn hash_xof_type() -> PyObjectRef {
    static TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_hashlib.HASHXOF",
            |ns| unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__new__",
                    crate::typedef::make_new_descr(|_| {
                        Err(crate::PyError::type_error(
                            "cannot create '_hashlib.HASHXOF' instances",
                        ))
                    }),
                )
            },
            hash_state_class::type_object(),
            <W_HashState as crate::PyreClassPyTypeOf>::PYTYPE,
        );
        // CPython 3.14 Modules/_hashopenssl.c creates HASHXOF from the same
        // immutable module heap family as HASH.
        crate::typedef::mark_cpython_heap_type(tp, true);
        tp as usize
    }) as PyObjectRef
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

/// Build a `PyError` raising `_hashlib.UnsupportedDigestmodError` with `msg`.
/// `hmac.py` catches this to fall back to its pure-Python HMAC when the
/// requested digest is not one of the native streaming implementations.
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

const HMAC_STATE_WORDS: usize = 128;

#[repr(C)]
struct HmacStateStorage {
    words: [usize; HMAC_STATE_WORDS + 1],
}

impl HmacStateStorage {
    fn as_ptr(&self) -> *const usize {
        let address = self.words.as_ptr() as usize;
        ((address + 15) & !15) as *const usize
    }

    fn as_mut_ptr(&mut self) -> *mut usize {
        self.as_ptr() as *mut usize
    }
}

// CPython 3.14 `_hashlib.HMAC` is the same immutable heap-spec family.
#[crate::pyre_class("_hashlib.HMAC", cpython_heaptype)]
pub struct W_Hmac {
    name: PyObjectRef,
    digest_size: i64,
    block_size: i64,
    storage: HmacStateStorage,
}

impl W_Hmac {
    fn new(name: &'static str, key: &[u8], msg: &[u8]) -> Result<PyObjectRef, crate::PyError> {
        assert_eq!(
            HMAC_STATE_WORDS,
            pyre_native::hash::HMAC_STATE_STORAGE_WORDS
        );
        assert_eq!(16, pyre_native::hash::HMAC_STATE_STORAGE_ALIGN);
        let digest_size = pyre_native::hash::digest_output_size(name)
            .ok_or_else(|| unsupported_digestmod("unsupported hash type"))?;
        let block_size = pyre_native::hash::digest_block_size(name)
            .ok_or_else(|| unsupported_digestmod("unsupported hash type"))?;
        let _roots = gc_roots::push_roots();
        let name_slot = gc_roots::shadow_stack_len();
        gc_roots::pin_root(w_str_new(&format!("hmac-{name}")));
        let obj = W_Hmac::allocate_stable(W_Hmac {
            ob: PyObject::default(),
            name: gc_roots::shadow_stack_get(name_slot),
            digest_size: digest_size as i64,
            block_size: block_size as i64,
            storage: HmacStateStorage {
                words: [0; HMAC_STATE_WORDS + 1],
            },
        });
        let this = W_Hmac::from_obj(obj).expect("fresh HMAC");
        assert!(unsafe {
            pyre_native::hash::hmac_state_init(
                this.storage.as_mut_ptr(),
                HMAC_STATE_WORDS,
                name,
                key,
            )
        });
        if !msg.is_empty() {
            unsafe {
                pyre_native::hash::hmac_state_update(
                    this.storage.as_mut_ptr(),
                    HMAC_STATE_WORDS,
                    msg,
                );
            }
        }
        Ok(obj)
    }
}

/// Sweep-time cleanup for the placement-initialized native HMAC context.
pub unsafe fn w_hmac_dealloc(obj: PyObjectRef) {
    let this = unsafe { &mut *(obj as *mut W_Hmac) };
    unsafe {
        pyre_native::hash::hmac_state_drop(this.storage.as_mut_ptr(), HMAC_STATE_WORDS);
    }
}

mod hmac_class {
    use super::*;

    // lib_pypy/_hashlib/__init__.py:232: `class HMAC(HASH)`.  HMAC keeps its
    // own native payload layout, but its Python type relationship is the same
    // TypeDef inheritance PyPy exposes.
    #[crate::pyre_methods(base = hash_state_class::type_object())]
    impl W_Hmac {
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            _args: &[PyObjectRef],
        ) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::PyError::type_error(
                "cannot create '_hashlib.HMAC' instances",
            ))
        }

        #[getter]
        fn name(&self) -> PyObjectRef {
            self.name
        }

        #[getter]
        fn digest_size(&self) -> i64 {
            self.digest_size
        }

        #[getter]
        fn block_size(&self) -> i64 {
            self.block_size
        }

        fn update(&mut self, msg: PyObjectRef) -> Result<(), crate::PyError> {
            let msg = read_hash_buffer(msg)?;
            unsafe {
                pyre_native::hash::hmac_state_update(
                    self.storage.as_mut_ptr(),
                    HMAC_STATE_WORDS,
                    &msg,
                );
            }
            Ok(())
        }

        fn digest(&self) -> PyObjectRef {
            let digest = unsafe {
                pyre_native::hash::hmac_state_digest(self.storage.as_ptr(), HMAC_STATE_WORDS)
            };
            w_bytes_from_bytes(&digest)
        }

        fn hexdigest(&self) -> String {
            let digest = unsafe {
                pyre_native::hash::hmac_state_digest(self.storage.as_ptr(), HMAC_STATE_WORDS)
            };
            let mut out = String::with_capacity(digest.len() * 2);
            for byte in digest {
                use std::fmt::Write;
                let _ = write!(out, "{byte:02x}");
            }
            out
        }

        fn copy(&self) -> Result<PyObjectRef, crate::PyError> {
            let _roots = gc_roots::push_roots();
            let name_slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(self.name);
            let obj = W_Hmac::allocate_stable(W_Hmac {
                ob: PyObject::default(),
                name: gc_roots::shadow_stack_get(name_slot),
                digest_size: self.digest_size,
                block_size: self.block_size,
                storage: HmacStateStorage {
                    words: [0; HMAC_STATE_WORDS + 1],
                },
            });
            let clone = W_Hmac::from_obj(obj).expect("fresh HMAC");
            unsafe {
                pyre_native::hash::hmac_state_copy(
                    self.storage.as_ptr(),
                    clone.storage.as_mut_ptr(),
                    HMAC_STATE_WORDS,
                );
            }
            Ok(obj)
        }

        fn __repr__(&self) -> String {
            let name = unsafe { w_str_get_value(self.name) };
            let name = name.strip_prefix("hmac-").unwrap_or(name);
            format!("<{name} HMAC object @ {:#x}>", self as *const Self as usize)
        }
    }
}

fn resolve_hmac_digestmod(digestmod: PyObjectRef) -> Result<&'static str, crate::PyError> {
    let name_obj = if unsafe { is_str(digestmod) } {
        digestmod
    } else {
        match crate::baseobjspace::getattr_str(digestmod, "__name__") {
            Ok(name) => name,
            // PyPy's structural rule is to accept every object that exposes
            // `__name__` (lib_pypy/_hashlib/__init__.py:547-554).  CPython
            // 3.14 additionally normalizes a missing name to the module's
            // public UnsupportedDigestmodError; preserve both without
            // restricting the accepted object type again.
            Err(err) if err.kind == crate::PyErrorKind::AttributeError => {
                return Err(unsupported_digestmod("unsupported hash type"));
            }
            Err(err) => return Err(err),
        }
    };
    if !unsafe { is_str(name_obj) } {
        return Err(unsupported_digestmod("unsupported hash type"));
    }
    let name = unsafe { w_str_get_wtf8(name_obj) };
    let bytes = name.as_bytes();
    let bytes = bytes.strip_prefix(b"openssl_").unwrap_or(bytes);
    lookup_digest_name(bytes).ok_or_else(|| unsupported_digestmod("unsupported hash type"))
}

fn hmac_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        "hmac_new",
        positional.len(),
        crate::builtins::real_kwarg_count(kwargs),
        1,
        3,
        0,
    )?;
    let key = crate::builtins::bind_pos_or_kw(positional, kwargs, 0, "key", "hmac_new", 1)?
        .ok_or_else(|| crate::PyError::type_error("hmac_new() missing required argument 'key'"))?;
    let msg = crate::builtins::bind_pos_or_kw(positional, kwargs, 1, "msg", "hmac_new", 2)?;
    let digestmod =
        crate::builtins::bind_pos_or_kw(positional, kwargs, 2, "digestmod", "hmac_new", 3)?
            .ok_or_else(|| {
                crate::PyError::type_error(
                    "hmac_new() missing required argument 'digestmod' (pos 3)",
                )
            })?;
    crate::builtins::kwarg_reject_unknown(kwargs, &["key", "msg", "digestmod"], "hmac_new")?;
    let key = read_hash_buffer(key)?;
    let msg = match msg {
        Some(msg) if unsafe { !is_none(msg) } => read_hash_buffer(msg)?,
        _ => Vec::new(),
    };
    W_Hmac::new(resolve_hmac_digestmod(digestmod)?, &key, &msg)
}

fn hmac_digest(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        "hmac_digest",
        positional.len(),
        crate::builtins::real_kwarg_count(kwargs),
        3,
        3,
        0,
    )?;
    let key = crate::builtins::bind_pos_or_kw(positional, kwargs, 0, "key", "hmac_digest", 1)?
        .ok_or_else(|| {
            crate::PyError::type_error("hmac_digest() missing required argument 'key'")
        })?;
    let msg = crate::builtins::bind_pos_or_kw(positional, kwargs, 1, "msg", "hmac_digest", 2)?
        .ok_or_else(|| {
            crate::PyError::type_error("hmac_digest() missing required argument 'msg'")
        })?;
    let digestmod =
        crate::builtins::bind_pos_or_kw(positional, kwargs, 2, "digest", "hmac_digest", 3)?
            .ok_or_else(|| {
                crate::PyError::type_error(
                    "hmac_digest() missing required argument 'digest' (pos 3)",
                )
            })?;
    crate::builtins::kwarg_reject_unknown(kwargs, &["key", "msg", "digest"], "hmac_digest")?;
    let key = read_hash_buffer(key)?;
    let msg = read_hash_buffer(msg)?;
    let state = W_Hmac::new(resolve_hmac_digestmod(digestmod)?, &key, &msg)?;
    let state = W_Hmac::from_obj(state).expect("fresh HMAC");
    let digest =
        unsafe { pyre_native::hash::hmac_state_digest(state.storage.as_ptr(), HMAC_STATE_WORDS) };
    Ok(w_bytes_from_bytes(&digest))
}

fn pbkdf2_hmac(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    const KEYWORDS: &[&str] = &["hash_name", "password", "salt", "iterations", "dklen"];
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        "pbkdf2_hmac",
        positional.len(),
        crate::builtins::real_kwarg_count(kwargs),
        4,
        5,
        0,
    )?;
    let required = |index, name, position| {
        crate::builtins::bind_pos_or_kw(positional, kwargs, index, name, "pbkdf2_hmac", position)?
            .ok_or_else(|| {
                crate::PyError::type_error(format!(
                    "pbkdf2_hmac() missing required argument '{name}' (pos {position})"
                ))
            })
    };
    let hash_name = required(0, "hash_name", 1)?;
    let password = required(1, "password", 2)?;
    let salt = required(2, "salt", 3)?;
    let iterations = required(3, "iterations", 4)?;
    let dklen = crate::builtins::bind_pos_or_kw(positional, kwargs, 4, "dklen", "pbkdf2_hmac", 5)?;
    crate::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "pbkdf2_hmac")?;

    check_digest_name(hash_name)?;
    let requested = unsafe { w_str_get_wtf8(hash_name) };
    let name = lookup_digest_name(requested.as_bytes())
        .ok_or_else(|| unsupported_digestmod("unsupported hash type"))?;
    let password = read_hash_buffer(password)?;
    let salt = read_hash_buffer(salt)?;
    let iterations = crate::baseobjspace::int_w(crate::baseobjspace::space_index(iterations)?)?;
    let iterations = usize::try_from(iterations)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| crate::PyError::value_error("iteration value must be greater than 0"))?;
    let dklen = match dklen {
        Some(obj) if unsafe { !is_none(obj) } => {
            let value = crate::baseobjspace::int_w(crate::baseobjspace::space_index(obj)?)?;
            usize::try_from(value)
                .ok()
                .filter(|&value| value > 0)
                .ok_or_else(|| crate::PyError::value_error("key length must be greater than 0"))?
        }
        _ => pyre_native::hash::digest_output_size(name).unwrap_or(0),
    };
    let result = pyre_native::hash::compute_pbkdf2_hmac(name, &password, &salt, iterations, dklen)
        .ok_or_else(|| unsupported_digestmod("unsupported hash type"))?;
    Ok(w_bytes_from_bytes(&result))
}

/// `scrypt(password, *, salt, n, r, p, maxmem=0, dklen=64)`.
///
/// CPython `_hashopenssl.c` takes only `password` positionally and sends the
/// work to `EVP_PBE_scrypt`.  The pure-Rust backend implements the same RFC
/// 7914 primitive; validation stays here so Python-visible errors and integer
/// bounds do not depend on the backend crate's narrower parameter types.
fn scrypt_kdf(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    const KEYWORDS: &[&str] = &["salt", "n", "r", "p", "maxmem", "dklen"];
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        "scrypt",
        positional.len(),
        crate::builtins::real_kwarg_count(kwargs),
        1,
        1,
        6,
    )?;
    let password = positional.first().copied().ok_or_else(|| {
        crate::PyError::type_error("scrypt() missing required argument 'password' (pos 1)")
    })?;
    crate::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "scrypt")?;
    let required = |name: &str| {
        crate::builtins::kwarg_get(kwargs, name).ok_or_else(|| {
            crate::PyError::type_error(format!(
                "scrypt() missing required keyword-only argument '{name}'"
            ))
        })
    };
    let salt = required("salt")?;
    let n_obj = required("n")?;
    let r_obj = required("r")?;
    let p_obj = required("p")?;

    let password = read_hash_buffer(password)?;
    let salt = read_hash_buffer(salt)?;
    let index = |obj| crate::baseobjspace::int_w(crate::baseobjspace::space_index(obj)?);
    let n = index(n_obj)?;
    let r = index(r_obj)?;
    let p = index(p_obj)?;
    let maxmem = match crate::builtins::kwarg_get(kwargs, "maxmem") {
        Some(obj) => index(obj)?,
        None => 0,
    };
    let dklen = match crate::builtins::kwarg_get(kwargs, "dklen") {
        Some(obj) => index(obj)?,
        None => 64,
    };

    let n = u64::try_from(n).unwrap_or(0);
    if n < 2 || !n.is_power_of_two() {
        return Err(crate::PyError::value_error(
            "n must be a power of 2 greater than 1",
        ));
    }
    let log_n = u8::try_from(n.trailing_zeros()).map_err(|_| {
        crate::PyError::value_error("Invalid parameter combination for n, r, p, maxmem")
    })?;
    let r = u32::try_from(r)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| {
            crate::PyError::value_error("Invalid parameter combination for n, r, p, maxmem")
        })?;
    let p = u32::try_from(p)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| {
            crate::PyError::value_error("Invalid parameter combination for n, r, p, maxmem")
        })?;
    let maxmem = usize::try_from(maxmem).map_err(|_| {
        crate::PyError::value_error("maxmem must be positive and smaller than 2147483647")
    })?;
    let dklen = usize::try_from(dklen)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| {
            crate::PyError::value_error("dklen must be greater than 0 and smaller than 2147483647")
        })?;
    if maxmem > i32::MAX as usize {
        return Err(crate::PyError::value_error(
            "maxmem must be positive and smaller than 2147483647",
        ));
    }
    if dklen > i32::MAX as usize {
        return Err(crate::PyError::value_error(
            "dklen must be greater than 0 and smaller than 2147483647",
        ));
    }

    // RFC 7914's dominant allocation is V[N] with 128*r-byte entries. OpenSSL
    // also needs B[p] and a working block.  PyPy passes maxmem=0 through to
    // EVP_PBE_scrypt (lib_pypy/_hashlib/__init__.py:430-433), where OpenSSL
    // applies its private 32 MiB default.  The Rust backend has no such layer,
    // so spell out that same default here rather than treating zero as
    // unlimited memory.
    let memory = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(r as usize))
        .and_then(|v| v.checked_mul(128))
        .and_then(|v| v.checked_add((p as usize).checked_mul(r as usize)?.checked_mul(128)?))
        .and_then(|v| v.checked_add((r as usize).checked_mul(256)?))
        .ok_or_else(|| {
            crate::PyError::value_error("Invalid parameter combination for n, r, p, maxmem")
        })?;
    const OPENSSL_DEFAULT_SCRYPT_MAXMEM: usize = 32 * 1024 * 1024;
    let effective_maxmem = if maxmem == 0 {
        OPENSSL_DEFAULT_SCRYPT_MAXMEM
    } else {
        maxmem
    };
    if memory > effective_maxmem {
        return Err(crate::PyError::value_error(
            "[digital envelope routines] memory limit exceeded",
        ));
    }
    let output = pyre_native::hash::compute_scrypt(&password, &salt, log_n, r, p, dklen)
        .ok_or_else(|| {
            crate::PyError::value_error("Invalid parameter combination for n, r, p, maxmem")
        })?;
    Ok(w_bytes_from_bytes(&output))
}

/// Private constructor used by PyPy-shaped `_blake2` app-level objects after
/// they validate the public clinic contract. The returned HASH is the native,
/// object-owned context held by the wrapper's `_state` attribute.
fn blake2_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::builtins::clinic_arity("_blake2_new", args.len(), 0, 13, 13, 0)?;
    let _roots = gc_roots::push_roots();
    let base = gc_roots::shadow_stack_len();
    for &arg in args {
        gc_roots::pin_root(arg);
    }
    let arg = |index| gc_roots::shadow_stack_get(base + index);
    let name_obj = arg(0);
    check_digest_name(name_obj)?;
    let requested = unsafe { w_str_get_wtf8(name_obj) };
    let name = match requested.as_bytes() {
        b"blake2b" => "blake2b",
        b"blake2s" => "blake2s",
        _ => return Err(crate::PyError::value_error("unsupported BLAKE2 type")),
    };
    let data = read_hash_buffer(arg(1))?;
    let key = read_hash_buffer(arg(3))?;
    let salt = read_hash_buffer(arg(4))?;
    let person = read_hash_buffer(arg(5))?;
    let (max_key_size, salt_size, person_size) = match name {
        "blake2b" => (64, 16, 16),
        "blake2s" => (32, 8, 8),
        _ => unreachable!(),
    };
    if key.len() > max_key_size {
        return Err(crate::PyError::value_error(format!(
            "maximum key length is {max_key_size} bytes"
        )));
    }
    if salt.len() > salt_size {
        return Err(crate::PyError::value_error(format!(
            "maximum salt length is {salt_size} bytes"
        )));
    }
    if person.len() > person_size {
        return Err(crate::PyError::value_error(format!(
            "maximum person length is {person_size} bytes"
        )));
    }
    let index =
        |position| crate::baseobjspace::int_w(crate::baseobjspace::space_index(arg(position))?);
    let digest_size = usize::try_from(index(2)?)
        .map_err(|_| crate::PyError::value_error("invalid digest size"))?;
    let fanout =
        u8::try_from(index(6)?).map_err(|_| crate::PyError::value_error("invalid fanout"))?;
    let depth =
        u8::try_from(index(7)?).map_err(|_| crate::PyError::value_error("invalid depth"))?;
    let leaf_size =
        u32::try_from(index(8)?).map_err(|_| crate::PyError::value_error("invalid leaf size"))?;
    // BLAKE2b accepts the full unsigned 64-bit range; `int_w` would reject
    // values above i64::MAX even though the app-level range check accepted
    // them. operator.index has already normalized the public argument.
    let node_offset = crate::baseobjspace::uint_w(arg(9))?;
    let node_depth =
        u8::try_from(index(10)?).map_err(|_| crate::PyError::value_error("invalid node depth"))?;
    let inner_size = usize::try_from(index(11)?)
        .map_err(|_| crate::PyError::value_error("invalid inner size"))?;
    let last_node = crate::baseobjspace::is_true(arg(12))?;
    W_HashState::new_blake2(
        name,
        &data,
        digest_size,
        &key,
        &salt,
        &person,
        fanout,
        depth,
        leaf_size,
        node_offset,
        node_depth,
        inner_size,
        last_node,
    )
}

/// Constant-time equality of two ASCII strings or two bytes-like objects.
fn compare_digest(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let a_obj = args.first().copied().unwrap_or_else(w_none);
    let b_obj = args.get(1).copied().unwrap_or_else(w_none);
    if unsafe { is_str(a_obj) } != unsafe { is_str(b_obj) } {
        return Err(crate::PyError::type_error(
            "unsupported operand types(s) or combination of types",
        ));
    }
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
    let a = read(a_obj)?;
    let b = read(b_obj)?;
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
    let name_obj = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let name = unsafe { w_str_get_wtf8(name_obj) };
    let name = lookup_digest_name(name.as_bytes())
        .ok_or_else(|| unsupported_digestmod(&format!("unsupported hash type {name}")))?;
    W_HashState::new(name, &data_bytes)
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
        "HASH" => hash_state_class::type_object(),
        "HASHXOF" => hash_xof_type(),
        "HMAC" => hmac_class::type_object(),
        "_GIL_MINSIZE" => w_int_new(2048),
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
        "compare_digest" / 2 = compare_digest,
        "hmac_new" / * = hmac_new,
        "hmac_digest" / * = hmac_digest,
        "pbkdf2_hmac" / * = pbkdf2_hmac,
        "scrypt" / * = scrypt_kdf,
        "_blake2_new" / * = blake2_new,
        "get_fips_mode" / 0 = |_| Ok(w_int_new(0)),
    },
    extra_init: |ns| {
        let _roots = gc_roots::push_roots();
        let mapping_slot = gc_roots::shadow_stack_len();
        gc_roots::pin_root(w_dict_new());
        for (constructor, name) in [
            ("openssl_md5", "md5"),
            ("openssl_sha1", "sha1"),
            ("openssl_sha224", "sha224"),
            ("openssl_sha256", "sha256"),
            ("openssl_sha384", "sha384"),
            ("openssl_sha512", "sha512"),
            ("openssl_sha3_224", "sha3_224"),
            ("openssl_sha3_256", "sha3_256"),
            ("openssl_sha3_384", "sha3_384"),
            ("openssl_sha3_512", "sha3_512"),
            ("openssl_shake_128", "shake_128"),
            ("openssl_shake_256", "shake_256"),
        ] {
            let function = crate::module_ns_get(ns, constructor)
                .expect("_hashlib constructor installed before extra_init");
            let function_slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(function);
            let name_slot = gc_roots::shadow_stack_len();
            gc_roots::pin_root(w_str_new(name));
            crate::baseobjspace::setitem(
                gc_roots::shadow_stack_get(mapping_slot),
                gc_roots::shadow_stack_get(function_slot),
                gc_roots::shadow_stack_get(name_slot),
            )
            .expect("populate _hashlib._constructors");
        }
        crate::module_ns_store(
            ns,
            "_constructors",
            pyre_object::w_dict_proxy_new(gc_roots::shadow_stack_get(mapping_slot)),
        );
    },
}
