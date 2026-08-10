//! zlib module — PyPy: `pypy/module/zlib/`.
//!
//! CRC-32 / Adler-32 checksums plus the DEFLATE compress/decompress surface.
//! The DEFLATE machinery is a deliberate duplication of RustPython's zlib
//! implementation, ported into `pyre_native::zlib` (flate2 / zlib-rs) and kept
//! outside the LLBC extraction; this module is the W_Root object glue.
//!
//! `Compress` / `Decompress` / `_ZlibDecompressor` hold flate2 streaming state
//! that cannot live in the Python dict, so it is parked in process-global
//! registries keyed by an id stashed in each instance dict (`_id`). A streamer
//! dropped by GC leaks its (post-finish, buffer-freed) registry entry; the
//! heavy flate2 buffers are released by the backend at finish/eof.

use pyre_native::zlib as backend;
use pyre_object::*;

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// ── streaming-state registries ──────────────────────────────────────────

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static COMPRESSORS: Mutex<BTreeMap<u64, backend::Compressor>> = Mutex::new(BTreeMap::new());
static DECOMPRESSORS: Mutex<BTreeMap<u64, backend::Decompressor>> = Mutex::new(BTreeMap::new());
static ZDECOMPRESSORS: Mutex<BTreeMap<u64, backend::ZlibDecompressor>> =
    Mutex::new(BTreeMap::new());

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn get_id(obj: PyObjectRef) -> u64 {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return 0;
    }
    match unsafe { w_dict_getitem_str(d, "_id") } {
        Some(v) if unsafe { is_int(v) } => (unsafe { w_int_get_value(v) }) as u64,
        _ => 0,
    }
}

fn set_id(obj: PyObjectRef, id: u64) {
    let d = crate::baseobjspace::getdict_native(obj);
    if !d.is_null() {
        unsafe { w_dict_setitem_str(d, "_id", w_int_new(id as i64)) };
    }
}

// ── errors ──────────────────────────────────────────────────────────────

fn zlib_error(msg: impl Into<String>) -> crate::PyError {
    let msg = msg.into();
    let mut err = crate::PyError::value_error(msg.clone());
    if let Some(cls) = crate::builtins::lookup_exc_class("zlib.error") {
        let args = [cls, w_str_new(&msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            err.exc_object = exc;
        }
    }
    err
}

fn eof_error(msg: &str) -> crate::PyError {
    crate::PyError::new(crate::PyErrorKind::EOFError, msg)
}

// ── argument helpers ────────────────────────────────────────────────────

fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    unsafe {
        if bytesobject::is_bytes_like(obj) {
            Ok(bytesobject::bytes_like_data(obj).to_vec())
        } else {
            Err(crate::PyError::type_error(format!(
                "a bytes-like object is required, not '{}'",
                crate::baseobjspace::object_functionstr_type_name(obj)
            )))
        }
    }
}

/// Coerce an argument slot to an integer, `None`/absent (`PY_NULL`) → default.
///
/// The Signature-bound call path fills every declared slot: a supplied
/// keyword / positional carries the value, an omitted optional carries
/// `PY_NULL`.
fn int_or_default(o: PyObjectRef, default: i64) -> Result<i64, crate::PyError> {
    if o.is_null() || unsafe { is_none(o) } {
        Ok(default)
    } else {
        crate::baseobjspace::int_w(o)
    }
}

/// Coerce an optional `zdict` slot to bytes, `None`/absent (`PY_NULL`) → None.
fn zdict_or_none(o: PyObjectRef) -> Result<Option<Vec<u8>>, crate::PyError> {
    if o.is_null() || unsafe { is_none(o) } {
        Ok(None)
    } else {
        Ok(Some(as_bytes(o)?))
    }
}

fn to_wbits(v: i64) -> i8 {
    v as i8
}

// ── checksums ───────────────────────────────────────────────────────────

fn crc32_compute(buf: &[u8], start: u32) -> u32 {
    let mut crc = !start;
    for &b in buf {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn adler32_compute(buf: &[u8], start: u32) -> u32 {
    let mut a = start & 0xffff;
    let mut b = (start >> 16) & 0xffff;
    for &x in buf {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

// ── Compress (compressobj) ──────────────────────────────────────────────

static COMPRESS_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static DECOMPRESS_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static ZDECOMPRESS_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn compress_type() -> PyObjectRef {
    *COMPRESS_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("Compress", init_compress_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

fn init_compress_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "compress",
            crate::make_builtin_function_with_arity(
                "compress",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("compress() missing data"));
                    }
                    let id = get_id(args[0]);
                    let data = as_bytes(args[1])?;
                    let mut reg = COMPRESSORS.lock().unwrap();
                    let c = reg
                        .get_mut(&id)
                        .ok_or_else(|| zlib_error("Error -2: inconsistent stream state"))?;
                    let out = c.compress(&data).map_err(zlib_error)?;
                    Ok(bytesobject::w_bytes_from_bytes(&out))
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "flush",
            crate::make_builtin_function("flush", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("flush() missing self"));
                }
                let id = get_id(args[0]);
                // interp_zlib.py:196 `@unwrap_spec(mode="c_int")` — the
                // converter reports a value outside the C `int` range rather
                // than truncating it into a different flush mode.
                let mode = match args.get(1).copied() {
                    Some(o) if !unsafe { is_none(o) } => crate::baseobjspace::c_int_w(o)?,
                    _ => backend::Z_FINISH,
                };
                let mut reg = COMPRESSORS.lock().unwrap();
                let c = reg
                    .get_mut(&id)
                    .ok_or_else(|| zlib_error("Error -2: inconsistent stream state"))?;
                let out = c.flush(mode).map_err(zlib_error)?;
                Ok(bytesobject::w_bytes_from_bytes(&out))
            }),
        )
    };
}

fn make_compress(
    level: i32,
    wbits: i8,
    zdict: Option<Vec<u8>>,
) -> Result<PyObjectRef, crate::PyError> {
    let c = backend::Compressor::new(level, wbits, zdict.as_deref()).map_err(zlib_error)?;
    let id = next_id();
    COMPRESSORS.lock().unwrap().insert(id, c);
    let obj = w_instance_new(compress_type());
    set_id(obj, id);
    Ok(obj)
}

// ── Decompress (decompressobj) ──────────────────────────────────────────

fn decompress_type() -> PyObjectRef {
    *DECOMPRESS_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("Decompress", init_decompress_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

fn decompress_getset(ns: PyObjectRef, name: &'static str, f: crate::gateway::BuiltinCodeFn) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            name,
            crate::typedef::make_getset_descriptor_named(
                crate::make_builtin_function_with_arity(name, f, 2),
                name,
            ),
        )
    };
}

fn init_decompress_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "decompress",
            crate::make_builtin_function("decompress", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("decompress() missing data"));
                }
                let id = get_id(args[0]);
                let data = as_bytes(args[1])?;
                // An omitted `max_length` is unlimited; a supplied value —
                // including `None` — goes through `int_w`, which raises for
                // `None`.  Zero also means unlimited here, and a negative
                // value is rejected.
                let max_length = match args.get(2).copied() {
                    Some(o) => {
                        let v = crate::baseobjspace::int_w(o)?;
                        if v < 0 {
                            return Err(crate::PyError::value_error(
                                "max_length must be non-negative",
                            ));
                        }
                        (v != 0).then_some(v as usize)
                    }
                    None => None,
                };
                let mut reg = DECOMPRESSORS.lock().unwrap();
                let d = reg
                    .get_mut(&id)
                    .ok_or_else(|| zlib_error("Error -2: inconsistent stream state"))?;
                let out = d.decompress(&data, max_length).map_err(zlib_error)?;
                Ok(bytesobject::w_bytes_from_bytes(&out))
            }),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "flush",
            crate::make_builtin_function("flush", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("flush() missing self"));
                }
                let id = get_id(args[0]);
                let length = match args.get(1).copied() {
                    Some(o) if !unsafe { is_none(o) } => {
                        let v = crate::baseobjspace::int_w(o)?;
                        if v <= 0 {
                            return Err(crate::PyError::value_error(
                                "length must be greater than zero",
                            ));
                        }
                        v as usize
                    }
                    _ => backend::DEF_BUF_SIZE,
                };
                let mut reg = DECOMPRESSORS.lock().unwrap();
                let d = reg
                    .get_mut(&id)
                    .ok_or_else(|| zlib_error("Error -2: inconsistent stream state"))?;
                let out = d.flush(length).map_err(zlib_error)?;
                Ok(bytesobject::w_bytes_from_bytes(&out))
            }),
        )
    };
    decompress_getset(ns, "unused_data", |args| {
        let id = get_id(args.get(1).copied().unwrap_or(PY_NULL));
        let reg = DECOMPRESSORS.lock().unwrap();
        let data = reg
            .get(&id)
            .map(|d| d.unused_data().to_vec())
            .unwrap_or_default();
        Ok(bytesobject::w_bytes_from_bytes(&data))
    });
    decompress_getset(ns, "unconsumed_tail", |args| {
        let id = get_id(args.get(1).copied().unwrap_or(PY_NULL));
        let reg = DECOMPRESSORS.lock().unwrap();
        let data = reg
            .get(&id)
            .map(|d| d.unconsumed_tail().to_vec())
            .unwrap_or_default();
        Ok(bytesobject::w_bytes_from_bytes(&data))
    });
    decompress_getset(ns, "eof", |args| {
        let id = get_id(args.get(1).copied().unwrap_or(PY_NULL));
        let reg = DECOMPRESSORS.lock().unwrap();
        Ok(w_bool_from(reg.get(&id).map(|d| d.eof()).unwrap_or(false)))
    });
}

fn make_decompress(wbits: i8, zdict: Option<Vec<u8>>) -> Result<PyObjectRef, crate::PyError> {
    let d = backend::Decompressor::new(wbits, zdict).map_err(zlib_error)?;
    let id = next_id();
    DECOMPRESSORS.lock().unwrap().insert(id, d);
    let obj = w_instance_new(decompress_type());
    set_id(obj, id);
    Ok(obj)
}

// ── _ZlibDecompressor (buffered; used by gzip reading) ──────────────────

fn zdecompress_type() -> PyObjectRef {
    *ZDECOMPRESS_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_ZlibDecompressor", init_zdecompress_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

fn zdecompress_getset(ns: PyObjectRef, name: &'static str, f: crate::gateway::BuiltinCodeFn) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            name,
            crate::typedef::make_getset_descriptor_named(
                crate::make_builtin_function_with_arity(name, f, 2),
                name,
            ),
        )
    };
}

// _ZlibDecompressor(wbits=MAX_WBITS, zdict=b'') — the DecompressReader factory
// gzip calls with wbits=-MAX_WBITS.  `cls` positional-only, `wbits`/`zdict`
// positional-or-keyword; the Signature-bound call path fills omitted optionals
// with PY_NULL.
fn zdecompress_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] is the type; args[1..] are the constructor arguments.
    let wbits = to_wbits(int_or_default(
        args.get(1).copied().unwrap_or(PY_NULL),
        backend::MAX_WBITS as i64,
    )?);
    let zdict = zdict_or_none(args.get(2).copied().unwrap_or(PY_NULL))?;
    let d = backend::ZlibDecompressor::new(wbits, zdict).map_err(zlib_error)?;
    let id = next_id();
    ZDECOMPRESSORS.lock().unwrap().insert(id, d);
    let obj = w_instance_new(zdecompress_type());
    set_id(obj, id);
    Ok(obj)
}

// `_ZlibDecompressor.decompress(self, /, data, max_length=-1)` — `self`
// positional-only, `data` positional-or-keyword.
fn zdecompress_decompress(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data_obj = args.get(1).copied().unwrap_or(PY_NULL);
    if data_obj.is_null() {
        return Err(crate::PyError::type_error("decompress() missing data"));
    }
    let data = as_bytes(data_obj)?;
    // `max_length=-1` (the default) means unlimited; an omitted slot behaves
    // the same.  Only PY_NULL selects that default: a supplied value —
    // including `None` — goes through `int_w`, which raises for `None` and on
    // ssize_t overflow.  A negative value is unlimited, zero caps the output
    // at zero bytes.
    let max_length = match args.get(2).copied() {
        Some(o) if !o.is_null() => {
            let v = crate::baseobjspace::int_w(o)?;
            (v >= 0).then_some(v as usize)
        }
        _ => None,
    };
    let id = get_id(args[0]);
    let mut reg = ZDECOMPRESSORS.lock().unwrap();
    let d = reg
        .get_mut(&id)
        .ok_or_else(|| zlib_error("Error -2: inconsistent stream state"))?;
    match d.decompress(&data, max_length) {
        Ok(out) => Ok(bytesobject::w_bytes_from_bytes(&out)),
        Err(backend::DecompressError::Zlib(m)) => Err(zlib_error(m)),
        Err(backend::DecompressError::Eof) => Err(eof_error("End of stream already reached")),
    }
}

fn init_zdecompress_type(ns: PyObjectRef) {
    let new_sig = {
        let mut b = crate::SignatureBuilder::default();
        b.append("cls");
        b.marker_posonly();
        b.append("wbits");
        b.append("zdict");
        b.signature()
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            crate::typedef::make_new_descr_maybe_sig(zdecompress_new, Some(new_sig)),
        )
    };
    let decompress_sig = {
        let mut b = crate::SignatureBuilder::default();
        b.append("self");
        b.marker_posonly();
        b.append("data");
        b.append("max_length");
        b.signature()
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "decompress",
            crate::make_builtin_function_maybe_sig(
                "decompress",
                zdecompress_decompress,
                Some(decompress_sig),
            ),
        )
    };
    zdecompress_getset(ns, "unused_data", |args| {
        let id = get_id(args.get(1).copied().unwrap_or(PY_NULL));
        let reg = ZDECOMPRESSORS.lock().unwrap();
        let data = reg
            .get(&id)
            .map(|d| d.unused_data().to_vec())
            .unwrap_or_default();
        Ok(bytesobject::w_bytes_from_bytes(&data))
    });
    zdecompress_getset(ns, "eof", |args| {
        let id = get_id(args.get(1).copied().unwrap_or(PY_NULL));
        let reg = ZDECOMPRESSORS.lock().unwrap();
        Ok(w_bool_from(reg.get(&id).map(|d| d.eof()).unwrap_or(false)))
    });
    zdecompress_getset(ns, "needs_input", |args| {
        let id = get_id(args.get(1).copied().unwrap_or(PY_NULL));
        let reg = ZDECOMPRESSORS.lock().unwrap();
        Ok(w_bool_from(
            reg.get(&id).map(|d| d.needs_input()).unwrap_or(true),
        ))
    });
}

crate::py_module! {
    "zlib",
    interpleveldefs: {
        "ZLIB_VERSION" => w_str_new("1.3.1"),
        "ZLIB_RUNTIME_VERSION" => w_str_new("1.3.1"),
        "_ZlibDecompressor" => zdecompress_type(),
    },
    int_constants: {
        "DEFLATED" => 8,
        "MAX_WBITS" => 15,
        "DEF_MEM_LEVEL" => 8,
        "DEF_BUF_SIZE" => 16384,
        "Z_DEFAULT_COMPRESSION" => -1,
        "Z_NO_COMPRESSION" => 0,
        "Z_BEST_SPEED" => 1,
        "Z_BEST_COMPRESSION" => 9,
        "Z_DEFAULT_STRATEGY" => 0,
        "Z_FILTERED" => 1,
        "Z_HUFFMAN_ONLY" => 2,
        "Z_RLE" => 3,
        "Z_FIXED" => 4,
        "Z_NO_FLUSH" => 0,
        "Z_PARTIAL_FLUSH" => 1,
        "Z_SYNC_FLUSH" => 2,
        "Z_FULL_FLUSH" => 3,
        "Z_FINISH" => 4,
        "Z_BLOCK" => 5,
        "Z_TREES" => 6,
    },
    exceptions: {
        "error" => crate::builtins::lookup_exc_class("Exception").expect("Exception installed"),
    },
    inline_functions: {
        // interp_zlib.py:66 `compress(data, __posonly__=None, level, wbits)` —
        // `data` positional-only, `level`/`wbits` positional-or-keyword.
        fn compress(
            data: PyBufferStr,
            #[posonly]
            #[default(w_none())]
            level: PyObjectRef,
            #[default(w_none())]
            wbits: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let level = int_or_default(level, -1)? as i32;
            let wbits = to_wbits(int_or_default(wbits, backend::MAX_WBITS as i64)?);
            let out = backend::compress(data, level, wbits).map_err(zlib_error)?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        }
        // interp_zlib.py:92 `decompress(string, __posonly__=None, wbits, bufsize)`.
        fn decompress(
            data: PyBufferStr,
            #[posonly]
            #[default(w_none())]
            wbits: PyObjectRef,
            #[default(w_none())]
            bufsize: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let wbits = to_wbits(int_or_default(wbits, backend::MAX_WBITS as i64)?);
            let bufsize = int_or_default(bufsize, backend::DEF_BUF_SIZE as i64)?;
            if bufsize < 0 {
                return Err(crate::PyError::value_error("bufsize must be non-negative"));
            }
            let out = backend::decompress(data, wbits, bufsize as usize).map_err(zlib_error)?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        }
        // interp_zlib.py:228 `Compress___new__(level, method, wbits, memLevel,
        // strategy, w_zdict)` — all six positional-or-keyword.  `method` /
        // `memLevel` / `strategy` are accepted-and-ignored: `Compressor::new`
        // threads only level / wbits / zdict (interp_zlib.py:244 notes the
        // undocumented pass-through).
        fn compressobj(
            #[default(w_none())] level: PyObjectRef,
            #[default(w_none())] method: PyObjectRef,
            #[default(w_none())] wbits: PyObjectRef,
            #[default(w_none())] memLevel: PyObjectRef,
            #[default(w_none())] strategy: PyObjectRef,
            #[default(w_none())] zdict: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let _ = (method, memLevel, strategy);
            let level = int_or_default(level, -1)? as i32;
            let wbits = to_wbits(int_or_default(wbits, backend::MAX_WBITS as i64)?);
            make_compress(level, wbits, zdict_or_none(zdict)?)
        }
        // interp_zlib.py:400 `Decompress___new__(wbits, w_zdict)`.
        fn decompressobj(
            #[default(w_none())] wbits: PyObjectRef,
            #[default(w_none())] zdict: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let wbits = to_wbits(int_or_default(wbits, backend::MAX_WBITS as i64)?);
            make_decompress(wbits, zdict_or_none(zdict)?)
        }
    },
    functions: {
        "crc32" / * = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let start = args.get(1).map(|&o| unsafe { w_int_get_value(o) } as u32).unwrap_or(0);
            Ok(w_int_new(crc32_compute(&data, start) as i64))
        },
        "adler32" / * = |args| {
            let data = as_bytes(args.first().copied().unwrap_or(w_none()))?;
            let start = args.get(1).map(|&o| unsafe { w_int_get_value(o) } as u32).unwrap_or(1);
            Ok(w_int_new(adler32_compute(&data, start) as i64))
        },
    },
}
