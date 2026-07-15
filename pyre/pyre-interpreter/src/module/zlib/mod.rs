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
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return 0;
    }
    match unsafe { w_dict_getitem_str(d, "_id") } {
        Some(v) if unsafe { is_int(v) } => (unsafe { w_int_get_value(v) }) as u64,
        _ => 0,
    }
}

fn set_id(obj: PyObjectRef, id: u64) {
    let d = crate::baseobjspace::getdict(obj);
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
    let mut err = crate::PyError::value_error(msg.to_string());
    if let Some(cls) = crate::builtins::lookup_exc_class("EOFError") {
        let args = [cls, w_str_new(msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            err.exc_object = exc;
        }
    }
    err
}

// ── argument helpers ────────────────────────────────────────────────────

fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    unsafe {
        if bytesobject::is_bytes_like(obj) {
            Ok(bytesobject::bytes_like_data(obj).to_vec())
        } else if is_str(obj) {
            Ok(w_str_get_value(obj).as_bytes().to_vec())
        } else {
            Err(crate::PyError::type_error(
                "a bytes-like object is required",
            ))
        }
    }
}

/// Fetch an integer argument by keyword or position, `None`/missing → default.
fn arg_int(
    pos: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    name: &str,
    index: usize,
    default: i64,
) -> Result<i64, crate::PyError> {
    match crate::builtins::kwarg_get(kwargs, name).or_else(|| pos.get(index).copied()) {
        Some(o) if unsafe { is_none(o) } => Ok(default),
        Some(o) => crate::baseobjspace::int_w(o),
        None => Ok(default),
    }
}

/// Fetch an optional `zdict` bytes argument by keyword or position.
fn arg_zdict(
    pos: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    index: usize,
) -> Result<Option<Vec<u8>>, crate::PyError> {
    match crate::builtins::kwarg_get(kwargs, "zdict").or_else(|| pos.get(index).copied()) {
        Some(o) if !unsafe { is_none(o) } => Ok(Some(as_bytes(o)?)),
        _ => Ok(None),
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

thread_local! {
    static COMPRESS_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    static DECOMPRESS_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    static ZDECOMPRESS_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
}

fn compress_type() -> PyObjectRef {
    COMPRESS_TYPE.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("Compress", init_compress_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
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
                let mode = match args.get(1).copied() {
                    Some(o) if !unsafe { is_none(o) } => crate::baseobjspace::int_w(o)? as i32,
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
    DECOMPRESS_TYPE.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("Decompress", init_decompress_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
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
                let max_length = match args.get(2).copied() {
                    Some(o) if !unsafe { is_none(o) } => {
                        let v = crate::baseobjspace::int_w(o)?;
                        if v < 0 {
                            return Err(crate::PyError::value_error(
                                "max_length must be non-negative",
                            ));
                        }
                        (v != 0).then_some(v as usize)
                    }
                    _ => None,
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
    ZDECOMPRESS_TYPE.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_ZlibDecompressor", init_zdecompress_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
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

fn init_zdecompress_type(ns: PyObjectRef) {
    // _ZlibDecompressor(wbits=MAX_WBITS, zdict=b'') — the DecompressReader
    // factory gzip calls with wbits=-MAX_WBITS.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            crate::make_builtin_function("__new__", |args| {
                // args[0] is the type; the rest are the constructor arguments.
                let (pos, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
                let wbits = to_wbits(arg_int(pos, kwargs, "wbits", 0, backend::MAX_WBITS as i64)?);
                let zdict = arg_zdict(pos, kwargs, 1)?;
                let d = backend::ZlibDecompressor::new(wbits, zdict).map_err(zlib_error)?;
                let id = next_id();
                ZDECOMPRESSORS.lock().unwrap().insert(id, d);
                let obj = w_instance_new(zdecompress_type());
                set_id(obj, id);
                Ok(obj)
            }),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "decompress",
            crate::make_builtin_function("decompress", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("decompress() missing data"));
                }
                let (pos, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
                let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
                let max_length = match crate::builtins::kwarg_get(kwargs, "max_length")
                    .or_else(|| pos.get(1).copied())
                {
                    Some(o) if !unsafe { is_none(o) } => {
                        let v = crate::baseobjspace::int_w(o)?;
                        usize::try_from(v).ok()
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
                    Err(backend::DecompressError::Eof) => {
                        Err(eof_error("End of stream already reached"))
                    }
                }
            }),
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
        "compress" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let level = arg_int(pos, kwargs, "level", 1, -1)? as i32;
            let wbits = to_wbits(arg_int(pos, kwargs, "wbits", 2, backend::MAX_WBITS as i64)?);
            let out = backend::compress(&data, level, wbits).map_err(zlib_error)?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        },
        "decompress" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let data = as_bytes(pos.first().copied().unwrap_or(w_none()))?;
            let wbits = to_wbits(arg_int(pos, kwargs, "wbits", 1, backend::MAX_WBITS as i64)?);
            let bufsize = arg_int(pos, kwargs, "bufsize", 2, backend::DEF_BUF_SIZE as i64)?;
            if bufsize < 0 {
                return Err(crate::PyError::value_error("bufsize must be non-negative"));
            }
            let out = backend::decompress(&data, wbits, bufsize as usize).map_err(zlib_error)?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        },
        "compressobj" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let level = arg_int(pos, kwargs, "level", 0, -1)? as i32;
            // positions 1 (method) and 3 (memLevel) / 4 (strategy) are accepted
            // but only level / wbits / zdict affect the stream.
            let wbits = to_wbits(arg_int(pos, kwargs, "wbits", 2, backend::MAX_WBITS as i64)?);
            let zdict = arg_zdict(pos, kwargs, 5)?;
            make_compress(level, wbits, zdict)
        },
        "decompressobj" / * = |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let wbits = to_wbits(arg_int(pos, kwargs, "wbits", 0, backend::MAX_WBITS as i64)?);
            let zdict = arg_zdict(pos, kwargs, 1)?;
            make_decompress(wbits, zdict)
        },
    },
}
