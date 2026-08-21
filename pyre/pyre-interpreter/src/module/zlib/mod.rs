//! zlib module — PyPy: `pypy/module/zlib/`.
//!
//! CRC-32 / Adler-32 checksums plus the DEFLATE compress/decompress surface.
//! The DEFLATE machinery is a deliberate duplication of RustPython's zlib
//! implementation, ported into `pyre_native::zlib` (flate2 / zlib-rs) and kept
//! outside the LLBC extraction; this module is the W_Root object glue.
//!
//! `Compress` / `Decompress` / `_ZlibDecompressor` own their native stream
//! directly, matching `interp_zlib.py`'s `self.stream`.  Each stream also owns
//! its PyPy-style per-object lock; there is no process-global side table.

use pyre_native::zlib as backend;
use pyre_object::*;

use std::sync::Mutex;

/// PyPy `interp_zlib.py Compress`: the stream and lock belong to the
/// wrapper object.  The mapdict prefix preserves PyPy's ability to subclass
/// the type without moving its native payload into a side table.
#[crate::pyre_class("zlib.Compress")]
#[derive(Default)]
pub struct W_Compress {
    pub map: usize,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    backend: *mut Mutex<backend::Compressor>,
}

/// PyPy `interp_zlib.py Decompress` object-owned stream state.
#[crate::pyre_class("zlib.Decompress")]
#[derive(Default)]
pub struct W_Decompress {
    pub map: usize,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    backend: *mut Mutex<backend::Decompressor>,
}

/// PyPy `interp_zlib.py:419 ZlibDecompressor` object-owned buffered stream.
#[crate::pyre_class("zlib._ZlibDecompressor")]
#[derive(Default)]
pub struct W_ZlibDecompressor {
    pub map: usize,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    backend: *mut Mutex<backend::ZlibDecompressor>,
}

macro_rules! assert_mapdict_prefix {
    ($ty:ty) => {
        const _: () = assert!(
            std::mem::offset_of!($ty, map)
                == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, map),
            "zlib wrapper must keep W_ObjectObject's map offset"
        );
        const _: () = assert!(
            std::mem::offset_of!($ty, storage)
                == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, storage),
            "zlib wrapper must keep W_ObjectObject's storage offset"
        );
    };
}

assert_mapdict_prefix!(W_Compress);
assert_mapdict_prefix!(W_Decompress);
assert_mapdict_prefix!(W_ZlibDecompressor);

fn compressor_this(obj: PyObjectRef) -> Result<&'static mut W_Compress, crate::PyError> {
    W_Compress::from_obj(obj).ok_or_else(|| crate::PyError::type_error("expected Compress object"))
}

fn decompressor_this(obj: PyObjectRef) -> Result<&'static mut W_Decompress, crate::PyError> {
    W_Decompress::from_obj(obj)
        .ok_or_else(|| crate::PyError::type_error("expected Decompress object"))
}

fn zdecompressor_this(obj: PyObjectRef) -> Result<&'static mut W_ZlibDecompressor, crate::PyError> {
    W_ZlibDecompressor::from_obj(obj)
        .ok_or_else(|| crate::PyError::type_error("expected _ZlibDecompressor object"))
}

/// Sweep-time counterparts of the three RPython `_finalize_` methods.  The
/// Box contains the per-object mutex and native stream; dropping it neither
/// allocates nor calls back into Python.
pub unsafe fn w_compress_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_Compress::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

pub unsafe fn w_decompress_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_Decompress::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

pub unsafe fn w_zdecompress_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_ZlibDecompressor::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
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

/// `interp_zlib.py` spells every data argument `'bufferstr'`, so an
/// `array.array` or a C-contiguous `memoryview` is as acceptable as `bytes`.
fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    crate::baseobjspace::charbuf_w(obj)
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

/// `rzlib` passes window bits through a C `int`.  Preserve that 32-bit
/// wrapping boundary; narrowing to `i8` would incorrectly turn e.g. 271 into
/// the valid value 15.
fn to_wbits(v: i64) -> i32 {
    v as i32
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

static COMPRESS_RUNTIME_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static DECOMPRESS_RUNTIME_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static ZDECOMPRESS_RUNTIME_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn compress_type() -> PyObjectRef {
    *COMPRESS_RUNTIME_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "Compress",
            init_compress_type,
            crate::typedef::w_object(),
            <W_Compress as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_Compress as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp as usize
    }) as PyObjectRef
}

fn init_compress_type(ns: PyObjectRef) {
    let new_sig = {
        let mut b = crate::SignatureBuilder::default();
        b.append("cls");
        b.marker_posonly();
        b.append("level");
        b.append("method");
        b.append("wbits");
        b.append("memLevel");
        b.append("strategy");
        b.append("zdict");
        b.signature()
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            crate::typedef::make_new_descr_maybe_sig(compress_new, Some(new_sig)),
        )
    };
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
                    let data = as_bytes(args[1])?;
                    let this = compressor_this(args[0])?;
                    if this.backend.is_null() {
                        return Err(zlib_error("Error -2: inconsistent stream state"));
                    }
                    let mut c = unsafe { &*this.backend }.lock().unwrap();
                    let out = c.compress(&data).map_err(zlib_error)?;
                    Ok(bytesobject::w_bytes_from_bytes(&out))
                },
                2,
            ),
        )
    };
    for name in ["copy", "__copy__"] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                crate::make_builtin_function_with_arity(name, compress_copy, 1),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__deepcopy__",
            crate::make_builtin_function_with_arity("__deepcopy__", compress_copy, 2),
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
                // interp_zlib.py `@unwrap_spec(mode="c_int")` — the
                // converter reports a value outside the C `int` range rather
                // than truncating it into a different flush mode.
                let mode = match args.get(1).copied() {
                    Some(o) if !unsafe { is_none(o) } => crate::baseobjspace::c_int_w(o)?,
                    _ => backend::Z_FINISH,
                };
                let this = compressor_this(args[0])?;
                if this.backend.is_null() {
                    return Err(zlib_error("Error -2: inconsistent stream state"));
                }
                let mut c = unsafe { &*this.backend }.lock().unwrap();
                let out = c.flush(mode).map_err(zlib_error)?;
                Ok(bytesobject::w_bytes_from_bytes(&out))
            }),
        )
    };
}

fn compress_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or_else(compress_type);
    let level = int_or_default(args.get(1).copied().unwrap_or(PY_NULL), -1)? as i32;
    let method = int_or_default(args.get(2).copied().unwrap_or(PY_NULL), 8)? as i32;
    let wbits = to_wbits(int_or_default(
        args.get(3).copied().unwrap_or(PY_NULL),
        backend::MAX_WBITS as i64,
    )?);
    let mem_level = int_or_default(args.get(4).copied().unwrap_or(PY_NULL), 8)? as i32;
    let strategy = int_or_default(args.get(5).copied().unwrap_or(PY_NULL), 0)? as i32;
    let zdict = zdict_or_none(args.get(6).copied().unwrap_or(PY_NULL))?;
    make_compress_for(cls, level, method, wbits, mem_level, strategy, zdict)
}

fn make_compress(
    level: i32,
    method: i32,
    wbits: i32,
    mem_level: i32,
    strategy: i32,
    zdict: Option<Vec<u8>>,
) -> Result<PyObjectRef, crate::PyError> {
    make_compress_for(
        compress_type(),
        level,
        method,
        wbits,
        mem_level,
        strategy,
        zdict,
    )
}

fn make_compress_for(
    cls: PyObjectRef,
    level: i32,
    method: i32,
    wbits: i32,
    mem_level: i32,
    strategy: i32,
    zdict: Option<Vec<u8>>,
) -> Result<PyObjectRef, crate::PyError> {
    if !(-1..=9).contains(&level)
        || method != 8
        || !((-15..=-9).contains(&wbits) || (8..=15).contains(&wbits) || (25..=31).contains(&wbits))
        || !(1..=9).contains(&mem_level)
        || !(0..=4).contains(&strategy)
    {
        // interp_zlib.py:244-247: rzlib.deflateInit reports invalid
        // initialization options as ValueError, distinct from operational
        // zlib.error failures.
        return Err(crate::PyError::value_error("Invalid initialization option"));
    }
    let c = backend::Compressor::new(level, method, wbits, mem_level, strategy, zdict.as_deref())
        .map_err(zlib_error)?;
    allocate_compress(cls, c)
}

/// `interp_zlib.py Compress.copy` and `descr_deepcopy`: copy the live
/// deflate stream while holding the source object's lock, then construct the
/// base `Compress` type (PyPy deliberately does not preserve a subclass).
fn compress_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args.first().copied().unwrap_or(PY_NULL);
    let copied = {
        let this = compressor_this(obj)?;
        if this.backend.is_null() {
            return Err(crate::PyError::value_error(
                "Compressor was already flushed",
            ));
        }
        let mut compressor = unsafe { &*this.backend }.lock().unwrap();
        if compressor.is_finished() {
            return Err(crate::PyError::value_error(
                "Compressor was already flushed",
            ));
        }
        compressor.copy().map_err(zlib_error)?
    };
    allocate_compress(compress_type(), copied)
}

fn allocate_compress(
    cls: PyObjectRef,
    backend: backend::Compressor,
) -> Result<PyObjectRef, crate::PyError> {
    let backend = Box::into_raw(Box::new(Mutex::new(backend)));
    let _roots = gc_roots::push_roots();
    gc_roots::pin_root(cls);
    let cls_slot = gc_roots::shadow_stack_len() - 1;
    let obj = W_Compress::allocate_stable(W_Compress {
        ob: PyObject::default(),
        map: 0,
        storage: std::ptr::null_mut(),
        backend,
    });
    Ok(crate::typedef::tag_subclass_instance(obj, unsafe {
        gc_roots::shadow_stack_get(cls_slot)
    }))
}

// ── Decompress (decompressobj) ──────────────────────────────────────────

fn decompress_type() -> PyObjectRef {
    *DECOMPRESS_RUNTIME_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "Decompress",
            init_decompress_type,
            crate::typedef::w_object(),
            <W_Decompress as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_Decompress as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
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

// interp_zlib.py `decompress(self, data, __posonly__=None,
// max_length=0)`, exposed as `($self, data, /, max_length=0)`.  The signature
// binder is essential here: pip's wheel reader passes max_length by keyword,
// and an unbound native callable otherwise sees the kwargs transport object as
// a positional integer.
fn decompress_decompress(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data_obj = args.get(1).copied().unwrap_or(PY_NULL);
    if data_obj.is_null() {
        return Err(crate::PyError::type_error("decompress() missing data"));
    }
    let data = as_bytes(data_obj)?;
    // An omitted `max_length` is unlimited; a supplied value — including
    // `None` — goes through `int_w`.  Zero also means unlimited here, and a
    // negative value is rejected.
    let max_length = match args.get(2).copied() {
        Some(o) if !o.is_null() => {
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
    let this = decompressor_this(args[0])?;
    if this.backend.is_null() {
        return Err(zlib_error("Error -2: inconsistent stream state"));
    }
    let mut d = unsafe { &*this.backend }.lock().unwrap();
    let out = d.decompress(&data, max_length).map_err(zlib_error)?;
    Ok(bytesobject::w_bytes_from_bytes(&out))
}

fn init_decompress_type(ns: PyObjectRef) {
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
            crate::typedef::make_new_descr_maybe_sig(decompress_new, Some(new_sig)),
        )
    };
    let decompress_sig = {
        let mut b = crate::SignatureBuilder::default();
        b.append("self");
        b.append("data");
        b.marker_posonly();
        b.append("max_length");
        b.signature()
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "decompress",
            crate::make_builtin_function_maybe_sig(
                "decompress",
                decompress_decompress,
                Some(decompress_sig),
            ),
        )
    };
    for name in ["copy", "__copy__"] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                crate::make_builtin_function_with_arity(name, decompress_copy, 1),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__deepcopy__",
            crate::make_builtin_function_with_arity("__deepcopy__", decompress_copy, 2),
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
                let this = decompressor_this(args[0])?;
                if this.backend.is_null() {
                    return Err(zlib_error("Error -2: inconsistent stream state"));
                }
                let mut d = unsafe { &*this.backend }.lock().unwrap();
                let out = d.flush(length).map_err(zlib_error)?;
                Ok(bytesobject::w_bytes_from_bytes(&out))
            }),
        )
    };
    decompress_getset(ns, "unused_data", |args| {
        let this = decompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
        let data = if this.backend.is_null() {
            Vec::new()
        } else {
            unsafe { &*this.backend }
                .lock()
                .unwrap()
                .unused_data()
                .to_vec()
        };
        Ok(bytesobject::w_bytes_from_bytes(&data))
    });
    decompress_getset(ns, "unconsumed_tail", |args| {
        let this = decompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
        let data = if this.backend.is_null() {
            Vec::new()
        } else {
            unsafe { &*this.backend }
                .lock()
                .unwrap()
                .unconsumed_tail()
                .to_vec()
        };
        Ok(bytesobject::w_bytes_from_bytes(&data))
    });
    decompress_getset(ns, "eof", |args| {
        let this = decompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
        Ok(w_bool_from(
            !this.backend.is_null() && unsafe { &*this.backend }.lock().unwrap().eof(),
        ))
    });
}

fn decompress_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or_else(decompress_type);
    let wbits = to_wbits(int_or_default(
        args.get(1).copied().unwrap_or(PY_NULL),
        backend::MAX_WBITS as i64,
    )?);
    let zdict = zdict_or_none(args.get(2).copied().unwrap_or(PY_NULL))?;
    make_decompress_for(cls, wbits, zdict)
}

fn make_decompress(wbits: i32, zdict: Option<Vec<u8>>) -> Result<PyObjectRef, crate::PyError> {
    make_decompress_for(decompress_type(), wbits, zdict)
}

fn make_decompress_for(
    cls: PyObjectRef,
    wbits: i32,
    zdict: Option<Vec<u8>>,
) -> Result<PyObjectRef, crate::PyError> {
    if !(wbits == 0
        || (-15..=-8).contains(&wbits)
        || (8..=15).contains(&wbits)
        || (24..=31).contains(&wbits)
        || (40..=47).contains(&wbits))
    {
        // interp_zlib.py:411-414 mirrors the deflate constructor's ValueError
        // conversion for inflateInit2.
        return Err(crate::PyError::value_error("Invalid initialization option"));
    }
    let d = backend::Decompressor::new(wbits, zdict).map_err(zlib_error)?;
    allocate_decompress(cls, d)
}

/// `interp_zlib.py Decompress.copy`: clone both inflate state and its
/// visible tails/dictionary under the per-object lock, returning the base type.
fn decompress_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args.first().copied().unwrap_or(PY_NULL);
    let copied = {
        let this = decompressor_this(obj)?;
        if this.backend.is_null() {
            return Err(crate::PyError::value_error(
                "Decompressor was already flushed",
            ));
        }
        let decompressor = unsafe { &*this.backend }.lock().unwrap();
        if decompressor.is_finished() {
            return Err(crate::PyError::value_error(
                "Decompressor was already flushed",
            ));
        }
        decompressor.copy().map_err(zlib_error)?
    };
    allocate_decompress(decompress_type(), copied)
}

fn allocate_decompress(
    cls: PyObjectRef,
    backend: backend::Decompressor,
) -> Result<PyObjectRef, crate::PyError> {
    let backend = Box::into_raw(Box::new(Mutex::new(backend)));
    let _roots = gc_roots::push_roots();
    gc_roots::pin_root(cls);
    let cls_slot = gc_roots::shadow_stack_len() - 1;
    let obj = W_Decompress::allocate_stable(W_Decompress {
        ob: PyObject::default(),
        map: 0,
        storage: std::ptr::null_mut(),
        backend,
    });
    Ok(crate::typedef::tag_subclass_instance(obj, unsafe {
        gc_roots::shadow_stack_get(cls_slot)
    }))
}

// ── _ZlibDecompressor (buffered; used by gzip reading) ──────────────────

fn zdecompress_type() -> PyObjectRef {
    *ZDECOMPRESS_RUNTIME_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_ZlibDecompressor",
            init_zdecompress_type,
            crate::typedef::w_object(),
            <W_ZlibDecompressor as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_ZlibDecompressor as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
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
    let cls = args.first().copied().unwrap_or_else(zdecompress_type);
    allocate_zdecompress(cls, d)
}

fn allocate_zdecompress(
    cls: PyObjectRef,
    backend: backend::ZlibDecompressor,
) -> Result<PyObjectRef, crate::PyError> {
    let backend = Box::into_raw(Box::new(Mutex::new(backend)));
    let _roots = gc_roots::push_roots();
    gc_roots::pin_root(cls);
    let cls_slot = gc_roots::shadow_stack_len() - 1;
    let obj = W_ZlibDecompressor::allocate_stable(W_ZlibDecompressor {
        ob: PyObject::default(),
        map: 0,
        storage: std::ptr::null_mut(),
        backend,
    });
    Ok(crate::typedef::tag_subclass_instance(obj, unsafe {
        gc_roots::shadow_stack_get(cls_slot)
    }))
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
    let this = zdecompressor_this(args[0])?;
    if this.backend.is_null() {
        return Err(zlib_error("Error -2: inconsistent stream state"));
    }
    let mut d = unsafe { &*this.backend }.lock().unwrap();
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
    for name in ["__reduce__", "__reduce_ex__"] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                crate::make_builtin_function(name, |_| {
                    Err(crate::PyError::type_error(
                        "cannot pickle 'zlib._ZlibDecompressor' object",
                    ))
                }),
            )
        };
    }
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
        let this = zdecompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
        let data = if this.backend.is_null() {
            Vec::new()
        } else {
            unsafe { &*this.backend }
                .lock()
                .unwrap()
                .unused_data()
                .to_vec()
        };
        Ok(bytesobject::w_bytes_from_bytes(&data))
    });
    zdecompress_getset(ns, "eof", |args| {
        let this = zdecompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
        Ok(w_bool_from(
            !this.backend.is_null() && unsafe { &*this.backend }.lock().unwrap().eof(),
        ))
    });
    zdecompress_getset(ns, "needs_input", |args| {
        let this = zdecompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
        Ok(w_bool_from(
            this.backend.is_null() || unsafe { &*this.backend }.lock().unwrap().needs_input(),
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
        // interp_zlib.py `compress(data, __posonly__=None, level, wbits)` —
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
            let out = backend::compress(&data, level, wbits).map_err(zlib_error)?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        }
        // interp_zlib.py `decompress(string, __posonly__=None, wbits, bufsize)`.
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
            let out = backend::decompress(&data, wbits, bufsize as usize).map_err(zlib_error)?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        }
        // interp_zlib.py `Compress___new__(level, method, wbits, memLevel,
        // strategy, w_zdict)` — all six positional-or-keyword and all native
        // initialization options are passed through to deflateInit2.
        fn compressobj(
            #[default(w_none())] level: PyObjectRef,
            #[default(w_none())] method: PyObjectRef,
            #[default(w_none())] wbits: PyObjectRef,
            #[default(w_none())] memLevel: PyObjectRef,
            #[default(w_none())] strategy: PyObjectRef,
            #[default(w_none())] zdict: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let level = int_or_default(level, -1)? as i32;
            let method = int_or_default(method, 8)? as i32;
            let wbits = to_wbits(int_or_default(wbits, backend::MAX_WBITS as i64)?);
            let mem_level = int_or_default(memLevel, 8)? as i32;
            let strategy = int_or_default(strategy, 0)? as i32;
            make_compress(
                level,
                method,
                wbits,
                mem_level,
                strategy,
                zdict_or_none(zdict)?,
            )
        }
        // interp_zlib.py `Decompress___new__(wbits, w_zdict)`.
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
