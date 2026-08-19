//! `_lzma` module — `Modules/_lzmamodule.c`.
//!
//! PyPy has no interpreter-level `_lzma`: `lib_pypy/_lzma.py` drives liblzma
//! through cffi and the applevel `lzma.py` stands on that, so the C module is
//! the structure this follows, the same footing `_heapq` sits on.
//!
//! The codec lives in `pyre_native::lzma` — the `xz-core` port of liblzma —
//! outside the LLBC extraction, exactly as `zlib` and `_bz2` keep theirs.
//! What is left here is the object glue: the two stream objects, the filter
//! dicts, the argument surface, and `LZMAError`.

use pyre_native::lzma as backend;
use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};
use pyre_object::*;

use std::sync::Mutex;

/// `Compressor`: the liblzma stream and its lock belong to the object, and
/// there is no process-global side table.
#[crate::pyre_class("_lzma.LZMACompressor")]
#[derive(Default)]
pub struct W_LZMACompressor {
    backend: *mut Mutex<backend::Compressor>,
}

/// `Decompressor`, object-owned the same way.  The unconsumed input and the
/// `unused_data` tail live with the stream rather than as Python attributes.
#[crate::pyre_class("_lzma.LZMADecompressor")]
#[derive(Default)]
pub struct W_LZMADecompressor {
    backend: *mut Mutex<backend::Decompressor>,
}

/// `LZMAError`, the module's own exception class.  The fallback kind only
/// matters if the class is somehow missing from the registry; the raised
/// object is what `except LZMAError` matches on.
fn lzma_exception(msg: impl Into<String>) -> crate::PyError {
    let msg = msg.into();
    let mut err = crate::PyError::value_error(msg.clone());
    if let Some(cls) = crate::builtins::lookup_exc_class("_lzma.LZMAError") {
        let args = [cls, w_str_new(&msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            err.exc_object = exc;
        }
    }
    err
}

/// `catch_lzma_error`, plus the three failures the filter conversion reports
/// before liblzma is reached.
fn lzma_error(error: backend::LzmaError) -> crate::PyError {
    use backend::LzmaError as E;
    match error {
        E::UnsupportedCheck => lzma_exception("Unsupported integrity check"),
        E::Mem => crate::PyError::memory_error("out of memory"),
        E::MemLimit => lzma_exception("Memory usage limit exceeded"),
        E::Format => lzma_exception("Input format not supported by decoder"),
        E::Options => lzma_exception("Invalid or unsupported options"),
        E::Data => lzma_exception("Corrupt input data"),
        E::Buf => lzma_exception("Insufficient buffer space"),
        E::Prog => lzma_exception("Internal error"),
        E::Unrecognized(ret) => lzma_exception(format!("Unrecognized error from liblzma: {ret}")),
        E::InvalidPreset(preset) => lzma_exception(format!("Invalid compression preset: {preset}")),
        E::InvalidFilterId(id) => crate::PyError::value_error(format!("Invalid filter ID: {id}")),
        E::AloneChainNotSingleLzma1 => crate::PyError::value_error(
            "Invalid filter chain for FORMAT_ALONE - must be a single LZMA1 filter",
        ),
    }
}

/// Both stream objects carry liblzma state no state dict can describe, which
/// is the `tp_basicsize` mismatch `_PyObject_GetState` refuses — so neither
/// can be pickled, however ordinary the rest of the object looks.
fn cannot_serialize(name: &str) -> crate::PyError {
    crate::PyError::type_error(format!("cannot pickle '{name}' object"))
}

/// `_PyLong_UInt32_Converter`: the value arrives through `__index__`, and one
/// outside a `uint32_t` is an error rather than a truncation.
fn uint32_w(obj: PyObjectRef) -> Result<u32, crate::PyError> {
    crate::baseobjspace::c_uint_w(crate::baseobjspace::space_index(obj)?)
}

// ── filter specifiers ───────────────────────────────────────────────────

/// What one filter id makes of its spec dict: the option names it reads, and
/// the message it reports for a spec carrying anything else.
///
/// `parse_filter_spec_*` hands the dict to `PyArg_ParseTupleAndKeywords` as
/// the keyword mapping, so an unrecognized key is exactly what makes the
/// parse fail — hence the entry count check below.
struct FilterKind {
    optnames: &'static [&'static str],
    message: &'static str,
}

/// `parse_filter_spec_lzma`.
const LZMA_KIND: FilterKind = FilterKind {
    optnames: &[
        "id",
        "preset",
        "dict_size",
        "lc",
        "lp",
        "pb",
        "mode",
        "nice_len",
        "mf",
        "depth",
    ],
    message: "Invalid filter specifier for LZMA filter",
};

/// `parse_filter_spec_delta`.
const DELTA_KIND: FilterKind = FilterKind {
    optnames: &["id", "dist"],
    message: "Invalid filter specifier for delta filter",
};

/// `parse_filter_spec_bcj`.
const BCJ_KIND: FilterKind = FilterKind {
    optnames: &["id", "start_offset"],
    message: "Invalid filter specifier for BCJ filter",
};

/// `lzma_filter_converter`.  Every read here can run a `__getitem__` of the
/// caller's own, so the spec is held in a shadow-stack slot rather than a
/// Rust local.
fn parse_filter_spec(spec: PyObjectRef) -> Result<backend::FilterSpec, crate::PyError> {
    let _roots = push_roots();
    let spec_slot = shadow_stack_len();
    pin_root(spec);
    let spec = || shadow_stack_get(spec_slot);

    if unsafe { crate::baseobjspace::lookup(spec(), "__getitem__") }.is_none() {
        return Err(crate::PyError::type_error(
            "Filter specifier must be a dict or dict-like object",
        ));
    }
    let Some(w_id) = crate::baseobjspace::finditem_str(spec(), "id")? else {
        return Err(crate::PyError::value_error(
            "Filter specifier must have an \"id\" entry",
        ));
    };
    // `lzma_vli` is a `uint64_t`, so the id reads as an unsigned 64-bit value
    // and one that overflows a *signed* read still converts. Whether the
    // result names a filter is the dispatch's answer below, not this one's;
    // only a width the type cannot hold is an overflow here.
    let w_index = crate::baseobjspace::space_index(w_id)?;
    let negative = || crate::PyError::value_error("cannot convert negative integer to unsigned");
    let id = match crate::baseobjspace::int_w(w_index) {
        Ok(value) => u64::try_from(value).map_err(|_| negative())?,
        Err(error) if error.kind == crate::PyErrorKind::OverflowError => {
            let big = unsafe { pyre_object::w_long_get_value(w_index) };
            if big.get_sign() < 0 {
                return Err(negative());
            }
            big.to_u64().ok_or_else(|| {
                crate::PyError::overflow_error("Python int too large for C lzma_vli")
            })?
        }
        Err(error) => return Err(error),
    };
    let kind = match id {
        backend::FILTER_LZMA1 | backend::FILTER_LZMA2 => LZMA_KIND,
        backend::FILTER_DELTA => DELTA_KIND,
        backend::FILTER_X86
        | backend::FILTER_POWERPC
        | backend::FILTER_IA64
        | backend::FILTER_ARM
        | backend::FILTER_ARMTHUMB
        | backend::FILTER_SPARC => BCJ_KIND,
        id => {
            return Err(crate::PyError::value_error(format!(
                "Invalid filter ID: {id}"
            )));
        }
    };

    let mut parsed = backend::FilterSpec {
        id,
        ..backend::FilterSpec::default()
    };
    // The "id" entry is already accounted for.
    let mut named = 1i64;
    for name in &kind.optnames[1..] {
        let Some(value) = crate::baseobjspace::finditem_str(spec(), name)? else {
            continue;
        };
        named += 1;
        let value = Some(uint32_w(value)?);
        match *name {
            "preset" => parsed.preset = value,
            "dict_size" => parsed.dict_size = value,
            "lc" => parsed.lc = value,
            "lp" => parsed.lp = value,
            "pb" => parsed.pb = value,
            "mode" => parsed.mode = value,
            "nice_len" => parsed.nice_len = value,
            "mf" => parsed.mf = value,
            "depth" => parsed.depth = value,
            "dist" => parsed.dist = value,
            "start_offset" => parsed.start_offset = value,
            other => unreachable!("filter option {other} has no slot"),
        }
    }
    // The options are read by a keyword parse whose keyword argument is the
    // spec itself, so a spec that is not a dict fails there — `__getitem__`
    // alone carries it past the dict-or-dict-like gate above but no further.
    // A dict carrying an entry the filter does not name fails there too, and
    // that is what the count catches.
    let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
    if !unsafe { crate::baseobjspace::isinstance_w(spec(), w_dict_type) } {
        return Err(crate::PyError::value_error(kind.message));
    }
    let entries = crate::baseobjspace::len(spec())
        .and_then(crate::baseobjspace::int_w)
        .map_err(|_| crate::PyError::value_error(kind.message))?;
    if entries != named {
        return Err(crate::PyError::value_error(kind.message));
    }
    Ok(parsed)
}

/// `parse_filter_chain_spec`.
fn parse_filter_chain(filters: PyObjectRef) -> Result<Vec<backend::FilterSpec>, crate::PyError> {
    let _roots = push_roots();
    let filters_slot = shadow_stack_len();
    pin_root(filters);
    let filters = || shadow_stack_get(filters_slot);

    let count = crate::runtime_ops::sequence_len(filters())?;
    if count > backend::FILTERS_MAX {
        return Err(crate::PyError::value_error(format!(
            "Too many filters - liblzma supports a maximum of {}",
            backend::FILTERS_MAX
        )));
    }
    let mut specs = Vec::with_capacity(count);
    for index in 0..count {
        let spec = crate::runtime_ops::sequence_getitem(filters(), index)?;
        specs.push(parse_filter_spec(spec)?);
    }
    Ok(specs)
}

/// The chain a `filters` argument names, or `None` when it was left out.
fn optional_filter_chain(
    filters: PyObjectRef,
) -> Result<Option<Vec<backend::FilterSpec>>, crate::PyError> {
    if unsafe { is_none(filters) } {
        return Ok(None);
    }
    parse_filter_chain(filters).map(Some)
}

impl W_LZMACompressor {
    fn compressor(&self) -> Result<&Mutex<backend::Compressor>, crate::PyError> {
        if self.backend.is_null() {
            return Err(crate::PyError::value_error(
                "Compressor was not initialized",
            ));
        }
        Ok(unsafe { &*self.backend })
    }
}

impl W_LZMADecompressor {
    fn decompressor(&self) -> Result<&Mutex<backend::Decompressor>, crate::PyError> {
        if self.backend.is_null() {
            return Err(crate::PyError::value_error(
                "Decompressor was not initialized",
            ));
        }
        Ok(unsafe { &*self.backend })
    }
}

mod compressor_methods {
    use super::*;

    #[crate::pyre_methods(
        doc = "Create a compressor object for compressing data incrementally.\n\n\
               The settings used by the compressor can be specified either as a\n\
               preset compression level (with the 'preset' argument), or in detail\n\
               as a custom filter chain (with the 'filters' argument).  For FORMAT_XZ\n\
               and FORMAT_ALONE, the default is to use the PRESET_DEFAULT preset\n\
               level.  For FORMAT_RAW, the caller must always specify a filter chain;\n\
               the raw compressor does not support preset compression levels.\n\n\
               For one-shot compression, use the compress() function instead."
    )]
    impl W_LZMACompressor {
        /// `Compressor_new`.  Everything the container format rules out is
        /// rejected before the encoder is initialized.
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            #[default(backend::FORMAT_XZ)] format: PyIndexCInt,
            #[default(-1i32)] check: PyIndexCInt,
            #[default(w_none())] preset: PyObjectRef,
            #[default(w_none())] filters: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // The preset conversion below can run a `__index__` of the
            // caller's own, so the chain is held in a shadow-stack slot from
            // here on rather than in this argument.
            let _roots = push_roots();
            let filters_slot = shadow_stack_len();
            pin_root(filters);
            let filters = || shadow_stack_get(filters_slot);

            if format != backend::FORMAT_XZ && check != -1 && check != backend::CHECK_NONE as i32 {
                return Err(crate::PyError::value_error(
                    "Integrity checks are only supported by FORMAT_XZ",
                ));
            }
            let preset_given = !unsafe { is_none(preset) };
            let filters_given = !unsafe { is_none(filters()) };
            if preset_given && filters_given {
                return Err(crate::PyError::value_error(
                    "Cannot specify both preset and filter chain",
                ));
            }
            let preset = if preset_given {
                uint32_w(preset)?
            } else {
                backend::PRESET_DEFAULT
            };
            // Only FORMAT_XZ carries an integrity check, and FORMAT_AUTO
            // names nothing that can be written.
            let check = match (format, check) {
                (backend::FORMAT_XZ, -1) => backend::CHECK_CRC64,
                (backend::FORMAT_XZ, check) => check as u32,
                (backend::FORMAT_ALONE | backend::FORMAT_RAW, _) => backend::CHECK_NONE,
                _ => {
                    return Err(crate::PyError::value_error(format!(
                        "Invalid container format: {format}"
                    )));
                }
            };
            if format == backend::FORMAT_RAW && !filters_given {
                return Err(crate::PyError::value_error(
                    "Must specify filters for FORMAT_RAW",
                ));
            }
            let specs = optional_filter_chain(filters())?;
            let compressor = backend::Compressor::new(format, check, preset, specs.as_deref())
                .map_err(lzma_error)?;
            Ok(W_LZMACompressor::allocate_stable(W_LZMACompressor {
                backend: Box::into_raw(Box::new(Mutex::new(compressor))),
                ..W_LZMACompressor::default()
            }))
        }

        /// `_lzma_LZMACompressor_compress_impl`.
        fn compress(&mut self, data: PyBufferStr) -> Result<Vec<u8>, crate::PyError> {
            let mut compressor = self.compressor()?.lock().unwrap();
            if compressor.is_flushed() {
                return Err(crate::PyError::value_error("Compressor has been flushed"));
            }
            compressor.compress(&data).map_err(lzma_error)
        }

        /// `_lzma_LZMACompressor_flush_impl` — the object may not be used
        /// afterwards.
        fn flush(&mut self) -> Result<Vec<u8>, crate::PyError> {
            let mut compressor = self.compressor()?.lock().unwrap();
            if compressor.is_flushed() {
                return Err(crate::PyError::value_error("Repeated call to flush()"));
            }
            compressor.flush().map_err(lzma_error)
        }

        fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
            Err(cannot_serialize("_lzma.LZMACompressor"))
        }
    }
} // compressor_methods

mod decompressor_methods {
    use super::*;

    #[crate::pyre_methods(
        doc = "Create a decompressor object for decompressing data incrementally.\n\n\
               For one-shot decompression, use the decompress() function instead."
    )]
    impl W_LZMADecompressor {
        /// `_lzma_LZMADecompressor_impl`.  A caller that names no memory
        /// limit gets the clinic default, an unlimited one.
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            #[default(backend::FORMAT_AUTO)] format: PyIndexCInt,
            #[default(w_none())] memlimit: PyObjectRef,
            #[default(w_none())] filters: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            // The memory-limit conversion below can run a `__index__` of the
            // caller's own, so the chain moves to a shadow-stack slot first.
            let _roots = push_roots();
            let filters_slot = shadow_stack_len();
            pin_root(filters);
            let filters = || shadow_stack_get(filters_slot);

            let memlimit = if unsafe { is_none(memlimit) } {
                u64::MAX
            } else {
                if format == backend::FORMAT_RAW {
                    return Err(crate::PyError::value_error(
                        "Cannot specify memory limit with FORMAT_RAW",
                    ));
                }
                crate::baseobjspace::uint_w(crate::baseobjspace::space_index(memlimit)?)?
            };
            let filters_given = !unsafe { is_none(filters()) };
            if format == backend::FORMAT_RAW && !filters_given {
                return Err(crate::PyError::value_error(
                    "Must specify filters for FORMAT_RAW",
                ));
            }
            if format != backend::FORMAT_RAW && filters_given {
                return Err(crate::PyError::value_error(
                    "Cannot specify filters except with FORMAT_RAW",
                ));
            }
            if !matches!(
                format,
                backend::FORMAT_AUTO
                    | backend::FORMAT_XZ
                    | backend::FORMAT_ALONE
                    | backend::FORMAT_RAW
            ) {
                return Err(crate::PyError::value_error(format!(
                    "Invalid container format: {format}"
                )));
            }
            let specs = optional_filter_chain(filters())?;
            let decompressor = backend::Decompressor::new(format, memlimit, specs.as_deref())
                .map_err(lzma_error)?;
            Ok(W_LZMADecompressor::allocate_stable(W_LZMADecompressor {
                backend: Box::into_raw(Box::new(Mutex::new(decompressor))),
                ..W_LZMADecompressor::default()
            }))
        }

        /// `_lzma_LZMADecompressor_decompress_impl` — a negative
        /// `max_length` is unlimited.
        fn decompress(
            &mut self,
            data: PyBufferStr,
            #[default(-1i64)] max_length: PyIndexInt,
        ) -> Result<Vec<u8>, crate::PyError> {
            let mut decompressor = self.decompressor()?.lock().unwrap();
            if decompressor.eof() {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::EOFError,
                    "Already at end of stream",
                ));
            }
            // A cap too large for the platform's index type is an error, not
            // silently unlimited -- only a negative value means unlimited.
            let max_length = if max_length < 0 {
                None
            } else {
                Some(usize::try_from(max_length).map_err(|_| {
                    crate::PyError::overflow_error("Python int too large to convert to C ssize_t")
                })?)
            };
            decompressor
                .decompress(&data, max_length)
                .map_err(lzma_error)
        }

        /// ID of the integrity check used by the input stream.
        #[getter]
        fn check(&self) -> Result<i64, crate::PyError> {
            Ok(self.decompressor()?.lock().unwrap().check() as i64)
        }

        /// True once the end-of-stream marker has been reached.
        #[getter]
        fn eof(&self) -> Result<bool, crate::PyError> {
            Ok(self.decompressor()?.lock().unwrap().eof())
        }

        /// True when more input is needed before more decompressed data can
        /// be produced.
        #[getter]
        fn needs_input(&self) -> Result<bool, crate::PyError> {
            Ok(self.decompressor()?.lock().unwrap().needs_input())
        }

        /// Data found after the end of the compressed stream.
        #[getter]
        fn unused_data(&self) -> Result<Vec<u8>, crate::PyError> {
            Ok(self.decompressor()?.lock().unwrap().unused_data().to_vec())
        }

        fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
            Err(cannot_serialize("_lzma.LZMADecompressor"))
        }
    }
} // decompressor_methods

/// The `LZMACompressor` type object.  Neither stream type is acceptable as a
/// base class, yet both constructors take keywords, so the call path needs to
/// name them.
pub fn compressor_type() -> PyObjectRef {
    compressor_methods::type_object()
}

/// The `LZMADecompressor` type object, named for the same reason.
pub fn decompressor_type() -> PyObjectRef {
    decompressor_methods::type_object()
}

/// Sweep-time counterpart of `Compressor_dealloc`.  The Box holds the
/// per-object mutex and the native stream; dropping it neither allocates nor
/// calls back into Python.
///
/// # Safety
/// `obj` must be a GC-dead `W_LZMACompressor`.
pub unsafe fn w_lzmacompressor_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_LZMACompressor::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

/// Sweep-time counterpart of `Decompressor_dealloc`.
///
/// # Safety
/// `obj` must be a GC-dead `W_LZMADecompressor`.
pub unsafe fn w_lzmadecompressor_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_LZMADecompressor::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

crate::py_module! {
    "_lzma",
    interpleveldefs: {
        "LZMACompressor" => compressor_methods::type_object(),
        "LZMADecompressor" => decompressor_methods::type_object(),
    },
    int_constants: {
        "FORMAT_AUTO" => backend::FORMAT_AUTO,
        "FORMAT_XZ" => backend::FORMAT_XZ,
        "FORMAT_ALONE" => backend::FORMAT_ALONE,
        "FORMAT_RAW" => backend::FORMAT_RAW,
        "CHECK_NONE" => backend::CHECK_NONE,
        "CHECK_CRC32" => backend::CHECK_CRC32,
        "CHECK_CRC64" => backend::CHECK_CRC64,
        "CHECK_SHA256" => backend::CHECK_SHA256,
        "CHECK_ID_MAX" => backend::CHECK_ID_MAX,
        "CHECK_UNKNOWN" => backend::CHECK_UNKNOWN,
        "FILTER_LZMA1" => backend::FILTER_LZMA1,
        "FILTER_LZMA2" => backend::FILTER_LZMA2,
        "FILTER_DELTA" => backend::FILTER_DELTA,
        "FILTER_X86" => backend::FILTER_X86,
        "FILTER_IA64" => backend::FILTER_IA64,
        "FILTER_ARM" => backend::FILTER_ARM,
        "FILTER_ARMTHUMB" => backend::FILTER_ARMTHUMB,
        "FILTER_SPARC" => backend::FILTER_SPARC,
        "FILTER_POWERPC" => backend::FILTER_POWERPC,
        "MF_HC3" => backend::MF_HC3,
        "MF_HC4" => backend::MF_HC4,
        "MF_BT2" => backend::MF_BT2,
        "MF_BT3" => backend::MF_BT3,
        "MF_BT4" => backend::MF_BT4,
        "MODE_FAST" => backend::MODE_FAST,
        "MODE_NORMAL" => backend::MODE_NORMAL,
        "PRESET_DEFAULT" => backend::PRESET_DEFAULT,
        "PRESET_EXTREME" => backend::PRESET_EXTREME,
    },
    exceptions: {
        "LZMAError" => crate::builtins::lookup_exc_class("Exception").expect("Exception installed"),
    },
    inline_functions: {
        // `_lzma_is_check_supported_impl`.
        fn is_check_supported(check_id: PyIndexCInt) -> bool {
            backend::is_check_supported(check_id as u32)
        }
        // `_lzma__encode_filter_properties_impl` — the options of one filter,
        // without the filter id itself.
        fn _encode_filter_properties(
            filter: PyObjectRef,
        ) -> Result<PyObjectRef, crate::PyError> {
            let spec = parse_filter_spec(filter)?;
            let properties = backend::encode_filter_properties(&spec).map_err(lzma_error)?;
            Ok(bytesobject::w_bytes_from_bytes(&properties))
        }
        // `_lzma__decode_filter_properties_impl` — the spec dict those options
        // describe, id included.
        fn _decode_filter_properties(
            filter_id: PyIndexInt,
            encoded_props: PyBufferStr,
        ) -> Result<PyObjectRef, crate::PyError> {
            let Ok(filter_id) = u64::try_from(filter_id) else {
                return Err(crate::PyError::value_error(
                    "cannot convert negative integer to unsigned",
                ));
            };
            let decoded = backend::decode_filter_properties(filter_id, &encoded_props)
                .map_err(lzma_error)?;
            // `build_filter_spec`.  Each store can grow the dict, so the dict
            // is read back out of its slot for every one of them.
            let _roots = push_roots();
            let dict_slot = shadow_stack_len();
            pin_root(w_dict_new());
            let dict = || shadow_stack_get(dict_slot);
            let id = w_int_new(decoded.id as i64);
            unsafe { w_dict_setitem_str(dict(), "id", id) };
            for (name, value) in &decoded.fields {
                let value = w_int_new(*value as i64);
                unsafe { w_dict_setitem_str(dict(), name, value) };
            }
            Ok(dict())
        }
    },
    extra_init: |ns| {
        let _ = ns;
        // Neither type spec carries `Py_TPFLAGS_BASETYPE`.
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(
                compressor_methods::type_object(),
                false,
            );
            pyre_object::w_type_set_acceptable_as_base_class(
                decompressor_methods::type_object(),
                false,
            );
        }
    },
}
