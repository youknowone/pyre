//! _bz2 module — PyPy: `pypy/module/bz2/` (`moduledef.py` publishes
//! `interp_bz2.W_BZ2Compressor` / `W_BZ2Decompressor` under the applevel
//! name `_bz2`).
//!
//! `interp_bz2.py` keeps the libbz2 stream and its per-object lock on the
//! wrapper object; here the stream lives in `pyre_native::bz2`
//! (bzip2 / libbz2-rs-sys) outside the LLBC extraction, exactly as `zlib`
//! keeps its DEFLATE codec, and this module is the W_Root object glue.

use pyre_native::bz2 as backend;
use pyre_object::*;

use std::sync::Mutex;

/// `interp_bz2.py W_BZ2Compressor`: the stream and its lock belong to
/// the wrapper object; there is no process-global side table.
#[crate::pyre_class("_bz2.BZ2Compressor")]
#[derive(Default)]
pub struct W_BZ2Compressor {
    backend: *mut Mutex<backend::Compressor>,
}

/// `interp_bz2.py W_BZ2Decompressor`, object-owned the same way.
#[crate::pyre_class("_bz2.BZ2Decompressor")]
#[derive(Default)]
pub struct W_BZ2Decompressor {
    backend: *mut Mutex<backend::Decompressor>,
}

/// `interp_bz2.py _catch_bz2_error`, carrying the messages
/// `lib-python/3/test/test_bz2.py:1042` pins.
fn bz2_error(error: backend::Bz2Error) -> crate::PyError {
    match error {
        backend::Bz2Error::Param => {
            crate::PyError::value_error("Internal error - invalid parameters passed to libbzip2")
        }
        backend::Bz2Error::Data => crate::PyError::os_error("Invalid data stream"),
        backend::Bz2Error::Sequence => crate::PyError::runtime_error(
            "Internal error - Invalid sequence of commands sent to libbzip2",
        ),
        backend::Bz2Error::Mem => crate::PyError::memory_error("out of memory"),
    }
}

/// `interp_bz2.py descr_getstate` — neither object is serialisable.
fn cannot_serialize(name: &str) -> crate::PyError {
    crate::PyError::type_error(format!("cannot pickle '{name}' object"))
}

impl W_BZ2Compressor {
    fn compressor(&self) -> Result<&Mutex<backend::Compressor>, crate::PyError> {
        if self.backend.is_null() {
            return Err(crate::PyError::value_error(
                "Compressor was not initialized",
            ));
        }
        Ok(unsafe { &*self.backend })
    }
}

impl W_BZ2Decompressor {
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
               For one-shot compression, use the compress() function instead."
    )]
    impl W_BZ2Compressor {
        /// `interp_bz2.py descr_compressor__new__` — the compression
        /// level is checked before `BZ2_bzCompressInit` runs.
        #[staticmethod]
        fn __new__(
            _cls: PyObjectRef,
            #[default(9i32)] compresslevel: PyIndexCInt,
        ) -> Result<PyObjectRef, crate::PyError> {
            let Some(compressor) = backend::Compressor::new(compresslevel as i64) else {
                return Err(crate::PyError::value_error(
                    "compresslevel must be between 1 and 9",
                ));
            };
            Ok(W_BZ2Compressor::allocate_stable(W_BZ2Compressor {
                backend: Box::into_raw(Box::new(Mutex::new(compressor))),
                ..W_BZ2Compressor::default()
            }))
        }

        /// `interp_bz2.py compress`.
        fn compress(&mut self, data: PyBufferStr) -> Result<Vec<u8>, crate::PyError> {
            let mut compressor = self.compressor()?.lock().unwrap();
            if compressor.is_flushed() {
                return Err(crate::PyError::value_error("Compressor has been flushed"));
            }
            compressor.compress(&data).map_err(bz2_error)
        }

        /// `interp_bz2.py flush` — the object may not be used afterwards.
        fn flush(&mut self) -> Result<Vec<u8>, crate::PyError> {
            let mut compressor = self.compressor()?.lock().unwrap();
            if compressor.is_flushed() {
                return Err(crate::PyError::value_error("Repeated call to flush()"));
            }
            compressor.flush().map_err(bz2_error)
        }

        /// `interp_bz2.py descr_getstate`.
        fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
            Err(cannot_serialize("_bz2.BZ2Compressor"))
        }
    }
} // compressor_methods

mod decompressor_methods {
    use super::*;

    #[crate::pyre_methods(
        doc = "Create a decompressor object for decompressing data incrementally.\n\n\
               For one-shot decompression, use the decompress() function instead."
    )]
    impl W_BZ2Decompressor {
        /// `interp_bz2.py descr_decompressor__new__`.
        #[staticmethod]
        fn __new__(_cls: PyObjectRef) -> PyObjectRef {
            W_BZ2Decompressor::allocate_stable(W_BZ2Decompressor {
                backend: Box::into_raw(Box::new(Mutex::new(backend::Decompressor::new()))),
                ..W_BZ2Decompressor::default()
            })
        }

        /// `interp_bz2.py decompress` — a negative `max_length` is
        /// unlimited.
        fn decompress(
            &mut self,
            data: PyBufferStr,
            #[default(-1i64)] max_length: PyIndexInt,
        ) -> Result<Vec<u8>, crate::PyError> {
            let mut decompressor = self.decompressor()?.lock().unwrap();
            if decompressor.eof() {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::EOFError,
                    "End of stream already reached",
                ));
            }
            if decompressor.failed() {
                // Re-entering BZ2_bzDecompress after a failure can write out
                // of bounds, so a latched error refuses every later call.
                return Err(crate::PyError::value_error(
                    "Decompressor is unusable after a previous error",
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
                .map_err(bz2_error)
        }

        /// `interp_bz2.py eof_w` — true once the end-of-stream marker
        /// has been reached.
        #[getter]
        fn eof(&self) -> Result<bool, crate::PyError> {
            Ok(self.decompressor()?.lock().unwrap().eof())
        }

        /// `interp_bz2.py:541 unused_data` — data found after the end of the
        /// compressed stream.
        #[getter]
        fn unused_data(&self) -> Result<Vec<u8>, crate::PyError> {
            Ok(self.decompressor()?.lock().unwrap().unused_data().to_vec())
        }

        /// `interp_bz2.py needs_input_w` — true when more input is
        /// needed before more decompressed data can be produced.
        #[getter]
        fn needs_input(&self) -> Result<bool, crate::PyError> {
            Ok(self.decompressor()?.lock().unwrap().needs_input())
        }

        /// `interp_bz2.py descr_getstate`.
        fn __getstate__(&self) -> Result<PyObjectRef, crate::PyError> {
            Err(cannot_serialize("_bz2.BZ2Decompressor"))
        }
    }
} // decompressor_methods

/// Sweep-time counterpart of `interp_bz2.py _finalize_`.  The Box holds
/// the per-object mutex and the native stream; dropping it neither allocates
/// nor calls back into Python.
///
/// # Safety
/// `obj` must be a GC-dead `W_BZ2Compressor`.
pub unsafe fn w_bz2compressor_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_BZ2Compressor::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

/// Sweep-time counterpart of `interp_bz2.py _finalize_`.
///
/// # Safety
/// `obj` must be a GC-dead `W_BZ2Decompressor`.
pub unsafe fn w_bz2decompressor_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_BZ2Decompressor::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

crate::py_module! {
    "_bz2",
    interpleveldefs: {
        "BZ2Compressor" => compressor_methods::type_object(),
        "BZ2Decompressor" => decompressor_methods::type_object(),
    },
    extra_init: |ns| {
        let _ = ns;
        // `interp_bz2.py:389` / `:547 acceptable_as_base_class = False`.
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
