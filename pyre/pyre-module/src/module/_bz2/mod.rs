//! _bz2 module — PyPy: `pypy/module/bz2/` (`moduledef.py` publishes
//! `interp_bz2.W_BZ2Compressor` / `W_BZ2Decompressor` under the applevel
//! name `_bz2`).
//!
//! `interp_bz2.py` keeps the libbz2 stream and its per-object lock on the
//! wrapper object.  The stream machinery is shared with RustPython through
//! `rustpython_common::compression::bz2`, which Charon never enters because it
//! is a git dependency; this module is the W_Root object glue and keeps the
//! per-object lock.

use pyre_object::*;
use rustpython_common::compression::bz2 as backend;

use parking_lot::Mutex;

/// `interp_bz2.py W_BZ2Compressor`: the stream and its lock belong to
/// the wrapper object; there is no process-global side table.
// CPython 3.14 Modules/_bz2module.c:bz2_exec uses
// PyType_FromModuleAndSpec with IMMUTABLETYPE.
#[pyre_interpreter::pyre_class("_bz2.BZ2Compressor", cpython_heaptype)]
#[derive(Default)]
pub struct W_BZ2Compressor {
    backend: *mut Mutex<backend::Compressor>,
}

/// `interp_bz2.py W_BZ2Decompressor`, object-owned the same way.
// Same `bz2_exec` owner and flags as BZ2Compressor.
#[pyre_interpreter::pyre_class("_bz2.BZ2Decompressor", cpython_heaptype)]
#[derive(Default)]
pub struct W_BZ2Decompressor {
    backend: *mut Mutex<backend::Decompressor>,
}

/// `interp_bz2.py _catch_bz2_error`, carrying the messages
/// `lib-python/3/test/test_bz2.py:1042` pins.
fn bz2_error(error: backend::Bz2Error) -> pyre_interpreter::PyError {
    match error {
        backend::Bz2Error::Param => pyre_interpreter::PyError::value_error(
            "Internal error - invalid parameters passed to libbzip2",
        ),
        backend::Bz2Error::Data => pyre_interpreter::PyError::os_error("Invalid data stream"),
        backend::Bz2Error::Sequence => pyre_interpreter::PyError::runtime_error(
            "Internal error - Invalid sequence of commands sent to libbzip2",
        ),
        backend::Bz2Error::Mem => pyre_interpreter::PyError::memory_error("out of memory"),
    }
}

/// `interp_bz2.py descr_getstate` — neither object is serialisable.
fn cannot_serialize(name: &str) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::type_error(format!("cannot pickle '{name}' object"))
}

impl W_BZ2Compressor {
    fn compressor(&self) -> Result<&Mutex<backend::Compressor>, pyre_interpreter::PyError> {
        if self.backend.is_null() {
            return Err(pyre_interpreter::PyError::value_error(
                "Compressor was not initialized",
            ));
        }
        Ok(unsafe { &*self.backend })
    }
}

impl W_BZ2Decompressor {
    fn decompressor(&self) -> Result<&Mutex<backend::Decompressor>, pyre_interpreter::PyError> {
        if self.backend.is_null() {
            return Err(pyre_interpreter::PyError::value_error(
                "Decompressor was not initialized",
            ));
        }
        Ok(unsafe { &*self.backend })
    }
}

mod compressor_methods {
    use super::*;

    #[pyre_interpreter::pyre_methods(
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
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let Some(compressor) = backend::Compressor::new(compresslevel as i64) else {
                return Err(pyre_interpreter::PyError::value_error(
                    "compresslevel must be between 1 and 9",
                ));
            };
            Ok(W_BZ2Compressor::allocate_stable(W_BZ2Compressor {
                backend: Box::into_raw(Box::new(Mutex::new(compressor))),
                ..W_BZ2Compressor::default()
            }))
        }

        /// `interp_bz2.py compress`.
        fn compress(&mut self, data: PyBufferStr) -> Result<Vec<u8>, pyre_interpreter::PyError> {
            let mut compressor = self.compressor()?.lock();
            if compressor.is_flushed() {
                return Err(pyre_interpreter::PyError::value_error(
                    "Compressor has been flushed",
                ));
            }
            compressor.compress(&data).map_err(bz2_error)
        }

        /// `interp_bz2.py flush` — the object may not be used afterwards.
        fn flush(&mut self) -> Result<Vec<u8>, pyre_interpreter::PyError> {
            let mut compressor = self.compressor()?.lock();
            if compressor.is_flushed() {
                return Err(pyre_interpreter::PyError::value_error(
                    "Repeated call to flush()",
                ));
            }
            compressor.flush().map_err(bz2_error)
        }

        /// `interp_bz2.py descr_getstate`.
        fn __getstate__(&self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            Err(cannot_serialize("_bz2.BZ2Compressor"))
        }
    }
} // compressor_methods

mod decompressor_methods {
    use super::*;

    #[pyre_interpreter::pyre_methods(
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
        ) -> Result<Vec<u8>, pyre_interpreter::PyError> {
            let mut decompressor = self.decompressor()?.lock();
            if decompressor.eof() {
                return Err(pyre_interpreter::PyError::new(
                    pyre_interpreter::PyErrorKind::EOFError,
                    "End of stream already reached",
                ));
            }
            if decompressor.failed() {
                // Re-entering BZ2_bzDecompress after a failure can write out
                // of bounds, so a latched error refuses every later call.
                return Err(pyre_interpreter::PyError::value_error(
                    "Decompressor is unusable after a previous error",
                ));
            }
            // A cap too large for the platform's index type is an error, not
            // silently unlimited -- only a negative value means unlimited.
            let max_length = if max_length < 0 {
                None
            } else {
                Some(usize::try_from(max_length).map_err(|_| {
                    pyre_interpreter::PyError::overflow_error(
                        "Python int too large to convert to C ssize_t",
                    )
                })?)
            };
            decompressor
                .decompress(&data, max_length)
                .map_err(bz2_error)
        }

        /// `interp_bz2.py eof_w` — true once the end-of-stream marker
        /// has been reached.
        #[getter]
        fn eof(&self) -> Result<bool, pyre_interpreter::PyError> {
            Ok(self.decompressor()?.lock().eof())
        }

        /// `interp_bz2.py:541 unused_data` — data found after the end of the
        /// compressed stream.
        #[getter]
        fn unused_data(&self) -> Result<Vec<u8>, pyre_interpreter::PyError> {
            Ok(self.decompressor()?.lock().unused_data().to_vec())
        }

        /// `interp_bz2.py needs_input_w` — true when more input is
        /// needed before more decompressed data can be produced.
        #[getter]
        fn needs_input(&self) -> Result<bool, pyre_interpreter::PyError> {
            Ok(self.decompressor()?.lock().needs_input())
        }

        /// `interp_bz2.py descr_getstate`.
        fn __getstate__(&self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
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

pyre_interpreter::py_module! {
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

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcAnchor, ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `interp_bz2.py` keeps each libbz2 stream and its lock on the W_Root
    // owner.  Neither type is subclassable, so they carry no mapdict prefix and
    // no inline GC edge beyond the header's `w_class`; the sweep destructor
    // releases the native stream.
    types.push(ModuleGcType {
        anchor: ModuleGcAnchor::AfterGcStats,
        descriptor: <W_BZ2Compressor as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::Object,
        destructor: Some(gc_destructor!(w_bz2compressor_dealloc)),
    });
    types.push(ModuleGcType {
        anchor: ModuleGcAnchor::AfterGcStats,
        descriptor: <W_BZ2Decompressor as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::Object,
        destructor: Some(gc_destructor!(w_bz2decompressor_dealloc)),
    });
}
