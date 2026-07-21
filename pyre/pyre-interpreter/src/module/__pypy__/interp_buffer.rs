//! `__pypy__.PickleBuffer` — `pypy/module/__pypy__/interp_buffer.py
//! W_PickleBuffer`. Wraps a bytes-like object so the `_pickle` accelerator
//! can serialize it either in-band or out-of-band (protocol 5). The
//! `_pickle` save path recognizes the wrapper via `from_obj` and reads its
//! contents through `buffer_view`.

use pyre_object::PyObjectRef;

use crate::PyError;

#[crate::pyre_class("__pypy__.PickleBuffer")]
pub struct W_PickleBuffer {
    /// The wrapped buffer-supporting object, or `None` after `release()`.
    w_obj: PyObjectRef,
    /// `self.buf is not None and self.buf.needs_release()`: owns exactly one
    /// exporter release until `release()` or `_finalize_` takes it.
    export_active: bool,
    /// The generic `W_Root.__buffer_w` path returns a temporary memoryview.
    /// Keep and release that carrier as part of `self.buf`; a concrete
    /// exporter supplied directly by the caller is borrowed instead.
    release_memoryview: bool,
    /// The PEP 688 exporter paired with the temporary memoryview, or `None`
    /// for a native exporter.  This is the inline owner corresponding to the
    /// buffer protocol's `__release_buffer__(view)` obligation.
    w_release_exporter: PyObjectRef,
}

#[crate::pyre_methods(
    doc = "PickleBuffer(buffer) -> wrapper for potentially out-of-band serialization.",
    // interp_buffer.py:209 make_weakref_descr(W_PickleBuffer)
    weakrefable
)]
impl W_PickleBuffer {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef, w_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
        // interp_buffer.py:201-203 descr_new_picklebuffer — acquire the
        // export while constructing the object; PickleBuffer has no separate
        // __init__ phase.
        let (w_buffer, release_memoryview, w_release_exporter) = acquire_pickle_buffer(w_obj)?;
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_buffer);
        pyre_object::gc_roots::pin_root(w_release_exporter);
        // `allocate` may collect; store the post-collection exporter rather
        // than the stale Rust-stack copy.
        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        let export_active = unsafe { crate::builtins::buffer_export_incref(r_obj) };
        // Like the other self-mutating interp-level payloads, keep the
        // wrapper stationary.  More importantly for `register_finalizer`, an
        // acquired export must enter the old-object finalizer queue at its
        // definitive address; a nursery registration is first promoted as a
        // finalizer root and otherwise delays `_release_buf` by a collection.
        let w_pickle_buffer = W_PickleBuffer::allocate_stable(W_PickleBuffer {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_obj: r_obj,
            export_active,
            release_memoryview,
            w_release_exporter: pyre_object::gc_roots::shadow_stack_get(sp + 1),
        });
        // Pyre also routes weakref invalidation through this finalizer queue,
        // so register immutable-buffer wrappers too: PyPy's collector handles
        // their weakrefs independently even when `buf.needs_release()` is
        // false.  The idempotent release body is a no-op for that export.
        crate::executioncontext::register_native_buffer_finalizer(w_pickle_buffer);
        Ok(w_pickle_buffer)
    }

    /// `raw()` — a memoryview of the raw bytes underlying the wrapped buffer.
    /// The result is a one-dimensional unsigned-byte view (`format='B'`,
    /// itemsize 1) that aliases the source and preserves its read-only flag,
    /// regardless of the source's element format; extracting it from a
    /// non-contiguous buffer raises `BufferError`.
    fn raw(&self) -> Result<PyObjectRef, PyError> {
        let w_obj = self.w_obj;
        if unsafe { pyre_object::is_none(w_obj) } {
            return Err(released_error());
        }
        let mv_type = memoryview_type()
            .ok_or_else(|| PyError::runtime_error("memoryview type unavailable"))?;
        let mv = crate::module::_pickle::call_fn(mv_type, &[w_obj])?;
        // Raw extraction is only defined for a C-contiguous buffer.
        let w_contig = crate::baseobjspace::getattr_str(mv, "c_contiguous")?;
        if !crate::baseobjspace::is_true(w_contig)? {
            return Err(PyError::new(
                crate::PyErrorKind::BufferError,
                "cannot extract raw buffer from non-contiguous buffer",
            ));
        }
        // Normalize to the raw byte layout via `cast('B')` so an `array('i')`
        // or other non-`'B'` source still yields a byte view.
        crate::module::_pickle::call_meth(mv, "cast", &[pyre_object::unicodeobject::w_str_new("B")])
    }

    /// `release()` — drop the reference to the underlying buffer.
    fn release(&mut self) -> Result<(), PyError> {
        self.release_export_result()
    }
}

impl W_PickleBuffer {
    /// `_release_buf` (`interp_buffer.py:146-150`) — clear ownership before
    /// calling the exporter release, making repeated explicit/finalizer calls
    /// idempotent.
    pub(crate) fn release_export(&mut self) {
        let _ = self.release_export_result();
    }

    fn release_export_result(&mut self) -> Result<(), PyError> {
        if unsafe { pyre_object::is_none(self.w_obj) } {
            return Ok(());
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(self.w_obj);
        pyre_object::gc_roots::pin_root(self.w_release_exporter);
        let w_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        if self.export_active {
            self.export_active = false;
            unsafe { crate::builtins::buffer_export_decref(w_obj) };
        }
        self.w_obj = pyre_object::w_none();
        self.w_release_exporter = pyre_object::w_none();
        if self.release_memoryview {
            self.release_memoryview = false;
            let w_exporter = pyre_object::gc_roots::shadow_stack_get(sp + 1);
            let callback_result = if let Some(w_release) =
                unsafe { crate::baseobjspace::lookup(w_exporter, "__release_buffer__") }
            {
                let w_type = crate::typedef::r#type(w_exporter).unwrap_or(w_exporter);
                unsafe {
                    crate::baseobjspace::get_and_call_function(
                        w_release,
                        w_exporter,
                        w_type,
                        &[pyre_object::gc_roots::shadow_stack_get(sp)],
                    )
                }
                .map(|_| ())
            } else {
                Ok(())
            };
            // The callback is advisory cleanup; the temporary carrier still
            // has to be released if it raises, matching a try/finally around
            // the acquired Py_buffer.
            let release_result =
                crate::builtins::memoryview_release(&[pyre_object::gc_roots::shadow_stack_get(sp)])
                    .map(|_| ());
            callback_result?;
            release_result?;
        }
        Ok(())
    }
}

/// `W_PickleBuffer.typedef.acceptable_as_base_class = False` — return the
/// shared type after applying PyPy's explicit final-type flag. Both
/// `__pypy__.PickleBuffer` and `_pickle.PickleBuffer` call this accessor, so
/// the flag is installed regardless of which module is imported first.
pub(crate) fn picklebuffer_type_object() -> PyObjectRef {
    let tp = type_object();
    unsafe { pyre_object::w_type_set_acceptable_as_base_class(tp, false) };
    tp
}

impl W_PickleBuffer {
    /// The wrapped buffer object (`None` after `release()`), read by the
    /// `_pickle` save path.
    pub(crate) fn wrapped(&self) -> PyObjectRef {
        self.w_obj
    }
}

/// If `obj` is a `PickleBuffer`, the wrapped exporter its buffer protocol
/// forwards to — `buffer_w` delegates to the underlying object, so `bytes(pb)`
/// / `memoryview(pb)` operate on the wrapped `bytes`/`bytearray`/`array`/
/// `memoryview`. `Some(Err(..))` once the buffer was released; `None` when
/// `obj` is not a `PickleBuffer`.
pub(crate) fn forwarded_exporter(obj: PyObjectRef) -> Option<Result<PyObjectRef, PyError>> {
    W_PickleBuffer::from_obj(obj).map(|pb| {
        let w = pb.wrapped();
        if unsafe { pyre_object::is_none(w) } {
            Err(released_error())
        } else {
            Ok(w)
        }
    })
}

fn released_error() -> PyError {
    PyError::value_error("operation forbidden on released PickleBuffer object")
}

fn type_name(obj: PyObjectRef) -> String {
    match crate::typedef::r#type(obj) {
        Some(t) => unsafe { pyre_object::w_type_get_name(t) }.to_string(),
        None => "object".to_string(),
    }
}

/// `space.buffer_w(w_obj, BUF_FULL_RO)` for PickleBuffer construction.
///
/// Concrete native exporters keep their identity.  The W_Root fallback is
/// the Python 3.14/PyPy PEP 688 path: look up and descriptor-bind
/// `__buffer__`, pass `BUF_FULL_RO`, and require a memoryview result.  The
/// returned boolean says that the result is the temporary carrier owned by
/// this acquisition and must be released with the PickleBuffer.
fn acquire_pickle_buffer(obj: PyObjectRef) -> Result<(PyObjectRef, bool, PyObjectRef), PyError> {
    unsafe {
        if pyre_object::is_bytes(obj)
            || pyre_object::is_bytearray(obj)
            || pyre_object::interp_array::is_array(obj)
            || is_memoryview(obj)
        {
            if is_memoryview(obj) {
                crate::builtins::memoryview_check_released(obj)?;
            }
            return Ok((obj, false, pyre_object::w_none()));
        }

        const BUF_FULL_RO: i64 = 0x011c;
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(obj);
        pyre_object::gc_roots::pin_root(pyre_object::w_int_new(BUF_FULL_RO));
        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        if let Some(w_impl) = crate::baseobjspace::lookup(r_obj, "__buffer__") {
            pyre_object::gc_roots::pin_root(w_impl);
            let w_type = crate::typedef::r#type(r_obj).unwrap_or(r_obj);
            let w_result = crate::baseobjspace::get_and_call_function(
                pyre_object::gc_roots::shadow_stack_get(sp + 2),
                r_obj,
                w_type,
                &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
            )?;
            if is_memoryview(w_result) {
                crate::builtins::memoryview_check_released(w_result)?;
                return Ok((w_result, true, pyre_object::gc_roots::shadow_stack_get(sp)));
            }
            return Err(PyError::type_error(format!(
                "a bytes-like object is required, not '{}'",
                type_name(pyre_object::gc_roots::shadow_stack_get(sp))
            )));
        }
        // Interp-level exporters such as mmap / ctypes expose their native
        // `buffer_w` implementation without a Python-visible `__buffer__`
        // descriptor.  Normalize those to the same owned memoryview carrier.
        let w_result =
            crate::builtins::w_memoryview_new(pyre_object::gc_roots::shadow_stack_get(sp))?;
        return Ok((w_result, true, pyre_object::w_none()));
    }
}

fn is_memoryview(obj: PyObjectRef) -> bool {
    unsafe { pyre_object::memoryview::is_w_memoryview(obj) }
}

/// Extract `(contents, readonly)` from a buffer exporter: `bytes` is
/// read-only, `bytearray` and `array` are mutable, and a `memoryview` reports
/// both through its own contents and `readonly` flag.
pub(crate) fn buffer_view(obj: PyObjectRef) -> Result<(Vec<u8>, bool), PyError> {
    unsafe {
        if pyre_object::is_bytes(obj) {
            return Ok((pyre_object::bytesobject::w_bytes_data(obj).to_vec(), true));
        }
        if pyre_object::is_bytearray(obj) {
            return Ok((
                pyre_object::bytearrayobject::w_bytearray_data(obj).to_vec(),
                false,
            ));
        }
        if pyre_object::interp_array::is_array(obj) {
            return Ok((
                pyre_object::interp_array::w_array_bytes(obj).to_vec(),
                false,
            ));
        }
    }
    if is_memoryview(obj) {
        let w_data = crate::module::_pickle::call_meth(obj, "tobytes", &[])?;
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_data) }.to_vec();
        let w_ro = crate::baseobjspace::getattr_str(obj, "readonly")?;
        return Ok((data, crate::baseobjspace::is_true(w_ro)?));
    }
    Err(PyError::type_error(format!(
        "a bytes-like object is required, not '{}'",
        type_name(obj)
    )))
}

/// Whether the wrapped exporter's buffer is C-contiguous, matching the
/// `_pickle` save path's `iscontiguous(buf)` guard. `bytes`/`bytearray`/`array`
/// are one-dimensional and always contiguous; a `memoryview` reports through
/// its `c_contiguous` flag.
pub(crate) fn is_contiguous(obj: PyObjectRef) -> Result<bool, PyError> {
    if is_memoryview(obj) {
        let w = crate::baseobjspace::getattr_str(obj, "c_contiguous")?;
        return crate::baseobjspace::is_true(w);
    }
    Ok(true)
}

/// The `memoryview` builtin type via the live execution context.
fn memoryview_type() -> Option<PyObjectRef> {
    let frame = crate::eval::CURRENT_FRAME.with(|f| f.get());
    if frame.is_null() {
        return None;
    }
    let ec = unsafe { (*frame).execution_context };
    if ec.is_null() {
        return None;
    }
    unsafe { (*ec).lookup_builtin("memoryview") }
}
