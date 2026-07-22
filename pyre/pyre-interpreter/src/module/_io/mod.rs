//! _io module — PyPy: pypy/module/_io/
//!
//! Pyre stubs the bulk of the C IO classes: ctors return None / "" and
//! ABC base classes (`_IOBase` / `_RawIOBase` / `_BufferedIOBase` /
//! `_TextIOBase`) are exposed as plain types so io.py's class
//! inheritance succeeds.

use pyre_object::*;

// The module-local exception class is process-global, like PyPy's module
// definition object.  Keep the immortal type pointer shared across threads;
// runtime semantic state must not be duplicated in TLS.
static UNSUPPORTED_OPERATION_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn type_method(ns: PyObjectRef, name: &str, function: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, function);
    }
}

fn io_closed(obj: PyObjectRef) -> bool {
    crate::baseobjspace::getattr_str(obj, "closed")
        .ok()
        .and_then(|value| unsafe {
            pyre_object::is_bool(value).then(|| pyre_object::w_bool_get_value(value))
        })
        .unwrap_or(false)
}

fn iobase_internal_closed(obj: PyObjectRef) -> bool {
    crate::baseobjspace::getattr_str(obj, "__iobase_closed__")
        .ok()
        .is_some_and(|value| unsafe {
            pyre_object::is_bool(value) && pyre_object::w_bool_get_value(value)
        })
}

fn iobase_set_internal_closed(obj: PyObjectRef, closed: bool) -> Result<(), crate::PyError> {
    if crate::baseobjspace::setdictvalue(obj, "__iobase_closed__", w_bool_from(closed)) {
        Ok(())
    } else {
        Err(crate::PyError::runtime_error(
            "_IOBase instance has no state dictionary",
        ))
    }
}

/// `interp_iobase.py:unsupported` — construct the module-local
/// `UnsupportedOperation`, preserving its OSError + ValueError MRO.
pub(crate) fn unsupported(message: &str) -> crate::PyError {
    let Some(&type_addr) = UNSUPPORTED_OPERATION_TYPE.get() else {
        return crate::PyError::value_error(message);
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_str_new(message));
    match crate::call::call_function_impl_result(
        type_addr as PyObjectRef,
        &[pyre_object::gc_roots::shadow_stack_get(sp)],
    ) {
        Ok(exc) => unsafe { crate::PyError::from_exc_object(exc) },
        Err(error) => error,
    }
}

fn iobase_close(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("close() requires self"))?;
    if iobase_internal_closed(self_obj) {
        return Ok(w_none());
    }
    // PyPy `close_w`: `flush()` runs while `closed` is still false, and the
    // internal flag is set in `finally`, even when the virtual flush raises.
    let flushed = call_method_result(self_obj, "flush", &[]);
    iobase_set_internal_closed(self_obj, true)?;
    flushed.map(|_| w_none())
}

fn iobase_flush(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("flush() requires self"))?;
    if iobase_internal_closed(self_obj) {
        return Err(crate::PyError::value_error("I/O operation on closed file"));
    }
    Ok(w_none())
}

fn iobase_closed_get(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .get(1)
        .copied()
        .ok_or_else(|| crate::PyError::type_error("descriptor requires an instance"))?;
    Ok(w_bool_from(iobase_internal_closed(self_obj)))
}

fn iobase_check_closed(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("_checkClosed() requires self"))?;
    let message = args
        .get(1)
        .copied()
        .filter(|value| !unsafe { pyre_object::is_none(*value) })
        .map(|value| unsafe { pyre_object::w_str_get_value(value).to_string() })
        .unwrap_or_else(|| "I/O operation on closed file".to_string());
    if io_closed(self_obj) {
        Err(crate::PyError::value_error(message))
    } else {
        Ok(w_none())
    }
}

fn iobase_check_capability(args: &[PyObjectRef], method: &str, message: &str) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error(format!("{method}() requires self")))?;
    let capable = call_method_result(self_obj, method, &[])?;
    if crate::baseobjspace::is_true(capable)? {
        Ok(w_none())
    } else {
        Err(unsupported(message))
    }
}

fn iobase_unsupported(args: &[PyObjectRef], operation: &str) -> crate::PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "{operation}() requires self"
        )));
    }
    Err(unsupported(operation))
}

fn iobase_seek(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "seek")
}

fn iobase_truncate(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "truncate")
}

fn iobase_fileno(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "fileno")
}

fn iobase_tell(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("tell() requires self"))?;
    call_method_result(self_obj, "seek", &[w_int_new(0), w_int_new(1)])
}

fn iobase_enter(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__enter__() requires self"))?;
    iobase_check_closed(&[self_obj])?;
    Ok(self_obj)
}

fn iobase_iter(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__iter__() requires self"))?;
    iobase_check_closed(&[self_obj])?;
    Ok(self_obj)
}

fn iobase_next(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__next__() requires self"))?;
    let line = call_method_result(self_obj, "readline", &[])?;
    if crate::baseobjspace::len_w(line)? == 0 {
        Err(crate::PyError::stop_iteration())
    } else {
        Ok(line)
    }
}

fn iobase_del(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&self_obj) = args.first() else {
        return Ok(w_none());
    };
    // PyPy `descr_del` deliberately suppresses failures while finalizing.
    if !io_closed(self_obj) {
        let _ = call_method_result(self_obj, "close", &[]);
    }
    Ok(w_none())
}

fn iobase_getstate(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__getstate__() requires self"))?;
    Err(crate::PyError::type_error(format!(
        "cannot serialize '{}' object",
        crate::type_methods::arg_type_name(self_obj)
    )))
}

fn iobase_isatty(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("isatty() requires self"))?;
    if io_closed(self_obj) {
        return Err(crate::PyError::value_error("I/O operation on closed file"));
    }
    Ok(w_bool_from(false))
}

/// `interp_iobase.py:303-322 W_IOBase.writelines_w` — validate the stream,
/// obtain the input iterator, and call the receiver's (possibly overridden)
/// `write` method once for each line.  The iteration is deliberately lazy;
/// no list snapshot is introduced.
pub(crate) fn iobase_writelines(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("writelines() requires self"))?;
    // The registered arity is only a fast-dispatch hint, so surplus
    // positionals still arrive here.  PyPy's interp2app gateway exposes one
    // argument after the bound receiver and rejects the call before it can
    // iterate or write anything.
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "{}.writelines() takes exactly one argument ({} given)",
            crate::type_methods::arg_type_name(self_obj),
            args.len() - 1,
        )));
    }
    let lines = args[1];
    if io_closed(self_obj) {
        return Err(crate::PyError::value_error("I/O operation on closed file."));
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    pyre_object::gc_roots::pin_root(lines);
    let iterator = crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(sp + 1))?;
    pyre_object::gc_roots::pin_root(iterator);
    loop {
        let iterator = pyre_object::gc_roots::shadow_stack_get(sp + 2);
        let line = match crate::baseobjspace::next(iterator) {
            Ok(line) => line,
            Err(err) if err.kind == crate::PyErrorKind::StopIteration => break,
            Err(err) => return Err(err),
        };
        let _line_root = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(line);
        let line =
            pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1);
        call_method_result(
            pyre_object::gc_roots::shadow_stack_get(sp),
            "write",
            &[line],
        )?;
    }
    Ok(w_none())
}

fn iobase_convert_size(value: Option<PyObjectRef>) -> Result<i64, crate::PyError> {
    match value {
        None => Ok(-1),
        Some(value) if unsafe { pyre_object::is_none(value) } => Ok(-1),
        Some(value) => crate::baseobjspace::int_w(crate::baseobjspace::space_index(value)?),
    }
}

/// `interp_iobase.py:W_IOBase.readline_w` — backwards-compatible mixin over
/// virtual `peek` and `read`, including the one-byte fallback.
fn iobase_readline(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("readline() requires self"))?;
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "readline() takes at most one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let limit = iobase_convert_size(args.get(1).copied())?;
    let peek = match crate::baseobjspace::getattr_str(self_obj, "peek") {
        Ok(method) => Some(method),
        Err(error) if error.kind == crate::PyErrorKind::AttributeError => None,
        Err(error) => return Err(error),
    };

    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    pyre_object::gc_roots::pin_root(peek.unwrap_or(PY_NULL));
    let mut output = Vec::new();
    while limit < 0 || output.len() < limit as usize {
        let mut nreadahead = 1usize;
        let peek = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        if !peek.is_null() {
            let readahead = crate::call::call_function_impl_result(peek, &[w_int_new(1)])?;
            if !unsafe { pyre_object::bytesobject::is_bytes_like(readahead) } {
                return Err(crate::PyError::os_error(format!(
                    "peek() should have returned a bytes object, not '{}'",
                    crate::type_methods::arg_type_name(readahead)
                )));
            }
            let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(readahead) };
            if !bytes.is_empty() {
                let remaining = if limit < 0 {
                    bytes.len()
                } else {
                    (limit as usize - output.len()).min(bytes.len())
                };
                nreadahead = bytes[..remaining]
                    .iter()
                    .position(|byte| *byte == b'\n')
                    .map_or(remaining, |index| index + 1);
            }
        }

        let read = call_method_result(
            pyre_object::gc_roots::shadow_stack_get(sp),
            "read",
            &[w_int_new(nreadahead as i64)],
        )?;
        if !unsafe { pyre_object::bytesobject::is_bytes_like(read) } {
            return Err(crate::PyError::os_error(format!(
                "peek() should have returned a bytes object, not '{}'",
                crate::type_methods::arg_type_name(read)
            )));
        }
        let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(read) };
        if bytes.is_empty() {
            break;
        }
        output.extend_from_slice(bytes);
        if bytes.last() == Some(&b'\n') {
            break;
        }
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
}

/// `interp_iobase.py:W_IOBase.readlines_w` — consume the stream iterator,
/// stopping after the accumulated line lengths exceed a positive hint.
fn iobase_readlines(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("readlines() requires self"))?;
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "readlines() takes at most one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let hint = iobase_convert_size(args.get(1).copied())?;
    let iterator = crate::baseobjspace::iter(self_obj)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(iterator);
    let lines_sp = pyre_object::gc_roots::shadow_stack_len();
    let mut count = 0usize;
    let mut length = 0i64;
    loop {
        let line = match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(sp)) {
            Ok(line) => line,
            Err(error) if error.kind == crate::PyErrorKind::StopIteration => break,
            Err(error) => return Err(error),
        };
        length = length.saturating_add(crate::baseobjspace::len_w(line)?);
        pyre_object::gc_roots::pin_root(line);
        count += 1;
        if hint > 0 && length > hint {
            break;
        }
    }
    let lines = (0..count)
        .map(|index| pyre_object::gc_roots::shadow_stack_get(lines_sp + index))
        .collect();
    Ok(w_list_new(lines))
}

fn init_iobase_type(ns: PyObjectRef) {
    // interp_iobase.py:333-358 W_IOBase.typedef declares both descriptors
    // in the raw typedef.  They must be present before the type/layout is
    // built: setting only the hasdict/weakrefable flags afterwards leaves
    // `_IOBase()` without the observable `__dict__` descriptor.
    type_method(ns, "__dict__", crate::typedef::dict_descr());
    type_method(ns, "__weakref__", crate::typedef::weakref_descr());
    let closed_getter = crate::make_builtin_function_with_arity("closed", iobase_closed_get, 2);
    type_method(
        ns,
        "closed",
        crate::typedef::make_getset_descriptor_named(closed_getter, "closed"),
    );
    type_method(
        ns,
        "close",
        crate::make_builtin_function_with_arity("close", iobase_close, 1),
    );
    type_method(
        ns,
        "flush",
        crate::make_builtin_function_with_arity("flush", iobase_flush, 1),
    );
    for (name, function) in [
        ("seek", iobase_seek as crate::gateway::BuiltinCodeFn),
        ("truncate", iobase_truncate as crate::gateway::BuiltinCodeFn),
        ("fileno", iobase_fileno as crate::gateway::BuiltinCodeFn),
    ] {
        type_method(ns, name, crate::make_builtin_function(name, function));
    }
    type_method(
        ns,
        "tell",
        crate::make_builtin_function_with_arity("tell", iobase_tell, 1),
    );
    for name in ["readable", "writable", "seekable"] {
        type_method(
            ns,
            name,
            crate::make_builtin_function_with_arity(name, |_| Ok(w_bool_from(false)), 1),
        );
    }
    type_method(
        ns,
        "_checkReadable",
        crate::make_builtin_function_with_arity(
            "_checkReadable",
            |args| iobase_check_capability(args, "readable", "File or stream is not readable"),
            1,
        ),
    );
    type_method(
        ns,
        "_checkWritable",
        crate::make_builtin_function_with_arity(
            "_checkWritable",
            |args| iobase_check_capability(args, "writable", "File or stream is not writable"),
            1,
        ),
    );
    type_method(
        ns,
        "_checkSeekable",
        crate::make_builtin_function_with_arity(
            "_checkSeekable",
            |args| iobase_check_capability(args, "seekable", "File or stream is not seekable"),
            1,
        ),
    );
    type_method(
        ns,
        "_checkClosed",
        crate::make_builtin_function("_checkClosed", iobase_check_closed),
    );
    type_method(
        ns,
        "isatty",
        crate::make_builtin_function_with_arity("isatty", iobase_isatty, 1),
    );
    type_method(
        ns,
        "readline",
        crate::make_builtin_function("readline", iobase_readline),
    );
    type_method(
        ns,
        "readlines",
        crate::make_builtin_function("readlines", iobase_readlines),
    );
    type_method(
        ns,
        "writelines",
        crate::make_builtin_function_with_arity("writelines", iobase_writelines, 2),
    );
    type_method(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity("__enter__", iobase_enter, 1),
    );
    type_method(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            iobase_close(&args[..1])?;
            Ok(w_none())
        }),
    );
    type_method(
        ns,
        "__iter__",
        crate::make_builtin_function_with_arity("__iter__", iobase_iter, 1),
    );
    type_method(
        ns,
        "__next__",
        crate::make_builtin_function_with_arity("__next__", iobase_next, 1),
    );
    type_method(
        ns,
        "__del__",
        crate::make_builtin_function_with_arity("__del__", iobase_del, 1),
    );
    type_method(
        ns,
        "__getstate__",
        crate::make_builtin_function_with_arity("__getstate__", iobase_getstate, 1),
    );
}

/// `interp_iobase.py:rawiobase_read_w` — the default raw `read` is a
/// one-shot `readinto` over a freshly allocated bytearray.  A negative or
/// omitted size delegates to the virtual `readall` method.
fn rawiobase_read(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("read() requires self"))?;
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "read() takes at most one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let size = match args.get(1).copied() {
        None => -1,
        Some(value) if unsafe { pyre_object::is_none(value) } => -1,
        Some(value) => crate::baseobjspace::int_w(crate::baseobjspace::space_index(value)?)?,
    };
    if size < 0 {
        return call_method_result(self_obj, "readall", &[]);
    }
    let size = usize::try_from(size).map_err(|_| {
        crate::PyError::overflow_error("Python int too large to convert to C ssize_t")
    })?;

    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    pyre_object::gc_roots::pin_root(pyre_object::bytearrayobject::w_bytearray_new(size));
    let length = call_method_result(
        pyre_object::gc_roots::shadow_stack_get(sp),
        "readinto",
        &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
    )?;
    if unsafe { pyre_object::is_none(length) } {
        return Ok(length);
    }
    let length = crate::baseobjspace::int_w(length)?;
    if length < 0 || length as u128 > size as u128 {
        return Err(crate::PyError::value_error(format!(
            "readinto returned {length} outside buffer size {size}"
        )));
    }
    let buffer = pyre_object::gc_roots::shadow_stack_get(sp + 1);
    let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(buffer) };
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(
        &data[..length as usize],
    ))
}

/// `interp_iobase.py:rawiobase_readall_w` — repeatedly invoke the virtual
/// limited `read(DEFAULT_BUFFER_SIZE)` until EOF.  `None` is propagated only
/// when no bytes have yet been accumulated.
fn rawiobase_readall(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("readall() requires self"))?;
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "readall() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_obj);
    let mut output = Vec::new();
    loop {
        let data = call_method_result(
            pyre_object::gc_roots::shadow_stack_get(sp),
            "read",
            &[w_int_new(8192)],
        )?;
        if unsafe { pyre_object::is_none(data) } {
            if output.is_empty() {
                return Ok(data);
            }
            break;
        }
        if !unsafe { pyre_object::bytesobject::is_bytes_like(data) } {
            return Err(crate::PyError::type_error("read() should return bytes"));
        }
        let chunk = unsafe { pyre_object::bytesobject::bytes_like_data(data) };
        if chunk.is_empty() {
            break;
        }
        output.extend_from_slice(chunk);
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&output))
}

fn init_rawiobase_type(ns: PyObjectRef) {
    type_method(
        ns,
        "read",
        crate::make_builtin_function("read", rawiobase_read),
    );
    type_method(
        ns,
        "readall",
        crate::make_builtin_function_with_arity("readall", rawiobase_readall, 1),
    );
}

fn buffered_iobase_read(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "read")
}

fn buffered_iobase_read1(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "read1")
}

fn buffered_iobase_write(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "write")
}

fn buffered_iobase_detach(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "detach")
}

/// `interp_bufferedio.py:W_BufferedIOBase._readinto` — acquire one writable
/// view, call the virtual `read`/`read1` once, bounds-check the returned bytes,
/// then copy them into the exact exported window.
fn buffered_iobase_readinto_impl(args: &[PyObjectRef], read_once: bool) -> crate::PyResult {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "{}() takes exactly one argument ({} given)",
            if read_once { "readinto1" } else { "readinto" },
            args.len().saturating_sub(1)
        )));
    }
    let self_obj = args[0];
    let mut buffer = unsafe { crate::builtins::WritableBuffer::acquire(args[1]) }?;
    let target = unsafe { buffer.as_mut_slice() };
    let method = if read_once { "read1" } else { "read" };
    let data = call_method_result(self_obj, method, &[w_int_new(target.len() as i64)])?;
    if !unsafe { pyre_object::bytesobject::is_bytes_like(data) } {
        return Err(crate::PyError::type_error(format!(
            "{method}() should return bytes"
        )));
    }
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(data) };
    if data.len() > target.len() {
        return Err(crate::PyError::value_error(format!(
            "{method}() returned too much data: {} bytes requested, {} returned",
            target.len(),
            data.len()
        )));
    }
    target[..data.len()].copy_from_slice(data);
    Ok(w_int_new(data.len() as i64))
}

fn buffered_iobase_readinto(args: &[PyObjectRef]) -> crate::PyResult {
    buffered_iobase_readinto_impl(args, false)
}

fn buffered_iobase_readinto1(args: &[PyObjectRef]) -> crate::PyResult {
    buffered_iobase_readinto_impl(args, true)
}

fn init_buffered_iobase_type(ns: PyObjectRef) {
    for (name, function) in [
        (
            "read",
            buffered_iobase_read as crate::gateway::BuiltinCodeFn,
        ),
        (
            "read1",
            buffered_iobase_read1 as crate::gateway::BuiltinCodeFn,
        ),
        (
            "write",
            buffered_iobase_write as crate::gateway::BuiltinCodeFn,
        ),
        (
            "detach",
            buffered_iobase_detach as crate::gateway::BuiltinCodeFn,
        ),
    ] {
        type_method(ns, name, crate::make_builtin_function(name, function));
    }
    type_method(
        ns,
        "readinto",
        crate::make_builtin_function_with_arity("readinto", buffered_iobase_readinto, 2),
    );
    type_method(
        ns,
        "readinto1",
        crate::make_builtin_function_with_arity("readinto1", buffered_iobase_readinto1, 2),
    );
}

fn text_iobase_read(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "read")
}

fn text_iobase_readline(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "readline")
}

fn text_iobase_write(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "write")
}

fn text_iobase_detach(args: &[PyObjectRef]) -> crate::PyResult {
    iobase_unsupported(args, "detach")
}

fn text_iobase_none_get(args: &[PyObjectRef]) -> crate::PyResult {
    if args.get(1).is_none() {
        return Err(crate::PyError::type_error(
            "descriptor requires an instance",
        ));
    }
    Ok(w_none())
}

fn init_text_iobase_type(ns: PyObjectRef) {
    for (name, function) in [
        ("read", text_iobase_read as crate::gateway::BuiltinCodeFn),
        (
            "readline",
            text_iobase_readline as crate::gateway::BuiltinCodeFn,
        ),
        ("write", text_iobase_write as crate::gateway::BuiltinCodeFn),
        (
            "detach",
            text_iobase_detach as crate::gateway::BuiltinCodeFn,
        ),
    ] {
        type_method(ns, name, crate::make_builtin_function(name, function));
    }
    for name in ["encoding", "newlines", "errors"] {
        let getter = crate::make_builtin_function_with_arity(name, text_iobase_none_get, 2);
        type_method(
            ns,
            name,
            crate::typedef::make_getset_descriptor_named(getter, name),
        );
    }
}

fn call_method_result(obj: PyObjectRef, name: &str, args: &[PyObjectRef]) -> crate::PyResult {
    let result = crate::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::runtime_error(format!("{name} failed"))))
    } else {
        Ok(result)
    }
}

fn buffered_reader_init(args: &[PyObjectRef]) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("BufferedReader.__init__() missing self"))?;
    let raw = args
        .get(1)
        .copied()
        .ok_or_else(|| crate::PyError::type_error("BufferedReader() missing raw argument"))?;
    crate::baseobjspace::setattr_str(self_obj, "raw", raw)?;
    crate::baseobjspace::setattr_str(self_obj, "closed", w_bool_from(false))?;
    Ok(w_none())
}

fn buffered_reader_call(args: &[PyObjectRef], method: &str) -> crate::PyResult {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error(format!("{method}() requires self")))?;
    let raw = crate::baseobjspace::getattr_str(self_obj, "raw")?;
    call_method_result(raw, method, &args[1..])
}

fn buffered_reader_close(args: &[PyObjectRef]) -> crate::PyResult {
    let result = buffered_reader_call(args, "close")?;
    crate::baseobjspace::setattr_str(args[0], "closed", w_bool_from(true))?;
    Ok(result)
}

fn init_buffered_reader_type(ns: PyObjectRef) {
    type_method(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", buffered_reader_init),
    );
    type_method(
        ns,
        "read",
        crate::make_builtin_function("read", |args| buffered_reader_call(args, "read")),
    );
    type_method(
        ns,
        "seekable",
        crate::make_builtin_function_with_arity(
            "seekable",
            |args| buffered_reader_call(args, "seekable"),
            1,
        ),
    );
    type_method(
        ns,
        "close",
        crate::make_builtin_function_with_arity("close", buffered_reader_close, 1),
    );
    type_method(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity("__enter__", |args| Ok(args[0]), 1),
    );
    type_method(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            buffered_reader_close(&args[..1])?;
            Ok(w_none())
        }),
    );
}

crate::py_module! {
    "_io",
    interpleveldefs: {
        "DEFAULT_BUFFER_SIZE" => w_int_new(8192),
    },
    // BytesIO / StringIO are the pure-Python in-memory streams: pickle's
    // Pickler/Unpickler use BytesIO; logging / traceback / csv use StringIO.
    appleveldefs: {
        "_io_app.py" => ["BytesIO", "StringIO", "IncrementalNewlineDecoder"],
    },
    functions: {
        "open"            / * = crate::builtins::builtin_open,
        // `io.open_code(path)` — `_PyIO_open_code` opens the path in binary
        // read mode ("rb"); pyre has no audit hooks so it is just `open`.
        "open_code"       / * = |args| {
            let path = args.first().copied().unwrap_or_else(w_none);
            crate::builtins::builtin_open(&[path, w_str_new("rb")])
        },
        "text_encoding"   / * = |args| Ok(args.first().copied().unwrap_or_else(|| w_str_new("utf-8"))),
    },
    extra_init: |ns| {
        // `Modules/_io/_iomodule.c`:
        //   UnsupportedOperation = class UnsupportedOperation(OSError, ValueError)
        // A real exception class so `raise`/`except` and io.py's
        // `UnsupportedOperation.__module__ = "io"` work.  Falls back to a
        // single OSError base if the builtin exceptions aren't registered.
        let os_error = crate::builtins::lookup_exc_class("OSError")
            .expect("OSError must be registered before _io init");
        let bases: &[pyre_object::PyObjectRef] =
            match crate::builtins::lookup_exc_class("ValueError") {
                Some(value_error) => &[os_error, value_error],
                None => &[os_error],
            };
        let unsupported = crate::builtins::make_exc_type_multi(
            "io.UnsupportedOperation",
            crate::builtins::exc_exception_new,
            bases,
        );
        let _ = UNSUPPORTED_OPERATION_TYPE.set(unsupported as usize);
        crate::module_ns_store(ns, "UnsupportedOperation", unsupported);

        // `_io.BlockingIOError` aliases the builtin BlockingIOError.
        if let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") {
            crate::module_ns_store(ns, "BlockingIOError", blocking);
        }

        // Abstract base classes as W_TypeObject (required for io.py class inheritance).
        // PyPy hierarchy: RawIOBase/BufferedIOBase/TextIOBase all derive IOBase.
        let obj_type = crate::typedef::w_object();
        let io_base = crate::typedef::make_builtin_type_with_base(
            "_IOBase",
            init_iobase_type,
            obj_type,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(io_base, true) };
        unsafe {
            pyre_object::w_type_set_weakrefable(io_base, true);
            pyre_object::typeobject::w_type_set_hasdict(io_base, true);
        }
        let raw_base = crate::typedef::make_builtin_type_with_base(
            "_RawIOBase",
            init_rawiobase_type,
            io_base,
        );
        let buffered_base = crate::typedef::make_builtin_type_with_base(
            "_BufferedIOBase",
            init_buffered_iobase_type,
            io_base,
        );
        let text_base = crate::typedef::make_builtin_type_with_base(
            "_TextIOBase",
            init_text_iobase_type,
            io_base,
        );
        for (name, typ) in [
            ("_IOBase", io_base),
            ("_RawIOBase", raw_base),
            ("_BufferedIOBase", buffered_base),
            ("_TextIOBase", text_base),
        ] {
            unsafe {
                pyre_object::w_type_set_acceptable_as_base_class(typ, true);
                pyre_object::w_type_set_weakrefable(typ, true);
                pyre_object::typeobject::w_type_set_hasdict(typ, true);
            };
            crate::module_ns_store(ns, name, typ);
        }

        // Concrete stream classes as subclassable W_TypeObjects.  stdlib
        // modules derive from them at import (`class ExFileObject(
        // io.BufferedReader)` in tarfile, `class _MockRawIO(...)` in
        // test_io), so they must be real types, not function stubs.
        // `FileIO` derives from `_RawIOBase`; the buffered classes from
        // `_BufferedIOBase` (`Modules/_io/_iomodule.c` PyInit__io).
        let file_io = crate::typedef::make_builtin_type_with_base(
            "FileIO",
            |type_ns| {
                crate::builtins::init_file_wrapper_type(type_ns);
                crate::builtins::init_fileio_type(type_ns);
                type_method(
                    type_ns,
                    "__init__",
                    crate::make_builtin_function("__init__", crate::builtins::fileio_init),
                );
            },
            raw_base,
        );
        // W_IOBase carries a weakref lifeline; W_FileIO instances therefore
        // accept weak references just like PyPy's concrete raw stream.
        unsafe { pyre_object::w_type_set_weakrefable(file_io, true) };
        let buffered_reader = crate::typedef::make_builtin_type_with_base(
            "BufferedReader",
            init_buffered_reader_type,
            buffered_base,
        );
        for (name, t) in [
            ("FileIO", file_io),
            ("BufferedReader", buffered_reader),
            (
                "BufferedWriter",
                crate::typedef::make_builtin_type_with_base(
                    "BufferedWriter",
                    |_| {},
                    buffered_base,
                ),
            ),
            (
                "BufferedRWPair",
                crate::typedef::make_builtin_type_with_base(
                    "BufferedRWPair",
                    |_| {},
                    buffered_base,
                ),
            ),
            (
                "BufferedRandom",
                crate::typedef::make_builtin_type_with_base(
                    "BufferedRandom",
                    |_| {},
                    buffered_base,
                ),
            ),
        ] {
            unsafe {
                pyre_object::w_type_set_acceptable_as_base_class(t, true);
                pyre_object::typeobject::w_type_set_hasdict(t, true);
            }
            crate::module_ns_store(ns, name, t);
        }

        // `TextIOWrapper` is a real (subclassable) type: stdlib modules such
        // as argparse / pickle / _android_support derive from it
        // (`class StdIOBuffer(io.TextIOWrapper)`).  Its `__init__` configures
        // the underlying buffer + encoding so `TextIOWrapper(buffer, ...)`
        // and a subclass's `super().__init__(...)` both work.
        let text_io_wrapper = crate::builtins::text_io_wrapper_type();
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(text_io_wrapper, true);
        }
        crate::module_ns_store(ns, "TextIOWrapper", text_io_wrapper);
    }
}
