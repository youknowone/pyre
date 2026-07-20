//! _io module — PyPy: pypy/module/_io/
//!
//! Pyre stubs the bulk of the C IO classes: ctors return None / "" and
//! ABC base classes (`_IOBase` / `_RawIOBase` / `_BufferedIOBase` /
//! `_TextIOBase`) are exposed as plain types so io.py's class
//! inheritance succeeds.

use pyre_object::*;

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

fn iobase_close(args: &[PyObjectRef]) -> crate::PyResult {
    if let Some(&self_obj) = args.first() {
        crate::baseobjspace::setattr_str(self_obj, "closed", w_bool_from(true))?;
    }
    Ok(w_none())
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

fn init_iobase_type(ns: PyObjectRef) {
    type_method(ns, "closed", w_bool_from(false));
    type_method(
        ns,
        "close",
        crate::make_builtin_function_with_arity("close", iobase_close, 1),
    );
    type_method(
        ns,
        "isatty",
        crate::make_builtin_function_with_arity("isatty", iobase_isatty, 1),
    );
    type_method(
        ns,
        "writelines",
        crate::make_builtin_function_with_arity("writelines", iobase_writelines, 2),
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
            iobase_close(&args[..1])?;
            Ok(w_none())
        }),
    );
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
        "_io_app.py" => ["BytesIO", "StringIO"],
    },
    functions: {
        "IncrementalNewlineDecoder" / * = |_| Ok(w_none()),
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
        let raw_base = crate::typedef::make_builtin_type_with_base("_RawIOBase", |_| {}, io_base);
        let buffered_base = crate::typedef::make_builtin_type_with_base(
            "_BufferedIOBase",
            |_| {},
            io_base,
        );
        let text_base = crate::typedef::make_builtin_type_with_base("_TextIOBase", |_| {}, io_base);
        for (name, typ) in [
            ("_IOBase", io_base),
            ("_RawIOBase", raw_base),
            ("_BufferedIOBase", buffered_base),
            ("_TextIOBase", text_base),
        ] {
            unsafe { pyre_object::w_type_set_acceptable_as_base_class(typ, true) };
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
                type_method(
                    type_ns,
                    "__init__",
                    crate::make_builtin_function("__init__", crate::builtins::fileio_init),
                );
            },
            raw_base,
        );
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
