//! _io module — PyPy: pypy/module/_io/
//!
//! Pyre stubs the bulk of the C IO classes: ctors return None / "" and
//! ABC base classes (`_IOBase` / `_RawIOBase` / `_BufferedIOBase` /
//! `_TextIOBase`) are exposed as plain types so io.py's class
//! inheritance succeeds.

use pyre_object::*;

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
        crate::dict_storage_store(ns, "UnsupportedOperation", unsupported);

        // `_io.BlockingIOError` aliases the builtin BlockingIOError.
        if let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") {
            crate::dict_storage_store(ns, "BlockingIOError", blocking);
        }

        // Abstract base classes as W_TypeObject (required for io.py class inheritance).
        let obj_type = crate::typedef::w_object();
        let mut io_base_types: std::collections::HashMap<&str, pyre_object::PyObjectRef> =
            std::collections::HashMap::new();
        for name in &["_IOBase", "_RawIOBase", "_BufferedIOBase", "_TextIOBase"] {
            let t = pyre_object::w_type_new(
                name,
                pyre_object::w_tuple_new(vec![obj_type]),
                std::ptr::null_mut(),
            );
            unsafe { pyre_object::w_type_set_mro(t, vec![t, obj_type]) };
            unsafe { pyre_object::typeobject::w_type_ready(t) };
            io_base_types.insert(name, t);
            crate::dict_storage_store(ns, name, t);
        }

        // Concrete stream classes as subclassable W_TypeObjects.  stdlib
        // modules derive from them at import (`class ExFileObject(
        // io.BufferedReader)` in tarfile, `class _MockRawIO(...)` in
        // test_io), so they must be real types, not function stubs.
        // `FileIO` derives from `_RawIOBase`; the buffered classes from
        // `_BufferedIOBase` (`Modules/_io/_iomodule.c` PyInit__io).
        for (name, base_name) in &[
            ("FileIO", "_RawIOBase"),
            ("BufferedReader", "_BufferedIOBase"),
            ("BufferedWriter", "_BufferedIOBase"),
            ("BufferedRWPair", "_BufferedIOBase"),
            ("BufferedRandom", "_BufferedIOBase"),
        ] {
            let base = io_base_types[base_name];
            let t = crate::typedef::make_builtin_type_with_base(name, |_ns| {}, base);
            unsafe {
                pyre_object::w_type_set_acceptable_as_base_class(t, true);
                pyre_object::typeobject::w_type_set_hasdict(t, true);
            }
            crate::dict_storage_store(ns, name, t);
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
        crate::dict_storage_store(ns, "TextIOWrapper", text_io_wrapper);
    }
}
