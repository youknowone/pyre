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
    // BytesIO is the pure-Python in-memory binary stream pickle's
    // Pickler/Unpickler write to and read from.
    appleveldefs: {
        "_io_app.py" => ["BytesIO"],
    },
    functions: {
        "StringIO"        / * = |_| Ok(w_str_new("")),
        "FileIO"          / * = |_| Ok(w_none()),
        "BufferedReader"  / * = |_| Ok(w_none()),
        "BufferedWriter"  / * = |_| Ok(w_none()),
        "BufferedRWPair"  / * = |_| Ok(w_none()),
        "BufferedRandom"  / * = |_| Ok(w_none()),
        "TextIOWrapper"   / * = |_| Ok(w_none()),
        "IncrementalNewlineDecoder" / * = |_| Ok(w_none()),
        "open"            / * = |_| Ok(w_none()),
        "open_code"       / * = |_| Ok(w_none()),
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
        for name in &["_IOBase", "_RawIOBase", "_BufferedIOBase", "_TextIOBase"] {
            let t = pyre_object::w_type_new(
                name,
                pyre_object::w_tuple_new(vec![obj_type]),
                std::ptr::null_mut(),
            );
            unsafe { pyre_object::w_type_set_mro(t, vec![t, obj_type]) };
            unsafe { pyre_object::typeobject::w_type_ready(t) };
            crate::dict_storage_store(ns, name, t);
        }
    }
}
