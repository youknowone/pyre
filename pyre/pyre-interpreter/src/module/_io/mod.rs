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
        // Exception types as strings (isinstance checks in io.py).
        "UnsupportedOperation" => w_str_new("UnsupportedOperation"),
        "BlockingIOError"      => w_str_new("BlockingIOError"),
    },
    functions: {
        "StringIO"        / * = |_| Ok(w_str_new("")),
        "BytesIO"         / * = |_| Ok(w_str_new("")),
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
