//! _io module — PyPy: pypy/module/_io/
//!
//! Pyre stubs the bulk of the C IO classes: ctors return None / "" and
//! ABC base classes (`_IOBase` / `_RawIOBase` / `_BufferedIOBase` /
//! `_TextIOBase`) are exposed as plain types so io.py's class
//! inheritance succeeds.

use pyre_object::*;

fn stub_stringio(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_str_new(""))
}

fn stub_bytesio(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_str_new(""))
}

fn stub_fileio(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn stub_noop_ctor(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn text_encoding(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(if args.is_empty() {
        w_str_new("utf-8")
    } else {
        args[0]
    })
}

crate::py_module! {
    "_io",
    interpleveldefs: {
        "DEFAULT_BUFFER_SIZE" => w_int_new(8192),
        "StringIO"        => crate::make_builtin_function("StringIO",        stub_stringio),
        "BytesIO"         => crate::make_builtin_function("BytesIO",         stub_bytesio),
        "FileIO"          => crate::make_builtin_function("FileIO",          stub_fileio),
        "BufferedReader"  => crate::make_builtin_function("BufferedReader",  stub_noop_ctor),
        "BufferedWriter"  => crate::make_builtin_function("BufferedWriter",  stub_noop_ctor),
        "BufferedRWPair"  => crate::make_builtin_function("BufferedRWPair",  stub_noop_ctor),
        "BufferedRandom"  => crate::make_builtin_function("BufferedRandom",  stub_noop_ctor),
        "TextIOWrapper"   => crate::make_builtin_function("TextIOWrapper",   stub_noop_ctor),
        "IncrementalNewlineDecoder" =>
            crate::make_builtin_function("IncrementalNewlineDecoder", stub_noop_ctor),
        "open"            => crate::make_builtin_function("open",      stub_noop_ctor),
        "open_code"       => crate::make_builtin_function("open_code", stub_noop_ctor),
        "text_encoding"   => crate::make_builtin_function("text_encoding", text_encoding),
        // Exception types as strings (isinstance checks in io.py).
        "UnsupportedOperation" => w_str_new("UnsupportedOperation"),
        "BlockingIOError"      => w_str_new("BlockingIOError"),
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
