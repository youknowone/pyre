//! _io module stub — C extension IO classes.
//!
//! PyPy equivalent: pypy/module/_io/

use crate::{PyNamespace, namespace_store, w_builtin_func_new};
use pyre_object::*;

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "DEFAULT_BUFFER_SIZE", w_int_new(8192));

    // StringIO stub — returns an empty object with write/getvalue
    namespace_store(
        ns,
        "StringIO",
        w_builtin_func_new("StringIO", stub_stringio),
    );
    namespace_store(ns, "BytesIO", w_builtin_func_new("BytesIO", stub_bytesio));
    namespace_store(ns, "FileIO", w_builtin_func_new("FileIO", stub_fileio));
    namespace_store(
        ns,
        "BufferedReader",
        w_builtin_func_new("BufferedReader", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedWriter",
        w_builtin_func_new("BufferedWriter", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedRWPair",
        w_builtin_func_new("BufferedRWPair", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedRandom",
        w_builtin_func_new("BufferedRandom", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "TextIOWrapper",
        w_builtin_func_new("TextIOWrapper", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "IncrementalNewlineDecoder",
        w_builtin_func_new("IncrementalNewlineDecoder", stub_noop_ctor),
    );
    namespace_store(ns, "open", w_builtin_func_new("open", stub_noop_ctor));

    // Exception types as strings (for isinstance checks in io.py)
    namespace_store(
        ns,
        "UnsupportedOperation",
        w_str_new("UnsupportedOperation"),
    );
    namespace_store(ns, "BlockingIOError", w_str_new("BlockingIOError"));

    // Abstract base classes (stubs)
    namespace_store(ns, "_IOBase", w_builtin_func_new("_IOBase", stub_noop_ctor));
    namespace_store(
        ns,
        "_RawIOBase",
        w_builtin_func_new("_RawIOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "_BufferedIOBase",
        w_builtin_func_new("_BufferedIOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "_TextIOBase",
        w_builtin_func_new("_TextIOBase", stub_noop_ctor),
    );
}

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
