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

    namespace_store(
        ns,
        "open_code",
        w_builtin_func_new("open_code", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "text_encoding",
        w_builtin_func_new("text_encoding", |args| {
            Ok(if args.is_empty() {
                w_str_new("utf-8")
            } else {
                args[0]
            })
        }),
    );

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

/// `io` module — re-exports _io names + IOBase/RawIOBase etc. as stubs.
/// Real io.py uses ABCMeta metaclass which requires metaclass support.
pub fn init_io(ns: &mut PyNamespace) {
    init(ns);
    // Additional names io.py would define
    namespace_store(ns, "IOBase", w_builtin_func_new("IOBase", stub_noop_ctor));
    namespace_store(
        ns,
        "RawIOBase",
        w_builtin_func_new("RawIOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedIOBase",
        w_builtin_func_new("BufferedIOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "TextIOBase",
        w_builtin_func_new("TextIOBase", stub_noop_ctor),
    );
    namespace_store(ns, "SEEK_SET", w_int_new(0));
    namespace_store(ns, "SEEK_CUR", w_int_new(1));
    namespace_store(ns, "SEEK_END", w_int_new(2));
    namespace_store(ns, "Reader", w_builtin_func_new("Reader", stub_noop_ctor));
    namespace_store(ns, "Writer", w_builtin_func_new("Writer", stub_noop_ctor));
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
