//! _io module stub — C extension IO classes.
//!
//! PyPy equivalent: pypy/module/_io/

use crate::{PyNamespace, builtin_code_new, namespace_store};
use pyre_object::*;

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "DEFAULT_BUFFER_SIZE", w_int_new(8192));

    // StringIO stub — returns an empty object with write/getvalue
    namespace_store(ns, "StringIO", builtin_code_new("StringIO", stub_stringio));
    namespace_store(ns, "BytesIO", builtin_code_new("BytesIO", stub_bytesio));
    namespace_store(ns, "FileIO", builtin_code_new("FileIO", stub_fileio));
    namespace_store(
        ns,
        "BufferedReader",
        builtin_code_new("BufferedReader", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedWriter",
        builtin_code_new("BufferedWriter", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedRWPair",
        builtin_code_new("BufferedRWPair", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedRandom",
        builtin_code_new("BufferedRandom", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "TextIOWrapper",
        builtin_code_new("TextIOWrapper", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "IncrementalNewlineDecoder",
        builtin_code_new("IncrementalNewlineDecoder", stub_noop_ctor),
    );
    namespace_store(ns, "open", builtin_code_new("open", stub_noop_ctor));

    namespace_store(
        ns,
        "open_code",
        builtin_code_new("open_code", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "text_encoding",
        builtin_code_new("text_encoding", |args| {
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

    // Abstract base classes as W_TypeObject (required for io.py class inheritance)
    let obj_type = crate::typedef::w_object();
    for name in &["_IOBase", "_RawIOBase", "_BufferedIOBase", "_TextIOBase"] {
        let t = pyre_object::w_type_new(
            name,
            pyre_object::w_tuple_new(vec![obj_type]),
            std::ptr::null_mut(),
        );
        unsafe { pyre_object::w_type_set_mro(t, vec![t, obj_type]) };
        namespace_store(ns, name, t);
    }
}

/// `io` module — re-exports _io names + IOBase/RawIOBase etc. as stubs.
/// Real io.py uses ABCMeta metaclass which requires metaclass support.
pub fn init_io(ns: &mut PyNamespace) {
    init(ns);
    // Additional names io.py would define
    namespace_store(ns, "IOBase", builtin_code_new("IOBase", stub_noop_ctor));
    namespace_store(
        ns,
        "RawIOBase",
        builtin_code_new("RawIOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedIOBase",
        builtin_code_new("BufferedIOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "TextIOBase",
        builtin_code_new("TextIOBase", stub_noop_ctor),
    );
    namespace_store(ns, "SEEK_SET", w_int_new(0));
    namespace_store(ns, "SEEK_CUR", w_int_new(1));
    namespace_store(ns, "SEEK_END", w_int_new(2));
    namespace_store(ns, "Reader", builtin_code_new("Reader", stub_noop_ctor));
    namespace_store(ns, "Writer", builtin_code_new("Writer", stub_noop_ctor));
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
