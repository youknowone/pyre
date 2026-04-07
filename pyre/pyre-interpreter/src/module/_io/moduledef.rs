//! _io module stub — C extension IO classes.
//!
//! PyPy equivalent: pypy/module/_io/

use crate::{PyNamespace, make_builtin_function, namespace_store};
use pyre_object::*;

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "DEFAULT_BUFFER_SIZE", w_int_new(8192));

    // StringIO stub — returns an empty object with write/getvalue
    namespace_store(
        ns,
        "StringIO",
        make_builtin_function("StringIO", stub_stringio),
    );
    namespace_store(
        ns,
        "BytesIO",
        make_builtin_function("BytesIO", stub_bytesio),
    );
    namespace_store(ns, "FileIO", make_builtin_function("FileIO", stub_fileio));
    namespace_store(
        ns,
        "BufferedReader",
        make_builtin_function("BufferedReader", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedWriter",
        make_builtin_function("BufferedWriter", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedRWPair",
        make_builtin_function("BufferedRWPair", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedRandom",
        make_builtin_function("BufferedRandom", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "TextIOWrapper",
        make_builtin_function("TextIOWrapper", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "IncrementalNewlineDecoder",
        make_builtin_function("IncrementalNewlineDecoder", stub_noop_ctor),
    );
    namespace_store(ns, "open", make_builtin_function("open", stub_noop_ctor));

    namespace_store(
        ns,
        "open_code",
        make_builtin_function("open_code", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "text_encoding",
        make_builtin_function("text_encoding", |args| {
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
    namespace_store(
        ns,
        "IOBase",
        make_builtin_function("IOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "RawIOBase",
        make_builtin_function("RawIOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "BufferedIOBase",
        make_builtin_function("BufferedIOBase", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "TextIOBase",
        make_builtin_function("TextIOBase", stub_noop_ctor),
    );
    namespace_store(ns, "SEEK_SET", w_int_new(0));
    namespace_store(ns, "SEEK_CUR", w_int_new(1));
    namespace_store(ns, "SEEK_END", w_int_new(2));
    namespace_store(
        ns,
        "Reader",
        make_builtin_function("Reader", stub_noop_ctor),
    );
    namespace_store(
        ns,
        "Writer",
        make_builtin_function("Writer", stub_noop_ctor),
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
