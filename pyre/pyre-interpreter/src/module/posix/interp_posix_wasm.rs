//! `posix` on wasm32 — PyPy: pypy/module/posix/interp_posix.py
//!
//! wasm32 has no operating system underneath it, so nothing here wraps a
//! syscall.  What the guest does have is the embedder's filesystem, reached
//! through the import machinery's [`SourceProvider`]: the same seam `import`
//! resolves modules with and traceback rendering reads source lines with.
//! This module publishes that seam under the names `os.py` reads.
//!
//! The list is exactly what `os.py`'s module body needs to run — `environ`,
//! `_have_functions`, `stat` and `cpu_count` — plus `lstat` and
//! `stat_result`.  Everything else `os` normally re-exports stays absent, so
//! `os._exists` answers False for it and `os` builds the same fallbacks it
//! builds on a platform whose C library lacks the call.
//!
//! [`SourceProvider`]: crate::importing::SourceProvider

use pyre_object::{PyObjectRef, is_none};

/// `S_IFREG` / `S_IFDIR` — the two file types the seam can tell apart.  It
/// answers `is_dir` and a byte length and nothing else, so a symlink, a device
/// or a socket is not distinguishable from the file it resolves to.
const S_IFDIR: i64 = 0o040000;
const S_IFREG: i64 = 0o100000;

/// `posix.stat(path)` — build a `stat_result` from the two facts the seam
/// reports.
///
/// `st_mode` carries the file type and a read-only permission set, and
/// `st_size` the byte length; every other field is zero, including all three
/// timestamps, because no timestamp reaches the guest.  A constant mtime is
/// self-consistent rather than merely absent: `linecache.checkcache` compares
/// a cached entry against a fresh `stat` of the same file, so equal values
/// keep the entry valid instead of invalidating it on every check.
fn stat(w_path: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let bytes = crate::gateway::fsencode_bytes_w(w_path)?;
    // The seam takes a `Path`, which on wasm32 is UTF-8; a path byte with no
    // UTF-8 spelling cannot address a host file through it either way.
    let text = String::from_utf8_lossy(&bytes);
    let path = std::path::Path::new(text.as_ref());
    let (mode, size) = if crate::importing::source_is_dir(path) {
        (S_IFDIR | 0o555, 0)
    } else {
        match crate::importing::source_file_size(path) {
            Ok(size) => (S_IFREG | 0o444, size as i64),
            Err(_) => {
                return Err(crate::PyError::os_error_syscall(
                    crate::builtins::wasm_errno::ENOENT,
                    pyre_object::w_str_new(&text),
                ));
            }
        }
    };
    Ok(make_stat_result(mode, size))
}

/// The `stat_result` layout of [`super::stat_result_seq_type`] with the
/// platform extras that target carries — on wasm32 neither the unix
/// block-device fields nor the Windows ones, leaving the float times, the
/// sub-second remainders and the full nanosecond timestamps.
fn make_stat_result(mode: i64, size: i64) -> PyObjectRef {
    let seq = vec![
        pyre_object::w_int_new(mode),
        pyre_object::w_int_new(0), // st_ino
        pyre_object::w_int_new(0), // st_dev
        pyre_object::w_int_new(1), // st_nlink
        pyre_object::w_int_new(0), // st_uid
        pyre_object::w_int_new(0), // st_gid
        pyre_object::w_int_new(size),
        pyre_object::w_int_new(0), // _integer_atime
        pyre_object::w_int_new(0), // _integer_mtime
        pyre_object::w_int_new(0), // _integer_ctime
    ];
    let extras = vec![
        ("st_atime", pyre_object::w_float_new(0.0)),
        ("st_mtime", pyre_object::w_float_new(0.0)),
        ("st_ctime", pyre_object::w_float_new(0.0)),
        ("nsec_atime", pyre_object::w_int_new(0)),
        ("nsec_mtime", pyre_object::w_int_new(0)),
        ("nsec_ctime", pyre_object::w_int_new(0)),
        ("st_atime_ns", pyre_object::w_int_new(0)),
        ("st_mtime_ns", pyre_object::w_int_new(0)),
        ("st_ctime_ns", pyre_object::w_int_new(0)),
    ];
    crate::_structseq::new_instance_with_extra(super::stat_result_seq_type(), seq, extras)
}

/// `posix.listdir(path)` — the entry names the seam reports, in its order.
///
/// `os.listdir` defaults its argument to the current directory, which this
/// target does not have; a call with no path is refused rather than answered
/// about some invented one.
fn listdir(w_path: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if w_path.is_null() || unsafe { is_none(w_path) } {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::ENOENT,
            pyre_object::PY_NULL,
        ));
    }
    let bytes = crate::gateway::fsencode_bytes_w(w_path)?;
    let text = String::from_utf8_lossy(&bytes);
    let path = std::path::Path::new(text.as_ref());
    let entries = crate::importing::source_list_dir(path).map_err(|_| {
        crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::ENOENT,
            pyre_object::w_str_new(&text),
        )
    })?;
    let _roots = pyre_object::gc_roots::push_roots();
    let names: Vec<PyObjectRef> = entries
        .iter()
        .map(|name| pyre_object::w_str_new(&name.to_string_lossy()))
        .collect();
    let first = pyre_object::gc_roots::pin_roots(&names);
    let rooted = (0..names.len())
        .map(|index| pyre_object::gc_roots::shadow_stack_get(first + index))
        .collect();
    Ok(pyre_object::w_list_new(rooted))
}

pub fn register_module(ns: PyObjectRef) {
    // `os._createenviron` reads this dict directly and wraps it, so `os.environ`
    // is a live view of it.  The guest is started with no environment — the
    // runner hands the interpreter the few variables it forwards, rather than
    // an environment block — so it starts empty and stays whatever the program
    // puts in it.
    crate::module_ns_store(ns, "environ", pyre_object::w_dict_new());
    // `os.py` builds `supports_dir_fd` and friends only when this exists, and
    // then reads `stat` out of its own namespace to seed `supports_fd`.  Empty:
    // no call here takes a directory or a descriptor argument.
    crate::module_ns_store(ns, "_have_functions", pyre_object::w_list_new(vec![]));
    crate::module_ns_store(ns, "stat_result", super::stat_result_seq_type());
    crate::module_ns_store(
        ns,
        "stat",
        crate::make_builtin_function_with_arity(
            "stat",
            |args| stat(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );
    // The seam resolves a path without reporting whether it went through a
    // symlink, so `lstat` can only answer what `stat` answers.
    crate::module_ns_store(
        ns,
        "lstat",
        crate::make_builtin_function_with_arity(
            "lstat",
            |args| stat(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "listdir",
        crate::make_builtin_function_with_arity(
            "listdir",
            |args| listdir(args.first().copied().unwrap_or(pyre_object::PY_NULL)),
            1,
        ),
    );
    // `_bootstrap_external` reads `_os.fspath`, not `os.fspath`, so the
    // pure-Python fallback `os.py` installs when `posix` lacks it does not
    // stand in for the import machinery.  The protocol itself touches no
    // filesystem, so both arms publish the one implementation.
    crate::module_ns_store(
        ns,
        "fspath",
        crate::make_builtin_function_with_arity(
            "fspath",
            |args| super::fspath(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );
    // `os.process_cpu_count` is an alias for this one, and reads it at module
    // level, so its absence would stop `import os` outright.  One: the guest is
    // a single wasm instance.
    crate::module_ns_store(
        ns,
        "cpu_count",
        crate::make_builtin_function_with_arity(
            "cpu_count",
            |_args| Ok(pyre_object::w_int_new(1)),
            0,
        ),
    );
}
