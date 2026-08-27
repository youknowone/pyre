//! `posix` on wasm32 — PyPy: pypy/module/posix/interp_posix.py
//!
//! wasm32 has no operating system underneath it, so nothing here wraps a
//! syscall.  What the guest does have is the embedder's filesystem, reached
//! through the import machinery's [`SourceProvider`]: the same seam `import`
//! resolves modules with and traceback rendering reads source lines with.
//! This module publishes that seam under the names `os.py` reads.
//!
//! The list is exactly what `os.py`'s module body needs to run — `environ`,
//! `_have_functions`, `stat` and `cpu_count` — plus `lstat`, `listdir`,
//! `fspath` and `stat_result`.  Everything else `os` normally re-exports
//! stays absent, so `os._exists` answers False for it and `os` builds the
//! same fallbacks it builds on a platform whose C library lacks the call.
//!
//! [`SourceProvider`]: crate::importing::SourceProvider

use pyre_object::{PyObjectRef, is_none};

/// `S_IFREG` / `S_IFDIR` — the two file types the seam can tell apart.  It
/// answers `is_dir` and a byte length and nothing else, so a symlink, a device
/// or a socket is not distinguishable from the file it resolves to.
const S_IFDIR: i64 = 0o040000;
const S_IFREG: i64 = 0o100000;

/// `posix.stat(path, *, dir_fd=None, follow_symlinks=True)` /
/// `posix.lstat(path, *, dir_fd=None)` — build a `stat_result` from the two
/// facts the seam reports.
///
/// `st_mode` carries the file type and a read-only permission set, and
/// `st_size` the byte length; every other field is zero, including all three
/// timestamps, because no timestamp reaches the guest.  A constant mtime is
/// self-consistent rather than merely absent: `linecache.checkcache` compares
/// a cached entry against a fresh `stat` of the same file, so equal values
/// keep the entry valid instead of invalidating it on every check.
///
/// The seam takes a name and nothing else, so it is the platform without
/// `fstatat` and without `fstat`: `dir_fd` is refused the way
/// `_DirFD_Unavailable` refuses it, and a descriptor is not one of the types
/// the argument accepts.  `follow_symlinks` is accepted and has no effect —
/// the seam resolves a path without reporting whether it went through a
/// symlink, which is the same reason `lstat` can only answer what `stat`
/// answers.
fn stat(args: &[PyObjectRef], default_follow: bool) -> Result<PyObjectRef, crate::PyError> {
    let name = if default_follow { "stat" } else { "lstat" };
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let allowed: &[&str] = if default_follow {
        &["path", "dir_fd", "follow_symlinks"]
    } else {
        &["path", "dir_fd"]
    };
    crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes at most 1 positional argument ({} given)",
            pos.len()
        )));
    }
    let Some(w_path) = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", name, 1)? else {
        return Err(crate::PyError::type_error(format!(
            "{name}() missing required argument 'path' (pos 1)"
        )));
    };
    // Unwrapped in signature order, because `__fspath__` for `path` and
    // `__index__` for `dir_fd` can both raise and both run user code.
    let resolved = crate::gateway::fsencode_path_or_fd_w(w_path, name, false)?;
    if crate::builtins::kwarg_get(kwargs, "dir_fd").is_some_and(|v| !unsafe { is_none(v) }) {
        return Err(crate::PyError::not_implemented(
            "dir_fd unavailable on this platform",
        ));
    }
    if let Some(v) = crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
        crate::baseobjspace::is_true(v)?;
    }
    let path = seam_path(&resolved.as_bytes);
    let (mode, size) = if crate::importing::source_is_dir(&path) {
        (S_IFDIR | 0o555, 0)
    } else {
        match crate::importing::source_file_size(&path) {
            Ok(size) => (S_IFREG | 0o444, size as i64),
            // The seam reports why it could not answer, and the path object the
            // caller passed is what names the failure.
            Err(e) => return Err(seam_error(&e, resolved.w_path())),
        }
    };
    Ok(make_stat_result(mode, size))
}

/// The seam takes a `&Path`, and on this target an `OsStr` is the byte string
/// itself, so the filesystem bytes cross it whole rather than through a text
/// spelling that a name without one would not survive: an entry `listdir`
/// reported is an entry `stat` can be called with again.
fn seam_path(bytes: &[u8]) -> std::path::PathBuf {
    crate::gateway::os_string_from_fs_bytes(bytes).into()
}

/// The seam's `io::Error` as the OSError it stands for, named by the path
/// object the caller passed rather than by a re-decoded spelling of it.
///
/// A provider that answers `import`'s probes need not be able to enumerate a
/// directory, and refusing is not the same answer as a missing one.
fn seam_error(e: &std::io::Error, w_path: PyObjectRef) -> crate::PyError {
    let errno = match e.kind() {
        std::io::ErrorKind::Unsupported => crate::builtins::wasm_errno::ENOTSUP,
        _ => crate::builtins::wasm_errno::ENOENT,
    };
    crate::PyError::os_error_syscall(crate::builtins::io_error_posix_errno(e, errno), w_path)
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

/// `posix.listdir(path=None)` — the entry names the seam reports, in its
/// order.
///
/// The omitted argument is the `None` the signature names, which resolves to
/// `"."`; the guest has no `getcwd` to spell that any other way, so what it
/// stands for is whatever directory the embedder started in.  A `bytes` path
/// asks for `bytes` names, as it does on every other target.
fn listdir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path"], "listdir")?;
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "listdir() takes at most 1 argument ({} given)",
            pos.len()
        )));
    }
    let w_path = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", "listdir", 1)?
        .unwrap_or(pyre_object::w_none());
    // One resolution yields both the path and its bytes-ness, so `__fspath__`
    // runs exactly once.
    let resolved = crate::gateway::fsencode_path_or_fd_nullable_w(w_path, "listdir", false)?;
    let bytes_mode = unsafe { resolved.is_bytes() };
    let path = seam_path(&resolved.as_bytes);
    let entries =
        crate::importing::source_list_dir(&path).map_err(|e| seam_error(&e, resolved.w_path()))?;
    // Each name is freshly allocated and the next one allocates again, so they
    // are pinned as they arrive.
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for name in &entries {
        items.push(super::fs_name_obj(bytes_mode, name));
    }
    Ok(pyre_object::w_list_new(items.take()))
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
    // `follow_symlinks` and `dir_fd` are keyword-only, so neither entry point
    // can take the fixed-arity carrier that rejects keywords.
    crate::module_ns_store(
        ns,
        "stat",
        crate::make_builtin_function("stat", |args| stat(args, true)),
    );
    crate::module_ns_store(
        ns,
        "lstat",
        crate::make_builtin_function("lstat", |args| stat(args, false)),
    );
    crate::module_ns_store(
        ns,
        "listdir",
        crate::make_builtin_function("listdir", listdir),
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
