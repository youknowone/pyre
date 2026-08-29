//! `posix` on wasm32 — PyPy: pypy/module/posix/interp_posix.py
//!
//! wasm32 has no operating system underneath it, so nothing here wraps a
//! syscall.  What the guest does have is the embedder's filesystem, reached
//! through the import machinery's [`SourceProvider`]: the same seam `import`
//! resolves modules with and traceback rendering reads source lines with.
//! This module publishes that seam under the names `os.py` reads.
//!
//! The list is what `os.py`'s module body needs to run — `environ`,
//! `_have_functions`, `stat` and `cpu_count` — plus `lstat`, `listdir`,
//! `fspath`, `scandir` and `stat_result`, and the three calls the stdlib reaches for
//! that the embedder can still answer: `getcwd`, `getcwdb` and `urandom`.
//! On top of that sits a descriptor half — `open`, `close`, `read`, `lseek`,
//! `fstat` — over a table this module keeps itself.  The seam answers a whole
//! path at a time, so a descriptor is the bytes a path resolved to plus a
//! position; the numbers are the guest's own and name nothing on the other
//! side of the host ABI.  The seam is a read-only mount, so every entry point
//! that would change it (`open` for write, `unlink`, `rmdir`) is published and
//! answers `EROFS`, which is what a read-only filesystem answers — a different
//! statement from the call not existing.
//!
//! Everything else `os` normally re-exports stays absent, so `os._exists`
//! answers False for it and `os` builds the same fallbacks it builds on a
//! platform whose C library lacks the call.
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
    stat_resolved(&resolved)
}

/// The guest's working directory, once something has changed it.
///
/// The seam resolves a relative name against the embedder's directory, which is
/// what `getcwd` reports until `chdir` runs.  After that the guest owns the
/// answer, and a relative path is joined here before it crosses -- the same
/// place `getcwd` reads, so the two never disagree.
static WORKING_DIRECTORY: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);

/// The directory a relative path is resolved against, as bytes.
fn cwd_bytes() -> Vec<u8> {
    if let Some(directory) = WORKING_DIRECTORY.lock().unwrap().clone() {
        return directory;
    }
    crate::importing::source_cwd().unwrap_or_default()
}

/// `path` as the seam should see it: an absolute name unchanged, a relative one
/// joined to the working directory.
///
/// An empty name stays empty so that the seam reports `ENOENT` for it, which is
/// what `open("")` owes its caller rather than the working directory's contents.
fn resolve_against_cwd(path: &[u8]) -> Vec<u8> {
    if path.is_empty() || path.starts_with(b"/") {
        return path.to_vec();
    }
    let mut resolved = cwd_bytes();
    if resolved.is_empty() {
        return path.to_vec();
    }
    if !resolved.ends_with(b"/") {
        resolved.push(b'/');
    }
    resolved.extend_from_slice(path);
    resolved
}

/// `posix.chdir(path)` — the directory later relative paths resolve against.
///
/// `fchdir` is what a descriptor argument would need, and the seam has no
/// directory descriptor to offer it, so a name is the only spelling accepted.
fn chdir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let resolved = path_arg(args, "chdir")?;
    let target = resolve_against_cwd(&resolved.as_bytes);
    let path: std::path::PathBuf = crate::gateway::os_string_from_fs_bytes(&target).into();
    if !crate::importing::source_is_dir(&path) {
        // The two answers a failed `chdir` gives: the name is not there, or it
        // is there and is not a directory.
        let errno = match crate::importing::source_file_size(&path) {
            Ok(_) => crate::builtins::wasm_errno::ENOTDIR,
            Err(e) => crate::builtins::wasm_errno::seam_errno(&e),
        };
        return Err(crate::PyError::os_error_syscall(errno, resolved.w_path()));
    }
    *WORKING_DIRECTORY.lock().unwrap() = Some(target);
    Ok(pyre_object::w_none())
}

/// The seam takes a `&Path`, and on this target an `OsStr` is the byte string
/// itself, so the filesystem bytes cross it whole rather than through a text
/// spelling that a name without one would not survive: an entry `listdir`
/// reported is an entry `stat` can be called with again.
///
/// This is the one place a relative name becomes an absolute one, so every
/// entry point that takes a path resolves it against the same directory
/// `getcwd` reports.
pub(crate) fn seam_path(bytes: &[u8]) -> std::path::PathBuf {
    crate::gateway::os_string_from_fs_bytes(&resolve_against_cwd(bytes)).into()
}

/// The seam's `io::Error` as the OSError it stands for, named by the path
/// object the caller passed rather than by a re-decoded spelling of it.
fn seam_error(e: &std::io::Error, w_path: PyObjectRef) -> crate::PyError {
    let errno = crate::builtins::wasm_errno::seam_errno(e);
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

/// `os.terminal_size` structseq — `(columns, lines)`.
fn terminal_size_seq_type() -> PyObjectRef {
    static TERMINAL_SIZE_SEQ_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TERMINAL_SIZE_SEQ_TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq("os.terminal_size", &["columns", "lines"]) as usize
    }) as PyObjectRef
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

/// One `stat_result` for a path the caller has already resolved, so that
/// `stat`, `lstat` and `DirEntry.stat` report one file the same way.
fn stat_resolved(
    resolved: &crate::gateway::FsEncodedPath,
) -> Result<PyObjectRef, crate::PyError> {
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

/// `join_path_filename` — the directory and the entry name with one separator
/// between them.  An empty directory leaves the name alone, which is what
/// `scandir("")` would otherwise turn into an absolute path.
fn join_entry_path(dir_bytes: &[u8], name: &[u8]) -> Vec<u8> {
    if dir_bytes.is_empty() {
        return name.to_vec();
    }
    let mut joined = dir_bytes.to_vec();
    if !joined.ends_with(b"/") {
        joined.push(b'/');
    }
    joined.extend_from_slice(name);
    joined
}

/// The path a `DirEntry` was built for, back in the form the seam takes.
///
/// It is read off the entry rather than kept beside it: `path` is the entry's
/// own public attribute, so one spelling answers both the caller and the seam.
fn entry_path(self_obj: PyObjectRef) -> Result<crate::gateway::FsEncodedPath, crate::PyError> {
    crate::gateway::fsencode_path_w(crate::baseobjspace::getattr_str(self_obj, "path")?)
}

/// The receiver of a `DirEntry` method, with `follow_symlinks` accepted and
/// without effect for the same reason `lstat` answers what `stat` answers.
fn entry_self(
    args: &[PyObjectRef],
    name: &'static str,
    follow: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let allowed: &[&str] = if follow { &["follow_symlinks"] } else { &[] };
    crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
    if let Some(v) = crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
        crate::baseobjspace::is_true(v)?;
    }
    pos.first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error(format!("{name}() requires self")))
}

/// `posix.DirEntry` — the record `scandir` yields for one name.
///
/// The seam reports a name, whether it is a directory, and a byte length, so
/// that is what an entry can answer.  It keeps its name and its path and asks
/// the seam again for the rest, which is the same on-demand `stat` the entry
/// performs anywhere its `readdir` did not carry a type.  `is_symlink` is
/// false for every entry, not as a claim that nothing is a link but because a
/// name and what it resolves to are the only two things this seam separates.
fn dir_entry_type() -> PyObjectRef {
    static DIR_ENTRY_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DIR_ENTRY_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("posix.DirEntry", init_dir_entry_type);
        crate::typedef::mark_cpython_heap_type(tp, false);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

fn init_dir_entry_type(ns: PyObjectRef) {
    let method = |name: &'static str, f: crate::gateway::BuiltinCodeFn| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            name,
            crate::make_builtin_function(name, f),
        );
    };
    method("is_dir", |args| {
        let resolved = entry_path(entry_self(args, "is_dir", true)?)?;
        Ok(pyre_object::w_bool_from(crate::importing::source_is_dir(
            &seam_path(&resolved.as_bytes),
        )))
    });
    method("is_file", |args| {
        let resolved = entry_path(entry_self(args, "is_file", true)?)?;
        let path = seam_path(&resolved.as_bytes);
        Ok(pyre_object::w_bool_from(
            !crate::importing::source_is_dir(&path)
                && crate::importing::source_file_size(&path).is_ok(),
        ))
    });
    method("is_symlink", |args| {
        entry_self(args, "is_symlink", false)?;
        Ok(pyre_object::w_bool_from(false))
    });
    method("is_junction", |args| {
        entry_self(args, "is_junction", false)?;
        Ok(pyre_object::w_bool_from(false))
    });
    method("stat", |args| {
        stat_resolved(&entry_path(entry_self(args, "stat", true)?)?)
    });
    // No inode number reaches the guest, so every entry reports the zero
    // `st_ino` its `stat_result` carries: one file has one identity either way.
    method("inode", |args| {
        entry_self(args, "inode", false)?;
        Ok(pyre_object::w_int_new(0))
    });
    method("__fspath__", |args| {
        crate::baseobjspace::getattr_str(entry_self(args, "__fspath__", false)?, "path")
    });
    method("__repr__", |args| {
        let name = crate::baseobjspace::getattr_str(entry_self(args, "__repr__", false)?, "name")?;
        Ok(pyre_object::w_str_from_wtf8_managed(
            crate::display::wtf8_format!("<DirEntry ", unsafe {
                crate::display::py_repr_wtf8(name)?
            }, ">"),
        ))
    });
}

/// The entry for one name inside the directory `scandir` was called on.
fn new_dir_entry(bytes_mode: bool, dir_bytes: &[u8], name: &[u8]) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let entry_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_instance_new(dir_entry_type()));
    // Each value is built before the receiver is read back, because building
    // it allocates and the allocation may move the entry.
    let store = |field: &str, value: PyObjectRef| {
        crate::baseobjspace::setdictvalue_native(
            pyre_object::gc_roots::shadow_stack_get(entry_slot),
            field,
            value,
        );
    };
    store("name", super::fs_name_obj(bytes_mode, name));
    store(
        "path",
        super::fs_name_obj(bytes_mode, &join_entry_path(dir_bytes, name)),
    );
    pyre_object::gc_roots::shadow_stack_get(entry_slot)
}

/// `posix.ScandirIterator` — the iterator `scandir` returns, and the context
/// manager `os.walk` enters.
///
/// The seam answers a whole directory at a time, so the entries exist before
/// the first `__next__`; what the iterator holds is the position in them.
/// `close` drops that, which is what makes an exhausted or closed iterator
/// report the end rather than the directory again.
fn scandir_iterator_type() -> PyObjectRef {
    static SCANDIR_ITERATOR_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SCANDIR_ITERATOR_TYPE.get_or_init(|| {
        let tp =
            crate::typedef::make_builtin_type("posix.ScandirIterator", init_scandir_iterator_type);
        crate::typedef::mark_cpython_heap_type(tp, false);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

/// The iterator's own position, or `None` once it has been closed.
const SCANDIR_POSITION: &str = "__scandir_iter__";

fn init_scandir_iterator_type(ns: PyObjectRef) {
    let method = |name: &'static str, f: crate::gateway::BuiltinCodeFn| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            name,
            crate::make_builtin_function(name, f),
        );
    };
    let close = |args: &[PyObjectRef]| -> Result<PyObjectRef, crate::PyError> {
        crate::baseobjspace::setdictvalue_native(args[0], SCANDIR_POSITION, pyre_object::w_none());
        Ok(pyre_object::w_none())
    };
    method("__iter__", |args| Ok(args[0]));
    method("__next__", |args| {
        let position = crate::baseobjspace::getattr_str(args[0], SCANDIR_POSITION)?;
        if unsafe { is_none(position) } {
            return Err(crate::PyError::stop_iteration());
        }
        crate::baseobjspace::next(position)
    });
    method("__enter__", |args| Ok(args[0]));
    method("__exit__", close);
    method("close", close);
}

/// `posix.scandir(path=".")` — the entries the seam reports, as the records
/// `os.walk`, `glob` and `shutil` read them through.
///
/// The omitted argument is the `"."` the signature names, and its entries
/// carry the `./name` paths that spelling joins to.  A `bytes` path asks for
/// `bytes` names, as it does for `listdir`.  `fd` is not one of the types the
/// argument accepts: the seam takes a name, so a descriptor names no directory
/// on the other side of it.
fn scandir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path"], "scandir")?;
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "scandir() takes at most 1 argument ({} given)",
            pos.len()
        )));
    }
    let w_path = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", "scandir", 1)?
        .unwrap_or(pyre_object::w_none());
    let resolved = crate::gateway::fsencode_path_or_fd_nullable_w(w_path, "scandir", false)?;
    let bytes_mode = unsafe { resolved.is_bytes() };
    // An omitted path is the directory `listdir` reads for the same argument,
    // and the prefix its entries are named relative to.
    let dir_bytes: &[u8] = if resolved.as_bytes.is_empty() && unsafe { is_none(w_path) } {
        b"."
    } else {
        &resolved.as_bytes
    };
    let names = crate::importing::source_list_dir(&seam_path(&resolved.as_bytes))
        .map_err(|e| seam_error(&e, resolved.w_path()))?;
    let _roots = pyre_object::gc_roots::push_roots();
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for name in &names {
        items.push(new_dir_entry(bytes_mode, dir_bytes, name));
    }
    let entries_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_list_new(items.take()));
    let iterator_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_instance_new(scandir_iterator_type()));
    let position =
        crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(entries_slot))?;
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(iterator_slot),
        SCANDIR_POSITION,
        position,
    );
    Ok(pyre_object::gc_roots::shadow_stack_get(iterator_slot))
}

/// `O_*` — the darwin/BSD numbering the guest's `errno` table already
/// follows, so a flag and the errno a call refuses it with are drawn from one
/// platform rather than two.
mod oflag {
    pub const RDONLY: i64 = 0x0000;
    pub const WRONLY: i64 = 0x0001;
    pub const RDWR: i64 = 0x0002;
    pub const ACCMODE: i64 = 0x0003;
    pub const NONBLOCK: i64 = 0x0004;
    pub const APPEND: i64 = 0x0008;
    pub const SYNC: i64 = 0x0080;
    pub const NOFOLLOW: i64 = 0x0100;
    pub const CREAT: i64 = 0x0200;
    pub const TRUNC: i64 = 0x0400;
    pub const EXCL: i64 = 0x0800;
    pub const DIRECTORY: i64 = 0x0010_0000;
    pub const CLOEXEC: i64 = 0x0100_0000;
}

/// An open descriptor: the bytes the path resolved to, and how far the guest
/// has read them.
///
/// A directory carries no bytes.  It is opened so that `O_DIRECTORY` can mean
/// what it means, and reading it is `EISDIR` — the same two answers a
/// directory descriptor gives anywhere else.
struct OpenFile {
    data: Vec<u8>,
    pos: u64,
    is_dir: bool,
}

/// The standard streams hold 0, 1 and 2, so a descriptor this table hands out
/// starts above them.
const FIRST_FD: i32 = 3;
/// The table is the whole descriptor space on this target, so it is what runs
/// out: past this, `open` reports `EMFILE` rather than growing without bound.
const MAX_OPEN_FILES: i32 = 4096;

static OPEN_FILES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::BTreeMap<i32, OpenFile>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));

/// Take the lowest free number, which is the number `open` is required to
/// return.
fn insert_open_file(file: OpenFile) -> Result<i32, crate::PyError> {
    let mut table = OPEN_FILES.lock().unwrap();
    let fd = (FIRST_FD..FIRST_FD + MAX_OPEN_FILES)
        .find(|n| !table.contains_key(n))
        .ok_or_else(|| {
            crate::PyError::os_error_syscall(
                crate::builtins::wasm_errno::EMFILE,
                pyre_object::w_none(),
            )
        })?;
    table.insert(fd, file);
    Ok(fd)
}

/// Run `f` over the descriptor numbered `fd`, or report the number as one no
/// descriptor is open on.
fn with_raw_fd<R>(
    fd: i32,
    f: impl FnOnce(&mut OpenFile) -> Result<R, crate::PyError>,
) -> Result<R, crate::PyError> {
    let mut table = OPEN_FILES.lock().unwrap();
    let Some(file) = table.get_mut(&fd) else {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::EBADF,
            pyre_object::w_none(),
        ));
    };
    f(file)
}

/// [`with_raw_fd`] where the descriptor arrives as a Python argument.
fn with_open_file<R>(
    w_fd: PyObjectRef,
    f: impl FnOnce(&mut OpenFile) -> Result<R, crate::PyError>,
) -> Result<R, crate::PyError> {
    with_raw_fd(crate::baseobjspace::c_int_w(w_fd)?, f)
}

/// Take up to `n` bytes from the descriptor's position, which advances by what
/// was taken.  `None` takes everything left, which is what a sizeless `read()`
/// asks for.
///
/// The `_io` layer reaches its descriptors through here too, so a number
/// `posix.open` returned is a number `io.open(fd)` can read.
pub(crate) fn fd_take(fd: i32, n: Option<usize>) -> Result<Vec<u8>, crate::PyError> {
    with_raw_fd(fd, |file| {
        if file.is_dir {
            return Err(crate::PyError::os_error_syscall(
                crate::builtins::wasm_errno::EISDIR,
                pyre_object::w_none(),
            ));
        }
        let start = (file.pos as usize).min(file.data.len());
        let end = match n {
            Some(n) => start.saturating_add(n).min(file.data.len()),
            None => file.data.len(),
        };
        file.pos = end as u64;
        Ok(file.data[start..end].to_vec())
    })
}

/// [`fd_take`] into a caller's buffer, reporting how much of it was filled.
pub(crate) fn fd_read_into(fd: i32, target: &mut [u8]) -> Result<usize, crate::PyError> {
    let data = fd_take(fd, Some(target.len()))?;
    target[..data.len()].copy_from_slice(&data);
    Ok(data.len())
}

/// Move the descriptor's position and report where it landed.
pub(crate) fn fd_lseek(fd: i32, offset: i64, whence: i32) -> Result<i64, crate::PyError> {
    with_raw_fd(fd, |file| {
        let base = match whence {
            0 => 0,
            1 => file.pos as i64,
            2 => file.data.len() as i64,
            _ => {
                return Err(crate::PyError::os_error_with_errno(
                    crate::builtins::wasm_errno::EINVAL,
                    "lseek: unknown whence",
                ));
            }
        };
        let Some(pos) = base.checked_add(offset).filter(|p| *p >= 0) else {
            return Err(crate::PyError::os_error_with_errno(
                crate::builtins::wasm_errno::EINVAL,
                "lseek: position out of range",
            ));
        };
        file.pos = pos as u64;
        Ok(pos)
    })
}

/// Whether the descriptor is open on a directory, which is the one fact a
/// stream needs before it adopts a number: `EISDIR` is what `io.open(fd)` owes
/// a caller who hands it one.
pub(crate) fn fd_is_dir(fd: i32) -> Result<bool, crate::PyError> {
    with_raw_fd(fd, |file| Ok(file.is_dir))
}

/// Drop the descriptor and the bytes it held.
pub(crate) fn fd_close(fd: i32) -> Result<(), crate::PyError> {
    if OPEN_FILES.lock().unwrap().remove(&fd).is_none() {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::EBADF,
            pyre_object::w_none(),
        ));
    }
    Ok(())
}

/// The answer a write to an open descriptor gets: the seam has no writing
/// half, so the mount is read-only rather than the descriptor unusable.
pub(crate) fn fd_refuse_write(fd: i32) -> crate::PyError {
    match with_raw_fd(fd, |_| Ok(())) {
        Ok(()) => crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::EROFS,
            pyre_object::w_none(),
        ),
        Err(e) => e,
    }
}

/// The path argument of a one-path entry point, with `dir_fd` refused the way
/// `stat` refuses it: the seam takes a name and nothing else.
fn path_arg(
    args: &[PyObjectRef],
    name: &'static str,
) -> Result<crate::gateway::FsEncodedPath, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path", "dir_fd"], name)?;
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
    let resolved = crate::gateway::fsencode_path_or_fd_w(w_path, name, false)?;
    if crate::builtins::kwarg_get(kwargs, "dir_fd").is_some_and(|v| !unsafe { is_none(v) }) {
        return Err(crate::PyError::not_implemented(
            "dir_fd unavailable on this platform",
        ));
    }
    Ok(resolved)
}

/// `posix.open(path, flags, mode=0o777, *, dir_fd=None)` — a descriptor over
/// the bytes the seam reports for `path`.
///
/// The whole file is read here rather than at the first `read`: the seam has
/// no way to resume a path part-way, so the descriptor is the only place the
/// bytes can live.  `mode` is accepted and unused — it describes a file that
/// is about to be created, and this seam creates none.
fn open_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path", "flags", "mode", "dir_fd"], "open")?;
    if pos.len() > 3 {
        return Err(crate::PyError::type_error(format!(
            "open() takes at most 3 positional arguments ({} given)",
            pos.len()
        )));
    }
    let Some(w_path) = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", "open", 3)? else {
        return Err(crate::PyError::type_error(
            "open() missing required argument 'path' (pos 1)",
        ));
    };
    let Some(w_flags) = crate::builtins::bind_pos_or_kw(pos, kwargs, 1, "flags", "open", 3)? else {
        return Err(crate::PyError::type_error(
            "open() missing required argument 'flags' (pos 2)",
        ));
    };
    let _ = crate::builtins::bind_pos_or_kw(pos, kwargs, 2, "mode", "open", 3)?;
    let resolved = crate::gateway::fsencode_path_or_fd_w(w_path, "open", false)?;
    if crate::builtins::kwarg_get(kwargs, "dir_fd").is_some_and(|v| !unsafe { is_none(v) }) {
        return Err(crate::PyError::not_implemented(
            "dir_fd unavailable on this platform",
        ));
    }
    let flags = crate::baseobjspace::int_w(w_flags)?;
    // Every bit that asks to change the mount, refused where the file is
    // named rather than at the write that would discover it.
    if flags & oflag::ACCMODE != oflag::RDONLY
        || flags & (oflag::CREAT | oflag::TRUNC | oflag::APPEND | oflag::EXCL) != 0
    {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::EROFS,
            resolved.w_path(),
        ));
    }
    let path = seam_path(&resolved.as_bytes);
    let is_dir = crate::importing::source_is_dir(&path);
    if flags & oflag::DIRECTORY != 0 && !is_dir {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::ENOTDIR,
            resolved.w_path(),
        ));
    }
    let data = if is_dir {
        Vec::new()
    } else {
        crate::importing::read_source_bytes(&path)
            .map_err(|e| seam_error(&e, resolved.w_path()))?
    };
    let fd = insert_open_file(OpenFile {
        data,
        pos: 0,
        is_dir,
    })?;
    Ok(pyre_object::w_int_new(fd as i64))
}

/// `posix.close(fd)` — drop the descriptor and the bytes it held.
fn close_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    fd_close(crate::baseobjspace::c_int_w(args[0])?)?;
    Ok(pyre_object::w_none())
}

/// `posix.read(fd, length)` — up to `length` bytes from the position, which
/// advances by what was returned.  A short answer at the end is the end.
fn read_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let n_signed = crate::baseobjspace::int_w(args[1])?;
    // A negative size would wrap to a huge `usize` (and allocation);
    // os.read rejects it with EINVAL, matching the host read(2).
    if n_signed < 0 {
        return Err(crate::PyError::os_error_with_errno(
            crate::builtins::wasm_errno::EINVAL,
            "read: negative size",
        ));
    }
    let fd = crate::baseobjspace::c_int_w(args[0])?;
    let data = fd_take(fd, Some(n_signed as usize))?;
    Ok(pyre_object::w_bytes_from_bytes(&data))
}

/// `posix.lseek(fd, position, whence)` — the new position.
///
/// A position past the end is a position, not an error: it reads as empty and
/// is what `SEEK_END` with a positive offset names.
fn lseek_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fd = crate::baseobjspace::c_int_w(args[0])?;
    let offset = crate::baseobjspace::int_w(args[1])?;
    let whence = crate::baseobjspace::int_w(args[2])?;
    Ok(pyre_object::w_int_new(fd_lseek(
        fd,
        offset,
        i32::try_from(whence).unwrap_or(i32::MAX),
    )?))
}

/// `posix.fstat(fd)` — the same two facts `stat` reports, taken from the
/// descriptor rather than from a name.
fn fstat_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    with_open_file(args[0], |file| {
        Ok(if file.is_dir {
            make_stat_result(S_IFDIR | 0o555, 0)
        } else {
            make_stat_result(S_IFREG | 0o444, file.data.len() as i64)
        })
    })
}

/// `posix.readlink(path, *, dir_fd=None)` — the seam resolves a name and does
/// not report whether it went through a link, so no path it answers for is a
/// symbolic link.  `EINVAL` is what `readlink` answers for one that is not.
fn readlink(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let resolved = path_arg(args, "readlink")?;
    let path = seam_path(&resolved.as_bytes);
    if !crate::importing::source_is_dir(&path)
        && let Err(e) = crate::importing::source_file_size(&path)
    {
        return Err(seam_error(&e, resolved.w_path()));
    }
    Err(crate::PyError::os_error_syscall(
        crate::builtins::wasm_errno::EINVAL,
        resolved.w_path(),
    ))
}

/// The guest's environment.
///
/// This is what a C library's `environ` array is: the store `putenv` and
/// `unsetenv` write to, and the one `posix.environ` is built from once.  `os.py`
/// wraps that dict rather than copying it, so `os.environ` tracks the dict and
/// these two track this map -- the same two-place split every platform has, and
/// the reason a bare `os.putenv` is not visible through `os.environ` there
/// either.  `_create_environ` is what reads this side back, which is how
/// `os.reload_environ()` sees what `putenv` wrote.
///
/// The guest is started with no environment: the runner forwards the few
/// variables the launcher options resolve against rather than an environment
/// block, so this begins empty and holds whatever the program puts in it.
static ENVIRONMENT: std::sync::LazyLock<
    std::sync::Mutex<std::collections::BTreeMap<Vec<u8>, Vec<u8>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));

/// A fresh dict holding what the environment holds now.
///
/// Bytes on both sides: `os.py` encodes a key before it stores one and decodes
/// it on the way back, and `os.environb` hands the same dict out undecoded.
fn environ_snapshot() -> PyObjectRef {
    let entries: Vec<(Vec<u8>, Vec<u8>)> = ENVIRONMENT
        .lock()
        .unwrap()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
    for (name, value) in entries {
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_bytes_from_bytes(&name));
        let w_value = pyre_object::w_bytes_from_bytes(&value);
        let _ = crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            w_value,
        );
    }
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

/// The name or value argument of `putenv` / `unsetenv`, as the bytes the
/// environment holds.
///
/// `os.environ` hands these down already encoded, and `os.environb` hands down
/// bytes to begin with, so both spellings arrive here and both are kept as
/// bytes: a name that came from the embedder is a name the guest can set again,
/// whatever it spells.
fn env_bytes_arg(
    w_arg: PyObjectRef,
    name: &'static str,
) -> Result<crate::gateway::FsEncodedPath, crate::PyError> {
    crate::gateway::fsencode_path_w(w_arg).map_err(|_| {
        crate::PyError::type_error(format!("{name}() argument must be str or bytes"))
    })
}

/// `posix.putenv(name, value)` — add or replace one variable.
fn putenv(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &[], "putenv")?;
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "putenv expected 2 arguments, got {}",
            pos.len()
        )));
    }
    let name = env_bytes_arg(pos[0], "putenv")?;
    // `putenv` is defined over `name=value`, so a name carrying the separator
    // could not be spelled back out of the environment it went into.
    if name.as_bytes.contains(&b'=') {
        return Err(crate::PyError::value_error(
            "illegal environment variable name",
        ));
    }
    let value = env_bytes_arg(pos[1], "putenv")?;
    ENVIRONMENT
        .lock()
        .unwrap()
        .insert(name.as_bytes.clone(), value.as_bytes.clone());
    Ok(pyre_object::w_none())
}

/// `posix.unsetenv(name)` — remove one variable, or leave an absent one absent.
fn unsetenv(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = env_bytes_arg(
        args.first()
            .copied()
            .ok_or_else(|| crate::PyError::type_error("unsetenv expected 1 argument, got 0"))?,
        "unsetenv",
    )?;
    // A name nothing set is a name already unset, which is what the syscall
    // reports and what `os.environ.__delitem__` relies on so that it can raise
    // the KeyError itself, against the caller's own spelling of the key.
    ENVIRONMENT.lock().unwrap().remove(&name.as_bytes);
    Ok(pyre_object::w_none())
}

/// `posix.unlink(path, *, dir_fd=None)` / `posix.rmdir(path, *, dir_fd=None)`
/// — the entry points a read-only mount publishes and refuses.
fn refuse_write(args: &[PyObjectRef], name: &'static str) -> Result<PyObjectRef, crate::PyError> {
    let resolved = path_arg(args, name)?;
    Err(crate::PyError::os_error_syscall(
        crate::builtins::wasm_errno::EROFS,
        resolved.w_path(),
    ))
}

pub fn register_module(ns: PyObjectRef) -> Result<(), crate::PyError> {
    // `os._create_environ_mapping` binds this dict itself rather than copying
    // it, so `os.environ._data` is this object and `os.environ` is a live view
    // of it.  The guest is started with no environment — the runner hands the
    // interpreter the few variables it forwards, rather than an environment
    // block — so this begins empty and holds whatever the program puts in it.
    crate::module_ns_store(ns, "environ", environ_snapshot());
    // `os.py` builds `supports_dir_fd` and friends only when this exists, and
    // then reads `stat` out of its own namespace to seed `supports_fd`.  Empty:
    // no call here takes a directory or a descriptor argument.
    crate::module_ns_store(ns, "_have_functions", pyre_object::w_list_new(vec![]));
    crate::module_ns_store(ns, "stat_result", super::stat_result_seq_type());
    // `posixmodule_exec` publishes this type on every platform, and only the
    // `get_terminal_size` that fills one is guarded: `shutil.get_terminal_size`
    // catches the AttributeError from the missing call and builds its fallback
    // out of the type, so a target with neither raises from the handler.
    crate::module_ns_store(ns, "terminal_size", terminal_size_seq_type());
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
    crate::module_ns_store(
        ns,
        "scandir",
        crate::make_builtin_function("scandir", scandir),
    );
    crate::module_ns_store(ns, "chdir", crate::make_builtin_function("chdir", chdir));
    // `os.environ.__setitem__` calls `putenv` by name, so a target without it
    // cannot set a variable at all.
    crate::module_ns_store(ns, "putenv", crate::make_builtin_function("putenv", putenv));
    // `os.reload_environ` exists only where this does, and it is the one reader
    // that can see what a bare `putenv` wrote.
    crate::module_ns_store(
        ns,
        "_create_environ",
        crate::make_builtin_function_with_arity(
            "_create_environ",
            |_args| Ok(environ_snapshot()),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "unsetenv",
        crate::make_builtin_function_with_arity("unsetenv", unsetenv, 1),
    );
    // One wasm instance, one process, and it is not one the embedder named.
    crate::module_ns_store(
        ns,
        "getpid",
        crate::make_builtin_function_with_arity("getpid", |_args| Ok(pyre_object::w_int_new(1)), 0),
    );
    // No terminal is reachable through the seam, so no descriptor is one.  The
    // answer is False rather than an error for the same reason it is on every
    // other platform: `isatty` reports, it does not validate.
    crate::module_ns_store(
        ns,
        "isatty",
        crate::make_builtin_function_with_arity(
            "isatty",
            |_args| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    // `os.walk` catches `OSError` from the iterator and `shutil` calls
    // `isinstance(entry, os.DirEntry)`, so both the type and the entries it
    // stamps are published.
    crate::module_ns_store(ns, "DirEntry", dir_entry_type());
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
    // The seam reports the embedder's working directory; without one there is
    // no directory to name, and `""` is what the syscall arm returns when the
    // host refuses too.  `posixpath.abspath` is this call, so every relative
    // path the stdlib resolves goes through it.
    crate::module_ns_store(
        ns,
        "getcwd",
        crate::make_builtin_function_with_arity(
            "getcwd",
            |_args| Ok(crate::gateway::fsdecode_filename_bytes(&cwd_bytes())),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "getcwdb",
        crate::make_builtin_function_with_arity(
            "getcwdb",
            |_args| Ok(pyre_object::w_bytes_from_bytes(&cwd_bytes())),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "urandom",
        crate::make_builtin_function_with_arity("urandom", urandom, 1),
    );
    // ── the descriptor half ──
    crate::module_ns_store(
        ns,
        "open",
        crate::make_builtin_function("open", open_file),
    );
    crate::module_ns_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity("close", close_file, 1),
    );
    crate::module_ns_store(
        ns,
        "read",
        crate::make_builtin_function_with_arity("read", read_file, 2),
    );
    crate::module_ns_store(
        ns,
        "lseek",
        crate::make_builtin_function_with_arity("lseek", lseek_file, 3),
    );
    crate::module_ns_store(
        ns,
        "fstat",
        crate::make_builtin_function_with_arity("fstat", fstat_file, 1),
    );
    crate::module_ns_store(
        ns,
        "readlink",
        crate::make_builtin_function("readlink", readlink),
    );
    crate::module_ns_store(
        ns,
        "unlink",
        crate::make_builtin_function("unlink", |args| refuse_write(args, "unlink")),
    );
    // `os.remove` is the same call under its other name on every platform.
    crate::module_ns_store(
        ns,
        "remove",
        crate::make_builtin_function("remove", |args| refuse_write(args, "remove")),
    );
    crate::module_ns_store(
        ns,
        "rmdir",
        crate::make_builtin_function("rmdir", |args| refuse_write(args, "rmdir")),
    );
    // ── the constants those entry points are called with ──
    let constants: &[(&str, i64)] = &[
        ("O_RDONLY", oflag::RDONLY),
        ("O_WRONLY", oflag::WRONLY),
        ("O_RDWR", oflag::RDWR),
        ("O_ACCMODE", oflag::ACCMODE),
        ("O_NONBLOCK", oflag::NONBLOCK),
        ("O_APPEND", oflag::APPEND),
        ("O_SYNC", oflag::SYNC),
        ("O_NOFOLLOW", oflag::NOFOLLOW),
        ("O_CREAT", oflag::CREAT),
        ("O_TRUNC", oflag::TRUNC),
        ("O_EXCL", oflag::EXCL),
        ("O_DIRECTORY", oflag::DIRECTORY),
        ("O_CLOEXEC", oflag::CLOEXEC),
        ("F_OK", 0),
        ("X_OK", 1),
        ("W_OK", 2),
        ("R_OK", 4),
        ("SEEK_SET", 0),
        ("SEEK_CUR", 1),
        ("SEEK_END", 2),
    ];
    for (name, value) in constants {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(*value));
    }
    Ok(())
}

/// `posix.urandom(n)` — the same entry point the syscall arm publishes, over
/// the entropy the embedder supplies.
///
/// `random.Random()` and `secrets` are both this call, so a guest without it
/// cannot import either; the bytes are the host's, not a stream the guest
/// generates, because that is what the callers are asking for.
fn urandom(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::gateway::check_declared_arity("urandom", 1, args.len())?;
    let n = crate::builtins::space_index_w(args[0])?;
    if n < 0 {
        return Err(crate::PyError::value_error("negative argument not allowed"));
    }
    let n =
        usize::try_from(n).map_err(|_| crate::PyError::overflow_error("argument out of range"))?;
    // Reporting an entropy failure rather than absorbing it: the alternative
    // is handing a caller who asked for unpredictable bytes a buffer of zeros.
    let buf = crate::importing::host::os::urandom(n).map_err(|e| {
        crate::PyError::os_error_syscall(
            crate::builtins::io_error_posix_errno(&e, crate::builtins::wasm_errno::ENOTSUP),
            pyre_object::w_none(),
        )
    })?;
    Ok(pyre_object::w_bytes_from_bytes(&buf))
}
