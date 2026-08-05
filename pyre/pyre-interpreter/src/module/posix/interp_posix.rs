//! posix implementation — PyPy: pypy/module/posix/interp_posix.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! shared `stat_result_type` helper is carried in here too; `init_posix`
//! is renamed to `register_module`.

use crate::importing::host::{fs as host_fs, os as host_os};
use pyre_object::PyObjectRef;
use std::sync::{LazyLock, Mutex};
// Under sandbox, name libc through the seam facade so any direct syscall call
// in this module is a compile error (only types/constants/pure fns resolve).
#[cfg(feature = "sandbox")]
use crate::host_seam::sys as libc;

/// PyPy `ApplevelForkCallbacks`, cached on the object space.
///
/// The callback collections are RPython lists, so use insertion-ordered Vecs;
/// this is process/interpreter state, never TLS.
#[derive(Default)]
struct ApplevelForkCallbacks {
    before_w: Vec<usize>,
    parent_w: Vec<usize>,
    child_w: Vec<usize>,
}

static APPLEVEL_FORK_CALLBACKS: LazyLock<Mutex<ApplevelForkCallbacks>> =
    LazyLock::new(|| Mutex::new(ApplevelForkCallbacks::default()));
// PyPy's GIL serializes concurrent fork entry.  Pyre is free-threaded, so the
// corresponding process operation has its own narrow serializer.
static FORK_SERIALIZER: Mutex<()> = Mutex::new(());

#[cfg(all(unix, feature = "host_env", not(target_os = "redox")))]
fn sysconf_names() -> &'static [(&'static str, i32)] {
    &[
        ("SC_2_CHAR_TERM", libc::_SC_2_CHAR_TERM),
        ("SC_2_C_BIND", libc::_SC_2_C_BIND),
        ("SC_2_C_DEV", libc::_SC_2_C_DEV),
        ("SC_2_FORT_DEV", libc::_SC_2_FORT_DEV),
        ("SC_2_FORT_RUN", libc::_SC_2_FORT_RUN),
        ("SC_2_LOCALEDEF", libc::_SC_2_LOCALEDEF),
        ("SC_2_SW_DEV", libc::_SC_2_SW_DEV),
        ("SC_2_UPE", libc::_SC_2_UPE),
        ("SC_2_VERSION", libc::_SC_2_VERSION),
        ("SC_AIO_LISTIO_MAX", libc::_SC_AIO_LISTIO_MAX),
        ("SC_AIO_MAX", libc::_SC_AIO_MAX),
        ("SC_AIO_PRIO_DELTA_MAX", libc::_SC_AIO_PRIO_DELTA_MAX),
        ("SC_ARG_MAX", libc::_SC_ARG_MAX),
        ("SC_ASYNCHRONOUS_IO", libc::_SC_ASYNCHRONOUS_IO),
        ("SC_ATEXIT_MAX", libc::_SC_ATEXIT_MAX),
        ("SC_BC_BASE_MAX", libc::_SC_BC_BASE_MAX),
        ("SC_BC_DIM_MAX", libc::_SC_BC_DIM_MAX),
        ("SC_BC_SCALE_MAX", libc::_SC_BC_SCALE_MAX),
        ("SC_BC_STRING_MAX", libc::_SC_BC_STRING_MAX),
        ("SC_CHILD_MAX", libc::_SC_CHILD_MAX),
        ("SC_CLK_TCK", libc::_SC_CLK_TCK),
        ("SC_COLL_WEIGHTS_MAX", libc::_SC_COLL_WEIGHTS_MAX),
        ("SC_DELAYTIMER_MAX", libc::_SC_DELAYTIMER_MAX),
        ("SC_EXPR_NEST_MAX", libc::_SC_EXPR_NEST_MAX),
        ("SC_FSYNC", libc::_SC_FSYNC),
        ("SC_GETGR_R_SIZE_MAX", libc::_SC_GETGR_R_SIZE_MAX),
        ("SC_GETPW_R_SIZE_MAX", libc::_SC_GETPW_R_SIZE_MAX),
        ("SC_IOV_MAX", libc::_SC_IOV_MAX),
        ("SC_JOB_CONTROL", libc::_SC_JOB_CONTROL),
        ("SC_LINE_MAX", libc::_SC_LINE_MAX),
        ("SC_LOGIN_NAME_MAX", libc::_SC_LOGIN_NAME_MAX),
        ("SC_MAPPED_FILES", libc::_SC_MAPPED_FILES),
        ("SC_MEMLOCK", libc::_SC_MEMLOCK),
        ("SC_MEMLOCK_RANGE", libc::_SC_MEMLOCK_RANGE),
        ("SC_MEMORY_PROTECTION", libc::_SC_MEMORY_PROTECTION),
        ("SC_MESSAGE_PASSING", libc::_SC_MESSAGE_PASSING),
        ("SC_MQ_OPEN_MAX", libc::_SC_MQ_OPEN_MAX),
        ("SC_MQ_PRIO_MAX", libc::_SC_MQ_PRIO_MAX),
        ("SC_NGROUPS_MAX", libc::_SC_NGROUPS_MAX),
        ("SC_NPROCESSORS_CONF", libc::_SC_NPROCESSORS_CONF),
        ("SC_NPROCESSORS_ONLN", libc::_SC_NPROCESSORS_ONLN),
        ("SC_OPEN_MAX", libc::_SC_OPEN_MAX),
        ("SC_PAGE_SIZE", libc::_SC_PAGE_SIZE),
        ("SC_PAGESIZE", libc::_SC_PAGE_SIZE),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_PASS_MAX", libc::_SC_PASS_MAX),
        ("SC_PHYS_PAGES", libc::_SC_PHYS_PAGES),
        ("SC_PRIORITIZED_IO", libc::_SC_PRIORITIZED_IO),
        ("SC_PRIORITY_SCHEDULING", libc::_SC_PRIORITY_SCHEDULING),
        ("SC_REALTIME_SIGNALS", libc::_SC_REALTIME_SIGNALS),
        ("SC_RE_DUP_MAX", libc::_SC_RE_DUP_MAX),
        ("SC_RTSIG_MAX", libc::_SC_RTSIG_MAX),
        ("SC_SAVED_IDS", libc::_SC_SAVED_IDS),
        ("SC_SEMAPHORES", libc::_SC_SEMAPHORES),
        ("SC_SEM_NSEMS_MAX", libc::_SC_SEM_NSEMS_MAX),
        ("SC_SEM_VALUE_MAX", libc::_SC_SEM_VALUE_MAX),
        ("SC_SHARED_MEMORY_OBJECTS", libc::_SC_SHARED_MEMORY_OBJECTS),
        ("SC_SIGQUEUE_MAX", libc::_SC_SIGQUEUE_MAX),
        ("SC_STREAM_MAX", libc::_SC_STREAM_MAX),
        ("SC_SYNCHRONIZED_IO", libc::_SC_SYNCHRONIZED_IO),
        ("SC_THREADS", libc::_SC_THREADS),
        ("SC_THREAD_ATTR_STACKADDR", libc::_SC_THREAD_ATTR_STACKADDR),
        ("SC_THREAD_ATTR_STACKSIZE", libc::_SC_THREAD_ATTR_STACKSIZE),
        (
            "SC_THREAD_DESTRUCTOR_ITERATIONS",
            libc::_SC_THREAD_DESTRUCTOR_ITERATIONS,
        ),
        ("SC_THREAD_KEYS_MAX", libc::_SC_THREAD_KEYS_MAX),
        (
            "SC_THREAD_PRIORITY_SCHEDULING",
            libc::_SC_THREAD_PRIORITY_SCHEDULING,
        ),
        ("SC_THREAD_PRIO_INHERIT", libc::_SC_THREAD_PRIO_INHERIT),
        ("SC_THREAD_PRIO_PROTECT", libc::_SC_THREAD_PRIO_PROTECT),
        (
            "SC_THREAD_PROCESS_SHARED",
            libc::_SC_THREAD_PROCESS_SHARED,
        ),
        ("SC_THREAD_SAFE_FUNCTIONS", libc::_SC_THREAD_SAFE_FUNCTIONS),
        ("SC_THREAD_STACK_MIN", libc::_SC_THREAD_STACK_MIN),
        ("SC_THREAD_THREADS_MAX", libc::_SC_THREAD_THREADS_MAX),
        ("SC_TIMERS", libc::_SC_TIMERS),
        ("SC_TIMER_MAX", libc::_SC_TIMER_MAX),
        ("SC_TTY_NAME_MAX", libc::_SC_TTY_NAME_MAX),
        ("SC_TZNAME_MAX", libc::_SC_TZNAME_MAX),
        ("SC_VERSION", libc::_SC_VERSION),
        ("SC_XOPEN_CRYPT", libc::_SC_XOPEN_CRYPT),
        ("SC_XOPEN_ENH_I18N", libc::_SC_XOPEN_ENH_I18N),
        ("SC_XOPEN_LEGACY", libc::_SC_XOPEN_LEGACY),
        ("SC_XOPEN_REALTIME", libc::_SC_XOPEN_REALTIME),
        (
            "SC_XOPEN_REALTIME_THREADS",
            libc::_SC_XOPEN_REALTIME_THREADS,
        ),
        ("SC_XOPEN_SHM", libc::_SC_XOPEN_SHM),
        ("SC_XOPEN_UNIX", libc::_SC_XOPEN_UNIX),
        ("SC_XOPEN_VERSION", libc::_SC_XOPEN_VERSION),
        ("SC_XOPEN_XCU_VERSION", libc::_SC_XOPEN_XCU_VERSION),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_ILP32_OFF32", libc::_SC_XBS5_ILP32_OFF32),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_ILP32_OFFBIG", libc::_SC_XBS5_ILP32_OFFBIG),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_LP64_OFF64", libc::_SC_XBS5_LP64_OFF64),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_LPBIG_OFFBIG", libc::_SC_XBS5_LPBIG_OFFBIG),
    ]
}

pub(crate) fn walk_fork_callback_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let mut callbacks = APPLEVEL_FORK_CALLBACKS.lock().unwrap();
    let ApplevelForkCallbacks {
        before_w,
        parent_w,
        child_w,
    } = &mut *callbacks;
    for callbacks in [before_w, parent_w, child_w] {
        for callback in callbacks {
            visitor(unsafe { &mut *(callback as *mut usize as *mut PyObjectRef) });
        }
    }
}

fn run_fork_callbacks(kind: &str) {
    let reverse = kind == "before";
    let initial_len = {
        let callbacks = APPLEVEL_FORK_CALLBACKS.lock().unwrap();
        match kind {
            "before" => callbacks.before_w.len(),
            "parent" => callbacks.parent_w.len(),
            "child" => callbacks.child_w.len(),
            _ => unreachable!(),
        }
    };
    let indices: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..initial_len).rev())
    } else {
        Box::new(0..initial_len)
    };
    for index in indices {
        let callback = {
            let callbacks = APPLEVEL_FORK_CALLBACKS.lock().unwrap();
            match kind {
                "before" => callbacks.before_w.get(index),
                "parent" => callbacks.parent_w.get(index),
                "child" => callbacks.child_w.get(index),
                _ => unreachable!(),
            }
            .copied()
        };
        let Some(callback) = callback else { continue };
        if let Err(mut error) = crate::call::call_function_impl_result(callback as PyObjectRef, &[])
        {
            error.write_unraisable(pyre_object::w_none(), "fork hook", callback as PyObjectRef);
        }
    }
}

fn register_at_fork(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if !pos.is_empty() {
        return Err(crate::PyError::type_error(
            "register_at_fork() takes no positional arguments",
        ));
    }
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &["before", "after_in_parent", "after_in_child"],
        "register_at_fork",
    )?;
    let before = crate::builtins::kwarg_get(kwargs, "before");
    let parent = crate::builtins::kwarg_get(kwargs, "after_in_parent");
    let child = crate::builtins::kwarg_get(kwargs, "after_in_child");
    if before.is_none() && parent.is_none() && child.is_none() {
        return Err(crate::PyError::type_error(
            "At least one argument is required.",
        ));
    }
    for (name, callback) in [
        ("before", before),
        ("after_in_parent", parent),
        ("after_in_child", child),
    ] {
        if callback.is_some_and(|callback| !crate::baseobjspace::callable_w(callback)) {
            return Err(crate::PyError::type_error(format!(
                "'{name}' must be callable",
            )));
        }
    }
    {
        let mut callbacks = APPLEVEL_FORK_CALLBACKS.lock().unwrap();
        if let Some(callback) = before {
            callbacks.before_w.push(callback as usize);
        }
        if let Some(callback) = parent {
            callbacks.parent_w.push(callback as usize);
        }
        if let Some(callback) = child {
            callbacks.child_w.push(callback as usize);
        }
    }
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    Ok(pyre_object::w_none())
}

/// `posix.stat_result` — a real structseq (tuple subclass) so `st[0]`,
/// `len(st)`, iteration and `isinstance(st, tuple)` all work, matching
/// `posixmodule.c` `stat_result_desc`.  The 10 sequence slots hold the
/// integer fields, with the integer-seconds times at 7..10 under the
/// hidden `_integer_atime`/`_integer_mtime`/`_integer_ctime` names; the
/// float `st_atime`/`st_mtime`/`st_ctime`, the `st_*_ns` integers, and the
/// `st_blksize`/`st_blocks`/`st_rdev` block-device fields are named-only
/// extras.
fn stat_result_seq_type() -> PyObjectRef {
    static STAT_RESULT_SEQ_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *STAT_RESULT_SEQ_TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq_with_extra(
            // Dotted name → `__name__` "stat_result", repr "os.stat_result(...)".
            "os.stat_result",
            // `app_posix.py:20-37` — slots 7..10 are the hidden integer
            // timestamps; the float `st_atime`/`st_mtime`/`st_ctime` are
            // named-only extras, never indexable.
            &[
                "st_mode",
                "st_ino",
                "st_dev",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "_integer_atime",
                "_integer_mtime",
                "_integer_ctime",
            ],
            // `app_posix.py:38-69` — named-only extras ordered by their
            // `structseqfield` index (11..13, 20..23, 40..42, 50..52).
            // `structseq_descr_new` fills surplus sequence items into
            // this list in order, so the list order must match PyPy's
            // index sort, not the build-time population order.
            &[
                // float times, indices 11..13.
                "st_atime",
                "st_mtime",
                "st_ctime",
                // `app_posix.py:45-52` — present where the platform's
                // `struct stat` carries them (every Unix target),
                // indices 20..23.
                #[cfg(unix)]
                "st_blksize",
                #[cfg(unix)]
                "st_blocks",
                #[cfg(unix)]
                "st_rdev",
                // `rposix_stat.py` exposes `st_flags` where the C
                // `struct stat` carries it (BSD family / macOS).
                #[cfg(target_os = "macos")]
                "st_flags",
                // `build_stat_result` (interp_posix.py:554-557) +
                // `rposix_stat.py STAT_FIELDS += ALL_STAT_FIELDS[-3:]`
                // — the sub-second nanosecond remainders, indices 40..42.
                "nsec_atime",
                "nsec_mtime",
                "nsec_ctime",
                // full nanosecond timestamps, indices 50..52.
                "st_atime_ns",
                "st_mtime_ns",
                "st_ctime_ns",
            ],
        ) as usize
    }) as PyObjectRef
}

/// `os.terminal_size` structseq — `(columns, lines)`.
fn terminal_size_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq("os.terminal_size", &["columns", "lines"]) as usize
    }) as PyObjectRef
}

/// `os.uname_result` structseq — `(sysname, nodename, release, version,
/// machine)`; repr renders "posix.uname_result(...)".
#[cfg(unix)]
fn uname_result_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            "posix.uname_result",
            &["sysname", "nodename", "release", "version", "machine"],
        ) as usize
    }) as PyObjectRef
}

/// `os.statvfs_result` structseq — 10 sequence slots with `f_fsid` as an
/// extra named field (`n_sequence_fields=10`, `n_fields=11`).
fn statvfs_result_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq_with_extra(
            "os.statvfs_result",
            &[
                "f_bsize",
                "f_frsize",
                "f_blocks",
                "f_bfree",
                "f_bavail",
                "f_files",
                "f_ffree",
                "f_favail",
                "f_flag",
                "f_namemax",
            ],
            &["f_fsid"],
        ) as usize
    }) as PyObjectRef
}

/// `os.times_result` structseq — `(user, system, children_user,
/// children_system, elapsed)`; repr renders "posix.times_result(...)".
fn times_result_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            "posix.times_result",
            &[
                "user",
                "system",
                "children_user",
                "children_system",
                "elapsed",
            ],
        ) as usize
    }) as PyObjectRef
}

/// Split `path` into the root and everything after it, the way
/// `ntpath.splitroot` splits it three ways with the drive and the root joined
/// back together.
///
/// `_bootstrap_external._path_join` maps this over its parts and classifies
/// each result: a root that starts or ends with a separator is absolute, one
/// ending in `:` is drive-relative, anything else is a plain relative part.
///
/// Both separators count, so the tests are taken on a copy with `/` rewritten
/// to `\`; that rewrite is character-for-character, so an offset into it
/// addresses the same character of the original. The offsets are per
/// character rather than per byte, which is what makes `"ä:\x"` take the
/// drive branch — at byte 1 it would be a continuation byte and take none.
///
/// Compiled everywhere so the tests run on every platform; only the Windows
/// build has a caller.
#[cfg_attr(not(windows), allow(dead_code))]
fn split_root(path: &str) -> (&str, &str) {
    const SEP: char = '\\';
    const UNC_PREFIX: &str = "\\\\?\\UNC\\";

    let norm: Vec<char> = path
        .chars()
        .map(|c| if c == '/' { SEP } else { c })
        .collect();
    let byte_at = |index: usize| {
        path.char_indices()
            .nth(index)
            .map_or(path.len(), |(offset, _)| offset)
    };
    let sep_from = |start: usize| {
        norm.get(start..)
            .and_then(|rest| rest.iter().position(|&c| c == SEP))
            .map(|offset| offset + start)
    };

    if norm.first() != Some(&SEP) {
        if norm.get(1) == Some(&':') {
            // `X:\Windows` keeps the separator in the root; `X:Windows` names
            // a location on the drive's own cursor and has no root at all.
            let split = if norm.get(2) == Some(&SEP) { 3 } else { 2 };
            return path.split_at(byte_at(split));
        }
        return ("", path);
    }
    if norm.get(1) != Some(&SEP) {
        // A path rooted on the current drive, e.g. `\Windows`.
        return path.split_at(byte_at(1));
    }
    // A UNC share (`\\server\share`, `\\?\UNC\server\share`) or a device
    // (`\\.\device`): the root runs to the separator after the share name,
    // and a path that never reaches a second separator is all root.
    let unc = norm.len() >= 8
        && norm[..8]
            .iter()
            .map(|c| c.to_ascii_uppercase())
            .eq(UNC_PREFIX.chars());
    let start = if unc { 8 } else { 2 };
    match sep_from(start).and_then(|index| sep_from(index + 1)) {
        Some(index) => path.split_at(byte_at(index + 1)),
        None => (path, ""),
    }
}

/// Windows-only `nt` calls — PyPy: pypy/module/posix/interp_nt.py and the
/// `if _WIN32` blocks of interp_posix.py. Registered under the `nt` module
/// name on Windows (moduledef.py `applevel_name = os.name`); `ntpath` reaches
/// for these through `from nt import _getfullpathname` and friends.
#[cfg(all(windows, feature = "host_env"))]
mod win_nt {
    use pyre_object::PyObjectRef;
    use rustpython_host_env::nt as host_nt;
    use std::path::Path;

    /// Wrap a host-layer `io::Error` as an OSError carrying the offending path.
    fn io_err(error: &std::io::Error, path: &str) -> crate::PyError {
        let errno = crate::builtins::io_error_posix_errno(error, 0);
        let filename = if path.is_empty() {
            pyre_object::PY_NULL
        } else {
            pyre_object::w_str_new(path)
        };
        crate::PyError::os_error_syscall(errno, filename)
    }

    fn io_err_with_filename(
        error: &std::io::Error,
        filename: PyObjectRef,
    ) -> crate::PyError {
        crate::PyError::os_error_syscall(
            crate::builtins::io_error_posix_errno(error, 0),
            filename,
        )
    }

    /// Read argument 0 as a filesystem path; the flag reports whether the
    /// input was bytes so the result can be encoded back to match.
    fn arg_path(
        args: &[PyObjectRef],
        func: &str,
    ) -> Result<(String, bool, crate::gateway::FsEncodedPath), crate::PyError> {
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(format!(
                "{func}() missing required argument 'path'"
            )));
        };
        let resolved = crate::gateway::fsencode_path_w(arg)?;
        let as_bytes = unsafe { resolved.is_bytes() };
        // Windows names files in UTF-16, so there is no byte spelling to keep
        // here the way there is on a unix path; the host API takes text.
        let path = String::from_utf8_lossy(&resolved.as_bytes).into_owned();
        Ok((path, as_bytes, resolved))
    }

    fn wrap_path(s: &std::ffi::OsStr, as_bytes: bool) -> PyObjectRef {
        let text = s.to_string_lossy();
        if as_bytes {
            pyre_object::w_bytes_from_bytes(text.as_bytes())
        } else {
            pyre_object::w_str_new(&text)
        }
    }

    /// ntpath.abspath helper — resolves `.`/`..` and the drive without
    /// requiring the path to exist.
    pub fn _getfullpathname(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, as_bytes, resolved) = arg_path(args, "_getfullpathname")?;
        match host_nt::getfullpathname(Path::new(&path)) {
            Ok(result) => Ok(wrap_path(&result, as_bytes)),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// ntpath.realpath helper — the canonical `\\?\`-prefixed path, via a
    /// backup-semantics handle so directories open too.
    pub fn _getfinalpathname(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, as_bytes, resolved) = arg_path(args, "_getfinalpathname")?;
        if path.contains('\0') {
            return Err(crate::PyError::value_error("embedded null character"));
        }
        match host_nt::getfinalpathname(Path::new(&path)) {
            Ok(result) => Ok(wrap_path(&result, as_bytes)),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// os.stat helper for ntpath.samefile — (volume serial, file index high,
    /// file index low) uniquely identifies a file across handles. host_env has
    /// no wrapper for GetFileInformationByHandle, so call windows-sys directly.
    pub fn _getfileinformation(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "_getfileinformation() missing required argument 'fd'",
            ));
        };
        let fd = crate::baseobjspace::c_int_w(arg)?;
        let handle = host_nt::handle_from_fd(fd);
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { GetFileInformationByHandle(handle, &mut info) } == 0 {
            return Err(io_err(&std::io::Error::last_os_error(), ""));
        }
        Ok(pyre_object::w_tuple_new(vec![
            pyre_object::w_int_new(info.dwVolumeSerialNumber as i64),
            pyre_object::w_int_new(info.nFileIndexHigh as i64),
            pyre_object::w_int_new(info.nFileIndexLow as i64),
        ]))
    }

    /// shutil.disk_usage helper — (total, free) bytes. host_env::getdiskusage
    /// retries against the parent directory when the path names a file
    /// (ERROR_DIRECTORY).
    pub fn _getdiskusage(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, _, resolved) = arg_path(args, "_getdiskusage")?;
        if path.contains('\0') {
            return Err(crate::PyError::value_error("embedded null character"));
        }
        match host_nt::getdiskusage(Path::new(&path)) {
            Ok((total, free)) => Ok(pyre_object::w_tuple_new(vec![
                pyre_object::w_int_new(total as i64),
                pyre_object::w_int_new(free as i64),
            ])),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// os.get_handle_inheritable — the argument is an OS handle value, not a
    /// CRT fd (rwin32.cast(HANDLE, fd)).
    pub fn get_handle_inheritable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "get_handle_inheritable() missing required argument 'handle'",
            ));
        };
        let handle = crate::baseobjspace::c_int_w(arg)? as libc::intptr_t;
        match host_nt::get_handle_inheritable(handle) {
            Ok(value) => Ok(pyre_object::w_bool_from(value)),
            Err(error) => Err(io_err(&error, "")),
        }
    }

    /// os.set_handle_inheritable.
    pub fn set_handle_inheritable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() < 2 {
            return Err(crate::PyError::type_error(
                "set_handle_inheritable() takes 2 arguments",
            ));
        }
        let handle = crate::baseobjspace::c_int_w(args[0])? as libc::intptr_t;
        let inheritable = crate::baseobjspace::is_true(args[1])?;
        match host_nt::set_handle_inheritable(handle, inheritable) {
            Ok(()) => Ok(pyre_object::w_none()),
            Err(error) => Err(io_err(&error, "")),
        }
    }

    /// os._add_dll_directory. PyPy hands back a W_DLLCapsule; os.py only
    /// round-trips the value into `_remove_dll_directory`, so the opaque
    /// DLL_DIRECTORY_COOKIE pointer is returned as an int instead. host_env has
    /// no AddDllDirectory wrapper, so call windows-sys directly.
    pub fn _add_dll_directory(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        use windows_sys::Win32::System::LibraryLoader::AddDllDirectory;
        let (path, _, resolved) = arg_path(args, "_add_dll_directory")?;
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        let cookie = unsafe { AddDllDirectory(wide.as_ptr()) };
        if cookie.is_null() {
            return Err(io_err_with_filename(
                &std::io::Error::last_os_error(),
                resolved.w_path(),
            ));
        }
        Ok(pyre_object::w_int_new(cookie as usize as i64))
    }

    /// os._remove_dll_directory — takes the cookie returned above.
    pub fn _remove_dll_directory(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        use windows_sys::Win32::System::LibraryLoader::RemoveDllDirectory;
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "_remove_dll_directory() missing required argument 'cookie'",
            ));
        };
        let cookie =
            (crate::baseobjspace::int_w(arg)? as usize) as *mut std::ffi::c_void;
        let ok = unsafe { RemoveDllDirectory(cookie) };
        Ok(pyre_object::w_bool_from(ok != 0))
    }

    /// os._supports_virtual_terminal — whether stderr's console mode carries
    /// ENABLE_VIRTUAL_TERMINAL_PROCESSING.
    pub fn _supports_virtual_terminal(
        _args: &[PyObjectRef],
    ) -> Result<PyObjectRef, crate::PyError> {
        Ok(pyre_object::w_bool_from(
            host_nt::supports_virtual_terminal(),
        ))
    }
}

/// A fresh dict holding the current process environment.
///
/// PyPy equivalent: posix.State.startup → `_convertenviron`, which copies the
/// environment into `posix.environ` at interpreter startup. os.py seeds
/// `environ` from it at import time and re-reads it from `reload_environ()`.
fn create_environ() -> pyre_object::PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
    // Both halves of an entry stay rooted across the store: either allocation
    // may move what the other one already produced.
    #[cfg_attr(
        not(any(feature = "sandbox", feature = "host_env")),
        expect(dead_code, reason = "no environment source is compiled in")
    )]
    fn store(
        dict_slot: usize,
        key: impl FnOnce() -> pyre_object::PyObjectRef,
        value: impl FnOnce() -> pyre_object::PyObjectRef,
    ) {
        let _entry = pyre_object::gc_roots::push_roots();
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(key());
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(value());
        unsafe {
            pyre_object::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            );
        }
    }
    #[cfg(feature = "sandbox")]
    {
        // The controller delivers the virtual environment as (bytes, bytes).
        if let Ok(items) = crate::host_seam::ops::envitems() {
            for (k_bytes, v_bytes) in items {
                store(
                    dict_slot,
                    || pyre_object::w_bytes_from_bytes(&k_bytes),
                    || pyre_object::w_bytes_from_bytes(&v_bytes),
                );
            }
        }
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox"), not(windows)))]
    {
        // On POSIX, posix.environ stores bytes → bytes. os.py's
        // _create_environ_mapping wraps this dict in an _Environ object that
        // encodes/decodes via surrogateescape when accessed.
        // (_convertenviron: `space.newbytes(key), space.newbytes(value)`.)
        for (key, value) in host_os::vars_os() {
            store(
                dict_slot,
                || pyre_object::w_bytes_from_bytes(key.as_encoded_bytes()),
                || pyre_object::w_bytes_from_bytes(value.as_encoded_bytes()),
            );
        }
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox"), windows))]
    {
        // On Windows nt.environ stores str → str; os.py's nt branch of
        // _create_environ_mapping demands str keys/values and upper-cases the
        // keys itself. (_convertenviron: `space.newtext(key), newtext(value)`.)
        for (key, value) in host_os::vars_os() {
            store(
                dict_slot,
                || pyre_object::w_str_new(&key.to_string_lossy()),
                || pyre_object::w_str_new(&value.to_string_lossy()),
            );
        }
    }
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

/// posix stub — PyPy: pypy/module/posix/ interp_posix.py
///
/// Provides the minimal surface that os.py module init needs to succeed.
/// Real posix calls are not implemented — they raise or return defaults.
pub fn register_module(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(ns, "environ", create_environ());
    crate::module_ns_store(
        ns,
        "_create_environ",
        crate::make_builtin_function_with_arity("_create_environ", |_args| Ok(create_environ()), 0),
    );

    // ── posix.putenv(name, value) / posix.unsetenv(name) ──
    // `os.environ.__setitem__` calls `putenv` before updating its own dict, so
    // without these the mapping and the real process environment drift apart:
    // child processes and any native reader of the environment keep seeing the
    // values captured at startup.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    {
        /// The environment block is a list of NUL-terminated `NAME=VALUE`
        /// strings, so neither half may embed a NUL.
        ///
        /// Bytes, like the path boundary: an entry the process was started with
        /// can hold a byte with no UTF-8 spelling, and `posix.environ` already
        /// hands those back as bytes, so writing one must not fold it first.
        fn env_bytes(arg: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
            let bytes = crate::gateway::fsencode_bytes_w(arg)?;
            if bytes.contains(&0) {
                return Err(crate::PyError::value_error("embedded null byte"));
            }
            Ok(bytes)
        }
        /// A name that is empty or contains `=` cannot be expressed in the
        /// environment block.
        fn illegal_name(name: &[u8]) -> bool {
            name.is_empty() || name.contains(&b'=')
        }
        crate::module_ns_store(
            ns,
            "putenv",
            crate::make_builtin_function_with_arity(
                "putenv",
                |args| {
                    // interp_posix.py putenv_impl rejects the name itself...
                    let name = env_bytes(args[0])?;
                    if illegal_name(&name) {
                        return Err(crate::PyError::value_error(
                            "illegal environment variable name",
                        ));
                    }
                    let value = env_bytes(args[1])?;
                    unsafe {
                        host_os::set_var(os_str_from_bytes(&name), os_str_from_bytes(&value))
                    };
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );
        crate::module_ns_store(
            ns,
            "unsetenv",
            crate::make_builtin_function_with_arity(
                "unsetenv",
                |args| {
                    let name = env_bytes(args[0])?;
                    // ...while unsetenv leaves the same rejection to the
                    // syscall, which reports EINVAL.
                    if illegal_name(&name) {
                        return Err(crate::PyError::os_error_syscall(
                            libc::EINVAL,
                            pyre_object::PY_NULL,
                        ));
                    }
                    unsafe { host_os::remove_var(os_str_from_bytes(&name)) };
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );
    }

    // _have_functions — list of HAVE_* macro names that were defined at
    // build time. os.py uses this to populate the supports_* capability sets
    // (supports_dir_fd / supports_fd / supports_follow_symlinks), which
    // callers like shutil.rmtree consult to choose between fd-relative and
    // path-based implementations. Only the macros whose functionality is
    // actually implemented may be listed: of the `*at` family only
    // HAVE_FSTATAT is listed, because stat/lstat are the only calls that
    // resolve a dir_fd-relative name, and HAVE_FDOPENDIR is omitted because
    // scandir/listdir do not accept a file descriptor. HAVE_LSTAT remains so
    // os.stat is reported in supports_follow_symlinks (follow_symlinks=False
    // works).
    // Under sandbox the fd-relative host probes/mutators (fchdir/fchmod/fchown/
    // fexecve/fpathconf/fstatvfs/ftruncate) are replaced with raising stubs, so
    // drop their capability bits — otherwise os.py picks an fd-relative path
    // that deterministically fails.
    let have_functions: &[&str] = &[
        #[cfg(not(feature = "sandbox"))]
        "HAVE_FCHDIR",
        #[cfg(not(feature = "sandbox"))]
        "HAVE_FCHMOD",
        #[cfg(not(feature = "sandbox"))]
        "HAVE_FCHOWN",
        // Do not advertise HAVE_FEXECVE until execve() accepts an open file
        // descriptor.  os.py uses this bit to add execve to supports_fd, and
        // test_posix then runs a fork+fexecve path whose child must never
        // return to the libregrtest worker.
        #[cfg(not(feature = "sandbox"))]
        "HAVE_FPATHCONF",
        // os.py:120-121 reads this as `stat` and `lstat` honouring dir_fd;
        // `stat_at` implements it with `fstatat`, which the sandbox seam and
        // the non-unix hosts do not carry.
        #[cfg(all(unix, not(feature = "sandbox")))]
        "HAVE_FSTATAT",
        #[cfg(not(feature = "sandbox"))]
        "HAVE_FSTATVFS",
        #[cfg(all(unix, not(feature = "sandbox")))]
        "HAVE_FTRUNCATE",
        "HAVE_FUTIMENS",
        "HAVE_FUTIMES",
        "HAVE_LSTAT",
    ];
    crate::module_ns_store(
        ns,
        "_have_functions",
        pyre_object::w_list_new(
            have_functions
                .iter()
                .map(|&n| pyre_object::w_str_new(n))
                .collect(),
        ),
    );
    // POSIX constants — real libc values (cross-platform subset).
    for (name, val) in [
        // F_OK/R_OK/W_OK/X_OK: Windows doesn't have them in libc crate,
        // define standard POSIX values directly.
        #[cfg(unix)]
        ("F_OK", libc::F_OK as i64),
        #[cfg(not(unix))]
        ("F_OK", 0i64),
        #[cfg(unix)]
        ("R_OK", libc::R_OK as i64),
        #[cfg(not(unix))]
        ("R_OK", 4i64),
        #[cfg(unix)]
        ("W_OK", libc::W_OK as i64),
        #[cfg(not(unix))]
        ("W_OK", 2i64),
        #[cfg(unix)]
        ("X_OK", libc::X_OK as i64),
        #[cfg(not(unix))]
        ("X_OK", 1i64),
        ("O_RDONLY", libc::O_RDONLY as i64),
        ("O_WRONLY", libc::O_WRONLY as i64),
        ("O_RDWR", libc::O_RDWR as i64),
        ("O_APPEND", libc::O_APPEND as i64),
        ("O_CREAT", libc::O_CREAT as i64),
        ("O_EXCL", libc::O_EXCL as i64),
        ("O_TRUNC", libc::O_TRUNC as i64),
        // O_NONBLOCK, O_DSYNC, O_SYNC are Unix-only.
        #[cfg(unix)]
        ("O_NONBLOCK", libc::O_NONBLOCK as i64),
        #[cfg(not(unix))]
        ("O_NONBLOCK", 0i64),
        #[cfg(unix)]
        ("O_NDELAY", libc::O_NONBLOCK as i64),
        #[cfg(not(unix))]
        ("O_NDELAY", 0i64),
        #[cfg(unix)]
        ("O_DSYNC", libc::O_DSYNC as i64),
        #[cfg(not(unix))]
        ("O_DSYNC", 0i64),
        #[cfg(unix)]
        ("O_SYNC", libc::O_SYNC as i64),
        #[cfg(not(unix))]
        ("O_SYNC", 0i64),
        ("SEEK_SET", libc::SEEK_SET as i64),
        ("SEEK_CUR", libc::SEEK_CUR as i64),
        ("SEEK_END", libc::SEEK_END as i64),
    ] {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
    }
    // Windows-only open() mode flags (fcntl.h). os.py exposes these off `nt`,
    // and stdlib callers reach for them behind `hasattr(os, 'O_BINARY')`
    // probes (tempfile) that only succeed once the names are bound.
    #[cfg(windows)]
    for (name, val) in [
        ("O_BINARY", libc::O_BINARY as i64),
        ("O_TEXT", libc::O_TEXT as i64),
        ("O_NOINHERIT", libc::O_NOINHERIT as i64),
        ("O_TEMPORARY", libc::O_TEMPORARY as i64),
        ("O_SHORT_LIVED", libc::_O_SHORT_LIVED as i64),
        ("O_RANDOM", libc::O_RANDOM as i64),
        ("O_SEQUENTIAL", libc::O_SEQUENTIAL as i64),
    ] {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
    }
    // Non-critical constants — zero stubs are fine for os.py init.
    for name in [
        "EX_OK",
        "EX_USAGE",
        "EX_DATAERR",
        "EX_NOINPUT",
        "EX_NOUSER",
        "EX_NOHOST",
        "EX_UNAVAILABLE",
        "EX_SOFTWARE",
        "EX_OSERR",
        "EX_OSFILE",
        "EX_CANTCREAT",
        "EX_IOERR",
        "EX_TEMPFAIL",
        "EX_PROTOCOL",
        "EX_NOPERM",
        "EX_CONFIG",
        "WNOHANG",
        "WCONTINUED",
        "WUNTRACED",
        "P_WAIT",
        "P_NOWAIT",
        "P_NOWAITO",
        "ST_RDONLY",
        "ST_NOSUID",
        "SCHED_OTHER",
        "SCHED_FIFO",
        "SCHED_RR",
        "SCHED_BATCH",
        "SCHED_IDLE",
        "RTLD_LAZY",
        "RTLD_NOW",
        "RTLD_GLOBAL",
        "RTLD_LOCAL",
        "RTLD_NODELETE",
        "RTLD_NOLOAD",
        "RTLD_DEEPBIND",
        "PRIO_PROCESS",
        "PRIO_PGRP",
        "PRIO_USER",
    ] {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(0));
    }
    // Remaining noop stubs — functions os.py references at module level.
    // Functions with real implementations are registered individually below.
    for name in [
        "fstatat",
        // Only the POSIX builds reach a real `statvfs` below, and the pair is
        // probed for presence rather than called blind: `os.py` gates
        // `supports_fd` on `_exists`, and `shutil` picks its `disk_usage`
        // implementation on `hasattr(os, 'statvfs')`, falling back to the
        // Windows one. A stub answering those probes with `None` would win the
        // POSIX branch on a host that cannot serve it.
        #[cfg(unix)]
        "statvfs",
        #[cfg(unix)]
        "fstatvfs",
        "dup",
        "dup2",
        "chdir",
        "fchdir",
        "link",
        "symlink",
        "chmod",
        "fchmod",
        "lchmod",
        "fchown",
        "access",
        "faccessat",
        "chflags",
        "lchflags",
        "futimens",
        "futimes",
        "fdopendir",
        "execve",
        "execv",
        "fork",
        "forkpty",
        "wait",
        "waitpid",
        "truncate",
        "ftruncate",
        "pathconf",
        "fpathconf",
        "getppid",
        "setuid",
        "setgid",
        "setsid",
        "setpgid",
        "setreuid",
        "setregid",
        "getgroups",
        "setgroups",
        "getpgrp",
        "setpgrp",
        "getpgid",
        "umask",
        "getlogin",
        "nice",
        "pipe",
        "pipe2",
        "dup3",
        "fsync",
        "fdatasync",
        "mkfifo",
        "mknod",
        "major",
        "minor",
        "makedev",
        "get_inheritable",
        "set_inheritable",
        // "get_terminal_size" — implemented below
        "cpu_count",
        "getloadavg",
        "kill",
        "killpg",
        "getpriority",
        "setpriority",
        "sched_get_priority_max",
        "sched_get_priority_min",
        "sched_getparam",
        "sched_setparam",
        "sched_getscheduler",
        "sched_setscheduler",
        "sched_yield",
        "confstr",
        "confstr_names",
        "sysconf",
        "sysconf_names",
        "setenv",
        // putenv/unsetenv are implemented above unless the host environment is
        // out of reach.
        #[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
        "unsetenv",
        #[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
        "putenv",
        "device_encoding",
        "ttyname",
        "openpty",
        "login_tty",
        "tcgetpgrp",
        "tcsetpgrp",
        "ctermid",
        "get_exec_path",
        "WIFEXITED",
        "WEXITSTATUS",
        "WIFSIGNALED",
        "WTERMSIG",
        "WIFSTOPPED",
        "WSTOPSIG",
        "WEXITED",
        "WNOWAIT",
        "WSTOPPED",
        "waitstatus_to_exitcode",
        "_exit",
        "_cpu_count",
        "abort",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
        "popen",
    ] {
        crate::module_ns_store(
            ns,
            name,
            crate::make_builtin_function(name, |_| Ok(pyre_object::w_none())),
        );
    }
    crate::module_ns_store(
        ns,
        "register_at_fork",
        crate::make_builtin_function("register_at_fork", register_at_fork),
    );

    // PyPy `interp_posix.get_blocking/set_blocking` → rposix
    // `get_blocking/set_blocking`: inspect or update O_NONBLOCK with fcntl.
    fn get_blocking(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() != 1 {
            return Err(crate::PyError::type_error(format!(
                "get_blocking() takes exactly one argument ({} given)",
                args.len()
            )));
        }
        let fd = libc::c_int::try_from(crate::builtins::space_index_w(args[0])?)
            .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            // PyPy passes `eintr_retry=False`; CPython 3.14's
            // `_Py_get_blocking` likewise performs one F_GETFL call.
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            if flags < 0 {
                let error = std::io::Error::last_os_error();
                return Err(errno_err(error.raw_os_error().unwrap_or(libc::EIO), ""));
            }
            return Ok(pyre_object::w_bool_from(flags & libc::O_NONBLOCK == 0));
        }
        #[cfg(any(not(unix), feature = "sandbox"))]
        {
            let _ = fd;
            Err(crate::PyError::not_implemented(
                "get_blocking is unavailable on this target",
            ))
        }
    }

    fn set_blocking(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() != 2 {
            return Err(crate::PyError::type_error(format!(
                "set_blocking() takes exactly two arguments ({} given)",
                args.len()
            )));
        }
        let fd = libc::c_int::try_from(crate::builtins::space_index_w(args[0])?)
            .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
        // CPython 3.14's Argument Clinic declares this parameter `bool`, so
        // it truth-tests arbitrary objects.  This intentionally differs from
        // PyPy's older `@unwrap_spec(blocking=int)` gateway.
        let blocking = crate::baseobjspace::is_true(args[1])?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let mut flags = loop {
                let result = unsafe { libc::fcntl(fd, libc::F_GETFL) };
                if result >= 0 {
                    break result;
                }
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EINTR) {
                    return Err(errno_err(error.raw_os_error().unwrap_or(libc::EIO), ""));
                }
            };
            if blocking {
                flags &= !libc::O_NONBLOCK;
            } else {
                flags |= libc::O_NONBLOCK;
            }
            loop {
                let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags) };
                if result >= 0 {
                    break;
                }
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EINTR) {
                    return Err(errno_err(error.raw_os_error().unwrap_or(libc::EIO), ""));
                }
            }
            return Ok(pyre_object::w_none());
        }
        #[cfg(any(not(unix), feature = "sandbox"))]
        {
            let _ = (fd, blocking);
            Err(crate::PyError::not_implemented(
                "set_blocking is unavailable on this target",
            ))
        }
    }

    crate::module_ns_store(
        ns,
        "get_blocking",
        crate::make_builtin_function("get_blocking", get_blocking),
    );
    crate::module_ns_store(
        ns,
        "set_blocking",
        crate::make_builtin_function("set_blocking", set_blocking),
    );

    // `baseobjspace.py:1970 fsencode_w` returns filesystem bytes; syscall
    // boundaries must not pass through a Rust `String`.
    use crate::gateway::fsencode_bytes_w as extract_path;

    /// The host-API view of OS bytes — a filename, or a half of an environment
    /// entry. Unix spells both in bytes and takes them back unchanged.
    fn os_str_from_bytes(bytes: &[u8]) -> std::borrow::Cow<'_, std::ffi::OsStr> {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            std::borrow::Cow::Borrowed(std::ffi::OsStr::from_bytes(bytes))
        }
        #[cfg(not(unix))]
        {
            // This platform has no byte spelling, so the host API necessarily
            // receives the best text representation of these bytes.
            std::borrow::Cow::Owned(std::ffi::OsString::from(
                String::from_utf8_lossy(bytes).into_owned(),
            ))
        }
    }

    fn path_from_bytes(path: &[u8]) -> std::borrow::Cow<'_, std::path::Path> {
        match os_str_from_bytes(path) {
            std::borrow::Cow::Borrowed(s) => std::borrow::Cow::Borrowed(std::path::Path::new(s)),
            std::borrow::Cow::Owned(s) => std::borrow::Cow::Owned(std::path::PathBuf::from(s)),
        }
    }

    // ── Helper: convert std::io::Error → PyError (OSError) ──
    fn errno_err(errno: i32, path: &str) -> crate::PyError {
        let w_filename = if path.is_empty() {
            pyre_object::PY_NULL
        } else {
            pyre_object::w_str_new(path)
        };
        crate::PyError::os_error_syscall(errno, w_filename)
    }

    // `interp_posix.py:211-219,332` keeps the resolved `Path.w_path` as
    // `OSError.filename`.
    fn errno_err_with_filename(errno: i32, w_path: PyObjectRef) -> crate::PyError {
        crate::PyError::os_error_syscall(errno, w_path)
    }

    fn io_err(e: std::io::Error, path: &str) -> crate::PyError {
        errno_err(crate::builtins::io_error_posix_errno(&e, 0), path)
    }

    fn io_err_with_filename(e: std::io::Error, w_path: PyObjectRef) -> crate::PyError {
        errno_err_with_filename(crate::builtins::io_error_posix_errno(&e, 0), w_path)
    }

    /// A filesystem name reported back to the caller: `bytes` when the path
    /// argument was `bytes` (`posixmodule.c path_converter`), else `str`.
    ///
    /// The name arrives as raw bytes because `readdir`'s `d_name` is handed
    /// back unchanged. Both arms keep it openable: bytes mode returns it
    /// verbatim, and the `str` arm takes the filesystem decoding
    /// (`interp_posix.py:1121 space.newfilename(f)`), so a name like
    /// `b"bad_\xff"` still names the file on disk instead of carrying U+FFFD.
    fn fs_name_obj(bytes_mode: bool, name: &[u8]) -> PyObjectRef {
        if bytes_mode {
            pyre_object::bytesobject::w_bytes_from_bytes(name)
        } else {
            crate::gateway::fsdecode_filename_bytes(name)
        }
    }

    /// interp_posix.py:259-272 `unwrap_fd`.
    ///
    /// ```python
    /// def unwrap_fd(space, w_value, allowed_types='integer'):
    ///     try:
    ///         result = space.c_int_w(w_value)
    ///     except OperationError as e:
    ///         if not e.match(space, space.w_OverflowError):
    ///             raise oefmt(space.w_TypeError,
    ///                 "argument should be %s, not %T", allowed_types, w_value)
    ///         else:
    ///             raise
    ///     if result == -1:
    ///         # -1 is used as sentinel value for not a fd
    ///         raise oefmt(space.w_OSError, "invalid file descriptor: -1")
    ///     return result
    /// ```
    ///
    /// `c_int_w` is the load-bearing part: it converts through `__index__`,
    /// so an `int` subclass — an `IntEnum` member, say — reaches the syscall,
    /// where an exact-type test would reject it and a raw payload read would
    /// interpret the instance's first word as the descriptor.
    fn unwrap_fd(value: PyObjectRef, allowed_types: &str) -> Result<i32, crate::PyError> {
        let result = crate::baseobjspace::c_int_w(value).map_err(|err| {
            if err.kind == crate::PyErrorKind::OverflowError {
                err
            } else {
                crate::PyError::type_error(format!(
                    "argument should be {allowed_types}, not {}",
                    crate::baseobjspace::object_functionstr_type_name(value)
                ))
            }
        })?;
        if result == -1 {
            return Err(crate::PyError::os_error("invalid file descriptor: -1"));
        }
        Ok(result)
    }

    // ── posix.open(path, flags, mode=0o777) → fd ──
    crate::module_ns_store(
        ns,
        "open",
        crate::make_builtin_function("open", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "open() requires at least 2 arguments",
                ));
            }
            let path = crate::gateway::fsencode_path_w(args[0])?;
            let flags = crate::baseobjspace::c_int_w(args[1])? as libc::c_int;
            let mode: u32 = if args.len() >= 3 {
                crate::baseobjspace::c_int_w(args[2])? as u32
            } else {
                0o777
            };
            #[cfg(not(feature = "sandbox"))]
            let fd = {
                // Open the fd non-inheritable (PEP 446) so the descriptor does
                // not leak across exec into child processes: O_CLOEXEC on unix,
                // O_NOINHERIT on Windows (O_CLOEXEC is unix-only in libc). Moot
                // under sandbox, where the controller hands out virtual fds, so
                // it is applied only here.
                #[cfg(unix)]
                let flags = flags | libc::O_CLOEXEC;
                #[cfg(windows)]
                let flags = flags | libc::O_NOINHERIT;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // Opening a FIFO without O_NONBLOCK waits for a peer.
                let (fd, errno) = crate::module::thread::call_external_function(|| unsafe {
                    libc::open(c_path.as_ptr(), flags, mode as libc::c_uint)
                });
                if fd < 0 {
                    return Err(io_err_with_filename(
                        std::io::Error::from_raw_os_error(errno),
                        path.w_path(),
                    ));
                }
                fd
            };
            #[cfg(feature = "sandbox")]
            let fd = crate::host_seam::ops::open(&path.as_bytes, flags, mode)
                .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            Ok(pyre_object::w_int_new(fd as i64))
        }),
    );

    // ── posix.close(fd) ──
    crate::module_ns_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("close() requires 1 argument"));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                #[cfg(not(feature = "sandbox"))]
                {
                    let ret = unsafe { libc::close(fd) };
                    if ret < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                }
                #[cfg(feature = "sandbox")]
                crate::host_seam::ops::close(fd)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.read(fd, n) → bytes ──
    crate::module_ns_store(
        ns,
        "read",
        crate::make_builtin_function_with_arity(
            "read",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("read() requires 2 arguments"));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                let n_signed = crate::baseobjspace::int_w(args[1])?;
                // A negative size would wrap to a huge `usize` (and allocation);
                // os.read rejects it with EINVAL, matching the host read(2).
                if n_signed < 0 {
                    return Err(crate::PyError::os_error_with_errno(
                        libc::EINVAL,
                        "read: negative size",
                    ));
                }
                let n = n_signed as usize;
                #[cfg(not(feature = "sandbox"))]
                let buf = {
                    let mut buf = vec![0u8; n];
                    // interp_posix.py:364-372 `read`: the syscall sits inside
                    // the `eintr_retry=True` loop, so an interrupted read runs
                    // the pending signal handlers and is re-issued rather than
                    // surfacing as `InterruptedError`.  The blocking guard is
                    // scoped to the syscall alone: `checksignals` runs Python.
                    loop {
                        let (ret, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, n as _)
                            });
                        if ret >= 0 {
                            buf.truncate(ret as usize);
                            break buf;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| io_err(e, ""),
                        )?;
                    }
                };
                #[cfg(feature = "sandbox")]
                let buf = crate::host_seam::ops::read(fd, n as i64)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            2,
        ),
    );

    // Python 3.14 `os.readinto`: acquire one writable buffer export for the
    // complete `_Py_read(fd, buffer->buf, buffer->len)` call and return the
    // number of bytes transferred without allocating an intermediate bytes
    // object on the real-host path.
    crate::module_ns_store(
        ns,
        "readinto",
        crate::make_builtin_function_with_arity(
            "readinto",
            |args| {
                let fd_value = crate::baseobjspace::int_w(args[0])?;
                let fd = libc::c_int::try_from(fd_value)
                    .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
                let mut buffer = unsafe { crate::builtins::WritableBuffer::acquire(args[1]) }?;
                let target = unsafe { buffer.as_mut_slice() };
                #[cfg(not(feature = "sandbox"))]
                let result = loop {
                    let result = {
                        let _blocked = crate::module::thread::before_external_block();
                        unsafe {
                            libc::read(
                                fd,
                                target.as_mut_ptr() as *mut libc::c_void,
                                target.len() as _,
                            )
                        }
                    };
                    if result >= 0 {
                        break result as i64;
                    }
                    // The retry used to skip the handlers, so an interrupted
                    // read never let the handler that supplies the remaining
                    // bytes run.  Guard dropped first: `checksignals` runs
                    // Python.
                    crate::builtins::eintr_retry_with(std::io::Error::last_os_error(), |e| {
                        io_err(e, "")
                    })?;
                };
                #[cfg(feature = "sandbox")]
                let result = {
                    let data = crate::host_seam::ops::read(fd, target.len() as i64)
                        .map_err(|error| crate::host_seam::seam_os_err(error, ""))?;
                    let length = data.len().min(target.len());
                    target[..length].copy_from_slice(&data[..length]);
                    length as i64
                };
                Ok(pyre_object::w_int_new(result))
            },
            2,
        ),
    );

    // ── posix.write(fd, data) → nbytes ──
    crate::module_ns_store(
        ns,
        "write",
        crate::make_builtin_function_with_arity(
            "write",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("write() requires 2 arguments"));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                // CPython `os_write_impl` receives a `Py_buffer`: text is not
                // accepted, while every contiguous readable exporter is.
                let data = unsafe { crate::builtins::file_write_buffer_bytes(args[1]) }
                    .map_err(|_| crate::PyError::type_error("write() arg 2 must be bytes-like"))?;
                #[cfg(not(feature = "sandbox"))]
                let ret = {
                    let (ret, errno) = crate::module::thread::call_external_function(|| unsafe {
                        libc::write(fd, data.as_ptr() as *const libc::c_void, data.len() as _)
                    });
                    if ret < 0 {
                        return Err(io_err(std::io::Error::from_raw_os_error(errno), ""));
                    }
                    ret as i64
                };
                #[cfg(feature = "sandbox")]
                let ret = crate::host_seam::ops::write(fd, &data)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_int_new(ret))
            },
            2,
        ),
    );

    // ── posix.lseek(fd, offset, whence) → position ──
    crate::module_ns_store(
        ns,
        "lseek",
        crate::make_builtin_function_with_arity(
            "lseek",
            |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error("lseek() requires 3 arguments"));
                }
                // interp_posix.py:340 `@unwrap_spec(fd=c_int, position=r_longlong,
                // how=c_int)` — the position is a 64-bit offset, not a C int.
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                let offset = crate::baseobjspace::int_w(args[1])? as libc::off_t;
                let whence = crate::baseobjspace::c_int_w(args[2])? as libc::c_int;
                #[cfg(not(feature = "sandbox"))]
                let ret = {
                    let ret = unsafe { libc::lseek(fd, offset, whence) };
                    if ret < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    ret as i64
                };
                #[cfg(feature = "sandbox")]
                let ret = crate::host_seam::ops::lseek(fd, offset as i64, whence)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_int_new(ret))
            },
            3,
        ),
    );

    // ── posix.unlink(path) / posix.remove(path) ──
    fn posix_unlink(
        args: &[pyre_object::PyObjectRef],
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("unlink() requires 1 argument"));
        }
        let path = crate::gateway::fsencode_path_w(args[0])?;
        #[cfg(not(feature = "sandbox"))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            let ret = unsafe { libc::unlink(c_path.as_ptr()) };
            if ret < 0 {
                return Err(io_err_with_filename(
                    std::io::Error::last_os_error(),
                    path.w_path(),
                ));
            }
        }
        #[cfg(feature = "sandbox")]
        crate::host_seam::ops::unlink(&path.as_bytes)
            .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
        Ok(pyre_object::w_none())
    }
    crate::module_ns_store(
        ns,
        "unlink",
        crate::make_builtin_function_with_arity("unlink", posix_unlink, 1),
    );
    crate::module_ns_store(
        ns,
        "remove",
        crate::make_builtin_function_with_arity("remove", posix_unlink, 1),
    );

    // ── posix.readlink(path, *, dir_fd=None) ──
    // Returns the symlink target; a non-symlink raises OSError(EINVAL), which
    // `posixpath.realpath` relies on to stop following links.
    // Under sandbox readlink is unavailable (the controller has no ll_os
    // readlink handler); the stub override loop registers a raising stub, so
    // keep the raw std::fs::read_link body out of the sandbox build.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "readlink",
        crate::make_builtin_function("readlink", |args| {
            let arg = args
                .first()
                .copied()
                .ok_or_else(|| crate::PyError::type_error("readlink() requires 1 argument"))?;
            let path = crate::gateway::fsencode_path_w(arg)?;
            let bytes_mode = unsafe { path.is_bytes() };
            match std::fs::read_link(path_from_bytes(&path.as_bytes).as_ref()) {
                Ok(target) => {
                    let target = target.as_os_str().as_encoded_bytes();
                    Ok(fs_name_obj(bytes_mode, target))
                }
                Err(e) => Err(io_err_with_filename(e, path.w_path())),
            }
        }),
    );

    // ── posix.mkdir(path, mode=0o777) ──
    crate::module_ns_store(
        ns,
        "mkdir",
        crate::make_builtin_function("mkdir", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("mkdir() requires 1 argument"));
            }
            let path = crate::gateway::fsencode_path_w(args[0])?;
            let _mode: u32 = if args.len() >= 2 {
                crate::baseobjspace::c_int_w(args[1])? as u32
            } else {
                0o777
            };
            #[cfg(not(feature = "sandbox"))]
            {
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                #[cfg(unix)]
                let ret = unsafe { libc::mkdir(c_path.as_ptr(), _mode as libc::mode_t) };
                #[cfg(windows)]
                let ret = unsafe { libc::mkdir(c_path.as_ptr()) };
                if ret < 0 {
                    return Err(io_err_with_filename(
                        std::io::Error::last_os_error(),
                        path.w_path(),
                    ));
                }
            }
            #[cfg(feature = "sandbox")]
            crate::host_seam::ops::mkdir(&path.as_bytes, _mode)
                .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            Ok(pyre_object::w_none())
        }),
    );

    // ── posix.rmdir(path) ──
    // Mutates the host filesystem; stubbed under sandbox, so the real body
    // (and its libc call) is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "rmdir",
        crate::make_builtin_function_with_arity(
            "rmdir",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("rmdir() requires 1 argument"));
                }
                let path = crate::gateway::fsencode_path_w(args[0])?;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let ret = unsafe { libc::rmdir(c_path.as_ptr()) };
                if ret < 0 {
                    return Err(io_err_with_filename(
                        std::io::Error::last_os_error(),
                        path.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.rename / posix.replace(src, dst, *, src_dir_fd=None,
    //    dst_dir_fd=None) ──
    // A non-None `src_dir_fd` / `dst_dir_fd` resolves the path relative to the
    // open directory descriptor (`renameat`); the descriptors are only usable
    // where `renameat` exists (unix).
    //
    // The two entry points take the same arguments and differ only in what a
    // pre-existing `dst` does: `replace` overwrites it on every platform,
    // `rename` leaves that to the platform call. On Windows the host layer's
    // `replace` uses MoveFileExW(MOVEFILE_REPLACE_EXISTING); plain `rename`
    // deliberately omits that flag.
    fn rename_impl(
        args: &[PyObjectRef],
        name: &'static str,
    ) -> Result<PyObjectRef, crate::PyError> {
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        if pos.len() < 2 {
            return Err(crate::PyError::type_error(format!(
                "{name}() requires 2 arguments"
            )));
        }
        if pos.len() > 2 {
            return Err(crate::PyError::type_error(format!(
                "{name}() takes exactly 2 positional arguments ({} given)",
                pos.len()
            )));
        }
        crate::builtins::kwarg_reject_unknown(kwargs, &["src_dir_fd", "dst_dir_fd"], name)?;
        let src = crate::gateway::fsencode_path_w(pos[0])?;
        let dst = crate::gateway::fsencode_path_w(pos[1])?;
        let dir_fd = |name: &str| -> Result<Option<i32>, crate::PyError> {
            match crate::builtins::kwarg_get(kwargs, name) {
                // interp_posix.py:274-278 `_unwrap_dirfd` — a non-`None` value
                // goes through `unwrap_fd` with `allowed_types="integer or
                // None"`.
                Some(v) if !unsafe { pyre_object::is_none(v) } => {
                    Ok(Some(unwrap_fd(v, "integer or None")?))
                }
                _ => Ok(None),
            }
        };
        let src_fd = dir_fd("src_dir_fd")?;
        let dst_fd = dir_fd("dst_dir_fd")?;
        #[cfg(unix)]
        let (src_b, dst_b) = {
            use rustpython_host_env::crt_fd::Borrowed;
            (
                src_fd.map(|fd| unsafe { Borrowed::borrow_raw(fd) }),
                dst_fd.map(|fd| unsafe { Borrowed::borrow_raw(fd) }),
            )
        };
        #[cfg(not(unix))]
        let (src_b, dst_b) = {
            if src_fd.is_some() || dst_fd.is_some() {
                return Err(crate::PyError::not_implemented(
                    "dir_fd unavailable on this platform",
                ));
            }
            (None, None)
        };
        let src_path = path_from_bytes(&src.as_bytes);
        let dst_path = path_from_bytes(&dst.as_bytes);
        let result = if name == "replace" {
            host_os::replace(src_path.as_ref(), src_b, dst_path.as_ref(), dst_b)
        } else {
            host_os::rename(src_path.as_ref(), src_b, dst_path.as_ref(), dst_b)
        };
        result.map_err(|e| {
            let errno = crate::builtins::io_error_posix_errno(&e, 0);
            // interp_posix.py:654 hands both resolved `Path.w_path` objects to
            // `wrap_oserror2`.
            crate::PyError::os_error_syscall2(errno, src.w_path(), dst.w_path())
        })?;
        Ok(pyre_object::w_none())
    }
    crate::module_ns_store(
        ns,
        "rename",
        crate::make_builtin_function("rename", |args| rename_impl(args, "rename")),
    );
    crate::module_ns_store(
        ns,
        "replace",
        crate::make_builtin_function("replace", |args| rename_impl(args, "replace")),
    );

    // os.utime(path, times=None, *, ns=None, dir_fd=None, follow_symlinks=True)
    // PyPy `interp_posix.utime` → rposix `utimensat`/`SetFileTime`.  `times` is a
    // `(atime, mtime)` pair in seconds; `ns` the same pair in integer
    // nanoseconds; the two are mutually exclusive.  Both `None` means "now".
    fn utime_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        if pos.is_empty() {
            return Err(crate::PyError::type_error(
                "utime() missing required argument 'path' (pos 1)",
            ));
        }
        if pos.len() > 2 {
            return Err(crate::PyError::type_error(format!(
                "utime() takes from 1 to 2 positional arguments but {} were given",
                pos.len()
            )));
        }
        crate::builtins::kwarg_reject_unknown(
            kwargs,
            &["ns", "dir_fd", "follow_symlinks"],
            "utime",
        )?;
        let path = crate::gateway::fsencode_path_w(pos[0])?;

        let present = |v: PyObjectRef| (!unsafe { pyre_object::is_none(v) }).then_some(v);
        let times = pos.get(1).copied().and_then(present);
        let ns = crate::builtins::kwarg_get(kwargs, "ns").and_then(present);
        let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v)?,
            None => true,
        };
        let dir_fd = match crate::builtins::kwarg_get(kwargs, "dir_fd").and_then(present) {
            // interp_posix.py:1863 types `dir_fd` as `DirFD(...)`, whose
            // `unwrap` is `_unwrap_dirfd` (:274-278).
            Some(v) => Some(unwrap_fd(v, "integer or None")?),
            None => None,
        };

        let unpack_two =
            |obj: PyObjectRef, what: &str| -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
                if !unsafe { pyre_object::is_tuple(obj) }
                    || unsafe { pyre_object::w_tuple_len(obj) } != 2
                {
                    return Err(crate::PyError::type_error(format!(
                        "utime: '{what}' must be a tuple of two ints"
                    )));
                }
                Ok((
                    unsafe { pyre_object::w_tuple_getitem(obj, 0) }.unwrap(),
                    unsafe { pyre_object::w_tuple_getitem(obj, 1) }.unwrap(),
                ))
            };
        let dur_from_secs = |v: PyObjectRef| -> Result<std::time::Duration, crate::PyError> {
            let f = crate::builtins::builtin_float(&[v])?;
            let secs = unsafe { pyre_object::w_float_get_value(f) };
            std::time::Duration::try_from_secs_f64(secs)
                .map_err(|_| crate::PyError::value_error("utime: timestamp out of range"))
        };
        let dur_from_ns = |v: PyObjectRef| -> Result<std::time::Duration, crate::PyError> {
            let n = crate::builtins::space_index_w(v)?;
            if n < 0 {
                return Err(crate::PyError::value_error("utime: timestamp out of range"));
            }
            Ok(std::time::Duration::from_nanos(n as u64))
        };

        let (access, modified) = match (times, ns) {
            (Some(_), Some(_)) => {
                return Err(crate::PyError::value_error(
                    "utime: you may specify either 'times' or 'ns' but not both",
                ));
            }
            (Some(t), None) => {
                let (a, m) = unpack_two(t, "times")?;
                (dur_from_secs(a)?, dur_from_secs(m)?)
            }
            (None, Some(n)) => {
                let (a, m) = unpack_two(n, "ns")?;
                (dur_from_ns(a)?, dur_from_ns(m)?)
            }
            (None, None) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(std::time::Duration::ZERO);
                (now, now)
            }
        };

        #[cfg(all(windows, feature = "host_env"))]
        {
            if dir_fd.is_some() || !follow_symlinks {
                return Err(crate::PyError::not_implemented(
                    "utime: dir_fd and follow_symlinks=False are unavailable on this platform",
                ));
            }
            host_os::set_file_times(path_from_bytes(&path.as_bytes).as_ref(), access, modified)
                .map_err(|e| io_err_with_filename(e, path.w_path()))?;
            return Ok(pyre_object::w_none());
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null character"))?;
            rustpython_host_env::posix::set_file_times_at(
                dir_fd.unwrap_or(libc::AT_FDCWD),
                &c_path,
                access,
                modified,
                follow_symlinks,
            )
            .map_err(|e| io_err_with_filename(e, path.w_path()))?;
            return Ok(pyre_object::w_none());
        }
        #[allow(unreachable_code)]
        {
            let _ = (access, modified, dir_fd, follow_symlinks, &path, pos);
            Err(crate::PyError::not_implemented(
                "utime is unavailable on this platform",
            ))
        }
    }
    crate::module_ns_store(
        ns,
        "utime",
        crate::make_builtin_function("utime", utime_impl),
    );

    // ── posix._path_splitroot(path) → (root, tail) ──
    // Registered only where `sys.platform` is `win32`, the same condition
    // `_bootstrap_external` gates its use on.
    #[cfg(windows)]
    crate::module_ns_store(
        ns,
        "_path_splitroot",
        crate::make_builtin_function_with_arity(
            "_path_splitroot",
            |args| {
                let Some(&arg) = args.first() else {
                    return Err(crate::PyError::type_error(
                        "_path_splitroot() missing required argument 'path'",
                    ));
                };
                let path = extract_path(arg)?;
                // Splitting a drive or UNC prefix is a text operation on a
                // Windows path, and both halves are handed back as `str`, so
                // this one stays in the text domain rather than the byte one.
                let path = String::from_utf8_lossy(&path);
                let (root, tail) = split_root(&path);
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_str_new(root),
                    pyre_object::w_str_new(tail),
                ]))
            },
            1,
        ),
    );

    // ── Windows-only nt calls — moduledef.py `if os.name == 'nt'` block ──
    // ntpath imports these from `nt` behind try/except ImportError, so a build
    // that omits one silently falls back to its pure-Python path implementation.
    #[cfg(all(windows, feature = "host_env"))]
    {
        for (name, func, arity) in [
            (
                "_getfullpathname",
                win_nt::_getfullpathname as crate::gateway::BuiltinCodeFn,
                1u16,
            ),
            ("_getfinalpathname", win_nt::_getfinalpathname, 1),
            ("_getfileinformation", win_nt::_getfileinformation, 1),
            ("_getdiskusage", win_nt::_getdiskusage, 1),
            ("get_handle_inheritable", win_nt::get_handle_inheritable, 1),
            ("set_handle_inheritable", win_nt::set_handle_inheritable, 2),
            ("_add_dll_directory", win_nt::_add_dll_directory, 1),
            ("_remove_dll_directory", win_nt::_remove_dll_directory, 1),
        ] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function_with_arity(name, func, arity),
            );
        }
        crate::module_ns_store(
            ns,
            "_supports_virtual_terminal",
            crate::make_builtin_function(
                "_supports_virtual_terminal",
                win_nt::_supports_virtual_terminal,
            ),
        );
    }

    // ── posix.listdir(path=".") → list of str ──
    crate::module_ns_store(
        ns,
        "listdir",
        crate::make_builtin_function("listdir", |args| {
            // One resolution yields both the path and its bytes-ness, so
            // `__fspath__` runs exactly once.
            let resolved = if args.is_empty() || unsafe { pyre_object::is_none(args[0]) } {
                None
            } else {
                Some(crate::gateway::fsencode_path_w(args[0])?)
            };
            let bytes_mode = unsafe { resolved.as_ref().is_some_and(|path| path.is_bytes()) };
            let path = resolved
                .as_ref()
                .map(|path| path.as_bytes.as_slice())
                .unwrap_or(b".");
            // The omitted argument defaults to `"."` but reports no filename,
            // since there was no path object for the failure to name.
            let w_path = || {
                resolved
                    .as_ref()
                    .map(|path| path.w_path())
                    .unwrap_or(pyre_object::PY_NULL)
            };
            #[cfg(feature = "sandbox")]
            {
                let names = crate::host_seam::ops::listdir(path)
                    .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, w_path()))?;
                let items = names
                    .into_iter()
                    .map(|n| fs_name_obj(bytes_mode, &n))
                    .collect();
                return Ok(pyre_object::w_list_new(items));
            }
            #[cfg(not(feature = "sandbox"))]
            {
                let entries = host_fs::read_dir(path_from_bytes(path).as_ref())
                    .map_err(|e| io_err_with_filename(e, w_path()))?;
                let mut items = Vec::new();
                for entry in entries {
                    let entry = entry.map_err(|e| io_err_with_filename(e, w_path()))?;
                    let name = entry.file_name();
                    items.push(fs_name_obj(bytes_mode, name.as_encoded_bytes()));
                }
                Ok(pyre_object::w_list_new(items))
            }
        }),
    );

    // ── posix.isatty(fd) → bool ──
    crate::module_ns_store(
        ns,
        "isatty",
        crate::make_builtin_function_with_arity(
            "isatty",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])?;
                #[cfg(feature = "sandbox")]
                {
                    return Ok(pyre_object::w_bool_from(
                        crate::host_seam::ops::isatty(fd).unwrap_or(false),
                    ));
                }
                #[cfg(not(feature = "sandbox"))]
                Ok(pyre_object::w_bool_from(host_os::isatty(fd)))
            },
            1,
        ),
    );

    // ── posix.urandom(n) → bytes ──
    crate::module_ns_store(
        ns,
        "urandom",
        crate::make_builtin_function_with_arity(
            "urandom",
            |args| {
                crate::gateway::check_declared_arity("urandom", 1, args.len())?;
                // `__index__` conversion, so a non-integer is a TypeError
                // instead of a raw field read that asks for an arbitrary
                // number of bytes.
                let n = crate::builtins::space_index_w(args[0])?;
                if n < 0 {
                    return Err(crate::PyError::value_error("negative argument not allowed"));
                }
                // A 32-bit target's `usize` is narrower than the `__index__`
                // result, so a size it cannot hold is reported rather than
                // wrapped to a smaller request.
                let n = usize::try_from(n)
                    .map_err(|_| crate::PyError::overflow_error("argument out of range"))?;
                #[cfg(not(feature = "sandbox"))]
                let buf = host_os::urandom(n).unwrap_or_else(|_| vec![0u8; n]);
                // Route host entropy through the trusted controller instead of
                // reaching host getrandom directly.
                #[cfg(feature = "sandbox")]
                let buf = crate::host_seam::ops::urandom(n as i64)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            1,
        ),
    );
    // os.terminal_size — structseq (columns, lines).
    fn make_terminal_size(cols: i64, lines: i64) -> pyre_object::PyObjectRef {
        crate::_structseq::new_instance(
            terminal_size_seq_type(),
            vec![pyre_object::w_int_new(cols), pyre_object::w_int_new(lines)],
        )
    }
    crate::module_ns_store(ns, "terminal_size", terminal_size_seq_type());
    crate::module_ns_store(ns, "statvfs_result", statvfs_result_seq_type());
    crate::module_ns_store(ns, "times_result", times_result_seq_type());
    // `uname_result` names `posix` as its module, which is what pickle imports
    // to resolve it; only the POSIX hosts have that module, and only they
    // register `uname` below.
    #[cfg(unix)]
    crate::module_ns_store(ns, "uname_result", uname_result_seq_type());

    // ── posix.get_terminal_size(fd=1) → os.terminal_size(columns, lines) ──
    // Inspects the controlling terminal via ioctl(TIOCGWINSZ); stubbed under
    // sandbox, so the real body is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "get_terminal_size",
        crate::make_builtin_function("get_terminal_size", |_args| {
            let (cols, rows) = {
                #[cfg(unix)]
                {
                    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                    let ret = unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) };
                    if ret == 0 && ws.ws_col > 0 {
                        (ws.ws_col as i64, ws.ws_row as i64)
                    } else {
                        (80, 24)
                    }
                }
                #[cfg(not(unix))]
                {
                    (80, 24)
                }
            };
            Ok(make_terminal_size(cols, rows))
        }),
    );
    // os.fspath() — posixmodule.c posix_fspath / PyOS_FSPath.  str/bytes
    // pass through unchanged (the protocol's identity case); any other
    // object is resolved via `type(path).__fspath__(path)`.
    crate::module_ns_store(
        ns,
        "fspath",
        crate::make_builtin_function_with_arity(
            "fspath",
            |args| {
                let arg = args.first().copied().unwrap_or(pyre_object::w_none());
                unsafe {
                    if pyre_object::is_str(arg) || pyre_object::bytesobject::is_bytes_like(arg) {
                        return Ok(arg);
                    }
                }
                // `path_type.__fspath__(path)` — the descriptor read off the
                // type is unbound, so `path` is supplied as the sole argument.
                let path_type = crate::typedef::r#type(arg);
                if let Some(pt) = path_type {
                    if let Some(fspath_fn) =
                        unsafe { crate::baseobjspace::lookup_in_type(pt.as_ptr(), "__fspath__") }
                    {
                        return crate::call::call_function_impl_result(fspath_fn, &[arg]);
                    }
                }
                let type_name = match path_type {
                    Some(pt) => unsafe { pyre_object::typeobject::w_type_get_name(pt.as_ptr()) },
                    None => "object",
                };
                Err(crate::PyError::type_error(format!(
                    "expected str, bytes or os.PathLike object, not {type_name}"
                )))
            },
            1,
        ),
    );
    // os.stat / os.lstat / os.fstat — return stat_result structseq.
    // PyPy: posixmodule.c posix_do_stat → build_stat_result.
    //
    // The returned object is a tuple subclass with named attributes
    // (st_mode, st_ino, ...). We expose it as a plain instance with
    // attributes so that both `os.stat(p).st_mode` and
    // `os.stat(p)[0]` work.
    // `st_flags` lives in the BSD/macOS `struct stat` but `std`'s
    // `Metadata`/`MetadataExt` does not surface it, so read it with a raw
    // `stat`/`lstat`/`fstat`; on failure default to 0 (the primary
    // metadata read already succeeded).
    // Under sandbox the stat path is mediated (st_flags arrives over the wire),
    // so this raw-libc helper is compiled out.
    #[cfg(all(target_os = "macos", not(feature = "sandbox")))]
    fn macos_path_st_flags(path: &[u8], follow: bool) -> u32 {
        let Ok(c) = std::ffi::CString::new(path) else {
            return 0;
        };
        unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            let rc = if follow {
                libc::stat(c.as_ptr(), &mut st)
            } else {
                libc::lstat(c.as_ptr(), &mut st)
            };
            if rc == 0 { st.st_flags } else { 0 }
        }
    }
    #[cfg(all(target_os = "macos", not(feature = "sandbox")))]
    fn macos_fd_st_flags(fd: i32) -> u32 {
        unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, &mut st) == 0 {
                st.st_flags
            } else {
                0
            }
        }
    }

    /// `st_flags` (macOS/BSD) is not surfaced by `std::fs::Metadata`, so
    /// the caller obtains it via a raw `stat`/`lstat`/`fstat` and passes
    /// it in; it is ignored (and unread) on platforms whose `struct stat`
    /// lacks the field.
    fn make_stat_result(meta: &std::fs::Metadata, st_flags: u32) -> pyre_object::PyObjectRef {
        // Extract stat fields in a cross-platform way.
        #[cfg(unix)]
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = {
            use std::os::unix::fs::MetadataExt;
            (
                meta.mode() as i64,
                meta.ino() as i64,
                meta.dev() as i64,
                meta.nlink() as i64,
                meta.uid() as i64,
                meta.gid() as i64,
                meta.size() as i64,
                meta.atime(),
                meta.mtime(),
                meta.ctime(),
                meta.atime() * 1_000_000_000 + meta.atime_nsec(),
                meta.mtime() * 1_000_000_000 + meta.mtime_nsec(),
                meta.ctime() * 1_000_000_000 + meta.ctime_nsec(),
            )
        };
        #[cfg(windows)]
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = {
            use std::os::windows::fs::MetadataExt;
            let ft = meta.file_type();
            let attrs = meta.file_attributes();
            let mode: i64 = if ft.is_symlink() {
                // S_IFLNK | 0o777
                0o120777
            } else if ft.is_dir() {
                0o40755
            } else if attrs & 0x1 != 0 {
                // FILE_ATTRIBUTE_READONLY
                0o100444
            } else {
                0o100644
            };
            let size = meta.file_size() as i64;
            // Windows FILETIME is 100-ns intervals since 1601-01-01.
            // Convert to Unix epoch seconds.
            const EPOCH_DIFF: i64 = 11_644_473_600;
            let atime_secs = (meta.last_access_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let mtime_secs = (meta.last_write_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let ctime_secs = (meta.creation_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let atime_ns =
                ((meta.last_access_time() as i64 % 10_000_000) * 100) + atime_secs * 1_000_000_000;
            let mtime_ns =
                ((meta.last_write_time() as i64 % 10_000_000) * 100) + mtime_secs * 1_000_000_000;
            let ctime_ns =
                ((meta.creation_time() as i64 % 10_000_000) * 100) + ctime_secs * 1_000_000_000;
            (
                mode, 0i64, // st_ino — not available on Windows
                0i64, // st_dev
                1i64, // nlink — not easily available on stable Windows
                0i64, // st_uid
                0i64, // st_gid
                size, atime_secs, mtime_secs, ctime_secs, atime_ns, mtime_ns, ctime_ns,
            )
        };

        #[cfg(unix)]
        let (st_blksize, st_blocks, st_rdev) = {
            use std::os::unix::fs::MetadataExt;
            (
                meta.blksize() as i64,
                meta.blocks() as i64,
                meta.rdev() as i64,
            )
        };

        stat_result_from_fields(
            &StatFields {
                mode: st_mode,
                ino: st_ino,
                dev: st_dev,
                nlink: st_nlink,
                uid: st_uid,
                gid: st_gid,
                size: st_size,
                atime: st_atime,
                mtime: st_mtime,
                ctime: st_ctime,
                atime_ns: st_atime_ns,
                mtime_ns: st_mtime_ns,
                ctime_ns: st_ctime_ns,
                #[cfg(unix)]
                blksize: st_blksize,
                #[cfg(unix)]
                blocks: st_blocks,
                #[cfg(unix)]
                rdev: st_rdev,
            },
            st_flags,
        )
    }

    /// The `stat_result` fields, read out of whichever source produced them:
    /// `std::fs::Metadata` for the path and descriptor forms, `libc::stat`
    /// for the `fstatat` form a `dir_fd`-relative name takes.
    struct StatFields {
        mode: i64,
        ino: i64,
        dev: i64,
        nlink: i64,
        uid: i64,
        gid: i64,
        size: i64,
        atime: i64,
        mtime: i64,
        ctime: i64,
        atime_ns: i64,
        mtime_ns: i64,
        ctime_ns: i64,
        #[cfg(unix)]
        blksize: i64,
        #[cfg(unix)]
        blocks: i64,
        #[cfg(unix)]
        rdev: i64,
    }

    fn stat_result_from_fields(f: &StatFields, st_flags: u32) -> pyre_object::PyObjectRef {
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = (
            f.mode,
            f.ino,
            f.dev,
            f.nlink,
            f.uid,
            f.gid,
            f.size,
            f.atime,
            f.mtime,
            f.ctime,
            f.atime_ns,
            f.mtime_ns,
            f.ctime_ns,
        );
        #[cfg(unix)]
        let (st_blksize, st_blocks, st_rdev) = (f.blksize, f.blocks, f.rdev);
        // The 10 sequence slots are the integer fields (integer-seconds
        // times at 7..10, named `_integer_*`); the float times, `st_*_ns`,
        // and the platform block/device extras are named-only fields.
        let seq = vec![
            pyre_object::w_int_new(st_mode),
            pyre_object::w_int_new(st_ino),
            pyre_object::w_int_new(st_dev),
            pyre_object::w_int_new(st_nlink),
            pyre_object::w_int_new(st_uid),
            pyre_object::w_int_new(st_gid),
            pyre_object::w_int_new(st_size),
            pyre_object::w_int_new(st_atime),
            pyre_object::w_int_new(st_mtime),
            pyre_object::w_int_new(st_ctime),
        ];
        // `_ll_get_st_atime` — float times keep sub-second precision:
        // `float(seconds) + 1e-9 * nanosecond_fraction`, where the
        // fraction is recovered from the full-nanosecond field.
        let st_atime_f = st_atime as f64 + 1e-9 * (st_atime_ns - st_atime * 1_000_000_000) as f64;
        let st_mtime_f = st_mtime as f64 + 1e-9 * (st_mtime_ns - st_mtime * 1_000_000_000) as f64;
        let st_ctime_f = st_ctime as f64 + 1e-9 * (st_ctime_ns - st_ctime * 1_000_000_000) as f64;
        #[allow(unused_mut)]
        let mut extras = vec![
            ("st_atime", pyre_object::w_float_new(st_atime_f)),
            ("st_mtime", pyre_object::w_float_new(st_mtime_f)),
            ("st_ctime", pyre_object::w_float_new(st_ctime_f)),
            ("st_atime_ns", pyre_object::w_int_new(st_atime_ns)),
            ("st_mtime_ns", pyre_object::w_int_new(st_mtime_ns)),
            ("st_ctime_ns", pyre_object::w_int_new(st_ctime_ns)),
            // `build_stat_result` (interp_posix.py:554-557): the
            // sub-second remainder of each full-nanosecond timestamp,
            // `value % 1_000_000_000` (non-negative for pre-1970 times).
            (
                "nsec_atime",
                pyre_object::w_int_new(st_atime_ns.rem_euclid(1_000_000_000)),
            ),
            (
                "nsec_mtime",
                pyre_object::w_int_new(st_mtime_ns.rem_euclid(1_000_000_000)),
            ),
            (
                "nsec_ctime",
                pyre_object::w_int_new(st_ctime_ns.rem_euclid(1_000_000_000)),
            ),
        ];
        #[cfg(unix)]
        {
            extras.push(("st_blksize", pyre_object::w_int_new(st_blksize)));
            extras.push(("st_blocks", pyre_object::w_int_new(st_blocks)));
            extras.push(("st_rdev", pyre_object::w_int_new(st_rdev)));
        }
        #[cfg(target_os = "macos")]
        extras.push(("st_flags", pyre_object::w_int_new(st_flags as i64)));
        #[cfg(not(target_os = "macos"))]
        let _ = st_flags;
        crate::_structseq::new_instance_with_extra(stat_result_seq_type(), seq, extras)
    }
    /// Build a `stat_result` from the sandbox wire `StatBuf` (sandbox build
    /// only, hence unix-only): the controller delivers the 10 protocol fields
    /// plus integer atime/mtime/ctime; the sub-second and block/device extras
    /// are whatever `StatBuf` carries (zero over the wire). Mirrors the unix
    /// slot/extra layout of `make_stat_result`.
    #[cfg(feature = "sandbox")]
    fn make_stat_result_from_statbuf(st: &crate::host_seam::StatBuf) -> pyre_object::PyObjectRef {
        let st_atime = st.atime;
        let st_mtime = st.mtime;
        let st_ctime = st.ctime;
        let st_atime_ns = st.atime * 1_000_000_000 + st.atime_nsec;
        let st_mtime_ns = st.mtime * 1_000_000_000 + st.mtime_nsec;
        let st_ctime_ns = st.ctime * 1_000_000_000 + st.ctime_nsec;
        let seq = vec![
            pyre_object::w_int_new(st.mode as i64),
            pyre_object::w_int_new(st.ino as i64),
            pyre_object::w_int_new(st.dev as i64),
            pyre_object::w_int_new(st.nlink as i64),
            pyre_object::w_int_new(st.uid as i64),
            pyre_object::w_int_new(st.gid as i64),
            pyre_object::w_int_new(st.size as i64),
            pyre_object::w_int_new(st_atime),
            pyre_object::w_int_new(st_mtime),
            pyre_object::w_int_new(st_ctime),
        ];
        let st_atime_f = st_atime as f64 + 1e-9 * (st_atime_ns - st_atime * 1_000_000_000) as f64;
        let st_mtime_f = st_mtime as f64 + 1e-9 * (st_mtime_ns - st_mtime * 1_000_000_000) as f64;
        let st_ctime_f = st_ctime as f64 + 1e-9 * (st_ctime_ns - st_ctime * 1_000_000_000) as f64;
        #[allow(unused_mut)]
        let mut extras = vec![
            ("st_atime", pyre_object::w_float_new(st_atime_f)),
            ("st_mtime", pyre_object::w_float_new(st_mtime_f)),
            ("st_ctime", pyre_object::w_float_new(st_ctime_f)),
            ("st_atime_ns", pyre_object::w_int_new(st_atime_ns)),
            ("st_mtime_ns", pyre_object::w_int_new(st_mtime_ns)),
            ("st_ctime_ns", pyre_object::w_int_new(st_ctime_ns)),
            (
                "nsec_atime",
                pyre_object::w_int_new(st_atime_ns.rem_euclid(1_000_000_000)),
            ),
            (
                "nsec_mtime",
                pyre_object::w_int_new(st_mtime_ns.rem_euclid(1_000_000_000)),
            ),
            (
                "nsec_ctime",
                pyre_object::w_int_new(st_ctime_ns.rem_euclid(1_000_000_000)),
            ),
            ("st_blksize", pyre_object::w_int_new(st.blksize as i64)),
            ("st_blocks", pyre_object::w_int_new(st.blocks as i64)),
            ("st_rdev", pyre_object::w_int_new(st.rdev as i64)),
        ];
        #[cfg(target_os = "macos")]
        extras.push(("st_flags", pyre_object::w_int_new(st.st_flags as i64)));
        crate::_structseq::new_instance_with_extra(stat_result_seq_type(), seq, extras)
    }
    /// `os.stat(path, *, dir_fd=None, follow_symlinks=True)` /
    /// `os.lstat(path, *, dir_fd=None)` — `follow_symlinks` is keyword-only,
    /// so `stat` cannot take the fixed-arity carrier that rejects keywords.
    /// The three argument forms are the three arms of `do_stat`
    /// (`interp_posix.py:633-649`): an open descriptor as `path` goes to
    /// `fstat`, a `dir_fd`-relative name to `fstatat`, and a bare name to
    /// `stat`/`lstat`.
    fn stat_entry(
        args: &[pyre_object::PyObjectRef],
        default_follow: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
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
        let path = match crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", name, 1)? {
            Some(path) => path,
            None => {
                return Err(crate::PyError::type_error(format!(
                    "{name}() missing required argument 'path' (pos 1)"
                )));
            }
        };
        // `stat`/`lstat` type `dir_fd` as `DirFD(rposix.HAVE_FSTATAT)`
        // (`interp_posix.py:612,660`), whose `unwrap` is `_unwrap_dirfd`
        // (:274-278).
        let dir_fd = match crate::builtins::kwarg_get(kwargs, "dir_fd")
            .filter(|&v| !unsafe { pyre_object::is_none(v) })
        {
            Some(v) => Some(unwrap_fd(v, "integer or None")?),
            None => None,
        };
        let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v)?,
            None => default_follow,
        };
        // interp_posix.py:611,659 — `stat` takes `path_or_fd(allow_fd=True)`
        // and `lstat` takes `allow_fd=False`, which is also what makes their
        // type errors name different allowed types.
        let path = crate::gateway::fsencode_path_or_fd_w(path, name, default_follow)?;
        // interp_posix.py:634-644 `do_stat` tests the descriptor first: with one
        // in hand neither other argument has anything to apply to, and both
        // rejections precede the platform's dir_fd availability.
        if path.as_fd != -1 {
            if dir_fd.is_some() {
                return Err(crate::PyError::value_error(format!(
                    "{name}: can't specify dir_fd without matching path"
                )));
            }
            if !follow_symlinks {
                return Err(crate::PyError::value_error(format!(
                    "{name}: cannot use fd and follow_symlinks together"
                )));
            }
            return fstat_fd(path.as_fd);
        }
        match dir_fd {
            Some(dir_fd) => stat_at(name, &path, dir_fd, follow_symlinks),
            None => stat_path(&path, follow_symlinks),
        }
    }

    /// `rposix_stat.build_stat_result` reads the same fields off the raw
    /// `struct stat` the `*at` calls fill in.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn stat_fields_from_libc(st: &libc::stat) -> StatFields {
        StatFields {
            mode: st.st_mode as i64,
            ino: st.st_ino as i64,
            dev: st.st_dev as i64,
            nlink: st.st_nlink as i64,
            uid: st.st_uid as i64,
            gid: st.st_gid as i64,
            size: st.st_size as i64,
            atime: st.st_atime as i64,
            mtime: st.st_mtime as i64,
            ctime: st.st_ctime as i64,
            atime_ns: st.st_atime as i64 * 1_000_000_000 + st.st_atime_nsec as i64,
            mtime_ns: st.st_mtime as i64 * 1_000_000_000 + st.st_mtime_nsec as i64,
            ctime_ns: st.st_ctime as i64 * 1_000_000_000 + st.st_ctime_nsec as i64,
            blksize: st.st_blksize as i64,
            blocks: st.st_blocks as i64,
            rdev: st.st_rdev as i64,
        }
    }

    /// `do_stat` (`interp_posix.py:649`) resolves a name against an open
    /// directory descriptor with `fstatat`, where `AT_SYMLINK_NOFOLLOW`
    /// carries `follow_symlinks=False`. An absolute name ignores `dir_fd`,
    /// which is why the caller does not have to test for one.
    fn stat_at(
        name: &str,
        path: &crate::gateway::FsEncodedPath,
        dir_fd: i32,
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null character"))?;
            let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
            let flags = if follow_symlinks {
                0
            } else {
                libc::AT_SYMLINK_NOFOLLOW
            };
            let ret = unsafe { libc::fstatat(dir_fd, c_path.as_ptr(), st.as_mut_ptr(), flags) };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                return Err(errno_err_with_filename(
                    crate::builtins::io_error_posix_errno(&err, libc::EBADF),
                    path.w_path(),
                ));
            }
            let st = unsafe { st.assume_init() };
            #[cfg(target_os = "macos")]
            let st_flags = st.st_flags;
            #[cfg(not(target_os = "macos"))]
            let st_flags = 0u32;
            return Ok(stat_result_from_fields(
                &stat_fields_from_libc(&st),
                st_flags,
            ));
        }
        // `DirFD(available=False)` (`interp_posix.py:285-292`): the platform
        // has no `fstatat`, so a `dir_fd` that reached this far has nothing
        // to resolve against.
        #[allow(unreachable_code)]
        {
            let _ = (path, dir_fd, follow_symlinks);
            Err(crate::PyError::not_implemented(format!(
                "{name}: dir_fd unavailable on this platform"
            )))
        }
    }

    fn stat_path(
        path: &crate::gateway::FsEncodedPath,
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        #[cfg(feature = "sandbox")]
        {
            let buf = if follow_symlinks {
                crate::host_seam::ops::stat(&path.as_bytes)
            } else {
                crate::host_seam::ops::lstat(&path.as_bytes)
            }
            .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            return Ok(make_stat_result_from_statbuf(&buf));
        }
        #[cfg(not(feature = "sandbox"))]
        {
            let meta = if follow_symlinks {
                host_fs::metadata(path_from_bytes(&path.as_bytes).as_ref())
            } else {
                host_fs::symlink_metadata(path_from_bytes(&path.as_bytes).as_ref())
            };
            match meta {
                Ok(m) => {
                    #[cfg(target_os = "macos")]
                    let st_flags = macos_path_st_flags(&path.as_bytes, follow_symlinks);
                    #[cfg(not(target_os = "macos"))]
                    let st_flags = 0u32;
                    Ok(make_stat_result(&m, st_flags))
                }
                Err(e) => Err(errno_err_with_filename(
                    crate::builtins::io_error_posix_errno(&e, 2),
                    path.w_path(),
                )),
            }
        }
    }

    // ── posix.scandir(path=".") → ScandirIterator of DirEntry ──
    // `posix_scandir` / `posixmodule.c` DirEntry + ScandirIterator. The
    // entries are read eagerly into a list backing a context-manager
    // iterator so `with os.scandir(p) as it:` and `for e in it:` both work.
    //
    // DirEntry holds `name`/`path` as instance attributes; the type carries
    // is_dir/is_file/is_symlink/is_junction/stat/inode/__fspath__ which stat
    // the stored path on demand.
    type PyObjectRef = pyre_object::PyObjectRef;

    fn dir_entry_path(self_obj: PyObjectRef) -> Result<(PyObjectRef, Vec<u8>), crate::PyError> {
        let p = crate::baseobjspace::getattr_str(self_obj, "path")?;
        let path = extract_path(p)?;
        Ok((p, path))
    }
    fn dir_entry_follow(args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
        let (_pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v),
            None => Ok(true),
        }
    }
    fn dir_entry_is_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (_, path) = dir_entry_path(args[0])?;
        let follow = dir_entry_follow(args)?;
        let meta = if follow {
            host_fs::metadata(path_from_bytes(&path).as_ref())
        } else {
            host_fs::symlink_metadata(path_from_bytes(&path).as_ref())
        };
        Ok(pyre_object::w_bool_from(
            meta.map(|m| m.is_dir()).unwrap_or(false),
        ))
    }
    fn dir_entry_is_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (_, path) = dir_entry_path(args[0])?;
        let follow = dir_entry_follow(args)?;
        let meta = if follow {
            host_fs::metadata(path_from_bytes(&path).as_ref())
        } else {
            host_fs::symlink_metadata(path_from_bytes(&path).as_ref())
        };
        Ok(pyre_object::w_bool_from(
            meta.map(|m| m.is_file()).unwrap_or(false),
        ))
    }
    fn dir_entry_is_symlink(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (_, path) = dir_entry_path(args[0])?;
        Ok(pyre_object::w_bool_from(
            host_fs::symlink_metadata(path_from_bytes(&path).as_ref())
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false),
        ))
    }
    fn dir_entry_is_junction(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // POSIX has no junction points.
        Ok(pyre_object::w_bool_from(false))
    }
    fn dir_entry_inode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (w_path, path) = dir_entry_path(args[0])?;
        let meta = host_fs::symlink_metadata(path_from_bytes(&path).as_ref())
            .map_err(|e| io_err_with_filename(e, w_path))?;
        #[cfg(unix)]
        let ino = {
            use std::os::unix::fs::MetadataExt;
            meta.ino() as i64
        };
        #[cfg(not(unix))]
        let ino = {
            let _ = &meta;
            0i64
        };
        Ok(pyre_object::w_int_new(ino))
    }
    fn dir_entry_stat(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (w_path, path) = dir_entry_path(args[0])?;
        let follow = dir_entry_follow(args)?;
        let meta = if follow {
            host_fs::metadata(path_from_bytes(&path).as_ref())
        } else {
            host_fs::symlink_metadata(path_from_bytes(&path).as_ref())
        };
        match meta {
            Ok(m) => {
                #[cfg(all(target_os = "macos", not(feature = "sandbox")))]
                let st_flags = macos_path_st_flags(&path, follow);
                #[cfg(not(all(target_os = "macos", not(feature = "sandbox"))))]
                let st_flags = 0u32;
                Ok(make_stat_result(&m, st_flags))
            }
            Err(e) => Err(io_err_with_filename(e, w_path)),
        }
    }
    fn dir_entry_fspath(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        crate::baseobjspace::getattr_str(args[0], "path")
    }
    /// `posixmodule.c DirEntry_repr` — `"<DirEntry %R>"`, so the name is
    /// rendered by its own `repr`.  That keeps `os.scandir(b'.')`'s bytes
    /// name spelled `b'…'` and lets a name whose bytes have no UTF-8 form
    /// keep the lone surrogate `fs_name_obj` decoded it to.
    fn dir_entry_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let name = crate::baseobjspace::getattr_str(args[0], "name")?;
        let mut out = rustpython_wtf8::Wtf8Buf::new();
        out.push_str("<DirEntry ");
        out.push_wtf8(&unsafe { crate::py_repr_wtf8(name)? });
        out.push_str(">");
        Ok(pyre_object::w_str_from_wtf8(out))
    }
    /// `interp_scandir.py:463-465 descr_reduce_ex` — an entry names a live
    /// position in a directory listing, so it refuses to be pickled.  `%T` is
    /// `error.py:592-593 space.type(value).name`, the qualified typedef name
    /// (`interp_scandir.py:469 'posix.DirEntry'`), which is what
    /// `w_type_get_name` returns; the flag-driven `reduce_newobj` refusal in
    /// `reduce_protocol_app.py:13-14` spells it with the bare `__name__`.
    fn dir_entry_reduce_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let type_name = match crate::typedef::r#type(args[0]) {
            Some(tp) => unsafe { pyre_object::typeobject::w_type_get_name(tp.as_ptr()) },
            None => "posix.DirEntry",
        };
        Err(crate::PyError::type_error(format!(
            "cannot pickle '{type_name}' object"
        )))
    }
    fn dir_entry_type() -> PyObjectRef {
        static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CELL.get_or_init(|| {
            // `interp_scandir.py:469` names the typedef `'posix.DirEntry'`.
            // typedef.rs:2285-2291 turns the leading component of a qualified
            // builtin name into a `__module__` entry, so `type(e).__module__`
            // reports `posix` and every type-name-bearing error message is
            // spelled the way the typedef spells it.
            let tp = crate::typedef::make_builtin_type("posix.DirEntry", |ns| {
                for (name, f) in [
                    ("is_dir", dir_entry_is_dir as crate::gateway::BuiltinCodeFn),
                    ("is_file", dir_entry_is_file),
                    ("is_symlink", dir_entry_is_symlink),
                    ("is_junction", dir_entry_is_junction),
                    ("inode", dir_entry_inode),
                    ("stat", dir_entry_stat),
                    ("__fspath__", dir_entry_fspath),
                    ("__repr__", dir_entry_repr),
                    ("__reduce_ex__", dir_entry_reduce_ex),
                ] {
                    unsafe {
                        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                            ns,
                            name,
                            crate::make_builtin_function(name, f),
                        )
                    };
                }
                // CPython 3.14 Modules/posixmodule.c DirEntry_methods —
                // Py_GenericAlias with METH_CLASS.
                unsafe {
                    pyre_object::w_dict_setitem_str(
                        ns,
                        "__class_getitem__",
                        pyre_object::function::w_classmethod_new(crate::make_builtin_function(
                            "__class_getitem__",
                            crate::_pypy_generic_alias::generic_alias_class_getitem,
                        )),
                    )
                };
            });
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            // `interp_scandir.py:468-487` declares no `__new__` on the typedef
            // and `:487` sets `acceptable_as_base_class = False`; `typedef.py:55
            // acceptable_as_base_class = '__new__' in rawdict` is the rule, and
            // `typedef.py:754 assert not PyFrame.typedef.acceptable_as_base_class
            // # no __new__` is the same shape typedef.rs:534-539 already ports.
            // `scandir_fn` below allocates entries with
            // `pyre_object::w_instance_new`, which never enters `type.__call__`,
            // so the producer is untouched by the instantiation gate.
            unsafe {
                pyre_object::typeobject::w_type_set_disallow_instantiation(tp);
                pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp as usize
        }) as PyObjectRef
    }

    fn scandir_iter_self(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(args[0])
    }
    fn scandir_iter_close(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(pyre_object::w_none())
    }
    fn scandir_iter_next(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = args[0];
        // The type carries an instance dict, so `_index` is writable from
        // Python and cannot be assumed to still hold the int this iterator
        // stored.
        let idx = crate::baseobjspace::int_w(crate::baseobjspace::getattr_str(self_obj, "_index")?)?;
        let entries = crate::baseobjspace::getattr_str(self_obj, "_entries")?;
        let len = unsafe { pyre_object::w_list_len(entries) } as i64;
        if idx >= len {
            return Err(crate::PyError::stop_iteration());
        }
        let item = unsafe { pyre_object::w_list_getitem(entries, idx) }
            .ok_or_else(crate::PyError::stop_iteration)?;
        let _ =
            crate::baseobjspace::setattr_str(self_obj, "_index", pyre_object::w_int_new(idx + 1));
        Ok(item)
    }
    fn scandir_iter_type() -> PyObjectRef {
        static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CELL.get_or_init(|| {
            // `interp_scandir.py:173` names the typedef `'posix.ScandirIterator'`.
            let tp = crate::typedef::make_builtin_type("posix.ScandirIterator", |ns| {
                for (name, f) in [
                    (
                        "__iter__",
                        scandir_iter_self as crate::gateway::BuiltinCodeFn,
                    ),
                    ("__next__", scandir_iter_next),
                    ("__enter__", scandir_iter_self),
                    ("__exit__", scandir_iter_close),
                    ("close", scandir_iter_close),
                ] {
                    unsafe {
                        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                            ns,
                            name,
                            crate::make_builtin_function(name, f),
                        )
                    };
                }
            });
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            // `interp_scandir.py:172-180` declares no `__new__` on the typedef
            // and `:180` sets `acceptable_as_base_class = False`.  The iterator
            // is produced only by `scandir_fn` below, through
            // `pyre_object::w_instance_new`.
            unsafe {
                pyre_object::typeobject::w_type_set_disallow_instantiation(tp);
                pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp as usize
        }) as PyObjectRef
    }

    fn scandir_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // One resolution yields both the path and its bytes-ness, so
        // `__fspath__` runs exactly once.
        let resolved = if args.is_empty() || unsafe { pyre_object::is_none(args[0]) } {
            None
        } else {
            Some(crate::gateway::fsencode_path_w(args[0])?)
        };
        let bytes_mode = unsafe { resolved.as_ref().is_some_and(|path| path.is_bytes()) };
        let path = resolved
            .as_ref()
            .map(|path| path.as_bytes.as_slice())
            .unwrap_or(b".");
        // The omitted argument defaults to `"."` but reports no filename,
        // since there was no path object for the failure to name.
        let w_path = || {
            resolved
                .as_ref()
                .map(|path| path.w_path())
                .unwrap_or(pyre_object::PY_NULL)
        };
        let entries = host_fs::read_dir(path_from_bytes(path).as_ref())
            .map_err(|e| io_err_with_filename(e, w_path()))?;
        let list = pyre_object::w_list_new(Vec::new());
        for entry in entries {
            let entry = entry.map_err(|e| io_err_with_filename(e, w_path()))?;
            let name = entry.file_name();
            let full = entry.path().into_os_string();
            let de = pyre_object::w_instance_new(dir_entry_type());
            let _ = crate::baseobjspace::setattr_str(
                de,
                "name",
                fs_name_obj(bytes_mode, name.as_encoded_bytes()),
            );
            let _ = crate::baseobjspace::setattr_str(
                de,
                "path",
                fs_name_obj(bytes_mode, full.as_encoded_bytes()),
            );
            unsafe { pyre_object::w_list_append(list, de) };
        }
        let it = pyre_object::w_instance_new(scandir_iter_type());
        let _ = crate::baseobjspace::setattr_str(it, "_entries", list);
        let _ = crate::baseobjspace::setattr_str(it, "_index", pyre_object::w_int_new(0));
        Ok(it)
    }
    crate::module_ns_store(
        ns,
        "scandir",
        crate::make_builtin_function("scandir", scandir_fn),
    );
    crate::module_ns_store(ns, "DirEntry", dir_entry_type());

    // os.uname() — returns structseq (sysname, nodename, release, version, machine).
    // Routed through `host_env::posix::uname_info` when available so the
    // result reports the host's real POSIX strings ("Darwin", "Linux",
    // node hostname, kernel release, etc.) instead of Rust's compile-time
    // `std::env::consts::OS` ("macos"/"linux"/...).
    //
    // POSIX only, the way `HAVE_UNAME` gates it. Its callers read presence as
    // "the POSIX identification is available": `platform.uname` falls back to
    // `sys.platform` on AttributeError, and `sysconfig.get_platform` tests
    // `hasattr(os, 'uname')` directly. A build with no host strings to report
    // would answer both with the compile-time constants instead.
    #[cfg(unix)]
    crate::module_ns_store(
        ns,
        "uname",
        crate::make_builtin_function_with_arity(
            "uname",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                let (sysname, nodename, release, version, machine) = {
                    let info = rustpython_host_env::posix::uname_info().unwrap_or(
                        rustpython_host_env::posix::UnameInfo {
                            sysname: String::new(),
                            nodename: String::new(),
                            release: String::new(),
                            version: String::new(),
                            machine: String::new(),
                        },
                    );
                    (
                        info.sysname,
                        info.nodename,
                        info.release,
                        info.version,
                        info.machine,
                    )
                };
                #[cfg(not(all(unix, feature = "host_env")))]
                let (sysname, nodename, release, version, machine) = (
                    std::env::consts::OS.to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    std::env::consts::ARCH.to_string(),
                );
                Ok(crate::_structseq::new_instance(
                    uname_result_seq_type(),
                    vec![
                        pyre_object::w_str_new(&sysname),
                        pyre_object::w_str_new(&nodename),
                        pyre_object::w_str_new(&release),
                        pyre_object::w_str_new(&version),
                        pyre_object::w_str_new(&machine),
                    ],
                ))
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "stat",
        crate::gateway::make_builtin_function_with_text_signature(
            "stat",
            |args| stat_entry(args, true),
            "(path, *, dir_fd=None, follow_symlinks=True)",
        ),
    );
    crate::module_ns_store(
        ns,
        "lstat",
        crate::make_builtin_function("lstat", |args| stat_entry(args, false)),
    );
    /// `rposix_stat.py fstat`: the descriptor form both `os.fstat` and
    /// `os.stat` with a descriptor answer through, so the two cannot drift.
    fn fstat_fd(fd: i32) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        // `rposix_stat.py:fstat` passes the descriptor to libc, where
        // `-1` reports EBADF.  Rust's `OwnedFd::from_raw_fd(-1)`
        // asserts before `File::metadata` can produce that error.
        if fd == -1 {
            return Err(crate::PyError::os_error_with_errno(
                libc::EBADF,
                std::io::Error::from_raw_os_error(libc::EBADF).to_string(),
            ));
        }
        #[cfg(feature = "sandbox")]
        {
            let buf = crate::host_seam::ops::fstat(fd)
                .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
            Ok(make_stat_result_from_statbuf(&buf))
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            use std::os::unix::io::FromRawFd;
            let f = unsafe { std::fs::File::from_raw_fd(fd) };
            let meta = f.metadata();
            let _ = std::mem::ManuallyDrop::new(f); // don't close
            match meta {
                Ok(m) => {
                    #[cfg(target_os = "macos")]
                    let st_flags = macos_fd_st_flags(fd);
                    #[cfg(not(target_os = "macos"))]
                    let st_flags = 0u32;
                    Ok(make_stat_result(&m, st_flags))
                }
                Err(e) => Err(crate::PyError::os_error_with_errno(
                    crate::builtins::io_error_posix_errno(&e, 9),
                    format!("{}", e),
                )),
            }
        }
        #[cfg(not(any(unix, feature = "sandbox")))]
        Err(crate::PyError::os_error_with_errno(
            9,
            "fstat unsupported".to_string(),
        ))
    }

    crate::module_ns_store(
        ns,
        "fstat",
        crate::make_builtin_function_with_arity(
            "fstat",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("fstat() missing argument"));
                }
                fstat_fd(crate::baseobjspace::c_int_w(args[0])?)
            },
            1,
        ),
    );
    // stat_result type — structseq (tuple subclass). Exported so that
    // `posix.stat_result` and `isinstance(os.stat(p), os.stat_result)` work.
    crate::module_ns_store(ns, "stat_result", stat_result_seq_type());
    // os.getcwd() — PyPy: posixmodule.c posix_getcwd.
    crate::module_ns_store(
        ns,
        "getcwd",
        crate::make_builtin_function_with_arity(
            "getcwd",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let cwd = crate::host_seam::ops::getcwd()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    Ok(pyre_object::w_str_new(&String::from_utf8_lossy(&cwd)))
                }
                #[cfg(not(feature = "sandbox"))]
                {
                    #[cfg(feature = "host_env")]
                    {
                        if let Ok(cwd) = host_os::current_dir() {
                            return Ok(pyre_object::w_str_new(&cwd.to_string_lossy()));
                        }
                    }
                    Ok(pyre_object::w_str_new(""))
                }
            },
            0,
        ),
    );
    // os.getcwdb() — bytes form of getcwd.
    crate::module_ns_store(
        ns,
        "getcwdb",
        crate::make_builtin_function_with_arity(
            "getcwdb",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let cwd = crate::host_seam::ops::getcwd()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    Ok(pyre_object::w_bytes_from_bytes(&cwd))
                }
                #[cfg(not(feature = "sandbox"))]
                {
                    #[cfg(feature = "host_env")]
                    {
                        if let Ok(cwd) = host_os::current_dir() {
                            return Ok(pyre_object::w_bytes_from_bytes(
                                cwd.as_os_str().as_encoded_bytes(),
                            ));
                        }
                    }
                    Ok(pyre_object::w_bytes_from_bytes(b""))
                }
            },
            0,
        ),
    );
    // os.getuid / geteuid / getgid / getegid — real syscalls (the sandbox
    // build routes these through the controller instead, see below).
    #[cfg(all(unix, not(feature = "sandbox")))]
    unsafe extern "C" {
        fn getuid() -> u32;
        fn geteuid() -> u32;
        fn getgid() -> u32;
        fn getegid() -> u32;
    }
    crate::module_ns_store(
        ns,
        "getuid",
        crate::make_builtin_function_with_arity(
            "getuid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getuid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    return Ok(pyre_object::w_int_new(getuid() as i64));
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "geteuid",
        crate::make_builtin_function_with_arity(
            "geteuid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::geteuid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    return Ok(pyre_object::w_int_new(geteuid() as i64));
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "getgid",
        crate::make_builtin_function_with_arity(
            "getgid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getgid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    return Ok(pyre_object::w_int_new(getgid() as i64));
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "getegid",
        crate::make_builtin_function_with_arity(
            "getegid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getegid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    return Ok(pyre_object::w_int_new(getegid() as i64));
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    // os.getpid — host_os::process_id (std::process::id).
    crate::module_ns_store(
        ns,
        "getpid",
        crate::make_builtin_function_with_arity(
            "getpid",
            |_| Ok(pyre_object::w_int_new(host_os::process_id() as i64)),
            0,
        ),
    );
    // os.environ lookups from setenv / unsetenv / putenv / getenv — mutate
    // posix.environ (the dict) rather than calling libc; os.py writes back
    // into that dict in its _Environ wrapper.
    crate::module_ns_store(
        ns,
        "getenv",
        crate::make_builtin_function("getenv", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let key = unsafe {
                if pyre_object::is_str(args[0]) {
                    crate::baseobjspace::str_utf8_w(args[0])?.to_string()
                } else {
                    return Ok(pyre_object::w_none());
                }
            };
            #[cfg(feature = "sandbox")]
            {
                if let Ok(Some(value)) = crate::host_seam::ops::getenv(key.as_bytes()) {
                    return Ok(pyre_object::w_str_new(&String::from_utf8_lossy(&value)));
                }
            }
            #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
            {
                if let Ok(value) = host_os::var(&key) {
                    return Ok(pyre_object::w_str_new(&value));
                }
            }
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Ok(pyre_object::w_none())
            }
        }),
    );
    // ── host_env::posix-backed real implementations (override the noop
    //    placeholders registered above) ───────────────────────────────
    #[cfg(all(unix, feature = "host_env"))]
    {
        use rustpython_host_env::posix as host_posix;

        fn exec_argv(
            w_argv: PyObjectRef,
            function: &str,
        ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
            // PyPy interp_posix.execv: `space.unpackiterable(w_argv)` followed
            // by `space.fsencode_w` for every argument.
            let items = crate::baseobjspace::unpackiterable(w_argv, -1).map_err(|error| {
                if error.kind == crate::PyErrorKind::TypeError {
                    crate::PyError::type_error(format!(
                        "{function}() arg 2 must be an iterable of strings"
                    ))
                } else {
                    error
                }
            })?;
            if items.is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 must not be empty"
                )));
            }
            let mut argv = Vec::with_capacity(items.len());
            for item in items {
                let value = extract_path(item)?;
                argv.push(std::ffi::CString::new(value).map_err(|_| {
                    crate::PyError::value_error(format!(
                        "{function}() arg 2 contains an embedded null byte"
                    ))
                })?);
            }
            if argv[0].as_bytes().is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 first element cannot be empty"
                )));
            }
            Ok(argv)
        }

        fn exec_pointer_array(values: &[std::ffi::CString]) -> Vec<*const libc::c_char> {
            let mut pointers: Vec<_> = values.iter().map(|value| value.as_ptr()).collect();
            pointers.push(std::ptr::null());
            pointers
        }

        // PyPy interp_posix.execv: this call replaces the current process and
        // returns only to translate the host errno into OSError.
        crate::module_ns_store(
            ns,
            "execv",
            crate::make_builtin_function_with_arity(
                "execv",
                |args| {
                    let command = extract_path(args[0])?;
                    let command_c = std::ffi::CString::new(command).map_err(|_| {
                        crate::PyError::value_error("execv() path contains an embedded null byte")
                    })?;
                    let argv = exec_argv(args[1], "execv")?;
                    let argv_ptrs = exec_pointer_array(&argv);
                    let errno =
                        host_posix::exec_replace(&[command_c], argv_ptrs.as_ptr(), None) as i32;
                    // interp_posix.py:1814-1817 uses `wrap_oserror`, which does
                    // not attach the command path.
                    Err(errno_err(errno, ""))
                },
                2,
            ),
        );

        // PyPy interp_posix.execve/_env2interp: accept a mapping, fsencode
        // names and values, reject illegal names, then replace the process.
        crate::module_ns_store(
            ns,
            "execve",
            crate::make_builtin_function_with_arity(
                "execve",
                |args| {
                    let command = extract_path(args[0])?;
                    let command_c = std::ffi::CString::new(command).map_err(|_| {
                        crate::PyError::value_error("execve() path contains an embedded null byte")
                    })?;
                    let argv = exec_argv(args[1], "execve")?;
                    let argv_ptrs = exec_pointer_array(&argv);

                    let keys_obj = crate::baseobjspace::call_method(args[2], "keys", &[]);
                    if keys_obj.is_null() {
                        return Err(crate::call::take_call_error().unwrap_or_else(|| {
                            crate::PyError::type_error("execve: env must be a mapping")
                        }));
                    }
                    let keys = crate::baseobjspace::unpackiterable(keys_obj, -1)?;
                    let mut env = Vec::with_capacity(keys.len());
                    for key_obj in keys {
                        let value_obj = crate::baseobjspace::getitem(args[2], key_obj)?;
                        let key = extract_path(key_obj)?;
                        let value = extract_path(value_obj)?;
                        if key.is_empty()
                            || key.get(1..).is_some_and(|tail| tail.contains(&b'='))
                        {
                            return Err(crate::PyError::value_error(
                                "illegal environment variable name",
                            ));
                        }
                        let mut entry = key;
                        entry.push(b'=');
                        entry.extend_from_slice(&value);
                        env.push(std::ffi::CString::new(entry).map_err(|_| {
                            crate::PyError::value_error(
                                "execve() environment contains an embedded null byte",
                            )
                        })?);
                    }
                    let env_ptrs = exec_pointer_array(&env);
                    let errno = host_posix::exec_replace(
                        &[command_c],
                        argv_ptrs.as_ptr(),
                        Some(env_ptrs.as_ptr()),
                    ) as i32;
                    // `interp_posix.py:1812,1817` wraps the failure with
                    // `wrap_oserror`, which names no file, so the path stays
                    // out of the error the same way `execv` leaves it out.
                    Err(errno_err(errno, ""))
                },
                3,
            ),
        );

        // os.strerror(code) -> str
        crate::module_ns_store(
            ns,
            "strerror",
            crate::make_builtin_function_with_arity(
                "strerror",
                |args| {
                    let code = match args.first() {
                        Some(&o) => crate::baseobjspace::c_int_w(o)?,
                        None => {
                            return Err(crate::PyError::type_error(
                                "strerror() requires 1 argument",
                            ));
                        }
                    };
                    #[cfg(feature = "sandbox")]
                    {
                        let msg = crate::host_seam::ops::strerror(code)
                            .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                        return Ok(pyre_object::w_str_new(&String::from_utf8_lossy(&msg)));
                    }
                    #[cfg(not(feature = "sandbox"))]
                    Ok(pyre_object::w_str_new(
                        &rustpython_host_env::time::strerror(code),
                    ))
                },
                1,
            ),
        );

        // os.pipe() -> (r_fd, w_fd)
        crate::module_ns_store(
            ns,
            "pipe",
            crate::make_builtin_function_with_arity(
                "pipe",
                |_| match host_posix::pipe() {
                    Ok((rfd, wfd)) => {
                        use std::os::fd::IntoRawFd;
                        Ok(pyre_object::w_tuple_new(vec![
                            pyre_object::w_int_new(rfd.into_raw_fd() as i64),
                            pyre_object::w_int_new(wfd.into_raw_fd() as i64),
                        ]))
                    }
                    Err(e) => Err(io_err(e, "")),
                },
                0,
            ),
        );

        // os.sched_yield()
        crate::module_ns_store(
            ns,
            "sched_yield",
            crate::make_builtin_function_with_arity(
                "sched_yield",
                |_| {
                    host_posix::sched_yield().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                0,
            ),
        );

        // os.nice(increment) -> new niceness
        crate::module_ns_store(
            ns,
            "nice",
            crate::make_builtin_function_with_arity(
                "nice",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("nice() requires 1 argument"));
                    }
                    // interp_posix.py:2565 `@unwrap_spec(increment=c_int)`.
                    let inc = crate::baseobjspace::c_int_w(args[0])?;
                    let n = host_posix::nice(inc).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );

        // os.umask(mask) -> previous mask
        crate::module_ns_store(
            ns,
            "umask",
            crate::make_builtin_function_with_arity(
                "umask",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("umask() requires 1 argument"));
                    }
                    // interp_posix.py:1372 `@unwrap_spec(mask=c_int)`.
                    let mask = crate::baseobjspace::c_int_w(args[0])? as libc::mode_t;
                    let prev = host_posix::umask(mask);
                    Ok(pyre_object::w_int_new(prev as i64))
                },
                1,
            ),
        );

        // os.getlogin() -> str
        crate::module_ns_store(
            ns,
            "getlogin",
            crate::make_builtin_function_with_arity(
                "getlogin",
                |_| match host_posix::getlogin() {
                    Some(name) => Ok(pyre_object::w_str_new(name.to_string_lossy().as_ref())),
                    None => Err(crate::PyError::os_error_with_errno(
                        crate::builtins::io_error_posix_errno(&std::io::Error::last_os_error(), 0),
                        "getlogin",
                    )),
                },
                0,
            ),
        );

        // os.getgroups() -> list[int]
        crate::module_ns_store(
            ns,
            "getgroups",
            crate::make_builtin_function_with_arity(
                "getgroups",
                |_| {
                    let gs = host_posix::getgroups().map_err(|e| io_err(e, ""))?;
                    let items: Vec<_> = gs
                        .into_iter()
                        .map(|g| pyre_object::w_int_new(g as i64))
                        .collect();
                    Ok(pyre_object::w_list_new(items))
                },
                0,
            ),
        );

        // os.sched_get_priority_max(policy) -> int
        crate::module_ns_store(
            ns,
            "sched_get_priority_max",
            crate::make_builtin_function_with_arity(
                "sched_get_priority_max",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sched_get_priority_max() requires 1 argument",
                        ));
                    }
                    // interp_posix.py:2977 `@unwrap_spec(policy=int)`.
                    let policy = crate::baseobjspace::int_w(args[0])? as i32;
                    let m =
                        host_posix::sched_get_priority_max(policy).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            ),
        );

        // os.sched_get_priority_min(policy) -> int
        crate::module_ns_store(
            ns,
            "sched_get_priority_min",
            crate::make_builtin_function_with_arity(
                "sched_get_priority_min",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sched_get_priority_min() requires 1 argument",
                        ));
                    }
                    // interp_posix.py:2991 `@unwrap_spec(policy=int)`.
                    let policy = crate::baseobjspace::int_w(args[0])? as i32;
                    let m =
                        host_posix::sched_get_priority_min(policy).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            ),
        );

        // os.sync()
        #[cfg(not(any(target_os = "redox", target_os = "android")))]
        crate::module_ns_store(
            ns,
            "sync",
            crate::make_builtin_function_with_arity(
                "sync",
                |_| {
                    host_posix::sync();
                    Ok(pyre_object::w_none())
                },
                0,
            ),
        );

        // os.chdir(path)
        crate::module_ns_store(
            ns,
            "chdir",
            crate::make_builtin_function_with_arity(
                "chdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chdir() requires 1 argument"));
                    }
                    let path = crate::gateway::fsencode_path_w(args[0])?;
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    host_posix::chdir(&c_path)
                        .map_err(|e| errno_err_with_filename(e as i32, path.w_path()))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fchdir(fd)
        crate::module_ns_store(
            ns,
            "fchdir",
            crate::make_builtin_function_with_arity(
                "fchdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fchdir() requires 1 argument"));
                    }
                    // interp_posix.py:458-460 `fchdir(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`, which takes an int or
                    // anything exposing `fileno()`.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    host_posix::fchdir(fd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fork() -> child pid in parent, 0 in child
        crate::module_ns_store(
            ns,
            "fork",
            crate::make_builtin_function_with_arity(
                "fork",
                |_| {
                    if majit_gc::gc_sync::registered_threads() > 1 {
                        crate::warn::warn_deprecation(
                            "This process is multi-threaded, use of fork() may lead to deadlocks",
                        )?;
                    }
                    let blocked = crate::module::thread::before_external_block();
                    let fork_serial = FORK_SERIALIZER
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    drop(blocked);
                    run_fork_callbacks("before");
                    // A free-threaded fork must snapshot native/Python lock
                    // state while every other mutator is parked.  Enter
                    // through the collector's full request path so no GC
                    // operation can overlap the STW window.
                    let mut fork_result = None;
                    majit_gc::gc_sync::request_stw(|_| {
                        fork_result = Some(host_posix::fork());
                    });
                    match fork_result.expect("fork STW closure must run") {
                        Ok(0) => {
                            crate::module::thread::after_fork_child();
                            run_fork_callbacks("child");
                            // rposix.py `_exit` from the fork wrapper returns
                            // directly after `gc_thread_after_fork` and the
                            // registered child hooks.  Do not introduce a
                            // full collection/finalizer drain here: arbitrary
                            // inherited Python objects may be mid-lifecycle,
                            // and PyPy only runs app-level finalizers at their
                            // ordinary safe points after the child resumes.
                            drop(fork_serial);
                            Ok(pyre_object::w_int_new(0))
                        }
                        Ok(pid) => {
                            run_fork_callbacks("parent");
                            drop(fork_serial);
                            Ok(pyre_object::w_int_new(pid as i64))
                        }
                        Err(error) => {
                            run_fork_callbacks("parent");
                            drop(fork_serial);
                            Err(io_err(error, ""))
                        }
                    }
                },
                0,
            ),
        );

        // os.getppid() -> int
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "getppid",
            crate::make_builtin_function_with_arity(
                "getppid",
                |_| Ok(pyre_object::w_int_new(unsafe { libc::getppid() } as i64)),
                0,
            ),
        );

        // PyPy interp_posix.getsid(pid) -> rposix.getsid(pid).
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "getsid",
            crate::make_builtin_function_with_arity(
                "getsid",
                |args| {
                    let pid = match args.first() {
                        // interp_posix.py:2246 `@unwrap_spec(pid=c_int)`.
                        Some(&obj) => crate::baseobjspace::c_int_w(obj)? as libc::pid_t,
                        None => {
                            return Err(crate::PyError::type_error("getsid() requires 1 argument"));
                        }
                    };
                    let sid = unsafe { libc::getsid(pid) };
                    if sid == -1 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_int_new(sid as i64))
                },
                1,
            ),
        );

        // os.waitpid(pid, options) -> (pid, status)
        crate::module_ns_store(
            ns,
            "waitpid",
            crate::make_builtin_function_with_arity(
                "waitpid",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("waitpid() requires 2 arguments"));
                    }
                    // interp_posix.py:1708 `@unwrap_spec(pid=c_int, options=c_int)`.
                    let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                    let options = crate::baseobjspace::c_int_w(args[1])?;
                    let mut status: i32 = 0;
                    let res = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_posix::waitpid(pid, &mut status, options)
                    }
                    .map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(res as i64),
                        pyre_object::w_int_new(status as i64),
                    ]))
                },
                2,
            ),
        );

        // os.wait() -> (pid, status)
        crate::module_ns_store(
            ns,
            "wait",
            crate::make_builtin_function_with_arity(
                "wait",
                |_| {
                    let mut status: i32 = 0;
                    let res = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_posix::waitpid(-1, &mut status, 0)
                    }
                    .map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(res as i64),
                        pyre_object::w_int_new(status as i64),
                    ]))
                },
                0,
            ),
        );

        // os._exit(code) — immediate process exit, no cleanup.
        crate::module_ns_store(
            ns,
            "_exit",
            crate::make_builtin_function_with_arity(
                "_exit",
                |args| {
                    let code = match args.first() {
                        // interp_posix.py:1724 `@unwrap_spec(status=c_int)`.
                        Some(&o) => crate::baseobjspace::c_int_w(o)?,
                        None => {
                            return Err(crate::PyError::type_error("_exit() requires 1 argument"));
                        }
                    };
                    rustpython_host_env::os::exit(code)
                },
                1,
            ),
        );

        // Wait-status decoding macros (WIFEXITED/WEXITSTATUS/...): override
        // the noop stubs registered above with the libc bit-math.
        macro_rules! reg_wstatus {
            ($name:literal, |$s:ident| $body:expr) => {
                crate::module_ns_store(
                    ns,
                    $name,
                    crate::make_builtin_function_with_arity(
                        $name,
                        |args| {
                            let $s = match args.first() {
                                // interp_posix.py:2363-2371 `declare_new_w_star`
                                // types every wait macro `@unwrap_spec(status=c_int)`.
                                Some(&o) => crate::baseobjspace::c_int_w(o)?,
                                None => {
                                    return Err(crate::PyError::type_error(concat!(
                                        $name,
                                        "() requires 1 argument"
                                    )));
                                }
                            };
                            Ok($body)
                        },
                        1,
                    ),
                );
            };
        }
        reg_wstatus!("WIFEXITED", |s| pyre_object::w_bool_from(libc::WIFEXITED(
            s
        )));
        reg_wstatus!("WEXITSTATUS", |s| pyre_object::w_int_new(
            libc::WEXITSTATUS(s) as i64
        ));
        reg_wstatus!("WIFSIGNALED", |s| pyre_object::w_bool_from(
            libc::WIFSIGNALED(s)
        ));
        reg_wstatus!("WTERMSIG", |s| pyre_object::w_int_new(
            libc::WTERMSIG(s) as i64
        ));
        reg_wstatus!("WIFSTOPPED", |s| pyre_object::w_bool_from(
            libc::WIFSTOPPED(s)
        ));
        reg_wstatus!("WSTOPSIG", |s| pyre_object::w_int_new(
            libc::WSTOPSIG(s) as i64
        ));

        // Wait option flags — override the `0` placeholders registered above
        // with their real libc values (os.WNOHANG must be non-zero for
        // subprocess.poll()).
        crate::module_ns_store(ns, "WNOHANG", pyre_object::w_int_new(libc::WNOHANG as i64));
        crate::module_ns_store(
            ns,
            "WUNTRACED",
            pyre_object::w_int_new(libc::WUNTRACED as i64),
        );
        crate::module_ns_store(
            ns,
            "WCONTINUED",
            pyre_object::w_int_new(libc::WCONTINUED as i64),
        );

        // os.dup(fd) -> new_fd
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "dup",
            crate::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dup() requires 1 argument"));
                    }
                    // interp_posix.py:722 `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let n = unsafe { libc::dup(fd) };
                    if n < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );

        // os.dup2(fd, fd2, inheritable=True) -> fd2
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "dup2",
            crate::make_builtin_function("dup2", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("dup2() requires 2 arguments"));
                }
                // interp_posix.py:733 `@unwrap_spec(fd=c_int, fd2=c_int, inheritable=bool)`.
                let fd = crate::baseobjspace::c_int_w(args[0])?;
                let fd2 = crate::baseobjspace::c_int_w(args[1])?;
                let n = unsafe { libc::dup2(fd, fd2) };
                if n < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(n as i64))
            }),
        );

        // os.fsync(fd)
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "fsync",
            crate::make_builtin_function_with_arity(
                "fsync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fsync() requires 1 argument"));
                    }
                    // interp_posix.py:433-435 `fsync(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    let (r, errno) = crate::module::thread::call_external_function(|| unsafe {
                        libc::fsync(fd)
                    });
                    if r < 0 {
                        return Err(io_err(std::io::Error::from_raw_os_error(errno), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fdatasync(fd) — falls back to fsync on macOS, which has no
        // fdatasync syscall but exposes the same semantics through fsync.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "fdatasync",
            crate::make_builtin_function_with_arity(
                "fdatasync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "fdatasync() requires 1 argument",
                        ));
                    }
                    // interp_posix.py:443-446 `fdatasync(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    let (r, errno) = crate::module::thread::call_external_function(|| unsafe {
                        #[cfg(any(target_os = "linux", target_os = "android"))]
                        {
                            libc::fdatasync(fd)
                        }
                        #[cfg(not(any(target_os = "linux", target_os = "android")))]
                        {
                            libc::fsync(fd)
                        }
                    });
                    if r < 0 {
                        return Err(io_err(std::io::Error::from_raw_os_error(errno), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // rpython/rlib/rposix.py `ftruncate(fd, length)` — this must be a
        // real fd mutation whenever HAVE_FTRUNCATE is advertised.  Shared
        // memory sizes its newly-created object through this call before
        // mapping it.
        #[cfg(all(unix, not(feature = "sandbox")))]
        crate::module_ns_store(
            ns,
            "ftruncate",
            crate::make_builtin_function_with_arity(
                "ftruncate",
                |args| {
                    // interp_posix.py:404 `@unwrap_spec(fd=c_int, length=r_longlong)`.
                    // CPython 3.14's clinic gateway accepts `__index__` but not
                    // `__int__`; that newer coercion wins over PyPy's legacy
                    // `gateway_int_w` behavior.
                    if args.len() != 2 {
                        return Err(crate::PyError::type_error(format!(
                            "ftruncate expected 2 arguments, got {}",
                            args.len(),
                        )));
                    }
                    let w_fd = crate::baseobjspace::space_index(args[0])?;
                    let fd_value = crate::baseobjspace::int_w(w_fd).map_err(|err| {
                        if err.kind == crate::PyErrorKind::OverflowError {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C int",
                            )
                        } else {
                            err
                        }
                    })?;
                    let fd = libc::c_int::try_from(fd_value).map_err(|_| {
                        crate::PyError::overflow_error("Python int too large to convert to C int")
                    })?;
                    let w_length = crate::baseobjspace::space_index(args[1])?;
                    let length = crate::baseobjspace::int_w(w_length).map_err(|err| {
                        if err.kind == crate::PyErrorKind::OverflowError {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C long",
                            )
                        } else {
                            err
                        }
                    })? as libc::off_t;
                    // interp_posix.py:407-412: retry EINTR, propagate every
                    // other OSError.
                    loop {
                        let r = unsafe { libc::ftruncate(fd, length) };
                        if r == 0 {
                            break;
                        }
                        let err = std::io::Error::last_os_error();
                        if err.kind() != std::io::ErrorKind::Interrupted {
                            return Err(io_err(err, ""));
                        }
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.mkfifo(path, mode=0o666) -> None
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "mkfifo",
            crate::make_builtin_function("mkfifo", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("mkfifo() requires 1 argument"));
                }
                let path = crate::gateway::fsencode_path_w(args[0])?;
                // interp_posix.py:1322 `@unwrap_spec(mode=c_int, ...)`.
                let mode = if args.len() >= 2 {
                    crate::baseobjspace::c_int_w(args[1])? as libc::mode_t
                } else {
                    0o666
                };
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let r = unsafe { libc::mkfifo(c_path.as_ptr(), mode) };
                if r < 0 {
                    return Err(io_err_with_filename(
                        std::io::Error::last_os_error(),
                        path.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.kill(pid, sig) / os.killpg(pgid, sig)
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "kill",
            crate::make_builtin_function_with_arity(
                "kill",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("kill() requires 2 arguments"));
                    }
                    // interp_posix.py:1386 `@unwrap_spec(pid=c_int, signal=c_int)`.
                    // Both arguments go through `c_int_w`: a raw payload read
                    // would accept any object, and `is_int` is an exact-type
                    // check, so an `int` subclass instance — `signal.SIGHUP` is
                    // an `IntEnum` member — never reaches the checked path.
                    let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                    let sig = crate::baseobjspace::c_int_w(args[1])? as libc::c_int;
                    let r = unsafe { libc::kill(pid, sig) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "killpg",
            crate::make_builtin_function_with_arity(
                "killpg",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("killpg() requires 2 arguments"));
                    }
                    // interp_posix.py:1394 `@unwrap_spec(pgid=c_int, signal=c_int)`.
                    let pgid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                    let sig = crate::baseobjspace::c_int_w(args[1])? as libc::c_int;
                    let r = unsafe { libc::killpg(pgid, sig) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.statvfs(path) / os.fstatvfs(fd) -> statvfs_result
        #[cfg(not(target_os = "redox"))]
        crate::module_ns_store(ns, "statvfs_result", statvfs_result_seq_type());

        #[cfg(not(target_os = "redox"))]
        fn statvfs_to_obj(
            info: rustpython_host_env::posix::StatVfsInfo,
        ) -> pyre_object::PyObjectRef {
            let seq = vec![
                pyre_object::w_int_new(info.f_bsize as i64),
                pyre_object::w_int_new(info.f_frsize as i64),
                pyre_object::w_int_new(info.f_blocks as i64),
                pyre_object::w_int_new(info.f_bfree as i64),
                pyre_object::w_int_new(info.f_bavail as i64),
                pyre_object::w_int_new(info.f_files as i64),
                pyre_object::w_int_new(info.f_ffree as i64),
                pyre_object::w_int_new(info.f_favail as i64),
                pyre_object::w_int_new(info.f_flag as i64),
                pyre_object::w_int_new(info.f_namemax as i64),
            ];
            let extras = vec![("f_fsid", pyre_object::w_int_new(info.f_fsid as i64))];
            crate::_structseq::new_instance_with_extra(statvfs_result_seq_type(), seq, extras)
        }
        #[cfg(not(target_os = "redox"))]
        crate::module_ns_store(
            ns,
            "statvfs",
            crate::make_builtin_function_with_arity(
                "statvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("statvfs() requires 1 argument"));
                    }
                    let path = crate::gateway::fsencode_path_w(args[0])?;
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let info = host_posix::statvfs_path(&c_path)
                        .map_err(|e| io_err_with_filename(e, path.w_path()))?;
                    Ok(statvfs_to_obj(info))
                },
                1,
            ),
        );
        #[cfg(not(target_os = "redox"))]
        crate::module_ns_store(
            ns,
            "fstatvfs",
            crate::make_builtin_function_with_arity(
                "fstatvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fstatvfs() requires 1 argument"));
                    }
                    // interp_posix.py:693 `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let info = host_posix::statvfs_fd(fd).map_err(|e| io_err(e, ""))?;
                    Ok(statvfs_to_obj(info))
                },
                1,
            ),
        );

        // os.cpu_count() -> int | None
        crate::module_ns_store(
            ns,
            "cpu_count",
            crate::make_builtin_function_with_arity(
                "cpu_count",
                |_| {
                    let n = host_posix::get_number_of_os_threads();
                    if n <= 0 {
                        Ok(pyre_object::w_none())
                    } else {
                        Ok(pyre_object::w_int_new(n as i64))
                    }
                },
                0,
            ),
        );
        // _cpu_count alias — newer CPython exposes both.
        crate::module_ns_store(
            ns,
            "_cpu_count",
            crate::make_builtin_function_with_arity(
                "_cpu_count",
                |_| {
                    let n = host_posix::get_number_of_os_threads();
                    if n <= 0 {
                        Ok(pyre_object::w_none())
                    } else {
                        Ok(pyre_object::w_int_new(n as i64))
                    }
                },
                0,
            ),
        );

        // os.symlink(src, dst, target_is_directory=False) -> None
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "symlink",
            crate::make_builtin_function("symlink", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("symlink() requires 2 arguments"));
                }
                let src = crate::gateway::fsencode_path_w(args[0])?;
                let dst = crate::gateway::fsencode_path_w(args[1])?;
                let c_src = std::ffi::CString::new(src.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in src"))?;
                let c_dst = std::ffi::CString::new(dst.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in dst"))?;
                // host_env::posix only exposes symlinkat on non-redox unices;
                // call libc::symlink directly so we don't need an at-cwd dance.
                let ret = unsafe { libc::symlink(c_src.as_ptr(), c_dst.as_ptr()) };
                if ret < 0 {
                    return Err(io_err_with_filename(
                        std::io::Error::last_os_error(),
                        dst.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.chmod(path, mode) -> None
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "chmod",
            crate::make_builtin_function_with_arity(
                "chmod",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("chmod() requires 2 arguments"));
                    }
                    let path = crate::gateway::fsencode_path_w(args[0])?;
                    // `posix.chmod` unwraps `mode` as `c_int`, so a non-integer
                    // raises TypeError instead of reinterpreting its layout.
                    let mode = crate::baseobjspace::c_int_w(args[1])? as u32;
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let ret = unsafe { libc::chmod(c_path.as_ptr(), mode as libc::mode_t) };
                    if ret < 0 {
                        return Err(io_err_with_filename(
                            std::io::Error::last_os_error(),
                            path.w_path(),
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.fchmod(fd, mode) -> None
        crate::module_ns_store(
            ns,
            "fchmod",
            crate::make_builtin_function_with_arity(
                "fchmod",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fchmod() requires 2 arguments"));
                    }
                    // interp_posix.py:1260 `@unwrap_spec(fd=c_int, mode=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let mode = crate::baseobjspace::c_int_w(args[1])? as u32;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::fchmod(bfd, mode).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.chown(path, uid, gid, *, dir_fd=None, follow_symlinks=True) -> None
        // os.lchown(path, uid, gid) -> None
        // `uid`/`gid` of -1 means "leave unchanged", as for fchown.
        fn chown_entry(
            args: &[pyre_object::PyObjectRef],
            name: &str,
            default_follow: bool,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            use std::os::fd::BorrowedFd;
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let allowed: &[&str] = if default_follow {
                &["path", "uid", "gid", "dir_fd", "follow_symlinks"]
            } else {
                &["path", "uid", "gid"]
            };
            crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
            if pos.len() > 3 {
                // `chown` declares keyword-only `dir_fd`/`follow_symlinks`, so
                // its surplus-positional report is the "positional arguments"
                // form; `lchown` takes no keywords at all and reports the
                // plain "arguments" form.
                let surplus = if default_follow {
                    format!(
                        "{name}() takes exactly 3 positional arguments ({} given)",
                        pos.len()
                    )
                } else {
                    format!("{name}() takes at most 3 arguments ({} given)", pos.len())
                };
                return Err(crate::PyError::type_error(surplus));
            }
            // Every parameter is bound duplicate-aware, so a call that supplies
            // one both ways raises before the ownership syscall runs.
            let arg = |index: usize, key: &'static str| -> Result<PyObjectRef, crate::PyError> {
                match crate::builtins::bind_pos_or_kw(pos, kwargs, index, key, name, index + 1)? {
                    Some(value) => Ok(value),
                    None => Err(crate::PyError::type_error(format!(
                        "{name}() missing required argument '{key}' (pos {})",
                        index + 1
                    ))),
                }
            };
            let (path_obj, uid_obj, gid_obj) =
                (arg(0, "path")?, arg(1, "uid")?, arg(2, "gid")?);
            // `posixmodule.c path_converter` calls `__fspath__` and lets what it
            // raises out: a `RuntimeError` from a user `__fspath__` is that
            // object's error, not a statement that the argument was the wrong
            // type.  Rewriting every failure into a `TypeError` here would also
            // swallow the `UnicodeEncodeError` a lone surrogate produces.
            let path = crate::gateway::fsencode_path_w(path_obj)?;
            // `_Py_Uid_Converter` / `_Py_Gid_Converter`: `uid_t` is unsigned, yet
            // -1 is always accepted as the "leave unchanged" sentinel.  Only
            // that one value means unchanged; every other id is judged by
            // round-tripping through `uid_t`, so nothing is silently wrapped —
            // 2**32 truncates to 0 and would otherwise request uid 0.
            //
            // The two range reports follow the C converter's own split: a value
            // that still fits a C long but fails the round trip is "less than
            // minimum" (including 2**32, whose truncation reads as underflow),
            // while one too wide for a long is "greater than maximum".
            let id_of =
                |w: pyre_object::PyObjectRef, what: &str| -> Result<Option<u32>, crate::PyError> {
                    if !unsafe { crate::builtins::index_check(w) } {
                        return Err(crate::PyError::type_error(format!(
                            "{what} should be integer, not {}",
                            crate::type_methods::arg_type_name(w)
                        )));
                    }
                    let w_index = crate::baseobjspace::space_index(w)?;
                    let raw = crate::baseobjspace::int_w(w_index).map_err(|_| {
                        crate::PyError::overflow_error(format!("{what} is greater than maximum"))
                    })?;
                    if raw == -1 {
                        return Ok(None);
                    }
                    let narrowed = raw as u32;
                    if i64::from(narrowed) != raw {
                        return Err(crate::PyError::overflow_error(format!(
                            "{what} is less than minimum"
                        )));
                    }
                    Ok(Some(narrowed))
                };
            let (uid, gid) = (id_of(uid_obj, "uid")?, id_of(gid_obj, "gid")?);
            if let Some(dir_fd) = crate::builtins::kwarg_get(kwargs, "dir_fd")
                && !unsafe { pyre_object::is_none(dir_fd) }
            {
                return Err(crate::PyError::not_implemented(format!(
                    "{name}: dir_fd unavailable on this platform"
                )));
            }
            let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                Some(v) => crate::baseobjspace::is_true(v)?,
                None => default_follow,
            };
            // `fchownat` with `AT_FDCWD` is the `chown` / `lchown` pair:
            // the flagless call follows the final symlink, `AT_SYMLINK_NOFOLLOW`
            // does not.
            let cwd = unsafe { BorrowedFd::borrow_raw(libc::AT_FDCWD) };
            host_posix::fchownat(
                cwd,
                path_from_bytes(&path.as_bytes).as_os_str(),
                uid,
                gid,
                follow_symlinks,
            )
            .map_err(|e| io_err_with_filename(e, path.w_path()))?;
            Ok(pyre_object::w_none())
        }
        crate::module_ns_store(
            ns,
            "chown",
            crate::make_builtin_function("chown", |args| chown_entry(args, "chown", true)),
        );
        crate::module_ns_store(
            ns,
            "lchown",
            crate::make_builtin_function("lchown", |args| chown_entry(args, "lchown", false)),
        );

        // os.fchown(fd, uid, gid) -> None  (uid/gid of -1 means "leave unchanged")
        crate::module_ns_store(
            ns,
            "fchown",
            crate::make_builtin_function_with_arity(
                "fchown",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("fchown() requires 3 arguments"));
                    }
                    // interp_posix.py:2527-2533 `@unwrap_spec(uid=c_uid_t,
                    // gid=c_gid_t)` with the descriptor taken by
                    // `space.c_filedescriptor_w`. The spec is applied by the
                    // gateway before the body runs, so a bad uid/gid is
                    // reported ahead of a bad descriptor.
                    //
                    // `c_uid_t_w` (baseobjspace.py:2110-2125) is what turns -1
                    // into `UINT_MAX`, i.e. the `(uid_t)-1` "leave unchanged"
                    // sentinel that `host_posix::fchown` spells as `None`.
                    let unchanged = |value: u32| (value != u32::MAX).then_some(value);
                    let uid = unchanged(crate::baseobjspace::c_uid_t_w(args[1])?);
                    let gid = unchanged(crate::baseobjspace::c_uid_t_w(args[2])?);
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::fchown(bfd, uid, gid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.set_inheritable(fd, inheritable) -> None
        crate::module_ns_store(
            ns,
            "set_inheritable",
            crate::make_builtin_function_with_arity(
                "set_inheritable",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "set_inheritable() requires 2 arguments",
                        ));
                    }
                    // interp_posix.py:1165 `@unwrap_spec(fd=c_int, inheritable=int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let inherit = crate::baseobjspace::int_w(args[1])? != 0;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::set_inheritable(bfd, inherit).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.access(path, mode) -> bool
        crate::module_ns_store(
            ns,
            "access",
            crate::make_builtin_function("access", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("access() requires 2 arguments"));
                }
                let path = extract_path(args[0])?;
                // interp_posix.py:744 `@unwrap_spec(mode=c_int, ...)`.
                let mode = crate::baseobjspace::c_int_w(args[1])?;
                #[cfg(feature = "sandbox")]
                {
                    return Ok(pyre_object::w_bool_from(
                        crate::host_seam::ops::access(&path, mode).unwrap_or(false),
                    ));
                }
                #[cfg(not(feature = "sandbox"))]
                {
                    // `check_access` takes the mask as a `u8` and rejects any
                    // bit outside `R_OK | W_OK | X_OK`. Narrowing before that
                    // check would fold a mode like 256 onto `F_OK` and answer
                    // "exists" for a mode `access(2)` rejects with EINVAL.
                    let Ok(mode) = u8::try_from(mode) else {
                        return Ok(pyre_object::w_bool_from(false));
                    };
                    match host_posix::check_access(path_from_bytes(&path).as_ref(), mode) {
                        Ok(ok) => Ok(pyre_object::w_bool_from(ok)),
                        Err(_) => Ok(pyre_object::w_bool_from(false)),
                    }
                }
            }),
        );

        // os.chroot(path) -> None
        crate::module_ns_store(
            ns,
            "chroot",
            crate::make_builtin_function_with_arity(
                "chroot",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chroot() requires 1 argument"));
                    }
                    let path = crate::gateway::fsencode_path_w(args[0])?;
                    host_posix::chroot(path_from_bytes(&path.as_bytes).as_ref())
                        .map_err(|e| io_err_with_filename(e, path.w_path()))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.getloadavg() -> (1m, 5m, 15m)
        crate::module_ns_store(
            ns,
            "getloadavg",
            crate::make_builtin_function_with_arity(
                "getloadavg",
                |_| {
                    let [l1, l5, l15] =
                        rustpython_host_env::time::getloadavg().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_float_new(l1),
                        pyre_object::w_float_new(l5),
                        pyre_object::w_float_new(l15),
                    ]))
                },
                0,
            ),
        );

        // os.times() -> posix.times_result(user, system, children_user,
        //                                  children_system, elapsed)
        crate::module_ns_store(
            ns,
            "times",
            crate::make_builtin_function_with_arity(
                "times",
                |_| {
                    let t =
                        rustpython_host_env::time::process_times().map_err(|e| io_err(e, ""))?;
                    Ok(crate::_structseq::new_instance(
                        times_result_seq_type(),
                        vec![
                            pyre_object::w_float_new(t.user),
                            pyre_object::w_float_new(t.system),
                            pyre_object::w_float_new(t.children_user),
                            pyre_object::w_float_new(t.children_system),
                            pyre_object::w_float_new(t.elapsed),
                        ],
                    ))
                },
                0,
            ),
        );

        // os.waitstatus_to_exitcode(status) -> int
        crate::module_ns_store(
            ns,
            "waitstatus_to_exitcode",
            crate::make_builtin_function_with_arity(
                "waitstatus_to_exitcode",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "waitstatus_to_exitcode() requires 1 argument",
                        ));
                    }
                    // app_posix.py:149-176 is app-level and reaches the status
                    // through `posix.WIFEXITED`/`WEXITSTATUS`, each of which is
                    // `@unwrap_spec(status=c_int)`.
                    let status = crate::baseobjspace::c_int_w(args[0])?;
                    match rustpython_host_env::time::waitstatus_to_exitcode(status) {
                        Some(code) => Ok(pyre_object::w_int_new(code as i64)),
                        None => Err(crate::PyError::value_error(
                            "waitstatus_to_exitcode: invalid status",
                        )),
                    }
                },
                1,
            ),
        );

        // os.system(command) -> exit_status
        crate::module_ns_store(
            ns,
            "system",
            crate::make_builtin_function_with_arity(
                "system",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("system() requires 1 argument"));
                    }
                    // `interp_posix.py:815 command='fsencode'`, which
                    // `gateway.py:365` unwraps with `space.fsencode_w`: the
                    // shell gets the filesystem bytes, so a command naming a
                    // byte with no UTF-8 spelling survives instead of being
                    // refused, and `bytes` / `__fspath__` are accepted as the
                    // converter accepts them.
                    let cmd = crate::gateway::fsencode_bytes_w(args[0])?;
                    let c_cmd = std::ffi::CString::new(cmd)
                        .map_err(|_| crate::PyError::value_error("embedded null in command"))?;
                    let rc = rustpython_host_env::os::system(&c_cmd);
                    Ok(pyre_object::w_int_new(rc as i64))
                },
                1,
            ),
        );

        // os.sendfile(out_fd, in_fd, offset, count) -> bytes_sent
        //
        // Ported from pypy/module/posix/interp_posix.py:2932-2961:
        //   * 4 positional args: out_fd, in_fd (called "in_" in PyPy because
        //     "in" is reserved), offset, count.
        //   * offset == None: linux-only "no-offset" path (NULL pointer);
        //     non-linux raises TypeError("an integer is required (got None)")
        //     verbatim from PyPy.
        //   * offset == int: read as i64 (PyPy uses
        //     space.gateway_r_longlong_w) and routed through
        //     rustpython_host_env::posix::sendfile (linux) or the BSD-form
        //     wrapper (macos).
        //   * Returns bytes-sent as int (PyPy: space.newint(res)).
        //
        // EINTR retry loop intentionally omitted — pyre's other os-syscall
        // wrappers don't do manual retry (relies on PEP 475 OS-level retry),
        // matching pyre-wide convention rather than introducing a single
        // outlier.
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(feature = "sandbox")
        ))]
        crate::module_ns_store(
            ns,
            "sendfile",
            crate::make_builtin_function("sendfile", |args| {
                use std::os::fd::BorrowedFd;
                if args.len() < 4 {
                    return Err(crate::PyError::type_error(
                        "sendfile() requires 4 arguments",
                    ));
                }
                // interp_posix.py:2946 `@unwrap_spec(out_fd=c_int, count=int)`,
                // with `in_ = space.c_int_w(w_in_fd)` in the body (:2955). The
                // spec runs in the gateway, so the count is converted before
                // the descriptor argument that follows it here.
                let out_fd = crate::baseobjspace::c_int_w(args[0])?;
                let count_raw = crate::baseobjspace::int_w(args[3])?;
                let in_fd = crate::baseobjspace::c_int_w(args[1])?;
                let w_offset = args[2];
                if unsafe { pyre_object::is_none(w_offset) } {
                    // linux-only no-offset path; non-linux raises TypeError
                    // matching interp_posix.py:2946.
                    #[cfg(not(target_os = "linux"))]
                    {
                        let _ = (out_fd, in_fd, count_raw);
                        return Err(crate::PyError::type_error(
                            "an integer is required (got None)",
                        ));
                    }
                    #[cfg(target_os = "linux")]
                    {
                        // host_env doesn't expose a NULL-offset variant; call
                        // libc::sendfile directly with a null pointer, matching
                        // rposix.sendfile_no_offset (rposix.py:3066-3069).
                        let count = count_raw as libc::size_t;
                        let (res, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::sendfile(out_fd, in_fd, core::ptr::null_mut(), count)
                            });
                        if res < 0 {
                            return Err(io_err(std::io::Error::from_raw_os_error(errno), ""));
                        }
                        return Ok(pyre_object::w_int_new(res as i64));
                    }
                }
                // interp_posix.py:2968 `space.gateway_r_longlong_w(w_offset)`.
                let offset_i64 = crate::baseobjspace::int_w(w_offset)?;
                let out_b = unsafe { BorrowedFd::borrow_raw(out_fd) };
                let in_b = unsafe { BorrowedFd::borrow_raw(in_fd) };
                #[cfg(target_os = "linux")]
                {
                    let count = count_raw as usize;
                    let mut offset: rustpython_host_env::crt_fd::Offset = offset_i64 as _;
                    let n = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_posix::sendfile(out_b, in_b, &mut offset, count)
                    }
                    .map_err(|e| io_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(n as i64));
                }
                #[cfg(target_os = "macos")]
                {
                    let (res, written) = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_posix::sendfile(
                            in_b,
                            out_b,
                            offset_i64 as rustpython_host_env::crt_fd::Offset,
                            count_raw,
                            None,
                            None,
                        )
                    };
                    // rposix.py:3086-3095: BSD sendfile reports a partial
                    // transfer through sbytes even when the syscall result is
                    // EAGAIN/EBUSY. Return that progress so asyncio advances
                    // its file offset instead of resending the same range.
                    if let Err(error) = res {
                        if written == 0
                            || !matches!(
                                error.raw_os_error(),
                                Some(libc::EAGAIN) | Some(libc::EBUSY)
                            )
                        {
                            return Err(io_err(error, ""));
                        }
                    }
                    return Ok(pyre_object::w_int_new(written));
                }
            }),
        );

        // os.posix_spawn(path, argv, env, *, file_actions=None) -> pid
        // os.posix_spawnp(file, argv, env, *, file_actions=None) -> pid
        // Currently supports path/argv/env + the file_actions sequence
        // ((POSIX_SPAWN_OPEN, fd, path, flags, mode) | (POSIX_SPAWN_CLOSE,
        // fd) | (POSIX_SPAWN_DUP2, fd, newfd)). Other CPython kwargs
        // (setpgroup, setsid, setsigmask, setsigdef, resetids, scheduler)
        // are not yet plumbed.
        #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "macos"))]
        {
            fn build_posix_spawn(
                args: &[pyre_object::PyObjectRef],
                spawnp: bool,
            ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
                let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
                if positional.len() < 3 {
                    return Err(crate::PyError::type_error(
                        "posix_spawn() requires path, argv, env",
                    ));
                }
                let path = crate::gateway::fsencode_path_w(positional[0])?;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice()).map_err(|_| {
                    crate::PyError::value_error("posix_spawn: embedded null in path")
                })?;
                let argv = collect_cstring_seq(positional[1], "posix_spawn", "argv")?;
                // posixmodule.c parses `env` as a mapping.  This is the same
                // owner/shape as PyPy's `_env2interp` path used by execve:
                // iterate `keys()`, fetch each value through `getitem`, then
                // filesystem-encode both sides into `key=value`.
                let env = collect_spawn_env(positional[2])?;
                let file_actions_obj = crate::builtins::kwarg_get(kwargs, "file_actions");
                let actions: Vec<rustpython_host_env::posix::PosixSpawnFileAction> =
                    if let Some(fa) = file_actions_obj {
                        if unsafe { pyre_object::is_none(fa) } {
                            Vec::new()
                        } else {
                            decode_file_actions(fa)?
                        }
                    } else {
                        Vec::new()
                    };
                let config = rustpython_host_env::posix::PosixSpawnConfig {
                    path: c_path.as_c_str(),
                    args: &argv,
                    env: &env,
                    file_actions: &actions,
                    setsigdef: None,
                    setpgroup: None,
                    resetids: false,
                    setsid: false,
                    setsigmask: None,
                    spawnp,
                };
                let pid = host_posix::posix_spawn(config)
                    .map_err(|e| io_err_with_filename(e, path.w_path()))?;
                Ok(pyre_object::w_int_new(pid as i64))
            }
            fn collect_spawn_env(
                mapping: pyre_object::PyObjectRef,
            ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
                // CPython's posix_spawn accepts None as "inherit environ";
                // subprocess._posix_spawn uses exactly this form when Popen
                // was called without an explicit env mapping.
                if unsafe { pyre_object::is_none(mapping) } {
                    let mut env = Vec::new();
                    for (key, value) in host_os::vars_os() {
                        let key = key.as_encoded_bytes();
                        let value = value.as_encoded_bytes();
                        let mut entry = Vec::with_capacity(key.len() + 1 + value.len());
                        entry.extend_from_slice(key);
                        entry.push(b'=');
                        entry.extend_from_slice(value);
                        env.push(std::ffi::CString::new(entry).map_err(|_| {
                            crate::PyError::value_error(
                                "posix_spawn() environment contains an embedded null byte",
                            )
                        })?);
                    }
                    return Ok(env);
                }
                let keys_obj = crate::baseobjspace::call_method(mapping, "keys", &[]);
                if keys_obj.is_null() {
                    return Err(crate::call::take_call_error().unwrap_or_else(|| {
                        crate::PyError::type_error("posix_spawn: env must be a mapping")
                    }));
                }
                let keys = crate::baseobjspace::unpackiterable(keys_obj, -1)?;
                // `getitem` runs the mapping's `__getitem__` and both encodes
                // allocate, so the mapping and every key are published once and
                // read back per iteration rather than kept in plain locals.
                let _env_roots = pyre_object::gc_roots::push_roots();
                let mapping_slot = pyre_object::gc_roots::pin_roots(&[mapping]);
                let keys_base = pyre_object::gc_roots::pin_roots(&keys);
                let mut env = Vec::with_capacity(keys.len());
                for i in 0..keys.len() {
                    let _entry_roots = pyre_object::gc_roots::push_roots();
                    let value_obj = crate::baseobjspace::getitem(
                        pyre_object::gc_roots::shadow_stack_get(mapping_slot),
                        pyre_object::gc_roots::shadow_stack_get(keys_base + i),
                    )?;
                    // Encoding the key can collect, so the value it was fetched
                    // beside has to be published before that call.
                    let value_slot = pyre_object::gc_roots::pin_roots(&[value_obj]);
                    let key = crate::gateway::fsencode_bytes_w(
                        pyre_object::gc_roots::shadow_stack_get(keys_base + i),
                    )?;
                    let value = crate::gateway::fsencode_bytes_w(
                        pyre_object::gc_roots::shadow_stack_get(value_slot),
                    )?;
                    // interp_posix.py:1762-1769 permits the Windows `=C:`
                    // spelling and rejects `=` only after the first byte.
                    if key.is_empty() || key.get(1..).is_some_and(|tail| tail.contains(&b'=')) {
                        return Err(crate::PyError::value_error(
                            "illegal environment variable name",
                        ));
                    }
                    let mut entry = Vec::with_capacity(key.len() + 1 + value.len());
                    entry.extend_from_slice(&key);
                    entry.push(b'=');
                    entry.extend_from_slice(&value);
                    env.push(std::ffi::CString::new(entry).map_err(|_| {
                        crate::PyError::value_error(
                            "posix_spawn() environment contains an embedded null byte",
                        )
                    })?);
                }
                Ok(env)
            }
            fn collect_cstring_seq(
                obj: pyre_object::PyObjectRef,
                fn_name: &str,
                arg_name: &str,
            ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
                let items: Vec<pyre_object::PyObjectRef> = if unsafe { pyre_object::is_list(obj) } {
                    let n = unsafe { pyre_object::w_list_len(obj) };
                    (0..n)
                        .filter_map(|i| unsafe { pyre_object::w_list_getitem(obj, i as i64) })
                        .collect()
                } else if unsafe { pyre_object::is_tuple(obj) } {
                    let n = unsafe { pyre_object::w_tuple_len(obj) };
                    (0..n)
                        .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(obj, i as i64) })
                        .collect()
                } else {
                    return Err(crate::PyError::type_error(format!(
                        "{fn_name}(): {arg_name} must be a list or tuple",
                    )));
                };
                // `interp_posix.py:1742 args = [space.fsencode_w(w_arg)
                // for w_arg in args_w]`: every argv/envp entry crosses
                // to the new process as filesystem bytes, and the same
                // converter decides what an entry may be.
                // `fsencode_bytes_w` reaches `__fspath__` for a non-str, non-bytes entry, so
                // encoding one entry can collect and move the entries not yet converted.
                // Publish the sequence once and read each entry back per iteration.
                let _seq_roots = pyre_object::gc_roots::push_roots();
                let items_base = pyre_object::gc_roots::pin_roots(&items);
                let mut out = Vec::with_capacity(items.len());
                for i in 0..items.len() {
                    let bytes = crate::gateway::fsencode_bytes_w(
                        pyre_object::gc_roots::shadow_stack_get(items_base + i),
                    )?;
                    out.push(std::ffi::CString::new(bytes).map_err(|_| {
                        crate::PyError::value_error(format!(
                            "{fn_name}(): embedded null in {arg_name}",
                        ))
                    })?);
                }
                Ok(out)
            }
            fn decode_file_actions(
                obj: pyre_object::PyObjectRef,
            ) -> Result<Vec<rustpython_host_env::posix::PosixSpawnFileAction>, crate::PyError>
            {
                use rustpython_host_env::posix::PosixSpawnFileAction;
                let len = if unsafe { pyre_object::is_list(obj) } {
                    unsafe { pyre_object::w_list_len(obj) }
                } else if unsafe { pyre_object::is_tuple(obj) } {
                    unsafe { pyre_object::w_tuple_len(obj) }
                } else {
                    return Err(crate::PyError::type_error(
                        "posix_spawn: file_actions must be a list or tuple",
                    ));
                };
                // Every field of a `file_actions` entry is an `int` argument of
                // `os.posix_spawn`, so it is converted rather than read as a
                // payload: the caller controls the tuple's contents, and an
                // `int` subclass or a plain non-int would otherwise be
                // reinterpreted as a descriptor, flag set or mode.
                let field = |entry: PyObjectRef, index: i64| -> Result<i32, crate::PyError> {
                    let value = unsafe { pyre_object::w_tuple_getitem(entry, index) }
                        .ok_or_else(|| {
                            crate::PyError::value_error("posix_spawn: file_actions entry too short")
                        })?;
                    crate::baseobjspace::c_int_w(value)
                };
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let entry = if unsafe { pyre_object::is_list(obj) } {
                        unsafe { pyre_object::w_list_getitem(obj, i as i64) }
                    } else {
                        unsafe { pyre_object::w_tuple_getitem(obj, i as i64) }
                    }
                    .ok_or_else(|| {
                        crate::PyError::value_error("posix_spawn: file_actions entry missing")
                    })?;
                    if unsafe { !pyre_object::is_tuple(entry) } {
                        return Err(crate::PyError::type_error(
                            "posix_spawn: each file_actions entry must be a tuple",
                        ));
                    }
                    let tlen = unsafe { pyre_object::w_tuple_len(entry) };
                    if tlen < 2 {
                        return Err(crate::PyError::value_error(
                            "posix_spawn: file_actions entry too short",
                        ));
                    }
                    let op = field(entry, 0)?;
                    match op {
                        0 => {
                            // POSIX_SPAWN_OPEN: (op, fd, path, flags, mode)
                            if tlen < 5 {
                                return Err(crate::PyError::value_error(
                                    "posix_spawn: OPEN action requires fd, path, flags, mode",
                                ));
                            }
                            let fd = field(entry, 1)?;
                            let path_obj =
                                unsafe { pyre_object::w_tuple_getitem(entry, 2).unwrap() };
                            let path = extract_path(path_obj)?;
                            let cpath =
                                std::ffi::CString::new(path).map_err(|_| {
                                    crate::PyError::value_error(
                                        "posix_spawn: embedded null in OPEN path",
                                    )
                                })?;
                            let oflag = field(entry, 3)?;
                            let mode = field(entry, 4)? as u32;
                            out.push(PosixSpawnFileAction::Open {
                                fd,
                                path: cpath,
                                oflag,
                                mode,
                            });
                        }
                        1 => {
                            // POSIX_SPAWN_CLOSE: (op, fd)
                            let fd = field(entry, 1)?;
                            out.push(PosixSpawnFileAction::Close { fd });
                        }
                        2 => {
                            // POSIX_SPAWN_DUP2: (op, fd, newfd)
                            if tlen < 3 {
                                return Err(crate::PyError::value_error(
                                    "posix_spawn: DUP2 action requires fd, newfd",
                                ));
                            }
                            let fd = field(entry, 1)?;
                            let newfd = field(entry, 2)?;
                            out.push(PosixSpawnFileAction::Dup2 { fd, newfd });
                        }
                        _ => {
                            return Err(crate::PyError::value_error(
                                "posix_spawn: unknown file_actions opcode",
                            ));
                        }
                    }
                }
                Ok(out)
            }
            crate::module_ns_store(
                ns,
                "posix_spawn",
                crate::make_builtin_function("posix_spawn", |args| build_posix_spawn(args, false)),
            );
            crate::module_ns_store(
                ns,
                "posix_spawnp",
                crate::make_builtin_function("posix_spawnp", |args| build_posix_spawn(args, true)),
            );
            crate::module_ns_store(ns, "POSIX_SPAWN_OPEN", pyre_object::w_int_new(0));
            crate::module_ns_store(ns, "POSIX_SPAWN_CLOSE", pyre_object::w_int_new(1));
            crate::module_ns_store(ns, "POSIX_SPAWN_DUP2", pyre_object::w_int_new(2));
        }

        // os.ttyname(fd) -> str
        crate::module_ns_store(
            ns,
            "ttyname",
            crate::make_builtin_function_with_arity(
                "ttyname",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("ttyname() requires fd"));
                    }
                    // interp_posix.py:2382 `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    let name = host_posix::ttyname(bfd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_str_new(&name.to_string_lossy()))
                },
                1,
            ),
        );

        // os.tcgetpgrp(fd) -> pgid
        crate::module_ns_store(
            ns,
            "tcgetpgrp",
            crate::make_builtin_function_with_arity(
                "tcgetpgrp",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("tcgetpgrp() requires fd"));
                    }
                    // interp_posix.py:2269 `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    let pgid = host_posix::tcgetpgrp(bfd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(pgid as i64))
                },
                1,
            ),
        );

        // os.tcsetpgrp(fd, pgid) -> None
        crate::module_ns_store(
            ns,
            "tcsetpgrp",
            crate::make_builtin_function_with_arity(
                "tcsetpgrp",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("tcsetpgrp() requires fd, pgid"));
                    }
                    // interp_posix.py:2281 `@unwrap_spec(fd=c_int, pgid=c_gid_t)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let pgid = crate::baseobjspace::c_uid_t_w(args[1])? as libc::pid_t;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::tcsetpgrp(bfd, pgid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.getpriority(which, who) -> int
        crate::module_ns_store(
            ns,
            "getpriority",
            crate::make_builtin_function_with_arity(
                "getpriority",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "getpriority() requires which, who",
                        ));
                    }
                    // interp_posix.py:2340 `@unwrap_spec(which=int, who=int)`.
                    let which =
                        crate::baseobjspace::int_w(args[0])? as host_posix::PriorityWhichType;
                    let who = crate::baseobjspace::int_w(args[1])? as host_posix::PriorityWhoType;
                    let prio = host_posix::getpriority(which, who).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(prio as i64))
                },
                2,
            ),
        );

        // os.setpriority(which, who, priority) -> None
        crate::module_ns_store(
            ns,
            "setpriority",
            crate::make_builtin_function_with_arity(
                "setpriority",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setpriority() requires which, who, priority",
                        ));
                    }
                    // interp_posix.py:2352 `@unwrap_spec(which=int, who=int,
                    // priority=int)`.
                    let which =
                        crate::baseobjspace::int_w(args[0])? as host_posix::PriorityWhichType;
                    let who = crate::baseobjspace::int_w(args[1])? as host_posix::PriorityWhoType;
                    let prio = crate::baseobjspace::int_w(args[2])? as i32;
                    host_posix::setpriority(which, who, prio).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        crate::module_ns_store(
            ns,
            "PRIO_PROCESS",
            pyre_object::w_int_new(libc::PRIO_PROCESS as i64),
        );
        crate::module_ns_store(
            ns,
            "PRIO_PGRP",
            pyre_object::w_int_new(libc::PRIO_PGRP as i64),
        );
        crate::module_ns_store(
            ns,
            "PRIO_USER",
            pyre_object::w_int_new(libc::PRIO_USER as i64),
        );

        // `posixmodule.c` `pathconf_names` — the `_PC_*` table
        // `conv_path_confname` resolves a string `name` argument through.
        // `libc` exports the constants only on the BSD family, so the glibc
        // values (`bits/confname.h`) are spelled out for Linux.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        const PATHCONF_NAMES: &[(&str, i32)] = &[
            ("PC_ALLOC_SIZE_MIN", libc::_PC_ALLOC_SIZE_MIN),
            ("PC_ASYNC_IO", libc::_PC_ASYNC_IO),
            ("PC_CHOWN_RESTRICTED", libc::_PC_CHOWN_RESTRICTED),
            ("PC_FILESIZEBITS", libc::_PC_FILESIZEBITS),
            ("PC_LINK_MAX", libc::_PC_LINK_MAX),
            ("PC_MAX_CANON", libc::_PC_MAX_CANON),
            ("PC_MAX_INPUT", libc::_PC_MAX_INPUT),
            ("PC_MIN_HOLE_SIZE", libc::_PC_MIN_HOLE_SIZE),
            ("PC_NAME_MAX", libc::_PC_NAME_MAX),
            ("PC_NO_TRUNC", libc::_PC_NO_TRUNC),
            ("PC_PATH_MAX", libc::_PC_PATH_MAX),
            ("PC_PIPE_BUF", libc::_PC_PIPE_BUF),
            ("PC_PRIO_IO", libc::_PC_PRIO_IO),
            ("PC_REC_INCR_XFER_SIZE", libc::_PC_REC_INCR_XFER_SIZE),
            ("PC_REC_MAX_XFER_SIZE", libc::_PC_REC_MAX_XFER_SIZE),
            ("PC_REC_MIN_XFER_SIZE", libc::_PC_REC_MIN_XFER_SIZE),
            ("PC_REC_XFER_ALIGN", libc::_PC_REC_XFER_ALIGN),
            ("PC_SYMLINK_MAX", libc::_PC_SYMLINK_MAX),
            ("PC_SYNC_IO", libc::_PC_SYNC_IO),
            ("PC_VDISABLE", libc::_PC_VDISABLE),
        ];
        #[cfg(target_os = "linux")]
        const PATHCONF_NAMES: &[(&str, i32)] = &[
            ("PC_2_SYMLINKS", 20),
            ("PC_ALLOC_SIZE_MIN", 18),
            ("PC_ASYNC_IO", 10),
            ("PC_CHOWN_RESTRICTED", 6),
            ("PC_FILESIZEBITS", 13),
            ("PC_LINK_MAX", 0),
            ("PC_MAX_CANON", 1),
            ("PC_MAX_INPUT", 2),
            ("PC_NAME_MAX", 3),
            ("PC_NO_TRUNC", 7),
            ("PC_PATH_MAX", 4),
            ("PC_PIPE_BUF", 5),
            ("PC_PRIO_IO", 11),
            ("PC_REC_INCR_XFER_SIZE", 14),
            ("PC_REC_MAX_XFER_SIZE", 15),
            ("PC_REC_MIN_XFER_SIZE", 16),
            ("PC_REC_XFER_ALIGN", 17),
            ("PC_SOCK_MAXBUF", 12),
            ("PC_SYMLINK_MAX", 19),
            ("PC_SYNC_IO", 9),
            ("PC_VDISABLE", 8),
        ];
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux")))]
        const PATHCONF_NAMES: &[(&str, i32)] = &[];
        let _pathconf_roots = pyre_object::gc_roots::push_roots();
        let names_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
        for (name, value) in PATHCONF_NAMES {
            // The value is allocated before the store, and the dict is reloaded
            // from its root slot every iteration because the insert itself can
            // grow — and so relocate — the dict.
            let w_value = pyre_object::w_int_new(*value as i64);
            unsafe {
                pyre_object::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(names_slot),
                    name,
                    w_value,
                )
            };
        }
        crate::module_ns_store(
            ns,
            "pathconf_names",
            pyre_object::gc_roots::shadow_stack_get(names_slot),
        );

        /// `posixmodule.c conv_path_confname`: an `int` passes through, a
        /// `str` is resolved through `pathconf_names`.
        fn confname_arg(w: PyObjectRef) -> Result<i32, crate::PyError> {
            if unsafe { pyre_object::is_str(w) } {
                // A str carrying a lone surrogate has no `&str` view.  It simply
                // matches no known name, which is the ValueError below — not an
                // interpreter abort, which is what reading the value unchecked
                // would produce.
                let name = unsafe { pyre_object::w_str_get_value_opt(w) };
                return name
                    .and_then(|name| {
                        PATHCONF_NAMES
                            .iter()
                            .find(|(known, _)| *known == name)
                            .map(|(_, value)| *value)
                    })
                    .ok_or_else(|| {
                        crate::PyError::value_error("unrecognized configuration name")
                    });
            }
            // `conv_confname` gates on `PyIndex_Check` before converting, so an
            // object that is neither a str nor index-able is this TypeError,
            // while an `__index__` that raises propagates its own exception.
            if !unsafe { crate::builtins::index_check(w) } {
                return Err(crate::PyError::type_error(
                    "configuration names must be strings or integers",
                ));
            }
            let value = crate::baseobjspace::int_w(crate::baseobjspace::space_index(w)?)?;
            Ok(value as i32)
        }

        // os.pathconf(path, name) -> int | None
        crate::module_ns_store(
            ns,
            "pathconf",
            crate::make_builtin_function_with_arity(
                "pathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("pathconf() requires path, name"));
                    }
                    let path = crate::gateway::fsencode_path_w(args[0])?;
                    let cpath = std::ffi::CString::new(path.as_bytes.as_slice()).map_err(|_| {
                        crate::PyError::value_error("pathconf: embedded null in path")
                    })?;
                    let name = confname_arg(args[1])?;
                    match host_posix::pathconf(&cpath, name)
                        .map_err(|e| io_err_with_filename(e, path.w_path()))?
                    {
                        Some(v) => Ok(pyre_object::w_int_new(v as i64)),
                        None => Ok(pyre_object::w_none()),
                    }
                },
                2,
            ),
        );

        // os.fpathconf(fd, name) -> int | None
        crate::module_ns_store(
            ns,
            "fpathconf",
            crate::make_builtin_function_with_arity(
                "fpathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fpathconf() requires fd, name"));
                    }
                    // interp_posix.py:2411 `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let name = confname_arg(args[1])?;
                    match host_posix::fpathconf(fd, name).map_err(|e| io_err(e, ""))? {
                        Some(v) => Ok(pyre_object::w_int_new(v as i64)),
                        None => Ok(pyre_object::w_none()),
                    }
                },
                2,
            ),
        );

        // os.sysconf(name) -> int
        crate::module_ns_store(
            ns,
            "sysconf",
            crate::make_builtin_function_with_arity(
                "sysconf",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("sysconf() requires name"));
                    }
                    // interp_posix.py:2388-2397 confname_w: symbolic names
                    // resolve through the same dictionary exported as
                    // `sysconf_names`; every other value follows space.int_w.
                    let name = if unsafe { pyre_object::is_str(args[0]) } {
                        let key = crate::baseobjspace::text_w(args[0])?;
                        sysconf_names()
                            .iter()
                            .find_map(|(name, value)| (*name == key).then_some(*value))
                            .ok_or_else(|| {
                                crate::PyError::value_error(
                                    "unrecognized configuration name",
                                )
                            })?
                    } else {
                        crate::baseobjspace::int_w(args[0])? as i32
                    };
                    let v = host_posix::sysconf(name).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(v as i64))
                },
                1,
            ),
        );
        let w_sysconf_names = pyre_object::w_dict_new();
        let _sysconf_names_root = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_sysconf_names);
        let names_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        for (name, value) in sysconf_names() {
            // Build the value first: call arguments evaluate left to right, so
            // reading the rooted dict inline would take its address before
            // `w_int_new` allocates, and a collection there would leave the
            // store writing through the pre-move address.
            let w_value = pyre_object::w_int_new(*value as i64);
            unsafe {
                pyre_object::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(names_slot),
                    name,
                    w_value,
                );
            }
        }
        crate::module_ns_store(
            ns,
            "sysconf_names",
            pyre_object::gc_roots::shadow_stack_get(names_slot),
        );

        // os.initgroups(username, gid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::module_ns_store(
            ns,
            "initgroups",
            crate::make_builtin_function_with_arity(
                "initgroups",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "initgroups() requires username, gid",
                        ));
                    }
                    let user = unsafe {
                        if pyre_object::is_str(args[0]) {
                            crate::baseobjspace::str_utf8_w(args[0])?.to_string()
                        } else {
                            return Err(crate::PyError::type_error(
                                "initgroups(): username must be str",
                            ));
                        }
                    };
                    let cuser = std::ffi::CString::new(user.as_bytes()).map_err(|_| {
                        crate::PyError::value_error("initgroups: embedded null in username")
                    })?;
                    // interp_posix.py:2137 `@unwrap_spec(username='text', gid=c_gid_t)`.
                    let gid = crate::baseobjspace::c_uid_t_w(args[1])?;
                    host_posix::initgroups(&cuser, gid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.openpty() -> (master_fd, slave_fd)
        crate::module_ns_store(
            ns,
            "openpty",
            crate::make_builtin_function_with_arity(
                "openpty",
                |_| {
                    use std::os::fd::IntoRawFd;
                    let (master, slave) = host_posix::openpty().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(master.into_raw_fd() as i64),
                        pyre_object::w_int_new(slave.into_raw_fd() as i64),
                    ]))
                },
                0,
            ),
        );

        // os.getresuid() -> (ruid, euid, suid)
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "openbsd"))]
        crate::module_ns_store(
            ns,
            "getresuid",
            crate::make_builtin_function_with_arity(
                "getresuid",
                |_| {
                    let (r, e, s) = host_posix::getresuid().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(r as i64),
                        pyre_object::w_int_new(e as i64),
                        pyre_object::w_int_new(s as i64),
                    ]))
                },
                0,
            ),
        );

        // os.getresgid() -> (rgid, egid, sgid)
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "openbsd"))]
        crate::module_ns_store(
            ns,
            "getresgid",
            crate::make_builtin_function_with_arity(
                "getresgid",
                |_| {
                    let (r, e, s) = host_posix::getresgid().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(r as i64),
                        pyre_object::w_int_new(e as i64),
                        pyre_object::w_int_new(s as i64),
                    ]))
                },
                0,
            ),
        );

        // os.setresuid(ruid, euid, suid) -> None
        #[cfg(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "openbsd"
        ))]
        crate::module_ns_store(
            ns,
            "setresuid",
            crate::make_builtin_function_with_arity(
                "setresuid",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setresuid() requires ruid, euid, suid",
                        ));
                    }
                    // interp_posix.py:2318 `@unwrap_spec(ruid=c_uid_t,
                    // euid=c_uid_t, suid=c_uid_t)`.
                    let r = crate::baseobjspace::c_uid_t_w(args[0])?;
                    let e = crate::baseobjspace::c_uid_t_w(args[1])?;
                    let s = crate::baseobjspace::c_uid_t_w(args[2])?;
                    host_posix::setresuid(r, e, s).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.setresgid(rgid, egid, sgid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::module_ns_store(
            ns,
            "setresgid",
            crate::make_builtin_function_with_arity(
                "setresgid",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setresgid() requires rgid, egid, sgid",
                        ));
                    }
                    // interp_posix.py:2329 `@unwrap_spec(rgid=c_gid_t,
                    // egid=c_gid_t, sgid=c_gid_t)`.
                    let r = crate::baseobjspace::c_uid_t_w(args[0])?;
                    let e = crate::baseobjspace::c_uid_t_w(args[1])?;
                    let s = crate::baseobjspace::c_uid_t_w(args[2])?;
                    host_posix::setresgid(r, e, s).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );
    }

    // The trampoline only mediates the curated ll_os/ll_time surface, so the
    // real impls registered above for process control, fd duplication, host
    // filesystem mutation and privilege changes would otherwise reach libc
    // directly under sandbox.  Overwrite each with a raising stub, mirroring the
    // RPython sandbox where unsupported externals are simply unavailable.  The
    // mediated names (open/read/write/close/lseek/stat/access/getcwd/listdir/
    // getenv/isatty/strerror/get{u,g}id/unlink/mkdir) are intentionally absent
    // here — they stay live through host_seam.
    #[cfg(feature = "sandbox")]
    {
        fn sandbox_unavailable(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::host_seam::stub("this OS operation"))
        }
        for name in [
            // process creation / control
            "fork",
            "forkpty",
            "system",
            "popen",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            "spawnv",
            "spawnve",
            "spawnvp",
            "spawnvpe",
            "posix_spawn",
            "posix_spawnp",
            "abort",
            "_exit",
            "register_at_fork",
            "wait",
            "waitpid",
            "kill",
            "killpg",
            // file-descriptor duplication / pipes / ttys / cross-fd copy +
            // inheritance control (set_inheritable would mutate a real fd).
            "dup",
            "dup2",
            "dup3",
            "pipe",
            "pipe2",
            "openpty",
            "login_tty",
            "sendfile",
            "set_inheritable",
            // host filesystem mutation that bypasses the controller
            "chmod",
            "fchmod",
            "lchmod",
            "chown",
            "fchown",
            "lchown",
            "chroot",
            "chdir",
            "fchdir",
            "link",
            "symlink",
            "truncate",
            "ftruncate",
            "rename",
            "replace",
            "rmdir",
            "mkfifo",
            "mknod",
            // privilege / scheduling
            "setuid",
            "setgid",
            "setreuid",
            "setregid",
            "setresuid",
            "setresgid",
            "setgroups",
            "initgroups",
            "setsid",
            "setpgid",
            "setpgrp",
            "nice",
            "setpriority",
            "sched_get_priority_max",
            "sched_get_priority_min",
            // durability + real process environment mutation
            "sync",
            "fsync",
            "fdatasync",
            "setenv",
            "unsetenv",
            "putenv",
            // host filesystem inspection that bypasses the controller VFS.
            // DirEntry is a type, but its is_dir/is_file/stat/inode methods stat
            // a guest-controlled `path` via host_fs, so neutralise it too (its
            // only producer, scandir, is already stubbed here).
            "readlink",
            "scandir",
            "DirEntry",
            "statvfs",
            "fstatvfs",
            // host process / environment information leaks
            "getpid",
            "getppid",
            "uname",
            "getlogin",
            "getloadavg",
            "getpriority",
            "times",
            "umask",
            "getgroups",
            "cpu_count",
            "_cpu_count",
            "getresuid",
            "getresgid",
            // host system-configuration probes; pathconf consults a
            // guest-controlled path on the real filesystem.
            "pathconf",
            "fpathconf",
            "sysconf",
            // terminal / tty inspection + control
            "tcgetpgrp",
            "tcsetpgrp",
            "get_terminal_size",
            "ttyname",
        ] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function(name, sandbox_unavailable),
            );
        }
    }

    crate::module_ns_store(ns, "error", crate::typedef::w_object());
}

#[cfg(test)]
mod split_root_tests {
    use super::split_root;

    /// Expectations taken from `ntpath.splitroot`, whose three-way split
    /// joins drive and root into the root this returns.
    #[test]
    fn matches_ntpath_splitroot() {
        let cases: &[(&str, &str, &str)] = &[
            ("", "", ""),
            ("Windows", "", "Windows"),
            ("a/b", "", "a/b"),
            ("\\", "\\", ""),
            ("/", "/", ""),
            ("\\Windows", "\\", "Windows"),
            ("/Windows", "/", "Windows"),
            ("C:", "C:", ""),
            ("C:a", "C:", "a"),
            ("C:\\", "C:\\", ""),
            ("C:/Windows", "C:/", "Windows"),
            (":a", "", ":a"),
            ("\\\\server", "\\\\server", ""),
            ("\\\\server\\share", "\\\\server\\share", ""),
            ("\\\\server\\share\\", "\\\\server\\share\\", ""),
            ("\\\\server\\share\\dir", "\\\\server\\share\\", "dir"),
            ("//server/share/dir", "//server/share/", "dir"),
            ("\\\\.\\device\\x", "\\\\.\\device\\", "x"),
            ("\\\\?\\C:\\x", "\\\\?\\C:\\", "x"),
            (
                "\\\\?\\UNC\\server\\share\\dir",
                "\\\\?\\UNC\\server\\share\\",
                "dir",
            ),
            ("//?/unc/server/share/dir", "//?/unc/server/share/", "dir"),
            // No separator after the share name, so the whole path is root.
            ("\\\\\\a", "\\\\\\a", ""),
            // Offsets address characters: the drive branch is taken here.
            ("\u{e4}:\\x", "\u{e4}:\\", "x"),
            (
                "\\\\s\u{e4}rver\\share\\dir",
                "\\\\s\u{e4}rver\\share\\",
                "dir",
            ),
        ];
        for &(path, root, tail) in cases {
            assert_eq!(split_root(path), (root, tail), "split_root({path:?})");
            assert_eq!(
                format!("{root}{tail}"),
                path,
                "split_root({path:?}) lost characters"
            );
        }
    }
}
