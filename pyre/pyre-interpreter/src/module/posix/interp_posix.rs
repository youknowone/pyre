//! posix implementation — PyPy: pypy/module/posix/interp_posix.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! shared `stat_result_type` helper is carried in here too; `init_posix`
//! is renamed to `register_module`.

use crate::DictStorage;
use crate::importing::host::{fs as host_fs, os as host_os};
use pyre_object::PyObjectRef;

/// `posix.stat_result` — a real structseq (tuple subclass) so `st[0]`,
/// `len(st)`, iteration and `isinstance(st, tuple)` all work, matching
/// `posixmodule.c` `stat_result_desc`.  The 10 sequence slots hold the
/// integer fields, with the integer-seconds times at 7..10 under the
/// hidden `_integer_atime`/`_integer_mtime`/`_integer_ctime` names; the
/// float `st_atime`/`st_mtime`/`st_ctime`, the `st_*_ns` integers, and the
/// `st_blksize`/`st_blocks`/`st_rdev` block-device fields are named-only
/// extras.
fn stat_result_seq_type() -> PyObjectRef {
    thread_local! {
        static STAT_RESULT_SEQ_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    STAT_RESULT_SEQ_TYPE.with(|c| {
        *c.get_or_init(|| {
            crate::_structseq::make_struct_seq_with_extra(
                // Dotted name → `__name__` "stat_result", repr "os.stat_result(...)".
                "os.stat_result",
                // `app_posix.py:20-37` — slots 7..10 are the hidden integer
                // timestamps; the float `st_atime`/`st_mtime`/`st_ctime` are
                // named-only extras, never indexable.
                &[
                    "st_mode", "st_ino", "st_dev", "st_nlink", "st_uid", "st_gid", "st_size",
                    "_integer_atime", "_integer_mtime", "_integer_ctime",
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
            )
        })
    })
}

/// `os.terminal_size` structseq — `(columns, lines)`.
fn terminal_size_seq_type() -> PyObjectRef {
    thread_local! {
        static T: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    T.with(|c| {
        *c.get_or_init(|| {
            crate::_structseq::make_struct_seq("os.terminal_size", &["columns", "lines"])
        })
    })
}

/// `os.uname_result` structseq — `(sysname, nodename, release, version,
/// machine)`; repr renders "posix.uname_result(...)".
fn uname_result_seq_type() -> PyObjectRef {
    thread_local! {
        static T: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    T.with(|c| {
        *c.get_or_init(|| {
            crate::_structseq::make_struct_seq(
                "posix.uname_result",
                &["sysname", "nodename", "release", "version", "machine"],
            )
        })
    })
}

/// `os.statvfs_result` structseq — 10 sequence slots with `f_fsid` as an
/// extra named field (`n_sequence_fields=10`, `n_fields=11`).
fn statvfs_result_seq_type() -> PyObjectRef {
    thread_local! {
        static T: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    T.with(|c| {
        *c.get_or_init(|| {
            crate::_structseq::make_struct_seq_with_extra(
                "os.statvfs_result",
                &[
                    "f_bsize", "f_frsize", "f_blocks", "f_bfree", "f_bavail", "f_files", "f_ffree",
                    "f_favail", "f_flag", "f_namemax",
                ],
                &["f_fsid"],
            )
        })
    })
}

/// `os.times_result` structseq — `(user, system, children_user,
/// children_system, elapsed)`; repr renders "posix.times_result(...)".
fn times_result_seq_type() -> PyObjectRef {
    thread_local! {
        static T: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    T.with(|c| {
        *c.get_or_init(|| {
            crate::_structseq::make_struct_seq(
                "posix.times_result",
                &[
                    "user",
                    "system",
                    "children_user",
                    "children_system",
                    "elapsed",
                ],
            )
        })
    })
}

/// posix stub — PyPy: pypy/module/posix/ interp_posix.py
///
/// Provides the minimal surface that os.py module init needs to succeed.
/// Real posix calls are not implemented — they raise or return defaults.
pub fn register_module(ns: &mut DictStorage) {
    // environ — dict populated from the host environment.
    // PyPy equivalent: posix.State.startup → _convertenviron copies
    // os.environ.items() into w_environ at interpreter startup.
    let w_environ = pyre_object::w_dict_new();
    #[cfg(feature = "host_env")]
    {
        // On POSIX, posix.environ stores bytes → bytes. os.py's
        // _create_environ_mapping wraps this dict in an _Environ object that
        // encodes/decodes via surrogateescape when accessed.
        for (key, value) in host_os::vars_os() {
            let k_bytes = key.as_encoded_bytes();
            let v_bytes = value.as_encoded_bytes();
            unsafe {
                pyre_object::w_dict_store(
                    w_environ,
                    pyre_object::w_bytes_from_bytes(k_bytes),
                    pyre_object::w_bytes_from_bytes(v_bytes),
                );
            }
        }
    }
    crate::dict_storage_store(ns, "environ", w_environ);
    // _have_functions — list of HAVE_* macro names that were defined at
    // build time. os.py uses this to populate the supports_* capability sets
    // (supports_dir_fd / supports_fd / supports_follow_symlinks), which
    // callers like shutil.rmtree consult to choose between fd-relative and
    // path-based implementations. Only the macros whose functionality is
    // actually implemented may be listed: the `*at` family is omitted because
    // dir_fd is not honored, and HAVE_FDOPENDIR is omitted because
    // scandir/listdir do not accept a file descriptor. HAVE_LSTAT remains so
    // os.stat is reported in supports_follow_symlinks (follow_symlinks=False
    // works).
    crate::dict_storage_store(
        ns,
        "_have_functions",
        pyre_object::w_list_new(vec![
            pyre_object::w_str_new("HAVE_FCHDIR"),
            pyre_object::w_str_new("HAVE_FCHMOD"),
            pyre_object::w_str_new("HAVE_FCHOWN"),
            pyre_object::w_str_new("HAVE_FEXECVE"),
            pyre_object::w_str_new("HAVE_FPATHCONF"),
            pyre_object::w_str_new("HAVE_FSTATVFS"),
            pyre_object::w_str_new("HAVE_FTRUNCATE"),
            pyre_object::w_str_new("HAVE_FUTIMENS"),
            pyre_object::w_str_new("HAVE_FUTIMES"),
            pyre_object::w_str_new("HAVE_LSTAT"),
        ]),
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
        crate::dict_storage_store(ns, name, pyre_object::w_int_new(val));
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
        crate::dict_storage_store(ns, name, pyre_object::w_int_new(0));
    }
    // Remaining noop stubs — functions os.py references at module level.
    // Functions with real implementations are registered individually below.
    for name in [
        "fstatat",
        "statvfs",
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
        "chown",
        "fchown",
        "lchown",
        "access",
        "faccessat",
        "chflags",
        "lchflags",
        "utime",
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
        "get_blocking",
        "set_blocking",
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
        "pathconf_names",
        "setenv",
        "unsetenv",
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
        "register_at_fork",
        "abort",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
        "popen",
    ] {
        crate::dict_storage_store(
            ns,
            name,
            crate::make_builtin_function(name, |_| Ok(pyre_object::w_none())),
        );
    }

    // PyPy `space.fsencode_w` — promoted to `crate::gateway::fsencode_w`
    // so the `#[pyre_function]` / `#[pyre_methods]` `PyPath` alias and
    // these posix call sites share one extraction path.
    use crate::gateway::fsencode_w as extract_path;

    // ── Helper: convert std::io::Error → PyError (OSError) ──
    fn io_err(e: std::io::Error, path: &str) -> crate::PyError {
        crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("{}: '{}'", e, path),
        )
    }

    // ── posix.open(path, flags, mode=0o777) → fd ──
    crate::dict_storage_store(
        ns,
        "open",
        crate::make_builtin_function("open", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "open() requires at least 2 arguments",
                ));
            }
            let path = extract_path(args[0])?;
            let flags = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
            let mode: u32 = if args.len() >= 3 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as u32
            } else {
                0o777
            };
            let c_path = std::ffi::CString::new(path.as_bytes())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            let fd = unsafe { libc::open(c_path.as_ptr(), flags, mode as libc::c_uint) };
            if fd < 0 {
                return Err(io_err(std::io::Error::last_os_error(), &path));
            }
            Ok(pyre_object::w_int_new(fd as i64))
        }),
    );

    // ── posix.close(fd) ──
    crate::dict_storage_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("close() requires 1 argument"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let ret = unsafe { libc::close(fd) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.read(fd, n) → bytes ──
    crate::dict_storage_store(
        ns,
        "read",
        crate::make_builtin_function_with_arity(
            "read",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("read() requires 2 arguments"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let n = (unsafe { pyre_object::w_int_get_value(args[1]) }) as usize;
                let mut buf = vec![0u8; n];
                let ret = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, n as _) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                buf.truncate(ret as usize);
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            2,
        ),
    );

    // ── posix.write(fd, data) → nbytes ──
    crate::dict_storage_store(
        ns,
        "write",
        crate::make_builtin_function_with_arity(
            "write",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("write() requires 2 arguments"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let data = unsafe {
                    if pyre_object::bytesobject::is_bytes_like(args[1]) {
                        pyre_object::bytesobject::bytes_like_data(args[1]).to_vec()
                    } else if pyre_object::is_str(args[1]) {
                        pyre_object::w_str_get_value(args[1]).as_bytes().to_vec()
                    } else {
                        return Err(crate::PyError::type_error(
                            "write() arg 2 must be bytes-like",
                        ));
                    }
                };
                let ret = unsafe {
                    libc::write(fd, data.as_ptr() as *const libc::c_void, data.len() as _)
                };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(ret as i64))
            },
            2,
        ),
    );

    // ── posix.lseek(fd, offset, whence) → position ──
    crate::dict_storage_store(
        ns,
        "lseek",
        crate::make_builtin_function_with_arity(
            "lseek",
            |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error("lseek() requires 3 arguments"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let offset = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::off_t;
                let whence = (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int;
                let ret = unsafe { libc::lseek(fd, offset, whence) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(ret as i64))
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
        let path = extract_path(args[0])?;
        let c_path = std::ffi::CString::new(path.as_bytes())
            .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
        let ret = unsafe { libc::unlink(c_path.as_ptr()) };
        if ret < 0 {
            return Err(io_err(std::io::Error::last_os_error(), &path));
        }
        Ok(pyre_object::w_none())
    }
    crate::dict_storage_store(
        ns,
        "unlink",
        crate::make_builtin_function_with_arity("unlink", posix_unlink, 1),
    );
    crate::dict_storage_store(
        ns,
        "remove",
        crate::make_builtin_function_with_arity("remove", posix_unlink, 1),
    );

    // ── posix.readlink(path, *, dir_fd=None) ──
    // Returns the symlink target; a non-symlink raises OSError(EINVAL), which
    // `posixpath.realpath` relies on to stop following links.
    crate::dict_storage_store(
        ns,
        "readlink",
        crate::make_builtin_function("readlink", |args| {
            let arg = args
                .first()
                .copied()
                .ok_or_else(|| crate::PyError::type_error("readlink() requires 1 argument"))?;
            let path = extract_path(arg)?;
            match std::fs::read_link(&path) {
                Ok(target) => Ok(pyre_object::w_str_new(&target.to_string_lossy())),
                Err(e) => Err(io_err(e, &path)),
            }
        }),
    );

    // ── posix.mkdir(path, mode=0o777) ──
    crate::dict_storage_store(
        ns,
        "mkdir",
        crate::make_builtin_function("mkdir", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("mkdir() requires 1 argument"));
            }
            let path = extract_path(args[0])?;
            let _mode: u32 = if args.len() >= 2 {
                (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32
            } else {
                0o777
            };
            let c_path = std::ffi::CString::new(path.as_bytes())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            #[cfg(unix)]
            let ret = unsafe { libc::mkdir(c_path.as_ptr(), _mode as libc::mode_t) };
            #[cfg(windows)]
            let ret = unsafe { libc::mkdir(c_path.as_ptr()) };
            if ret < 0 {
                return Err(io_err(std::io::Error::last_os_error(), &path));
            }
            Ok(pyre_object::w_none())
        }),
    );

    // ── posix.rmdir(path) ──
    crate::dict_storage_store(
        ns,
        "rmdir",
        crate::make_builtin_function_with_arity(
            "rmdir",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("rmdir() requires 1 argument"));
                }
                let path = extract_path(args[0])?;
                let c_path = std::ffi::CString::new(path.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let ret = unsafe { libc::rmdir(c_path.as_ptr()) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), &path));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None) ──
    // A non-None `src_dir_fd` / `dst_dir_fd` resolves the path relative to the
    // open directory descriptor (`renameat`); the descriptors are only usable
    // where `renameat` exists (unix).
    crate::dict_storage_store(
        ns,
        "rename",
        crate::make_builtin_function("rename", |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            if pos.len() < 2 {
                return Err(crate::PyError::type_error("rename() requires 2 arguments"));
            }
            if pos.len() > 2 {
                return Err(crate::PyError::type_error(format!(
                    "rename() takes exactly 2 positional arguments ({} given)",
                    pos.len()
                )));
            }
            crate::builtins::kwarg_reject_unknown(
                kwargs,
                &["src_dir_fd", "dst_dir_fd"],
                "rename",
            )?;
            let src = extract_path(pos[0])?;
            let dst = extract_path(pos[1])?;
            let dir_fd = |name: &str| -> Result<Option<i32>, crate::PyError> {
                match crate::builtins::kwarg_get(kwargs, name) {
                    Some(v) if !unsafe { pyre_object::is_none(v) } => {
                        if !unsafe { pyre_object::is_int(v) } {
                            let type_name = crate::typedef::r#type(v)
                                .map(|t| unsafe { pyre_object::typeobject::w_type_get_name(t) })
                                .unwrap_or("object");
                            return Err(crate::PyError::type_error(format!(
                                "argument should be integer or None, not {type_name}"
                            )));
                        }
                        Ok(Some((unsafe { pyre_object::w_int_get_value(v) }) as i32))
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
            host_os::rename(&src, src_b, &dst, dst_b).map_err(|e| io_err(e, &src))?;
            Ok(pyre_object::w_none())
        }),
    );

    // ── posix.listdir(path=".") → list of str ──
    crate::dict_storage_store(
        ns,
        "listdir",
        crate::make_builtin_function("listdir", |args| {
            let path = if args.is_empty() || unsafe { pyre_object::is_none(args[0]) } {
                ".".to_string()
            } else {
                extract_path(args[0])?
            };
            let entries = host_fs::read_dir(&path).map_err(|e| io_err(e, &path))?;
            let mut items = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| io_err(e, &path))?;
                let name = entry.file_name();
                items.push(pyre_object::w_str_new(&name.to_string_lossy()));
            }
            Ok(pyre_object::w_list_new(items))
        }),
    );

    // ── posix.isatty(fd) → bool ──
    crate::dict_storage_store(
        ns,
        "isatty",
        crate::make_builtin_function_with_arity(
            "isatty",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                Ok(pyre_object::w_bool_from(host_os::isatty(fd)))
            },
            1,
        ),
    );

    // ── posix.urandom(n) → bytes ──
    crate::dict_storage_store(
        ns,
        "urandom",
        crate::make_builtin_function_with_arity(
            "urandom",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("urandom() requires 1 argument"));
                }
                let n = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                let buf = host_os::urandom(n).unwrap_or_else(|_| vec![0u8; n]);
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
    crate::dict_storage_store(ns, "terminal_size", terminal_size_seq_type());

    // ── posix.get_terminal_size(fd=1) → os.terminal_size(columns, lines) ──
    crate::dict_storage_store(
        ns,
        "get_terminal_size",
        crate::make_builtin_function_with_arity(
            "get_terminal_size",
            |_args| {
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
            },
            0,
        ),
    );
    // os.fspath() — posixmodule.c posix_fspath / PyOS_FSPath.  str/bytes
    // pass through unchanged (the protocol's identity case); any other
    // object is resolved via `type(path).__fspath__(path)`.
    crate::dict_storage_store(
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
                        unsafe { crate::baseobjspace::lookup_in_type(pt, "__fspath__") }
                    {
                        return crate::call::call_function_impl_result(fspath_fn, &[arg]);
                    }
                }
                let type_name = match path_type {
                    Some(pt) => unsafe { pyre_object::typeobject::w_type_get_name(pt) },
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
    #[cfg(target_os = "macos")]
    fn macos_path_st_flags(path: &str, follow: bool) -> u32 {
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
            if rc == 0 {
                st.st_flags
            } else {
                0
            }
        }
    }
    #[cfg(target_os = "macos")]
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
            (meta.blksize() as i64, meta.blocks() as i64, meta.rdev() as i64)
        };

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
    fn stat_impl(
        args: &[pyre_object::PyObjectRef],
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("stat() missing argument"));
        }
        let path_obj = args[0];
        let path_str = crate::gateway::fsencode_w(path_obj).map_err(|_| {
            crate::PyError::type_error("stat: path should be string, bytes, os.PathLike")
        })?;
        let meta = if follow_symlinks {
            host_fs::metadata(&path_str)
        } else {
            host_fs::symlink_metadata(&path_str)
        };
        match meta {
            Ok(m) => {
                #[cfg(target_os = "macos")]
                let st_flags = macos_path_st_flags(&path_str, follow_symlinks);
                #[cfg(not(target_os = "macos"))]
                let st_flags = 0u32;
                Ok(make_stat_result(&m, st_flags))
            }
            Err(e) => {
                let kind = e.raw_os_error().unwrap_or(2);
                Err(crate::PyError::os_error_with_errno(
                    kind,
                    format!("{}: '{}'", e, path_str),
                ))
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

    fn dir_entry_path(self_obj: PyObjectRef) -> Result<String, crate::PyError> {
        let p = crate::baseobjspace::getattr_str(self_obj, "path")?;
        Ok(unsafe { pyre_object::w_str_get_value(p) }.to_string())
    }
    fn dir_entry_follow(args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
        let (_pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v),
            None => Ok(true),
        }
    }
    fn dir_entry_is_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let path = dir_entry_path(args[0])?;
        let follow = dir_entry_follow(args)?;
        let meta = if follow {
            host_fs::metadata(&path)
        } else {
            host_fs::symlink_metadata(&path)
        };
        Ok(pyre_object::w_bool_from(meta.map(|m| m.is_dir()).unwrap_or(false)))
    }
    fn dir_entry_is_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let path = dir_entry_path(args[0])?;
        let follow = dir_entry_follow(args)?;
        let meta = if follow {
            host_fs::metadata(&path)
        } else {
            host_fs::symlink_metadata(&path)
        };
        Ok(pyre_object::w_bool_from(
            meta.map(|m| m.is_file()).unwrap_or(false),
        ))
    }
    fn dir_entry_is_symlink(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let path = dir_entry_path(args[0])?;
        Ok(pyre_object::w_bool_from(
            host_fs::symlink_metadata(&path)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false),
        ))
    }
    fn dir_entry_is_junction(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // POSIX has no junction points.
        Ok(pyre_object::w_bool_from(false))
    }
    fn dir_entry_inode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let path = dir_entry_path(args[0])?;
        let meta = host_fs::symlink_metadata(&path).map_err(|e| io_err(e, &path))?;
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
        let path = dir_entry_path(args[0])?;
        let follow = dir_entry_follow(args)?;
        let meta = if follow {
            host_fs::metadata(&path)
        } else {
            host_fs::symlink_metadata(&path)
        };
        match meta {
            Ok(m) => {
                #[cfg(target_os = "macos")]
                let st_flags = macos_path_st_flags(&path, follow);
                #[cfg(not(target_os = "macos"))]
                let st_flags = 0u32;
                Ok(make_stat_result(&m, st_flags))
            }
            Err(e) => Err(io_err(e, &path)),
        }
    }
    fn dir_entry_fspath(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        crate::baseobjspace::getattr_str(args[0], "path")
    }
    fn dir_entry_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let name = crate::baseobjspace::getattr_str(args[0], "name")?;
        let name = unsafe { pyre_object::w_str_get_value(name) };
        Ok(pyre_object::w_str_new(&format!("<DirEntry {name:?}>")))
    }
    fn dir_entry_type() -> PyObjectRef {
        thread_local! {
            static CELL: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
        }
        CELL.with(|c| {
            *c.get_or_init(|| {
                let tp = crate::typedef::make_builtin_type("DirEntry", |ns| {
                    for (name, f) in [
                        ("is_dir", dir_entry_is_dir as crate::gateway::BuiltinCodeFn),
                        ("is_file", dir_entry_is_file),
                        ("is_symlink", dir_entry_is_symlink),
                        ("is_junction", dir_entry_is_junction),
                        ("inode", dir_entry_inode),
                        ("stat", dir_entry_stat),
                        ("__fspath__", dir_entry_fspath),
                        ("__repr__", dir_entry_repr),
                    ] {
                        crate::dict_storage_store(ns, name, crate::make_builtin_function(name, f));
                    }
                });
                unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
                tp
            })
        })
    }

    fn scandir_iter_self(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(args[0])
    }
    fn scandir_iter_close(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(pyre_object::w_none())
    }
    fn scandir_iter_next(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = args[0];
        let idx = unsafe {
            pyre_object::w_int_get_value(crate::baseobjspace::getattr_str(self_obj, "_index")?)
        };
        let entries = crate::baseobjspace::getattr_str(self_obj, "_entries")?;
        let len = unsafe { pyre_object::w_list_len(entries) } as i64;
        if idx >= len {
            return Err(crate::PyError::stop_iteration());
        }
        let item = unsafe { pyre_object::w_list_getitem(entries, idx) }
            .ok_or_else(crate::PyError::stop_iteration)?;
        let _ = crate::baseobjspace::setattr_str(self_obj, "_index", pyre_object::w_int_new(idx + 1));
        Ok(item)
    }
    fn scandir_iter_type() -> PyObjectRef {
        thread_local! {
            static CELL: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
        }
        CELL.with(|c| {
            *c.get_or_init(|| {
                let tp = crate::typedef::make_builtin_type("ScandirIterator", |ns| {
                    for (name, f) in [
                        ("__iter__", scandir_iter_self as crate::gateway::BuiltinCodeFn),
                        ("__next__", scandir_iter_next),
                        ("__enter__", scandir_iter_self),
                        ("__exit__", scandir_iter_close),
                        ("close", scandir_iter_close),
                    ] {
                        crate::dict_storage_store(ns, name, crate::make_builtin_function(name, f));
                    }
                });
                unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
                tp
            })
        })
    }

    fn scandir_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let path = if args.is_empty() || unsafe { pyre_object::is_none(args[0]) } {
            ".".to_string()
        } else {
            crate::gateway::fsencode_w(args[0])?
        };
        let entries = host_fs::read_dir(&path).map_err(|e| io_err(e, &path))?;
        let list = pyre_object::w_list_new(Vec::new());
        for entry in entries {
            let entry = entry.map_err(|e| io_err(e, &path))?;
            let name = entry.file_name().to_string_lossy().to_string();
            let full = entry.path().to_string_lossy().to_string();
            let de = pyre_object::w_instance_new(dir_entry_type());
            let _ = crate::baseobjspace::setattr_str(de, "name", pyre_object::w_str_new(&name));
            let _ = crate::baseobjspace::setattr_str(de, "path", pyre_object::w_str_new(&full));
            unsafe { pyre_object::w_list_append(list, de) };
        }
        let it = pyre_object::w_instance_new(scandir_iter_type());
        let _ = crate::baseobjspace::setattr_str(it, "_entries", list);
        let _ = crate::baseobjspace::setattr_str(it, "_index", pyre_object::w_int_new(0));
        Ok(it)
    }
    crate::dict_storage_store(ns, "scandir", crate::make_builtin_function("scandir", scandir_fn));
    crate::dict_storage_store(ns, "DirEntry", dir_entry_type());

    // os.uname() — returns structseq (sysname, nodename, release, version, machine).
    // Routed through `host_env::posix::uname_info` when available so the
    // result reports the host's real POSIX strings ("Darwin", "Linux",
    // node hostname, kernel release, etc.) instead of Rust's compile-time
    // `std::env::consts::OS` ("macos"/"linux"/...).
    crate::dict_storage_store(
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
    crate::dict_storage_store(
        ns,
        "stat",
        crate::make_builtin_function_with_arity("stat", |args| stat_impl(args, true), 1),
    );
    crate::dict_storage_store(
        ns,
        "lstat",
        crate::make_builtin_function_with_arity("lstat", |args| stat_impl(args, false), 1),
    );
    crate::dict_storage_store(
        ns,
        "fstat",
        crate::make_builtin_function_with_arity(
            "fstat",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("fstat() missing argument"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                #[cfg(unix)]
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
                            e.raw_os_error().unwrap_or(9),
                            format!("{}", e),
                        )),
                    }
                }
                #[cfg(not(unix))]
                Err(crate::PyError::os_error_with_errno(
                    9,
                    "fstat unsupported".to_string(),
                ))
            },
            1,
        ),
    );
    // stat_result type — structseq (tuple subclass). Exported so that
    // `posix.stat_result` and `isinstance(os.stat(p), os.stat_result)` work.
    crate::dict_storage_store(ns, "stat_result", stat_result_seq_type());
    // os.getcwd() — PyPy: posixmodule.c posix_getcwd.
    crate::dict_storage_store(
        ns,
        "getcwd",
        crate::make_builtin_function_with_arity(
            "getcwd",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    if let Ok(cwd) = host_os::current_dir() {
                        return Ok(pyre_object::w_str_new(&cwd.to_string_lossy()));
                    }
                }
                Ok(pyre_object::w_str_new(""))
            },
            0,
        ),
    );
    // os.getcwdb() — bytes form of getcwd.
    crate::dict_storage_store(
        ns,
        "getcwdb",
        crate::make_builtin_function_with_arity(
            "getcwdb",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    if let Ok(cwd) = host_os::current_dir() {
                        return Ok(pyre_object::w_bytes_from_bytes(
                            cwd.as_os_str().as_encoded_bytes(),
                        ));
                    }
                }
                Ok(pyre_object::w_bytes_from_bytes(b""))
            },
            0,
        ),
    );
    // os.getuid / geteuid / getgid / getegid — real syscalls.
    #[cfg(unix)]
    unsafe extern "C" {
        fn getuid() -> u32;
        fn geteuid() -> u32;
        fn getgid() -> u32;
        fn getegid() -> u32;
    }
    crate::dict_storage_store(
        ns,
        "getuid",
        crate::make_builtin_function_with_arity(
            "getuid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getuid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "geteuid",
        crate::make_builtin_function_with_arity(
            "geteuid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(geteuid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getgid",
        crate::make_builtin_function_with_arity(
            "getgid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getgid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getegid",
        crate::make_builtin_function_with_arity(
            "getegid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getegid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    // os.getpid — host_os::process_id (std::process::id).
    crate::dict_storage_store(
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
    crate::dict_storage_store(
        ns,
        "getenv",
        crate::make_builtin_function("getenv", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let key = unsafe {
                if pyre_object::is_str(args[0]) {
                    pyre_object::w_str_get_value(args[0]).to_string()
                } else {
                    return Ok(pyre_object::w_none());
                }
            };
            #[cfg(feature = "host_env")]
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

        // os.strerror(code) -> str
        crate::dict_storage_store(
            ns,
            "strerror",
            crate::make_builtin_function_with_arity(
                "strerror",
                |args| {
                    let code = match args.first() {
                        Some(&o) => (unsafe { pyre_object::w_int_get_value(o) }) as i32,
                        None => {
                            return Err(crate::PyError::type_error("strerror() requires 1 argument"));
                        }
                    };
                    Ok(pyre_object::w_str_new(
                        &rustpython_host_env::time::strerror(code),
                    ))
                },
                1,
            ),
        );

        // os.pipe() -> (r_fd, w_fd)
        crate::dict_storage_store(
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
        crate::dict_storage_store(
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
        crate::dict_storage_store(
            ns,
            "nice",
            crate::make_builtin_function_with_arity(
                "nice",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("nice() requires 1 argument"));
                    }
                    let inc = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let n = host_posix::nice(inc).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );

        // os.umask(mask) -> previous mask
        crate::dict_storage_store(
            ns,
            "umask",
            crate::make_builtin_function_with_arity(
                "umask",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("umask() requires 1 argument"));
                    }
                    let mask = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::mode_t;
                    let prev = host_posix::umask(mask);
                    Ok(pyre_object::w_int_new(prev as i64))
                },
                1,
            ),
        );

        // os.getlogin() -> str
        crate::dict_storage_store(
            ns,
            "getlogin",
            crate::make_builtin_function_with_arity(
                "getlogin",
                |_| match host_posix::getlogin() {
                    Some(name) => Ok(pyre_object::w_str_new(name.to_string_lossy().as_ref())),
                    None => Err(crate::PyError::os_error_with_errno(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                        "getlogin",
                    )),
                },
                0,
            ),
        );

        // os.getgroups() -> list[int]
        crate::dict_storage_store(
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
        crate::dict_storage_store(
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
                    let policy = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let m =
                        host_posix::sched_get_priority_max(policy).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            ),
        );

        // os.sched_get_priority_min(policy) -> int
        crate::dict_storage_store(
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
                    let policy = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let m =
                        host_posix::sched_get_priority_min(policy).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            ),
        );

        // os.sync()
        #[cfg(not(any(target_os = "redox", target_os = "android")))]
        crate::dict_storage_store(
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
        crate::dict_storage_store(
            ns,
            "chdir",
            crate::make_builtin_function_with_arity(
                "chdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chdir() requires 1 argument"));
                    }
                    let path = extract_path(args[0])?;
                    let c_path = std::ffi::CString::new(path.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    host_posix::chdir(&c_path).map_err(|e| {
                        crate::PyError::os_error_with_errno(e as i32, format!("chdir: '{}'", path))
                    })?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fchdir(fd)
        crate::dict_storage_store(
            ns,
            "fchdir",
            crate::make_builtin_function_with_arity(
                "fchdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fchdir() requires 1 argument"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    host_posix::fchdir(fd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fork() -> child pid in parent, 0 in child
        crate::dict_storage_store(
            ns,
            "fork",
            crate::make_builtin_function_with_arity(
                "fork",
                |_| {
                    let pid = host_posix::fork().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(pid as i64))
                },
                0,
            ),
        );

        // os.getppid() -> int
        crate::dict_storage_store(
            ns,
            "getppid",
            crate::make_builtin_function_with_arity(
                "getppid",
                |_| Ok(pyre_object::w_int_new(unsafe { libc::getppid() } as i64)),
                0,
            ),
        );

        // os.waitpid(pid, options) -> (pid, status)
        crate::dict_storage_store(
            ns,
            "waitpid",
            crate::make_builtin_function_with_arity(
                "waitpid",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("waitpid() requires 2 arguments"));
                    }
                    let pid = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::pid_t;
                    let options = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    let mut status: i32 = 0;
                    let res =
                        host_posix::waitpid(pid, &mut status, options).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(res as i64),
                        pyre_object::w_int_new(status as i64),
                    ]))
                },
                2,
            ),
        );

        // os.wait() -> (pid, status)
        crate::dict_storage_store(
            ns,
            "wait",
            crate::make_builtin_function_with_arity(
                "wait",
                |_| {
                    let mut status: i32 = 0;
                    let res =
                        host_posix::waitpid(-1, &mut status, 0).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(res as i64),
                        pyre_object::w_int_new(status as i64),
                    ]))
                },
                0,
            ),
        );

        // os._exit(code) — immediate process exit, no cleanup.
        crate::dict_storage_store(
            ns,
            "_exit",
            crate::make_builtin_function_with_arity(
                "_exit",
                |args| {
                    let code = match args.first() {
                        Some(&o) => (unsafe { pyre_object::w_int_get_value(o) }) as i32,
                        None => return Err(crate::PyError::type_error("_exit() requires 1 argument")),
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
                crate::dict_storage_store(
                    ns,
                    $name,
                    crate::make_builtin_function_with_arity(
                        $name,
                        |args| {
                            let $s = match args.first() {
                                Some(&o) => (unsafe { pyre_object::w_int_get_value(o) }) as libc::c_int,
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
        reg_wstatus!("WIFEXITED", |s| pyre_object::w_bool_from(libc::WIFEXITED(s)));
        reg_wstatus!("WEXITSTATUS", |s| pyre_object::w_int_new(
            libc::WEXITSTATUS(s) as i64
        ));
        reg_wstatus!("WIFSIGNALED", |s| pyre_object::w_bool_from(
            libc::WIFSIGNALED(s)
        ));
        reg_wstatus!("WTERMSIG", |s| pyre_object::w_int_new(libc::WTERMSIG(s) as i64));
        reg_wstatus!("WIFSTOPPED", |s| pyre_object::w_bool_from(
            libc::WIFSTOPPED(s)
        ));
        reg_wstatus!("WSTOPSIG", |s| pyre_object::w_int_new(libc::WSTOPSIG(s) as i64));

        // Wait option flags — override the `0` placeholders registered above
        // with their real libc values (os.WNOHANG must be non-zero for
        // subprocess.poll()).
        crate::dict_storage_store(ns, "WNOHANG", pyre_object::w_int_new(libc::WNOHANG as i64));
        crate::dict_storage_store(
            ns,
            "WUNTRACED",
            pyre_object::w_int_new(libc::WUNTRACED as i64),
        );
        crate::dict_storage_store(
            ns,
            "WCONTINUED",
            pyre_object::w_int_new(libc::WCONTINUED as i64),
        );

        // os.dup(fd) -> new_fd
        crate::dict_storage_store(
            ns,
            "dup",
            crate::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dup() requires 1 argument"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
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
        crate::dict_storage_store(
            ns,
            "dup2",
            crate::make_builtin_function("dup2", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("dup2() requires 2 arguments"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let fd2 = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                let n = unsafe { libc::dup2(fd, fd2) };
                if n < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(n as i64))
            }),
        );

        // os.fsync(fd)
        crate::dict_storage_store(
            ns,
            "fsync",
            crate::make_builtin_function_with_arity(
                "fsync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fsync() requires 1 argument"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    let r = unsafe { libc::fsync(fd) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fdatasync(fd) — falls back to fsync on macOS, which has no
        // fdatasync syscall but exposes the same semantics through fsync.
        crate::dict_storage_store(
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
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    let r = unsafe { libc::fdatasync(fd) };
                    #[cfg(not(any(target_os = "linux", target_os = "android")))]
                    let r = unsafe { libc::fsync(fd) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.mkfifo(path, mode=0o666) -> None
        crate::dict_storage_store(
            ns,
            "mkfifo",
            crate::make_builtin_function("mkfifo", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("mkfifo() requires 1 argument"));
                }
                let path = extract_path(args[0])?;
                let mode = if args.len() >= 2 {
                    (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::mode_t
                } else {
                    0o666
                };
                let c_path = std::ffi::CString::new(path.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let r = unsafe { libc::mkfifo(c_path.as_ptr(), mode) };
                if r < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), &path));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.kill(pid, sig) / os.killpg(pgid, sig)
        crate::dict_storage_store(
            ns,
            "kill",
            crate::make_builtin_function_with_arity(
                "kill",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("kill() requires 2 arguments"));
                    }
                    let pid = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::pid_t;
                    let sig = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                    let r = unsafe { libc::kill(pid, sig) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );
        crate::dict_storage_store(
            ns,
            "killpg",
            crate::make_builtin_function_with_arity(
                "killpg",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("killpg() requires 2 arguments"));
                    }
                    let pgid = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::pid_t;
                    let sig = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
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
        crate::dict_storage_store(
            ns,
            "statvfs",
            crate::make_builtin_function_with_arity(
                "statvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("statvfs() requires 1 argument"));
                    }
                    let path = extract_path(args[0])?;
                    let c_path = std::ffi::CString::new(path.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let info = host_posix::statvfs_path(&c_path).map_err(|e| io_err(e, &path))?;
                    Ok(statvfs_to_obj(info))
                },
                1,
            ),
        );
        #[cfg(not(target_os = "redox"))]
        crate::dict_storage_store(
            ns,
            "fstatvfs",
            crate::make_builtin_function_with_arity(
                "fstatvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fstatvfs() requires 1 argument"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let info = host_posix::statvfs_fd(fd).map_err(|e| io_err(e, ""))?;
                    Ok(statvfs_to_obj(info))
                },
                1,
            ),
        );

        // os.cpu_count() -> int | None
        crate::dict_storage_store(
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
        crate::dict_storage_store(
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
        crate::dict_storage_store(
            ns,
            "symlink",
            crate::make_builtin_function("symlink", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("symlink() requires 2 arguments"));
                }
                let src = extract_path(args[0])?;
                let dst = extract_path(args[1])?;
                let c_src = std::ffi::CString::new(src.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in src"))?;
                let c_dst = std::ffi::CString::new(dst.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in dst"))?;
                // host_env::posix only exposes symlinkat on non-redox unices;
                // call libc::symlink directly so we don't need an at-cwd dance.
                let ret = unsafe { libc::symlink(c_src.as_ptr(), c_dst.as_ptr()) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), &dst));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.fchmod(fd, mode) -> None
        crate::dict_storage_store(
            ns,
            "fchmod",
            crate::make_builtin_function_with_arity(
                "fchmod",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fchmod() requires 2 arguments"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let mode = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::fchmod(bfd, mode).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.fchown(fd, uid, gid) -> None  (uid/gid of -1 means "leave unchanged")
        crate::dict_storage_store(
            ns,
            "fchown",
            crate::make_builtin_function_with_arity(
                "fchown",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("fchown() requires 3 arguments"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let uid_raw = unsafe { pyre_object::w_int_get_value(args[1]) };
                    let gid_raw = unsafe { pyre_object::w_int_get_value(args[2]) };
                    let uid = if uid_raw < 0 {
                        None
                    } else {
                        Some(uid_raw as u32)
                    };
                    let gid = if gid_raw < 0 {
                        None
                    } else {
                        Some(gid_raw as u32)
                    };
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::fchown(bfd, uid, gid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.set_inheritable(fd, inheritable) -> None
        crate::dict_storage_store(
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
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let inherit = unsafe { pyre_object::w_int_get_value(args[1]) } != 0;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::set_inheritable(bfd, inherit).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.access(path, mode) -> bool
        crate::dict_storage_store(
            ns,
            "access",
            crate::make_builtin_function("access", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("access() requires 2 arguments"));
                }
                let path = extract_path(args[0])?;
                let mode = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u8;
                match host_posix::check_access(std::path::Path::new(&path), mode) {
                    Ok(ok) => Ok(pyre_object::w_bool_from(ok)),
                    Err(_) => Ok(pyre_object::w_bool_from(false)),
                }
            }),
        );

        // os.chroot(path) -> None
        crate::dict_storage_store(
            ns,
            "chroot",
            crate::make_builtin_function_with_arity(
                "chroot",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chroot() requires 1 argument"));
                    }
                    let path = extract_path(args[0])?;
                    host_posix::chroot(std::path::Path::new(&path))
                        .map_err(|e| io_err(e, &path))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.getloadavg() -> (1m, 5m, 15m)
        crate::dict_storage_store(
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
        crate::dict_storage_store(
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
        crate::dict_storage_store(
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
                    let status = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
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
        crate::dict_storage_store(
            ns,
            "system",
            crate::make_builtin_function_with_arity(
                "system",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("system() requires 1 argument"));
                    }
                    let cmd = unsafe {
                        if pyre_object::is_str(args[0]) {
                            pyre_object::w_str_get_value(args[0]).to_string()
                        } else {
                            return Err(crate::PyError::type_error(
                                "system(): command must be a string",
                            ));
                        }
                    };
                    let c_cmd = std::ffi::CString::new(cmd.as_bytes())
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
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        crate::dict_storage_store(
            ns,
            "sendfile",
            crate::make_builtin_function("sendfile", |args| {
                use std::os::fd::BorrowedFd;
                if args.len() < 4 {
                    return Err(crate::PyError::type_error(
                        "sendfile() requires 4 arguments",
                    ));
                }
                let out_fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let in_fd = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                let w_offset = args[2];
                let count_raw = unsafe { pyre_object::w_int_get_value(args[3]) };
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
                        let res =
                            unsafe { libc::sendfile(out_fd, in_fd, core::ptr::null_mut(), count) };
                        if res < 0 {
                            return Err(io_err(std::io::Error::last_os_error(), ""));
                        }
                        return Ok(pyre_object::w_int_new(res as i64));
                    }
                }
                let offset_i64 = unsafe { pyre_object::w_int_get_value(w_offset) };
                let out_b = unsafe { BorrowedFd::borrow_raw(out_fd) };
                let in_b = unsafe { BorrowedFd::borrow_raw(in_fd) };
                #[cfg(target_os = "linux")]
                {
                    let count = count_raw as usize;
                    let mut offset: rustpython_host_env::crt_fd::Offset = offset_i64 as _;
                    let n = host_posix::sendfile(out_b, in_b, &mut offset, count)
                        .map_err(|e| io_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(n as i64));
                }
                #[cfg(target_os = "macos")]
                {
                    let (res, written) = host_posix::sendfile(
                        in_b,
                        out_b,
                        offset_i64 as rustpython_host_env::crt_fd::Offset,
                        count_raw,
                        None,
                        None,
                    );
                    res.map_err(|e| io_err(e, ""))?;
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
                let path_str = extract_path(positional[0])?;
                let c_path = std::ffi::CString::new(path_str.as_bytes()).map_err(|_| {
                    crate::PyError::value_error("posix_spawn: embedded null in path")
                })?;
                let argv = collect_cstring_seq(positional[1], "posix_spawn", "argv")?;
                let env = collect_cstring_seq(positional[2], "posix_spawn", "env")?;
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
                let pid = host_posix::posix_spawn(config).map_err(|e| io_err(e, ""))?;
                Ok(pyre_object::w_int_new(pid as i64))
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
                items
                    .into_iter()
                    .map(|s| {
                        let bytes = unsafe {
                            if pyre_object::is_str(s) {
                                pyre_object::w_str_get_value(s).as_bytes().to_vec()
                            } else if pyre_object::is_bytes(s) {
                                pyre_object::w_bytes_data(s).to_vec()
                            } else {
                                return Err(crate::PyError::type_error(format!(
                                    "{fn_name}(): {arg_name} entries must be str or bytes",
                                )));
                            }
                        };
                        std::ffi::CString::new(bytes).map_err(|_| {
                            crate::PyError::value_error(format!(
                                "{fn_name}(): embedded null in {arg_name}",
                            ))
                        })
                    })
                    .collect()
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
                    let op = (unsafe {
                        pyre_object::w_int_get_value(
                            pyre_object::w_tuple_getitem(entry, 0).unwrap(),
                        )
                    }) as i32;
                    match op {
                        0 => {
                            // POSIX_SPAWN_OPEN: (op, fd, path, flags, mode)
                            if tlen < 5 {
                                return Err(crate::PyError::value_error(
                                    "posix_spawn: OPEN action requires fd, path, flags, mode",
                                ));
                            }
                            let fd = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 1).unwrap(),
                                )
                            }) as i32;
                            let path_obj =
                                unsafe { pyre_object::w_tuple_getitem(entry, 2).unwrap() };
                            let path_str = extract_path(path_obj)?;
                            let cpath =
                                std::ffi::CString::new(path_str.as_bytes()).map_err(|_| {
                                    crate::PyError::value_error(
                                        "posix_spawn: embedded null in OPEN path",
                                    )
                                })?;
                            let oflag = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 3).unwrap(),
                                )
                            }) as i32;
                            let mode = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 4).unwrap(),
                                )
                            }) as u32;
                            out.push(PosixSpawnFileAction::Open {
                                fd,
                                path: cpath,
                                oflag,
                                mode,
                            });
                        }
                        1 => {
                            // POSIX_SPAWN_CLOSE: (op, fd)
                            let fd = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 1).unwrap(),
                                )
                            }) as i32;
                            out.push(PosixSpawnFileAction::Close { fd });
                        }
                        2 => {
                            // POSIX_SPAWN_DUP2: (op, fd, newfd)
                            if tlen < 3 {
                                return Err(crate::PyError::value_error(
                                    "posix_spawn: DUP2 action requires fd, newfd",
                                ));
                            }
                            let fd = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 1).unwrap(),
                                )
                            }) as i32;
                            let newfd = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 2).unwrap(),
                                )
                            }) as i32;
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
            crate::dict_storage_store(
                ns,
                "posix_spawn",
                crate::make_builtin_function("posix_spawn", |args| build_posix_spawn(args, false)),
            );
            crate::dict_storage_store(
                ns,
                "posix_spawnp",
                crate::make_builtin_function("posix_spawnp", |args| build_posix_spawn(args, true)),
            );
            crate::dict_storage_store(ns, "POSIX_SPAWN_OPEN", pyre_object::w_int_new(0));
            crate::dict_storage_store(ns, "POSIX_SPAWN_CLOSE", pyre_object::w_int_new(1));
            crate::dict_storage_store(ns, "POSIX_SPAWN_DUP2", pyre_object::w_int_new(2));
        }

        // os.ttyname(fd) -> str
        crate::dict_storage_store(
            ns,
            "ttyname",
            crate::make_builtin_function_with_arity(
                "ttyname",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("ttyname() requires fd"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    let name = host_posix::ttyname(bfd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_str_new(&name.to_string_lossy()))
                },
                1,
            ),
        );

        // os.tcgetpgrp(fd) -> pgid
        crate::dict_storage_store(
            ns,
            "tcgetpgrp",
            crate::make_builtin_function_with_arity(
                "tcgetpgrp",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("tcgetpgrp() requires fd"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    let pgid = host_posix::tcgetpgrp(bfd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(pgid as i64))
                },
                1,
            ),
        );

        // os.tcsetpgrp(fd, pgid) -> None
        crate::dict_storage_store(
            ns,
            "tcsetpgrp",
            crate::make_builtin_function_with_arity(
                "tcsetpgrp",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("tcsetpgrp() requires fd, pgid"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let pgid = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::pid_t;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::tcsetpgrp(bfd, pgid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.getpriority(which, who) -> int
        crate::dict_storage_store(
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
                    let which = (unsafe { pyre_object::w_int_get_value(args[0]) })
                        as host_posix::PriorityWhichType;
                    let who = (unsafe { pyre_object::w_int_get_value(args[1]) })
                        as host_posix::PriorityWhoType;
                    let prio = host_posix::getpriority(which, who).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(prio as i64))
                },
                2,
            ),
        );

        // os.setpriority(which, who, priority) -> None
        crate::dict_storage_store(
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
                    let which = (unsafe { pyre_object::w_int_get_value(args[0]) })
                        as host_posix::PriorityWhichType;
                    let who = (unsafe { pyre_object::w_int_get_value(args[1]) })
                        as host_posix::PriorityWhoType;
                    let prio = (unsafe { pyre_object::w_int_get_value(args[2]) }) as i32;
                    host_posix::setpriority(which, who, prio).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        crate::dict_storage_store(
            ns,
            "PRIO_PROCESS",
            pyre_object::w_int_new(libc::PRIO_PROCESS as i64),
        );
        crate::dict_storage_store(
            ns,
            "PRIO_PGRP",
            pyre_object::w_int_new(libc::PRIO_PGRP as i64),
        );
        crate::dict_storage_store(
            ns,
            "PRIO_USER",
            pyre_object::w_int_new(libc::PRIO_USER as i64),
        );

        // os.pathconf(path, name) -> int | None
        crate::dict_storage_store(
            ns,
            "pathconf",
            crate::make_builtin_function_with_arity(
                "pathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("pathconf() requires path, name"));
                    }
                    let path = extract_path(args[0])?;
                    let cpath = std::ffi::CString::new(path.as_bytes()).map_err(|_| {
                        crate::PyError::value_error("pathconf: embedded null in path")
                    })?;
                    let name = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    match host_posix::pathconf(&cpath, name).map_err(|e| io_err(e, ""))? {
                        Some(v) => Ok(pyre_object::w_int_new(v as i64)),
                        None => Ok(pyre_object::w_none()),
                    }
                },
                2,
            ),
        );

        // os.fpathconf(fd, name) -> int | None
        crate::dict_storage_store(
            ns,
            "fpathconf",
            crate::make_builtin_function_with_arity(
                "fpathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fpathconf() requires fd, name"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let name = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    match host_posix::fpathconf(fd, name).map_err(|e| io_err(e, ""))? {
                        Some(v) => Ok(pyre_object::w_int_new(v as i64)),
                        None => Ok(pyre_object::w_none()),
                    }
                },
                2,
            ),
        );

        // os.sysconf(name) -> int
        crate::dict_storage_store(
            ns,
            "sysconf",
            crate::make_builtin_function_with_arity(
                "sysconf",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("sysconf() requires name"));
                    }
                    let name = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let v = host_posix::sysconf(name).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(v as i64))
                },
                1,
            ),
        );

        // os.initgroups(username, gid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::dict_storage_store(
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
                            pyre_object::w_str_get_value(args[0]).to_string()
                        } else {
                            return Err(crate::PyError::type_error(
                                "initgroups(): username must be str",
                            ));
                        }
                    };
                    let cuser = std::ffi::CString::new(user.as_bytes()).map_err(|_| {
                        crate::PyError::value_error("initgroups: embedded null in username")
                    })?;
                    let gid = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32;
                    host_posix::initgroups(&cuser, gid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.openpty() -> (master_fd, slave_fd)
        crate::dict_storage_store(
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
        crate::dict_storage_store(
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
        crate::dict_storage_store(
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
        crate::dict_storage_store(
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
                    let r = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u32;
                    let e = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32;
                    let s = (unsafe { pyre_object::w_int_get_value(args[2]) }) as u32;
                    host_posix::setresuid(r, e, s).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.setresgid(rgid, egid, sgid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::dict_storage_store(
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
                    let r = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u32;
                    let e = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32;
                    let s = (unsafe { pyre_object::w_int_get_value(args[2]) }) as u32;
                    host_posix::setresgid(r, e, s).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );
    }

    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
}
