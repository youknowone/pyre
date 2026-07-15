//! fcntl implementation — PyPy: pypy/module/fcntl/interp_fcntl.py
//!
//! Verbatim move of the inline block previously in importing.rs.


/// fcntl module — PyPy: pypy/module/fcntl/interp_fcntl.py.
///
/// fcntl(fd, cmd, arg=0) / ioctl(fd, request, arg=0) / flock(fd, op) /
/// lockf(fd, cmd, len=0, start=0, whence=0).  Backed by
/// `rustpython_host_env::fcntl`.  Only the integer-argument forms are
/// implemented; bytes-buffer (out-arg) variants are out of scope.
pub fn register_module(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(
        ns,
        "fcntl",
        crate::make_builtin_function("fcntl", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                if !(2..=3).contains(&args.len()) {
                    return Err(crate::PyError::type_error(
                        "fcntl() takes 2 or 3 arguments",
                    ));
                }
                if !unsafe { pyre_object::is_int(args[0]) }
                    || !unsafe { pyre_object::is_int(args[1]) }
                    || (args.len() >= 3 && !unsafe { pyre_object::is_int(args[2]) })
                {
                    return Err(crate::PyError::type_error(
                        "fcntl() arguments must be integers",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let cmd = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                let arg = if args.len() >= 3 {
                    unsafe { pyre_object::w_int_get_value(args[2]) as i32 }
                } else {
                    0
                };
                match rustpython_host_env::fcntl::fcntl_int(fd, cmd, arg) {
                    Ok(v) => Ok(pyre_object::w_int_new(v as i64)),
                    Err(e) => Err(crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("fcntl: {e}"),
                    )),
                }
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "fcntl.fcntl requires host_env feature",
                ))
            }
        }),
    );
    crate::module_ns_store(
        ns,
        "ioctl",
        crate::make_builtin_function("ioctl", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                if !(2..=3).contains(&args.len()) {
                    return Err(crate::PyError::type_error(
                        "ioctl() takes 2 or 3 arguments",
                    ));
                }
                if !unsafe { pyre_object::is_int(args[0]) }
                    || !unsafe { pyre_object::is_int(args[1]) }
                    || (args.len() >= 3 && !unsafe { pyre_object::is_int(args[2]) })
                {
                    return Err(crate::PyError::type_error(
                        "ioctl() arguments must be integers",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let raw_req = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i64;
                let request = rustpython_host_env::fcntl::normalize_ioctl_request(raw_req);
                let arg = if args.len() >= 3 {
                    unsafe { pyre_object::w_int_get_value(args[2]) as i32 }
                } else {
                    0
                };
                match rustpython_host_env::fcntl::ioctl_int(fd, request, arg) {
                    Ok(v) => Ok(pyre_object::w_int_new(v as i64)),
                    Err(e) => Err(crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("ioctl: {e}"),
                    )),
                }
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "fcntl.ioctl requires host_env feature",
                ))
            }
        }),
    );
    crate::module_ns_store(
        ns,
        "flock",
        crate::make_builtin_function_with_arity(
            "flock",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("flock() requires 2 arguments"));
                    }
                    if !unsafe { pyre_object::is_int(args[0]) }
                        || !unsafe { pyre_object::is_int(args[1]) }
                    {
                        return Err(crate::PyError::type_error(
                            "flock() arguments must be integers",
                        ));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let op = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    match rustpython_host_env::fcntl::flock(fd, op) {
                        Ok(_) => Ok(pyre_object::w_none()),
                        Err(e) => Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("flock: {e}"),
                        )),
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "fcntl.flock requires host_env feature",
                    ))
                }
            },
            2,
        ),
    );
    crate::module_ns_store(
        ns,
        "lockf",
        crate::make_builtin_function("lockf", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                if !(2..=5).contains(&args.len()) {
                    return Err(crate::PyError::type_error(
                        "lockf() takes from 2 to 5 arguments",
                    ));
                }
                for (i, &a) in args.iter().enumerate().take(5) {
                    if !unsafe { pyre_object::is_int(a) } {
                        let _ = i;
                        return Err(crate::PyError::type_error(
                            "lockf() arguments must be integers",
                        ));
                    }
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let cmd = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                let len = if args.len() >= 3 {
                    unsafe { pyre_object::w_int_get_value(args[2]) }
                } else {
                    0
                };
                let start = if args.len() >= 4 {
                    unsafe { pyre_object::w_int_get_value(args[3]) }
                } else {
                    0
                };
                let whence = if args.len() >= 5 {
                    unsafe { pyre_object::w_int_get_value(args[4]) as i32 }
                } else {
                    0
                };
                match rustpython_host_env::fcntl::lockf(fd, cmd, len, start, whence) {
                    // `interp_fcntl.py:226 fcntl_lockf` returns
                    // space.w_None; the integer return value of the C
                    // helper was an internal pyre detail.
                    Ok(_) => Ok(pyre_object::w_none()),
                    Err(rustpython_host_env::fcntl::LockfError::InvalidCmd) => {
                        Err(crate::PyError::value_error("lockf: invalid cmd"))
                    }
                    Err(rustpython_host_env::fcntl::LockfError::Overflow(s)) => {
                        Err(crate::PyError::value_error(format!("lockf: overflow: {s}")))
                    }
                    Err(rustpython_host_env::fcntl::LockfError::Io(e)) => {
                        Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("lockf: {e}"),
                        ))
                    }
                }
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "fcntl.lockf requires host_env feature",
                ))
            }
        }),
    );
    // `interp_fcntl.py:25-37 constant_names` — POSIX subset always
    // exposed; Linux-specific block gated below.  I_* (System V
    // STREAMS) are listed by PyPy but `if value is not None` filters
    // them out at platform.configure time on every supported platform;
    // not exposed here.
    #[cfg(unix)]
    {
        macro_rules! cst {
            ($name:literal, $val:expr) => {
                crate::module_ns_store(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        cst!("F_GETFD", libc::F_GETFD);
        cst!("F_SETFD", libc::F_SETFD);
        cst!("F_GETFL", libc::F_GETFL);
        cst!("F_SETFL", libc::F_SETFL);
        cst!("F_DUPFD", libc::F_DUPFD);
        cst!("F_DUPFD_CLOEXEC", libc::F_DUPFD_CLOEXEC);
        cst!("F_GETLK", libc::F_GETLK);
        cst!("F_SETLK", libc::F_SETLK);
        cst!("F_SETLKW", libc::F_SETLKW);
        cst!("F_GETOWN", libc::F_GETOWN);
        cst!("F_SETOWN", libc::F_SETOWN);
        cst!("F_RDLCK", libc::F_RDLCK);
        cst!("F_WRLCK", libc::F_WRLCK);
        cst!("F_UNLCK", libc::F_UNLCK);
        cst!("FD_CLOEXEC", libc::FD_CLOEXEC);
        cst!("LOCK_SH", libc::LOCK_SH);
        cst!("LOCK_EX", libc::LOCK_EX);
        cst!("LOCK_UN", libc::LOCK_UN);
        cst!("LOCK_NB", libc::LOCK_NB);

        // Linux-only fcntl constants.  Values for ones libc does not
        // expose (F_GETSIG/F_SETSIG/F_GETLK64/F_SETLK64/F_SETLKW64/
        // F_EXLCK/F_SHLCK/LOCK_MAND/LOCK_READ/LOCK_WRITE/LOCK_RW/DN_*)
        // come straight from Linux <fcntl.h>, matching the hardcoded
        // overrides at `interp_fcntl.py:48-52`.
        #[cfg(target_os = "linux")]
        {
            cst!("F_SETLEASE", libc::F_SETLEASE);
            cst!("F_GETLEASE", libc::F_GETLEASE);
            cst!("F_NOTIFY", libc::F_NOTIFY);
            cst!("F_GETSIG", 11);
            cst!("F_SETSIG", 10);
            cst!("F_GETLK64", 12);
            cst!("F_SETLK64", 13);
            cst!("F_SETLKW64", 14);
            cst!("F_EXLCK", 4);
            cst!("F_SHLCK", 8);
            cst!("LOCK_MAND", 32);
            cst!("LOCK_READ", 64);
            cst!("LOCK_WRITE", 128);
            cst!("LOCK_RW", 192);
            cst!("DN_ACCESS", 1);
            cst!("DN_MODIFY", 2);
            cst!("DN_CREATE", 4);
            cst!("DN_DELETE", 8);
            cst!("DN_RENAME", 16);
            cst!("DN_ATTRIB", 32);
            cst!("DN_MULTISHOT", 0x80000000u32);
            cst!("F_ADD_SEALS", libc::F_ADD_SEALS);
            cst!("F_GET_SEALS", libc::F_GET_SEALS);
            cst!("F_SEAL_SEAL", libc::F_SEAL_SEAL);
            cst!("F_SEAL_SHRINK", libc::F_SEAL_SHRINK);
            cst!("F_SEAL_GROW", libc::F_SEAL_GROW);
            cst!("F_SEAL_WRITE", libc::F_SEAL_WRITE);
            cst!("F_SETPIPE_SZ", libc::F_SETPIPE_SZ);
            cst!("F_GETPIPE_SZ", libc::F_GETPIPE_SZ);
        }
    }
}
