//! fcntl implementation — PyPy: pypy/module/fcntl/interp_fcntl.py
//!
//! Verbatim move of the inline block previously in importing.rs.


/// fcntl module — PyPy: pypy/module/fcntl/interp_fcntl.py.
///
/// fcntl(fd, cmd, arg=0) / ioctl(fd, request, arg=0) / flock(fd, op) /
/// lockf(fd, cmd, len=0, start=0, whence=0).  Backed by
/// `rustpython_host_env::fcntl`.  `ioctl` is still limited to the
/// integer-argument form; its buffer form needs writable-buffer acquisition
/// and `mutate_flag` handling from `interp_fcntl.py:252-300`.
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
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(crate::PyError::type_error(
                        "fcntl() arguments must be integers",
                    ));
                }
                // `fcntl(space, w_fd, op, w_arg)` takes its descriptor through
                // `space.c_filedescriptor_w`, so an open file answers for the
                // number it wraps.
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                let cmd = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                // `interp_fcntl.py fcntl` tries the string-buffer path before
                // falling back to the integer one and returns exactly the
                // original buffer's length; `fcntl_fcntl_impl` takes its
                // integer arm first, on `PyIndex_Check`.
                if args.len() >= 3 && !unsafe { pyre_object::is_int(args[2]) } {
                    let data = arg_readbuf(args[2], "fcntl")?;
                    let Some(mut buf) = stage_arg(data) else {
                        return Err(crate::PyError::value_error(
                            "fcntl argument 3 is too long",
                        ));
                    };
                    loop {
                        let outcome = {
                            let _blocked = crate::module::thread::before_external_block();
                            rustpython_host_env::fcntl::fcntl_with_bytes(fd, cmd, &mut buf)
                        };
                        match outcome {
                            Ok(_) => {
                                guard_intact(&buf, data.len())?;
                                return Ok(pyre_object::bytesobject::w_bytes_from_bytes(
                                    &buf[..data.len()],
                                ));
                            }
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| {
                                crate::PyError::os_error_with_errno(
                                    e.raw_os_error().unwrap_or(0),
                                    format!("fcntl: {e}"),
                                )
                            })?,
                        }
                    }
                }
                let arg = if args.len() >= 3 {
                    unsafe { pyre_object::w_int_get_value(args[2]) as i32 }
                } else {
                    0
                };
                // F_SETLKW waits for the lock, so this is a blocking call.
                // `_raise_error_maybe` is `eintr_retry=True`: an interrupted
                // wait runs the pending handlers and goes back to waiting.
                loop {
                    let outcome = {
                        let _blocked = crate::module::thread::before_external_block();
                        rustpython_host_env::fcntl::fcntl_int(fd, cmd, arg)
                    };
                    match outcome {
                        Ok(v) => return Ok(pyre_object::w_int_new(v as i64)),
                        Err(e) => crate::builtins::eintr_retry_with(e, |e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("fcntl: {e}"),
                            )
                        })?,
                    }
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
                // `interp_fcntl.py ioctl(space, w_fd, w_request, w_arg,
                // mutate_flag=-1)` / `fcntl_ioctl_impl(module, fd, code, arg,
                // mutate_arg)`.
                if !(2..=4).contains(&args.len()) {
                    return Err(crate::PyError::type_error(format!(
                        "ioctl expected at most 4 arguments, got {}",
                        args.len()
                    )));
                }
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(crate::PyError::type_error(
                        "ioctl() arguments must be integers",
                    ));
                }
                // `ioctl` reads its descriptor the same way the rest of the
                // module does.  It alone raises through `_raise_error_always`,
                // so an interrupted call surfaces rather than being re-issued.
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                let raw_req = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i64;
                let request = rustpython_host_env::fcntl::normalize_ioctl_request(raw_req);
                // The integer arm comes first, before the argument is ever
                // looked at as a buffer.
                if args.len() >= 3 && !unsafe { pyre_object::is_int(args[2]) } {
                    let arg = args[2];
                    // `mutate_arg` defaults true, and is consulted only for an
                    // exporter that is neither `bytes` nor `str` — those two
                    // always take the read-only form however it is set.
                    let mutate = if args.len() >= 4 {
                        crate::baseobjspace::is_true(args[3])?
                    } else {
                        true
                    };
                    let immutable =
                        unsafe { pyre_object::bytesobject::is_bytes(arg) || pyre_object::is_str(arg) };
                    if mutate
                        && !immutable
                        && let Ok((slice, _owner, _made_view)) =
                            unsafe { crate::builtins::fileio_writebuf(arg) }
                    {
                        return ioctl_mutable(fd, request, slice);
                    }
                    return ioctl_readonly(fd, request, arg_readbuf(arg, "ioctl")?);
                }
                let arg = if args.len() >= 3 {
                    unsafe { pyre_object::w_int_get_value(args[2]) as i32 }
                } else {
                    0
                };
                let outcome = {
                    let _blocked = crate::module::thread::before_external_block();
                    rustpython_host_env::fcntl::ioctl_int(fd, request, arg)
                };
                match outcome {
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
                    if !unsafe { pyre_object::is_int(args[1]) } {
                        return Err(crate::PyError::type_error(
                            "flock() arguments must be integers",
                        ));
                    }
                    // `flock(space, w_fd, op)` unwraps through
                    // `space.c_filedescriptor_w`, so `fcntl.flock(f, LOCK_EX)`
                    // on an open file is the documented spelling.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    let op = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    // Without LOCK_NB this waits for the lock, and
                    // `_raise_error_maybe` is `eintr_retry=True`: the wait runs
                    // the pending handlers and resumes rather than surfacing.
                    loop {
                        let outcome = {
                            let _blocked = crate::module::thread::before_external_block();
                            rustpython_host_env::fcntl::flock(fd, op)
                        };
                        match outcome {
                            Ok(_) => return Ok(pyre_object::w_none()),
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| {
                                crate::PyError::os_error_with_errno(
                                    e.raw_os_error().unwrap_or(0),
                                    format!("flock: {e}"),
                                )
                            })?,
                        }
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
                for &a in args.iter().take(5).skip(1) {
                    if !unsafe { pyre_object::is_int(a) } {
                        return Err(crate::PyError::type_error(
                            "lockf() arguments must be integers",
                        ));
                    }
                }
                // `lockf(space, w_fd, op, length, start, whence)` unwraps its
                // descriptor through `space.c_filedescriptor_w`.
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
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
                // F_LOCK waits for the lock, and `lockf` reports through
                // `_raise_error_maybe`, which is `eintr_retry=True`.
                loop {
                    let outcome = {
                        let _blocked = crate::module::thread::before_external_block();
                        rustpython_host_env::fcntl::lockf(fd, cmd, len, start, whence)
                    };
                    match outcome {
                        // `interp_fcntl.py:226 fcntl_lockf` returns
                        // space.w_None; the integer return value of the C
                        // helper was an internal pyre detail.
                        Ok(_) => return Ok(pyre_object::w_none()),
                        Err(rustpython_host_env::fcntl::LockfError::InvalidCmd) => {
                            return Err(crate::PyError::value_error("lockf: invalid cmd"));
                        }
                        Err(rustpython_host_env::fcntl::LockfError::Overflow(s)) => {
                            return Err(crate::PyError::value_error(format!(
                                "lockf: overflow: {s}"
                            )));
                        }
                        Err(rustpython_host_env::fcntl::LockfError::Io(e)) => {
                            crate::builtins::eintr_retry_with(e, |e| {
                                crate::PyError::os_error_with_errno(
                                    e.raw_os_error().unwrap_or(0),
                                    format!("lockf: {e}"),
                                )
                            })?
                        }
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
    // `interp_fcntl.py constant_names` — POSIX subset always
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
        // The darwin half of the same list.  `F_SETLEASE`/`F_GETLEASE` carry
        // different numbers here than under linux, so they are spelled per
        // platform rather than shared.
        #[cfg(target_vendor = "apple")]
        {
            cst!("FASYNC", 64);
            cst!("F_FULLFSYNC", libc::F_FULLFSYNC);
            cst!("F_GETLEASE", 107);
            cst!("F_SETLEASE", 106);
            cst!("F_GETNOSIGPIPE", 74);
            cst!("F_SETNOSIGPIPE", 73);
            cst!("F_GETPATH", libc::F_GETPATH);
            cst!("F_NOCACHE", libc::F_NOCACHE);
            cst!("F_RDAHEAD", libc::F_RDAHEAD);
            cst!("F_OFD_GETLK", libc::F_OFD_GETLK);
            cst!("F_OFD_SETLK", libc::F_OFD_SETLK);
            cst!("F_OFD_SETLKW", libc::F_OFD_SETLKW);
        }
    }
}

/// `fcntl_ioctl_impl`'s `IOCTL_BUFSZ` / `fcntl_fcntl_impl`'s `FCNTL_BUFSZ` —
/// the staging buffer a copied argument is handed to the kernel in.  Both are
/// 1024.
#[cfg(all(unix, feature = "host_env"))]
const ARG_BUFSZ: usize = 1024;

/// The `guard` both impls write after the staged argument.  A request
/// whose payload is longer than the argument the caller supplied overwrites
/// it, and that is the only way the overrun can be seen at all — so the bytes
/// are the module's, verbatim, starting with the NUL the staged copy is
/// terminated by.
#[cfg(all(unix, feature = "host_env"))]
const ARG_GUARD: [u8; 8] = [0x00, 0xfa, 0x69, 0xc4, 0x67, 0xa3, 0x6c, 0x58];

/// The third argument as `PyArg_Parse(arg, "s*")` reads it: any readable
/// buffer, or a `str`'s UTF-8, which `readbuf_w` alone does not accept.
#[cfg(all(unix, feature = "host_env"))]
fn arg_readbuf(
    arg: pyre_object::PyObjectRef,
    callable: &str,
) -> Result<&'static [u8], crate::PyError> {
    if unsafe { pyre_object::is_str(arg) } {
        return Ok(crate::baseobjspace::str_utf8_w(arg)?.as_bytes());
    }
    unsafe { crate::builtins::acquire_readbuf(arg) }.map_err(|_| {
        let type_name = unsafe { pyre_object::type_name_of(arg) };
        crate::PyError::type_error(format!(
            "{callable}() argument 3 must be an integer, a bytes-like object, \
             or a string, not {type_name}"
        ))
    })
}

/// Run `ioctl` with `arg` as its third argument, reporting the errno on
/// failure.
#[cfg(all(unix, feature = "host_env"))]
fn ioctl_ptr(fd: i32, request: libc::c_ulong, ptr: *mut u8) -> Result<i32, crate::PyError> {
    let ret = {
        let _blocked = crate::module::thread::before_external_block();
        unsafe { libc::ioctl(fd, request, ptr as *mut libc::c_void) }
    };
    if ret < 0 {
        let e = std::io::Error::last_os_error();
        return Err(crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("ioctl: {e}"),
        ));
    }
    Ok(ret)
}

/// A staging buffer holding `arg` followed by the guard, or `None` when `arg`
/// is longer than the staging buffer and has to be handed over as it is.
#[cfg(all(unix, feature = "host_env"))]
fn stage_arg(arg: &[u8]) -> Option<Vec<u8>> {
    if arg.len() > ARG_BUFSZ {
        return None;
    }
    let mut buf = vec![0u8; ARG_BUFSZ + ARG_GUARD.len()];
    buf[..arg.len()].copy_from_slice(arg);
    buf[arg.len()..arg.len() + ARG_GUARD.len()].copy_from_slice(&ARG_GUARD);
    Some(buf)
}

#[cfg(all(unix, feature = "host_env"))]
fn guard_intact(buf: &[u8], len: usize) -> Result<(), crate::PyError> {
    if buf[len..len + ARG_GUARD.len()] == ARG_GUARD {
        return Ok(());
    }
    Err(crate::PyError::system_error("buffer overflow"))
}

/// The writable-exporter arm: the kernel's answer lands back in the caller's
/// own storage and the call returns the syscall's value.  An argument longer
/// than the staging buffer is handed over directly, so there is no guard to
/// check and no length to refuse.
#[cfg(all(unix, feature = "host_env"))]
fn ioctl_mutable(
    fd: i32,
    request: libc::c_ulong,
    arg: &mut [u8],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let Some(mut buf) = stage_arg(arg) else {
        let ret = ioctl_ptr(fd, request, arg.as_mut_ptr())?;
        return Ok(pyre_object::w_int_new(ret as i64));
    };
    let ret = ioctl_ptr(fd, request, buf.as_mut_ptr())?;
    arg.copy_from_slice(&buf[..arg.len()]);
    guard_intact(&buf, arg.len())?;
    Ok(pyre_object::w_int_new(ret as i64))
}

/// The read-only arm: the answer is the staged copy, returned as bytes of the
/// argument's own length.  This one does refuse an over-long argument, because
/// the kernel can only be given the staging buffer.
#[cfg(all(unix, feature = "host_env"))]
fn ioctl_readonly(
    fd: i32,
    request: libc::c_ulong,
    arg: &[u8],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let Some(mut buf) = stage_arg(arg) else {
        return Err(crate::PyError::value_error("ioctl argument 3 is too long"));
    };
    ioctl_ptr(fd, request, buf.as_mut_ptr())?;
    guard_intact(&buf, arg.len())?;
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(
        &buf[..arg.len()],
    ))
}
