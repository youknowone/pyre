//! resource implementation — `lib_pypy/resource.py`.
//!
//! Verbatim move of the inline block previously in importing.rs.

/// `lib_pypy/resource.py class struct_rusage(
/// metaclass=structseqtype)` — process-wide cached subclass-of-tuple
/// type.
static STRUCT_RUSAGE_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

fn struct_rusage_type() -> pyre_object::PyObjectRef {
    STRUCT_RUSAGE_TYPE.get_or_init(|| {
        pyre_interpreter::_structseq::make_struct_seq(
            "resource.struct_rusage",
            &[
                "ru_utime",
                "ru_stime",
                "ru_maxrss",
                "ru_ixrss",
                "ru_idrss",
                "ru_isrss",
                "ru_minflt",
                "ru_majflt",
                "ru_nswap",
                "ru_inblock",
                "ru_oublock",
                "ru_msgsnd",
                "ru_msgrcv",
                "ru_nsignals",
                "ru_nvcsw",
                "ru_nivcsw",
            ],
        )
    })
}

/// `sys/resource.h` `getrlimit` / `setrlimit`. C names `getrlimit` /
/// `setrlimit`, `releasegil=False`, no `save_err`. Args match libc:
/// `(int, *mut rlimit) -> int` and `(int, *const rlimit) -> int`.
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
mod ll {
    use majit_rlib::rffi::INT;

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["sys/resource.h"],
        };
    }

    majit_rlib::rffi::llexternal!(
        pub(super) c_getrlimit = "getrlimit",
        [INT, *mut libc::rlimit],
        INT,
        compilation_info = ECI,
        releasegil = false
    );
    majit_rlib::rffi::llexternal!(
        pub(super) c_setrlimit = "setrlimit",
        [INT, *const libc::rlimit],
        INT,
        compilation_info = ECI,
        releasegil = false
    );
}

/// `resource.getrusage` `struct_rusage` from `rtime.RUSAGE` / `libc::rusage`.
/// `ru_utime` / `ru_stime` are timeval floats, then the 14 integer fields.
/// This is `resource.getrusage`, not `time.clock`.
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
fn make_struct_rusage(r: &majit_rlib::rtime::RUSAGE) -> pyre_object::PyObjectRef {
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::floatobject::w_float_new(
        majit_rlib::rtime::decode_timeval(&r.ru_utime),
    ));
    fields.push(pyre_object::floatobject::w_float_new(
        majit_rlib::rtime::decode_timeval(&r.ru_stime),
    ));
    fields.push(pyre_object::w_int_new(r.ru_maxrss as i64));
    fields.push(pyre_object::w_int_new(r.ru_ixrss as i64));
    fields.push(pyre_object::w_int_new(r.ru_idrss as i64));
    fields.push(pyre_object::w_int_new(r.ru_isrss as i64));
    fields.push(pyre_object::w_int_new(r.ru_minflt as i64));
    fields.push(pyre_object::w_int_new(r.ru_majflt as i64));
    fields.push(pyre_object::w_int_new(r.ru_nswap as i64));
    fields.push(pyre_object::w_int_new(r.ru_inblock as i64));
    fields.push(pyre_object::w_int_new(r.ru_oublock as i64));
    fields.push(pyre_object::w_int_new(r.ru_msgsnd as i64));
    fields.push(pyre_object::w_int_new(r.ru_msgrcv as i64));
    fields.push(pyre_object::w_int_new(r.ru_nsignals as i64));
    fields.push(pyre_object::w_int_new(r.ru_nvcsw as i64));
    fields.push(pyre_object::w_int_new(r.ru_nivcsw as i64));
    pyre_interpreter::_structseq::new_instance(struct_rusage_type(), fields.take())
}

/// Sandbox keeps the `rustpython_host_env::resource` RUsage layout.
#[cfg(all(unix, feature = "host_env", feature = "sandbox"))]
fn make_struct_rusage(r: &rustpython_host_env::resource::RUsage) -> pyre_object::PyObjectRef {
    let tv_to_f = |tv: libc::timeval| tv.tv_sec as f64 + (tv.tv_usec as f64) * 1e-6;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::floatobject::w_float_new(tv_to_f(r.ru_utime)));
    fields.push(pyre_object::floatobject::w_float_new(tv_to_f(r.ru_stime)));
    fields.push(pyre_object::w_int_new(r.ru_maxrss));
    fields.push(pyre_object::w_int_new(r.ru_ixrss));
    fields.push(pyre_object::w_int_new(r.ru_idrss));
    fields.push(pyre_object::w_int_new(r.ru_isrss));
    fields.push(pyre_object::w_int_new(r.ru_minflt));
    fields.push(pyre_object::w_int_new(r.ru_majflt));
    fields.push(pyre_object::w_int_new(r.ru_nswap));
    fields.push(pyre_object::w_int_new(r.ru_inblock));
    fields.push(pyre_object::w_int_new(r.ru_oublock));
    fields.push(pyre_object::w_int_new(r.ru_msgsnd));
    fields.push(pyre_object::w_int_new(r.ru_msgrcv));
    fields.push(pyre_object::w_int_new(r.ru_nsignals));
    fields.push(pyre_object::w_int_new(r.ru_nvcsw));
    fields.push(pyre_object::w_int_new(r.ru_nivcsw));
    pyre_interpreter::_structseq::new_instance(struct_rusage_type(), fields.take())
}

/// resource module — `lib_pypy/resource.py` (PyPy keeps it app-level
/// via `_resource_cffi`).  pyre takes CPython's `Modules/resource.c`
/// shape since pyre has no app-level stdlib.
///
/// Exposes getrusage / getrlimit / setrlimit plus the standard RUSAGE_*
/// and RLIMIT_* constants, the `struct_rusage` type attribute, and the
/// `error = OSError` alias. Unix + `host_env` + not-sandbox calls
/// `rtime.c_getrusage` and `c_getrlimit` / `c_setrlimit`. Sandbox keeps
/// `rustpython_host_env::resource`.
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    // `lib_pypy/resource.py error = OSError` and
    // `:15-37 class struct_rusage`.
    let w_os_error = pyre_interpreter::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before init_resource");
    pyre_interpreter::module_ns_store(ns, "error", w_os_error);
    pyre_interpreter::module_ns_store(ns, "struct_rusage", struct_rusage_type());
    pyre_interpreter::module_ns_store(
        ns,
        "getrusage",
        pyre_interpreter::make_builtin_function_with_arity(
            "getrusage",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let mut w_who = if let Some(&a) = args.first() {
                        if unsafe { pyre_object::is_int(a) } {
                            a
                        } else {
                            return Err(pyre_interpreter::PyError::type_error(
                                "getrusage(): who should be an integer",
                            ));
                        }
                    } else {
                        return Err(pyre_interpreter::PyError::type_error(
                            "getrusage() missing argument",
                        ));
                    };
                    let who = unsafe { pyre_object::w_int_get_value(w_who) as i32 };
                    // `rtime.c_getrusage` (`releasegil=False`, no `save_err`).
                    // This is `resource.getrusage`, not `time.clock`.
                    #[cfg(not(feature = "sandbox"))]
                    {
                        let mut ru =
                            unsafe { std::mem::zeroed::<majit_rlib::rtime::RUSAGE>() };
                        let ret = pyre_object::with_roots!(w_who => unsafe {
                            majit_rlib::rtime::c_getrusage(who, &mut ru)
                        });
                        if ret == -1 {
                            let errno = majit_rlib::rposix::_get_errno();
                            // `lib_pypy/resource.py getrusage` raises ValueError for
                            // an invalid `who`; only other errno values are
                            // surfaced as OSError.
                            if errno == libc::EINVAL {
                                return Err(pyre_interpreter::PyError::value_error(
                                    "invalid who parameter",
                                ));
                            }
                            let e = std::io::Error::from_raw_os_error(errno);
                            Err(pyre_interpreter::PyError::os_error_with_errno(
                                errno,
                                format!("getrusage: {e}"),
                            ))
                        } else {
                            Ok(make_struct_rusage(&ru))
                        }
                    }
                    #[cfg(feature = "sandbox")]
                    {
                        match rustpython_host_env::resource::getrusage(who) {
                            Ok(r) => Ok(make_struct_rusage(&r)),
                            Err(e) => {
                                let errno = e.raw_os_error().unwrap_or(0);
                                // `lib_pypy/resource.py getrusage` raises ValueError for
                                // an invalid `who`; only other errno values are
                                // surfaced as OSError.
                                if errno == libc::EINVAL {
                                    return Err(pyre_interpreter::PyError::value_error(
                                        "invalid who parameter",
                                    ));
                                }
                                Err(pyre_interpreter::PyError::os_error_with_errno(
                                    errno,
                                    format!("getrusage: {e}"),
                                ))
                            }
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(pyre_interpreter::PyError::not_implemented(
                        "resource.getrusage requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    pyre_interpreter::module_ns_store(
        ns,
        "getrlimit",
        pyre_interpreter::make_builtin_function_with_arity(
            "getrlimit",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let mut w_res = if let Some(&a) = args.first() {
                        if unsafe { pyre_object::is_int(a) } {
                            a
                        } else {
                            return Err(pyre_interpreter::PyError::type_error(
                                "getrlimit(): resource should be an integer",
                            ));
                        }
                    } else {
                        return Err(pyre_interpreter::PyError::type_error(
                            "getrlimit() missing argument",
                        ));
                    };
                    let res = unsafe { pyre_object::w_int_get_value(w_res) as libc::rlim_t };
                    #[cfg(not(feature = "sandbox"))]
                    {
                        let mut rl = unsafe { std::mem::zeroed::<libc::rlimit>() };
                        let ret = pyre_object::with_roots!(w_res => unsafe {
                            ll::c_getrlimit(res as majit_rlib::rffi::INT, &mut rl)
                        });
                        if ret == -1 {
                            let errno = majit_rlib::rposix::_get_errno();
                            let e = std::io::Error::from_raw_os_error(errno);
                            Err(pyre_interpreter::PyError::os_error_with_errno(
                                errno,
                                format!("getrlimit: {e}"),
                            ))
                        } else {
                            let mut fields = pyre_object::gc_roots::RootedItems::new();
                            fields.push(pyre_object::w_int_new(rl.rlim_cur as i64));
                            fields.push(pyre_object::w_int_new(rl.rlim_max as i64));
                            Ok(pyre_object::w_tuple_new(fields.take()))
                        }
                    }
                    #[cfg(feature = "sandbox")]
                    {
                        match rustpython_host_env::resource::getrlimit(res) {
                            Ok(rl) => {
                                let mut fields = pyre_object::gc_roots::RootedItems::new();
                                fields.push(pyre_object::w_int_new(rl.rlim_cur as i64));
                                fields.push(pyre_object::w_int_new(rl.rlim_max as i64));
                                Ok(pyre_object::w_tuple_new(fields.take()))
                            }
                            Err(e) => Err(pyre_interpreter::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getrlimit: {e}"),
                            )),
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(pyre_interpreter::PyError::not_implemented(
                        "resource.getrlimit requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    pyre_interpreter::module_ns_store(
        ns,
        "setrlimit",
        pyre_interpreter::make_builtin_function_with_arity(
            "setrlimit",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    if args.len() < 2 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "setrlimit() requires 2 arguments",
                        ));
                    }
                    let mut w_res = args[0];
                    let res = unsafe {
                        if !pyre_object::is_int(w_res) {
                            return Err(pyre_interpreter::PyError::type_error(
                                "setrlimit(): resource should be an integer",
                            ));
                        }
                        pyre_object::w_int_get_value(w_res) as libc::rlim_t
                    };
                    // `lib_pypy/resource.py setrlimit` — `soft, hard = limits;
                    // soft = int(soft); hard = int(hard)`.  Accept any
                    // 2-item tuple or list and coerce each entry to int
                    // (PyPy unpacks via Python iteration; pyre's surface
                    // covers the two concrete sequence shapes callers
                    // actually use).
                    let (mut w_soft, mut w_hard) = unsafe {
                        if pyre_object::is_tuple(args[1]) && pyre_object::w_tuple_len(args[1]) == 2
                        {
                            (
                                pyre_object::w_tuple_getitem(args[1], 0).unwrap(),
                                pyre_object::w_tuple_getitem(args[1], 1).unwrap(),
                            )
                        } else if pyre_object::is_list(args[1])
                            && pyre_object::w_list_len(args[1]) == 2
                        {
                            (
                                pyre_object::w_list_getitem(args[1], 0).unwrap(),
                                pyre_object::w_list_getitem(args[1], 1).unwrap(),
                            )
                        } else {
                            return Err(pyre_interpreter::PyError::type_error(
                                "expected a tuple of 2 integers",
                            ));
                        }
                    };
                    let soft = pyre_object::with_roots!(w_res, w_soft, w_hard => pyre_interpreter::baseobjspace::int_w(w_soft))? as libc::rlim_t;
                    let hard = pyre_object::with_roots!(w_res, w_soft, w_hard => pyre_interpreter::baseobjspace::int_w(w_hard))? as libc::rlim_t;
                    let rl = libc::rlimit {
                        rlim_cur: soft,
                        rlim_max: hard,
                    };
                    #[cfg(not(feature = "sandbox"))]
                    {
                        let ret = pyre_object::with_roots!(w_res, w_soft, w_hard => unsafe {
                            ll::c_setrlimit(res as majit_rlib::rffi::INT, &rl)
                        });
                        if ret == -1 {
                            // `lib_pypy/resource.py setrlimit` — EINVAL and
                            // EPERM both surface as ValueError with
                            // distinct messages; all other errnos stay
                            // as OSError.
                            let errno = majit_rlib::rposix::_get_errno();
                            if errno == libc::EINVAL {
                                return Err(pyre_interpreter::PyError::value_error(
                                    "current limit exceeds maximum limit",
                                ));
                            }
                            if errno == libc::EPERM {
                                return Err(pyre_interpreter::PyError::value_error(
                                    "not allowed to raise maximum limit",
                                ));
                            }
                            let e = std::io::Error::from_raw_os_error(errno);
                            Err(pyre_interpreter::PyError::os_error_with_errno(
                                errno,
                                format!("setrlimit: {e}"),
                            ))
                        } else {
                            Ok(pyre_object::w_none())
                        }
                    }
                    #[cfg(feature = "sandbox")]
                    {
                        match rustpython_host_env::resource::setrlimit(res, rl) {
                            Ok(()) => Ok(pyre_object::w_none()),
                            Err(e) => {
                                // `lib_pypy/resource.py setrlimit` — EINVAL and
                                // EPERM both surface as ValueError with
                                // distinct messages; all other errnos stay
                                // as OSError.
                                let errno = e.raw_os_error().unwrap_or(0);
                                if errno == libc::EINVAL {
                                    return Err(pyre_interpreter::PyError::value_error(
                                        "current limit exceeds maximum limit",
                                    ));
                                }
                                if errno == libc::EPERM {
                                    return Err(pyre_interpreter::PyError::value_error(
                                        "not allowed to raise maximum limit",
                                    ));
                                }
                                Err(pyre_interpreter::PyError::os_error_with_errno(
                                    errno,
                                    format!("setrlimit: {e}"),
                                ))
                            }
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(pyre_interpreter::PyError::not_implemented(
                        "resource.setrlimit requires host_env feature",
                    ))
                }
            },
            2,
        ),
    );
    // ── Constants (POSIX subset matching CPython) ──
    #[cfg(unix)]
    {
        #[cfg(not(feature = "host_env"))]
        use libc as host_resource;
        #[cfg(feature = "host_env")]
        use rustpython_host_env::resource as host_resource;
        pyre_interpreter::module_ns_store(
            ns,
            "RUSAGE_SELF",
            pyre_object::w_int_new(host_resource::RUSAGE_SELF as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RUSAGE_CHILDREN",
            pyre_object::w_int_new(host_resource::RUSAGE_CHILDREN as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_CPU",
            pyre_object::w_int_new(host_resource::RLIMIT_CPU as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_FSIZE",
            pyre_object::w_int_new(host_resource::RLIMIT_FSIZE as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_DATA",
            pyre_object::w_int_new(host_resource::RLIMIT_DATA as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_STACK",
            pyre_object::w_int_new(host_resource::RLIMIT_STACK as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_CORE",
            pyre_object::w_int_new(host_resource::RLIMIT_CORE as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_NOFILE",
            pyre_object::w_int_new(host_resource::RLIMIT_NOFILE as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_AS",
            pyre_object::w_int_new(host_resource::RLIMIT_AS as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_RSS",
            pyre_object::w_int_new(host_resource::RLIMIT_RSS as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_NPROC",
            pyre_object::w_int_new(host_resource::RLIMIT_NPROC as i64),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "RLIMIT_MEMLOCK",
            pyre_object::w_int_new(host_resource::RLIMIT_MEMLOCK as i64),
        );
        // RLIM_INFINITY: unsigned max — pyre stores as i64 (-1 on signed widen).
        pyre_interpreter::module_ns_store(
            ns,
            "RLIM_INFINITY",
            pyre_object::w_int_new(host_resource::RLIM_INFINITY as i64),
        );
    }
    Ok(())
}
