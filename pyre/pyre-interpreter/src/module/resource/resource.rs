//! resource implementation — `lib_pypy/resource.py`.
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

thread_local! {
    /// `lib_pypy/resource.py:15-37 class struct_rusage(
    /// metaclass=structseqtype)` — process-wide cached subclass-of-tuple
    /// type.
    static STRUCT_RUSAGE_TYPE: std::cell::OnceCell<pyre_object::PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

fn struct_rusage_type() -> pyre_object::PyObjectRef {
    STRUCT_RUSAGE_TYPE.with(|c| {
        *c.get_or_init(|| {
            crate::_structseq::make_struct_seq(
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
    })
}

/// resource module — `lib_pypy/resource.py` (PyPy keeps it app-level
/// via `_resource_cffi`).  pyre takes CPython's `Modules/resource.c`
/// shape since pyre has no app-level stdlib.
///
/// Exposes getrusage / getrlimit / setrlimit plus the standard RUSAGE_*
/// and RLIMIT_* constants, the `struct_rusage` type attribute, and the
/// `error = OSError` alias.  Backed by `rustpython_host_env::resource`.
pub fn register_module(ns: &mut DictStorage) {
    // `lib_pypy/resource.py:13 error = OSError` and
    // `:15-37 class struct_rusage`.
    let w_os_error = crate::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before init_resource");
    crate::dict_storage_store(ns, "error", w_os_error);
    crate::dict_storage_store(ns, "struct_rusage", struct_rusage_type());
    // ── struct_rusage tuple (16-field layout matches CPython) ──
    #[cfg(all(unix, feature = "host_env"))]
    fn make_struct_rusage(r: &rustpython_host_env::resource::RUsage) -> pyre_object::PyObjectRef {
        let tv_to_f = |tv: libc::timeval| tv.tv_sec as f64 + (tv.tv_usec as f64) * 1e-6;
        crate::_structseq::new_instance(
            struct_rusage_type(),
            vec![
                pyre_object::floatobject::w_float_new(tv_to_f(r.ru_utime)),
                pyre_object::floatobject::w_float_new(tv_to_f(r.ru_stime)),
                pyre_object::w_int_new(r.ru_maxrss as i64),
                pyre_object::w_int_new(r.ru_ixrss as i64),
                pyre_object::w_int_new(r.ru_idrss as i64),
                pyre_object::w_int_new(r.ru_isrss as i64),
                pyre_object::w_int_new(r.ru_minflt as i64),
                pyre_object::w_int_new(r.ru_majflt as i64),
                pyre_object::w_int_new(r.ru_nswap as i64),
                pyre_object::w_int_new(r.ru_inblock as i64),
                pyre_object::w_int_new(r.ru_oublock as i64),
                pyre_object::w_int_new(r.ru_msgsnd as i64),
                pyre_object::w_int_new(r.ru_msgrcv as i64),
                pyre_object::w_int_new(r.ru_nsignals as i64),
                pyre_object::w_int_new(r.ru_nvcsw as i64),
                pyre_object::w_int_new(r.ru_nivcsw as i64),
            ],
        )
    }
    crate::dict_storage_store(
        ns,
        "getrusage",
        crate::make_builtin_function_with_arity(
            "getrusage",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let who = if let Some(&a) = args.first() {
                        if unsafe { pyre_object::is_int(a) } {
                            unsafe { pyre_object::w_int_get_value(a) as i32 }
                        } else {
                            return Err(crate::PyError::type_error(
                                "getrusage(): who should be an integer",
                            ));
                        }
                    } else {
                        return Err(crate::PyError::type_error("getrusage() missing argument"));
                    };
                    match rustpython_host_env::resource::getrusage(who) {
                        Ok(r) => return Ok(make_struct_rusage(&r)),
                        Err(e) => {
                            let errno = e.raw_os_error().unwrap_or(0);
                            // `lib_pypy/resource.py:106` raises ValueError for
                            // an invalid `who`; only other errno values are
                            // surfaced as OSError.
                            if errno == libc::EINVAL {
                                return Err(crate::PyError::value_error("invalid who parameter"));
                            }
                            return Err(crate::PyError::os_error_with_errno(
                                errno,
                                format!("getrusage: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "resource.getrusage requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getrlimit",
        crate::make_builtin_function_with_arity(
            "getrlimit",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let res = if let Some(&a) = args.first() {
                        if unsafe { pyre_object::is_int(a) } {
                            unsafe { pyre_object::w_int_get_value(a) as libc::rlim_t }
                        } else {
                            return Err(crate::PyError::type_error(
                                "getrlimit(): resource should be an integer",
                            ));
                        }
                    } else {
                        return Err(crate::PyError::type_error("getrlimit() missing argument"));
                    };
                    match rustpython_host_env::resource::getrlimit(res) {
                        Ok(rl) => {
                            return Ok(pyre_object::w_tuple_new(vec![
                                pyre_object::w_int_new(rl.rlim_cur as i64),
                                pyre_object::w_int_new(rl.rlim_max as i64),
                            ]));
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getrlimit: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "resource.getrlimit requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "setrlimit",
        crate::make_builtin_function_with_arity(
            "setrlimit",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "setrlimit() requires 2 arguments",
                        ));
                    }
                    let res = unsafe {
                        if !pyre_object::is_int(args[0]) {
                            return Err(crate::PyError::type_error(
                                "setrlimit(): resource should be an integer",
                            ));
                        }
                        pyre_object::w_int_get_value(args[0]) as libc::rlim_t
                    };
                    // `lib_pypy/resource.py:81-86` — `soft, hard = limits;
                    // soft = int(soft); hard = int(hard)`.  Accept any
                    // 2-item tuple or list and coerce each entry to int
                    // (PyPy unpacks via Python iteration; pyre's surface
                    // covers the two concrete sequence shapes callers
                    // actually use).
                    let (w_soft, w_hard) = unsafe {
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
                            return Err(crate::PyError::type_error(
                                "expected a tuple of 2 integers",
                            ));
                        }
                    };
                    let soft = crate::baseobjspace::int_w(w_soft)? as libc::rlim_t;
                    let hard = crate::baseobjspace::int_w(w_hard)? as libc::rlim_t;
                    let rl = libc::rlimit {
                        rlim_cur: soft,
                        rlim_max: hard,
                    };
                    match rustpython_host_env::resource::setrlimit(res, rl) {
                        Ok(()) => return Ok(pyre_object::w_none()),
                        Err(e) => {
                            // `lib_pypy/resource.py:89-95` — EINVAL and
                            // EPERM both surface as ValueError with
                            // distinct messages; all other errnos stay
                            // as OSError.
                            let errno = e.raw_os_error().unwrap_or(0);
                            if errno == libc::EINVAL {
                                return Err(crate::PyError::value_error(
                                    "current limit exceeds maximum limit",
                                ));
                            }
                            if errno == libc::EPERM {
                                return Err(crate::PyError::value_error(
                                    "not allowed to raise maximum limit",
                                ));
                            }
                            return Err(crate::PyError::os_error_with_errno(
                                errno,
                                format!("setrlimit: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
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
        crate::dict_storage_store(
            ns,
            "RUSAGE_SELF",
            pyre_object::w_int_new(libc::RUSAGE_SELF as i64),
        );
        crate::dict_storage_store(
            ns,
            "RUSAGE_CHILDREN",
            pyre_object::w_int_new(libc::RUSAGE_CHILDREN as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_CPU",
            pyre_object::w_int_new(libc::RLIMIT_CPU as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_FSIZE",
            pyre_object::w_int_new(libc::RLIMIT_FSIZE as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_DATA",
            pyre_object::w_int_new(libc::RLIMIT_DATA as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_STACK",
            pyre_object::w_int_new(libc::RLIMIT_STACK as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_CORE",
            pyre_object::w_int_new(libc::RLIMIT_CORE as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_NOFILE",
            pyre_object::w_int_new(libc::RLIMIT_NOFILE as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_AS",
            pyre_object::w_int_new(libc::RLIMIT_AS as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_RSS",
            pyre_object::w_int_new(libc::RLIMIT_RSS as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_NPROC",
            pyre_object::w_int_new(libc::RLIMIT_NPROC as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_MEMLOCK",
            pyre_object::w_int_new(libc::RLIMIT_MEMLOCK as i64),
        );
        // RLIM_INFINITY: unsigned max — pyre stores as i64 (-1 on signed widen).
        crate::dict_storage_store(
            ns,
            "RLIM_INFINITY",
            pyre_object::w_int_new(libc::RLIM_INFINITY as i64),
        );
    }
}
