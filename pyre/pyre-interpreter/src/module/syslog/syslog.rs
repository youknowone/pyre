//! syslog implementation — PyPy: lib_pypy/syslog.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

#[cfg(all(unix, feature = "host_env"))]
thread_local! {
    /// `lib_pypy/syslog.py:35-44` — track whether `openlog()` has been
    /// called so the first `syslog()` can auto-open with the default
    /// libc ident (NULL → program name).
    static SYSLOG_OPENED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// syslog module — PyPy: lib_pypy/syslog.py.
///
/// openlog / syslog / closelog / setlogmask.  Backed by
/// `rustpython_host_env::syslog`.  Unix-only.
pub fn register_module(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "openlog",
        crate::make_builtin_function("openlog", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                let ident = args.first().and_then(|&a| unsafe {
                    if pyre_object::is_str(a) {
                        std::ffi::CString::new(pyre_object::w_str_get_value(a))
                            .ok()
                            .map(|c| c.into_boxed_c_str())
                    } else {
                        None
                    }
                });
                if args
                    .iter()
                    .skip(1)
                    .any(|&a| !unsafe { pyre_object::is_int(a) })
                {
                    return Err(crate::PyError::type_error(
                        "openlog(): logoption and facility must be integers",
                    ));
                }
                let logoption = args
                    .get(1)
                    .map(|&a| unsafe { pyre_object::w_int_get_value(a) } as i32)
                    .unwrap_or(0);
                let facility = args
                    .get(2)
                    .map(|&a| unsafe { pyre_object::w_int_get_value(a) } as i32)
                    .unwrap_or(libc::LOG_USER);
                rustpython_host_env::syslog::openlog(ident, logoption, facility);
                SYSLOG_OPENED.with(|f| f.set(true));
                Ok(pyre_object::w_none())
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "syslog.openlog requires host_env feature",
                ))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "syslog",
        crate::make_builtin_function("syslog", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                let (priority, msg_obj) = if args.len() >= 2 {
                    if !unsafe { pyre_object::is_int(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "syslog(): priority must be an integer",
                        ));
                    }
                    (
                        unsafe { pyre_object::w_int_get_value(args[0]) as i32 },
                        args[1],
                    )
                } else if args.len() == 1 {
                    (libc::LOG_INFO, args[0])
                } else {
                    return Err(crate::PyError::type_error("syslog() requires a message"));
                };
                if !unsafe { pyre_object::is_str(msg_obj) } {
                    return Err(crate::PyError::type_error(
                        "syslog(): message must be a string",
                    ));
                }
                let msg = unsafe { pyre_object::w_str_get_value(msg_obj) };
                if let Ok(cmsg) = std::ffi::CString::new(msg) {
                    // `lib_pypy/syslog.py:42-44` — auto-call openlog() with
                    // a NULL ident (libc falls back to argv[0]) so the
                    // first syslog() call delivers correctly even when the
                    // caller skipped openlog().
                    if !SYSLOG_OPENED.with(|f| f.get()) {
                        rustpython_host_env::syslog::openlog(None, 0, libc::LOG_USER);
                        SYSLOG_OPENED.with(|f| f.set(true));
                    }
                    rustpython_host_env::syslog::syslog(priority, &cmsg);
                }
                Ok(pyre_object::w_none())
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "syslog.syslog requires host_env feature",
                ))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "closelog",
        crate::make_builtin_function_with_arity(
            "closelog",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    rustpython_host_env::syslog::closelog();
                }
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "setlogmask",
        crate::make_builtin_function_with_arity(
            "setlogmask",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let mask = if let Some(&a) = args.first() {
                        if !unsafe { pyre_object::is_int(a) } {
                            return Err(crate::PyError::type_error(
                                "setlogmask(): argument must be an integer",
                            ));
                        }
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(crate::PyError::type_error("setlogmask() missing argument"));
                    };
                    return Ok(pyre_object::w_int_new(
                        rustpython_host_env::syslog::setlogmask(mask) as i64,
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "syslog.setlogmask requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    // Priorities + facilities (POSIX subset matching CPython).
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "LOG_EMERG",
            pyre_object::w_int_new(libc::LOG_EMERG as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_ALERT",
            pyre_object::w_int_new(libc::LOG_ALERT as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_CRIT",
            pyre_object::w_int_new(libc::LOG_CRIT as i64),
        );
        crate::dict_storage_store(ns, "LOG_ERR", pyre_object::w_int_new(libc::LOG_ERR as i64));
        crate::dict_storage_store(
            ns,
            "LOG_WARNING",
            pyre_object::w_int_new(libc::LOG_WARNING as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_NOTICE",
            pyre_object::w_int_new(libc::LOG_NOTICE as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_INFO",
            pyre_object::w_int_new(libc::LOG_INFO as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_DEBUG",
            pyre_object::w_int_new(libc::LOG_DEBUG as i64),
        );
        crate::dict_storage_store(ns, "LOG_PID", pyre_object::w_int_new(libc::LOG_PID as i64));
        crate::dict_storage_store(
            ns,
            "LOG_CONS",
            pyre_object::w_int_new(libc::LOG_CONS as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_NDELAY",
            pyre_object::w_int_new(libc::LOG_NDELAY as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_NOWAIT",
            pyre_object::w_int_new(libc::LOG_NOWAIT as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_PERROR",
            pyre_object::w_int_new(libc::LOG_PERROR as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_KERN",
            pyre_object::w_int_new(libc::LOG_KERN as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_USER",
            pyre_object::w_int_new(libc::LOG_USER as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_MAIL",
            pyre_object::w_int_new(libc::LOG_MAIL as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_DAEMON",
            pyre_object::w_int_new(libc::LOG_DAEMON as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_AUTH",
            pyre_object::w_int_new(libc::LOG_AUTH as i64),
        );
        crate::dict_storage_store(ns, "LOG_LPR", pyre_object::w_int_new(libc::LOG_LPR as i64));
        crate::dict_storage_store(
            ns,
            "LOG_NEWS",
            pyre_object::w_int_new(libc::LOG_NEWS as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_UUCP",
            pyre_object::w_int_new(libc::LOG_UUCP as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_CRON",
            pyre_object::w_int_new(libc::LOG_CRON as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_SYSLOG",
            pyre_object::w_int_new(libc::LOG_SYSLOG as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL0",
            pyre_object::w_int_new(libc::LOG_LOCAL0 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL1",
            pyre_object::w_int_new(libc::LOG_LOCAL1 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL2",
            pyre_object::w_int_new(libc::LOG_LOCAL2 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL3",
            pyre_object::w_int_new(libc::LOG_LOCAL3 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL4",
            pyre_object::w_int_new(libc::LOG_LOCAL4 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL5",
            pyre_object::w_int_new(libc::LOG_LOCAL5 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL6",
            pyre_object::w_int_new(libc::LOG_LOCAL6 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL7",
            pyre_object::w_int_new(libc::LOG_LOCAL7 as i64),
        );
    }
    // `Modules/syslogmodule.c syslog_log_mask / syslog_log_upto` —
    // helpers for building setlogmask() arguments.
    //   LOG_MASK(pri)  → 1 << pri
    //   LOG_UPTO(pri)  → (1 << (pri + 1)) - 1
    crate::dict_storage_store(
        ns,
        "LOG_MASK",
        crate::make_builtin_function_with_arity(
            "LOG_MASK",
            |args| {
                let pri =
                    crate::baseobjspace::int_w(args.first().copied().ok_or_else(|| {
                        crate::PyError::type_error("LOG_MASK() missing argument")
                    })?)?;
                Ok(pyre_object::w_int_new(1i64 << pri))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "LOG_UPTO",
        crate::make_builtin_function_with_arity(
            "LOG_UPTO",
            |args| {
                let pri =
                    crate::baseobjspace::int_w(args.first().copied().ok_or_else(|| {
                        crate::PyError::type_error("LOG_UPTO() missing argument")
                    })?)?;
                Ok(pyre_object::w_int_new((1i64 << (pri + 1)) - 1))
            },
            1,
        ),
    );
}
