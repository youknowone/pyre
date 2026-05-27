//! _signal implementation — PyPy: pypy/module/signal/interp_signal.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

// `interp_signal.py:set_wakeup_fd` — stores the configured wakeup fd
// for read-back via set_wakeup_fd(new) → previous_fd.  Real signal-to-fd
// delivery is not wired up; this cell is the get/set contract only.
// PyPy keeps this fd process-wide (it lives on the signal action
// handler, not per Python-thread), so we mirror that with an atomic.
use std::sync::atomic::{AtomicI32, Ordering};
static WAKEUP_FD: AtomicI32 = AtomicI32::new(-1);

/// _signal module — PyPy: pypy/module/signal/.
///
/// signal() / getsignal() / set_wakeup_fd() remain stubs because the
/// real implementations need interpreter-side trampolines to invoke
/// Python handlers from a Rust signal context.  alarm / pause /
/// raise_signal / strsignal / valid_signals are full implementations
/// backed by `rustpython_host_env::signal`.  Signal-number constants
/// are sourced from `libc::*` so they match the host's POSIX numbering
/// (the previous macOS-flavoured hard-coded list disagreed with Linux
/// for SIGUSR1/SIGUSR2/SIGCHLD).
pub fn register_module(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "signal",
        crate::make_builtin_function_with_arity(
            "signal",
            // signal(signalnum, handler) — pyre does not actually
            // install handlers, so the "previous handler" is always
            // SIG_DFL/None.  Returning `handler` would lie about the
            // prior state and confuse callers that swap+restore.
            |_| Ok(pyre_object::w_none()),
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getsignal",
        crate::make_builtin_function_with_arity("getsignal", |_| Ok(pyre_object::w_none()), 1),
    );
    // `interp_signal.py:default_int_handler` — `raise KeyboardInterrupt`.
    crate::dict_storage_store(
        ns,
        "default_int_handler",
        crate::make_builtin_function_with_arity(
            "default_int_handler",
            |_| {
                let cls = crate::builtins::lookup_exc_class("KeyboardInterrupt")
                    .expect("KeyboardInterrupt must be installed");
                let exc = crate::builtins::exc_exception_new(&[cls])
                    .expect("exc_exception_new is infallible for empty args");
                Err(unsafe { crate::PyError::from_exc_object(exc) })
            },
            2,
        ),
    );
    // `interp_signal.py:set_wakeup_fd` — stores the fd in a
    // process-wide cell and returns the previous value.  Real signal
    // delivery on the fd needs interpreter-side trampolines (still
    // unimplemented per the header comment); we still surface the
    // get/set contract so callers like `signal.set_wakeup_fd(-1)` no
    // longer silently report a stale −1.
    crate::dict_storage_store(
        ns,
        "set_wakeup_fd",
        crate::make_builtin_function("set_wakeup_fd", |args| {
            let fd = if let Some(&a) = args.first() {
                if !unsafe { pyre_object::is_int(a) } {
                    return Err(crate::PyError::type_error(
                        "set_wakeup_fd() argument must be an int",
                    ));
                }
                (unsafe { pyre_object::w_int_get_value(a) }) as i32
            } else {
                return Err(crate::PyError::type_error(
                    "set_wakeup_fd() requires an argument",
                ));
            };
            if fd < -1 {
                return Err(crate::PyError::value_error(
                    "set_wakeup_fd(): fd must be -1 or a valid file descriptor",
                ));
            }
            let prev = WAKEUP_FD.swap(fd, Ordering::SeqCst);
            Ok(pyre_object::w_int_new(prev as i64))
        }),
    );
    // ── real host_env-backed entry points ──
    crate::dict_storage_store(
        ns,
        "raise_signal",
        crate::make_builtin_function_with_arity(
            "raise_signal",
            |args| {
                #[cfg(feature = "host_env")]
                {
                    let signum = if let Some(&a) = args.first() {
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(crate::PyError::type_error(
                            "raise_signal() missing argument",
                        ));
                    };
                    match rustpython_host_env::signal::raise_signal(signum) {
                        Ok(()) => return Ok(pyre_object::w_none()),
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("raise_signal: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.raise_signal requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "strsignal",
        crate::make_builtin_function_with_arity(
            "strsignal",
            |args| {
                #[cfg(feature = "host_env")]
                {
                    let signum = if let Some(&a) = args.first() {
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(crate::PyError::type_error("strsignal() missing argument"));
                    };
                    return Ok(rustpython_host_env::signal::strsignal(signum)
                        .map(|s| pyre_object::w_str_new(&s))
                        .unwrap_or(pyre_object::w_none()));
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.strsignal requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "valid_signals",
        crate::make_builtin_function_with_arity(
            "valid_signals",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    // `interp_signal.py:550-574 valid_signals` returns
                    // `set(...)` via `_sigset_to_signals` (line 513),
                    // not a frozenset.  PyPy passes NSIG (64) here.
                    let sigs = rustpython_host_env::signal::valid_signals(64).unwrap_or_default();
                    let items: Vec<pyre_object::PyObjectRef> = sigs
                        .into_iter()
                        .map(|n| pyre_object::w_int_new(n as i64))
                        .collect();
                    return Ok(pyre_object::w_set_from_items(&items));
                }
                #[cfg(not(feature = "host_env"))]
                Err(crate::PyError::not_implemented(
                    "signal.valid_signals requires host_env feature",
                ))
            },
            0,
        ),
    );
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "alarm",
            crate::make_builtin_function_with_arity(
                "alarm",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        let secs = if let Some(&a) = args.first() {
                            unsafe { pyre_object::w_int_get_value(a) as u32 }
                        } else {
                            return Err(crate::PyError::type_error("alarm() missing argument"));
                        };
                        return Ok(pyre_object::w_int_new(
                            rustpython_host_env::signal::alarm(secs) as i64,
                        ));
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.alarm requires host_env feature",
                        ))
                    }
                },
                1,
            ),
        );
        crate::dict_storage_store(
            ns,
            "pause",
            crate::make_builtin_function_with_arity(
                "pause",
                |_| {
                    #[cfg(feature = "host_env")]
                    {
                        rustpython_host_env::signal::pause();
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        Err(crate::PyError::not_implemented(
                            "signal.pause requires host_env feature",
                        ))
                    }
                },
                0,
            ),
        );
        // setitimer(which, seconds, interval=0.0) -> (delay, interval)
        crate::dict_storage_store(
            ns,
            "setitimer",
            crate::make_builtin_function("setitimer", |args| {
                #[cfg(feature = "host_env")]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "setitimer() requires at least 2 arguments",
                        ));
                    }
                    let which = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let read_f = |o: pyre_object::PyObjectRef| -> f64 {
                        unsafe {
                            if pyre_object::is_float(o) {
                                pyre_object::w_float_get_value(o)
                            } else {
                                pyre_object::w_int_get_value(o) as f64
                            }
                        }
                    };
                    let new_value = libc::itimerval {
                        it_value: rustpython_host_env::signal::double_to_timeval(read_f(args[1])),
                        it_interval: if args.len() >= 3 {
                            rustpython_host_env::signal::double_to_timeval(read_f(args[2]))
                        } else {
                            rustpython_host_env::signal::double_to_timeval(0.0)
                        },
                    };
                    let old =
                        rustpython_host_env::signal::setitimer(which, &new_value).map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("setitimer: {e}"),
                            )
                        })?;
                    let (delay, interval) = rustpython_host_env::signal::itimerval_to_tuple(&old);
                    return Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_float_new(delay),
                        pyre_object::w_float_new(interval),
                    ]));
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.setitimer requires host_env feature",
                    ))
                }
            }),
        );
        // getitimer(which) -> (delay, interval)
        crate::dict_storage_store(
            ns,
            "getitimer",
            crate::make_builtin_function_with_arity(
                "getitimer",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "getitimer() requires 1 argument",
                            ));
                        }
                        let which = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                        let it = rustpython_host_env::signal::getitimer(which).map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getitimer: {e}"),
                            )
                        })?;
                        let (delay, interval) =
                            rustpython_host_env::signal::itimerval_to_tuple(&it);
                        return Ok(pyre_object::w_tuple_new(vec![
                            pyre_object::w_float_new(delay),
                            pyre_object::w_float_new(interval),
                        ]));
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.getitimer requires host_env feature",
                        ))
                    }
                },
                1,
            ),
        );
        // siginterrupt(signalnum, flag) -> None
        crate::dict_storage_store(
            ns,
            "siginterrupt",
            crate::make_builtin_function_with_arity(
                "siginterrupt",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "siginterrupt() requires 2 arguments",
                            ));
                        }
                        let sig = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                        let flag = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                        rustpython_host_env::signal::siginterrupt(sig, flag).map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("siginterrupt: {e}"),
                            )
                        })?;
                        return Ok(pyre_object::w_none());
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.siginterrupt requires host_env feature",
                        ))
                    }
                },
                2,
            ),
        );
        // ITIMER_REAL/VIRTUAL/PROF
        crate::dict_storage_store(
            ns,
            "ITIMER_REAL",
            pyre_object::w_int_new(libc::ITIMER_REAL as i64),
        );
        crate::dict_storage_store(
            ns,
            "ITIMER_VIRTUAL",
            pyre_object::w_int_new(libc::ITIMER_VIRTUAL as i64),
        );
        crate::dict_storage_store(
            ns,
            "ITIMER_PROF",
            pyre_object::w_int_new(libc::ITIMER_PROF as i64),
        );
        // pthread_sigmask(how, mask) -> previous mask (set of signums)
        crate::dict_storage_store(
            ns,
            "pthread_sigmask",
            crate::make_builtin_function_with_arity(
                "pthread_sigmask",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "pthread_sigmask() requires 2 arguments",
                            ));
                        }
                        let how = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                        let mask_arg = args[1];
                        let items: Vec<pyre_object::PyObjectRef> =
                            if unsafe { pyre_object::is_list(mask_arg) } {
                                let n = unsafe { pyre_object::w_list_len(mask_arg) };
                                (0..n)
                                    .filter_map(|i| unsafe {
                                        pyre_object::w_list_getitem(mask_arg, i as i64)
                                    })
                                    .collect()
                            } else if unsafe { pyre_object::is_tuple(mask_arg) } {
                                let n = unsafe { pyre_object::w_tuple_len(mask_arg) };
                                (0..n)
                                    .filter_map(|i| unsafe {
                                        pyre_object::w_tuple_getitem(mask_arg, i as i64)
                                    })
                                    .collect()
                            } else if unsafe { pyre_object::is_set_or_frozenset(mask_arg) } {
                                unsafe { pyre_object::w_set_items(mask_arg) }
                            } else {
                                return Err(crate::PyError::type_error(
                                    "pthread_sigmask: mask must be a list, tuple, or set",
                                ));
                            };
                        let mut set = rustpython_host_env::signal::sigemptyset().map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("sigemptyset: {e}"),
                            )
                        })?;
                        for it in items {
                            let signum = (unsafe { pyre_object::w_int_get_value(it) }) as i32;
                            rustpython_host_env::signal::sigaddset(&mut set, signum).map_err(
                                |e| {
                                    crate::PyError::os_error_with_errno(
                                        e.raw_os_error().unwrap_or(0),
                                        format!("sigaddset: {e}"),
                                    )
                                },
                            )?;
                        }
                        let prev = rustpython_host_env::signal::pthread_sigmask(how, &set)
                            .map_err(|e| {
                                crate::PyError::os_error_with_errno(
                                    e.raw_os_error().unwrap_or(0),
                                    format!("pthread_sigmask: {e}"),
                                )
                            })?;
                        let out: Vec<pyre_object::PyObjectRef> = (1..=64)
                            .filter(|s| {
                                rustpython_host_env::signal::sigset_contains(&prev, *s as i32)
                            })
                            .map(|s| pyre_object::w_int_new(s as i64))
                            .collect();
                        return Ok(pyre_object::w_set_from_items(&out));
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.pthread_sigmask requires host_env feature",
                        ))
                    }
                },
                2,
            ),
        );
        crate::dict_storage_store(
            ns,
            "SIG_BLOCK",
            pyre_object::w_int_new(libc::SIG_BLOCK as i64),
        );
        crate::dict_storage_store(
            ns,
            "SIG_UNBLOCK",
            pyre_object::w_int_new(libc::SIG_UNBLOCK as i64),
        );
        crate::dict_storage_store(
            ns,
            "SIG_SETMASK",
            pyre_object::w_int_new(libc::SIG_SETMASK as i64),
        );
        // pidfd_send_signal(pidfd, sig, siginfo=None, flags=0) - Linux-only
        #[cfg(target_os = "linux")]
        crate::dict_storage_store(
            ns,
            "pidfd_send_signal",
            crate::make_builtin_function("pidfd_send_signal", |args| {
                #[cfg(feature = "host_env")]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "pidfd_send_signal() requires at least 2 arguments",
                        ));
                    }
                    let pidfd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let sig = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    let flags = if args.len() >= 4 {
                        (unsafe { pyre_object::w_int_get_value(args[3]) }) as u32
                    } else {
                        0
                    };
                    rustpython_host_env::signal::pidfd_send_signal(pidfd, sig, flags).map_err(
                        |e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("pidfd_send_signal: {e}"),
                            )
                        },
                    )?;
                    return Ok(pyre_object::w_none());
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.pidfd_send_signal requires host_env feature",
                    ))
                }
            }),
        );
    }
    crate::dict_storage_store(ns, "SIG_DFL", pyre_object::w_int_new(0));
    crate::dict_storage_store(ns, "SIG_IGN", pyre_object::w_int_new(1));
    // libc crate doesn't surface NSIG portably; use POSIX 64-signal cap.
    crate::dict_storage_store(ns, "NSIG", pyre_object::w_int_new(64));
    // Common signal numbers (POSIX subset, sourced from libc so numerics
    // match the host — Linux SIGUSR1=10 / macOS SIGUSR1=30, etc.).
    #[cfg(unix)]
    {
        crate::dict_storage_store(ns, "SIGHUP", pyre_object::w_int_new(libc::SIGHUP as i64));
        crate::dict_storage_store(ns, "SIGINT", pyre_object::w_int_new(libc::SIGINT as i64));
        crate::dict_storage_store(ns, "SIGQUIT", pyre_object::w_int_new(libc::SIGQUIT as i64));
        crate::dict_storage_store(ns, "SIGILL", pyre_object::w_int_new(libc::SIGILL as i64));
        crate::dict_storage_store(ns, "SIGTRAP", pyre_object::w_int_new(libc::SIGTRAP as i64));
        crate::dict_storage_store(ns, "SIGABRT", pyre_object::w_int_new(libc::SIGABRT as i64));
        crate::dict_storage_store(ns, "SIGBUS", pyre_object::w_int_new(libc::SIGBUS as i64));
        crate::dict_storage_store(ns, "SIGFPE", pyre_object::w_int_new(libc::SIGFPE as i64));
        crate::dict_storage_store(ns, "SIGKILL", pyre_object::w_int_new(libc::SIGKILL as i64));
        crate::dict_storage_store(ns, "SIGUSR1", pyre_object::w_int_new(libc::SIGUSR1 as i64));
        crate::dict_storage_store(ns, "SIGSEGV", pyre_object::w_int_new(libc::SIGSEGV as i64));
        crate::dict_storage_store(ns, "SIGUSR2", pyre_object::w_int_new(libc::SIGUSR2 as i64));
        crate::dict_storage_store(ns, "SIGPIPE", pyre_object::w_int_new(libc::SIGPIPE as i64));
        crate::dict_storage_store(ns, "SIGALRM", pyre_object::w_int_new(libc::SIGALRM as i64));
        crate::dict_storage_store(ns, "SIGTERM", pyre_object::w_int_new(libc::SIGTERM as i64));
        crate::dict_storage_store(ns, "SIGCHLD", pyre_object::w_int_new(libc::SIGCHLD as i64));
        crate::dict_storage_store(ns, "SIGCONT", pyre_object::w_int_new(libc::SIGCONT as i64));
        crate::dict_storage_store(ns, "SIGSTOP", pyre_object::w_int_new(libc::SIGSTOP as i64));
        crate::dict_storage_store(ns, "SIGTSTP", pyre_object::w_int_new(libc::SIGTSTP as i64));
        crate::dict_storage_store(ns, "SIGTTIN", pyre_object::w_int_new(libc::SIGTTIN as i64));
        crate::dict_storage_store(ns, "SIGTTOU", pyre_object::w_int_new(libc::SIGTTOU as i64));
        crate::dict_storage_store(ns, "SIGURG", pyre_object::w_int_new(libc::SIGURG as i64));
        crate::dict_storage_store(ns, "SIGXCPU", pyre_object::w_int_new(libc::SIGXCPU as i64));
        crate::dict_storage_store(ns, "SIGXFSZ", pyre_object::w_int_new(libc::SIGXFSZ as i64));
        crate::dict_storage_store(
            ns,
            "SIGVTALRM",
            pyre_object::w_int_new(libc::SIGVTALRM as i64),
        );
        crate::dict_storage_store(ns, "SIGPROF", pyre_object::w_int_new(libc::SIGPROF as i64));
        crate::dict_storage_store(
            ns,
            "SIGWINCH",
            pyre_object::w_int_new(libc::SIGWINCH as i64),
        );
        crate::dict_storage_store(ns, "SIGIO", pyre_object::w_int_new(libc::SIGIO as i64));
        crate::dict_storage_store(ns, "SIGSYS", pyre_object::w_int_new(libc::SIGSYS as i64));
    }
}
