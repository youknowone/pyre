//! faulthandler implementation — PyPy: pypy/module/faulthandler/handler.py
//!
//! Verbatim move of the inline block previously in importing.rs.  `init_faulthandler`
//! was renamed to `register_module`; the host_env signal handlers and the
//! `faulthandler_extract_fd` helper stay private.

use crate::DictStorage;

// faulthandler module — PyPy: pypy/module/faulthandler/.
//
// CPython's faulthandler dumps the Python traceback on fatal signals.
// Pyre has no Python-level traceback machinery yet, so our handler
// writes a short "Fatal Python error: <name>" line to fd 2 and then
// restores the default disposition + reraises the signal so the
// process dies the normal way.
// ──────────────────────────────────────────────────────────────────────

#[cfg(all(unix, feature = "host_env"))]
thread_local! {
    static FAULTHANDLER_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(unix, feature = "host_env"))]
extern "C" fn faulthandler_signal_handler(signum: libc::c_int) {
    // Stay async-signal-safe: write to fd 2 with raw libc::write and
    // restore the default disposition before reraising.
    let name =
        rustpython_host_env::faulthandler::fatal_signal_name(signum).unwrap_or("unknown signal");
    let msg = format!("Fatal Python error: {name}\n");
    rustpython_host_env::faulthandler::write_fd(2, msg.as_bytes());
    rustpython_host_env::faulthandler::signal_default_and_raise(signum);
}

/// `handler.py:35-49 Handler.get_fileno_and_file` — extract a fileno
/// from a python file-or-fd-or-None argument.  None → fd 2 (stderr);
/// int → used directly; any other object → call `.fileno()`.
fn faulthandler_extract_fd(w_file: pyre_object::PyObjectRef) -> Result<i32, crate::PyError> {
    if w_file.is_null() || unsafe { pyre_object::is_none(w_file) } {
        return Ok(2);
    }
    if unsafe { pyre_object::is_int(w_file) } {
        let fd = unsafe { pyre_object::w_int_get_value(w_file) } as i32;
        if fd < 0 {
            return Err(crate::PyError::value_error(
                "file is not a valid file descriptor",
            ));
        }
        return Ok(fd);
    }
    let method = crate::baseobjspace::getattr_str(w_file, "fileno")?;
    let res = crate::call_function(method, &[]);
    if res.is_null() || !unsafe { pyre_object::is_int(res) } {
        return Err(crate::PyError::type_error("fileno() returned non-integer"));
    }
    Ok(unsafe { pyre_object::w_int_get_value(res) } as i32)
}

pub fn register_module(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "enable",
        crate::make_builtin_function_with_signature(
            "enable",
            |args| {
            // `handler.py:141-145 enable` — file=None, all_threads=True.
            let _fd =
                faulthandler_extract_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
            #[cfg(all(unix, feature = "host_env"))]
            {
                let ok = rustpython_host_env::faulthandler::enable_fatal_handlers(
                    faulthandler_signal_handler,
                    libc::SA_NODEFER | libc::SA_ONSTACK,
                );
                if ok {
                    FAULTHANDLER_ENABLED.with(|c| c.set(true));
                    return Ok(pyre_object::w_none());
                }
                return Err(crate::PyError::runtime_error(
                    "faulthandler.enable: sigaction failed",
                ));
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            Err(crate::PyError::not_implemented(
                "faulthandler.enable requires host_env feature",
            ))
            },
            crate::Signature::new(vec!["file", "all_threads"], None, None, 0, 0),
        ),
    );
    crate::dict_storage_store(
        ns,
        "disable",
        crate::make_builtin_function_with_arity(
            "disable",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    rustpython_host_env::faulthandler::disable_fatal_handlers();
                    FAULTHANDLER_ENABLED.with(|c| c.set(false));
                }
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_enabled",
        crate::make_builtin_function_with_arity(
            "is_enabled",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    return Ok(pyre_object::w_bool_from(
                        FAULTHANDLER_ENABLED.with(|c| c.get()),
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                Ok(pyre_object::w_bool_from(false))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "dump_traceback",
        crate::make_builtin_function("dump_traceback", |_| {
            // No Python-level traceback machinery — emit a placeholder
            // so callers that want a forensic dump at least see *something*
            // instead of silent success.
            #[cfg(unix)]
            {
                let msg = b"<faulthandler: pyre has no Python-level traceback yet>\n";
                let _ =
                    unsafe { libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len() as _) };
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "dump_traceback_later",
        crate::make_builtin_function("dump_traceback_later", |_| Ok(pyre_object::w_none())),
    );
    crate::dict_storage_store(
        ns,
        "cancel_dump_traceback_later",
        crate::make_builtin_function_with_arity(
            "cancel_dump_traceback_later",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    // register/unregister user signals: host_env supports the full API,
    // but it needs the user-signal handler to be a fixed extern "C" fn.
    // Provide a "registered → no-op" pattern: install the handler when
    // registering, restore on unregister.  The handler writes a short
    // "user signal NN delivered" message to fd 2 (no traceback).
    // `handler.py:115-128 register(signum, file=None, all_threads=True, chain=False)`.
    crate::dict_storage_store(
        ns,
        "register",
        crate::make_builtin_function_with_signature(
            "register",
            |args| {
            let w_signum = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if w_signum.is_null() {
                return Err(crate::PyError::type_error("register() missing signal"));
            }
            let signum = (unsafe { pyre_object::w_int_get_value(w_signum) }) as libc::c_int;
            let fd = faulthandler_extract_fd(args.get(1).copied().unwrap_or(pyre_object::PY_NULL))?;
            // handler.py:174 `@unwrap_spec(all_threads=int, chain=int)`
            // with defaults `all_threads=1, chain=0`: the arguments are
            // coerced as integers (`gateway_int_w`, raising on a non-int),
            // not by truthiness; `register` then tests `if all_threads:`.  An
            // omitted keyword leaves a null slot from the signature binding, so
            // treat null as the default.
            let all_threads = args
                .get(2)
                .copied()
                .filter(|a| !a.is_null())
                .map(crate::baseobjspace::gateway_int_w)
                .transpose()?
                .unwrap_or(1)
                != 0;
            let chain = args
                .get(3)
                .copied()
                .filter(|a| !a.is_null())
                .map(crate::baseobjspace::gateway_int_w)
                .transpose()?
                .unwrap_or(0)
                != 0;
            #[cfg(all(unix, feature = "host_env"))]
            {
                rustpython_host_env::faulthandler::register_user_signal(
                    signum,
                    fd,
                    all_threads,
                    chain,
                    faulthandler_user_handler,
                )
                .map_err(|e| {
                    crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("register: {e}"),
                    )
                })?;
                return Ok(pyre_object::w_none());
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = (fd, all_threads, chain);
                Err(crate::PyError::not_implemented(
                    "faulthandler.register requires host_env feature",
                ))
            }
            },
            crate::Signature::new(
                vec!["signum", "file", "all_threads", "chain"],
                None,
                None,
                0,
                0,
            ),
        ),
    );
    crate::dict_storage_store(
        ns,
        "unregister",
        crate::make_builtin_function_with_arity(
            "unregister",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("unregister() missing signal"));
                    }
                    let signum = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    return Ok(pyre_object::w_bool_from(
                        rustpython_host_env::faulthandler::unregister_user_signal(signum),
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Ok(pyre_object::w_bool_from(false))
                }
            },
            1,
        ),
    );

    // `handler.py:225-245` test-only crash helpers from
    // `moduledef.py:14-22`.  Each unconditionally takes down the
    // process — only ever called from test_faulthandler.py in a
    // subprocess.  Pyre cannot construct an OperationError here
    // because the abort/segfault leaves no caller to catch it.
    crate::dict_storage_store(
        ns,
        "_read_null",
        crate::make_builtin_function_with_arity(
            "_read_null",
            |_| {
                // `handler.py:225 read_null` — null-pointer deref.
                let p: *const u8 = std::ptr::null();
                let _ = unsafe { p.read_volatile() };
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_sigsegv",
        crate::make_builtin_function_with_arity(
            "_sigsegv",
            |_| {
                #[cfg(unix)]
                unsafe {
                    libc::raise(libc::SIGSEGV);
                }
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_sigfpe",
        crate::make_builtin_function_with_arity(
            "_sigfpe",
            |_| {
                #[cfg(unix)]
                unsafe {
                    libc::raise(libc::SIGFPE);
                }
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_sigabrt",
        crate::make_builtin_function_with_arity(
            "_sigabrt",
            |_| {
                #[cfg(unix)]
                unsafe {
                    libc::abort();
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_stack_overflow",
        crate::make_builtin_function_with_arity(
            "_stack_overflow",
            |_| {
                // `handler.py:240 stack_overflow` — infinite recursion.
                fn blow() {
                    let _buf = [0u8; 4096];
                    blow();
                    std::hint::black_box(_buf);
                }
                blow();
                #[allow(unreachable_code)]
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
}

#[cfg(all(unix, feature = "host_env"))]
extern "C" fn faulthandler_user_handler(signum: libc::c_int) {
    let msg = format!("User signal {signum} delivered (faulthandler)\n");
    rustpython_host_env::faulthandler::write_fd(2, msg.as_bytes());
}
