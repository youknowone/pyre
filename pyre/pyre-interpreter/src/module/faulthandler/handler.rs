//! faulthandler implementation — PyPy: pypy/module/faulthandler/handler.py
//!
//! Verbatim move of the inline block previously in importing.rs.  `init_faulthandler`
//! was renamed to `register_module`; the host_env signal handlers and the
//! `faulthandler_extract_fd` helper stay private.

// faulthandler module — PyPy: pypy/module/faulthandler/.
//
// CPython's faulthandler dumps the Python traceback on fatal signals.
// Pyre has no Python-level traceback machinery yet, so our handler
// writes a short "Fatal Python error: <name>" line to the descriptor
// `enable` was given and then restores the default disposition +
// reraises the signal so the process dies the normal way.
// ──────────────────────────────────────────────────────────────────────

#[cfg(all(unix, feature = "host_env"))]
static FAULTHANDLER_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The descriptor `enable` resolved from its `file` argument.  `handler.py`
/// keeps it on the Handler instance, but the signal handler below is a bare
/// `extern "C" fn` and cannot capture, so it is handed over through a static.
/// Defaults to 2, which is what `get_fileno_and_file` answers for None.
#[cfg(all(unix, feature = "host_env"))]
static FAULTHANDLER_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(2);

/// `handler.py:145` `self.fatal_error_w_file = w_file` / `handler.py:150`
/// `self.fatal_error_w_file = None`: the descriptor the handler writes to
/// belongs to this object, so it has to outlive the installed handlers rather
/// than be collected and have its finalizer close the fd under them.  pyre has
/// no Handler instance and the signal callback is a bare `extern "C" fn` that
/// cannot capture one, so the owner is a process-global slot, walked as a GC
/// root by [`walk_fatal_error_file`].
#[cfg(all(unix, feature = "host_env"))]
static FAULTHANDLER_FILE: std::sync::atomic::AtomicPtr<pyre_object::PyObject> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Take ownership of the object owning the fatal-error descriptor; a null
/// drops it (`enable` with a plain fd, and `disable`).
#[cfg(all(unix, feature = "host_env"))]
fn set_fatal_error_file(w_file: pyre_object::PyObjectRef) {
    FAULTHANDLER_FILE.store(w_file, std::sync::atomic::Ordering::Relaxed);
}

/// Root walker for [`FAULTHANDLER_FILE`], registered alongside the other
/// process-global interpreter roots.  Forwards the slot in place so a moving
/// collection relocates the held file rather than leaving a stale address the
/// next `enable` would compare against.
#[cfg(all(unix, feature = "host_env"))]
pub fn walk_fatal_error_file(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let mut slot: pyre_object::PyObjectRef =
        FAULTHANDLER_FILE.load(std::sync::atomic::Ordering::Relaxed);
    if slot.is_null() {
        return;
    }
    visitor(unsafe {
        &mut *(&mut slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef)
    });
    FAULTHANDLER_FILE.store(slot, std::sync::atomic::Ordering::Relaxed);
}

/// No fatal-signal handlers to own a file for off the host_env unix path.
#[cfg(not(all(unix, feature = "host_env")))]
pub fn walk_fatal_error_file(_visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {}

#[cfg(all(unix, feature = "host_env"))]
extern "C" fn faulthandler_signal_handler(signum: libc::c_int) {
    // Stay async-signal-safe: write with raw libc::write and restore the
    // default disposition before reraising.
    let name =
        rustpython_host_env::faulthandler::fatal_signal_name(signum).unwrap_or("unknown signal");
    let msg = format!("Fatal Python error: {name}\n");
    let fd = FAULTHANDLER_FD.load(std::sync::atomic::Ordering::Relaxed);
    rustpython_host_env::faulthandler::write_fd(fd, msg.as_bytes());
    rustpython_host_env::faulthandler::signal_default_and_raise(signum);
}

/// `handler.py:35-49 Handler.get_fileno_and_file` — resolve a
/// file-or-fd-or-None argument to `(fileno, file)`.  None resolves the CURRENT
/// `sys.stderr` rather than a hard-coded fd 2, so a redirected stderr is
/// honoured; an int is used directly and names no file; anything else is asked
/// for its `fileno()` and then flushed, with an ordinary flush error ignored.
///
/// The returned file is the object the descriptor belongs to, and the caller
/// parks it for as long as the handlers are installed — a null means there is
/// nothing to own.
fn faulthandler_get_fileno_and_file(
    w_file: pyre_object::PyObjectRef,
) -> Result<(i32, pyre_object::PyObjectRef), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let resolved = if w_file.is_null() || unsafe { pyre_object::is_none(w_file) } {
        let sys = crate::importing::get_sys_module("sys")
            .ok_or_else(|| crate::PyError::runtime_error("sys.stderr is None"))?;
        let w_stderr = crate::baseobjspace::getattr_str(sys, "stderr")?;
        if w_stderr.is_null() || unsafe { pyre_object::is_none(w_stderr) } {
            return Err(crate::PyError::runtime_error("sys.stderr is None"));
        }
        w_stderr
    } else if unsafe { pyre_object::is_int(w_file) } {
        let fd = unsafe { pyre_object::w_int_get_value(w_file) } as i32;
        if fd < 0 {
            return Err(crate::PyError::value_error(
                "file is not a valid file descriptor",
            ));
        }
        return Ok((fd, pyre_object::PY_NULL));
    } else {
        w_file
    };
    // `fileno` and `flush` both run Python; keep the file addressable across
    // them so the caller receives the relocated pointer, not a stale one.
    pyre_object::gc_roots::pin_root(resolved);
    let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let method = crate::baseobjspace::getattr_str(resolved, "fileno")?;
    let res = crate::call::call_function_impl_result(method, &[])?;
    if !unsafe { pyre_object::is_int(res) } {
        return Err(crate::PyError::type_error("fileno() returned non-integer"));
    }
    let fd = unsafe { pyre_object::w_int_get_value(res) } as i32;
    // `handler.py:44-48` `try: file.flush() except OperationError: pass`.
    let resolved = pyre_object::gc_roots::shadow_stack_get(file_slot);
    if let Ok(flush) = crate::baseobjspace::getattr_str(resolved, "flush") {
        let _ = crate::call::call_function_impl_result(flush, &[]);
    }
    Ok((fd, pyre_object::gc_roots::shadow_stack_get(file_slot)))
}

pub fn register_module(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(
        ns,
        "enable",
        crate::make_builtin_function_with_signature(
            "enable",
            |args| {
                // `handler.py:141-145 enable` — file=None, all_threads=True.
                let (fd, w_file) = faulthandler_get_fileno_and_file(
                    args.first().copied().unwrap_or(pyre_object::PY_NULL),
                )?;
                #[cfg(all(unix, feature = "host_env"))]
                {
                    // `set_fatal_error_file` allocates its key, so the file has
                    // to stay addressable from here to the store below.
                    let _roots = pyre_object::gc_roots::push_roots();
                    let file_slot = (!w_file.is_null()).then(|| {
                        pyre_object::gc_roots::pin_root(w_file);
                        pyre_object::gc_roots::shadow_stack_len() - 1
                    });
                    // `pypy_faulthandler_enable(fileno, all_threads)` takes the
                    // descriptor as an argument, so the handler never observes
                    // a descriptor the install did not commit to. Split across
                    // two statements here, the new fd has to be visible before
                    // the handlers go in — a fatal signal in between would
                    // otherwise dump to the old one — and has to be rolled back
                    // when the install fails, or a failed re-enable would
                    // redirect the handlers already installed.
                    let previous_fd =
                        FAULTHANDLER_FD.swap(fd, std::sync::atomic::Ordering::Relaxed);
                    let ok = rustpython_host_env::faulthandler::enable_fatal_handlers(
                        faulthandler_signal_handler,
                        libc::SA_NODEFER | libc::SA_ONSTACK,
                    );
                    if ok {
                        FAULTHANDLER_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
                        // `handler.py:145` `self.fatal_error_w_file = w_file`.
                        set_fatal_error_file(file_slot.map_or(
                            pyre_object::PY_NULL,
                            pyre_object::gc_roots::shadow_stack_get,
                        ));
                        return Ok(pyre_object::w_none());
                    }
                    FAULTHANDLER_FD.store(previous_fd, std::sync::atomic::Ordering::Relaxed);
                    return Err(crate::PyError::runtime_error(
                        "faulthandler.enable: sigaction failed",
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = (fd, w_file);
                    Err(crate::PyError::not_implemented(
                        "faulthandler.enable requires host_env feature",
                    ))
                }
            },
            // `enable(file=sys.stderr, all_threads=True, c_stack=True)` —
            // `c_stack` (3.14) selects C-stack dumping; accept and ignore it.
            crate::Signature::new(vec!["file", "all_threads", "c_stack"], None, None, 0, 0),
        ),
    );
    crate::module_ns_store(
        ns,
        "disable",
        crate::make_builtin_function_with_arity(
            "disable",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    rustpython_host_env::faulthandler::disable_fatal_handlers();
                    FAULTHANDLER_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
                    // `handler.py:150` `self.fatal_error_w_file = None`.
                    set_fatal_error_file(pyre_object::PY_NULL);
                }
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "is_enabled",
        crate::make_builtin_function_with_arity(
            "is_enabled",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    return Ok(pyre_object::w_bool_from(
                        FAULTHANDLER_ENABLED.load(std::sync::atomic::Ordering::Relaxed),
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                Ok(pyre_object::w_bool_from(false))
            },
            0,
        ),
    );
    crate::module_ns_store(
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
    crate::module_ns_store(
        ns,
        "dump_traceback_later",
        crate::make_builtin_function("dump_traceback_later", |_| Ok(pyre_object::w_none())),
    );
    crate::module_ns_store(
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
    crate::module_ns_store(
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
                let (fd, _w_file) = faulthandler_get_fileno_and_file(
                    args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
                )?;
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
    crate::module_ns_store(
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
    crate::module_ns_store(
        ns,
        "_read_null",
        crate::make_builtin_function("_read_null", |args| {
            if args.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "_read_null() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            if let Some(&release_gil) = args.first() {
                let _ = crate::baseobjspace::int_w(release_gil)?;
            }
            // `handler.py:225 read_null` — null-pointer deref.
            let p: *const u8 = std::ptr::null();
            let _ = unsafe { p.read_volatile() };
            Ok(pyre_object::w_none())
        }),
    );
    crate::module_ns_store(
        ns,
        "_sigsegv",
        crate::make_builtin_function("_sigsegv", |args| {
            if args.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "_sigsegv() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            if let Some(&release_gil) = args.first() {
                let _ = crate::baseobjspace::int_w(release_gil)?;
            }
            #[cfg(unix)]
            unsafe {
                libc::raise(libc::SIGSEGV);
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::module_ns_store(
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
    crate::module_ns_store(
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
    crate::module_ns_store(
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
