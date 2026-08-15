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

#[cfg(all(any(unix, windows), feature = "host_env"))]
static FAULTHANDLER_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The descriptor `enable` resolved from its `file` argument.  `handler.py`
/// keeps it on the Handler instance, but the signal handler below is a bare
/// `extern "C" fn` and cannot capture, so it is handed over through a static.
/// Defaults to 2, which is what `get_fileno_and_file` answers for None.
#[cfg(all(any(unix, windows), feature = "host_env"))]
static FAULTHANDLER_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(2);

/// `handler.py:145` `self.fatal_error_w_file = w_file` / `handler.py:150`
/// `self.fatal_error_w_file = None`: the descriptor the handler writes to
/// belongs to this object, so it has to outlive the installed handlers rather
/// than be collected and have its finalizer close the fd under them.  pyre has
/// no Handler instance and the signal callback is a bare `extern "C" fn` that
/// cannot capture one, so the owner is a process-global slot, walked as a GC
/// root by [`walk_faulthandler_roots`].
#[cfg(all(any(unix, windows), feature = "host_env"))]
static FAULTHANDLER_FILE: std::sync::atomic::AtomicPtr<pyre_object::PyObject> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Take ownership of the object owning the fatal-error descriptor; a null
/// drops it (`enable` with a plain fd, and `disable`).
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn set_fatal_error_file(w_file: pyre_object::PyObjectRef) {
    FAULTHANDLER_FILE.store(w_file, std::sync::atomic::Ordering::Relaxed);
}

/// `enable` / `disable` / `register` / `unregister` each install (or remove) a
/// handler and then hand the matching file to the owner slot in a second
/// statement.  `Handler` runs those two statements under the GIL, so no other
/// thread can interleave between them; pyre has no GIL, and an interleaving
/// leaves the installed descriptor owned by the *other* call's file — the file
/// this call resolved is then unrooted, collected, and its finalizer closes the
/// descriptor the handler still writes through.  Serialize the pairs.
///
/// Held across the host install or removal and the owner bookkeeping that
/// follows it.  The file is resolved — which runs Python — before the guard is
/// taken, and the fatal-signal handler reads `FAULTHANDLER_FD` atomically
/// without acquiring anything.  A failed install still builds its exception
/// inside the guard, so take it through `lock_faulthandler_state`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
static FAULTHANDLER_STATE_LOCK: parking_lot::Mutex<()> = parking_lot::const_mutex(());

/// Only a contended acquisition blocks, and a thread parked in the futex can no
/// longer poll the eval breaker, so it has to leave the collector's RUNNING
/// census for that wait — otherwise a holder that allocates (`register` builds
/// an `OSError` when the install fails, and `os_error_syscall2` allocates the
/// exception and its args) requests a stop-the-world the waiter can never
/// acknowledge, and both threads hang.  Same try-then-block split as
/// `w_list_lock`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn lock_faulthandler_state() -> parking_lot::MutexGuard<'static, ()> {
    if let Some(guard) = FAULTHANDLER_STATE_LOCK.try_lock() {
        return guard;
    }
    let blocked = crate::module::thread::before_external_block();
    let guard = FAULTHANDLER_STATE_LOCK.lock();
    drop(blocked);
    guard
}

/// `handler.py:22` `self.user_w_files = None` / `:125`
/// `self.user_w_files[signum] = w_file` / `:132`
/// `self.user_w_files.pop(signum, None)`: `register` owns the file per signal
/// for the same reason `enable` owns one, and `unregister` releases it.
/// Addresses rather than `PyObjectRef` so the table is `Sync`; the walker below
/// forwards them.  Process-global, matching one `Handler` per space, and tiny —
/// one entry per registered signal.
#[cfg(all(unix, feature = "host_env"))]
static FAULTHANDLER_USER_FILES: parking_lot::Mutex<Vec<(libc::c_int, usize)>> =
    parking_lot::const_mutex(Vec::new());

/// `handler.py:123-125` — take ownership of a registered signal's file; a null
/// (a plain fd, which `get_fileno_and_file` answers `None` for) drops any
/// previous owner for that signal.
#[cfg(all(unix, feature = "host_env"))]
fn set_user_signal_file(signum: libc::c_int, w_file: pyre_object::PyObjectRef) {
    let mut files = FAULTHANDLER_USER_FILES.lock();
    files.retain(|&(s, _)| s != signum);
    if !w_file.is_null() {
        files.push((signum, w_file as usize));
    }
}

/// `handler.py:131-132` `self.user_w_files.pop(signum, None)`.
#[cfg(all(unix, feature = "host_env"))]
fn clear_user_signal_file(signum: libc::c_int) {
    FAULTHANDLER_USER_FILES.lock().retain(|&(s, _)| s != signum);
}

/// Root walker for the two file-owner tables, registered alongside the other
/// process-global interpreter roots.  Forwards each slot in place so a moving
/// collection relocates the held file rather than leaving a stale address the
/// installed handler would keep writing through.
///
/// Runs from the collector inside the stop-the-world window, so no load/forward/
/// store here can be torn by a concurrent owner update, and it must NOT take
/// `FAULTHANDLER_STATE_LOCK`: the thread that requested the collection may be
/// holding that guard (`register` allocates its `OSError` inside it), and the
/// collector would then wait on a lock only a quiesced mutator can release.
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub fn walk_faulthandler_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let mut forward = |addr: usize| -> usize {
        let mut slot: pyre_object::PyObjectRef = addr as pyre_object::PyObjectRef;
        // SAFETY: `PyObjectRef` and `GcRef` are layout-compatible.
        visitor(unsafe { &mut *(&mut slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef) });
        slot as usize
    };
    let fatal = FAULTHANDLER_FILE.load(std::sync::atomic::Ordering::Relaxed);
    if !fatal.is_null() {
        FAULTHANDLER_FILE.store(
            forward(fatal as usize) as pyre_object::PyObjectRef,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    #[cfg(unix)]
    {
        for entry in FAULTHANDLER_USER_FILES.lock().iter_mut() {
            entry.1 = forward(entry.1);
        }
    }
}

/// No fatal-signal handlers to own a file without a native host environment.
#[cfg(not(all(any(unix, windows), feature = "host_env")))]
pub fn walk_faulthandler_roots(_visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {}

#[cfg(all(any(unix, windows), feature = "host_env"))]
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
    // `pin_root` normalizes a forwarding stub into the slot, not into the
    // caller's copy, so read the pinned value back before using it.
    pyre_object::gc_roots::pin_root(resolved);
    let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let resolved = pyre_object::gc_roots::shadow_stack_get(file_slot);
    let method = crate::baseobjspace::getattr_str(resolved, "fileno")?;
    let res = crate::call::call_function_impl_result(method, &[])?;
    if !unsafe { pyre_object::is_int(res) } {
        return Err(crate::PyError::type_error("fileno() returned non-integer"));
    }
    let fd = unsafe { pyre_object::w_int_get_value(res) } as i32;
    if fd < 0 {
        // `handler.py:42` hands the value straight to
        // `pypy_faulthandler_enable`, and PyPy accepts a negative descriptor
        // (measured: `enable()` succeeds); 3.14 rejects it, with a message
        // naming `fileno()` to distinguish it from the direct-int arm above.
        return Err(crate::PyError::runtime_error(
            "file.fileno() is not a valid file descriptor",
        ));
    }
    // `handler.py:44-48`:
    //
    //     try:
    //         space.call_method(w_file, 'flush')
    //     except OperationError as e:
    //         if e.async(space):
    //             raise
    //         pass   # ignore flush() error
    //
    // 3.14 clears the flush error unconditionally instead, so a `SystemExit`
    // raised out of `flush` does NOT abort `enable` there (measured on 3.14.5
    // and on PyPy: they disagree, and 3.14 is the behaviour target).  Ignore
    // every flush failure, including an asynchronous one.
    let resolved = pyre_object::gc_roots::shadow_stack_get(file_slot);
    let _ = crate::baseobjspace::getattr_str(resolved, "flush")
        .and_then(|flush| crate::call::call_function_impl_result(flush, &[]));
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
                // Resolving the argument runs a user `fileno()` and `flush()`;
                // keep those side effects behind the support gate, so a build
                // that can only answer NotImplementedError does not run them
                // first.
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
                    let (fd, w_file) = faulthandler_get_fileno_and_file(
                        args.first().copied().unwrap_or(pyre_object::PY_NULL),
                    )?;
                    // Nothing else names the resolved file between here and the
                    // owner store below, so keep it rooted across the install.
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
                    let _state = lock_faulthandler_state();
                    let previous_fd =
                        FAULTHANDLER_FD.swap(fd, std::sync::atomic::Ordering::Relaxed);
                    #[cfg(unix)]
                    let flags = libc::SA_NODEFER | libc::SA_ONSTACK;
                    #[cfg(windows)]
                    let flags = 0;
                    let ok = rustpython_host_env::faulthandler::enable_fatal_handlers(
                        faulthandler_signal_handler,
                        flags,
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
                    Err(crate::PyError::runtime_error(
                        "faulthandler.enable: sigaction failed",
                    ))
                }
                #[cfg(not(all(any(unix, windows), feature = "host_env")))]
                {
                    let _ = args;
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
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
                    let _state = lock_faulthandler_state();
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
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
                    Ok(pyre_object::w_bool_from(
                        FAULTHANDLER_ENABLED.load(std::sync::atomic::Ordering::Relaxed),
                    ))
                }
                #[cfg(not(all(any(unix, windows), feature = "host_env")))]
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
                    // Resolving the file runs a user `fileno()` and `flush()`.
                    // `register(signum, file, all_threads, chain)` parses and
                    // coerces the two integer arguments before touching the
                    // file, and a build that can only answer NotImplementedError
                    // must not run those side effects at all — so this sits
                    // after the coercions above and inside the support gate.
                    let (fd, w_file) = faulthandler_get_fileno_and_file(
                        args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
                    )?;
                    // Nothing else names the resolved file between here and the
                    // owner store below, so keep it rooted across the install.
                    let _roots = pyre_object::gc_roots::push_roots();
                    let file_slot = (!w_file.is_null()).then(|| {
                        pyre_object::gc_roots::pin_root(w_file);
                        pyre_object::gc_roots::shadow_stack_len() - 1
                    });
                    let _state = lock_faulthandler_state();
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
                    // `handler.py:123-125` `self.user_w_files[signum] = w_file`,
                    // after `check_err` — a register that failed owns nothing.
                    set_user_signal_file(
                        signum,
                        file_slot.map_or(
                            pyre_object::PY_NULL,
                            pyre_object::gc_roots::shadow_stack_get,
                        ),
                    );
                    Ok(pyre_object::w_none())
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = (signum, all_threads, chain);
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
                    let _state = lock_faulthandler_state();
                    let changed =
                        rustpython_host_env::faulthandler::unregister_user_signal(signum);
                    // `handler.py:131-132` `self.user_w_files.pop(signum, None)`,
                    // run whether or not the signal was registered.
                    clear_user_signal_file(signum);
                    Ok(pyre_object::w_bool_from(changed))
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
