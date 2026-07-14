//! _signal implementation — PyPy: pypy/module/signal/interp_signal.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use super::signalstate;
use crate::DictStorage;
use crate::executioncontext::{
    AsyncAction, AsyncActionOps, ExecutionContext, PeriodicAsyncAction, PeriodicAsyncActionOps,
};
use crate::pyframe::PyFrame;
use pyre_object::PyObjectRef;

/// executioncontext.py:436-441 `space.getexecutioncontext().checksignals()`
/// — deliver a signal that may have arrived during a syscall.  Resolves
/// the live EC from the thread-local slot (`getexecutioncontext`).
///
/// Called from EINTR retry branches in the blocking-IO modules (socket,
/// select / poll, kqueue, time.sleep, `_multiprocessing` semaphores) so a
/// signal received mid-syscall runs its handler and propagates
/// (e.g. `KeyboardInterrupt`) instead of being swallowed by the retry.
pub fn checksignals_now() -> Result<(), crate::PyError> {
    let ec = crate::call::getexecutioncontext() as *mut ExecutionContext;
    if ec.is_null() {
        return Ok(());
    }
    unsafe { (*ec).checksignals() }
}

/// interp_signal.py:485-497 `SignalMask.__enter__` — unpack a list / tuple
/// / set of signal numbers, the argument shape `sigwait` / `sigpending`
/// share with `pthread_sigmask`.
#[cfg(feature = "host_env")]
fn signal_set_items(arg: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    unsafe {
        if pyre_object::is_list(arg) {
            let n = pyre_object::w_list_len(arg);
            Ok((0..n)
                .filter_map(|i| pyre_object::w_list_getitem(arg, i as i64))
                .collect())
        } else if pyre_object::is_tuple(arg) {
            let n = pyre_object::w_tuple_len(arg);
            Ok((0..n)
                .filter_map(|i| pyre_object::w_tuple_getitem(arg, i as i64))
                .collect())
        } else if pyre_object::is_set_or_frozenset(arg) {
            Ok(pyre_object::w_set_items(arg))
        } else {
            Err(crate::PyError::type_error(
                "argument must be an iterable of signal numbers",
            ))
        }
    }
}

/// error.py `exception_from_saved_errno(space, w_type)` — build an instance
/// of `class_name` (an `OSError` or subclass) carrying the errno and its
/// strerror, so `.errno` / `str()` behave like any `OSError`.  Falls back
/// to `OSError` if `class_name` is not registered.
#[cfg(feature = "host_env")]
fn errno_exception(class_name: &str, errno: i32) -> crate::PyError {
    let strerror = unsafe {
        std::ffi::CStr::from_ptr(libc::strerror(errno))
            .to_string_lossy()
            .into_owned()
    };
    let cls = crate::builtins::lookup_exc_class(class_name)
        .or_else(|| crate::builtins::lookup_exc_class("OSError"))
        .expect("OSError must be installed");
    let args = vec![
        cls,
        pyre_object::w_int_new(errno as i64),
        pyre_object::w_str_new(&strerror),
    ];
    let exc = crate::builtins::exc_os_error_new(&args)
        .expect("exc_os_error_new is infallible for int/str args");
    let mut err = crate::PyError::os_error(&strerror);
    err.exc_object = exc;
    err
}

thread_local! {
    /// interp_signal.py:157-167 `Handlers.handlers_w` — signum → handler
    /// (a callable, or the SIG_DFL/SIG_IGN ints).  A real Python dict so
    /// the GC traces the handler callables; pinned for the process
    /// lifetime.  Missing keys mean SIG_DFL (the pre-fill in PyPy's
    /// `Handlers.__init__`).
    static HANDLERS: std::cell::Cell<PyObjectRef> = const { std::cell::Cell::new(pyre_object::PY_NULL) };
    /// moduledef.py:15 `default_int_handler` — cached so
    /// `getsignal(SIGINT) is signal.default_int_handler` holds after the
    /// startup install.
    static DEFAULT_INT_HANDLER: std::cell::Cell<PyObjectRef> = const { std::cell::Cell::new(pyre_object::PY_NULL) };
}

fn handlers_dict() -> PyObjectRef {
    HANDLERS.with(|cell| {
        let mut d = cell.get();
        if d.is_null() {
            d = pyre_object::w_dict_new();
            pyre_object::gc_roots::pin_root(d);
            cell.set(d);
        }
        d
    })
}

/// interp_signal.py:196-209 `handlers_w[n]` lookup.  Returns `PY_NULL`
/// for a signum with no registered handler (the KeyError → ignore arm).
pub fn get_handler(signum: i32) -> PyObjectRef {
    let d = handlers_dict();
    unsafe { pyre_object::w_dict_getitem(d, signum as i64).unwrap_or(pyre_object::PY_NULL) }
}

/// interp_signal.py:323-325 `handlers_w[signum] = w_handler`.
pub fn set_handler(signum: i32, handler: PyObjectRef) {
    let d = handlers_dict();
    unsafe { pyre_object::w_dict_setitem(d, signum as i64, handler) };
}

/// GC root walker over the signal-handler table and its value slots.
///
/// The HANDLERS dict pointer itself is visited as a root so the GC can
/// relocate it (nursery → oldgen) and update the Cell in place.
/// Without this, a minor collection would move the dict while the Cell
/// retains the stale nursery address, and the next `handlers_dict()`
/// read would dereference freed memory.  The value slots are then
/// walked so handler callables reachable only through this dict survive
/// collection.
pub fn walk_signal_handler_roots(mut visitor: impl FnMut(&mut PyObjectRef)) {
    HANDLERS.with(|cell| {
        let mut d = cell.get();
        if d.is_null() {
            return;
        }
        // Visit the dict pointer itself as a root.  If the GC
        // relocates the dict, `visitor` updates `d` in place;
        // write the (possibly new) address back to the Cell.
        let old = d;
        visitor(&mut d);
        if !std::ptr::eq(d, old) {
            cell.set(d);
        }
        unsafe {
            let strategy = pyre_object::dictmultiobject::w_dict_get_strategy(d);
            strategy.walk_gc_refs(d, &mut |slot: *mut PyObjectRef| {
                visitor(&mut *slot);
            });
        }
    });
}

pub fn capture_signal_handler_root_area() -> *const () {
    HANDLERS.with(|handlers| handlers as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_signal_handler_root_area`], and the owning
/// thread must be quiesced.
pub unsafe fn walk_signal_handler_roots_area(
    data: *const (),
    mut visitor: impl FnMut(&mut PyObjectRef),
) {
    let handlers = unsafe { &*(data as *const std::cell::Cell<PyObjectRef>) };
    let mut dict = handlers.get();
    if dict.is_null() {
        return;
    }
    visitor(&mut dict);
    handlers.set(dict);
    unsafe {
        let strategy = pyre_object::dictmultiobject::w_dict_get_strategy(dict);
        strategy.walk_gc_refs(dict, &mut |slot: *mut PyObjectRef| {
            visitor(&mut *slot);
        });
    }
}

/// `@unwrap_spec(signum=int)` — coerce the signal-number argument to an
/// `i32`.  The gateway `int` converter is `space.gateway_int_w`
/// (`gateway.py:646-665` → `int_w(allow_conversion=True)`), which runs
/// `__index__`/`__int__` and accepts `int` subclasses, so route through
/// the matching helper rather than an exact-tag check.
fn signum_arg(w_signum: PyObjectRef) -> Result<i32, crate::PyError> {
    Ok(crate::baseobjspace::gateway_int_w(w_signum)? as i32)
}

/// interp_signal.py:285-288 `check_signum_in_range`.
fn check_signum_in_range(signum: i32) -> Result<(), crate::PyError> {
    if (1..signalstate::NSIG).contains(&signum) {
        Ok(())
    } else {
        Err(crate::PyError::value_error("signal number out of range"))
    }
}

/// interp_signal.py:291-326 `signal(signum, handler) -> previous`.
fn signal_signal(w_signum: PyObjectRef, w_handler: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let signum = signum_arg(w_signum)?;
    // interp_signal.py:307-310 — `signals_enabled()` is always true in
    // single-threaded pyre, so the main-thread guard is omitted.
    check_signum_in_range(signum)?;

    // interp_signal.py:313-321 — SIG_DFL / SIG_IGN are the ints 0 / 1;
    // anything else must be callable.  PyPy compares with
    // `space.eq_w(w_handler, space.newint(SIG_DFL/SIG_IGN))`, so any
    // equality-compatible object (int subclass, bool, custom `__eq__`)
    // is accepted — use `eq_w` rather than an exact-int read.  The
    // `if / elif` short-circuits: when the SIG_DFL compare is true the
    // SIG_IGN compare never runs, so a handler equal to SIG_DFL whose
    // `__eq__` raises against SIG_IGN is still accepted.
    if crate::baseobjspace::eq_w(w_handler, pyre_object::w_int_new(0))? {
        signalstate::pypysig_default(signum);
    } else if crate::baseobjspace::eq_w(w_handler, pyre_object::w_int_new(1))? {
        signalstate::pypysig_ignore(signum);
    } else if !crate::baseobjspace::callable_w(w_handler) {
        return Err(crate::PyError::type_error(
            "'handler' must be a callable or SIG_DFL or SIG_IGN",
        ));
    } else {
        signalstate::pypysig_setflag(signum);
    }

    // interp_signal.py:323-326 — swap in the new handler, return the old
    // (SIG_DFL when none was registered, matching `Handlers.__init__`).
    let old = get_handler(signum);
    let old = if old.is_null() {
        pyre_object::w_int_new(0)
    } else {
        old
    };
    set_handler(signum, w_handler);
    Ok(old)
}

/// interp_signal.py:238-251 `getsignal(signum) -> action`.
fn signal_getsignal(w_signum: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let signum = signum_arg(w_signum)?;
    check_signum_in_range(signum)?;
    let h = get_handler(signum);
    Ok(if h.is_null() {
        pyre_object::w_int_new(0)
    } else {
        h
    })
}

/// interp_signal.py:254-262 `default_int_handler` — `raise
/// KeyboardInterrupt`.  Cached so the module attribute and the SIGINT
/// handler-dict entry share one identity.
pub fn default_int_handler_obj() -> PyObjectRef {
    DEFAULT_INT_HANDLER.with(|cell| {
        let mut h = cell.get();
        if h.is_null() {
            h = crate::make_builtin_function(
                "default_int_handler",
                // issue #2780: accept and ignore any non-keyword arguments.
                |_| {
                    let cls = crate::builtins::lookup_exc_class("KeyboardInterrupt")
                        .expect("KeyboardInterrupt must be installed");
                    let exc = crate::builtins::exc_exception_new(&[cls])
                        .expect("exc_exception_new is infallible for empty args");
                    Err(unsafe { crate::PyError::from_exc_object(exc) })
                },
            );
            pyre_object::gc_roots::pin_root(h);
            cell.set(h);
        }
        h
    })
}

/// interp_signal.py:54-152 `CheckSignalAction` — a periodic action run
/// whenever the C-level ticker goes negative.  It polls the pending-signal
/// bitmask and invokes the registered app-level handler for each signal,
/// which for SIGINT (default) raises `KeyboardInterrupt`.
pub struct CheckSignalAction {
    base: PeriodicAsyncAction,
    /// interp_signal.py:65 — a signal seen but not yet reported (used by
    /// the threaded fire-in-another-thread path; always -1 in pyre).
    pending_signal: i32,
    /// interp_signal.py:66/103 — re-entrancy guard so a handler that
    /// itself triggers a checkpoint does not recurse into polling.
    in_poll: bool,
}

impl CheckSignalAction {
    /// interp_signal.py:62-86 `CheckSignalAction.__init__`.
    pub fn new(space: PyObjectRef) -> Box<Self> {
        Box::new(Self {
            base: PeriodicAsyncAction {
                base: AsyncAction::new_periodic_base(space),
            },
            pending_signal: -1,
            in_poll: false,
        })
    }

    /// interp_signal.py:101-109 `_poll_for_signals` — the re-entrancy
    /// guard around the unlocked poll.
    fn poll_for_signals(&mut self, ec: &mut ExecutionContext) -> Result<(), crate::PyError> {
        if self.in_poll {
            return Ok(());
        }
        self.in_poll = true;
        let result = self.poll_for_signals_unlocked(ec);
        self.in_poll = false;
        result
    }

    /// interp_signal.py:111-141 `_poll_for_signals_unlocked`.  pyre is
    /// single-threaded, so `signals_enabled()` is always true and the
    /// fire-in-another-thread branch is unreachable; the remote-debugger
    /// arm is not surfaced.
    fn poll_for_signals_unlocked(
        &mut self,
        ec: &mut ExecutionContext,
    ) -> Result<(), crate::PyError> {
        // interp_signal.py:112-115 — report any wakeup-fd write error the
        // async handler stashed, before polling pending signals.
        let werr = signalstate::get_wakeup_fd_write_errno();
        if werr != 0 {
            report_wakeup_fd_error(werr);
        }
        let mut n = self.pending_signal;
        if n < 0 {
            n = signalstate::signal_poll();
        }
        while n >= 0 {
            self.pending_signal = -1;
            report_signal(ec, n)?;
            n = self.pending_signal;
            if n < 0 {
                n = signalstate::signal_poll();
            }
        }
        Ok(())
    }
}

/// interp_signal.py:196-209 `report_signal`.
fn report_signal(ec: &mut ExecutionContext, n: i32) -> Result<(), crate::PyError> {
    let w_handler = get_handler(n);
    if w_handler.is_null() {
        return Ok(()); // no handler, ignore signal
    }
    if !crate::baseobjspace::callable_w(w_handler) {
        return Ok(()); // w_handler is SIG_IGN or SIG_DFL (an int)
    }
    // interp_signal.py:205 — re-install for OSes that clear the handler
    // (no-op on SA_RESTART platforms).
    signalstate::pypysig_reinstall(n);
    // interp_signal.py:207-209 — call the handler with (signum, frame).
    // pyre does not wrap the executing frame as a Python object, so the
    // second argument is None (a valid value per the language spec when
    // the frame is unavailable).
    let _ = ec;
    let w_n = pyre_object::w_int_new(n as i64);
    let w_frame = pyre_object::w_none();
    let res = crate::baseobjspace::call_function(w_handler, &[w_n, w_frame]);
    if res.is_null() {
        if let Some(err) = crate::call::take_call_error() {
            return Err(err);
        }
    }
    Ok(())
}

/// interp_signal.py:169-193 `_report_wakeup_fd_error` — surface the errno
/// of a failed wakeup-fd write.  PyPy reports it through
/// `write_unraisable_default`; pyre has no unraisable hook, so it writes
/// the equivalent `OSError` line to stderr directly.
fn report_wakeup_fd_error(errno_val: i32) {
    #[cfg(unix)]
    let msg = unsafe {
        let p = libc::strerror(errno_val);
        if p.is_null() {
            format!("error {errno_val}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    #[cfg(not(unix))]
    let msg = format!("error {errno_val}");
    crate::host_seam::emit_stderr(
        b"Exception ignored when trying to write to the signal wakeup fd:\n",
    );
    crate::host_seam::emit_stderr(format!("OSError: [Errno {errno_val}] {msg}\n").as_bytes());
}

impl AsyncActionOps for CheckSignalAction {
    /// interp_signal.py:94-99 `perform`.  The
    /// `w_async_exception_type` arm only fires across threads, which pyre
    /// does not have, so it stays a no-op guard and we proceed straight
    /// to polling.
    fn perform(
        &mut self,
        ec: &mut ExecutionContext,
        _frame: *mut PyFrame,
    ) -> Result<(), crate::PyError> {
        self.poll_for_signals(ec)
    }

    fn async_action(&self) -> &AsyncAction {
        &self.base.base
    }

    fn async_action_mut(&mut self) -> &mut AsyncAction {
        &mut self.base.base
    }
}

impl PeriodicAsyncActionOps for CheckSignalAction {}

/// moduledef.py:62-66 + app_main.py:926 — install the signal-checking
/// action on the execution context and route SIGINT to
/// `default_int_handler` (so Ctrl-C raises `KeyboardInterrupt`).  Called
/// once at interpreter startup.  Idempotent: a second call is a no-op.
pub fn install_signal_handling(ec: &mut ExecutionContext) {
    if ec.check_signal_action.is_some() {
        return;
    }
    // moduledef.py:64-66 — register the periodic signal-check action.
    // The action outlives the call (the EC and actionflag hold pointers
    // into it for the whole run), so it is leaked deliberately.
    let action: &'static mut CheckSignalAction = Box::leak(CheckSignalAction::new(ec.space));
    let async_ptr: *mut dyn AsyncActionOps = &mut *action;
    action.register_periodic_action(&mut ec.actionflag, false);
    ec.check_signal_action = Some(async_ptr);

    // Hand the ticker cell address to the OS handler so it can force the
    // ticker negative (rsignal.py:31-32 `pypysig_getaddr_occurred`).
    let ticker_addr = ec.actionflag.ticker_addr();
    signalstate::register_ticker(ticker_addr);

    // app_main.py:926 — `signal.signal(SIGINT, default_int_handler)`.
    #[cfg(unix)]
    {
        let sigint = libc::SIGINT;
        if signalstate::pypysig_setflag(sigint) {
            set_handler(sigint, default_int_handler_obj());
        }
    }
}

/// _signal module — PyPy: pypy/module/signal/.
///
/// `signal()` / `getsignal()` register real handlers (sigaction +
/// pending-flag, delivered by `CheckSignalAction`).  `set_wakeup_fd`
/// records the fd but the handler does not yet write to it.  alarm /
/// pause / raise_signal / strsignal / valid_signals are backed by
/// `rustpython_host_env::signal`.  Signal-number constants are sourced
/// from `libc::*` so they match the host's POSIX numbering (the previous
/// macOS-flavoured hard-coded list disagreed with Linux for
/// SIGUSR1/SIGUSR2/SIGCHLD).
pub fn register_module(ns: &mut DictStorage) {
    // interp_signal.py:291-326 `signal(signum, handler) -> previous`.
    crate::dict_storage_store(
        ns,
        "signal",
        crate::make_builtin_function_with_arity("signal", |args| signal_signal(args[0], args[1]), 2),
    );
    // interp_signal.py:238-251 `getsignal(signum) -> action`.
    crate::dict_storage_store(
        ns,
        "getsignal",
        crate::make_builtin_function_with_arity("getsignal", |args| signal_getsignal(args[0]), 1),
    );
    // `interp_signal.py:default_int_handler` — `raise KeyboardInterrupt`.
    // Shares one identity with the SIGINT handler installed at startup so
    // `getsignal(SIGINT) is signal.default_int_handler`.
    crate::dict_storage_store(ns, "default_int_handler", default_int_handler_obj());
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
            // interp_signal.py:330-331 — `set_wakeup_fd(fd, *,
            // warn_on_full_buffer=True)`: the flag is keyword-only, so a
            // second positional argument is rejected.
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(
                kwargs,
                &["warn_on_full_buffer"],
                "set_wakeup_fd",
            )?;
            if positional.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "set_wakeup_fd() takes 1 positional argument but {} were given",
                    positional.len()
                )));
            }
            let warn_on_full_buffer = crate::builtins::kwarg_get(kwargs, "warn_on_full_buffer")
                .map(crate::baseobjspace::is_true)
                .transpose()?
                .unwrap_or(true);
            let fd = if let Some(&a) = positional.first() {
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
            // interp_signal.py:343-360 — a real fd is validated with
            // `os.fstat` then `get_status_flags`: a bad fd is a ValueError
            // and the fd must already be in non-blocking mode.
            if fd != -1 {
                #[cfg(unix)]
                unsafe {
                    let mut st: libc::stat = std::mem::zeroed();
                    let bad_fd = libc::fstat(fd, &mut st) != 0;
                    let flags = if bad_fd {
                        -1
                    } else {
                        libc::fcntl(fd, libc::F_GETFL)
                    };
                    if bad_fd || flags < 0 {
                        let e = std::io::Error::last_os_error();
                        if e.raw_os_error() == Some(libc::EBADF) {
                            return Err(crate::PyError::value_error("invalid fd"));
                        }
                        return Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("{e}"),
                        ));
                    }
                    if flags & libc::O_NONBLOCK == 0 {
                        return Err(crate::PyError::value_error(format!(
                            "the fd {fd} must be in non-blocking mode"
                        )));
                    }
                }
                #[cfg(not(unix))]
                if fd < -1 {
                    return Err(crate::PyError::value_error(
                        "set_wakeup_fd(): fd must be -1 or a valid file descriptor",
                    ));
                }
            }
            // interp_signal.py:376 — `pypysig_set_wakeup_fd`.  The OS
            // handler writes the signal-number byte to this fd so a
            // select/poll loop blocked elsewhere wakes up.
            let prev = signalstate::set_wakeup_fd(fd, warn_on_full_buffer);
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
                #[cfg(feature = "sandbox")]
                {
                    let _ = args;
                    return Err(crate::host_seam::stub("signal.raise_signal"));
                }
                #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                {
                    let signum = if let Some(&a) = args.first() {
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(crate::PyError::type_error(
                            "raise_signal() missing argument",
                        ));
                    };
                    match rustpython_host_env::signal::raise_signal(signum) {
                        Ok(()) => {
                            // interp_signal.py:583-584 — the signal may
                            // have been delivered to this thread; run the
                            // pending handler now (may raise).
                            checksignals_now()?;
                            return Ok(pyre_object::w_none());
                        }
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
    // moduledef.py:17 `'ItimerError': 'interp_signal.get_itimer_error(space)'`
    // — `signal.new_exception_class("signal.ItimerError", space.w_IOError)`.
    // An OSError subclass so `setitimer`'s `exception_from_saved_errno`
    // instance carries errno / strerror.
    let w_os_error = crate::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before _signal init");
    crate::dict_storage_store(
        ns,
        "ItimerError",
        crate::builtins::make_exc_type(
            "signal.ItimerError",
            crate::builtins::exc_os_error_new,
            w_os_error,
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
                    #[cfg(feature = "sandbox")]
                    {
                        let _ = args;
                        return Err(crate::host_seam::stub("signal.alarm"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
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
                    #[cfg(feature = "sandbox")]
                    {
                        return Err(crate::host_seam::stub("signal.pause"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
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
                #[cfg(feature = "sandbox")]
                {
                    let _ = args;
                    return Err(crate::host_seam::stub("signal.setitimer"));
                }
                #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
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
                    let old = rustpython_host_env::signal::setitimer(which, &new_value)
                        .map_err(|e| {
                            errno_exception("signal.ItimerError", e.raw_os_error().unwrap_or(0))
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
                    #[cfg(feature = "sandbox")]
                    {
                        let _ = args;
                        return Err(crate::host_seam::stub("signal.getitimer"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
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
        // sigwait(sigset) -> signum — interp_signal.py:515-524
        crate::dict_storage_store(
            ns,
            "sigwait",
            crate::make_builtin_function_with_arity(
                "sigwait",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sigwait() takes exactly one argument (0 given)",
                            ));
                        }
                        let mut set =
                            rustpython_host_env::signal::sigemptyset().map_err(|e| {
                                crate::PyError::os_error_with_errno(
                                    e.raw_os_error().unwrap_or(0),
                                    format!("sigemptyset: {e}"),
                                )
                            })?;
                        for it in signal_set_items(args[0])? {
                            let signum = (unsafe { pyre_object::w_int_get_value(it) }) as i32;
                            // interp_signal.py:285-288 check_signum_in_range
                            if !(1..signalstate::NSIG).contains(&signum) {
                                return Err(crate::PyError::value_error(
                                    "signal number out of range",
                                ));
                            }
                            rustpython_host_env::signal::sigaddset(&mut set, signum).map_err(
                                |e| {
                                    crate::PyError::os_error_with_errno(
                                        e.raw_os_error().unwrap_or(0),
                                        format!("sigaddset: {e}"),
                                    )
                                },
                            )?;
                        }
                        let mut signum: libc::c_int = 0;
                        // sigwait returns the error number directly, not via errno.
                        let ret = unsafe { libc::sigwait(&set, &mut signum) };
                        if ret != 0 {
                            return Err(errno_exception("OSError", ret));
                        }
                        return Ok(pyre_object::w_int_new(signum as i64));
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.sigwait requires host_env feature",
                        ))
                    }
                },
                1,
            ),
        );
        // sigpending() -> set of pending signals — interp_signal.py:526-535
        crate::dict_storage_store(
            ns,
            "sigpending",
            crate::make_builtin_function_with_arity(
                "sigpending",
                |_args| {
                    #[cfg(feature = "host_env")]
                    {
                        let mut mask: libc::sigset_t = unsafe { std::mem::zeroed() };
                        let ret = unsafe { libc::sigpending(&mut mask) };
                        if ret != 0 {
                            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                            return Err(errno_exception("OSError", errno));
                        }
                        // interp_signal.py:502-513 _sigset_to_signals
                        let items: Vec<pyre_object::PyObjectRef> = (1..signalstate::NSIG)
                            .filter(|s| {
                                rustpython_host_env::signal::sigset_contains(mask, *s as i32)
                            })
                            .map(|s| pyre_object::w_int_new(s as i64))
                            .collect();
                        return Ok(pyre_object::w_set_from_items(&items));
                    }
                    #[cfg(not(feature = "host_env"))]
                    Err(crate::PyError::not_implemented(
                        "signal.sigpending requires host_env feature",
                    ))
                },
                0,
            ),
        );
        // pthread_kill(tid, signum) -> None — interp_signal.py:466-474
        crate::dict_storage_store(
            ns,
            "pthread_kill",
            crate::make_builtin_function_with_arity(
                "pthread_kill",
                |args| {
                    #[cfg(feature = "sandbox")]
                    {
                        let _ = args;
                        return Err(crate::host_seam::stub("signal.pthread_kill"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
                    {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "pthread_kill() takes exactly 2 arguments",
                            ));
                        }
                        let tid =
                            (unsafe { pyre_object::w_int_get_value(args[0]) }) as u64;
                        let signum =
                            (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                        let ret =
                            unsafe { libc::pthread_kill(tid as libc::pthread_t, signum) };
                        if ret != 0 {
                            return Err(errno_exception("OSError", ret));
                        }
                        // interp_signal.py:473-474 — the signal may have been
                        // sent to the current thread.
                        checksignals_now()?;
                        return Ok(pyre_object::w_none());
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.pthread_kill requires host_env feature",
                        ))
                    }
                },
                2,
            ),
        );
        // pthread_sigmask(how, mask) -> previous mask (set of signums)
        crate::dict_storage_store(
            ns,
            "pthread_sigmask",
            crate::make_builtin_function_with_arity(
                "pthread_sigmask",
                |args| {
                    #[cfg(feature = "sandbox")]
                    {
                        let _ = args;
                        return Err(crate::host_seam::stub("signal.pthread_sigmask"));
                    }
                    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
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
                        // interp_signal.py:546-547 — if signals were
                        // unblocked, their handlers may now be pending.
                        checksignals_now()?;
                        let out: Vec<pyre_object::PyObjectRef> = (1..=64)
                            .filter(|s| {
                                rustpython_host_env::signal::sigset_contains(prev, *s as i32)
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
                // Delivers a signal cross-process via a direct syscall, bypassing
                // the controller; the `kill`/`killpg` twins are already stubbed.
                #[cfg(feature = "sandbox")]
                {
                    let _ = args;
                    return Err(crate::host_seam::stub("signal.pidfd_send_signal"));
                }
                #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
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
