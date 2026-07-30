//! _multiprocessing module — PyPy: `pypy/module/_multiprocessing/`.
//!
//! Exposes `SemLock(kind, value, maxvalue, name, unlink)` and
//! `sem_unlink(name)`.  Single-threaded pyre still needs the methods to
//! exist so multiprocessing.py teardown survives.  Backed by libc
//! `sem_t` via `rustpython_host_env::multiprocessing`; unix + host_env
//! only — other platforms get an empty module so `import
//! _multiprocessing` succeeds.

#[cfg(all(unix, feature = "host_env"))]
use pyre_object::*;

#[cfg(all(unix, feature = "host_env"))]
fn semlock_get_handle(obj: PyObjectRef) -> *mut libc::sem_t {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return core::ptr::null_mut();
    }
    if let Some(v) = unsafe { w_dict_getitem_str(d, "_handle") } {
        if unsafe { is_int(v) } {
            return unsafe { w_int_get_value(v) } as usize as *mut libc::sem_t;
        }
    }
    core::ptr::null_mut()
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_instance(
    w_subtype: PyObjectRef,
    raw: *mut libc::sem_t,
    kind: i64,
    maxvalue: i64,
    kept_name: Option<String>,
) -> Result<PyObjectRef, crate::PyError> {
    let obj = w_instance_new(w_subtype);
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(obj);
    let dict = crate::baseobjspace::getdict_native(obj);
    if dict.is_null() {
        return Err(crate::PyError::runtime_error(
            "SemLock instance has no storage",
        ));
    }
    pyre_object::gc_roots::pin_root(dict);
    macro_rules! store {
        ($name:literal, $value:expr) => {{
            let value = $value;
            unsafe {
                w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(root_base + 1),
                    $name,
                    value,
                )
            };
        }};
    }
    store!("_handle", w_int_new(raw as usize as i64));
    store!("handle", w_int_new(raw as usize as i64));
    store!("kind", w_int_new(kind));
    store!("maxvalue", w_int_new(maxvalue));
    store!(
        "name",
        kept_name.map_or_else(w_none, |name| w_str_new(&name))
    );
    Ok(pyre_object::gc_roots::shadow_stack_get(root_base))
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 6 {
        return Err(crate::PyError::type_error(
            "SemLock() needs (kind, value, maxvalue, name, unlink)",
        ));
    }
    let w_subtype = args[0];
    let kind = crate::baseobjspace::int_w(args[1])?;
    let value = crate::baseobjspace::int_w(args[2])?;
    let maxvalue = crate::baseobjspace::int_w(args[3])?;
    let name = if unsafe { is_str(args[4]) } {
        crate::baseobjspace::str_utf8_w(args[4])?.to_string()
    } else {
        return Err(crate::PyError::type_error("SemLock: name must be a string"));
    };
    let unlink = crate::baseobjspace::is_true(args[5])?;
    let (handle, kept_name) = rustpython_host_env::multiprocessing::SemHandle::create(
        &name,
        value as libc::c_uint,
        unlink,
    )
    .map_err(|error| {
        crate::PyError::os_error_with_errno(error.raw_os_error(), error.description())
    })?;
    let raw = handle.as_ptr();
    // SemHandle::Drop closes the semaphore. Ownership belongs to the Python
    // W_SemLock until its registered finalizer grows a typed payload.
    core::mem::forget(handle);
    semlock_instance(w_subtype, raw, kind, maxvalue, kept_name)
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_rebuild(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 4 {
        return Err(crate::PyError::type_error(
            "_rebuild() takes exactly 4 arguments",
        ));
    }
    let kind = crate::baseobjspace::int_w(args[1])?;
    let maxvalue = crate::baseobjspace::int_w(args[2])?;
    let name = if unsafe { is_str(args[3]) } {
        crate::baseobjspace::str_utf8_w(args[3])?.to_string()
    } else {
        return Err(crate::PyError::type_error(
            "SemLock._rebuild requires a semaphore name",
        ));
    };
    let handle =
        rustpython_host_env::multiprocessing::SemHandle::open_existing(&name).map_err(|error| {
            crate::PyError::os_error_with_errno(error.raw_os_error(), error.description())
        })?;
    let raw = handle.as_ptr();
    core::mem::forget(handle);
    semlock_instance(type_object(), raw, kind, maxvalue, Some(name))
}

#[cfg(all(unix, feature = "host_env"))]
crate::py_class! {
    "SemLock",
    methods: {
        fn acquire(
            self_obj: PyObjectRef,
            blocking: Option<i64>,
            timeout: Option<PyObjectRef>,
        ) -> Result<bool, crate::PyError> {
            let handle = semlock_get_handle(self_obj);
            if handle.is_null() {
                return Err(crate::PyError::value_error("SemLock handle is null"));
            }
            let blocking = blocking.map(|v| v != 0).unwrap_or(true);
            let timeout = match timeout {
                Some(value) if unsafe { !is_none(value) } => {
                    Some(crate::baseobjspace::float_w(value)?)
                }
                _ => None,
            };
            // PEP 475 — sem_wait/sem_trywait retry on EINTR; otherwise
            // EAGAIN (only meaningful for trywait) yields False and the
            // remaining errnos propagate as OSError instead of being
            // silently mapped to False.
            // `interp_semaphore.py:378-397 semlock_acquire` — on EINTR deliver
            // a pending signal then retry; on success deliver one too before
            // returning (`_check_signals(space)`).
            if blocking && timeout.is_none() {
                loop {
                    let r = unsafe { libc::sem_wait(handle) };
                    if r == 0 {
                        break;
                    }
                    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    if errno == libc::EINTR {
                        crate::module::signal::interp_signal::checksignals_now()?;
                        continue;
                    }
                    return Err(crate::PyError::os_error_with_errno(errno, "sem_wait"));
                }
                crate::module::signal::interp_signal::checksignals_now()?;
                Ok(true)
            } else if !blocking {
                loop {
                    let r = unsafe { libc::sem_trywait(handle) };
                    if r == 0 {
                        crate::module::signal::interp_signal::checksignals_now()?;
                        return Ok(true);
                    }
                    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    if errno == libc::EINTR {
                        crate::module::signal::interp_signal::checksignals_now()?;
                        continue;
                    }
                    if errno == libc::EAGAIN {
                        return Ok(false);
                    }
                    return Err(crate::PyError::os_error_with_errno(errno, "sem_trywait"));
                }
            } else {
                let deadline =
                    rustpython_host_env::multiprocessing::deadline_from_timeout(timeout.unwrap())
                        .map_err(|error| {
                            crate::PyError::os_error_with_errno(
                                error.raw_os_error(),
                                error.description(),
                            )
                        })?;
                #[cfg(target_vendor = "apple")]
                {
                    let mut delay = 0;
                    loop {
                        use rustpython_host_env::multiprocessing::PollWaitStep;
                        match rustpython_host_env::multiprocessing::sem_timedwait_poll_step(
                            handle, &deadline, delay,
                        )
                        .map_err(|error| {
                            crate::PyError::os_error_with_errno(
                                error.raw_os_error(),
                                error.description(),
                            )
                        })? {
                            PollWaitStep::Acquired => {
                                crate::module::signal::interp_signal::checksignals_now()?;
                                return Ok(true);
                            }
                            PollWaitStep::Timeout => return Ok(false),
                            PollWaitStep::Continue(next_delay) => delay = next_delay,
                        }
                    }
                }
                #[cfg(not(target_vendor = "apple"))]
                loop {
                    use rustpython_host_env::multiprocessing::WaitStatus;
                    match rustpython_host_env::multiprocessing::sem_wait_status(
                        handle,
                        Some(&deadline),
                    ) {
                        WaitStatus::Acquired => {
                            crate::module::signal::interp_signal::checksignals_now()?;
                            return Ok(true);
                        }
                        WaitStatus::TimedOut => return Ok(false),
                        WaitStatus::Interrupted => {
                            crate::module::signal::interp_signal::checksignals_now()?;
                        }
                        WaitStatus::Error(error) => {
                            return Err(crate::PyError::os_error_with_errno(
                                error.raw_os_error(),
                                error.description(),
                            ));
                        }
                    }
                }
            }
        }
        fn release(self_obj: PyObjectRef) -> Result<(), crate::PyError> {
            let handle = semlock_get_handle(self_obj);
            if handle.is_null() {
                return Err(crate::PyError::value_error("SemLock handle is null"));
            }
            let r = unsafe { libc::sem_post(handle) };
            if r != 0 {
                return Err(crate::PyError::os_error_with_errno(
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                    "sem_post",
                ));
            }
            Ok(())
        }
        fn _count(self_obj: PyObjectRef) -> i64 {
            let _ = self_obj;
            0
        }
        fn _is_mine(self_obj: PyObjectRef) -> bool {
            let _ = self_obj;
            false
        }
        fn _after_fork(self_obj: PyObjectRef) {
            let _ = self_obj;
        }
        // `sem_getvalue` isn't available on macOS; just return false —
        // multiprocessing.Queue teardown is the only consumer and it
        // tolerates a conservative "not zero" answer.
        fn _is_zero(self_obj: PyObjectRef) -> bool {
            let handle = semlock_get_handle(self_obj);
            handle.is_null()
        }
        fn __enter__(self_obj: PyObjectRef) -> PyObjectRef {
            let handle = semlock_get_handle(self_obj);
            if !handle.is_null() {
                let _ = unsafe { libc::sem_wait(handle) };
            }
            self_obj
        }
        fn __exit__(
            self_obj: PyObjectRef,
            exc_type: Option<PyObjectRef>,
            exc_value: Option<PyObjectRef>,
            traceback: Option<PyObjectRef>,
        ) -> bool {
            let _ = (exc_type, exc_value, traceback);
            let handle = semlock_get_handle(self_obj);
            if !handle.is_null() {
                let _ = unsafe { libc::sem_post(handle) };
            }
            false
        }
    }
}

#[cfg(all(unix, feature = "host_env"))]
#[crate::pyre_function]
fn sem_unlink(name: &str) -> Result<(), crate::PyError> {
    rustpython_host_env::multiprocessing::sem_unlink(name)
        .map_err(|_| crate::PyError::os_error("sem_unlink failed"))
}

crate::py_module! {
    "_multiprocessing",
    extra_init: |ns| {
        #[cfg(all(unix, feature = "host_env"))]
        {
            let semlock_type = type_object();
            crate::module_ns_store(ns, "SemLock", semlock_type);
            // interp_semaphore.py:593-610 W_SemLock.typedef publishes this
            // constant on the class (the module also exports its own copy).
            let sem_value_max =
                w_int_new(rustpython_host_env::multiprocessing::sem_value_max() as i64);
            let semlock_ns =
                unsafe { pyre_object::w_type_get_dict_ptr(semlock_type) } as PyObjectRef;
            unsafe {
                pyre_object::w_dict_setitem_str_no_proxy(
                    semlock_ns,
                    "SEM_VALUE_MAX",
                    sem_value_max,
                );
                pyre_object::w_dict_setitem_str_no_proxy(
                    semlock_ns,
                    "__new__",
                    crate::make_builtin_function("__new__", semlock_descr_new),
                );
                pyre_object::w_dict_setitem_str_no_proxy(
                    semlock_ns,
                    "_rebuild",
                    crate::make_builtin_function("_rebuild", semlock_rebuild),
                );
            }

            crate::module_ns_store(
                ns,
                "sem_unlink",
                crate::make_builtin_function_with_arity("sem_unlink", sem_unlink, 1),
            );

            crate::module_ns_store(
                ns,
                "SEM_VALUE_MAX",
                sem_value_max,
            );
            crate::module_ns_store(ns, "RECURSIVE_MUTEX", w_int_new(0));
            crate::module_ns_store(ns, "SEMAPHORE", w_int_new(1));
        }
    }
}
