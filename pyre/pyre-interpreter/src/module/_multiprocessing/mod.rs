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
    let d = crate::baseobjspace::getdict(obj);
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
crate::py_class! {
    "SemLock",
    methods: {
        fn acquire(self_obj: PyObjectRef, blocking: Option<i64>) -> Result<bool, crate::PyError> {
            let handle = semlock_get_handle(self_obj);
            if handle.is_null() {
                return Err(crate::PyError::value_error("SemLock handle is null"));
            }
            let blocking = blocking.map(|v| v != 0).unwrap_or(true);
            // PEP 475 — sem_wait/sem_trywait retry on EINTR; otherwise
            // EAGAIN (only meaningful for trywait) yields False and the
            // remaining errnos propagate as OSError instead of being
            // silently mapped to False.
            // `interp_semaphore.py:378-397 semlock_acquire` — on EINTR deliver
            // a pending signal then retry; on success deliver one too before
            // returning (`_check_signals(space)`).
            if blocking {
                loop {
                    let r = unsafe { libc::sem_wait(handle) };
                    if r == 0 {
                        break;
                    }
                    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    if errno == libc::EINTR {
                        crate::module::_signal::interp_signal::checksignals_now()?;
                        continue;
                    }
                    return Err(crate::PyError::os_error_with_errno(errno, "sem_wait"));
                }
                crate::module::_signal::interp_signal::checksignals_now()?;
                Ok(true)
            } else {
                loop {
                    let r = unsafe { libc::sem_trywait(handle) };
                    if r == 0 {
                        crate::module::_signal::interp_signal::checksignals_now()?;
                        return Ok(true);
                    }
                    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    if errno == libc::EINTR {
                        crate::module::_signal::interp_signal::checksignals_now()?;
                        continue;
                    }
                    if errno == libc::EAGAIN {
                        return Ok(false);
                    }
                    return Err(crate::PyError::os_error_with_errno(errno, "sem_trywait"));
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
        fn __exit__(self_obj: PyObjectRef) -> bool {
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
            crate::dict_storage_store(ns, "SemLock", type_object());

            // `_SemLock_new(kind, value, maxvalue, name, unlink)` —
            // Python-side factory; allocates a libc sem_t and stamps
            // its raw pointer onto a fresh SemLock instance.
            crate::dict_storage_store(
                ns,
                "_SemLock_new",
                crate::make_builtin_function("_SemLock_new", |args| {
                    // Fail fast on arity mismatch: declared signature is
                    // (kind, value, maxvalue, name, unlink) with no
                    // optional/positional spillover.
                    if args.len() != 5 {
                        return Err(crate::PyError::type_error(
                            "SemLock() needs (kind, value, maxvalue, name, unlink)",
                        ));
                    }
                    let value = (unsafe { w_int_get_value(args[1]) }) as libc::c_uint;
                    let name = unsafe {
                        if !is_str(args[3]) {
                            return Err(crate::PyError::type_error(
                                "SemLock: name must be a string",
                            ));
                        }
                        w_str_get_value(args[3]).to_string()
                    };
                    let unlink = unsafe { w_int_get_value(args[4]) } != 0;
                    let (handle, _kept_name) =
                        rustpython_host_env::multiprocessing::SemHandle::create(&name, value, unlink)
                            .map_err(|_| crate::PyError::os_error("SemLock create failed"))?;
                    let raw = handle.as_ptr();
                    // SemHandle::Drop closes the sem fd; we cannot let
                    // it run yet because the Python instance still
                    // holds the raw pointer.  The handle currently
                    // leaks per process — a typed-payload migration
                    // (#31-style) would attach a __finalize__ that
                    // calls sem_close on instance death.
                    core::mem::forget(handle);
                    let obj = w_instance_new(type_object());
                    let d = crate::baseobjspace::getdict(obj);
                    if !d.is_null() {
                        unsafe {
                            w_dict_setitem_str(
                                d,
                                "_handle",
                                w_int_new(raw as usize as i64),
                            );
                            w_dict_setitem_str(d, "name", w_str_new(&name));
                        }
                    }
                    Ok(obj)
                }),
            );

            crate::dict_storage_store(
                ns,
                "sem_unlink",
                crate::make_builtin_function_with_arity("sem_unlink", sem_unlink, 1),
            );

            crate::dict_storage_store(
                ns,
                "SEM_VALUE_MAX",
                w_int_new(rustpython_host_env::multiprocessing::sem_value_max() as i64),
            );
            crate::dict_storage_store(ns, "RECURSIVE_MUTEX", w_int_new(0));
            crate::dict_storage_store(ns, "SEMAPHORE", w_int_new(1));
        }
    }
}
