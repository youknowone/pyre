//! _multiprocessing module — PyPy: `pypy/module/_multiprocessing/`.
//!
//! Exposes `SemLock(kind, value, maxvalue, name, unlink)` and
//! `sem_unlink(name)`.  Backed by libc `sem_t` via
//! `rustpython_host_env::multiprocessing`; unix + host_env only — other
//! platforms get an empty module so `import _multiprocessing` succeeds.
//!
//! `W_SemLock`'s fields (`interp_semaphore.py:458-466`) live in the instance
//! dict rather than a typed payload: `handle`, `kind`, `maxvalue` and `name`
//! are the values behind the `GetSetProperty`s of
//! `interp_semaphore.py:593-599`, and `count`/`last_tid` are the recursion
//! bookkeeping `_ismine` reads.  A dict-backed field is also readable as a
//! plain attribute, which is wider than the typedef; the alternative — a
//! handle-keyed side table — has no upstream counterpart.

#[cfg(all(unix, feature = "host_env"))]
use pyre_object::*;

/// `interp_semaphore.py:17 RECURSIVE_MUTEX, SEMAPHORE = range(2)`.
#[cfg(all(unix, feature = "host_env"))]
const RECURSIVE_MUTEX: i64 = 0;
#[cfg(all(unix, feature = "host_env"))]
const SEMAPHORE: i64 = 1;

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

/// Read one of the integer fields of `interp_semaphore.py:458-466`.  A missing
/// or non-int entry reads as 0, which only happens on an instance whose dict a
/// caller has torn up.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_get_i64(obj: PyObjectRef, key: &str) -> i64 {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return 0;
    }
    match unsafe { w_dict_getitem_str(d, key) } {
        Some(v) if unsafe { is_int(v) } => unsafe { w_int_get_value(v) },
        _ => 0,
    }
}

/// Write one of those fields.  Boxing the value and materialising the dict can
/// both collect, so every operand is published and re-read from the shadow
/// stack at the store, exactly as `semlock_instance` does.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_set_i64(obj: PyObjectRef, key: &str, value: i64) {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::pin_roots(&[obj]);
    let boxed_slot = obj_slot + 1;
    pyre_object::gc_roots::pin_root(w_int_new(value));
    let dict =
        crate::baseobjspace::getdict_native(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    if dict.is_null() {
        return;
    }
    let dict_slot = boxed_slot + 1;
    pyre_object::gc_roots::pin_root(dict);
    unsafe {
        w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            key,
            pyre_object::gc_roots::shadow_stack_get(boxed_slot),
        )
    };
}

/// `interp_semaphore.py:486-487 W_SemLock._ismine`.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_ismine(obj: PyObjectRef) -> bool {
    semlock_get_i64(obj, "count") > 0
        && crate::module::thread::current_ident() == semlock_get_i64(obj, "last_tid")
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_post(handle: *mut libc::sem_t) -> Result<(), crate::PyError> {
    if unsafe { libc::sem_post(handle) } != 0 {
        return Err(crate::PyError::os_error_with_errno(
            std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            "sem_post",
        ));
    }
    Ok(())
}

/// `interp_semaphore.py:431-441 semlock_getvalue`.  Not built on darwin, where
/// `sem_getvalue` always fails (`HAVE_BROKEN_SEM_GETVALUE`,
/// `interp_semaphore.py:86-89`) and the `sem_trywait` fallbacks run instead.
#[cfg(all(unix, feature = "host_env", not(target_vendor = "apple")))]
fn semlock_getvalue(handle: *mut libc::sem_t) -> Result<i64, crate::PyError> {
    let mut val: libc::c_int = 0;
    if unsafe { libc::sem_getvalue(handle, &mut val) } != 0 {
        return Err(crate::PyError::os_error_with_errno(
            std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            "sem_getvalue",
        ));
    }
    // some posix implementations use negative numbers to indicate the number of
    // waiting threads
    Ok(if val < 0 { 0 } else { val as i64 })
}

/// `interp_semaphore.py:443-455 semlock_iszero`.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_iszero(handle: *mut libc::sem_t) -> Result<bool, crate::PyError> {
    #[cfg(target_vendor = "apple")]
    {
        if unsafe { libc::sem_trywait(handle) } == 0 {
            semlock_post(handle)?;
            return Ok(false);
        }
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if errno != libc::EAGAIN {
            return Err(crate::PyError::os_error_with_errno(errno, "sem_trywait"));
        }
        Ok(true)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        Ok(semlock_getvalue(handle)? == 0)
    }
}

/// `interp_semaphore.py:357-400 semlock_acquire` — the platform wait alone.
/// Upstream bumps `self.last_tid`/`self.count` here (`:395-396`); the receiver
/// stays with the caller instead, which does it on the success return.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_acquire(
    handle: *mut libc::sem_t,
    block: bool,
    timeout: Option<f64>,
) -> Result<bool, crate::PyError> {
    // PEP 475 — sem_wait/sem_trywait retry on EINTR; otherwise
    // EAGAIN (only meaningful for trywait) yields False and the
    // remaining errnos propagate as OSError instead of being
    // silently mapped to False.
    // `interp_semaphore.py:378-397 semlock_acquire` — on EINTR deliver
    // a pending signal then retry; on success deliver one too before
    // returning (`_check_signals(space)`).
    if block && timeout.is_none() {
        loop {
            let (r, errno) =
                crate::module::thread::call_external_function(|| unsafe { libc::sem_wait(handle) });
            if r == 0 {
                break;
            }
            if errno == libc::EINTR {
                crate::module::signal::interp_signal::checksignals_now()?;
                continue;
            }
            return Err(crate::PyError::os_error_with_errno(errno, "sem_wait"));
        }
        crate::module::signal::interp_signal::checksignals_now()?;
        Ok(true)
    } else if !block {
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
        let deadline = rustpython_host_env::multiprocessing::deadline_from_timeout(
            timeout.unwrap(),
        )
        .map_err(|error| {
            crate::PyError::os_error_with_errno(error.raw_os_error(), error.description())
        })?;
        #[cfg(target_vendor = "apple")]
        {
            let mut delay = 0;
            loop {
                use rustpython_host_env::multiprocessing::PollWaitStep;
                // The poll step sleeps between `sem_trywait` attempts.
                let step = {
                    let _blocked = crate::module::thread::before_external_block();
                    rustpython_host_env::multiprocessing::sem_timedwait_poll_step(
                        handle, &deadline, delay,
                    )
                };
                match step.map_err(|error| {
                    crate::PyError::os_error_with_errno(error.raw_os_error(), error.description())
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
            let status = {
                let _blocked = crate::module::thread::before_external_block();
                rustpython_host_env::multiprocessing::sem_wait_status(handle, Some(&deadline))
            };
            match status {
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

/// `interp_semaphore.py:403-429 semlock_release`.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_release(
    handle: *mut libc::sem_t,
    kind: i64,
    maxvalue: i64,
) -> Result<(), crate::PyError> {
    if kind == RECURSIVE_MUTEX {
        return semlock_post(handle);
    }
    #[cfg(target_vendor = "apple")]
    {
        // `HAVE_BROKEN_SEM_GETVALUE`: only the maxvalue == 1 case can be
        // checked properly.
        if maxvalue == 1 {
            // make sure that already locked
            if unsafe { libc::sem_trywait(handle) } == 0 {
                // it was not locked so undo wait and raise
                let _ = unsafe { libc::sem_post(handle) };
                return Err(crate::PyError::value_error(
                    "semaphore or lock released too many times",
                ));
            }
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            if errno != libc::EAGAIN {
                return Err(crate::PyError::os_error_with_errno(errno, "sem_trywait"));
            }
            // it is already locked as expected
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        // This check is not an absolute guarantee that the semaphore does not
        // rise above maxvalue.
        if semlock_getvalue(handle)? >= maxvalue {
            return Err(crate::PyError::value_error(
                "semaphore or lock released too many times",
            ));
        }
    }
    semlock_post(handle)
}

/// `interp_semaphore.py:506-523 W_SemLock.acquire` — shared by `acquire` and
/// `__enter__`, which the class methods cannot reach through each other.
#[cfg(all(unix, feature = "host_env"))]
fn w_semlock_acquire(
    self_obj: PyObjectRef,
    block: bool,
    timeout: Option<f64>,
) -> Result<bool, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::pin_roots(&[self_obj]);
    // Every field helper can collect — `semlock_get_i64` materialises the
    // instance dict, `semlock_set_i64` boxes the value — so the receiver is
    // read back from its slot at each use rather than kept in a local. Only
    // `pin_roots` normalises a forwarded pointer on the way in; the reads do
    // not, so a stale binding would have them consult a moved object's dict.
    let me = || pyre_object::gc_roots::shadow_stack_get(self_slot);
    // check whether we already own the lock
    if semlock_get_i64(me(), "kind") == RECURSIVE_MUTEX && semlock_ismine(me()) {
        semlock_set_i64(me(), "count", semlock_get_i64(me(), "count") + 1);
        return Ok(true);
    }
    let handle = semlock_get_handle(me());
    if handle.is_null() {
        return Err(crate::PyError::value_error("SemLock handle is null"));
    }
    let got = semlock_acquire(handle, block, timeout)?;
    if got {
        // `interp_semaphore.py:512-516` — these steps need to be as close as
        // possible to acquiring the semlock for `_ismine` to support multiple
        // threads.  The wait can run signal handlers, so the receiver comes
        // back off the shadow stack, and again after the `last_tid` store
        // boxes its value.
        semlock_set_i64(me(), "last_tid", crate::module::thread::current_ident());
        semlock_set_i64(me(), "count", semlock_get_i64(me(), "count") + 1);
    }
    Ok(got)
}

/// `interp_semaphore.py:525-541 W_SemLock.release` — shared by `release` and
/// `__exit__`.
#[cfg(all(unix, feature = "host_env"))]
fn w_semlock_release(self_obj: PyObjectRef) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::pin_roots(&[self_obj]);
    // As in `w_semlock_acquire`: every field helper can collect, so the
    // receiver is read back from its slot at each use.
    let me = || pyre_object::gc_roots::shadow_stack_get(self_slot);
    let kind = semlock_get_i64(me(), "kind");
    if kind == RECURSIVE_MUTEX {
        if !semlock_ismine(me()) {
            return Err(crate::PyError::new(
                crate::error::PyErrorKind::AssertionError,
                "attempt to release recursive lock not owned by thread",
            ));
        }
        let count = semlock_get_i64(me(), "count");
        if count > 1 {
            semlock_set_i64(me(), "count", count - 1);
            return Ok(());
        }
    }
    let handle = semlock_get_handle(me());
    if handle.is_null() {
        return Err(crate::PyError::value_error("SemLock handle is null"));
    }
    semlock_release(handle, kind, semlock_get_i64(me(), "maxvalue"))?;
    semlock_set_i64(me(), "count", semlock_get_i64(me(), "count") - 1);
    Ok(())
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
    // `getdict_native` materialises the instance dict, so it can collect and
    // move `obj`; read the receiver back from its slot the way every store
    // below does, rather than handing over the pre-pin copy.
    let dict =
        crate::baseobjspace::getdict_native(pyre_object::gc_roots::shadow_stack_get(root_base));
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
    // interp_semaphore.py:462,465 `self.count = 0`, `self.last_tid = -1`.
    store!("count", w_int_new(0));
    store!("last_tid", w_int_new(-1));
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
    // interp_semaphore.py:574-575.
    if kind != RECURSIVE_MUTEX && kind != SEMAPHORE {
        return Err(crate::PyError::value_error("unrecognized kind"));
    }
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

/// `interp_semaphore.py:547-561 W_SemLock.rebuild`, registered as a
/// classmethod (`:606`), so `args[0]` is the bound class.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_rebuild(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 5 {
        return Err(crate::PyError::type_error(
            "_rebuild() takes exactly 4 arguments",
        ));
    }
    let w_cls = args[0];
    let kind = crate::baseobjspace::int_w(args[2])?;
    let maxvalue = crate::baseobjspace::int_w(args[3])?;
    // `unwrap_spec(name='text_or_none')` — an unlinked semaphore carries no
    // name and travels as its raw handle instead.
    let name = if unsafe { is_none(args[4]) } {
        None
    } else if unsafe { is_str(args[4]) } {
        Some(crate::baseobjspace::str_utf8_w(args[4])?.to_string())
    } else {
        return Err(crate::PyError::type_error(
            "_rebuild() argument 'name' must be str or None",
        ));
    };
    let raw = match &name {
        // interp_semaphore.py:550-555 — with a name, reopen it and ignore
        // `w_handle`.
        Some(name) => {
            let handle = rustpython_host_env::multiprocessing::SemHandle::open_existing(name)
                .map_err(|error| {
                    crate::PyError::os_error_with_errno(error.raw_os_error(), error.description())
                })?;
            let raw = handle.as_ptr();
            core::mem::forget(handle);
            raw
        }
        // interp_semaphore.py:557 `handle = handle_w(space, w_handle)`
        // (`:223-224`).
        None => crate::baseobjspace::int_w(args[1])? as usize as *mut libc::sem_t,
    };
    semlock_instance(w_cls, raw, kind, maxvalue, name)
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
            let block = blocking.map(|v| v != 0).unwrap_or(true);
            let timeout = match timeout {
                Some(value) if unsafe { !is_none(value) } => {
                    Some(crate::baseobjspace::float_w(value)?)
                }
                _ => None,
            };
            w_semlock_acquire(self_obj, block, timeout)
        }
        fn release(self_obj: PyObjectRef) -> Result<(), crate::PyError> {
            w_semlock_release(self_obj)
        }
        // interp_semaphore.py:483-484 W_SemLock.get_count
        fn _count(self_obj: PyObjectRef) -> i64 {
            semlock_get_i64(self_obj, "count")
        }
        // interp_semaphore.py:489-490 W_SemLock.is_mine
        fn _is_mine(self_obj: PyObjectRef) -> bool {
            semlock_ismine(self_obj)
        }
        // interp_semaphore.py:544-545 W_SemLock.after_fork
        fn _after_fork(self_obj: PyObjectRef) {
            semlock_set_i64(self_obj, "count", 0);
        }
        // interp_semaphore.py:492-497 W_SemLock.is_zero
        fn _is_zero(self_obj: PyObjectRef) -> Result<bool, crate::PyError> {
            let handle = semlock_get_handle(self_obj);
            if handle.is_null() {
                return Err(crate::PyError::value_error("SemLock handle is null"));
            }
            semlock_iszero(handle)
        }
        // interp_semaphore.py:499-504 W_SemLock.get_value
        fn _get_value(self_obj: PyObjectRef) -> Result<i64, crate::PyError> {
            let handle = semlock_get_handle(self_obj);
            if handle.is_null() {
                return Err(crate::PyError::value_error("SemLock handle is null"));
            }
            #[cfg(target_vendor = "apple")]
            {
                // interp_semaphore.py:432-434.
                let _ = handle;
                Err(crate::PyError::not_implemented(
                    "sem_getvalue is not implemented on this system",
                ))
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                semlock_getvalue(handle)
            }
        }
        // interp_semaphore.py:563-564 W_SemLock.enter
        fn __enter__(self_obj: PyObjectRef) -> Result<bool, crate::PyError> {
            w_semlock_acquire(self_obj, true, None)
        }
        // interp_semaphore.py:566-567 W_SemLock.exit
        fn __exit__(
            self_obj: PyObjectRef,
            exc_type: Option<PyObjectRef>,
            exc_value: Option<PyObjectRef>,
            traceback: Option<PyObjectRef>,
        ) -> Result<(), crate::PyError> {
            let _ = (exc_type, exc_value, traceback);
            w_semlock_release(self_obj)
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
                // interp_semaphore.py:606 `as_classmethod=True` — `_rebuild`
                // allocates on the class it is called through.
                pyre_object::w_dict_setitem_str_no_proxy(
                    semlock_ns,
                    "_rebuild",
                    pyre_object::function::w_classmethod_new(crate::make_builtin_function(
                        "_rebuild",
                        semlock_rebuild,
                    )),
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
            crate::module_ns_store(ns, "RECURSIVE_MUTEX", w_int_new(RECURSIVE_MUTEX));
            crate::module_ns_store(ns, "SEMAPHORE", w_int_new(SEMAPHORE));
        }
    }
}
