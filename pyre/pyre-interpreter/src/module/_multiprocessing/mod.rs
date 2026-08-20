//! _multiprocessing module — PyPy: `pypy/module/_multiprocessing/`.
//!
//! Exposes `SemLock(kind, value, maxvalue, name, unlink)` and
//! `sem_unlink(name)`, plus the three socket calls `connection.py` reaches for
//! on Windows.  Backed by `rustpython_host_env::multiprocessing` — libc
//! `sem_t` on unix, a `CreateSemaphoreW` handle on Windows; host_env only, so
//! other platforms get an empty module and `import _multiprocessing`
//! still succeeds.
//!
//! `W_SemLock`'s fields (`interp_semaphore.py:458-466`) live in the instance
//! dict rather than a typed payload: `handle`, `kind`, `maxvalue` and `name`
//! are the values behind the `GetSetProperty`s of
//! `interp_semaphore.py:593-599`, and `count`/`last_tid` are the recursion
//! bookkeeping `_ismine` reads.  A dict-backed field is also readable as a
//! plain attribute, which is wider than the typedef; the alternative — a
//! handle-keyed side table — has no upstream counterpart.

#[cfg(all(any(unix, windows), feature = "host_env"))]
use pyre_object::*;

#[cfg(all(any(unix, windows), feature = "host_env"))]
use rustpython_host_env::multiprocessing as host_mp;

/// The platform's semaphore, as the instance stores it: an integer `handle`
/// that `_rebuild` takes back.  Both spellings are raw pointers, so the
/// round trip through `usize` is the same on either.
#[cfg(all(unix, feature = "host_env"))]
type SemRaw = *mut libc::sem_t;
#[cfg(all(windows, feature = "host_env"))]
type SemRaw = host_mp::RawHandle;

/// The Win32 code the last call left behind, as an `OSError` carrying it in
/// `.winerror` (`PyErr_SetExcFromWindowsErr`).
#[cfg(all(windows, feature = "host_env"))]
fn last_windows_error() -> crate::PyError {
    windows_error(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
}

#[cfg(all(windows, feature = "host_env"))]
fn windows_error(winerror: i32) -> crate::PyError {
    crate::PyError::os_error_win32_syscall2(winerror, PY_NULL, PY_NULL)
}

/// `interp_semaphore.py:17 RECURSIVE_MUTEX, SEMAPHORE = range(2)`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
const RECURSIVE_MUTEX: i64 = 0;
#[cfg(all(any(unix, windows), feature = "host_env"))]
const SEMAPHORE: i64 = 1;

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_get_handle(obj: PyObjectRef) -> SemRaw {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return core::ptr::null_mut();
    }
    if let Some(v) = unsafe { w_dict_getitem_str(d, "_handle") }
        && unsafe { is_int(v) }
    {
        return unsafe { w_int_get_value(v) } as usize as SemRaw;
    }
    core::ptr::null_mut()
}

/// Read one of the integer fields of `interp_semaphore.py:458-466`.  A missing
/// or non-int entry reads as 0, which only happens on an instance whose dict a
/// caller has torn up.
#[cfg(all(any(unix, windows), feature = "host_env"))]
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
#[cfg(all(any(unix, windows), feature = "host_env"))]
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
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_ismine(obj: PyObjectRef) -> bool {
    semlock_get_i64(obj, "count") > 0
        && crate::module::thread::current_ident() == semlock_get_i64(obj, "last_tid")
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_post(handle: SemRaw) -> Result<(), crate::PyError> {
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
fn semlock_getvalue(handle: SemRaw) -> Result<i64, crate::PyError> {
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
fn semlock_iszero(handle: SemRaw) -> Result<bool, crate::PyError> {
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

/// The value the semaphore currently holds, for `_get_value`.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_value(handle: SemRaw) -> Result<i64, crate::PyError> {
    // interp_semaphore.py:432-434.
    #[cfg(target_vendor = "apple")]
    {
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

/// `semaphore.c semlock_getvalue`'s Windows arm: take the count down by one
/// and give it straight back, which is the only way to read it.  A wait that
/// times out is a semaphore holding nothing.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_value(handle: SemRaw) -> Result<i64, crate::PyError> {
    host_mp::get_semaphore_value(handle)
        .map(i64::from)
        .map_err(|()| last_windows_error())
}

/// `semaphore.c semlock_iszero`'s Windows arm.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_iszero(handle: SemRaw) -> Result<bool, crate::PyError> {
    let status = host_mp::wait_for_single_object(handle, 0);
    if status == host_mp::wait_object_0() {
        host_mp::release_semaphore(handle).map_err(|code| windows_error(code as i32))?;
        return Ok(false);
    }
    if status == host_mp::wait_timeout() {
        return Ok(true);
    }
    Err(last_windows_error())
}

/// `semaphore.c semlock_acquire`'s Windows arm, with the recursion
/// bookkeeping left to the caller as on the POSIX side.
///
/// The wait runs in slices rather than as one call: it is the runtime's own
/// wait, which a Python signal does not interrupt, and the event handshake
/// `semaphore.c` interrupts it with needs the signal module to own a Win32
/// event.  Between slices a pending signal is delivered, which is how every
/// other blocking call in the interpreter answers a Ctrl-C.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_acquire(
    handle: SemRaw,
    block: bool,
    timeout: Option<f64>,
) -> Result<bool, crate::PyError> {
    const SLICE_MS: u32 = 100;
    // `None` waits forever; `Some(0)` is the non-blocking poll.
    let mut remaining = match (block, timeout) {
        (false, _) => Some(0),
        (true, None) => None,
        (true, Some(seconds)) => {
            // `interp_semaphore.py:268-275` — a negative timeout is a poll,
            // and one at half of `INFINITE` (about 25 days) is refused rather
            // than saturated, so no wait silently becomes a different one.
            let msecs = (seconds * 1000.0).max(0.0);
            if msecs >= 0.5 * f64::from(u32::MAX) {
                return Err(crate::PyError::overflow_error("timeout is too large"));
            }
            Some((msecs + 0.5) as u32)
        }
    };
    loop {
        let slice = remaining.map_or(SLICE_MS, |left| left.min(SLICE_MS));
        let status = {
            let _blocked = crate::module::thread::before_external_block();
            host_mp::wait_for_single_object(handle, slice)
        };
        // `interp_semaphore.py:311-315` — the wait has taken the count, so it
        // is reported before anything that can raise.  A signal pending at
        // this moment is delivered at the next checkpoint like any other;
        // raising here would consume the semaphore without handing it over.
        if status == host_mp::wait_object_0() {
            return Ok(true);
        }
        if status != host_mp::wait_timeout() {
            return Err(last_windows_error());
        }
        crate::module::signal::interp_signal::checksignals_now()?;
        if let Some(left) = &mut remaining {
            *left -= slice;
            if *left == 0 {
                return Ok(false);
            }
        }
    }
}

/// `semaphore.c semlock_release`'s Windows arm.  Neither the kind nor the
/// maximum is read: `ReleaseSemaphore` enforces the maximum itself, and the
/// refusal it reports is the one the POSIX arm makes out of `sem_getvalue`.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_release(handle: SemRaw, kind: i64, maxvalue: i64) -> Result<(), crate::PyError> {
    let _ = (kind, maxvalue);
    host_mp::release_semaphore(handle).map_err(|code| {
        if code == rustpython_host_env::errno::errors::ERROR_TOO_MANY_POSTS {
            crate::PyError::value_error("semaphore or lock released too many times")
        } else {
            windows_error(code as i32)
        }
    })
}

/// The semaphore a `SemLock()` call is built on, and the name the instance
/// then reports (`None` once it has been unlinked).
#[cfg(all(unix, feature = "host_env"))]
fn semlock_create(
    name: &str,
    value: i64,
    maxvalue: i64,
    unlink: bool,
) -> Result<(SemRaw, Option<String>), crate::PyError> {
    let _ = maxvalue;
    let (handle, kept_name) = host_mp::SemHandle::create(name, value as libc::c_uint, unlink)
        .map_err(|error| {
            crate::PyError::os_error_with_errno(error.raw_os_error(), error.description())
        })?;
    let raw = handle.as_ptr();
    // SemHandle::Drop closes the semaphore. Ownership belongs to the Python
    // W_SemLock until its registered finalizer grows a typed payload.
    core::mem::forget(handle);
    Ok((raw, kept_name))
}

/// A Windows semaphore is anonymous — `CreateSemaphoreW` takes the two counts
/// and nothing else, and it is the handle that travels to another process —
/// so the name is only what the instance reports back.  A `value` above
/// `maxvalue` is the call's own `ERROR_INVALID_PARAMETER`.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_create(
    name: &str,
    value: i64,
    maxvalue: i64,
    unlink: bool,
) -> Result<(SemRaw, Option<String>), crate::PyError> {
    let (Ok(value), Ok(maxvalue)) = (i32::try_from(value), i32::try_from(maxvalue)) else {
        return Err(crate::PyError::overflow_error(
            "SemLock() value out of range",
        ));
    };
    let handle = host_mp::SemHandle::create(value, maxvalue)
        .map_err(|error| windows_error(error.raw_os_error().unwrap_or(0)))?;
    let raw = handle.as_raw();
    // As on the POSIX arm: the drop closes the handle, and the Python object
    // owns it from here.
    core::mem::forget(handle);
    Ok((raw, (!unlink).then(|| name.to_owned())))
}

/// The semaphore `_rebuild` reattaches to.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_rebuild_raw(
    w_handle: PyObjectRef,
    name: Option<&str>,
) -> Result<SemRaw, crate::PyError> {
    match name {
        // interp_semaphore.py:550-555 — with a name, reopen it and ignore
        // `w_handle`.
        Some(name) => {
            let handle = host_mp::SemHandle::open_existing(name).map_err(|error| {
                crate::PyError::os_error_with_errno(error.raw_os_error(), error.description())
            })?;
            let raw = handle.as_ptr();
            core::mem::forget(handle);
            Ok(raw)
        }
        // interp_semaphore.py:557 `handle = handle_w(space, w_handle)`
        // (`:223-224`).
        None => Ok(crate::baseobjspace::int_w(w_handle)? as usize as SemRaw),
    }
}

/// There is no name to reopen through on Windows, so the handle is what
/// `_rebuild` reattaches to whether or not a name came with it.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_rebuild_raw(
    w_handle: PyObjectRef,
    name: Option<&str>,
) -> Result<SemRaw, crate::PyError> {
    let _ = name;
    Ok(crate::baseobjspace::int_w(w_handle)? as usize as SemRaw)
}

/// `interp_semaphore.py:357-400 semlock_acquire` — the platform wait alone.
/// Upstream bumps `self.last_tid`/`self.count` here (`:395-396`); the receiver
/// stays with the caller instead, which does it on the success return.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_acquire(
    handle: SemRaw,
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
fn semlock_release(handle: SemRaw, kind: i64, maxvalue: i64) -> Result<(), crate::PyError> {
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
#[cfg(all(any(unix, windows), feature = "host_env"))]
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
#[cfg(all(any(unix, windows), feature = "host_env"))]
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

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_instance(
    w_subtype: PyObjectRef,
    raw: SemRaw,
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

/// `_multiprocessing.SemLock.__new__` declares
/// `kind: int, value: int, maxvalue: int, name: str, unlink: int`, all five
/// positional-or-keyword, so each binds by name and each converter reports
/// its own argument.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some((&w_subtype, rest)) = args.split_first() else {
        return Err(crate::PyError::type_error(
            "_multiprocessing.SemLock.__new__(): not enough arguments",
        ));
    };
    let scope = crate::builtins::bind_builtin_kwargs(
        rest,
        &["kind", "value", "maxvalue", "name", "unlink"],
        &[true; 5],
        "SemLock",
    )?;
    let kind = crate::builtins::space_index_w(scope[0])?;
    let value = crate::builtins::space_index_w(scope[1])?;
    let maxvalue = crate::builtins::space_index_w(scope[2])?;
    if !unsafe { is_str(scope[3]) } {
        // `_PyArg_BadArgument` renders the None singleton as `None` rather
        // than as its class name.
        let type_name = if unsafe { is_none(scope[3]) } {
            "None"
        } else {
            unsafe { pyre_object::type_name_of(scope[3]) }
        };
        return Err(crate::PyError::type_error(format!(
            "SemLock() argument 'name' must be str, not {type_name}"
        )));
    }
    let name = crate::baseobjspace::str_utf8_w(scope[3])?.to_string();
    // `unwrap_spec(unlink=int)` (interp_semaphore.py:572) — the flag is
    // converted the same way `kind`, `value` and `maxvalue` beside it are,
    // so a type whose `__bool__` and `__index__` disagree does not decide it.
    let unlink = crate::builtins::space_index_w(scope[4])? != 0;
    // interp_semaphore.py:574-575.
    if kind != RECURSIVE_MUTEX && kind != SEMAPHORE {
        return Err(crate::PyError::value_error("unrecognized kind"));
    }
    let (raw, kept_name) = semlock_create(&name, value, maxvalue, unlink)?;
    semlock_instance(w_subtype, raw, kind, maxvalue, kept_name)
}

/// `interp_semaphore.py:547-561 W_SemLock.rebuild`, registered as a
/// classmethod (`:606`), so `args[0]` is the bound class.
#[cfg(all(any(unix, windows), feature = "host_env"))]
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
    let raw = semlock_rebuild_raw(args[1], name.as_deref())?;
    semlock_instance(w_cls, raw, kind, maxvalue, name)
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
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
            semlock_value(handle)
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

#[cfg(all(any(unix, windows), feature = "host_env"))]
#[crate::pyre_function]
fn sem_unlink(name: &str) -> Result<(), crate::PyError> {
    #[cfg(unix)]
    {
        host_mp::sem_unlink(name).map_err(|_| crate::PyError::os_error("sem_unlink failed"))
    }
    // A Windows semaphore has no name in the filesystem sense, so there is
    // nothing to remove and `SEM_UNLINK` is the constant success the call
    // reads (`semaphore.c`).
    #[cfg(windows)]
    {
        let _ = name;
        Ok(())
    }
}

/// The three socket calls `multiprocessing/connection.py` binds as default
/// arguments on Windows, where a `Connection` is a socket rather than a
/// descriptor.  A failure is reported by `WSAGetLastError`, so it carries a
/// Win32 code rather than an errno.
#[cfg(all(windows, feature = "host_env"))]
#[crate::pyre_function]
fn closesocket(handle: i64) -> Result<(), crate::PyError> {
    host_mp::close_socket(handle as usize as host_mp::RawSocket)
        .map_err(|error| windows_error(error.raw_os_error().unwrap_or(0)))
}

#[cfg(all(windows, feature = "host_env"))]
#[crate::pyre_function]
fn recv(handle: i64, size: i64) -> Result<PyObjectRef, crate::PyError> {
    let size =
        usize::try_from(size).map_err(|_| crate::PyError::value_error("negative buffer size"))?;
    let data = host_mp::recv_socket(handle as usize as host_mp::RawSocket, size)
        .map_err(|error| windows_error(error.raw_os_error().unwrap_or(0)))?;
    Ok(pyre_object::w_bytes_from_bytes(&data))
}

#[cfg(all(windows, feature = "host_env"))]
#[crate::pyre_function]
fn send(handle: i64, buf: &[u8]) -> Result<i64, crate::PyError> {
    host_mp::send_socket(handle as usize as host_mp::RawSocket, buf)
        .map(i64::from)
        .map_err(|error| windows_error(error.raw_os_error().unwrap_or(0)))
}

crate::py_module! {
    "_multiprocessing",
    extra_init: |ns| {
        #[cfg(all(any(unix, windows), feature = "host_env"))]
        {
            let semlock_type = type_object();
            crate::module_ns_store(ns, "SemLock", semlock_type);
            // interp_semaphore.py:593-610 W_SemLock.typedef publishes this
            // constant on the class (the module also exports its own copy).
            // `SEM_VALUE_MAX` is what the platform will count to: the
            // POSIX limit, or `LONG_MAX` where `CreateSemaphoreW` takes the
            // maximum as its own argument.
            #[cfg(unix)]
            let value_max = i64::from(host_mp::sem_value_max());
            #[cfg(windows)]
            let value_max = i64::from(i32::MAX);
            let sem_value_max = w_int_new(value_max);
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
                    // interp_semaphore.py:572-573 declares the parameters
                    // through `@unwrap_spec(kind=int, value=int, maxvalue=int,
                    // name='text', unlink=int)`, so `interp2app` binds them by
                    // name as well as by position.  Registering the same names
                    // here routes a keyword call through the gateway binder,
                    // which hands the body the slots in positional order.
                    // `subtype` stays positional-only: it is the class the
                    // descriptor was reached through, not a parameter.
                    crate::make_builtin_function_with_signature(
                        "__new__",
                        semlock_descr_new,
                        crate::gateway::Signature::new(
                            vec!["subtype", "kind", "value", "maxvalue", "name", "unlink"],
                            None,
                            None,
                            0,
                            1,
                        ),
                    ),
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
        #[cfg(all(windows, feature = "host_env"))]
        {
            crate::module_ns_store(
                ns,
                "closesocket",
                crate::make_builtin_function_with_arity("closesocket", closesocket, 1),
            );
            crate::module_ns_store(
                ns,
                "recv",
                crate::make_builtin_function_with_arity("recv", recv, 2),
            );
            crate::module_ns_store(
                ns,
                "send",
                crate::make_builtin_function_with_arity("send", send, 2),
            );
            // `flags` reports the build-time semaphore capabilities the
            // POSIX build is configured with; this one has none to report.
            crate::module_ns_store(ns, "flags", pyre_object::w_dict_new());
        }
    }
}
