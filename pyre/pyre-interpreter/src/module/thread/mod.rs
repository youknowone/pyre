//! _thread module — PyPy: `pypy/module/thread/`.
//!
//! Single-threaded pyre: `Lock` / `RLock` state lives in the instance
//! dict as `_locked_count`.  `allocate_lock` / `start_new_thread` etc.
//! are stubs; `_ThreadHandle` lives long enough for `threading.py` to
//! call `is_done()` during shutdown.

use pyre_object::*;

fn lock_count(obj: PyObjectRef) -> i64 {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return 0;
    }
    if let Some(v) = unsafe { w_dict_getitem_str(d, "_locked_count") } {
        if unsafe { is_int(v) } {
            return unsafe { w_int_get_value(v) };
        }
    }
    0
}

fn lock_set_count(obj: PyObjectRef, v: i64) {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return;
    }
    unsafe {
        w_dict_setitem_str(d, "_locked_count", w_int_new(v));
    }
}

/// `pypy/module/thread/os_lock.py Lock / W_RLock` — single-threaded
/// pyre treats both the same: `_locked_count` is bumped on acquire,
/// decremented on release.  RLock ownership semantics
/// (Condition._is_owned) work because every acquire from the only
/// thread succeeds.
mod lock_class {
    use super::*;

    crate::py_class! {
        "lock",
        methods: {
            fn __enter__(self_obj: PyObjectRef) -> PyObjectRef {
                lock_set_count(self_obj, lock_count(self_obj) + 1);
                self_obj
            }
            fn __exit__(self_obj: PyObjectRef) -> Result<bool, crate::PyError> {
                let cur = lock_count(self_obj);
                if cur <= 0 {
                    return Err(crate::PyError::runtime_error("release unlocked lock"));
                }
                lock_set_count(self_obj, cur - 1);
                Ok(false)
            }
            fn acquire(self_obj: PyObjectRef) -> bool {
                lock_set_count(self_obj, lock_count(self_obj) + 1);
                true
            }
            fn release(self_obj: PyObjectRef) -> Result<(), crate::PyError> {
                let cur = lock_count(self_obj);
                if cur <= 0 {
                    return Err(crate::PyError::runtime_error("release unlocked lock"));
                }
                lock_set_count(self_obj, cur - 1);
                Ok(())
            }
            fn locked(self_obj: PyObjectRef) -> bool {
                lock_count(self_obj) > 0
            }
            fn _is_owned(self_obj: PyObjectRef) -> bool {
                lock_count(self_obj) > 0
            }
            fn _at_fork_reinit(self_obj: PyObjectRef) {
                lock_set_count(self_obj, 0);
            }
        }
    }
}

/// `lib-python/3/threading.py` `_ThreadHandle` support — stubs that keep
/// `_make_thread_handle` callable through module shutdown.
mod thread_handle_class {
    use super::*;

    crate::py_class! {
        "_ThreadHandle",
        methods: {
            fn is_done(self_obj: PyObjectRef) -> bool {
                let _ = self_obj;
                true
            }
            fn join(self_obj: PyObjectRef) {
                let _ = self_obj;
            }
            fn set_result(self_obj: PyObjectRef, result: PyObjectRef) {
                let _ = (self_obj, result);
            }
            fn _set_done(self_obj: PyObjectRef) {
                let _ = self_obj;
            }
        }
    }
}

/// `pypy/module/thread/os_local.py Local` — instances need
/// `__dict__` for per-thread attribute storage; pyre is single-threaded
/// so there's no real per-thread isolation.
fn local_type() -> PyObjectRef {
    thread_local! {
        static CELL: std::cell::OnceCell<PyObjectRef> =
            const { std::cell::OnceCell::new() };
    }
    CELL.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_local", |_| {});
            unsafe { typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

// `_thread.start_new_thread(function, args[, kwargs])` — pyre is
// single-threaded, so the callable runs synchronously and the returned
// ident is the sole thread's sentinel (1).  A raising target is swallowed
// (real threads report it via `_excepthook`, never to the spawner).
fn start_new_thread(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "start_new_thread expected at least 2 arguments",
        ));
    }
    let function = args[0];
    let call_args = unsafe {
        if is_tuple(args[1]) {
            w_tuple_items_copy_as_vec(args[1])
        } else {
            return Err(crate::PyError::type_error("2nd arg must be a tuple"));
        }
    };
    let _ = crate::call::call_function_impl_result(function, &call_args);
    Ok(w_int_new(1))
}

// PyPy `_thread.get_ident` returns the pthread handle; pyre routes
// through `rustpython_host_env::thread::current_thread_id`.  Without
// host_env we always return 1 (single-threaded sentinel).
#[crate::pyre_function]
fn get_ident() -> i64 {
    #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
    {
        return rustpython_host_env::thread::current_thread_id() as i64;
    }
    #[allow(unreachable_code)]
    {
        1
    }
}

// `_thread.get_native_id()` — kernel-level TID, NOT the pthread
// handle.  Mirrors `rthread.c_get_native_id` (pypy/module/thread/
// os_thread.py:204-210):
//   * Linux/Android: syscall(SYS_gettid)
//   * macOS:         pthread_threadid_np(NULL, &tid)
//   * Other Unix:    pthread_self  (no true TID concept)
#[crate::pyre_function]
fn get_native_id() -> i64 {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        return unsafe { libc::syscall(libc::SYS_gettid) } as i64;
    }
    #[cfg(target_os = "macos")]
    {
        let mut tid: u64 = 0;
        let rc = unsafe { libc::pthread_threadid_np(0, &mut tid as *mut u64) };
        if rc == 0 {
            return tid as i64;
        }
        return unsafe { libc::pthread_self() } as i64;
    }
    #[cfg(all(
        unix,
        not(any(target_os = "linux", target_os = "android", target_os = "macos"))
    ))]
    {
        return unsafe { libc::pthread_self() } as i64;
    }
    #[cfg(not(unix))]
    {
        1
    }
}

crate::py_module! {
    "_thread",
    interpleveldefs: {
        "LockType"      => lock_class::type_object(),
        "_ThreadHandle" => thread_handle_class::type_object(),
        "_local"        => local_type(),
        "TIMEOUT_MAX"   => w_float_new(f64::MAX),
        "error"         => crate::typedef::w_object(),
    },
    functions: {
        "RLock"                  / 0 = |_| Ok(w_instance_new(lock_class::type_object())),
        "allocate_lock"          / 0 = |_| Ok(w_instance_new(lock_class::type_object())),
        "_set_sentinel"          / 0 = |_| Ok(w_instance_new(lock_class::type_object())),
        "_make_thread_handle"    / 1 = |_| Ok(w_instance_new(thread_handle_class::type_object())),
        "get_ident"              / 0 = get_ident,
        "get_native_id"          / 0 = get_native_id,
        "_count"                 / 0 = |_| Ok(w_int_new(1)),
        "_is_main_interpreter"   / 0 = |_| Ok(w_bool_from(true)),
        "daemon_threads_allowed" / 0 = |_| Ok(w_bool_from(true)),
        "_shutdown"              / 0 = |_| Ok(w_none()),
        "stack_size"             / 1 = |_| Ok(w_int_new(0)),
        "set_name"               / 1 = |_| Ok(w_none()),
        "_excepthook"            / 1 = |_| Ok(w_none()),
        "_get_main_thread_ident" / 0 = |_| Ok(w_int_new(1)),
        "start_joinable_thread"  / * = |_| Ok(w_int_new(0)),
        "start_new_thread"       / * = start_new_thread,
        "start_new"              / * = start_new_thread,
    },
}
