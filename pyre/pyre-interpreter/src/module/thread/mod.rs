//! _thread module — PyPy: `pypy/module/thread/`.
//!
//! Single-threaded pyre: `Lock` / `RLock` state lives in the instance
//! dict as `_locked_count`.  `allocate_lock` / `start_new_thread` etc.
//! are stubs; `_ThreadHandle` lives long enough for `threading.py` to
//! call `is_done()` during shutdown.

use pyre_object::*;
use std::cell::Cell;
use std::sync::OnceLock;

thread_local! {
    // The runtime is single-OS-threaded, but a synchronous emulation of a
    // joinable Python thread still needs a distinct logical ident so
    // threading._active never replaces the main-thread entry.
    static LOGICAL_THREAD_IDENT: Cell<i64> = const { Cell::new(0) };
}

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
            fn _release_save(self_obj: PyObjectRef) -> i64 {
                let count = lock_count(self_obj);
                lock_set_count(self_obj, 0);
                count
            }
            fn _acquire_restore(self_obj: PyObjectRef, count: i64) {
                lock_set_count(self_obj, count.max(1));
            }
            fn _recursion_count(self_obj: PyObjectRef) -> i64 {
                lock_count(self_obj)
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
    // PyPy's Local.typedef is shared; only each Local instance's dictionaries
    // are execution-context-specific.
    static TYPE: OnceLock<usize> = OnceLock::new();
    *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_local", |_| {});
        unsafe { typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
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

fn start_joinable_thread(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _kwargs) = crate::builtins::split_builtin_kwargs(args);
    let function = pos
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("missing function argument"))?;
    let thread_obj = if unsafe { pyre_object::is_method(function) } {
        unsafe { pyre_object::w_method_get_self(function) }
    } else {
        pyre_object::PY_NULL
    };
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(function);
    if !thread_obj.is_null() {
        pyre_object::gc_roots::pin_root(thread_obj);
    }
    let thread_slot =
        (!thread_obj.is_null()).then(|| pyre_object::gc_roots::shadow_stack_len() - 1);
    // A real OS thread starts with a distinct ctypes TLS errno slot.  The
    // synchronous thread emulator must therefore bracket the call rather than
    // letting the worker overwrite its caller's slot.
    let caller_errno = rustpython_host_env::ctypes::get_errno();
    rustpython_host_env::ctypes::set_errno(0);
    LOGICAL_THREAD_IDENT.with(|ident| {
        let previous = ident.replace(2);
        let _ = crate::call::call_function_impl_result(function, &[]);
        ident.set(previous);
    });
    rustpython_host_env::ctypes::set_errno(caller_errno);
    // The emulated thread has already completed before this function returns.
    // Remove its debugging-only weak entry now, matching the state a real
    // thread reaches once its Thread object becomes unreachable.  This also
    // avoids leaving a dead weak target for the next nursery collection.
    if let (Some(slot), Some(threading)) =
        (thread_slot, crate::importing::get_sys_module("threading"))
    {
        let thread_obj = pyre_object::gc_roots::shadow_stack_get(slot);
        if let Ok(dangling) = crate::baseobjspace::getattr_str(threading, "_dangling") {
            if let Ok(discard) = crate::baseobjspace::getattr_str(dangling, "discard") {
                let _ = crate::call::call_function_impl_result(discard, &[thread_obj]);
            }
        }
    }
    Ok(w_int_new(1))
}

// PyPy `_thread.get_ident` returns the pthread handle; pyre routes
// through `rustpython_host_env::thread::current_thread_id`.  Without
// host_env we always return 1 (single-threaded sentinel).
fn current_ident() -> i64 {
    let logical = LOGICAL_THREAD_IDENT.with(Cell::get);
    if logical != 0 {
        return logical;
    }
    // The sandboxed child is a single logical thread; do not leak the real
    // thread id (host state), return the single-threaded sentinel instead.
    #[cfg(all(
        feature = "host_env",
        not(target_arch = "wasm32"),
        not(feature = "sandbox")
    ))]
    {
        return rustpython_host_env::thread::current_thread_id() as i64;
    }
    #[allow(unreachable_code)]
    {
        1
    }
}

#[crate::pyre_function]
fn get_ident() -> i64 {
    current_ident()
}

// `_thread.get_native_id()` — kernel-level TID, NOT the pthread
// handle.  Mirrors `rthread.c_get_native_id` (pypy/module/thread/
// os_thread.py:204-210):
//   * Linux/Android: syscall(SYS_gettid)
//   * macOS:         pthread_threadid_np(NULL, &tid)
//   * Other Unix:    pthread_self  (no true TID concept)
#[crate::pyre_function]
fn get_native_id() -> i64 {
    #[cfg(all(
        not(feature = "sandbox"),
        any(target_os = "linux", target_os = "android")
    ))]
    {
        return unsafe { libc::syscall(libc::SYS_gettid) } as i64;
    }
    #[cfg(all(not(feature = "sandbox"), target_os = "macos"))]
    {
        let mut tid: u64 = 0;
        let rc = unsafe { libc::pthread_threadid_np(0, &mut tid as *mut u64) };
        if rc == 0 {
            return tid as i64;
        }
        return unsafe { libc::pthread_self() } as i64;
    }
    #[cfg(all(
        not(feature = "sandbox"),
        unix,
        not(any(target_os = "linux", target_os = "android", target_os = "macos"))
    ))]
    {
        return unsafe { libc::pthread_self() } as i64;
    }
    // Sandbox and non-unix: a fixed single-thread sentinel — the sandboxed
    // child must not issue the raw `gettid`/`pthread_self` that would expose the
    // real kernel/pthread id.
    #[allow(unreachable_code)]
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
        "_get_main_thread_ident" / 0 = |_| Ok(w_int_new(current_ident())),
        "start_joinable_thread"  / * = start_joinable_thread,
        "start_new_thread"       / * = start_new_thread,
        "start_new"              / * = start_new_thread,
    },
}
