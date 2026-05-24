//! _thread implementation — PyPy: pypy/module/thread/interp_thread.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! init_lock_type helper is kept private; init_thread is renamed to
//! register_module.

use crate::DictStorage;
use pyre_object::PyObjectRef;

/// Lock methods — PyPy: pypy/module/thread/os_lock.py W_Lock / W_RLock
///
/// Single-threaded pyre: state lives in the instance dict as `_locked_count`.
/// Methods increment/decrement this counter so Condition/RLock ownership
/// checks see the correct state.
fn init_lock_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| {
                if let Some(&obj) = args.first() {
                    lock_acquire_impl(obj)?;
                }
                Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                lock_release_impl(obj)?;
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    );
    // descr_lock_acquire — PyPy: os_lock.Lock.descr_lock_acquire
    crate::dict_storage_store(
        ns,
        "acquire",
        crate::make_builtin_function("acquire", |args| {
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            lock_acquire_impl(obj)?;
            Ok(pyre_object::w_bool_from(true))
        }),
    );
    // descr_lock_release — PyPy: os_lock.Lock.descr_lock_release
    crate::dict_storage_store(
        ns,
        "release",
        crate::make_builtin_function_with_arity(
            "release",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                lock_release_impl(obj)?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    // descr_lock_locked — PyPy: os_lock.Lock.descr_lock_locked
    crate::dict_storage_store(
        ns,
        "locked",
        crate::make_builtin_function_with_arity(
            "locked",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_bool_from(lock_count(obj) > 0))
            },
            1,
        ),
    );
    // _is_owned — used by RLock/Condition in threading.py
    crate::dict_storage_store(
        ns,
        "_is_owned",
        crate::make_builtin_function_with_arity(
            "_is_owned",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_bool_from(lock_count(obj) > 0))
            },
            1,
        ),
    );
    // _at_fork_reinit — PyPy: os_lock.Lock._at_fork_reinit (reset to unlocked)
    crate::dict_storage_store(
        ns,
        "_at_fork_reinit",
        crate::make_builtin_function_with_arity(
            "_at_fork_reinit",
            |args| {
                if let Some(&obj) = args.first() {
                    lock_set_count(obj, 0);
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
}

/// Read the lock's internal count. Single-threaded: 0 = unlocked, >0 = locked.
fn lock_count(obj: pyre_object::PyObjectRef) -> i64 {
    let w_dict = crate::baseobjspace::getdict(obj);
    if w_dict.is_null() {
        return 0;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(w_dict, "_locked_count") } {
        unsafe {
            if pyre_object::is_int(v) {
                return pyre_object::w_int_get_value(v);
            }
        }
    }
    0
}

fn lock_set_count(obj: pyre_object::PyObjectRef, v: i64) {
    let w_dict = crate::baseobjspace::getdict(obj);
    if w_dict.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(w_dict, "_locked_count", pyre_object::w_int_new(v));
    }
}

fn lock_acquire_impl(obj: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    lock_set_count(obj, lock_count(obj) + 1);
    Ok(())
}

fn lock_release_impl(obj: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let cur = lock_count(obj);
    if cur <= 0 {
        return Err(crate::PyError::runtime_error("release unlocked lock"));
    }
    lock_set_count(obj, cur - 1);
    Ok(())
}

thread_local! {
    static LOCK_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    static THREAD_HANDLE_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
}

fn lock_type() -> PyObjectRef {
    LOCK_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("lock", init_lock_type);
            // Store per-instance `_locked_count` in the instance dict.
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

fn thread_handle_type() -> PyObjectRef {
    THREAD_HANDLE_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            crate::typedef::make_builtin_type("_ThreadHandle", |ns| {
                crate::dict_storage_store(
                    ns,
                    "is_done",
                    crate::make_builtin_function_with_arity(
                        "is_done",
                        |_| Ok(pyre_object::w_bool_from(true)),
                        1,
                    ),
                );
                crate::dict_storage_store(
                    ns,
                    "join",
                    crate::make_builtin_function("join", |_| Ok(pyre_object::w_none())),
                );
                crate::dict_storage_store(
                    ns,
                    "set_result",
                    crate::make_builtin_function_with_arity(
                        "set_result",
                        |_| Ok(pyre_object::w_none()),
                        2,
                    ),
                );
                crate::dict_storage_store(
                    ns,
                    "_set_done",
                    crate::make_builtin_function_with_arity(
                        "_set_done",
                        |_| Ok(pyre_object::w_none()),
                        1,
                    ),
                );
            })
        })
    })
}

/// _thread stub
pub fn register_module(ns: &mut DictStorage) {
    let lock_tp = lock_type();
    crate::dict_storage_store(ns, "LockType", lock_tp);
    crate::dict_storage_store(
        ns,
        "RLock",
        crate::make_builtin_function_with_arity(
            "RLock",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "allocate_lock",
        crate::make_builtin_function_with_arity(
            "allocate_lock",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_ident",
        crate::make_builtin_function_with_arity(
            "get_ident",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    return Ok(pyre_object::w_int_new(
                        rustpython_host_env::thread::current_thread_id() as i64,
                    ));
                }
                #[cfg(not(feature = "host_env"))]
                Ok(pyre_object::w_int_new(1))
            },
            0,
        ),
    );
    // _thread.get_native_id() — returns the kernel-level TID, NOT the
    // pthread handle.  Mirrors rthread.c_get_native_id (rpython/rlib/
    // rthread.py) used by pypy/module/thread/os_thread.py:204-210.
    //
    // host_env::thread::current_thread_id always returns pthread_self
    // (suitable for get_ident above), so we drop to libc here:
    //   * Linux/Android: syscall(SYS_gettid) — kernel TID, distinct
    //     from pthread_self.
    //   * macOS:         pthread_threadid_np(NULL, &tid) — 64-bit TID.
    //   * Other Unix:    fall back to pthread_self (best effort; the
    //     same as get_ident, matching the lack of a true TID concept).
    crate::dict_storage_store(
        ns,
        "get_native_id",
        crate::make_builtin_function_with_arity(
            "get_native_id",
            |_| {
                #[cfg(any(target_os = "linux", target_os = "android"))]
                {
                    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
                    return Ok(pyre_object::w_int_new(tid as i64));
                }
                #[cfg(target_os = "macos")]
                {
                    let mut tid: u64 = 0;
                    let rc = unsafe { libc::pthread_threadid_np(0, &mut tid as *mut u64) };
                    if rc == 0 {
                        return Ok(pyre_object::w_int_new(tid as i64));
                    }
                    return Ok(pyre_object::w_int_new(
                        unsafe { libc::pthread_self() } as i64
                    ));
                }
                #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos",)))]
                {
                    #[cfg(unix)]
                    {
                        return Ok(pyre_object::w_int_new(
                            unsafe { libc::pthread_self() } as i64
                        ));
                    }
                    #[cfg(not(unix))]
                    Ok(pyre_object::w_int_new(1))
                }
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_count",
        crate::make_builtin_function_with_arity("_count", |_| Ok(pyre_object::w_int_new(1)), 0),
    );
    crate::dict_storage_store(ns, "TIMEOUT_MAX", pyre_object::w_float_new(f64::MAX));
    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
    crate::dict_storage_store(
        ns,
        "start_joinable_thread",
        crate::make_builtin_function("start_joinable_thread", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::dict_storage_store(
        ns,
        "_set_sentinel",
        crate::make_builtin_function_with_arity(
            "_set_sentinel",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "stack_size",
        crate::make_builtin_function_with_arity("stack_size", |_| Ok(pyre_object::w_int_new(0)), 1),
    );
    crate::dict_storage_store(
        ns,
        "_is_main_interpreter",
        crate::make_builtin_function_with_arity(
            "_is_main_interpreter",
            |_| Ok(pyre_object::w_bool_from(true)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "daemon_threads_allowed",
        crate::make_builtin_function_with_arity(
            "daemon_threads_allowed",
            |_| Ok(pyre_object::w_bool_from(true)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_shutdown",
        crate::make_builtin_function_with_arity("_shutdown", |_| Ok(pyre_object::w_none()), 0),
    );
    // _make_thread_handle / _ThreadHandle — threading.py:40-41
    crate::dict_storage_store(ns, "_ThreadHandle", thread_handle_type());
    crate::dict_storage_store(
        ns,
        "_make_thread_handle",
        crate::make_builtin_function_with_arity(
            "_make_thread_handle",
            |_| Ok(pyre_object::w_instance_new(thread_handle_type())),
            1,
        ),
    );
    // _get_main_thread_ident — threading.py:43
    crate::dict_storage_store(
        ns,
        "_get_main_thread_ident",
        crate::make_builtin_function_with_arity(
            "_get_main_thread_ident",
            |_| Ok(pyre_object::w_int_new(1)),
            0,
        ),
    );
    // get_native_id — threading.py:46
    crate::dict_storage_store(
        ns,
        "get_native_id",
        crate::make_builtin_function_with_arity(
            "get_native_id",
            |_| Ok(pyre_object::w_int_new(1)),
            0,
        ),
    );
    // set_name — threading.py:52
    crate::dict_storage_store(
        ns,
        "set_name",
        crate::make_builtin_function_with_arity("set_name", |_| Ok(pyre_object::w_none()), 1),
    );
    // _excepthook — threading.py:1262
    crate::dict_storage_store(
        ns,
        "_excepthook",
        crate::make_builtin_function_with_arity("_excepthook", |_| Ok(pyre_object::w_none()), 1),
    );
    // _local — PyPy: pypy/module/thread/os_local.py Local
    // Thread-local data. Single-threaded: equivalent to a plain object with dict.
    crate::dict_storage_store(ns, "_local", local_type());
}

fn local_type() -> PyObjectRef {
    thread_local! {
        static LOCAL_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    LOCAL_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_local", |_ns| {});
            // Instances need __dict__ for per-thread attribute storage.
            // PyPy: os_local.py Local has getdict(space) → w_dict
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}
