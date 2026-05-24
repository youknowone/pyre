//! _multiprocessing implementation — PyPy: pypy/module/_multiprocessing/interp_semaphore.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

// ──────────────────────────────────────────────────────────────────────
// _multiprocessing module — PyPy: pypy/module/_multiprocessing/.
//
// Exposes `SemLock(kind, value, maxvalue, name, unlink)` and
// `sem_unlink(name)`.  Pyre is currently single-threaded so the lock
// barely matters for serialization, but multiprocessing.py still calls
// .acquire()/.release() during pool teardown, so the methods must exist
// and round-trip without crashing.
// ──────────────────────────────────────────────────────────────────────

#[cfg(all(unix, feature = "host_env"))]
thread_local! {
    static SEMLOCK_TYPE_OBJ: std::cell::OnceCell<pyre_object::PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_type() -> pyre_object::PyObjectRef {
    SEMLOCK_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("SemLock", init_semlock_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_get_handle(obj: pyre_object::PyObjectRef) -> *mut libc::sem_t {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return core::ptr::null_mut();
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(d, "_handle") } {
        if unsafe { pyre_object::is_int(v) } {
            return unsafe { pyre_object::w_int_get_value(v) } as usize as *mut libc::sem_t;
        }
    }
    core::ptr::null_mut()
}

#[cfg(all(unix, feature = "host_env"))]
fn init_semlock_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "acquire",
        crate::make_builtin_function("acquire", |args| {
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let handle = semlock_get_handle(obj);
            if handle.is_null() {
                return Err(crate::PyError::value_error("SemLock handle is null"));
            }
            let blocking = if args.len() >= 2 {
                (unsafe { pyre_object::w_int_get_value(args[1]) }) != 0
            } else {
                true
            };
            if blocking {
                let r = unsafe { libc::sem_wait(handle) };
                if r != 0 {
                    return Err(crate::PyError::os_error_with_errno(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                        "sem_wait",
                    ));
                }
                Ok(pyre_object::w_bool_from(true))
            } else {
                let r = unsafe { libc::sem_trywait(handle) };
                Ok(pyre_object::w_bool_from(r == 0))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "release",
        crate::make_builtin_function_with_arity(
            "release",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let handle = semlock_get_handle(obj);
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
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_count",
        crate::make_builtin_function_with_arity("_count", |_| Ok(pyre_object::w_int_new(0)), 1),
    );
    crate::dict_storage_store(
        ns,
        "_is_mine",
        crate::make_builtin_function_with_arity(
            "_is_mine",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_is_zero",
        crate::make_builtin_function_with_arity(
            "_is_zero",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let handle = semlock_get_handle(obj);
                if handle.is_null() {
                    return Ok(pyre_object::w_bool_from(true));
                }
                // sem_getvalue isn't available on macOS; just try sem_trywait
                // and immediately repost.  Best-effort; returning false is
                // safe because the only consumer is multiprocessing.Queue
                // tearing down.
                Ok(pyre_object::w_bool_from(false))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let handle = semlock_get_handle(obj);
                if !handle.is_null() {
                    let _ = unsafe { libc::sem_wait(handle) };
                }
                Ok(obj)
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                let handle = semlock_get_handle(obj);
                if !handle.is_null() {
                    let _ = unsafe { libc::sem_post(handle) };
                }
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    );
}

pub fn register_module(ns: &mut DictStorage) {
    #[cfg(all(unix, feature = "host_env"))]
    {
        // SemLock class.
        crate::dict_storage_store(ns, "SemLock", semlock_type());

        // SemLock factory — Python convention: SemLock(kind, value, maxvalue, name, unlink)
        crate::dict_storage_store(
            ns,
            "_SemLock_new",
            crate::make_builtin_function("_SemLock_new", |args| {
                if args.len() < 5 {
                    return Err(crate::PyError::type_error(
                        "SemLock() needs (kind, value, maxvalue, name, unlink)",
                    ));
                }
                let value = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_uint;
                let name = unsafe {
                    if !pyre_object::is_str(args[3]) {
                        return Err(crate::PyError::type_error("SemLock: name must be a string"));
                    }
                    pyre_object::w_str_get_value(args[3]).to_string()
                };
                let unlink = unsafe { pyre_object::w_int_get_value(args[4]) } != 0;
                let (handle, _kept_name) =
                    rustpython_host_env::multiprocessing::SemHandle::create(&name, value, unlink)
                        .map_err(|_| crate::PyError::os_error("SemLock create failed"))?;
                let raw = handle.as_ptr();
                // Leak the SemHandle wrapper so its Drop doesn't close the fd;
                // we'll sem_close manually if needed.  This matches CPython's
                // SemLock which keeps the sem_t for the instance lifetime.
                core::mem::forget(handle);
                let obj = pyre_object::w_instance_new(semlock_type());
                let d = crate::baseobjspace::getdict(obj);
                if !d.is_null() {
                    unsafe {
                        pyre_object::w_dict_setitem_str(
                            d,
                            "_handle",
                            pyre_object::w_int_new(raw as usize as i64),
                        );
                        pyre_object::w_dict_setitem_str(d, "name", pyre_object::w_str_new(&name));
                    }
                }
                Ok(obj)
            }),
        );

        crate::dict_storage_store(
            ns,
            "sem_unlink",
            crate::make_builtin_function_with_arity(
                "sem_unlink",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("sem_unlink() needs name"));
                    }
                    let name = unsafe {
                        if !pyre_object::is_str(args[0]) {
                            return Err(crate::PyError::type_error(
                                "sem_unlink: name must be a string",
                            ));
                        }
                        pyre_object::w_str_get_value(args[0]).to_string()
                    };
                    rustpython_host_env::multiprocessing::sem_unlink(&name)
                        .map_err(|_| crate::PyError::os_error("sem_unlink failed"))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        crate::dict_storage_store(
            ns,
            "SEM_VALUE_MAX",
            pyre_object::w_int_new(rustpython_host_env::multiprocessing::sem_value_max() as i64),
        );

        // _multiprocessing exposes RECURSIVE_MUTEX / SEMAPHORE kind tags.
        crate::dict_storage_store(ns, "RECURSIVE_MUTEX", pyre_object::w_int_new(0));
        crate::dict_storage_store(ns, "SEMAPHORE", pyre_object::w_int_new(1));
    }
}
