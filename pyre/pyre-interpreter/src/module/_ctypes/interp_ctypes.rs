//! _ctypes implementation — PyPy: lib_pypy/_ctypes/
//!
//! Verbatim move of the inline block previously in importing.rs.


// ──────────────────────────────────────────────────────────────────────
// _ctypes module — PyPy: `pypy/module/_rawffi/` plus `lib_pypy/_ctypes/`.
//
// **Slice C1: dlopen / dlsym / dlclose + size/align/memmove constants.**
//
// Provides the dynamic-linker primitives that ctypes.CDLL builds on
// top of, plus the simple-type size/align table and POSIX RTLD_* flags.
// The full c_int / Structure / CFUNCTYPE / Pointer machinery still
// requires libffi-style argument marshalling and per-instance heap
// state — those are later slices.
// ──────────────────────────────────────────────────────────────────────

pub fn register_module(ns: pyre_object::PyObjectRef) {
    #[cfg(all(unix, feature = "host_env"))]
    {
        use rustpython_host_env::ctypes as host_ctypes;

        // dlopen flags (POSIX).
        crate::module_ns_store(
            ns,
            "RTLD_LOCAL",
            pyre_object::w_int_new(libc::RTLD_LOCAL as i64),
        );
        crate::module_ns_store(
            ns,
            "RTLD_GLOBAL",
            pyre_object::w_int_new(libc::RTLD_GLOBAL as i64),
        );
        crate::module_ns_store(
            ns,
            "RTLD_LAZY",
            pyre_object::w_int_new(libc::RTLD_LAZY as i64),
        );
        crate::module_ns_store(
            ns,
            "RTLD_NOW",
            pyre_object::w_int_new(libc::RTLD_NOW as i64),
        );
        crate::module_ns_store(
            ns,
            "DEFAULT_MODE",
            pyre_object::w_int_new(host_ctypes::dlopen_mode(None) as i64),
        );

        // dlopen(name, mode=DEFAULT_MODE) → handle (opaque integer that
        // indexes into host_env's libcache).
        crate::module_ns_store(
            ns,
            "dlopen",
            crate::make_builtin_function("dlopen", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("dlopen() missing library name"));
                }
                let name = unsafe {
                    if pyre_object::is_none(args[0]) {
                        // dlopen(None) → process handle
                        let mode = if args.len() >= 2 {
                            pyre_object::w_int_get_value(args[1]) as libc::c_int
                        } else {
                            libc::RTLD_NOW
                        };
                        let ptr = rustpython_host_env::ctypes::dlopen_self(mode)
                            .map_err(|e| crate::PyError::os_error(format!("dlopen(None): {e}")))?;
                        let h = rustpython_host_env::ctypes::insert_raw_library_handle(ptr);
                        return Ok(pyre_object::w_int_new(h as i64));
                    }
                    if !pyre_object::is_str(args[0]) {
                        return Err(crate::PyError::type_error(
                            "dlopen: name must be a string or None",
                        ));
                    }
                    pyre_object::w_str_get_value(args[0]).to_string()
                };
                let mode = if args.len() >= 2 {
                    (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32
                } else {
                    rustpython_host_env::ctypes::dlopen_mode(None)
                };
                let h = rustpython_host_env::ctypes::open_library_with_mode(&name, mode)
                    .map_err(|e| crate::PyError::os_error(format!("dlopen({name}): {e}")))?;
                Ok(pyre_object::w_int_new(h as i64))
            }),
        );

        // dlsym(handle, name) → address (int).  Returns the function
        // pointer; for data symbols use dlsym(handle, name) the same way.
        crate::module_ns_store(
            ns,
            "dlsym",
            crate::make_builtin_function_with_arity(
                "dlsym",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("dlsym() needs 2 arguments"));
                    }
                    let h = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                    let name = unsafe {
                        if !pyre_object::is_str(args[1]) {
                            return Err(crate::PyError::type_error("dlsym: name must be a string"));
                        }
                        pyre_object::w_str_get_value(args[1]).to_string()
                    };
                    let addr = rustpython_host_env::ctypes::lookup_function_symbol_addr(
                        h,
                        name.as_bytes(),
                    )
                    .map_err(|e| {
                        use rustpython_host_env::ctypes::LookupSymbolError as L;
                        let msg = match e {
                            L::LibraryNotFound => "library not found".to_string(),
                            L::LibraryClosed => "library closed".to_string(),
                            L::Load(s) => s,
                        };
                        crate::PyError::os_error(format!("dlsym({name}): {msg}"))
                    })?;
                    Ok(pyre_object::w_int_new(addr as i64))
                },
                2,
            ),
        );

        // dlclose(handle) → None
        crate::module_ns_store(
            ns,
            "dlclose",
            crate::make_builtin_function_with_arity(
                "dlclose",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dlclose() needs handle"));
                    }
                    let h = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                    rustpython_host_env::ctypes::drop_library(h);
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // get_errno / set_errno — ctypes routes them through host_env so
        // a saved-errno round-trip across foreign calls survives the
        // global libc::errno being overwritten by intermediate syscalls.
        crate::module_ns_store(
            ns,
            "get_errno",
            crate::make_builtin_function_with_arity(
                "get_errno",
                |_| {
                    Ok(pyre_object::w_int_new(
                        rustpython_host_env::ctypes::get_errno() as i64,
                    ))
                },
                0,
            ),
        );
        crate::module_ns_store(
            ns,
            "set_errno",
            crate::make_builtin_function_with_arity(
                "set_errno",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("set_errno() needs value"));
                    }
                    let v = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let prev = rustpython_host_env::ctypes::set_errno(v);
                    Ok(pyre_object::w_int_new(prev as i64))
                },
                1,
            ),
        );

        // sizeof / alignment of simple ctypes type codes ('i', 'l', 'd', etc.).
        crate::module_ns_store(
            ns,
            "_sizeof_typecode",
            crate::make_builtin_function_with_arity(
                "_sizeof_typecode",
                |args| {
                    if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "_sizeof_typecode() needs typecode string",
                        ));
                    }
                    let code = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    match rustpython_host_env::ctypes::simple_type_size(&code) {
                        Some(n) => Ok(pyre_object::w_int_new(n as i64)),
                        None => Err(crate::PyError::value_error(format!(
                            "unknown type code: {code}"
                        ))),
                    }
                },
                1,
            ),
        );
        crate::module_ns_store(
            ns,
            "_alignof_typecode",
            crate::make_builtin_function_with_arity(
                "_alignof_typecode",
                |args| {
                    if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "_alignof_typecode() needs typecode string",
                        ));
                    }
                    let code = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    match rustpython_host_env::ctypes::simple_type_align(&code) {
                        Some(n) => Ok(pyre_object::w_int_new(n as i64)),
                        None => Err(crate::PyError::value_error(format!(
                            "unknown type code: {code}"
                        ))),
                    }
                },
                1,
            ),
        );

        // Address of memmove / memset for ctypes.memmove / memset.
        crate::module_ns_store(
            ns,
            "memmove",
            crate::make_builtin_function_with_arity(
                "memmove",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "memmove() needs (dst, src, count)",
                        ));
                    }
                    let dst = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize
                        as *mut libc::c_void;
                    let src = (unsafe { pyre_object::w_int_get_value(args[1]) }) as usize
                        as *const libc::c_void;
                    let n = (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize;
                    unsafe { libc::memmove(dst, src, n) };
                    Ok(pyre_object::w_int_new(dst as usize as i64))
                },
                3,
            ),
        );
        crate::module_ns_store(
            ns,
            "memset",
            crate::make_builtin_function_with_arity(
                "memset",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("memset() needs (dst, c, count)"));
                    }
                    let dst = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize
                        as *mut libc::c_void;
                    let c = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                    let n = (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize;
                    unsafe { libc::memset(dst, c, n) };
                    Ok(pyre_object::w_int_new(dst as usize as i64))
                },
                3,
            ),
        );

        // string_at(ptr, size=-1) -> bytes
        crate::module_ns_store(
            ns,
            "string_at",
            crate::make_builtin_function("string_at", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("string_at() needs ptr"));
                }
                let ptr = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                let size = if args.len() >= 2 {
                    unsafe { pyre_object::w_int_get_value(args[1]) }
                } else {
                    -1
                };
                let bytes =
                    rustpython_host_env::ctypes::string_at(ptr, size as isize).map_err(|e| {
                        use rustpython_host_env::ctypes::StringAtError as S;
                        let msg = match e {
                            S::NullPointer => "NULL pointer access",
                            S::TooLong => "size too large",
                        };
                        crate::PyError::os_error(format!("string_at: {msg}"))
                    })?;
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
            }),
        );

        // FFI library helpers used by stdlib ctypes/util.py:
        //   _ctypes.dlopen + DEFAULT_MODE typically come above, but stdlib
        //   also looks for _ctypes.SIZEOF_TIME_T to size struct timespec.
        crate::module_ns_store(
            ns,
            "SIZEOF_TIME_T",
            pyre_object::w_int_new(rustpython_host_env::ctypes::SIZEOF_TIME_T as i64),
        );
    }

    // Error type alias.
    crate::module_ns_store(ns, "ArgumentError", crate::typedef::w_object());
    crate::module_ns_store(ns, "_Pointer", crate::typedef::w_object());
    crate::module_ns_store(ns, "Structure", crate::typedef::w_object());
    crate::module_ns_store(ns, "Union", crate::typedef::w_object());
    crate::module_ns_store(ns, "Array", crate::typedef::w_object());
    crate::module_ns_store(ns, "_CFuncPtr", crate::typedef::w_object());
    crate::module_ns_store(ns, "_SimpleCData", crate::typedef::w_object());
    crate::module_ns_store(ns, "CFuncPtr", crate::typedef::w_object());
    crate::module_ns_store(ns, "POINTER", crate::typedef::w_object());
    crate::module_ns_store(ns, "pointer", crate::typedef::w_object());
    crate::module_ns_store(ns, "byref", crate::typedef::w_object());
    crate::module_ns_store(ns, "addressof", crate::typedef::w_object());
    crate::module_ns_store(ns, "sizeof", crate::typedef::w_object());
    crate::module_ns_store(ns, "alignment", crate::typedef::w_object());
    crate::module_ns_store(ns, "_check_HRESULT", crate::typedef::w_object());
}
