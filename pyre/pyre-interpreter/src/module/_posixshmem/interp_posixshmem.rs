//! _posixshmem implementation — PyPy: pypy/module/_posixshmem/interp_posixshmem.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// _posixshmem module — PyPy: pypy/module/_posixshmem/.
/// Backs `multiprocessing.shared_memory` on POSIX.
pub fn register_module(ns: &mut DictStorage) {
    #[cfg(all(unix, feature = "host_env"))]
    {
        crate::dict_storage_store(
            ns,
            "shm_open",
            crate::make_builtin_function("shm_open", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "shm_open() requires (path, flags[, mode])",
                    ));
                }
                let name = unsafe {
                    if !pyre_object::is_str(args[0]) {
                        return Err(crate::PyError::type_error(
                            "shm_open: path must be a string",
                        ));
                    }
                    pyre_object::w_str_get_value(args[0]).to_string()
                };
                let flags = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                let mode = if args.len() >= 3 {
                    (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_uint
                } else {
                    0o600
                };
                let c_name = std::ffi::CString::new(name.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let fd = rustpython_host_env::shm::shm_open(&c_name, flags, mode).map_err(|e| {
                    crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("shm_open: {e}"),
                    )
                })?;
                Ok(pyre_object::w_int_new(fd as i64))
            }),
        );
        crate::dict_storage_store(
            ns,
            "shm_unlink",
            crate::make_builtin_function_with_arity(
                "shm_unlink",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("shm_unlink() needs path"));
                    }
                    let name = unsafe {
                        if !pyre_object::is_str(args[0]) {
                            return Err(crate::PyError::type_error(
                                "shm_unlink: path must be a string",
                            ));
                        }
                        pyre_object::w_str_get_value(args[0]).to_string()
                    };
                    let c_name = std::ffi::CString::new(name.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    rustpython_host_env::shm::shm_unlink(&c_name).map_err(|e| {
                        crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("shm_unlink: {e}"),
                        )
                    })?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );
    }
}
