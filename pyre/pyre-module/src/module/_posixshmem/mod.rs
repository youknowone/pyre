//! _posixshmem module — PyPy: `lib_pypy/_posixshmem.py`.
//!
//! Backs `multiprocessing.shared_memory` on POSIX.  Entire surface is
//! gated on `cfg(all(unix, feature = "host_env"))`; non-Unix /
//! `host_env = off` builds expose an empty module so `import
//! _posixshmem` still succeeds (matching PyPy's mixedmodule behaviour
//! when the conditional `interpleveldefs` entry is absent).

use pyre_object::*;

#[cfg(all(unix, feature = "host_env"))]
fn shm_open(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if !(2..=3).contains(&args.len()) {
        return Err(pyre_interpreter::PyError::type_error(
            "shm_open() requires (path, flags[, mode])",
        ));
    }
    let name = unsafe {
        if !is_str(args[0]) {
            return Err(pyre_interpreter::PyError::type_error(
                "shm_open: path must be a string",
            ));
        }
        pyre_interpreter::baseobjspace::str_utf8_w(args[0])?.to_string()
    };
    let flags = (unsafe { w_int_get_value(args[1]) }) as libc::c_int;
    let mode = if args.len() >= 3 {
        (unsafe { w_int_get_value(args[2]) }) as libc::c_uint
    } else {
        0o600
    };
    let c_name = std::ffi::CString::new(name.as_bytes())
        .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))?;
    // `lib_pypy/_posixshmem.py:13-20` retries on EINTR.
    let fd = loop {
        match rustpython_host_env::shm::shm_open(&c_name, flags, mode) {
            Ok(fd) => break fd,
            Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
            Err(e) => {
                return Err(pyre_interpreter::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(0),
                    format!("shm_open: {e}"),
                ));
            }
        }
    };
    Ok(w_int_new(fd as i64))
}

#[cfg(all(unix, feature = "host_env"))]
fn shm_unlink(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "shm_unlink() needs path",
        ));
    }
    let name = unsafe {
        if !is_str(args[0]) {
            return Err(pyre_interpreter::PyError::type_error(
                "shm_unlink: path must be a string",
            ));
        }
        pyre_interpreter::baseobjspace::str_utf8_w(args[0])?.to_string()
    };
    let c_name = std::ffi::CString::new(name.as_bytes())
        .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))?;
    // `lib_pypy/_posixshmem.py:33-40` retries on EINTR.
    loop {
        match rustpython_host_env::shm::shm_unlink(&c_name) {
            Ok(()) => break,
            Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
            Err(e) => {
                return Err(pyre_interpreter::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(0),
                    format!("shm_unlink: {e}"),
                ));
            }
        }
    }
    Ok(w_none())
}

pyre_interpreter::py_module! {
    "_posixshmem",
    extra_init: |ns| {
        #[cfg(all(unix, feature = "host_env"))]
        {
            pyre_interpreter::module_ns_store(ns, "shm_open",
                pyre_interpreter::make_builtin_function("shm_open", shm_open));
            pyre_interpreter::module_ns_store(ns, "shm_unlink",
                pyre_interpreter::make_builtin_function_with_arity("shm_unlink", shm_unlink, 1));
        }
        #[cfg(not(all(unix, feature = "host_env")))]
        let _ = ns;
    }
}
