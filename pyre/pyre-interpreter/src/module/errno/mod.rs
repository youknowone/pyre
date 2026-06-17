//! errno module — PyPy: `pypy/module/errno/`.
//!
//! Numerics differ per OS (e.g. `EAGAIN` is 11 on Linux but 35 on
//! macOS), so when `host_env` is enabled every constant resolves
//! through `rustpython_host_env::errno::errors` (a `pub use libc::*`
//! re-export).  The `host_env = off` build keeps a darwin/BSD-flavoured
//! fallback so pyre-wasm preserves its previous behaviour.

crate::py_module! {
    "errno",
    extra_init: |ns| {
        // `interp_errno.py` builds `errorcode = {code: name, ...}`
        // alongside each exported constant.  We populate it incrementally
        // as we register the constants below.
        let errorcode = pyre_object::w_dict_new();
        crate::dict_storage_store(ns, "errorcode", errorcode);
        let mut store = |name: &str, value: i64| {
            crate::dict_storage_store(ns, name, pyre_object::w_int_new(value));
            unsafe {
                pyre_object::w_dict_store(
                    errorcode,
                    pyre_object::w_int_new(value),
                    pyre_object::w_str_new(name),
                );
            }
        };
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        {
            use rustpython_host_env::errno::errors as host_errno;
            let entries: &[(&str, i32)] = &[
                ("EPERM", host_errno::EPERM),
                ("ENOENT", host_errno::ENOENT),
                ("ESRCH", host_errno::ESRCH),
                ("EINTR", host_errno::EINTR),
                ("EIO", host_errno::EIO),
                ("ENXIO", host_errno::ENXIO),
                ("E2BIG", host_errno::E2BIG),
                ("ENOEXEC", host_errno::ENOEXEC),
                ("EBADF", host_errno::EBADF),
                ("ECHILD", host_errno::ECHILD),
                ("EAGAIN", host_errno::EAGAIN),
                ("EWOULDBLOCK", host_errno::EWOULDBLOCK),
                ("ENOMEM", host_errno::ENOMEM),
                ("EACCES", host_errno::EACCES),
                ("EFAULT", host_errno::EFAULT),
                ("EBUSY", host_errno::EBUSY),
                ("EEXIST", host_errno::EEXIST),
                ("EXDEV", host_errno::EXDEV),
                ("ENODEV", host_errno::ENODEV),
                ("ENOTDIR", host_errno::ENOTDIR),
                ("EISDIR", host_errno::EISDIR),
                ("EINVAL", host_errno::EINVAL),
                ("ENFILE", host_errno::ENFILE),
                ("EMFILE", host_errno::EMFILE),
                ("ENOTTY", host_errno::ENOTTY),
                ("EFBIG", host_errno::EFBIG),
                ("ENOSPC", host_errno::ENOSPC),
                ("ESPIPE", host_errno::ESPIPE),
                ("EROFS", host_errno::EROFS),
                ("EMLINK", host_errno::EMLINK),
                ("EPIPE", host_errno::EPIPE),
                ("EDOM", host_errno::EDOM),
                ("ERANGE", host_errno::ERANGE),
                ("EDEADLK", host_errno::EDEADLK),
                ("ENAMETOOLONG", host_errno::ENAMETOOLONG),
                ("ENOLCK", host_errno::ENOLCK),
                ("ENOSYS", host_errno::ENOSYS),
                ("ENOTEMPTY", host_errno::ENOTEMPTY),
                ("ELOOP", host_errno::ELOOP),
                ("EOVERFLOW", host_errno::EOVERFLOW),
                ("EPROTO", host_errno::EPROTO),
                ("EDESTADDRREQ", host_errno::EDESTADDRREQ),
                ("EAFNOSUPPORT", host_errno::EAFNOSUPPORT),
                ("EALREADY", host_errno::EALREADY),
                ("EDQUOT", host_errno::EDQUOT),
            ];
            for (name, value) in entries {
                store(name, *value as i64);
            }
            #[cfg(unix)]
            {
                let unix_entries: &[(&str, i32)] = &[
                    ("ENOTBLK", host_errno::ENOTBLK),
                    ("ETXTBSY", host_errno::ETXTBSY),
                    ("ENOMSG", host_errno::ENOMSG),
                    ("EIDRM", host_errno::EIDRM),
                    ("EBADMSG", host_errno::EBADMSG),
                    ("EMULTIHOP", host_errno::EMULTIHOP),
                    ("ENODATA", host_errno::ENODATA),
                    ("ENOLINK", host_errno::ENOLINK),
                    ("ENOSR", host_errno::ENOSR),
                    ("ENOSTR", host_errno::ENOSTR),
                    ("ETIME", host_errno::ETIME),
                ];
                for (name, value) in unix_entries {
                    store(name, *value as i64);
                }
            }
        }
        #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
        {
            // darwin/BSD fallback so pyre-wasm keeps the same numeric
            // surface as before host_env existed.
            let entries: &[(&str, i64)] = &[
                ("EPERM", 1),
                ("ENOENT", 2),
                ("ESRCH", 3),
                ("EINTR", 4),
                ("EIO", 5),
                ("ENXIO", 6),
                ("E2BIG", 7),
                ("ENOEXEC", 8),
                ("EBADF", 9),
                ("ECHILD", 10),
                ("EAGAIN", 35),
                ("EWOULDBLOCK", 35),
                ("ENOMEM", 12),
                ("EACCES", 13),
                ("EFAULT", 14),
                ("ENOTBLK", 15),
                ("EBUSY", 16),
                ("EEXIST", 17),
                ("EXDEV", 18),
                ("ENODEV", 19),
                ("ENOTDIR", 20),
                ("EISDIR", 21),
                ("EINVAL", 22),
                ("ENFILE", 23),
                ("EMFILE", 24),
                ("ENOTTY", 25),
                ("ETXTBSY", 26),
                ("EFBIG", 27),
                ("ENOSPC", 28),
                ("ESPIPE", 29),
                ("EROFS", 30),
                ("EMLINK", 31),
                ("EPIPE", 32),
                ("EDOM", 33),
                ("ERANGE", 34),
                ("EDEADLK", 11),
                ("ENAMETOOLONG", 63),
                ("ENOLCK", 77),
                ("ENOSYS", 78),
                ("ENOTEMPTY", 66),
                ("ELOOP", 62),
                ("ENOMSG", 91),
                ("EIDRM", 90),
                ("EBADMSG", 94),
                ("EMULTIHOP", 95),
                ("ENODATA", 96),
                ("ENOLINK", 97),
                ("ENOSR", 98),
                ("ENOSTR", 99),
                ("EOVERFLOW", 84),
                ("EPROTO", 100),
                ("ETIME", 101),
                ("EDESTADDRREQ", 39),
                ("EAFNOSUPPORT", 47),
                ("EALREADY", 37),
                ("EDQUOT", 69),
            ];
            for (name, value) in entries {
                store(name, *value);
            }
        }
    }
}
