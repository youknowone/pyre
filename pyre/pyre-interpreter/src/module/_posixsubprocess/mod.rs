//! _posixsubprocess module — PyPy: `pypy/module/_posixsubprocess/`.
//!
//! Backs `subprocess` on POSIX through `fork_exec`.  The whole surface is
//! gated on `cfg(all(unix, feature = "host_env"))`; non-Unix /
//! `host_env = off` builds expose an empty module so `import
//! _posixsubprocess` still succeeds (matching PyPy's mixedmodule
//! behaviour when the conditional `interpleveldefs` entry is absent).

use pyre_object::*;

#[cfg(all(unix, feature = "host_env"))]
mod imp {
    use super::*;
    use crate::PyError;
    use core::{convert::Infallible, ffi::CStr, marker::PhantomData};
    use rustpython_host_env::posix as host_posix;
    use std::ffi::CString;
    use std::os::fd::{AsFd, BorrowedFd};

    /// Null-terminated `*const c_char` array, kept alive by the borrowed
    /// `CString`s it points into.  `argv`/`envp` for `exec*`.
    #[derive(Default)]
    struct CharPtrVec<'a> {
        vec: Vec<*const libc::c_char>,
        marker: PhantomData<Vec<&'a CStr>>,
    }

    impl<'a, T: AsRef<CStr>> FromIterator<&'a T> for CharPtrVec<'a> {
        fn from_iter<I: IntoIterator<Item = &'a T>>(iter: I) -> Self {
            let vec = iter
                .into_iter()
                .map(|x| x.as_ref().as_ptr())
                .chain(core::iter::once(core::ptr::null()))
                .collect();
            Self {
                vec,
                marker: PhantomData,
            }
        }
    }

    impl CharPtrVec<'_> {
        fn as_ptr(&self) -> *const *const libc::c_char {
            self.vec.as_ptr()
        }
    }

    fn io_err(e: std::io::Error) -> PyError {
        PyError::os_error_with_errno(e.raw_os_error().unwrap_or(0), e.to_string())
    }

    fn is_none_obj(o: PyObjectRef) -> bool {
        unsafe { is_none(o) }
    }

    fn fd_arg(o: PyObjectRef) -> i32 {
        (unsafe { w_int_get_value(o) }) as i32
    }

    fn seq_items(o: PyObjectRef, what: &str) -> Result<Vec<PyObjectRef>, PyError> {
        unsafe {
            if is_list(o) {
                let n = w_list_len(o);
                Ok((0..n).filter_map(|i| w_list_getitem(o, i as i64)).collect())
            } else if is_tuple(o) {
                let n = w_tuple_len(o);
                Ok((0..n)
                    .filter_map(|i| w_tuple_getitem(o, i as i64))
                    .collect())
            } else {
                Err(PyError::type_error(format!(
                    "fork_exec(): {what} must be a list or tuple"
                )))
            }
        }
    }

    fn obj_to_cstring(o: PyObjectRef, what: &str) -> Result<CString, PyError> {
        let bytes = unsafe {
            if is_str(o) {
                w_str_get_value(o).as_bytes().to_vec()
            } else if is_bytes(o) {
                w_bytes_data(o).to_vec()
            } else {
                return Err(PyError::type_error(format!(
                    "fork_exec(): {what} must be str or bytes"
                )));
            }
        };
        CString::new(bytes)
            .map_err(|_| PyError::value_error(format!("fork_exec(): embedded null in {what}")))
    }

    fn collect_cstrings(o: PyObjectRef, what: &str) -> Result<Vec<CString>, PyError> {
        seq_items(o, what)?
            .into_iter()
            .map(|x| obj_to_cstring(x, what))
            .collect()
    }

    fn opt_cstring(o: PyObjectRef, what: &str) -> Result<Option<CString>, PyError> {
        if is_none_obj(o) {
            Ok(None)
        } else {
            Ok(Some(obj_to_cstring(o, what)?))
        }
    }

    fn collect_fds(o: PyObjectRef) -> Result<Vec<BorrowedFd<'static>>, PyError> {
        Ok(seq_items(o, "fds_to_keep")?
            .into_iter()
            .map(|x| unsafe { BorrowedFd::borrow_raw((w_int_get_value(x)) as i32) })
            .collect())
    }

    /// `_Py_Gid_Converter`/`_Py_Uid_Converter`: accept `id >= -1`, with
    /// `-1` mapping to `u32::MAX` (an unset sentinel the `set*id_if_needed`
    /// helpers skip).
    fn try_from_id(o: PyObjectRef, name: &str) -> Result<u32, PyError> {
        use core::cmp::Ordering;
        let i = unsafe { w_int_get_value(o) };
        match i.cmp(&-1) {
            Ordering::Greater => u32::try_from(i)
                .map_err(|_| PyError::overflow_error(format!("{name} is larger than maximum"))),
            Ordering::Less => Err(PyError::overflow_error(format!(
                "{name} is less than minimum"
            ))),
            Ordering::Equal => Ok(-1i32 as u32),
        }
    }

    fn opt_id(o: PyObjectRef, name: &str) -> Result<Option<u32>, PyError> {
        if is_none_obj(o) {
            Ok(None)
        } else {
            Ok(Some(try_from_id(o, name)?))
        }
    }

    fn collect_gids(o: PyObjectRef) -> Result<Vec<u32>, PyError> {
        seq_items(o, "gids")?
            .into_iter()
            .map(|x| try_from_id(x, "gid"))
            .collect()
    }

    /// Decoded `fork_exec` arguments, allocated before `fork()` so the
    /// child does no further allocation before `exec`.
    struct Decoded<'a> {
        exec_list: &'a [CString],
        argv: *const *const libc::c_char,
        envp: Option<*const *const libc::c_char>,
        fds_to_keep: &'a [BorrowedFd<'static>],
        extra_groups: Option<&'a [u32]>,
        cwd: Option<&'a CString>,
        preexec_fn: Option<PyObjectRef>,
        close_fds: bool,
        restore_signals: bool,
        call_setsid: bool,
        pgid_to_set: libc::pid_t,
        gid: Option<u32>,
        uid: Option<u32>,
        child_umask: i32,
        p2cread: i32,
        p2cwrite: i32,
        c2pread: i32,
        c2pwrite: i32,
        errread: i32,
        errwrite: i32,
        errpipe_read: i32,
        errpipe_write: i32,
    }

    enum ExecErrorContext {
        NoExec,
        ChDir,
        PreExec,
        Exec,
    }

    impl ExecErrorContext {
        const fn as_msg(&self) -> &'static str {
            match self {
                Self::NoExec => "noexec",
                Self::ChDir => "noexec:chdir",
                Self::PreExec => "Exception occurred in preexec_fn.",
                Self::Exec => "",
            }
        }
    }

    fn exec_inner(d: &Decoded<'_>, ctx: &mut ExecErrorContext) -> std::io::Result<Infallible> {
        let errpipe_write = unsafe { BorrowedFd::borrow_raw(d.errpipe_write) };
        host_posix::setup_child_fds(
            d.fds_to_keep,
            errpipe_write.as_fd(),
            d.p2cread,
            d.p2cwrite,
            d.c2pread,
            d.c2pwrite,
            d.errread,
            d.errwrite,
            d.errpipe_read,
        )?;

        if let Some(cwd) = d.cwd {
            host_posix::chdir(cwd.as_c_str()).inspect_err(|_| *ctx = ExecErrorContext::ChDir)?;
        }

        host_posix::set_umask(d.child_umask);

        if d.restore_signals {
            host_posix::restore_signals();
        }

        host_posix::setsid_if_needed(d.call_setsid)?;
        host_posix::setpgid_if_needed(d.pgid_to_set)?;
        host_posix::setgroups_if_needed(d.extra_groups)?;
        host_posix::setregid_if_needed(d.gid)?;
        host_posix::setreuid_if_needed(d.uid)?;

        // Call preexec_fn after all process setup but before closing FDs.
        if let Some(preexec_fn) = d.preexec_fn {
            let r = crate::baseobjspace::call_function(preexec_fn, &[]);
            if r.is_null() {
                // Cannot safely stringify the exception after fork.
                let _ = crate::call::take_call_error();
                *ctx = ExecErrorContext::PreExec;
                return Err(std::io::Error::from_raw_os_error(0));
            }
        }

        *ctx = ExecErrorContext::Exec;

        if d.close_fds {
            host_posix::close_fds(2, d.fds_to_keep);
        }

        let err = host_posix::exec_replace(d.exec_list, d.argv, d.envp);
        Err(std::io::Error::from_raw_os_error(err as i32))
    }

    fn exec(d: &Decoded<'_>) -> ! {
        let mut ctx = ExecErrorContext::NoExec;
        match exec_inner(d, &mut ctx) {
            Ok(infallible) => match infallible {},
            Err(e) => {
                let errpipe = unsafe { BorrowedFd::borrow_raw(d.errpipe_write) };
                let msg = if matches!(ctx, ExecErrorContext::PreExec) {
                    // preexec_fn failures use SubprocessError format (errno=0).
                    format!("SubprocessError:0:{}", ctx.as_msg())
                } else {
                    // errno is written in hex.
                    let errno = e.raw_os_error().unwrap_or(0);
                    format!("OSError:{errno:x}:{}", ctx.as_msg())
                };
                let _ = host_posix::write_fd(errpipe.as_fd(), msg.as_bytes());
                rustpython_host_env::os::exit(255)
            }
        }
    }

    pub fn fork_exec(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        let (pos, _kwargs) = crate::builtins::split_builtin_kwargs(args);
        if pos.len() != 22 {
            return Err(PyError::type_error(format!(
                "fork_exec() takes exactly 22 arguments ({} given)",
                pos.len()
            )));
        }

        // Decode everything (and pre-allocate the argv/envp arrays) before
        // fork(): the child must not allocate before exec.
        let args_list = collect_cstrings(pos[0], "args")?;
        let exec_list = collect_cstrings(pos[1], "executable_list")?;
        let close_fds = crate::baseobjspace::is_true(pos[2])?;
        let fds_to_keep = collect_fds(pos[3])?;
        let cwd = opt_cstring(pos[4], "cwd")?;
        let env_list = if is_none_obj(pos[5]) {
            None
        } else {
            Some(collect_cstrings(pos[5], "env_list")?)
        };
        let p2cread = fd_arg(pos[6]);
        let p2cwrite = fd_arg(pos[7]);
        let c2pread = fd_arg(pos[8]);
        let c2pwrite = fd_arg(pos[9]);
        let errread = fd_arg(pos[10]);
        let errwrite = fd_arg(pos[11]);
        let errpipe_read = fd_arg(pos[12]);
        let errpipe_write = fd_arg(pos[13]);
        let restore_signals = crate::baseobjspace::is_true(pos[14])?;
        let call_setsid = crate::baseobjspace::is_true(pos[15])?;
        let pgid_to_set = (unsafe { w_int_get_value(pos[16]) }) as libc::pid_t;
        let gid = opt_id(pos[17], "gid")?;
        let extra_groups = if is_none_obj(pos[18]) {
            None
        } else {
            Some(collect_gids(pos[18])?)
        };
        let uid = opt_id(pos[19], "uid")?;
        let child_umask = (unsafe { w_int_get_value(pos[20]) }) as i32;
        let preexec_fn = if is_none_obj(pos[21]) {
            None
        } else {
            Some(pos[21])
        };

        let argv = args_list.iter().collect::<CharPtrVec<'_>>();
        let envp = env_list
            .as_ref()
            .map(|e| e.iter().collect::<CharPtrVec<'_>>());

        let decoded = Decoded {
            exec_list: &exec_list,
            argv: argv.as_ptr(),
            envp: envp.as_ref().map(CharPtrVec::as_ptr),
            fds_to_keep: &fds_to_keep,
            extra_groups: extra_groups.as_deref(),
            cwd: cwd.as_ref(),
            preexec_fn,
            close_fds,
            restore_signals,
            call_setsid,
            pgid_to_set,
            gid,
            uid,
            child_umask,
            p2cread,
            p2cwrite,
            c2pread,
            c2pwrite,
            errread,
            errwrite,
            errpipe_read,
            errpipe_write,
        };

        match host_posix::fork().map_err(io_err)? {
            0 => exec(&decoded),
            child => Ok(w_int_new(child as i64)),
        }
    }
}

crate::py_module! {
    "_posixsubprocess",
    extra_init: |ns| {
        #[cfg(all(unix, feature = "host_env"))]
        crate::dict_storage_store(
            ns,
            "fork_exec",
            crate::make_builtin_function("fork_exec", imp::fork_exec),
        );
        #[cfg(not(all(unix, feature = "host_env")))]
        let _ = ns;
    }
}
