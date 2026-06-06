//! select.kqueue — PyPy: pypy/module/select/interp_kqueue.py W_Kqueue.
//!
//! Kept separate from `interp_kevent` because each `#[pyre_class]`
//! emits its own module-scoped `type_object()`.

#![allow(dead_code)]

#[cfg(all(target_os = "macos", feature = "host_env"))]
use super::interp_kevent::W_Kevent;
#[cfg(all(target_os = "macos", feature = "host_env"))]
use pyre_object::PyObjectRef;

/// `select.kqueue` object — PyPy: `interp_kqueue.py:118 class W_Kqueue`.
///
/// Wraps a kqueue file descriptor (`-1` once closed).  `control()`
/// marshals a changelist of `kevent`s into the syscall and returns the
/// triggered events as fresh `kevent` instances.
#[cfg(all(target_os = "macos", feature = "host_env"))]
#[crate::pyre_class("select.kqueue")]
pub struct W_Kqueue {
    kqfd: i32,
}

#[cfg(all(target_os = "macos", feature = "host_env"))]
impl Default for W_Kqueue {
    fn default() -> Self {
        W_Kqueue {
            ob: Default::default(),
            kqfd: -1,
        }
    }
}

#[cfg(all(target_os = "macos", feature = "host_env"))]
#[crate::pyre_methods(doc = "kqueue() -> kqueue object")]
impl W_Kqueue {
    /// `interp_kqueue.py:124 descr__new__` — opens a fresh kqueue fd,
    /// clearing its inheritable flag.
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let kqfd = unsafe { libc::kqueue() };
        if kqfd < 0 {
            let e = std::io::Error::last_os_error();
            return Err(crate::PyError::os_error_with_errno(
                e.raw_os_error().unwrap_or(0),
                format!("kqueue: {e}"),
            ));
        }
        unsafe {
            let flags = libc::fcntl(kqfd, libc::F_GETFD);
            if flags >= 0 {
                libc::fcntl(kqfd, libc::F_SETFD, flags | libc::FD_CLOEXEC);
            }
        }
        Ok(W_Kqueue::allocate(W_Kqueue {
            kqfd,
            ..Default::default()
        }))
    }

    /// `interp_kqueue.py:135 descr_fromfd` — wraps an existing fd.
    #[classmethod]
    fn fromfd(_cls: PyObjectRef, fd: i64) -> PyObjectRef {
        W_Kqueue::allocate(W_Kqueue {
            kqfd: fd as i32,
            ..Default::default()
        })
    }

    #[getter]
    fn closed(&self) -> bool {
        self.kqfd < 0
    }

    fn fileno(&self) -> Result<i64, crate::PyError> {
        if self.kqfd < 0 {
            return Err(crate::PyError::value_error(
                "I/O operation on closed kqueue fd",
            ));
        }
        Ok(self.kqfd as i64)
    }

    fn close(&mut self) {
        if self.kqfd >= 0 {
            let fd = self.kqfd;
            self.kqfd = -1;
            unsafe {
                libc::close(fd);
            }
        }
    }

    /// `interp_kqueue.py:167 descr_control` — apply `changelist` (a list
    /// of kevents or None) and collect up to `max_events` triggered
    /// events.  `timeout` is in seconds (float) or None to block.
    fn control(
        &mut self,
        w_changelist: PyObjectRef,
        max_events: i64,
        #[default(pyre_object::w_none())] w_timeout: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        if self.kqfd < 0 {
            return Err(crate::PyError::value_error(
                "I/O operation on closed kqueue fd",
            ));
        }
        if max_events < 0 {
            return Err(crate::PyError::value_error(format!(
                "Length of eventlist must be 0 or positive, got {max_events}"
            )));
        }

        // Build the changelist from the supplied kevent objects.
        let mut changelist: Vec<libc::kevent> = Vec::new();
        if !unsafe { pyre_object::is_none(w_changelist) } {
            // `interp_kqueue.py:179` — space.listview accepts any iterable.
            let items = crate::baseobjspace::unpackiterable(w_changelist, -1)?;
            for item in items {
                let ev = W_Kevent::from_obj(item).ok_or_else(|| {
                    crate::PyError::type_error("arg 1 must be a sequence of kevent objects")
                })?;
                changelist.push(libc::kevent {
                    ident: ev.ident as libc::uintptr_t,
                    filter: ev.filter,
                    flags: ev.flags,
                    fflags: ev.fflags,
                    data: ev.data as libc::intptr_t,
                    udata: ev.udata as *mut libc::c_void,
                });
            }
        }

        // Resolve the timeout into a `timespec` (None blocks forever).
        let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
        let have_timeout = !unsafe { pyre_object::is_none(w_timeout) };
        if have_timeout {
            // `interp_kqueue.py:187` — space.float_w honours __float__.
            let w_secs = crate::builtins::builtin_float(&[w_timeout])?;
            let secs = unsafe { pyre_object::w_float_get_value(w_secs) };
            if secs < 0.0 {
                return Err(crate::PyError::value_error(format!(
                    "Timeout must be None or >= 0, got {secs}"
                )));
            }
            ts.tv_sec = secs as libc::time_t;
            ts.tv_nsec = ((secs - secs.floor()) * 1e9) as libc::c_long;
        }

        let mut eventlist: Vec<libc::kevent> = Vec::with_capacity(max_events as usize);
        let ptimeout: *const libc::timespec = if have_timeout { &ts } else { std::ptr::null() };
        let pchangelist: *const libc::kevent = if changelist.is_empty() {
            std::ptr::null()
        } else {
            changelist.as_ptr()
        };

        // `interp_kqueue.py:214` — EINTR retry, recomputing the remaining
        // timeout each pass (`ptimeout` aliases the mutable `ts`).
        let deadline = if have_timeout {
            Some(
                std::time::Instant::now()
                    + std::time::Duration::new(ts.tv_sec.max(0) as u64, ts.tv_nsec.max(0) as u32),
            )
        } else {
            None
        };
        let nfds = loop {
            let r = unsafe {
                libc::kevent(
                    self.kqfd,
                    pchangelist,
                    changelist.len() as libc::c_int,
                    eventlist.as_mut_ptr(),
                    max_events as libc::c_int,
                    ptimeout,
                )
            };
            if r >= 0 {
                break r;
            }
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                // `interp_kqueue.py:223-226` — deliver a pending signal, then
                // retry with the remaining timeout recomputed.
                crate::module::_signal::interp_signal::checksignals_now()?;
                if let Some(dl) = deadline {
                    let now = std::time::Instant::now();
                    let rem = if now >= dl {
                        std::time::Duration::ZERO
                    } else {
                        dl - now
                    };
                    ts.tv_sec = rem.as_secs() as libc::time_t;
                    ts.tv_nsec = rem.subsec_nanos() as libc::c_long;
                }
                continue;
            }
            return Err(crate::PyError::os_error_with_errno(
                e.raw_os_error().unwrap_or(0),
                format!("kevent: {e}"),
            ));
        };
        unsafe { eventlist.set_len(nfds as usize) };

        let result: Vec<PyObjectRef> = eventlist
            .iter()
            .map(|evt| {
                W_Kevent::allocate(W_Kevent {
                    ident: evt.ident as u64,
                    filter: evt.filter,
                    flags: evt.flags,
                    fflags: evt.fflags,
                    data: evt.data as i64,
                    udata: evt.udata as u64,
                    ..Default::default()
                })
            })
            .collect();
        Ok(pyre_object::w_list_new(result))
    }
}
