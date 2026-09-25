//! select.kqueue — PyPy: pypy/module/select/interp_kqueue.py W_Kqueue.
//!
//! Kept separate from `interp_kevent` because each `#[pyre_class]`
//! emits its own module-scoped `type_object()`.

#![allow(dead_code)]

#[cfg(all(target_os = "macos", feature = "host_env"))]
use super::interp_kevent::W_Kevent;
#[cfg(all(target_os = "macos", feature = "host_env"))]
use pyre_object::PyObjectRef;
#[cfg(all(target_os = "macos", feature = "host_env"))]
use rustpython_host_env::select::kqueue as host_kqueue;

/// `select.kqueue` object — PyPy: `interp_kqueue.py class W_Kqueue`.
///
/// Wraps a kqueue file descriptor (`-1` once closed).  `control()`
/// marshals a changelist of `kevent`s into the syscall and returns the
/// triggered events as fresh `kevent` instances.
#[cfg(all(target_os = "macos", feature = "host_env"))]
// CPython 3.14 Modules/selectmodule.c:select_exec creates
// kqueue_queue_Type_spec as a mutable module heap type.
#[pyre_interpreter::pyre_class("select.kqueue", cpython_mutable)]
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
#[pyre_interpreter::pyre_methods(doc = "kqueue() -> kqueue object")]
impl W_Kqueue {
    /// `interp_kqueue.py descr__new__` — opens a fresh kqueue fd,
    /// clearing its inheritable flag.
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let cell = host_kqueue::create().map_err(|e| {
            pyre_interpreter::PyError::os_error_with_errno(
                e.raw_os_error().unwrap_or(0),
                format!("kqueue: {e}"),
            )
        })?;
        Ok(W_Kqueue::allocate(W_Kqueue {
            kqfd: host_kqueue::fd(&cell),
            ..Default::default()
        }))
    }

    /// `interp_kqueue.py descr_fromfd` — wraps an existing fd.
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

    fn fileno(&self) -> Result<i64, pyre_interpreter::PyError> {
        if self.kqfd < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "I/O operation on closed kqueue fd",
            ));
        }
        Ok(self.kqfd as i64)
    }

    fn close(&mut self) {
        if self.kqfd >= 0 {
            let cell = host_kqueue::from_fd(self.kqfd);
            self.kqfd = -1;
            let _ = host_kqueue::close(&cell);
        }
    }

    /// `interp_kqueue.py descr_control` — apply `changelist` (a list
    /// of kevents or None) and collect up to `max_events` triggered
    /// events.  `timeout` is in seconds (float) or None to block.
    fn control(
        &mut self,
        w_changelist: PyObjectRef,
        max_events: i64,
        #[default(pyre_object::w_none())] w_timeout: PyObjectRef,
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        if self.kqfd < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "I/O operation on closed kqueue fd",
            ));
        }
        if max_events < 0 {
            return Err(pyre_interpreter::PyError::value_error(format!(
                "Length of eventlist must be 0 or positive, got {max_events}"
            )));
        }

        // Build the changelist from the supplied kevent objects.
        let mut changelist: Vec<host_kqueue::Event> = Vec::new();
        if !unsafe { pyre_object::is_none(w_changelist) } {
            // `interp_kqueue.py descr_control` — space.listview accepts any iterable.
            let items = pyre_interpreter::baseobjspace::unpackiterable(w_changelist, -1)?;
            for item in items {
                let ev = W_Kevent::from_obj(item).ok_or_else(|| {
                    pyre_interpreter::PyError::type_error(
                        "arg 1 must be a sequence of kevent objects",
                    )
                })?;
                changelist.push(host_kqueue::Event {
                    ident: ev.ident as usize,
                    filter: ev.filter,
                    flags: ev.flags,
                    fflags: ev.fflags,
                    data: ev.data as isize,
                    udata: ev.udata as usize,
                });
            }
        }

        // Resolve the timeout into a `Timespec` (None blocks forever).
        let mut timeout =
            if unsafe { pyre_object::is_none(w_timeout) } {
                None
            } else {
                // `interp_kqueue.py descr_control` — space.float_w honours __float__.
                let w_secs = pyre_interpreter::builtins::builtin_float(&[w_timeout])?;
                let secs = unsafe { pyre_object::w_float_get_value(w_secs) };
                if secs < 0.0 {
                    return Err(pyre_interpreter::PyError::value_error(format!(
                        "Timeout must be None or >= 0, got {secs}"
                    )));
                }
                Some(host_kqueue::Timespec::from_secs(secs).ok_or_else(|| {
                    pyre_interpreter::PyError::overflow_error("timeout is too large")
                })?)
            };

        let mut eventlist = vec![host_kqueue::Event::default(); max_events as usize];

        // `interp_kqueue.py descr_control` — EINTR retry, recomputing the
        // remaining timeout each pass.
        let deadline = match timeout.as_ref().and_then(|ts| ts.to_duration()) {
            Some(d) => Some(std::time::Instant::now().checked_add(d).ok_or_else(|| {
                pyre_interpreter::PyError::overflow_error("timeout is too large")
            })?),
            None => None,
        };
        let nfds = loop {
            let (result, errno) = pyre_interpreter::module::thread::call_external_function(|| {
                host_kqueue::kevent(self.kqfd, &changelist, &mut eventlist, timeout.as_ref())
            });
            match result {
                Ok(n) => break n,
                Err(e) if rustpython_host_env::io::is_interrupted_error(&e) => {
                    // `interp_kqueue.py descr_control` — deliver a pending
                    // signal, then retry with the remaining timeout recomputed.
                    pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                    if let Some(dl) = deadline {
                        timeout = Some(host_kqueue::Timespec::from_duration(
                            dl.saturating_duration_since(std::time::Instant::now()),
                        ));
                    }
                    continue;
                }
                Err(e) => {
                    let errno = e.raw_os_error().unwrap_or(errno);
                    return Err(pyre_interpreter::PyError::os_error_with_errno(
                        errno,
                        format!("kevent: {e}"),
                    ));
                }
            }
        };

        let result: Vec<PyObjectRef> = eventlist
            .iter()
            .take(nfds)
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
