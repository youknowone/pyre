//! select implementation — PyPy: pypy/module/select/interp_select.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;
#[cfg(all(unix, feature = "host_env"))]
use pyre_object::PyObjectRef;

/// `select.poll` object — PyPy: `interp_select.py:26 class Poll`.
///
/// Holds the registered `{fd: events}` map and a re-entrancy guard.
/// Instances are created only through the module-level `select.poll()`
/// factory (`interp_select.py:18`); the type has no public constructor.
#[cfg(all(unix, feature = "host_env"))]
#[crate::pyre_class("select.poll")]
#[derive(Default)]
pub struct W_Poll {
    fddict: std::collections::HashMap<i32, i16>,
    running: bool,
}

/// `interp_select.py:15 defaultevents = POLLIN | POLLOUT | POLLPRI`.
#[cfg(all(unix, feature = "host_env"))]
fn default_poll_events() -> i16 {
    (libc::POLLIN | libc::POLLOUT | libc::POLLPRI) as i16
}

/// Resolve a Python fd argument (int or object with `fileno()`) to a
/// raw descriptor — `space.c_filedescriptor_w`.
#[cfg(all(unix, feature = "host_env"))]
pub(crate) fn filedescriptor_w(w_fd: PyObjectRef) -> Result<i32, crate::PyError> {
    unsafe {
        // A real int (or int subclass / bignum) is taken directly; otherwise
        // `fileno()` is called.  An object with only `__int__` is rejected.
        let w_int = if pyre_object::is_int_or_long(w_fd) {
            w_fd
        } else {
            let fileno = crate::baseobjspace::getattr_str(w_fd, "fileno").map_err(|_| {
                crate::PyError::type_error("argument must be an int, or have a fileno() method.")
            })?;
            let res = crate::call::call_function_impl_result(fileno, &[])?;
            if !pyre_object::is_int_or_long(res) {
                return Err(crate::PyError::type_error("fileno() returned a non-integer"));
            }
            res
        };
        // `c_int_w` — OverflowError if it does not fit a 32-bit int.
        let fd = crate::baseobjspace::c_int_w(w_int)?;
        if fd < 0 {
            return Err(crate::PyError::value_error(format!(
                "file descriptor cannot be a negative integer ({fd})"
            )));
        }
        Ok(fd)
    }
}

#[cfg(all(unix, feature = "host_env"))]
#[crate::pyre_methods(
    doc = "Returns a polling object.\n\nSee the poll() documentation.",
    unhashable
)]
impl W_Poll {
    /// `interp_select.py:115-117 descr_new` — the type is not directly
    /// instantiable; `select.poll()` is the module-level factory.
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        Err(crate::PyError::type_error(
            "cannot create 'select.poll' instances",
        ))
    }

    /// `interp_select.py:32 Poll.register` — `events` defaults to
    /// `POLLIN | POLLOUT | POLLPRI`.
    fn register(
        &mut self,
        w_fd: PyObjectRef,
        #[default(pyre_object::w_none())] w_events: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let fd = filedescriptor_w(w_fd)?;
        // @unwrap_spec(events="c_ushort"): reject negative / >0xffff.
        let events = if unsafe { pyre_object::is_none(w_events) } {
            default_poll_events()
        } else {
            crate::baseobjspace::c_ushort_w(w_events)? as i16
        };
        self.fddict.insert(fd, events);
        Ok(())
    }

    /// `interp_select.py:43 Poll.modify` — raises `OSError(ENOENT)` for
    /// a descriptor that was never registered.
    fn modify(&mut self, w_fd: PyObjectRef, w_events: PyObjectRef) -> Result<(), crate::PyError> {
        let fd = filedescriptor_w(w_fd)?;
        // @unwrap_spec(events="c_ushort"): reject negative / >0xffff.
        let events = crate::baseobjspace::c_ushort_w(w_events)? as i16;
        let known = self.fddict.contains_key(&fd);
        if known {
            self.fddict.insert(fd, events);
            Ok(())
        } else {
            Err(crate::PyError::os_error_with_errno(
                libc::ENOENT,
                "poll.modify",
            ))
        }
    }

    /// `interp_select.py:56 Poll.unregister` — raises `KeyError(fd)` for
    /// an unknown descriptor.
    fn unregister(&mut self, w_fd: PyObjectRef) -> Result<(), crate::PyError> {
        let fd = filedescriptor_w(w_fd)?;
        if self.fddict.remove(&fd).is_none() {
            return Err(crate::PyError::key_error_with_key(pyre_object::w_int_new(
                fd as i64,
            )));
        }
        Ok(())
    }

    /// `interp_select.py:67 Poll.poll` — `timeout` is in milliseconds;
    /// `None` or a negative value blocks indefinitely.  Returns a list
    /// of `(fd, revents)` for the descriptors with pending events.
    fn poll(
        &mut self,
        #[default(pyre_object::w_none())] w_timeout: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        // `None` / negative → block indefinitely (timeout = -1).  Otherwise
        // `c_int_w(space.int(w_timeout))`: truncate a float to int, then
        // range-check to a 32-bit C int (millisecond count).
        let timeout: i32 = if unsafe { pyre_object::is_none(w_timeout) } {
            -1
        } else if unsafe { pyre_object::is_int(w_timeout) } {
            let t = unsafe { pyre_object::w_int_get_value(w_timeout) };
            if t < 0 {
                -1
            } else if t > i32::MAX as i64 {
                return Err(crate::PyError::overflow_error("expected a 32-bit integer"));
            } else {
                t as i32
            }
        } else if unsafe { pyre_object::is_float(w_timeout) } {
            let t = unsafe { pyre_object::w_float_get_value(w_timeout) };
            if t < 0.0 {
                -1
            } else {
                let trunc = t.trunc();
                if trunc > i32::MAX as f64 {
                    return Err(crate::PyError::overflow_error("expected a 32-bit integer"));
                }
                trunc as i32
            }
        } else {
            return Err(crate::PyError::type_error(
                "timeout must be an integer or None",
            ));
        };

        if self.running {
            return Err(crate::PyError::runtime_error(
                "concurrent poll() invocation",
            ));
        }

        let mut pollfds: Vec<libc::pollfd> = self
            .fddict
            .iter()
            .map(|(&fd, &events)| libc::pollfd {
                fd,
                events,
                revents: 0,
            })
            .collect();

        // EINTR retry with a recomputed timeout, mirroring
        // `interp_select.py:89` (round the remaining time up to the next ms).
        let deadline = (timeout >= 0)
            .then(|| std::time::Instant::now() + std::time::Duration::from_millis(timeout as u64));
        let mut cur_timeout = timeout;
        self.running = true;
        let ret = loop {
            let r =
                unsafe { libc::poll(pollfds.as_mut_ptr(), pollfds.len() as _, cur_timeout) };
            if r >= 0 {
                break r;
            }
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                // `interp_select.py:94-100` — deliver a pending signal, then
                // retry with a recomputed timeout.  Reset `running` first so
                // a raised handler does not leave the poll object wedged
                // (PyPy's `finally: self.running = False`).
                if let Err(err) = crate::module::_signal::interp_signal::checksignals_now() {
                    self.running = false;
                    return Err(err);
                }
                if let Some(dl) = deadline {
                    let now = std::time::Instant::now();
                    cur_timeout = if now >= dl {
                        0
                    } else {
                        ((dl - now).as_secs_f64() * 1000.0 + 0.999) as i32
                    };
                }
                continue;
            }
            self.running = false;
            return Err(crate::PyError::os_error_with_errno(
                e.raw_os_error().unwrap_or(0),
                format!("poll: {e}"),
            ));
        };
        self.running = false;
        let _ = ret;

        let retval: Vec<PyObjectRef> = pollfds
            .iter()
            .filter(|pfd| pfd.revents != 0)
            .map(|pfd| {
                pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(pfd.fd as i64),
                    pyre_object::w_int_new(pfd.revents as i64),
                ])
            })
            .collect();
        Ok(pyre_object::w_list_new(retval))
    }
}

/// _select module — PyPy: pypy/module/select/.
///
/// Implements `select.select(rlist, wlist, xlist, timeout=None)` via
/// `rustpython_host_env::select::{FdSet, select, sec_to_timeval}` and the
/// `select.poll()` polling object.  epoll / kqueue object types are not
/// implemented yet.
pub fn register_module(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "select",
        crate::make_builtin_function("select", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                use rustpython_host_env::select as host_select;

                if args.len() < 3 {
                    return Err(crate::PyError::type_error(
                        "select() takes at least 3 arguments",
                    ));
                }

                // `interp_select.py:226` — `space.unpackiterable` accepts any
                // iterable (list, tuple, generator, …); each item is an int
                // fd or an object exposing fileno().
                fn collect_fds(
                    seq: pyre_object::PyObjectRef,
                ) -> Result<Vec<(pyre_object::PyObjectRef, i32)>, crate::PyError> {
                    let items = crate::baseobjspace::unpackiterable(seq, -1)?;
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        // `interp_select.py:132 _build_fd_set` — each item is
                        // resolved through `space.c_filedescriptor_w`.
                        let fd = filedescriptor_w(item)?;
                        // `fd >= FD_SETSIZE` is rejected: `FD_SET` on such an
                        // fd writes outside the `fd_set` bitmap.
                        if fd >= libc::FD_SETSIZE as i32 {
                            return Err(crate::PyError::value_error(
                                "file descriptor out of range in select()",
                            ));
                        }
                        out.push((item, fd));
                    }
                    Ok(out)
                }

                let rfds = collect_fds(args[0])?;
                let wfds = collect_fds(args[1])?;
                let xfds = collect_fds(args[2])?;

                let mut nfds: i32 = -1;
                for fds in [&rfds, &wfds, &xfds] {
                    for &(_, fd) in fds {
                        if fd > nfds {
                            nfds = fd;
                        }
                    }
                }

                // `interp_select.py:230-235` — `None` blocks forever, else
                // `space.float_w` (applies `__float__`); a negative count is
                // a ValueError.
                let timeout_secs: Option<f64> = match args.get(3) {
                    None => None,
                    Some(&t) if unsafe { pyre_object::is_none(t) } => None,
                    Some(&t) => {
                        let secs = crate::baseobjspace::float_w(t)?;
                        if secs < 0.0 {
                            return Err(crate::PyError::value_error(
                                "timeout must be non-negative",
                            ));
                        }
                        Some(secs)
                    }
                };

                // `interp_select.py:166` — EINTR retry, recomputing the
                // remaining timeout each pass and rebuilding the fd sets
                // (select() clobbers them on every call).
                // `Duration::from_secs_f64` panics on a NaN/inf/overflowing
                // timeout; `float_w` lets such a value through, so convert it
                // into a `ValueError` instead of aborting the host process.
                let deadline = match timeout_secs {
                    None => None,
                    Some(s) => Some(
                        std::time::Duration::try_from_secs_f64(s)
                            .ok()
                            .and_then(|d| std::time::Instant::now().checked_add(d))
                            .ok_or_else(|| {
                                crate::PyError::value_error("timeout is too large")
                            })?,
                    ),
                };
                let mut rset = host_select::FdSet::new();
                let mut wset = host_select::FdSet::new();
                let mut xset = host_select::FdSet::new();
                loop {
                    rset = host_select::FdSet::new();
                    wset = host_select::FdSet::new();
                    xset = host_select::FdSet::new();
                    for &(_, fd) in &rfds {
                        rset.insert(fd);
                    }
                    for &(_, fd) in &wfds {
                        wset.insert(fd);
                    }
                    for &(_, fd) in &xfds {
                        xset.insert(fd);
                    }
                    let mut tv_storage;
                    let timeout_ref: Option<&mut host_select::timeval> = match timeout_secs {
                        None => None,
                        Some(_) => {
                            let remaining = deadline
                                .map(|dl| {
                                    let now = std::time::Instant::now();
                                    if now >= dl {
                                        0.0
                                    } else {
                                        (dl - now).as_secs_f64()
                                    }
                                })
                                .unwrap_or(0.0);
                            tv_storage = host_select::sec_to_timeval(remaining);
                            Some(&mut tv_storage)
                        }
                    };
                    match host_select::select(
                        nfds + 1,
                        &mut rset,
                        &mut wset,
                        &mut xset,
                        timeout_ref,
                    ) {
                        Ok(_) => break,
                        Err(e) if e.raw_os_error() == Some(libc::EINTR) => {
                            // `interp_select.py:182` — deliver a pending
                            // signal, then retry with the remaining timeout
                            // recomputed at the loop head.
                            crate::module::_signal::interp_signal::checksignals_now()?;
                            continue;
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("select: {e}"),
                            ));
                        }
                    }
                }

                fn build_ready(
                    set: &mut host_select::FdSet,
                    inputs: &[(pyre_object::PyObjectRef, i32)],
                ) -> pyre_object::PyObjectRef {
                    let items: Vec<_> = inputs
                        .iter()
                        .filter_map(|&(obj, fd)| if set.contains(fd) { Some(obj) } else { None })
                        .collect();
                    pyre_object::w_list_new(items)
                }

                let r_ready = build_ready(&mut rset, &rfds);
                let w_ready = build_ready(&mut wset, &wfds);
                let x_ready = build_ready(&mut xset, &xfds);
                Ok(pyre_object::w_tuple_new(vec![r_ready, w_ready, x_ready]))
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "select.select requires host_env feature on a Unix platform",
                ))
            }
        }),
    );

    // `interp_select.py:18 poll()` — factory returning a fresh polling
    // object.  The type has no public constructor, matching
    // `interp_select.py:115 descr_new` which raises TypeError.
    #[cfg(all(unix, feature = "host_env"))]
    {
        // Force the `select.poll` type to register so instances carry a
        // valid `ob_type`.  `interp_select.py:123
        // Poll.typedef.acceptable_as_base_class = False`.
        let _ = type_object();
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(type_object(), false) };
        crate::dict_storage_store(
            ns,
            "poll",
            crate::make_builtin_function_with_arity(
                "poll",
                |_args| Ok(W_Poll::allocate(W_Poll::default())),
                0,
            ),
        );
        // `interp_select.py` exposes the rpoll event names as module
        // constants (`rpoll.eventnames`).
        macro_rules! ev {
            ($name:literal, $val:expr) => {
                crate::dict_storage_store(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        ev!("POLLIN", libc::POLLIN);
        ev!("POLLPRI", libc::POLLPRI);
        ev!("POLLOUT", libc::POLLOUT);
        ev!("POLLERR", libc::POLLERR);
        ev!("POLLHUP", libc::POLLHUP);
        ev!("POLLNVAL", libc::POLLNVAL);
        ev!("POLLRDNORM", libc::POLLRDNORM);
        ev!("POLLRDBAND", libc::POLLRDBAND);
        ev!("POLLWRNORM", libc::POLLWRNORM);
        ev!("POLLWRBAND", libc::POLLWRBAND);
    }

    // `interp_kqueue.py` — kqueue() / kevent objects plus the KQ_* event
    // filter and flag constants (BSD/macOS only).
    #[cfg(all(target_os = "macos", feature = "host_env"))]
    {
        crate::dict_storage_store(ns, "kqueue", super::interp_kqueue::type_object());
        crate::dict_storage_store(ns, "kevent", super::interp_kevent::type_object());
        // `interp_kqueue.py:262 W_Kqueue.typedef.acceptable_as_base_class
        // = False` / `:406 W_Kevent.typedef.acceptable_as_base_class =
        // False`.
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(
                super::interp_kqueue::type_object(),
                false,
            );
            pyre_object::w_type_set_acceptable_as_base_class(
                super::interp_kevent::type_object(),
                false,
            );
        }
        macro_rules! kq {
            ($name:literal, $val:expr) => {
                crate::dict_storage_store(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        // `interp_kqueue.py:62 symbol_map` — KQ_FILTER_* / KQ_EV_*.
        kq!("KQ_FILTER_READ", libc::EVFILT_READ);
        kq!("KQ_FILTER_WRITE", libc::EVFILT_WRITE);
        kq!("KQ_FILTER_AIO", libc::EVFILT_AIO);
        kq!("KQ_FILTER_VNODE", libc::EVFILT_VNODE);
        kq!("KQ_FILTER_PROC", libc::EVFILT_PROC);
        kq!("KQ_FILTER_SIGNAL", libc::EVFILT_SIGNAL);
        kq!("KQ_FILTER_TIMER", libc::EVFILT_TIMER);
        kq!("KQ_EV_ADD", libc::EV_ADD);
        kq!("KQ_EV_DELETE", libc::EV_DELETE);
        kq!("KQ_EV_ENABLE", libc::EV_ENABLE);
        kq!("KQ_EV_DISABLE", libc::EV_DISABLE);
        kq!("KQ_EV_ONESHOT", libc::EV_ONESHOT);
        kq!("KQ_EV_CLEAR", libc::EV_CLEAR);
        kq!("KQ_EV_EOF", libc::EV_EOF);
        kq!("KQ_EV_ERROR", libc::EV_ERROR);
    }

    // `interp_select.py:35 W_Error = OSError` — expose the real type so
    // `except select.error` catches what selectors raise.
    let w_os_error = crate::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before select init");
    crate::dict_storage_store(ns, "error", w_os_error);
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "PIPE_BUF",
            pyre_object::w_int_new(libc::PIPE_BUF as i64),
        );
    }
}
