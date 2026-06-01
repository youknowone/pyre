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
fn filedescriptor_w(w_fd: PyObjectRef) -> Result<i32, crate::PyError> {
    unsafe {
        let fd_val = if pyre_object::is_int(w_fd) {
            pyre_object::w_int_get_value(w_fd)
        } else {
            let fileno = crate::baseobjspace::getattr(w_fd, "fileno").map_err(|_| {
                crate::PyError::type_error("argument must be an int, or have a fileno() method")
            })?;
            let res = crate::call::call_function_impl_result(fileno, &[])?;
            if !pyre_object::is_int(res) {
                return Err(crate::PyError::type_error("fileno() must return an integer"));
            }
            pyre_object::w_int_get_value(res)
        };
        if fd_val < 0 {
            return Err(crate::PyError::value_error(
                "file descriptor cannot be a negative integer",
            ));
        }
        if fd_val > i32::MAX as i64 {
            return Err(crate::PyError::overflow_error("file descriptor out of range"));
        }
        Ok(fd_val as i32)
    }
}

#[cfg(all(unix, feature = "host_env"))]
#[crate::pyre_methods(
    doc = "Returns a polling object.\n\nSee the poll() documentation.",
    unhashable
)]
impl W_Poll {
    /// `interp_select.py:32 Poll.register` — `events` defaults to
    /// `POLLIN | POLLOUT | POLLPRI`.
    fn register(
        &mut self,
        w_fd: PyObjectRef,
        #[default(pyre_object::w_none())] w_events: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let fd = filedescriptor_w(w_fd)?;
        let events = if unsafe { pyre_object::is_none(w_events) } {
            default_poll_events()
        } else if unsafe { pyre_object::is_int(w_events) } {
            unsafe { pyre_object::w_int_get_value(w_events) as i16 }
        } else {
            return Err(crate::PyError::type_error("events must be an integer"));
        };
        self.fddict.insert(fd, events);
        Ok(())
    }

    /// `interp_select.py:43 Poll.modify` — raises `OSError(ENOENT)` for
    /// a descriptor that was never registered.
    fn modify(&mut self, w_fd: PyObjectRef, w_events: PyObjectRef) -> Result<(), crate::PyError> {
        let fd = filedescriptor_w(w_fd)?;
        if !unsafe { pyre_object::is_int(w_events) } {
            return Err(crate::PyError::type_error("events must be an integer"));
        }
        let events = unsafe { pyre_object::w_int_get_value(w_events) as i16 };
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
        let timeout: i32 = if unsafe { pyre_object::is_none(w_timeout) } {
            -1
        } else if unsafe { pyre_object::is_int(w_timeout) } {
            let t = unsafe { pyre_object::w_int_get_value(w_timeout) };
            if t < 0 { -1 } else { t.min(i32::MAX as i64) as i32 }
        } else if unsafe { pyre_object::is_float(w_timeout) } {
            let t = unsafe { pyre_object::w_float_get_value(w_timeout) };
            if t < 0.0 { -1 } else { t as i32 }
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

        self.running = true;
        let ret = unsafe { libc::poll(pollfds.as_mut_ptr(), pollfds.len() as _, timeout) };
        self.running = false;
        if ret < 0 {
            let e = std::io::Error::last_os_error();
            return Err(crate::PyError::os_error_with_errno(
                e.raw_os_error().unwrap_or(0),
                format!("poll: {e}"),
            ));
        }

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

                // `interp_select.py:as_fdescr` — each item is either an
                // int file descriptor or an object exposing fileno().
                // pyre's list/tuple coverage matches CPython's
                // PySequence_Fast usage; bare iterables (generators)
                // would require iterator-protocol plumbing not yet
                // exposed at this layer.
                fn collect_fds(
                    seq: pyre_object::PyObjectRef,
                ) -> Result<Vec<(pyre_object::PyObjectRef, i32)>, crate::PyError> {
                    unsafe {
                        let is_list = pyre_object::is_list(seq);
                        let is_tuple = pyre_object::is_tuple(seq);
                        if !is_list && !is_tuple {
                            return Err(crate::PyError::type_error(
                                "select() arguments 1-3 must be sequences",
                            ));
                        }
                        let n = if is_list {
                            pyre_object::w_list_len(seq)
                        } else {
                            pyre_object::w_tuple_len(seq)
                        };
                        let mut out = Vec::with_capacity(n);
                        for i in 0..n {
                            let item = if is_list {
                                pyre_object::w_list_getitem(seq, i as i64)
                            } else {
                                pyre_object::w_tuple_getitem(seq, i as i64)
                            }
                            .ok_or_else(|| {
                                crate::PyError::value_error("select() sequence item missing")
                            })?;
                            let fd_val = if pyre_object::is_int(item) {
                                pyre_object::w_int_get_value(item)
                            } else {
                                let fileno =
                                    crate::baseobjspace::getattr(item, "fileno").map_err(|_| {
                                        crate::PyError::type_error(
                                            "argument must be an int, or have a fileno() method",
                                        )
                                    })?;
                                let res = crate::call::call_function_impl_result(fileno, &[])?;
                                if !pyre_object::is_int(res) {
                                    return Err(crate::PyError::type_error(
                                        "fileno() must return an integer",
                                    ));
                                }
                                pyre_object::w_int_get_value(res)
                            };
                            if fd_val < 0 {
                                return Err(crate::PyError::value_error(
                                    "file descriptor cannot be a negative integer",
                                ));
                            }
                            if fd_val > i32::MAX as i64 {
                                return Err(crate::PyError::overflow_error(
                                    "file descriptor out of range",
                                ));
                            }
                            out.push((item, fd_val as i32));
                        }
                        Ok(out)
                    }
                }

                let rfds = collect_fds(args[0])?;
                let wfds = collect_fds(args[1])?;
                let xfds = collect_fds(args[2])?;

                let mut rset = host_select::FdSet::new();
                let mut wset = host_select::FdSet::new();
                let mut xset = host_select::FdSet::new();
                let mut nfds: i32 = -1;
                for &(_, fd) in &rfds {
                    rset.insert(fd);
                    if fd > nfds {
                        nfds = fd;
                    }
                }
                for &(_, fd) in &wfds {
                    wset.insert(fd);
                    if fd > nfds {
                        nfds = fd;
                    }
                }
                for &(_, fd) in &xfds {
                    xset.insert(fd);
                    if fd > nfds {
                        nfds = fd;
                    }
                }

                let mut tv_storage;
                let timeout_ref: Option<&mut host_select::timeval> = match args.get(3) {
                    None => None,
                    Some(&t) if unsafe { pyre_object::is_none(t) } => None,
                    Some(&t) => {
                        let secs = unsafe {
                            if pyre_object::is_float(t) {
                                pyre_object::w_float_get_value(t)
                            } else if pyre_object::is_int(t) {
                                pyre_object::w_int_get_value(t) as f64
                            } else {
                                return Err(crate::PyError::type_error(
                                    "timeout must be a float or None",
                                ));
                            }
                        };
                        if secs < 0.0 {
                            return Err(crate::PyError::value_error(
                                "timeout must be non-negative",
                            ));
                        }
                        tv_storage = host_select::sec_to_timeval(secs);
                        Some(&mut tv_storage)
                    }
                };

                let n = host_select::select(nfds + 1, &mut rset, &mut wset, &mut xset, timeout_ref)
                    .map_err(|e| {
                        crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("select: {e}"),
                        )
                    })?;
                let _ = n;

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
        // valid `ob_type`.
        let _ = type_object();
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
