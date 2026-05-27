//! select implementation — PyPy: pypy/module/select/interp_select.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// _select module — PyPy: pypy/module/select/.
///
/// Implements `select.select(rlist, wlist, xlist, timeout=None)` via
/// `rustpython_host_env::select::{FdSet, select, sec_to_timeval}`.  poll()
/// / epoll / kqueue object types are not implemented yet; they need
/// per-instance heap state which the current pyre builtin-module wiring
/// doesn't expose.
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
