//! termios implementation — PyPy: pypy/module/termios/interp_termios.py
//!
//! Verbatim move of the inline block previously in importing.rs.  Both
//! the host_env real impl and the no-host_env stub are renamed to
//! `register_module` so moduledef::init can call a single name.

use crate::DictStorage;

/// `interp_termios.py:13-19 convert_error` — every termios syscall
/// failure is raised as the cached module exception `termios.error`
/// (`wrap_oserror(space, e, w_exception_class=w_error)`), not a bare
/// `OSError`, so `except termios.error` catches it.  Mirrors
/// `_socket`'s `socket_converted_error`: build an instance of the
/// registered `termios.error` class (falling back to `OSError` before
/// the module finishes installing) and stamp it onto the `PyError`.
#[cfg(all(unix, feature = "host_env"))]
fn termios_converted_error(errno: i32, message: impl Into<String>) -> crate::PyError {
    let message = message.into();
    let cls = crate::builtins::lookup_exc_class("termios.error")
        .or_else(|| crate::builtins::lookup_exc_class("OSError"))
        .expect("OSError must be installed");
    let args = vec![
        cls,
        pyre_object::w_int_new(errno as i64),
        pyre_object::w_str_new(&message),
    ];
    let exc = crate::builtins::exc_exception_new(&args)
        .expect("exc_exception_new is infallible for str/int args");
    let mut err = crate::PyError::os_error(message);
    err.exc_object = exc;
    err
}

/// _termios module — PyPy: pypy/module/termios/.
///
/// `tcgetattr(fd)` returns the 7-list `[iflag, oflag, cflag, lflag,
/// ispeed, ospeed, [cc_chars]]`.  `tcsetattr(fd, when, attrs)` takes the
/// same shape and writes it back via `termios::Termios`.  The simpler
/// `tcdrain` / `tcflush` / `tcflow` / `tcsendbreak` / `cfgetispeed` /
/// `cfgetospeed` calls are direct wrappers.  All constants come from
/// `rustpython_host_env::termios::*` so the values match the platform.
#[cfg(all(unix, feature = "host_env"))]
pub fn register_module(ns: &mut DictStorage) {
    use rustpython_host_env::termios as host_termios;

    fn make_cc_bytes(cc: &[libc::cc_t]) -> pyre_object::PyObjectRef {
        // Each cc[i] becomes a 1-byte bytes object (CPython does the same).
        let items: Vec<_> = cc
            .iter()
            .map(|&b| pyre_object::bytesobject::w_bytes_from_bytes(&[b as u8]))
            .collect();
        pyre_object::w_list_new(items)
    }

    crate::dict_storage_store(
        ns,
        "tcgetattr",
        crate::make_builtin_function_with_arity(
            "tcgetattr",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "tcgetattr() requires 1 argument",
                    ));
                }
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                let t = host_termios::tcgetattr(fd).map_err(|e| {
                    termios_converted_error(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcgetattr: {e}"),
                    )
                })?;
                let ispeed = host_termios::cfgetispeed(&t);
                let ospeed = host_termios::cfgetospeed(&t);
                let cc_list = make_cc_bytes(&t.c_cc[..]);
                // interp_termios.py:53 — in noncanonical mode VMIN/VTIME are
                // single-byte counters, surfaced as ints rather than bytes.
                if (t.c_lflag & libc::ICANON) == 0 {
                    let vmin = libc::VMIN as usize;
                    let vtime = libc::VTIME as usize;
                    unsafe {
                        pyre_object::w_list_setitem(
                            cc_list,
                            vmin as i64,
                            pyre_object::w_int_new(t.c_cc[vmin] as i64),
                        );
                        pyre_object::w_list_setitem(
                            cc_list,
                            vtime as i64,
                            pyre_object::w_int_new(t.c_cc[vtime] as i64),
                        );
                    }
                }
                Ok(pyre_object::w_list_new(vec![
                    pyre_object::w_int_new(t.c_iflag as i64),
                    pyre_object::w_int_new(t.c_oflag as i64),
                    pyre_object::w_int_new(t.c_cflag as i64),
                    pyre_object::w_int_new(t.c_lflag as i64),
                    pyre_object::w_int_new(ispeed as i64),
                    pyre_object::w_int_new(ospeed as i64),
                    cc_list,
                ]))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcsetattr",
        crate::make_builtin_function("tcsetattr", |args| {
            if args.len() < 3 {
                return Err(crate::PyError::type_error(
                    "tcsetattr() requires 3 arguments",
                ));
            }
            let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
            if !unsafe { pyre_object::is_int(args[1]) } {
                return Err(crate::PyError::type_error(
                    "tcsetattr: when must be an integer",
                ));
            }
            let when = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
            // interp_termios.py:24-27 — arg 3 must be a 7-element list,
            // unpacked via space.unpackiterable.
            let attrs = args[2];
            if !unsafe { pyre_object::is_list(attrs) }
                || unsafe { pyre_object::w_list_len(attrs) } != 7
            {
                return Err(crate::PyError::type_error(
                    "tcsetattr, arg 3: must be 7 element list",
                ));
            }
            let fields = crate::baseobjspace::unpackiterable(attrs, 7)?;
            let iflag = crate::baseobjspace::int_w(fields[0])? as libc::tcflag_t;
            let oflag = crate::baseobjspace::int_w(fields[1])? as libc::tcflag_t;
            let cflag = crate::baseobjspace::int_w(fields[2])? as libc::tcflag_t;
            let lflag = crate::baseobjspace::int_w(fields[3])? as libc::tcflag_t;
            let ispeed = crate::baseobjspace::int_w(fields[4])? as libc::speed_t;
            let ospeed = crate::baseobjspace::int_w(fields[5])? as libc::speed_t;
            let cc_obj = fields[6];

            // Start from the current settings so we preserve any platform-private fields.
            let mut t = host_termios::tcgetattr(fd).map_err(|e| {
                termios_converted_error(
                    e.raw_os_error().unwrap_or(0),
                    format!("tcsetattr: {e}"),
                )
            })?;
            t.c_iflag = iflag;
            t.c_oflag = oflag;
            t.c_cflag = cflag;
            t.c_lflag = lflag;
            host_termios::cfsetispeed(&mut t, ispeed).map_err(|e| {
                termios_converted_error(
                    e.raw_os_error().unwrap_or(0),
                    format!("cfsetispeed: {e}"),
                )
            })?;
            host_termios::cfsetospeed(&mut t, ospeed).map_err(|e| {
                termios_converted_error(
                    e.raw_os_error().unwrap_or(0),
                    format!("cfsetospeed: {e}"),
                )
            })?;

            // interp_termios.py:30-33 — c_cc is any iterable; an int element
            // goes through bytes([x]) (range 0..=255), a bytes element keeps
            // its first byte.
            let cc_items = crate::baseobjspace::unpackiterable(cc_obj, -1)?;
            let nccs = t.c_cc.len();
            for (i, &item) in cc_items.iter().enumerate() {
                if i >= nccs {
                    break;
                }
                let byte = unsafe {
                    if pyre_object::is_int(item) {
                        let v = pyre_object::w_int_get_value(item);
                        if !(0..=255).contains(&v) {
                            return Err(crate::PyError::value_error(
                                "bytes must be in range(0, 256)",
                            ));
                        }
                        v as libc::cc_t
                    } else if pyre_object::bytesobject::is_bytes_like(item) {
                        let data = pyre_object::bytesobject::bytes_like_data(item);
                        if data.is_empty() {
                            0
                        } else {
                            data[0] as libc::cc_t
                        }
                    } else {
                        return Err(crate::PyError::type_error(
                            "tcsetattr: c_cc element must be int or bytes",
                        ));
                    }
                };
                t.c_cc[i] = byte;
            }
            host_termios::tcsetattr(fd, when, &t).map_err(|e| {
                termios_converted_error(
                    e.raw_os_error().unwrap_or(0),
                    format!("tcsetattr: {e}"),
                )
            })?;
            Ok(pyre_object::w_none())
        }),
    );

    crate::dict_storage_store(
        ns,
        "tcsendbreak",
        crate::make_builtin_function_with_arity(
            "tcsendbreak",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "tcsendbreak() requires 2 arguments",
                    ));
                }
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(crate::PyError::type_error(
                        "tcsendbreak: duration must be an integer",
                    ));
                }
                let dur = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                host_termios::tcsendbreak(fd, dur).map_err(|e| {
                    termios_converted_error(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcsendbreak: {e}"),
                    )
                })?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcdrain",
        crate::make_builtin_function_with_arity(
            "tcdrain",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("tcdrain() requires 1 argument"));
                }
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                host_termios::tcdrain(fd).map_err(|e| {
                    termios_converted_error(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcdrain: {e}"),
                    )
                })?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcflush",
        crate::make_builtin_function_with_arity(
            "tcflush",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("tcflush() requires 2 arguments"));
                }
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(crate::PyError::type_error(
                        "tcflush: queue must be an integer",
                    ));
                }
                let q = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                host_termios::tcflush(fd, q).map_err(|e| {
                    termios_converted_error(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcflush: {e}"),
                    )
                })?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcflow",
        crate::make_builtin_function_with_arity(
            "tcflow",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("tcflow() requires 2 arguments"));
                }
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(crate::PyError::type_error(
                        "tcflow: action must be an integer",
                    ));
                }
                let action = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                host_termios::tcflow(fd, action).map_err(|e| {
                    termios_converted_error(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcflow: {e}"),
                    )
                })?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcgetwinsize",
        crate::make_builtin_function_with_arity(
            "tcgetwinsize",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "tcgetwinsize() requires 1 argument",
                    ));
                }
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                let ret = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) };
                if ret != 0 {
                    let e = std::io::Error::last_os_error();
                    return Err(termios_converted_error(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcgetwinsize: {e}"),
                    ));
                }
                // `interp_termios.py:99-101` returns `(ws_row, ws_col)`.
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(ws.ws_row as i64),
                    pyre_object::w_int_new(ws.ws_col as i64),
                ]))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcsetwinsize",
        crate::make_builtin_function_with_arity(
            "tcsetwinsize",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "tcsetwinsize() requires 2 arguments",
                    ));
                }
                let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                // `interp_termios.py:110-114` — argument 2 must be a
                // 2-sequence (any iterable); a length mismatch (ValueError
                // from unpackiterable) is reported as a TypeError.
                let winsz = crate::baseobjspace::unpackiterable(args[1], 2).map_err(|e| {
                    if e.kind == crate::PyErrorKind::ValueError {
                        crate::PyError::type_error(
                            "tcsetwinsize: argument 2 must be a 2-sequence",
                        )
                    } else {
                        e
                    }
                })?;
                let rows = crate::baseobjspace::int_w(winsz[0])?;
                let cols = crate::baseobjspace::int_w(winsz[1])?;
                let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                // `interp_termios.py:120` reads the current winsize first so
                // `ws_xpixel` / `ws_ypixel` are preserved across the set.
                let ret = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) };
                if ret != 0 {
                    let e = std::io::Error::last_os_error();
                    return Err(termios_converted_error(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcsetwinsize: {e}"),
                    ));
                }
                ws.ws_row = rows as libc::c_ushort;
                ws.ws_col = cols as libc::c_ushort;
                // `interp_termios.py:126-128` overflow guard.
                if ws.ws_row as i64 != rows || ws.ws_col as i64 != cols {
                    return Err(crate::PyError::overflow_error(
                        "winsize value(s) out of range",
                    ));
                }
                let ret = unsafe { libc::ioctl(fd, libc::TIOCSWINSZ, &mut ws) };
                if ret != 0 {
                    let e = std::io::Error::last_os_error();
                    return Err(termios_converted_error(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcsetwinsize: {e}"),
                    ));
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    // ── Constants ──
    crate::dict_storage_store(ns, "B0", pyre_object::w_int_new(host_termios::B0 as i64));
    crate::dict_storage_store(ns, "B50", pyre_object::w_int_new(host_termios::B50 as i64));
    crate::dict_storage_store(ns, "B75", pyre_object::w_int_new(host_termios::B75 as i64));
    crate::dict_storage_store(
        ns,
        "B110",
        pyre_object::w_int_new(host_termios::B110 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B134",
        pyre_object::w_int_new(host_termios::B134 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B150",
        pyre_object::w_int_new(host_termios::B150 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B200",
        pyre_object::w_int_new(host_termios::B200 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B300",
        pyre_object::w_int_new(host_termios::B300 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B600",
        pyre_object::w_int_new(host_termios::B600 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B1200",
        pyre_object::w_int_new(host_termios::B1200 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B1800",
        pyre_object::w_int_new(host_termios::B1800 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B2400",
        pyre_object::w_int_new(host_termios::B2400 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B4800",
        pyre_object::w_int_new(host_termios::B4800 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B9600",
        pyre_object::w_int_new(host_termios::B9600 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B19200",
        pyre_object::w_int_new(host_termios::B19200 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B38400",
        pyre_object::w_int_new(host_termios::B38400 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B57600",
        pyre_object::w_int_new(host_termios::B57600 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B115200",
        pyre_object::w_int_new(host_termios::B115200 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B230400",
        pyre_object::w_int_new(host_termios::B230400 as i64),
    );

    crate::dict_storage_store(
        ns,
        "BRKINT",
        pyre_object::w_int_new(host_termios::BRKINT as i64),
    );
    crate::dict_storage_store(
        ns,
        "CLOCAL",
        pyre_object::w_int_new(host_termios::CLOCAL as i64),
    );
    crate::dict_storage_store(
        ns,
        "CREAD",
        pyre_object::w_int_new(host_termios::CREAD as i64),
    );
    crate::dict_storage_store(ns, "CS5", pyre_object::w_int_new(host_termios::CS5 as i64));
    crate::dict_storage_store(ns, "CS6", pyre_object::w_int_new(host_termios::CS6 as i64));
    crate::dict_storage_store(ns, "CS7", pyre_object::w_int_new(host_termios::CS7 as i64));
    crate::dict_storage_store(ns, "CS8", pyre_object::w_int_new(host_termios::CS8 as i64));
    crate::dict_storage_store(
        ns,
        "CSIZE",
        pyre_object::w_int_new(host_termios::CSIZE as i64),
    );
    crate::dict_storage_store(
        ns,
        "CSTOPB",
        pyre_object::w_int_new(host_termios::CSTOPB as i64),
    );
    crate::dict_storage_store(
        ns,
        "ECHO",
        pyre_object::w_int_new(host_termios::ECHO as i64),
    );
    crate::dict_storage_store(
        ns,
        "ECHOE",
        pyre_object::w_int_new(host_termios::ECHOE as i64),
    );
    crate::dict_storage_store(
        ns,
        "ECHOK",
        pyre_object::w_int_new(host_termios::ECHOK as i64),
    );
    crate::dict_storage_store(
        ns,
        "ECHONL",
        pyre_object::w_int_new(host_termios::ECHONL as i64),
    );
    crate::dict_storage_store(
        ns,
        "HUPCL",
        pyre_object::w_int_new(host_termios::HUPCL as i64),
    );
    crate::dict_storage_store(
        ns,
        "ICANON",
        pyre_object::w_int_new(host_termios::ICANON as i64),
    );
    crate::dict_storage_store(
        ns,
        "ICRNL",
        pyre_object::w_int_new(host_termios::ICRNL as i64),
    );
    crate::dict_storage_store(
        ns,
        "IEXTEN",
        pyre_object::w_int_new(host_termios::IEXTEN as i64),
    );
    crate::dict_storage_store(
        ns,
        "IGNBRK",
        pyre_object::w_int_new(host_termios::IGNBRK as i64),
    );
    crate::dict_storage_store(
        ns,
        "IGNCR",
        pyre_object::w_int_new(host_termios::IGNCR as i64),
    );
    crate::dict_storage_store(
        ns,
        "IGNPAR",
        pyre_object::w_int_new(host_termios::IGNPAR as i64),
    );
    crate::dict_storage_store(
        ns,
        "INLCR",
        pyre_object::w_int_new(host_termios::INLCR as i64),
    );
    crate::dict_storage_store(
        ns,
        "INPCK",
        pyre_object::w_int_new(host_termios::INPCK as i64),
    );
    crate::dict_storage_store(
        ns,
        "ISIG",
        pyre_object::w_int_new(host_termios::ISIG as i64),
    );
    crate::dict_storage_store(
        ns,
        "ISTRIP",
        pyre_object::w_int_new(host_termios::ISTRIP as i64),
    );
    crate::dict_storage_store(
        ns,
        "IXANY",
        pyre_object::w_int_new(host_termios::IXANY as i64),
    );
    crate::dict_storage_store(
        ns,
        "IXOFF",
        pyre_object::w_int_new(host_termios::IXOFF as i64),
    );
    crate::dict_storage_store(
        ns,
        "IXON",
        pyre_object::w_int_new(host_termios::IXON as i64),
    );
    crate::dict_storage_store(
        ns,
        "NOFLSH",
        pyre_object::w_int_new(host_termios::NOFLSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "OCRNL",
        pyre_object::w_int_new(host_termios::OCRNL as i64),
    );
    crate::dict_storage_store(
        ns,
        "ONLCR",
        pyre_object::w_int_new(host_termios::ONLCR as i64),
    );
    crate::dict_storage_store(
        ns,
        "ONLRET",
        pyre_object::w_int_new(host_termios::ONLRET as i64),
    );
    crate::dict_storage_store(
        ns,
        "ONOCR",
        pyre_object::w_int_new(host_termios::ONOCR as i64),
    );
    crate::dict_storage_store(
        ns,
        "OPOST",
        pyre_object::w_int_new(host_termios::OPOST as i64),
    );
    crate::dict_storage_store(
        ns,
        "PARENB",
        pyre_object::w_int_new(host_termios::PARENB as i64),
    );
    crate::dict_storage_store(
        ns,
        "PARMRK",
        pyre_object::w_int_new(host_termios::PARMRK as i64),
    );
    crate::dict_storage_store(
        ns,
        "PARODD",
        pyre_object::w_int_new(host_termios::PARODD as i64),
    );

    crate::dict_storage_store(
        ns,
        "TCIFLUSH",
        pyre_object::w_int_new(host_termios::TCIFLUSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCOFLUSH",
        pyre_object::w_int_new(host_termios::TCOFLUSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCIOFLUSH",
        pyre_object::w_int_new(host_termios::TCIOFLUSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCIOFF",
        pyre_object::w_int_new(host_termios::TCIOFF as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCION",
        pyre_object::w_int_new(host_termios::TCION as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCOOFF",
        pyre_object::w_int_new(host_termios::TCOOFF as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCOON",
        pyre_object::w_int_new(host_termios::TCOON as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCSANOW",
        pyre_object::w_int_new(host_termios::TCSANOW as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCSADRAIN",
        pyre_object::w_int_new(host_termios::TCSADRAIN as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCSAFLUSH",
        pyre_object::w_int_new(host_termios::TCSAFLUSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "TOSTOP",
        pyre_object::w_int_new(host_termios::TOSTOP as i64),
    );

    crate::dict_storage_store(
        ns,
        "VEOF",
        pyre_object::w_int_new(host_termios::VEOF as i64),
    );
    crate::dict_storage_store(
        ns,
        "VEOL",
        pyre_object::w_int_new(host_termios::VEOL as i64),
    );
    crate::dict_storage_store(
        ns,
        "VERASE",
        pyre_object::w_int_new(host_termios::VERASE as i64),
    );
    crate::dict_storage_store(
        ns,
        "VINTR",
        pyre_object::w_int_new(host_termios::VINTR as i64),
    );
    crate::dict_storage_store(
        ns,
        "VKILL",
        pyre_object::w_int_new(host_termios::VKILL as i64),
    );
    crate::dict_storage_store(
        ns,
        "VMIN",
        pyre_object::w_int_new(host_termios::VMIN as i64),
    );
    crate::dict_storage_store(
        ns,
        "VQUIT",
        pyre_object::w_int_new(host_termios::VQUIT as i64),
    );
    crate::dict_storage_store(
        ns,
        "VSTART",
        pyre_object::w_int_new(host_termios::VSTART as i64),
    );
    crate::dict_storage_store(
        ns,
        "VSTOP",
        pyre_object::w_int_new(host_termios::VSTOP as i64),
    );
    crate::dict_storage_store(
        ns,
        "VSUSP",
        pyre_object::w_int_new(host_termios::VSUSP as i64),
    );
    crate::dict_storage_store(
        ns,
        "VTIME",
        pyre_object::w_int_new(host_termios::VTIME as i64),
    );

    // `interp_termios.py:18 class W_TermiosError(OperationError)` —
    // wraps OSError so `except termios.error` catches tcsetattr failures.
    let w_os_error = crate::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before termios init");
    let w_error = crate::builtins::make_exc_type(
        "termios.error",
        crate::builtins::exc_exception_new,
        w_os_error,
    );
    crate::dict_storage_store(ns, "error", w_error);
}

#[cfg(not(all(unix, feature = "host_env")))]
pub fn register_module(_ns: &mut DictStorage) {}
