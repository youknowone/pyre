//! pwd implementation — PyPy: pypy/module/pwd/interp_pwd.py
//!
//! Verbatim move of the inline block previously in importing.rs.

#[cfg(unix)]
thread_local! {
    /// `app_pwd.py:3-19 class struct_passwd(metaclass=structseqtype)`.
    /// Process-wide cached subclass-of-tuple type so every getpwuid /
    /// getpwnam / getpwall result materialises into the same structseq.
    static STRUCT_PASSWD_TYPE: std::cell::OnceCell<pyre_object::PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

#[cfg(unix)]
fn struct_passwd_type() -> pyre_object::PyObjectRef {
    STRUCT_PASSWD_TYPE.with(|c| {
        *c.get_or_init(|| {
            crate::_structseq::make_struct_seq(
                "pwd.struct_passwd",
                &[
                    "pw_name",
                    "pw_passwd",
                    "pw_uid",
                    "pw_gid",
                    "pw_gecos",
                    "pw_dir",
                    "pw_shell",
                ],
            )
        })
    })
}

/// `interp_pwd.py:50-73 uid_converter` — narrow a python int to `uid_t`.
///
/// `-1` is the "current uid" sentinel and passes through unchanged
/// (cast to `uid_t` it becomes the max value, matching the C convention
/// most BSDs use).  Other negative inputs raise OverflowError "user id
/// is less than minimum"; values that don't fit in `uid_t` raise
/// OverflowError "user id is greater than maximum".  Floats / non-int
/// inputs raise TypeError via `int_w`.
#[cfg(unix)]
fn pwd_uid_converter(w_uid: pyre_object::PyObjectRef) -> Result<libc::uid_t, crate::PyError> {
    let val = match crate::baseobjspace::int_w(w_uid) {
        Ok(v) => v,
        Err(e) if matches!(e.kind, crate::PyErrorKind::OverflowError) => {
            return Err(crate::PyError::overflow_error(
                "user id is greater than maximum",
            ));
        }
        Err(e) => return Err(e),
    };
    if val == -1 {
        return Ok((-1i64) as libc::uid_t);
    }
    if val < 0 {
        return Err(crate::PyError::overflow_error(
            "user id is less than minimum",
        ));
    }
    let uid = val as libc::uid_t;
    if uid as i64 != val {
        return Err(crate::PyError::overflow_error(
            "user id is greater than maximum",
        ));
    }
    Ok(uid)
}

/// pwd module — `pypy/module/pwd/interp_pwd.py`.
///
/// getpwuid / getpwnam / getpwall return 7-tuples with the
/// `(pw_name, pw_passwd, pw_uid, pw_gid, pw_gecos, pw_dir, pw_shell)`
/// layout.  `struct_passwd` / `struct_pwent` are exposed as the same
/// builtin type so `isinstance(pwd.struct_passwd, type)` succeeds and
/// `pwd.struct_passwd` is identity-equal to `pwd.struct_pwent`
/// (`app_pwd.py:1-21`).  Full structseq instance materialisation
/// (so `pw_entry.pw_name` returns a string) is a framework prereq
/// tracked separately.
///
/// Backed by `rustpython_host_env::pwd` (a thin `nix` wrapper).
#[cfg(unix)]
pub fn register_module(ns: pyre_object::PyObjectRef) {
    #[cfg(feature = "host_env")]
    fn make_struct_passwd(pw: &rustpython_host_env::pwd::Passwd) -> pyre_object::PyObjectRef {
        crate::_structseq::new_instance(
            struct_passwd_type(),
            vec![
                pyre_object::w_str_new(&pw.name),
                pyre_object::w_str_new(&pw.passwd),
                pyre_object::w_int_new(pw.uid as i64),
                pyre_object::w_int_new(pw.gid as i64),
                pyre_object::w_str_new(&pw.gecos),
                pyre_object::w_str_new(&pw.dir),
                pyre_object::w_str_new(&pw.shell),
            ],
        )
    }
    // `interp_pwd.py:75-87 make_struct_passwd` libc backend, used when
    // the host_env abstraction layer is disabled.  Mirrors the same
    // rffi.charp2str / int construction PyPy uses.
    #[cfg(not(feature = "host_env"))]
    unsafe fn make_struct_passwd_libc(pw: *const libc::passwd) -> pyre_object::PyObjectRef {
        unsafe fn cstr(p: *const libc::c_char) -> String {
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
        crate::_structseq::new_instance(
            struct_passwd_type(),
            vec![
                pyre_object::w_str_new(&cstr((*pw).pw_name)),
                pyre_object::w_str_new(&cstr((*pw).pw_passwd)),
                pyre_object::w_int_new((*pw).pw_uid as i64),
                pyre_object::w_int_new((*pw).pw_gid as i64),
                pyre_object::w_str_new(&cstr((*pw).pw_gecos)),
                pyre_object::w_str_new(&cstr((*pw).pw_dir)),
                pyre_object::w_str_new(&cstr((*pw).pw_shell)),
            ],
        )
    }
    // `app_pwd.py:1-21 class struct_passwd(metaclass=structseqtype)`.
    crate::module_ns_store(ns, "struct_passwd", struct_passwd_type());
    crate::module_ns_store(ns, "struct_pwent", struct_passwd_type());
    crate::module_ns_store(
        ns,
        "getpwuid",
        crate::make_builtin_function_with_arity(
            "getpwuid",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getpwuid() missing argument"));
                }
                // `interp_pwd.py:50-73 uid_converter`: -1 sentinel passes
                // through; negative-other → OverflowError "less than
                // minimum"; positive-too-big → OverflowError "greater
                // than maximum".  `interp_pwd.py:97-100 getpwuid` catches
                // OverflowError and converts it to KeyError "uid not
                // found".
                let uid = match pwd_uid_converter(args[0]) {
                    Ok(u) => u,
                    Err(e) if matches!(e.kind, crate::PyErrorKind::OverflowError) => {
                        return Err(crate::PyError::key_error("getpwuid(): uid not found"));
                    }
                    Err(e) => return Err(e),
                };
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::pwd::getpwuid(uid) {
                        Ok(Some(pw)) => return Ok(make_struct_passwd(&pw)),
                        Ok(None) => {
                            return Err(crate::PyError::key_error(format!(
                                "getpwuid(): uid not found: {}",
                                uid as i64
                            )));
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getpwuid: {e}"),
                            ));
                        }
                    }
                }
                // `interp_pwd.py:90-108` — libc fallback path; host_env
                // is a pyre-only abstraction layer over the same
                // getpwuid() call PyPy makes via rffi.llexternal.
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let pw = libc::getpwuid(uid);
                    if pw.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getpwuid(): uid not found: {}",
                            uid as i64
                        )));
                    }
                    return Ok(make_struct_passwd_libc(pw));
                }
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "getpwnam",
        crate::make_builtin_function_with_arity(
            "getpwnam",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getpwnam() missing argument"));
                }
                if !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "getpwnam(): name should be a string",
                    ));
                }
                let name = unsafe { pyre_object::w_str_get_value(args[0]) };
                // `interp_pwd.py:111 @unwrap_spec(name='text0')` rejects
                // embedded NULs.  CString::new() enforces that here.
                let c_name = std::ffi::CString::new(name).map_err(|_| {
                    crate::PyError::value_error("getpwnam: name must not contain NUL bytes")
                })?;
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::pwd::getpwnam(name) {
                        Some(pw) => return Ok(make_struct_passwd(&pw)),
                        None => {
                            return Err(crate::PyError::key_error(format!(
                                "getpwnam(): name not found: {}",
                                name
                            )));
                        }
                    }
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let pw = libc::getpwnam(c_name.as_ptr());
                    if pw.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getpwnam(): name not found: {}",
                            name
                        )));
                    }
                    return Ok(make_struct_passwd_libc(pw));
                }
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "getpwall",
        crate::make_builtin_function_with_arity(
            "getpwall",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    let items: Vec<pyre_object::PyObjectRef> = rustpython_host_env::pwd::getpwall()
                        .iter()
                        .map(make_struct_passwd)
                        .collect();
                    return Ok(pyre_object::w_list_new(items));
                }
                // `interp_pwd.py:123-134` — setpwent / loop getpwent /
                // endpwent.
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let mut items: Vec<pyre_object::PyObjectRef> = Vec::new();
                    libc::setpwent();
                    loop {
                        let pw = libc::getpwent();
                        if pw.is_null() {
                            break;
                        }
                        items.push(make_struct_passwd_libc(pw));
                    }
                    libc::endpwent();
                    return Ok(pyre_object::w_list_new(items));
                }
            },
            0,
        ),
    );
}
