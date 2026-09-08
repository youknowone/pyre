//! grp implementation — `lib_pypy/grp.py`.
//!
//! Verbatim move of the inline block previously in importing.rs.

#[cfg(unix)]
/// `lib_pypy/grp.py:14-20 class struct_group(metaclass=structseqtype)`
/// — process-wide cached subclass-of-tuple type so every getgrgid /
/// getgrnam / getgrall call materialises into the same structseq.
static STRUCT_GROUP_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

#[cfg(unix)]
fn struct_group_type() -> pyre_object::PyObjectRef {
    *STRUCT_GROUP_TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            "grp.struct_group",
            &["gr_name", "gr_passwd", "gr_gid", "gr_mem"],
        ) as usize
    }) as pyre_object::PyObjectRef
}

/// grp module — `lib_pypy/grp.py` (PyPy keeps it app-level via
/// `_pwdgrp_cffi`).  pyre takes CPython's `Modules/grpmodule.c`
/// shape since pyre has no app-level stdlib.
///
/// getgrgid / getgrnam / getgrall return a `grp.struct_group`
/// structseq (subclass of tuple) with named fields `gr_name`,
/// `gr_passwd`, `gr_gid`, `gr_mem` per `lib_pypy/grp.py:14-20`.
#[cfg(unix)]
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    #[cfg(feature = "host_env")]
    fn make_struct_group(g: &rustpython_host_env::grp::Group) -> pyre_object::PyObjectRef {
        // Each `w_str_new_managed` is collectable.  A plain Vec is not a
        // root, so later member/name allocations can sweep earlier strings
        // before `new_instance` pins its argument vector.  Pin each mint
        // as it is produced; close the member bracket before the field
        // bracket (`RootedItems` cannot grow under another open set).
        let mem_list = {
            let mut mem = pyre_object::gc_roots::RootedItems::new();
            for s in &g.mem {
                mem.push(pyre_object::w_str_new_managed(s));
            }
            pyre_object::w_list_new(mem.take())
        };
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_str_new_managed(&g.name));
        fields.push(pyre_object::w_str_new_managed(&g.passwd));
        fields.push(pyre_object::w_int_new(g.gid as i64));
        fields.push(mem_list);
        crate::_structseq::new_instance(struct_group_type(), fields.take())
    }
    // `lib_pypy/grp.py:21-34 _group_from_gstruct` libc backend, used when
    // the host_env abstraction layer is disabled.
    #[cfg(not(feature = "host_env"))]
    unsafe fn make_struct_group_libc(g: *const libc::group) -> pyre_object::PyObjectRef {
        unsafe fn cstr(p: *const libc::c_char) -> String {
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
        let mem_list = {
            let mut mem = pyre_object::gc_roots::RootedItems::new();
            let mut p = (*g).gr_mem;
            if !p.is_null() {
                while !(*p).is_null() {
                    mem.push(pyre_object::w_str_new_managed(&cstr(*p)));
                    p = p.add(1);
                }
            }
            pyre_object::w_list_new(mem.take())
        };
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_str_new_managed(&cstr((*g).gr_name)));
        fields.push(pyre_object::w_str_new_managed(&cstr((*g).gr_passwd)));
        fields.push(pyre_object::w_int_new((*g).gr_gid as i64));
        fields.push(mem_list);
        crate::_structseq::new_instance(struct_group_type(), fields.take())
    }
    // `lib_pypy/grp.py:14-20 class struct_group` — exposed as
    // `grp.struct_group`; every result type uses this same class.
    crate::module_ns_store(ns, "struct_group", struct_group_type());
    crate::module_ns_store(
        ns,
        "getgrgid",
        crate::make_builtin_function_with_arity(
            "getgrgid",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getgrgid() missing argument"));
                }
                // `Modules/grpmodule.c grp_getgrgid` routes through
                // `_Py_Gid_Converter`, which permits `-1` as a sentinel
                // and rejects other out-of-range values rather than
                // silently truncating.  Mirror that here so a Python
                // bigint that doesn't fit in `gid_t` raises OverflowError.
                let val = crate::baseobjspace::int_w(args[0])?;
                let gid_min = libc::gid_t::MIN as i64;
                let gid_max = libc::gid_t::MAX as i64;
                let gid = if val == -1 {
                    libc::gid_t::MAX
                } else if (gid_min..=gid_max).contains(&val) {
                    val as libc::gid_t
                } else {
                    return Err(crate::PyError::overflow_error(
                        "getgrgid: gid is out of range",
                    ));
                };
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::grp::getgrgid(gid) {
                        Ok(Some(g)) => Ok(make_struct_group(&g)),
                        Ok(None) => Err(crate::PyError::key_error(format!(
                            "getgrgid(): gid not found: {}",
                            gid
                        ))),
                        Err(e) => Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("getgrgid: {e}"),
                        )),
                    }
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let g = libc::getgrgid(gid);
                    if g.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getgrgid(): gid not found: {}",
                            gid
                        )));
                    }
                    return Ok(make_struct_group_libc(g));
                }
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "getgrnam",
        crate::make_builtin_function_with_arity(
            "getgrnam",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getgrnam() missing argument"));
                }
                if !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "getgrnam(): name should be a string",
                    ));
                }
                let name = crate::baseobjspace::str_utf8_w(args[0])?;
                // Reject embedded NULs (parity with PyPy's @unwrap_spec
                // text0 used for similar lookup APIs).
                let c_name = std::ffi::CString::new(name).map_err(|_| {
                    crate::PyError::value_error("getgrnam: name must not contain NUL bytes")
                })?;
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::grp::getgrnam(name) {
                        Ok(Some(g)) => Ok(make_struct_group(&g)),
                        Ok(None) => Err(crate::PyError::key_error(format!(
                            "getgrnam(): name not found: {}",
                            name
                        ))),
                        Err(e) => Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("getgrnam: {e}"),
                        )),
                    }
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let g = libc::getgrnam(c_name.as_ptr());
                    if g.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getgrnam(): name not found: {}",
                            name
                        )));
                    }
                    return Ok(make_struct_group_libc(g));
                }
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "getgrall",
        crate::make_builtin_function_with_arity(
            "getgrall",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    // Each struct_group is freshly allocated and building the
                    // next one allocates again, so they are pinned as they
                    // arrive.
                    let groups = rustpython_host_env::grp::getgrall();
                    let mut items = pyre_object::gc_roots::RootedItems::new();
                    for g in groups.iter() {
                        items.push(make_struct_group(g));
                    }
                    Ok(pyre_object::w_list_new(items.take()))
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    // Each struct_group is freshly allocated and the next
                    // `getgrent` entry allocates again, so they are pinned as
                    // they arrive.
                    let mut items = pyre_object::gc_roots::RootedItems::new();
                    libc::setgrent();
                    loop {
                        let g = libc::getgrent();
                        if g.is_null() {
                            break;
                        }
                        items.push(make_struct_group_libc(g));
                    }
                    libc::endgrent();
                    return Ok(pyre_object::w_list_new(items.take()));
                }
            },
            0,
        ),
    );
    Ok(())
}
