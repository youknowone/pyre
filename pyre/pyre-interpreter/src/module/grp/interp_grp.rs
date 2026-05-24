//! grp implementation — derived from Modules/grpmodule.c (CPython).
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// grp module — `lib_pypy/grp.py` (PyPy keeps it app-level via
/// `_pwdgrp_cffi`).  pyre takes CPython's `Modules/grpmodule.c`
/// shape since pyre has no app-level stdlib.
///
/// getgrgid / getgrnam / getgrall return 4-tuples `(gr_name,
/// gr_passwd, gr_gid, gr_mem)` matching CPython.  `grp.struct_group`
/// is exposed as a builtin type attribute; full structseq instance
/// materialisation (so `entry.gr_name` works) is blocked on the
/// structseq framework task.
#[cfg(unix)]
pub fn register_module(ns: &mut DictStorage) {
    #[cfg(feature = "host_env")]
    fn make_struct_group(g: &rustpython_host_env::grp::Group) -> pyre_object::PyObjectRef {
        let mem_items: Vec<pyre_object::PyObjectRef> =
            g.mem.iter().map(|s| pyre_object::w_str_new(s)).collect();
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&g.name),
            pyre_object::w_str_new(&g.passwd),
            pyre_object::w_int_new(g.gid as i64),
            pyre_object::w_list_new(mem_items),
        ])
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
        let mut mem_items: Vec<pyre_object::PyObjectRef> = Vec::new();
        let mut p = (*g).gr_mem;
        if !p.is_null() {
            while !(*p).is_null() {
                mem_items.push(pyre_object::w_str_new(&cstr(*p)));
                p = p.add(1);
            }
        }
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&cstr((*g).gr_name)),
            pyre_object::w_str_new(&cstr((*g).gr_passwd)),
            pyre_object::w_int_new((*g).gr_gid as i64),
            pyre_object::w_list_new(mem_items),
        ])
    }
    // `lib_pypy/grp.py:13-19 class struct_group` — exposed so
    // `grp.struct_group` is observable on the module even though
    // returned values are still raw tuples.
    let struct_group_type = crate::typedef::make_builtin_type("grp.struct_group", |_| {});
    crate::dict_storage_store(ns, "struct_group", struct_group_type);
    crate::dict_storage_store(
        ns,
        "getgrgid",
        crate::make_builtin_function_with_arity(
            "getgrgid",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getgrgid() missing argument"));
                }
                // `Modules/grpmodule.c grp_getgrgid` — accept any
                // python int (including bigint) via int_w; reject
                // floats as TypeError.  PyPy's `lib_pypy/grp.py`
                // forwards directly through ctypes which would
                // do the same conversion.
                let val = crate::baseobjspace::int_w(args[0])?;
                let gid = val as libc::gid_t;
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::grp::getgrgid(gid) {
                        Ok(Some(g)) => return Ok(make_struct_group(&g)),
                        Ok(None) => {
                            return Err(crate::PyError::key_error(format!(
                                "getgrgid(): gid not found: {}",
                                gid
                            )));
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getgrgid: {e}"),
                            ));
                        }
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
    crate::dict_storage_store(
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
                let name = unsafe { pyre_object::w_str_get_value(args[0]) };
                // Reject embedded NULs (parity with PyPy's @unwrap_spec
                // text0 used for similar lookup APIs).
                let c_name = std::ffi::CString::new(name).map_err(|_| {
                    crate::PyError::value_error("getgrnam: name must not contain NUL bytes")
                })?;
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::grp::getgrnam(name) {
                        Ok(Some(g)) => return Ok(make_struct_group(&g)),
                        Ok(None) => {
                            return Err(crate::PyError::key_error(format!(
                                "getgrnam(): name not found: {}",
                                name
                            )));
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getgrnam: {e}"),
                            ));
                        }
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
    crate::dict_storage_store(
        ns,
        "getgrall",
        crate::make_builtin_function_with_arity(
            "getgrall",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    let items: Vec<pyre_object::PyObjectRef> = rustpython_host_env::grp::getgrall()
                        .iter()
                        .map(make_struct_group)
                        .collect();
                    return Ok(pyre_object::w_list_new(items));
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let mut items: Vec<pyre_object::PyObjectRef> = Vec::new();
                    libc::setgrent();
                    loop {
                        let g = libc::getgrent();
                        if g.is_null() {
                            break;
                        }
                        items.push(make_struct_group_libc(g));
                    }
                    libc::endgrent();
                    return Ok(pyre_object::w_list_new(items));
                }
            },
            0,
        ),
    );
}
