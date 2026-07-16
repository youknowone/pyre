//! `_ctypes` — the native surface the CPython `ctypes` package sits on.
//!
//! On unix with the `host_env` feature this provides a working end-to-end
//! slice: the dynamic-linker primitives (`dlopen`/`dlsym`/`dlclose`), the
//! scalar data type (`_SimpleCData`, see [`super::cdata`]), the foreign
//! function object (`CFuncPtr`, see [`super::funcptr`]), `sizeof`/`addressof`/
//! `byref`/`alignment`/`resize`, and the import-time constants the package
//! requires.  `Structure`/`Union`/`Array`/`_Pointer`/`CField` are real
//! (see [`super::metaclass`]); the metaclasses compute each type's layout and
//! the buffer-view machinery aliases nested/pointed-to memory.
//!
//! All host/FFI work is delegated to `rustpython_host_env::ctypes`; the module
//! contains no direct `libc::` FFI.


pub fn register_module(ns: pyre_object::PyObjectRef) {
    #[cfg(all(unix, feature = "host_env"))]
    register_host_ctypes(ns);
    #[cfg(not(all(unix, feature = "host_env")))]
    register_stub_ctypes(ns);
}

// ──────────────────────────────────────────────────────────────────────
// Functional surface (unix + host_env)
// ──────────────────────────────────────────────────────────────────────

#[cfg(all(unix, feature = "host_env"))]
fn register_host_ctypes(ns: pyre_object::PyObjectRef) {
    use rustpython_host_env::ctypes as host_ctypes;

    // ── dlopen flags (POSIX) ──
    crate::module_ns_store(
        ns,
        "RTLD_LOCAL",
        pyre_object::w_int_new(libc::RTLD_LOCAL as i64),
    );
    crate::module_ns_store(
        ns,
        "RTLD_GLOBAL",
        pyre_object::w_int_new(libc::RTLD_GLOBAL as i64),
    );
    crate::module_ns_store(
        ns,
        "RTLD_LAZY",
        pyre_object::w_int_new(libc::RTLD_LAZY as i64),
    );
    crate::module_ns_store(
        ns,
        "RTLD_NOW",
        pyre_object::w_int_new(libc::RTLD_NOW as i64),
    );
    crate::module_ns_store(
        ns,
        "DEFAULT_MODE",
        pyre_object::w_int_new(host_ctypes::dlopen_mode(None) as i64),
    );

    // ── dlopen(name, mode=DEFAULT_MODE) → integer handle into host libcache ──
    crate::module_ns_store(
        ns,
        "dlopen",
        crate::make_builtin_function("dlopen", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("dlopen() missing library name"));
            }
            let name = unsafe {
                if pyre_object::is_none(args[0]) {
                    // dlopen(None) → process handle
                    let mode = if args.len() >= 2 {
                        pyre_object::w_int_get_value(args[1]) as libc::c_int
                    } else {
                        libc::RTLD_NOW
                    };
                    let ptr = rustpython_host_env::ctypes::dlopen_self(mode)
                        .map_err(|e| crate::PyError::os_error(format!("dlopen(None): {e}")))?;
                    let h = rustpython_host_env::ctypes::insert_raw_library_handle(ptr);
                    return Ok(pyre_object::w_int_new(h as i64));
                }
                if !pyre_object::is_str(args[0]) {
                    return Err(crate::PyError::type_error(
                        "dlopen: name must be a string or None",
                    ));
                }
                pyre_object::w_str_get_value(args[0]).to_string()
            };
            let mode = if args.len() >= 2 {
                (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32
            } else {
                rustpython_host_env::ctypes::dlopen_mode(None)
            };
            let h = rustpython_host_env::ctypes::open_library_with_mode(&name, mode)
                .map_err(|e| crate::PyError::os_error(format!("dlopen({name}): {e}")))?;
            Ok(pyre_object::w_int_new(h as i64))
        }),
    );

    // ── dlsym(handle, name) → address (int) ──
    crate::module_ns_store(
        ns,
        "dlsym",
        crate::make_builtin_function_with_arity(
            "dlsym",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("dlsym() needs 2 arguments"));
                }
                let h = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                let name = unsafe {
                    if !pyre_object::is_str(args[1]) {
                        return Err(crate::PyError::type_error("dlsym: name must be a string"));
                    }
                    pyre_object::w_str_get_value(args[1]).to_string()
                };
                let addr =
                    rustpython_host_env::ctypes::lookup_function_symbol_addr(h, name.as_bytes())
                        .map_err(|e| {
                            use rustpython_host_env::ctypes::LookupSymbolError as L;
                            let msg = match e {
                                L::LibraryNotFound => "library not found".to_string(),
                                L::LibraryClosed => "library closed".to_string(),
                                L::Load(s) => s,
                            };
                            crate::PyError::os_error(format!("dlsym({name}): {msg}"))
                        })?;
                Ok(pyre_object::w_int_new(addr as i64))
            },
            2,
        ),
    );

    // ── dlclose(handle) → None ──
    crate::module_ns_store(
        ns,
        "dlclose",
        crate::make_builtin_function_with_arity(
            "dlclose",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("dlclose() needs handle"));
                }
                let h = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                rustpython_host_env::ctypes::drop_library(h);
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── get_errno / set_errno — routed through host_env's ctypes-local errno ──
    crate::module_ns_store(
        ns,
        "get_errno",
        crate::make_builtin_function_with_arity(
            "get_errno",
            |_| {
                Ok(pyre_object::w_int_new(
                    rustpython_host_env::ctypes::get_errno() as i64,
                ))
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "set_errno",
        crate::make_builtin_function_with_arity(
            "set_errno",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("set_errno() needs value"));
                }
                let v = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let prev = rustpython_host_env::ctypes::set_errno(v);
                Ok(pyre_object::w_int_new(prev as i64))
            },
            1,
        ),
    );

    // ── sizeof / alignment of raw type codes ('i', 'l', 'd', …) ──
    crate::module_ns_store(
        ns,
        "_sizeof_typecode",
        crate::make_builtin_function_with_arity(
            "_sizeof_typecode",
            |args| {
                if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "_sizeof_typecode() needs typecode string",
                    ));
                }
                let code = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                match rustpython_host_env::ctypes::simple_type_size(&code) {
                    Some(n) => Ok(pyre_object::w_int_new(n as i64)),
                    None => Err(crate::PyError::value_error(format!(
                        "unknown type code: {code}"
                    ))),
                }
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "_alignof_typecode",
        crate::make_builtin_function_with_arity(
            "_alignof_typecode",
            |args| {
                if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "_alignof_typecode() needs typecode string",
                    ));
                }
                let code = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                match rustpython_host_env::ctypes::simple_type_align(&code) {
                    Some(n) => Ok(pyre_object::w_int_new(n as i64)),
                    None => Err(crate::PyError::value_error(format!(
                        "unknown type code: {code}"
                    ))),
                }
            },
            1,
        ),
    );

    // ── string_at(ptr, size=-1) → bytes ──
    crate::module_ns_store(
        ns,
        "string_at",
        crate::make_builtin_function("string_at", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("string_at() needs ptr"));
            }
            let ptr = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
            let size = if args.len() >= 2 {
                unsafe { pyre_object::w_int_get_value(args[1]) }
            } else {
                -1
            };
            let bytes =
                rustpython_host_env::ctypes::string_at(ptr, size as isize).map_err(|e| {
                    use rustpython_host_env::ctypes::StringAtError as S;
                    let msg = match e {
                        S::NullPointer => "NULL pointer access",
                        S::TooLong => "size too large",
                    };
                    crate::PyError::os_error(format!("string_at: {msg}"))
                })?;
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
        }),
    );

    crate::module_ns_store(
        ns,
        "SIZEOF_TIME_T",
        pyre_object::w_int_new(rustpython_host_env::ctypes::SIZEOF_TIME_T as i64),
    );

    // ── import-time constants ──
    crate::module_ns_store(ns, "__version__", pyre_object::w_str_new("1.1.0"));
    // `crates/vm/src/stdlib/_ctypes.rs`: CDECL=0x1, PYTHONAPI=0x4,
    // USE_ERRNO=0x8, USE_LASTERROR=0x10.
    crate::module_ns_store(ns, "FUNCFLAG_CDECL", pyre_object::w_int_new(0x1));
    crate::module_ns_store(ns, "FUNCFLAG_PYTHONAPI", pyre_object::w_int_new(0x4));
    crate::module_ns_store(ns, "FUNCFLAG_USE_ERRNO", pyre_object::w_int_new(0x8));
    crate::module_ns_store(ns, "FUNCFLAG_USE_LASTERROR", pyre_object::w_int_new(0x10));

    // Real addresses so `memmove`/`memset = CFUNCTYPE(...)(_memmove_addr)`
    // build callable foreign functions; the other three are import-only
    // sentinels (cast/string_at/memoryview PYFUNCTYPE targets are not
    // exercised by this slice).
    crate::module_ns_store(
        ns,
        "_memmove_addr",
        pyre_object::w_int_new(host_ctypes::memmove_addr() as i64),
    );
    crate::module_ns_store(
        ns,
        "_memset_addr",
        pyre_object::w_int_new(host_ctypes::memset_addr() as i64),
    );
    crate::module_ns_store(ns, "_cast_addr", pyre_object::w_int_new(1));
    crate::module_ns_store(ns, "_string_at_addr", pyre_object::w_int_new(2));
    crate::module_ns_store(ns, "_memoryview_at_addr", pyre_object::w_int_new(4));

    // ── ArgumentError — a real Exception subclass ──
    let w_exception = crate::builtins::lookup_exc_class("Exception")
        .expect("Exception must be installed before _ctypes init");
    crate::module_ns_store(
        ns,
        "ArgumentError",
        crate::builtins::make_exc_type(
            "ArgumentError",
            crate::builtins::exc_exception_new,
            w_exception,
        ),
    );

    // ── aggregate + array/pointer types: real `Structure`/`Union`/`Array`/
    //    `_Pointer`/`CField` ──
    use super::metaclass;
    let structure_tp = metaclass::structure_type();
    let union_tp = metaclass::union_type();
    let array_tp = metaclass::array_type();
    let pointer_tp = metaclass::pointer_base_type();
    crate::module_ns_store(ns, "Structure", structure_tp);
    crate::module_ns_store(ns, "Union", union_tp);
    crate::module_ns_store(ns, "Array", array_tp);
    crate::module_ns_store(ns, "_Pointer", pointer_tp);
    crate::module_ns_store(ns, "CField", metaclass::cfield_type());

    // ── the functional scalar + foreign-function types ──
    let simplecdata_tp = super::cdata::simplecdata_type();
    // `_SimpleCData`'s metaclass routes `class c_int(_SimpleCData): _type_="i"`
    // through `PyCSimpleType.__new__` (validation + StgInfo).
    unsafe { (*simplecdata_tp).w_class = metaclass::pycsimpletype_type() };
    crate::module_ns_store(ns, "_SimpleCData", simplecdata_tp);
    crate::module_ns_store(ns, "CFuncPtr", super::funcptr::cfuncptr_type());

    // Widen `is_cdata_instance` (sizeof/addressof/byref) to every CData base.
    for base in [simplecdata_tp, structure_tp, union_tp, array_tp, pointer_tp] {
        super::cdata::register_cdata_base(base);
    }

    // ── sizeof / addressof / byref / alignment / resize ──
    crate::module_ns_store(
        ns,
        "sizeof",
        crate::make_builtin_function("sizeof", ctypes_sizeof),
    );
    crate::module_ns_store(
        ns,
        "alignment",
        crate::make_builtin_function("alignment", ctypes_alignment),
    );
    crate::module_ns_store(
        ns,
        "addressof",
        crate::make_builtin_function("addressof", ctypes_addressof),
    );
    crate::module_ns_store(
        ns,
        "byref",
        crate::make_builtin_function("byref", ctypes_byref),
    );
    crate::module_ns_store(
        ns,
        "resize",
        crate::make_builtin_function("resize", ctypes_resize),
    );
}

// ── sizeof / alignment (types and instances) ──────────────────────────

#[cfg(all(unix, feature = "host_env"))]
fn ctypes_sizeof(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    use super::{cdata, stginfo};
    use rustpython_host_env::ctypes as host_ctypes;
    let obj = *args
        .first()
        .ok_or_else(|| crate::PyError::type_error("sizeof() missing argument"))?;
    if unsafe { pyre_object::is_type(obj) } {
        if let Some(info) = stginfo::stginfo_of(obj) {
            return Ok(pyre_object::w_int_new(stginfo::stginfo_size(info) as i64));
        }
        // Simple type without a StgInfo yet: derive from `_type_`.
        return match cdata::type_code_of(obj) {
            Some(tc) => match host_ctypes::simple_type_size(&tc) {
                Some(sz) => Ok(pyre_object::w_int_new(sz as i64)),
                None => Err(cdata::invalid_type_code_error()),
            },
            None => Err(crate::PyError::type_error("this type has no size")),
        };
    }
    if cdata::is_cdata_instance(obj) {
        return Ok(pyre_object::w_int_new(
            cdata::cdata_len(obj).unwrap_or(0) as i64
        ));
    }
    Err(crate::PyError::type_error("this type has no size"))
}

#[cfg(all(unix, feature = "host_env"))]
fn ctypes_alignment(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    use super::{cdata, stginfo};
    use rustpython_host_env::ctypes as host_ctypes;
    let obj = *args
        .first()
        .ok_or_else(|| crate::PyError::type_error("alignment() missing argument"))?;
    let target = if unsafe { pyre_object::is_type(obj) } {
        obj
    } else if cdata::is_cdata_instance(obj) {
        unsafe { pyre_object::w_instance_get_type(obj) }
    } else {
        return Err(crate::PyError::type_error("no alignment info"));
    };
    if let Some(info) = stginfo::stginfo_of(target) {
        return Ok(pyre_object::w_int_new(stginfo::stginfo_align(info) as i64));
    }
    match cdata::type_code_of(target) {
        Some(tc) => match host_ctypes::simple_type_align(&tc) {
            Some(a) => Ok(pyre_object::w_int_new(a as i64)),
            None => Err(cdata::invalid_type_code_error()),
        },
        None => Err(crate::PyError::type_error("no alignment info")),
    }
}

// ── addressof / byref / resize (instances) ────────────────────────────

#[cfg(all(unix, feature = "host_env"))]
fn ctypes_addressof(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    use super::cdata;
    let obj = *args
        .first()
        .ok_or_else(|| crate::PyError::type_error("addressof() missing argument"))?;
    if !cdata::is_cdata_instance(obj) {
        return Err(crate::PyError::type_error("invalid type"));
    }
    let addr = cdata::cdata_addr(obj)
        .ok_or_else(|| crate::PyError::type_error("instance has no buffer"))?;
    Ok(pyre_object::w_int_new(addr as i64))
}

#[cfg(all(unix, feature = "host_env"))]
fn ctypes_byref(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    use super::cdata;
    use rustpython_host_env::ctypes as host_ctypes;
    let obj = *args
        .first()
        .ok_or_else(|| crate::PyError::type_error("byref() missing argument"))?;
    if !cdata::is_cdata_instance(obj) {
        return Err(crate::PyError::type_error(
            "byref() argument must be a ctypes instance",
        ));
    }
    let offset = if args.len() >= 2 {
        crate::baseobjspace::int_w(args[1])? as isize
    } else {
        0
    };
    let base = cdata::cdata_addr(obj)
        .ok_or_else(|| crate::PyError::type_error("instance has no buffer"))?;
    let addr = host_ctypes::offset_address(base, offset);
    Ok(make_carg(addr, obj))
}

#[cfg(all(unix, feature = "host_env"))]
fn ctypes_resize(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    use super::cdata;
    if args.len() < 2 {
        return Err(crate::PyError::type_error("resize() needs (obj, size)"));
    }
    let obj = args[0];
    if !cdata::is_cdata_instance(obj) {
        return Err(crate::PyError::type_error("excepted ctypes instance"));
    }
    if !cdata::owns_buffer(obj) {
        return Err(crate::PyError::value_error(
            "Memory cannot be resized because this object doesn't own it",
        ));
    }
    let size = crate::baseobjspace::int_w(args[1])? as usize;
    let cur = cdata::cdata_len(obj).unwrap_or(0);
    if size < cur {
        return Err(crate::PyError::value_error(format!(
            "minimum size is {cur}"
        )));
    }
    if size > cur {
        if let Some(ba) = cdata::cdata_buffer(obj) {
            unsafe { pyre_object::w_bytearray_vec_mut(ba).resize(size, 0) };
        }
    }
    Ok(pyre_object::w_none())
}

// ── byref carrier ──────────────────────────────────────────────────────

#[cfg(all(unix, feature = "host_env"))]
thread_local! {
    static CARG_TYPE_OBJ: std::cell::OnceCell<pyre_object::PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

/// The minimal `byref` carrier type — holds `_ptr` (address) and `_obj`
/// (the referenced instance, kept alive).  Foreign-call consumption of the
/// carrier (the CArgObject P-tag path) is a later slice.
#[cfg(all(unix, feature = "host_env"))]
fn carg_type() -> pyre_object::PyObjectRef {
    CARG_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("CArgObject", |_| {});
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

#[cfg(all(unix, feature = "host_env"))]
fn make_carg(addr: usize, obj: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    let carg = pyre_object::w_instance_new(carg_type());
    let d = crate::baseobjspace::getdict(carg);
    if !d.is_null() {
        unsafe {
            pyre_object::w_dict_setitem_str(d, "_ptr", pyre_object::w_int_new(addr as i64));
            pyre_object::w_dict_setitem_str(d, "_obj", obj);
        }
    }
    carg
}

/// Whether `obj` is a `byref()` carrier (consumed by [`super::funcptr`]).
#[cfg(all(unix, feature = "host_env"))]
pub(super) fn is_carg(obj: pyre_object::PyObjectRef) -> bool {
    !obj.is_null() && unsafe { pyre_object::w_instance_get_type(obj) } == carg_type()
}

/// The address a `byref()` carrier points at.
#[cfg(all(unix, feature = "host_env"))]
pub(super) fn carg_ptr(carg: pyre_object::PyObjectRef) -> usize {
    let d = crate::baseobjspace::getdict(carg);
    if d.is_null() {
        return 0;
    }
    match unsafe { pyre_object::w_dict_getitem_str(d, "_ptr") } {
        Some(o) if unsafe { pyre_object::is_int(o) } => {
            (unsafe { pyre_object::w_int_get_value(o) }) as usize
        }
        _ => 0,
    }
}

// ──────────────────────────────────────────────────────────────────────
// Stub surface (non-unix or no host_env) — keeps names importable.
// ──────────────────────────────────────────────────────────────────────

#[cfg(not(all(unix, feature = "host_env")))]
fn register_stub_ctypes(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(ns, "ArgumentError", crate::typedef::w_object());
    crate::module_ns_store(ns, "_Pointer", crate::typedef::w_object());
    crate::module_ns_store(ns, "Structure", crate::typedef::w_object());
    crate::module_ns_store(ns, "Union", crate::typedef::w_object());
    crate::module_ns_store(ns, "Array", crate::typedef::w_object());
    crate::module_ns_store(ns, "CField", crate::typedef::w_object());
    crate::module_ns_store(ns, "CFuncPtr", crate::typedef::w_object());
    crate::module_ns_store(ns, "_SimpleCData", crate::typedef::w_object());
    crate::module_ns_store(ns, "sizeof", crate::typedef::w_object());
    crate::module_ns_store(ns, "alignment", crate::typedef::w_object());
    crate::module_ns_store(ns, "addressof", crate::typedef::w_object());
    crate::module_ns_store(ns, "byref", crate::typedef::w_object());
    crate::module_ns_store(ns, "resize", crate::typedef::w_object());
}
