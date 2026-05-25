//! mmap class + module-level helpers — PyPy: pypy/module/mmap/interp_mmap.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! `init_mmap` entry point has been renamed to `register_module` so that
//! moduledef.rs can call it directly; `init_mmap_type` remains private.

use crate::DictStorage;

// ──────────────────────────────────────────────────────────────────────
// mmap module — PyPy: pypy/module/mmap/.
//
// The `mmap.mmap(fileno, length, ...)` class wraps libc::mmap directly.
// Per-instance state lives in the instance dict: `_ptr` (raw pointer as
// i64), `_len` (i64), `_pos` (i64 cursor), `_access` (int).  The
// pointer is invalidated on close()/`__exit__` via munmap(2); leaking
// it (e.g. GC drops the instance before close) is acceptable, matching
// CPython behaviour.
// ──────────────────────────────────────────────────────────────────────

#[cfg(unix)]
thread_local! {
    static MMAP_TYPE_OBJ: std::cell::OnceCell<pyre_object::PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

#[cfg(unix)]
fn mmap_type() -> pyre_object::PyObjectRef {
    MMAP_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("mmap", init_mmap_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

#[cfg(unix)]
fn mmap_get_attr_i64(obj: pyre_object::PyObjectRef, key: &str) -> i64 {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return 0;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(d, key) } {
        if unsafe { pyre_object::is_int(v) } {
            return unsafe { pyre_object::w_int_get_value(v) };
        }
    }
    0
}

#[cfg(unix)]
fn mmap_set_attr(obj: pyre_object::PyObjectRef, key: &str, v: pyre_object::PyObjectRef) {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(d, key, v);
    }
}

#[cfg(unix)]
fn mmap_ptr(obj: pyre_object::PyObjectRef) -> Result<(*mut u8, usize), crate::PyError> {
    let p = mmap_get_attr_i64(obj, "_ptr") as usize as *mut u8;
    let len = mmap_get_attr_i64(obj, "_len") as usize;
    if p.is_null() {
        return Err(crate::PyError::value_error("mmap closed or invalid"));
    }
    Ok((p, len))
}

#[cfg(unix)]
fn init_mmap_type(ns: &mut DictStorage) {
    // close() — munmap and zero the pointer.
    crate::dict_storage_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let p = mmap_get_attr_i64(obj, "_ptr") as usize;
                let len = mmap_get_attr_i64(obj, "_len") as usize;
                if p != 0 && len != 0 {
                    let _ = unsafe { libc::munmap(p as *mut libc::c_void, len) };
                    mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(0));
                    mmap_set_attr(obj, "_len", pyre_object::w_int_new(0));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // closed — bool property; CPython exposes it as a get-only attribute.
    crate::dict_storage_store(
        ns,
        "closed",
        crate::make_builtin_function_with_arity(
            "closed",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_bool_from(
                    mmap_get_attr_i64(obj, "_ptr") == 0,
                ))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "size",
        crate::make_builtin_function_with_arity(
            "size",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_int_new(mmap_get_attr_i64(obj, "_len")))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tell",
        crate::make_builtin_function_with_arity(
            "tell",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_int_new(mmap_get_attr_i64(obj, "_pos")))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "seek",
        crate::make_builtin_function("seek", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("seek() missing argument"));
            }
            let obj = args[0];
            let (_, len) = mmap_ptr(obj)?;
            let off = unsafe { pyre_object::w_int_get_value(args[1]) };
            let whence = if args.len() >= 3 {
                unsafe { pyre_object::w_int_get_value(args[2]) }
            } else {
                0
            };
            let cur = mmap_get_attr_i64(obj, "_pos");
            let new_pos = match whence {
                0 => off,
                1 => cur + off,
                2 => len as i64 + off,
                _ => {
                    return Err(crate::PyError::value_error("invalid whence"));
                }
            };
            if new_pos < 0 || (new_pos as usize) > len {
                return Err(crate::PyError::value_error("seek out of range"));
            }
            mmap_set_attr(obj, "_pos", pyre_object::w_int_new(new_pos));
            Ok(pyre_object::w_none())
        }),
    );

    crate::dict_storage_store(
        ns,
        "read",
        crate::make_builtin_function("read", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("read() missing self"));
            }
            let obj = args[0];
            let (p, len) = mmap_ptr(obj)?;
            let pos = mmap_get_attr_i64(obj, "_pos") as usize;
            let remaining = len.saturating_sub(pos);
            let n = if args.len() >= 2 {
                let req = unsafe { pyre_object::w_int_get_value(args[1]) };
                if req < 0 {
                    remaining
                } else {
                    (req as usize).min(remaining)
                }
            } else {
                remaining
            };
            let slice = unsafe { std::slice::from_raw_parts(p.add(pos), n) };
            let data: Vec<u8> = slice.to_vec();
            mmap_set_attr(obj, "_pos", pyre_object::w_int_new((pos + n) as i64));
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
        }),
    );

    crate::dict_storage_store(
        ns,
        "read_byte",
        crate::make_builtin_function_with_arity(
            "read_byte",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let (p, len) = mmap_ptr(obj)?;
                let pos = mmap_get_attr_i64(obj, "_pos") as usize;
                if pos >= len {
                    return Err(crate::PyError::value_error("read byte out of range"));
                }
                let b = unsafe { *p.add(pos) };
                mmap_set_attr(obj, "_pos", pyre_object::w_int_new((pos + 1) as i64));
                Ok(pyre_object::w_int_new(b as i64))
            },
            1,
        ),
    );

    // `interp_mmap.py:42 readline` — read bytes from current pos until
    // the first '\n' (inclusive); if absent, read to end.  Mirrors
    // `rmmap.py:421-432`.
    crate::dict_storage_store(
        ns,
        "readline",
        crate::make_builtin_function_with_arity(
            "readline",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let (p, len) = mmap_ptr(obj)?;
                let pos = mmap_get_attr_i64(obj, "_pos") as usize;
                if pos >= len {
                    return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
                }
                let tail = unsafe { std::slice::from_raw_parts(p.add(pos), len - pos) };
                let eol = tail.iter().position(|&b| b == b'\n').map_or(len, |i| pos + i + 1);
                let data = unsafe { std::slice::from_raw_parts(p.add(pos), eol - pos) }.to_vec();
                mmap_set_attr(obj, "_pos", pyre_object::w_int_new(eol as i64));
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "write",
        crate::make_builtin_function_with_arity(
            "write",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("write() missing buffer"));
                }
                let obj = args[0];
                let (p, len) = mmap_ptr(obj)?;
                let access = mmap_get_attr_i64(obj, "_access");
                if access == MMAP_ACCESS_READ {
                    return Err(crate::PyError::type_error("mmap is read-only"));
                }
                let buf = unsafe {
                    if !pyre_object::bytesobject::is_bytes_like(args[1]) {
                        return Err(crate::PyError::type_error(
                            "write: buffer must be bytes-like",
                        ));
                    }
                    pyre_object::bytesobject::bytes_like_data(args[1])
                };
                let pos = mmap_get_attr_i64(obj, "_pos") as usize;
                if pos + buf.len() > len {
                    return Err(crate::PyError::value_error("data out of range"));
                }
                unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), p.add(pos), buf.len()) };
                mmap_set_attr(
                    obj,
                    "_pos",
                    pyre_object::w_int_new((pos + buf.len()) as i64),
                );
                Ok(pyre_object::w_int_new(buf.len() as i64))
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "write_byte",
        crate::make_builtin_function_with_arity(
            "write_byte",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("write_byte() missing arg"));
                }
                let obj = args[0];
                let (p, len) = mmap_ptr(obj)?;
                let access = mmap_get_attr_i64(obj, "_access");
                if access == MMAP_ACCESS_READ {
                    return Err(crate::PyError::type_error("mmap is read-only"));
                }
                let pos = mmap_get_attr_i64(obj, "_pos") as usize;
                if pos >= len {
                    return Err(crate::PyError::value_error("write_byte out of range"));
                }
                let b = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u8;
                unsafe { *p.add(pos) = b };
                mmap_set_attr(obj, "_pos", pyre_object::w_int_new((pos + 1) as i64));
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "flush",
        crate::make_builtin_function("flush", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("flush() missing self"));
            }
            let obj = args[0];
            let (p, len) = mmap_ptr(obj)?;
            let off = if args.len() >= 2 {
                (unsafe { pyre_object::w_int_get_value(args[1]) }) as usize
            } else {
                0
            };
            let n = if args.len() >= 3 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize
            } else {
                len - off
            };
            if off + n > len {
                return Err(crate::PyError::value_error("flush range out of bounds"));
            }
            let r = unsafe { libc::msync(p.add(off) as *mut libc::c_void, n, libc::MS_SYNC) };
            if r != 0 {
                return Err(crate::PyError::os_error_with_errno(
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                    "msync",
                ));
            }
            Ok(pyre_object::w_none())
        }),
    );

    crate::dict_storage_store(
        ns,
        "find",
        crate::make_builtin_function("find", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("find() missing pattern"));
            }
            let obj = args[0];
            let (p, len) = mmap_ptr(obj)?;
            let needle = unsafe {
                if !pyre_object::bytesobject::is_bytes_like(args[1]) {
                    return Err(crate::PyError::type_error(
                        "find: pattern must be bytes-like",
                    ));
                }
                pyre_object::bytesobject::bytes_like_data(args[1])
            };
            let cur = mmap_get_attr_i64(obj, "_pos") as usize;
            let start = if args.len() >= 3 {
                let s = unsafe { pyre_object::w_int_get_value(args[2]) };
                if s < 0 { cur } else { s as usize }
            } else {
                cur
            };
            let end = if args.len() >= 4 {
                let e = unsafe { pyre_object::w_int_get_value(args[3]) };
                if e < 0 { len } else { (e as usize).min(len) }
            } else {
                len
            };
            if start >= end || needle.is_empty() {
                return Ok(pyre_object::w_int_new(-1));
            }
            let hay = unsafe { std::slice::from_raw_parts(p.add(start), end - start) };
            let pos = (0..=hay.len().saturating_sub(needle.len()))
                .find(|&i| &hay[i..i + needle.len()] == needle)
                .map(|i| (start + i) as i64)
                .unwrap_or(-1);
            Ok(pyre_object::w_int_new(pos))
        }),
    );

    crate::dict_storage_store(
        ns,
        "rfind",
        crate::make_builtin_function("rfind", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("rfind() missing pattern"));
            }
            let obj = args[0];
            let (p, len) = mmap_ptr(obj)?;
            let needle = unsafe {
                if !pyre_object::bytesobject::is_bytes_like(args[1]) {
                    return Err(crate::PyError::type_error(
                        "rfind: pattern must be bytes-like",
                    ));
                }
                pyre_object::bytesobject::bytes_like_data(args[1])
            };
            let start = if args.len() >= 3 {
                let s = unsafe { pyre_object::w_int_get_value(args[2]) };
                if s < 0 { 0 } else { s as usize }
            } else {
                0
            };
            let end = if args.len() >= 4 {
                let e = unsafe { pyre_object::w_int_get_value(args[3]) };
                if e < 0 { len } else { (e as usize).min(len) }
            } else {
                len
            };
            if start >= end || needle.is_empty() {
                return Ok(pyre_object::w_int_new(-1));
            }
            let hay = unsafe { std::slice::from_raw_parts(p.add(start), end - start) };
            let pos = (0..=hay.len().saturating_sub(needle.len()))
                .rev()
                .find(|&i| &hay[i..i + needle.len()] == needle)
                .map(|i| (start + i) as i64)
                .unwrap_or(-1);
            Ok(pyre_object::w_int_new(pos))
        }),
    );

    crate::dict_storage_store(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                let p = mmap_get_attr_i64(obj, "_ptr") as usize;
                let len = mmap_get_attr_i64(obj, "_len") as usize;
                if p != 0 && len != 0 {
                    let _ = unsafe { libc::munmap(p as *mut libc::c_void, len) };
                    mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(0));
                    mmap_set_attr(obj, "_len", pyre_object::w_int_new(0));
                }
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    );

    crate::dict_storage_store(
        ns,
        "__len__",
        crate::make_builtin_function_with_arity(
            "__len__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_int_new(mmap_get_attr_i64(obj, "_len")))
            },
            1,
        ),
    );

    // `interp_mmap.py:descr_madvise` — call madvise(addr+start, length,
    // advice).  Defaults: start=0, length=remaining bytes.
    crate::dict_storage_store(
        ns,
        "madvise",
        crate::make_builtin_function("madvise", |args| {
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let p = mmap_get_attr_i64(obj, "_ptr") as usize;
            let total = mmap_get_attr_i64(obj, "_len") as usize;
            if args.len() < 2 {
                return Err(crate::PyError::type_error("madvise() requires option"));
            }
            let option = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
            let start: usize = args
                .get(2)
                .map(|&a| unsafe { pyre_object::w_int_get_value(a) } as usize)
                .unwrap_or(0);
            let length: usize = args
                .get(3)
                .map(|&a| unsafe { pyre_object::w_int_get_value(a) } as usize)
                .unwrap_or(total.saturating_sub(start));
            if start > total || start.saturating_add(length) > total {
                return Err(crate::PyError::value_error(
                    "madvise: start or length out of range",
                ));
            }
            #[cfg(unix)]
            {
                let rc = unsafe { libc::madvise((p + start) as *mut libc::c_void, length, option) };
                if rc != 0 {
                    return Err(crate::PyError::os_error_with_errno(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                        "madvise",
                    ));
                }
            }
            #[cfg(not(unix))]
            {
                let _ = (p, length, option);
            }
            Ok(pyre_object::w_none())
        }),
    );

    // `interp_mmap.py:descr_move` — copy `length` bytes from source
    // offset to dest offset within the mapping (memmove semantics).
    crate::dict_storage_store(
        ns,
        "move",
        crate::make_builtin_function_with_arity(
            "move",
            |args| {
                if args.len() < 4 {
                    return Err(crate::PyError::type_error(
                        "move() requires dest, src, count",
                    ));
                }
                let obj = args[0];
                let dest = (unsafe { pyre_object::w_int_get_value(args[1]) }) as usize;
                let src = (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize;
                let count = (unsafe { pyre_object::w_int_get_value(args[3]) }) as usize;
                let p = mmap_get_attr_i64(obj, "_ptr") as usize;
                let total = mmap_get_attr_i64(obj, "_len") as usize;
                if dest.saturating_add(count) > total || src.saturating_add(count) > total {
                    return Err(crate::PyError::value_error(
                        "source or destination out of range",
                    ));
                }
                #[cfg(unix)]
                unsafe {
                    libc::memmove(
                        (p + dest) as *mut libc::c_void,
                        (p + src) as *const libc::c_void,
                        count,
                    );
                }
                #[cfg(not(unix))]
                let _ = (p, dest, src, count);
                Ok(pyre_object::w_none())
            },
            4,
        ),
    );

    // `interp_mmap.py:descr_repr` — `<mmap.mmap closed=False, access=...>`.
    crate::dict_storage_store(
        ns,
        "__repr__",
        crate::make_builtin_function_with_arity(
            "__repr__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let closed = mmap_get_attr_i64(obj, "_ptr") == 0;
                let len = mmap_get_attr_i64(obj, "_len");
                let access = mmap_get_attr_i64(obj, "_access");
                let access_str = match access {
                    1 => "ACCESS_READ",
                    2 => "ACCESS_WRITE",
                    3 => "ACCESS_COPY",
                    _ => "ACCESS_DEFAULT",
                };
                Ok(pyre_object::w_str_new(&format!(
                    "<mmap.mmap closed={closed}, access={access_str}, length={len}, pos={}, offset=0>",
                    mmap_get_attr_i64(obj, "_pos")
                )))
            },
            1,
        ),
    );
}

#[cfg(unix)]
const MMAP_ACCESS_DEFAULT: i64 = 0;
#[cfg(unix)]
const MMAP_ACCESS_READ: i64 = 1;
#[cfg(unix)]
const MMAP_ACCESS_WRITE: i64 = 2;
#[cfg(unix)]
const MMAP_ACCESS_COPY: i64 = 3;

pub fn register_module(ns: &mut DictStorage) {
    #[cfg(unix)]
    {
        // `interp_mmap.py:42 error = OSError` alias.
        let w_os_error = crate::builtins::lookup_exc_class("OSError")
            .expect("OSError must be installed before init_mmap");
        crate::dict_storage_store(ns, "error", w_os_error);

        // Constants.  CPython exposes both POSIX MAP_/PROT_/MADV_ and the
        // Python ACCESS_* aliases.
        crate::dict_storage_store(
            ns,
            "MAP_SHARED",
            pyre_object::w_int_new(libc::MAP_SHARED as i64),
        );
        crate::dict_storage_store(
            ns,
            "MAP_PRIVATE",
            pyre_object::w_int_new(libc::MAP_PRIVATE as i64),
        );
        crate::dict_storage_store(
            ns,
            "MAP_ANON",
            pyre_object::w_int_new(libc::MAP_ANON as i64),
        );
        crate::dict_storage_store(
            ns,
            "MAP_ANONYMOUS",
            pyre_object::w_int_new(libc::MAP_ANON as i64),
        );
        crate::dict_storage_store(
            ns,
            "MAP_FIXED",
            pyre_object::w_int_new(libc::MAP_FIXED as i64),
        );
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            crate::dict_storage_store(
                ns,
                "MAP_POPULATE",
                pyre_object::w_int_new(libc::MAP_POPULATE as i64),
            );
            crate::dict_storage_store(
                ns,
                "MAP_STACK",
                pyre_object::w_int_new(libc::MAP_STACK as i64),
            );
            crate::dict_storage_store(
                ns,
                "MAP_HUGETLB",
                pyre_object::w_int_new(libc::MAP_HUGETLB as i64),
            );
            crate::dict_storage_store(
                ns,
                "MAP_NORESERVE",
                pyre_object::w_int_new(libc::MAP_NORESERVE as i64),
            );
            crate::dict_storage_store(
                ns,
                "MAP_LOCKED",
                pyre_object::w_int_new(libc::MAP_LOCKED as i64),
            );
            crate::dict_storage_store(
                ns,
                "MAP_NONBLOCK",
                pyre_object::w_int_new(libc::MAP_NONBLOCK as i64),
            );
        }
        crate::dict_storage_store(
            ns,
            "PROT_READ",
            pyre_object::w_int_new(libc::PROT_READ as i64),
        );
        crate::dict_storage_store(
            ns,
            "PROT_WRITE",
            pyre_object::w_int_new(libc::PROT_WRITE as i64),
        );
        crate::dict_storage_store(
            ns,
            "PROT_EXEC",
            pyre_object::w_int_new(libc::PROT_EXEC as i64),
        );
        crate::dict_storage_store(
            ns,
            "PROT_NONE",
            pyre_object::w_int_new(libc::PROT_NONE as i64),
        );
        crate::dict_storage_store(
            ns,
            "ACCESS_DEFAULT",
            pyre_object::w_int_new(MMAP_ACCESS_DEFAULT),
        );
        crate::dict_storage_store(ns, "ACCESS_READ", pyre_object::w_int_new(MMAP_ACCESS_READ));
        crate::dict_storage_store(
            ns,
            "ACCESS_WRITE",
            pyre_object::w_int_new(MMAP_ACCESS_WRITE),
        );
        crate::dict_storage_store(ns, "ACCESS_COPY", pyre_object::w_int_new(MMAP_ACCESS_COPY));
        crate::dict_storage_store(
            ns,
            "MADV_NORMAL",
            pyre_object::w_int_new(libc::MADV_NORMAL as i64),
        );
        crate::dict_storage_store(
            ns,
            "MADV_RANDOM",
            pyre_object::w_int_new(libc::MADV_RANDOM as i64),
        );
        crate::dict_storage_store(
            ns,
            "MADV_SEQUENTIAL",
            pyre_object::w_int_new(libc::MADV_SEQUENTIAL as i64),
        );
        crate::dict_storage_store(
            ns,
            "MADV_WILLNEED",
            pyre_object::w_int_new(libc::MADV_WILLNEED as i64),
        );
        crate::dict_storage_store(
            ns,
            "MADV_DONTNEED",
            pyre_object::w_int_new(libc::MADV_DONTNEED as i64),
        );

        // Page-related constants (sys.PAGESIZE in CPython mmap module).
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        crate::dict_storage_store(ns, "PAGESIZE", pyre_object::w_int_new(page));
        crate::dict_storage_store(ns, "ALLOCATIONGRANULARITY", pyre_object::w_int_new(page));

        // Register the type itself.
        crate::dict_storage_store(ns, "mmap", mmap_type());

        // mmap.mmap(fileno, length, flags=MAP_SHARED, prot=PROT_READ|WRITE,
        //          access=ACCESS_DEFAULT, offset=0) factory.  Resolves
        // access→flags/prot per CPython if access != ACCESS_DEFAULT.
        crate::dict_storage_store(
            ns,
            "_mmap_new",
            crate::make_builtin_function("_mmap_new", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "mmap() requires fileno + length",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let length = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::size_t;
                let flags_arg = if args.len() >= 3 {
                    (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int
                } else {
                    libc::MAP_SHARED
                };
                let prot_arg = if args.len() >= 4 {
                    (unsafe { pyre_object::w_int_get_value(args[3]) }) as libc::c_int
                } else {
                    libc::PROT_READ | libc::PROT_WRITE
                };
                let access = if args.len() >= 5 {
                    unsafe { pyre_object::w_int_get_value(args[4]) }
                } else {
                    MMAP_ACCESS_DEFAULT
                };
                let offset = if args.len() >= 6 {
                    (unsafe { pyre_object::w_int_get_value(args[5]) }) as libc::off_t
                } else {
                    0
                };
                let (flags, prot) = match access {
                    x if x == MMAP_ACCESS_READ => (libc::MAP_SHARED, libc::PROT_READ),
                    x if x == MMAP_ACCESS_WRITE => {
                        (libc::MAP_SHARED, libc::PROT_READ | libc::PROT_WRITE)
                    }
                    x if x == MMAP_ACCESS_COPY => {
                        (libc::MAP_PRIVATE, libc::PROT_READ | libc::PROT_WRITE)
                    }
                    _ => (flags_arg, prot_arg),
                };
                // fileno == -1 → anonymous mapping.
                let real_fd = if fd == -1 { -1 } else { fd };
                let final_flags = if real_fd == -1 {
                    flags | libc::MAP_ANON
                } else {
                    flags
                };
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        length,
                        prot,
                        final_flags,
                        real_fd,
                        offset,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    return Err(crate::PyError::os_error_with_errno(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                        "mmap",
                    ));
                }
                let obj = pyre_object::w_instance_new(mmap_type());
                mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(ptr as usize as i64));
                mmap_set_attr(obj, "_len", pyre_object::w_int_new(length as i64));
                mmap_set_attr(obj, "_pos", pyre_object::w_int_new(0));
                mmap_set_attr(obj, "_access", pyre_object::w_int_new(access));
                Ok(obj)
            }),
        );
    }
}
