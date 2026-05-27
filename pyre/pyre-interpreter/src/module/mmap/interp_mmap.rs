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
    // `interp_mmap.py:341 __new__ = interp2app(mmap)` — the class call
    // `mmap.mmap(fileno, length, ...)` lands here.  args[0] is the
    // type, the rest are the constructor positionals.
    crate::dict_storage_store(
        ns,
        "__new__",
        crate::make_builtin_function("__new__", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "mmap() requires fileno + length",
                ));
            }
            mmap_construct(&args[1..])
        }),
    );

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

    // `interp_mmap.py:391 closed = GetSetProperty(W_MMap.closed_get)` —
    // bare attribute access (`m.closed`) returns the bool directly via
    // descriptor lookup, not a bound method.
    crate::dict_storage_store(
        ns,
        "closed",
        crate::typedef::make_getset_descriptor_named(
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
            "closed",
        ),
    );

    // `interp_mmap.py:98-103 descr_size` returns `mmap.file_size()` —
    // the underlying file's current size via fstat, not the mapped
    // length.  The two diverge after `resize()`, and an anonymous mmap
    // (no fd) raises ValueError per rmmap.py:MMap.file_size.
    crate::dict_storage_store(
        ns,
        "size",
        crate::make_builtin_function_with_arity(
            "size",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                if mmap_get_attr_i64(obj, "_ptr") == 0 {
                    return Err(crate::PyError::value_error("mmap closed or invalid"));
                }
                let fd = mmap_get_attr_i64(obj, "_fd") as libc::c_int;
                if fd < 0 {
                    return Err(crate::PyError::os_error(
                        "mmap: cannot find file size for anonymous map",
                    ));
                }
                let mut st: libc::stat = unsafe { core::mem::zeroed() };
                let r = unsafe { libc::fstat(fd, &mut st as *mut libc::stat) };
                if r != 0 {
                    return Err(crate::PyError::os_error_with_errno(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                        "mmap.size: fstat failed",
                    ));
                }
                Ok(pyre_object::w_int_new(st.st_size as i64))
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
            if !unsafe { pyre_object::is_int(args[1]) } {
                return Err(crate::PyError::type_error(
                    "seek: offset must be an integer",
                ));
            }
            let off = unsafe { pyre_object::w_int_get_value(args[1]) };
            let whence = if args.len() >= 3 {
                if !unsafe { pyre_object::is_int(args[2]) } {
                    return Err(crate::PyError::type_error(
                        "seek: whence must be an integer",
                    ));
                }
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
            // `interp_mmap.py:60-69 read(num=-1)` — None or -1 reads to
            // end; positive value caps at remaining bytes.
            let n = if args.len() >= 2 && !unsafe { pyre_object::is_none(args[1]) } {
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(crate::PyError::type_error(
                        "read: argument must be int or None",
                    ));
                }
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
                let eol = tail
                    .iter()
                    .position(|&b| b == b'\n')
                    .map_or(len, |i| pos + i + 1);
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
                // `interp_mmap.py:114-121 write_byte(byte=int)` —
                // `@unwrap_spec(byte=int)` rejects non-ints, then
                // `chr(byte)` raises ValueError on values outside
                // 0..256.
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(crate::PyError::type_error(
                        "write_byte: byte must be an integer",
                    ));
                }
                let raw = unsafe { pyre_object::w_int_get_value(args[1]) };
                if !(0..=255).contains(&raw) {
                    return Err(crate::PyError::value_error(
                        "byte must be in range(0, 256)",
                    ));
                }
                unsafe { *p.add(pos) = raw as u8 };
                mmap_set_attr(obj, "_pos", pyre_object::w_int_new((pos + 1) as i64));
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        // `interp_mmap.py:123-134 flush(offset=0, size=0)` —
        // `@unwrap_spec(offset=int, size=int)` then `mmap.flush(offset,
        // size)`.  rmmap.flush passes size==0 through as "whole map",
        // which we mirror via `len - off`.
        "flush",
        crate::make_builtin_function("flush", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("flush() missing self"));
            }
            let obj = args[0];
            let (p, len) = mmap_ptr(obj)?;
            for (idx, label) in [(1usize, "offset"), (2, "size")] {
                if args.len() > idx && !unsafe { pyre_object::is_int(args[idx]) } {
                    return Err(crate::PyError::type_error(format!(
                        "flush: {label} must be an integer"
                    )));
                }
            }
            // Read as signed so negative user input does not wrap into
            // a huge `usize` and underflow the `len - off` subtraction
            // below (Critical: previously panicked / arbitrary length).
            let off_raw = if args.len() >= 2 {
                unsafe { pyre_object::w_int_get_value(args[1]) }
            } else {
                0
            };
            let raw_size_raw = if args.len() >= 3 {
                unsafe { pyre_object::w_int_get_value(args[2]) }
            } else {
                0
            };
            if off_raw < 0 || raw_size_raw < 0 {
                return Err(crate::PyError::value_error("flush range out of bounds"));
            }
            let off = off_raw as usize;
            let raw_size = raw_size_raw as usize;
            if off > len {
                return Err(crate::PyError::value_error("flush range out of bounds"));
            }
            let n = if raw_size == 0 { len - off } else { raw_size };
            if off.checked_add(n).map(|s| s > len).unwrap_or(true) {
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
            // `interp_mmap.py:56-69 find(w_tofind, w_start=None,
            // w_end=None)` defaults w_start to `self.mmap.pos` then
            // routes through rmmap.find which handles negative start /
            // end by adding `size` and clamping to 0.
            let cur = mmap_get_attr_i64(obj, "_pos") as usize;
            let start = if args.len() >= 3 {
                let s = unsafe { pyre_object::w_int_get_value(args[2]) };
                if s < 0 {
                    ((s + len as i64).max(0)) as usize
                } else {
                    (s as usize).min(len)
                }
            } else {
                cur
            };
            let end = if args.len() >= 4 {
                let e = unsafe { pyre_object::w_int_get_value(args[3]) };
                if e < 0 {
                    ((e + len as i64).max(0)) as usize
                } else {
                    (e as usize).min(len)
                }
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
            // `interp_mmap.py:71-84 rfind(w_tofind, w_start=None,
            // w_end=None)` defaults w_start to `self.mmap.pos`, not 0.
            // Negative args run through rmmap.find which adds `size`
            // and clamps to 0.
            let cur = mmap_get_attr_i64(obj, "_pos") as usize;
            let start = if args.len() >= 3 {
                let s = unsafe { pyre_object::w_int_get_value(args[2]) };
                if s < 0 {
                    ((s + len as i64).max(0)) as usize
                } else {
                    (s as usize).min(len)
                }
            } else {
                cur
            };
            let end = if args.len() >= 4 {
                let e = unsafe { pyre_object::w_int_get_value(args[3]) };
                if e < 0 {
                    ((e + len as i64).max(0)) as usize
                } else {
                    (e as usize).min(len)
                }
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

    // `interp_mmap.py:188 descr_getitem` — integer index returns a
    // single int byte; slice returns bytes (contiguous fast path for
    // step=1, stepped extraction otherwise).
    crate::dict_storage_store(
        ns,
        "__getitem__",
        crate::make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("__getitem__() requires index"));
                }
                let obj = args[0];
                let index = args[1];
                let (p, len) = mmap_ptr(obj)?;
                let len_i64 = len as i64;
                if unsafe { pyre_object::is_slice(index) } {
                    let (start, stop, step) =
                        unsafe { crate::baseobjspace::normalize_slice(index, len_i64)? };
                    if step == 1 {
                        if stop <= start {
                            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
                        }
                        let n = (stop - start) as usize;
                        let data = unsafe { std::slice::from_raw_parts(p.add(start as usize), n) };
                        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(data));
                    }
                    let mut out = Vec::new();
                    let mut i = start;
                    while (step > 0 && i < stop) || (step < 0 && i > stop) {
                        out.push(unsafe { *p.add(i as usize) });
                        i += step;
                    }
                    return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&out));
                }
                if !unsafe { pyre_object::is_int(index) } {
                    return Err(crate::PyError::type_error(
                        "mmap indices must be integers or slices",
                    ));
                }
                let mut idx = unsafe { pyre_object::w_int_get_value(index) };
                if idx < 0 {
                    idx += len_i64;
                }
                if idx < 0 || idx >= len_i64 {
                    return Err(crate::PyError::index_error("mmap index out of range"));
                }
                let b = unsafe { *p.add(idx as usize) };
                Ok(pyre_object::w_int_new(b as i64))
            },
            2,
        ),
    );

    // `interp_mmap.py:206 descr_setitem` — integer index writes a
    // single byte (0..256); slice writes a buffer whose length matches
    // the slice length.  Read-only mmaps raise TypeError.
    crate::dict_storage_store(
        ns,
        "__setitem__",
        crate::make_builtin_function_with_arity(
            "__setitem__",
            |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error(
                        "__setitem__() requires index and value",
                    ));
                }
                let obj = args[0];
                let index = args[1];
                let value = args[2];
                let access = mmap_get_attr_i64(obj, "_access");
                if access == MMAP_ACCESS_READ {
                    return Err(crate::PyError::type_error("mmap is read-only"));
                }
                let (p, len) = mmap_ptr(obj)?;
                let len_i64 = len as i64;
                if unsafe { pyre_object::is_slice(index) } {
                    let (start, stop, step) =
                        unsafe { crate::baseobjspace::normalize_slice(index, len_i64)? };
                    let length = if step > 0 {
                        ((stop - start).max(0) + step - 1) / step
                    } else {
                        ((start - stop).max(0) + (-step) - 1) / (-step)
                    };
                    if !unsafe { pyre_object::bytesobject::is_bytes_like(value) } {
                        return Err(crate::PyError::type_error(
                            "mmap slice assignment must be bytes-like",
                        ));
                    }
                    let buf = unsafe { pyre_object::bytesobject::bytes_like_data(value) };
                    if (buf.len() as i64) != length {
                        return Err(crate::PyError::value_error(
                            "mmap slice assignment is wrong size",
                        ));
                    }
                    if step == 1 {
                        if length > 0 {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    buf.as_ptr(),
                                    p.add(start as usize),
                                    length as usize,
                                );
                            }
                        }
                    } else {
                        let mut i = start;
                        let mut k = 0usize;
                        while (step > 0 && i < stop) || (step < 0 && i > stop) {
                            unsafe { *p.add(i as usize) = buf[k] };
                            i += step;
                            k += 1;
                        }
                    }
                    return Ok(pyre_object::w_none());
                }
                if !unsafe { pyre_object::is_int(index) } {
                    return Err(crate::PyError::type_error(
                        "mmap indices must be integers or slices",
                    ));
                }
                let mut idx = unsafe { pyre_object::w_int_get_value(index) };
                if idx < 0 {
                    idx += len_i64;
                }
                if idx < 0 || idx >= len_i64 {
                    return Err(crate::PyError::index_error("mmap index out of range"));
                }
                if !unsafe { pyre_object::is_int(value) } {
                    return Err(crate::PyError::type_error(
                        "mmap item value must be an integer",
                    ));
                }
                let v = unsafe { pyre_object::w_int_get_value(value) };
                if !(0..256).contains(&v) {
                    return Err(crate::PyError::value_error(
                        "mmap item value must be in range(0, 256)",
                    ));
                }
                unsafe { *p.add(idx as usize) = v as u8 };
                Ok(pyre_object::w_none())
            },
            3,
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
            for (idx, label) in [(1usize, "option"), (2, "start"), (3, "length")] {
                if args.len() > idx && !unsafe { pyre_object::is_int(args[idx]) } {
                    return Err(crate::PyError::type_error(format!(
                        "madvise: {label} must be an integer"
                    )));
                }
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
                // `interp_mmap.py:136-143 move(dest, src, count)` —
                // `@unwrap_spec(dest=int, src=int, count=int)` plus
                // `self.check_writeable()` upfront.  We require all
                // three args to be ints and reject ACCESS_READ.
                for (idx, label) in [(1, "dest"), (2, "src"), (3, "count")] {
                    if !unsafe { pyre_object::is_int(args[idx]) } {
                        return Err(crate::PyError::type_error(format!(
                            "move: {label} must be an integer"
                        )));
                    }
                }
                if mmap_get_attr_i64(obj, "_access") == MMAP_ACCESS_READ {
                    return Err(crate::PyError::type_error("mmap is read-only"));
                }
                let dest = (unsafe { pyre_object::w_int_get_value(args[1]) }) as usize;
                let src = (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize;
                let count = (unsafe { pyre_object::w_int_get_value(args[3]) }) as usize;
                let p = mmap_get_attr_i64(obj, "_ptr") as usize;
                if p == 0 {
                    return Err(crate::PyError::value_error("mmap closed or invalid"));
                }
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

    // `interp_mmap.py:146 resize` → `rmmap.py:589-601`.  POSIX path:
    // ftruncate the backing fd (if any) to `offset + newsize`, then
    // mremap(MREMAP_MAYMOVE).  Platforms without mremap (e.g. macOS)
    // raise SystemError to match PyPy's RValueError→SystemError
    // translation at `interp_mmap.py:155-157`.  Read-only / copy
    // mappings reject with TypeError.
    crate::dict_storage_store(
        ns,
        "resize",
        crate::make_builtin_function_with_arity(
            "resize",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("resize() requires newsize"));
                }
                let obj = args[0];
                let access = mmap_get_attr_i64(obj, "_access");
                if !(access == MMAP_ACCESS_WRITE || access == MMAP_ACCESS_DEFAULT) {
                    return Err(crate::PyError::type_error(
                        "mmap can't resize a readonly or copy-on-write memory map.",
                    ));
                }
                let (p, old_len) = mmap_ptr(obj)?;
                let newsize = unsafe { pyre_object::w_int_get_value(args[1]) };
                if newsize < 0 {
                    return Err(crate::PyError::value_error("new_size must be positive"));
                }
                let newsize = newsize as usize;
                let fd = mmap_get_attr_i64(obj, "_fd") as libc::c_int;
                let offset = mmap_get_attr_i64(obj, "_offset");

                #[cfg(any(target_os = "linux", target_os = "android"))]
                {
                    if fd >= 0 {
                        let r = unsafe {
                            libc::ftruncate(fd, (offset as libc::off_t) + newsize as libc::off_t)
                        };
                        if r != 0 {
                            return Err(crate::PyError::os_error_with_errno(
                                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                                "ftruncate",
                            ));
                        }
                    }
                    let newptr = unsafe {
                        libc::mremap(
                            p as *mut libc::c_void,
                            old_len,
                            newsize,
                            libc::MREMAP_MAYMOVE,
                        )
                    };
                    if newptr == libc::MAP_FAILED {
                        return Err(crate::PyError::os_error_with_errno(
                            std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                            "mremap",
                        ));
                    }
                    mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(newptr as usize as i64));
                    mmap_set_attr(obj, "_len", pyre_object::w_int_new(newsize as i64));
                    Ok(pyre_object::w_none())
                }
                #[cfg(not(any(target_os = "linux", target_os = "android")))]
                {
                    let _ = (p, old_len, fd, offset, newsize);
                    Err(crate::PyError::new(
                        crate::error::PyErrorKind::SystemError,
                        "mmap: resizing not available--no mremap()",
                    ))
                }
            },
            2,
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
                // `interp_mmap.py:297-316 descr_repr`: closed mmaps
                // suppress the inner fields and emit just
                // `<mmap.mmap closed=True>`; otherwise dump
                // access / length / pos / offset.  Capitalised True /
                // False matches CPython's bool repr.
                if mmap_get_attr_i64(obj, "_ptr") == 0 {
                    return Ok(pyre_object::w_str_new("<mmap.mmap closed=True>"));
                }
                let len = mmap_get_attr_i64(obj, "_len");
                let pos = mmap_get_attr_i64(obj, "_pos");
                let offset = mmap_get_attr_i64(obj, "_offset");
                let access = mmap_get_attr_i64(obj, "_access");
                let access_str = match access {
                    x if x == MMAP_ACCESS_READ => "ACCESS_READ",
                    x if x == MMAP_ACCESS_WRITE => "ACCESS_WRITE",
                    x if x == MMAP_ACCESS_COPY => "ACCESS_COPY",
                    _ => "ACCESS_DEFAULT",
                };
                Ok(pyre_object::w_str_new(&format!(
                    "<mmap.mmap closed=False, access={access_str}, length={len}, pos={pos}, offset={offset}>"
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

// `interp_mmap.py:55-130 mmap_new` — `args` carries the positional
// constructor arguments (fileno, length, flags, prot, access, offset)
// starting at index 0; the `__new__` typecall wrapper drops the class
// from args[0] before invoking this helper.
#[cfg(unix)]
fn mmap_construct(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "mmap() requires fileno + length",
        ));
    }
    for (idx, label) in [
        (0usize, "fileno"),
        (1, "length"),
        (2, "flags"),
        (3, "prot"),
        (4, "access"),
        (5, "offset"),
    ] {
        if args.len() > idx && !unsafe { pyre_object::is_int(args[idx]) } {
            return Err(crate::PyError::type_error(format!(
                "mmap() {label} must be an integer"
            )));
        }
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
        x if x == MMAP_ACCESS_WRITE => (libc::MAP_SHARED, libc::PROT_READ | libc::PROT_WRITE),
        x if x == MMAP_ACCESS_COPY => (libc::MAP_PRIVATE, libc::PROT_READ | libc::PROT_WRITE),
        _ => (flags_arg, prot_arg),
    };
    // fileno == -1 → anonymous mapping.
    let real_fd = fd;
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
    mmap_set_attr(obj, "_fd", pyre_object::w_int_new(real_fd as i64));
    mmap_set_attr(obj, "_offset", pyre_object::w_int_new(offset as i64));
    Ok(obj)
}

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
    }
}
