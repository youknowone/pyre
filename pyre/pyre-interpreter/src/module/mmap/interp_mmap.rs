//! mmap class + module-level helpers — PyPy: pypy/module/mmap/interp_mmap.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! `init_mmap` entry point has been renamed to `register_module` so that
//! moduledef.rs can call it directly; `init_mmap_type` remains private.


// ──────────────────────────────────────────────────────────────────────
// mmap module — PyPy: pypy/module/mmap/.
//
// `mmap.mmap(fileno, length, ...)` maps through `host_env::mmap`
// (memmap2-based, cross-platform), not raw libc.  Per-instance state
// lives in the instance dict: `_ptr` (mapping pointer as i64), `_len`
// (i64), `_pos` (i64 cursor), `_access` (int), `_mode` (resolved mapping
// protection), `_id` (registry key), plus the descriptor the object owns —
// `_fd` on POSIX, `_handle` and `_tagname` on Windows, where the constructor
// duplicates the file's handle
// (`rmmap.py:953-970`).  The mapping is invalidated on close()/`__exit__`
// by dropping the registry entry (→ unmap); leaking it (e.g. GC drops the
// instance before close) is acceptable, matching CPython behaviour.
// ──────────────────────────────────────────────────────────────────────

// host_env's `MappedFile` is an RAII handle (memmap2) that unmaps on Drop,
// but pyre's mmap object keeps its state in a Python dict, which cannot own
// a Rust value.  The live `MappedFile` is therefore parked in this
// process-global table keyed by an id stashed in the instance dict (`_id`);
// `_ptr`/`_len` mirror `MappedFile::as_ptr()`/len so every read/write path
// stays a raw-pointer access.  close()/`__exit__`/resize drop or replace the
// entry; a map dropped by GC without close leaks its entry, exactly as the
// previous raw-pointer code leaked the mapping.
#[cfg(any(unix, windows))]
use rustpython_host_env::mmap as host_mmap;

/// The live mapping one registry slot owns.  A Windows `mmap(…, tagname=…)`
/// goes through `CreateFileMappingW`/`MapViewOfFile` (`rmmap.py:999-1004`)
/// rather than memmap2, so the two mapping flavours share one entry type.
#[cfg(any(unix, windows))]
enum MappedObj {
    Mapped(host_mmap::MappedFile),
    #[cfg(windows)]
    Named(host_mmap::NamedMmap),
}

#[cfg(any(unix, windows))]
impl MappedObj {
    fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Mapped(m) => m.as_ptr(),
            #[cfg(windows)]
            Self::Named(m) => m.as_slice().as_ptr(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Mapped(m) => m.as_slice().len(),
            #[cfg(windows)]
            Self::Named(m) => m.as_slice().len(),
        }
    }

    fn flush_range(&self, offset: usize, size: usize) -> std::io::Result<()> {
        match self {
            Self::Mapped(m) => m.flush_range(offset, size),
            #[cfg(windows)]
            Self::Named(m) => m.flush_range(offset, size),
        }
    }

    #[cfg(windows)]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Mapped(m) => m.as_mut_slice(),
            Self::Named(m) => m.as_mut_slice(),
        }
    }
}

#[cfg(any(unix, windows))]
static MMAP_REGISTRY: std::sync::Mutex<std::collections::BTreeMap<u64, MappedObj>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
#[cfg(any(unix, windows))]
static MMAP_NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[cfg(any(unix, windows))]
fn mmap_registry_insert(m: MappedObj) -> (u64, *const u8, usize) {
    let ptr = m.as_ptr();
    let len = m.len();
    let id = MMAP_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    MMAP_REGISTRY.lock().unwrap().insert(id, m);
    (id, ptr, len)
}

#[cfg(any(unix, windows))]
fn mmap_registry_remove(id: u64) {
    if id != 0 {
        MMAP_REGISTRY.lock().unwrap().remove(&id);
    }
}

#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn mmap_registry_replace(id: u64, m: MappedObj) -> (*const u8, usize) {
    let ptr = m.as_ptr();
    let len = m.len();
    // Inserting over the same key drops the previous mapping → unmaps it.
    MMAP_REGISTRY.lock().unwrap().insert(id, m);
    (ptr, len)
}

#[cfg(any(unix, windows))]
fn mmap_registry_flush(id: u64, offset: usize, size: usize) -> std::io::Result<()> {
    match MMAP_REGISTRY.lock().unwrap().get(&id) {
        Some(m) => m.flush_range(offset, size),
        None => Ok(()),
    }
}

#[cfg(all(unix, not(target_os = "redox")))]
fn mmap_registry_madvise(
    id: u64,
    start: usize,
    length: usize,
    advice: i32,
) -> std::io::Result<()> {
    match MMAP_REGISTRY.lock().unwrap().get(&id) {
        Some(MappedObj::Mapped(m)) => m.madvise_range(start, length, advice),
        None => Ok(()),
    }
}

#[cfg(unix)]
fn mmap_io_err(e: std::io::Error, ctx: &str) -> crate::PyError {
    crate::PyError::os_error_with_errno(e.raw_os_error().unwrap_or(0), ctx)
}

/// The host layer reaches the mapping through Win32, so the code an
/// `io::Error` carries here is a Win32 error: it belongs in `.winerror`, with
/// `.errno` and the OSError subclass derived from it (`rmmap.py:1010`
/// `lastSavedWindowsError`).
#[cfg(windows)]
fn mmap_io_err(e: std::io::Error, ctx: &str) -> crate::PyError {
    match e.raw_os_error() {
        Some(code) => crate::PyError::os_error_win32_syscall2(
            code,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
        None => crate::PyError::os_error(format!("{ctx}: {e}")),
    }
}

#[cfg(any(unix, windows))]
static MMAP_TYPE_OBJ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Reads (and lazily installs) the runtime-assigned `mmap` type object, not a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the `gc_interp::enabled` shape). The
/// `-> PyObjectRef` return fits a single word and it cannot raise.
#[cfg(any(unix, windows))]
#[majit_macros::dont_look_inside]
pub(crate) fn mmap_type() -> pyre_object::PyObjectRef {
    *MMAP_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("mmap", init_mmap_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        // A view dropped by the collector never reaches `__release_buffer__`,
        // so the buffer layer needs a way back here to drop the count.
        unsafe { pyre_object::buffer::set_external_release_hook(mmap_exports_decref) };
        tp as usize
    }) as pyre_object::PyObjectRef
}

#[cfg(any(unix, windows))]
fn mmap_get_attr_i64(obj: pyre_object::PyObjectRef, key: &str) -> i64 {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return 0;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(d, key) }
        && unsafe { pyre_object::is_int(v) } {
            return unsafe { pyre_object::w_int_get_value(v) };
        }
    0
}

#[cfg(windows)]
fn mmap_get_attr_str(obj: pyre_object::PyObjectRef, key: &str) -> String {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return String::new();
    }
    unsafe { pyre_object::w_dict_getitem_str(d, key) }
        .and_then(|v| unsafe { pyre_object::w_str_get_value_opt(v) })
        .unwrap_or_default()
        .to_owned()
}

#[cfg(any(unix, windows))]
fn mmap_set_attr(obj: pyre_object::PyObjectRef, key: &str, v: pyre_object::PyObjectRef) {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(d, key, v);
    }
}

#[cfg(any(unix, windows))]
fn mmap_ptr(obj: pyre_object::PyObjectRef) -> Result<(*mut u8, usize), crate::PyError> {
    let p = mmap_get_attr_i64(obj, "_ptr") as usize as *mut u8;
    let len = mmap_get_attr_i64(obj, "_len") as usize;
    if p.is_null() {
        return Err(crate::PyError::value_error("mmap closed or invalid"));
    }
    Ok((p, len))
}

/// `rmmap.py:387-405 MMap.close` — drop the mapping and the descriptor the
/// object owns.  POSIX keeps the caller's own fd (nothing to release);
/// Windows holds a handle it duplicated at construction, and leaving it open
/// would keep the file locked after `close()`.
#[cfg(any(unix, windows))]
fn mmap_close(obj: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    if mmap_get_attr_i64(obj, "_ptr") == 0 {
        return Ok(());
    }
    mmap_check_exports(obj, "cannot close exported pointers exist")?;
    mmap_registry_remove(mmap_get_attr_i64(obj, "_id") as u64);
    #[cfg(windows)]
    mmap_close_handle(obj);
    mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(0));
    mmap_set_attr(obj, "_len", pyre_object::w_int_new(0));
    mmap_set_attr(obj, "_id", pyre_object::w_int_new(0));
    Ok(())
}

/// The file handle the object owns, or `None` for a mapping backed by no file
/// of its own — an anonymous one, and the pagefile-backed tagged mappings.
#[cfg(windows)]
fn mmap_handle(obj: pyre_object::PyObjectRef) -> Option<host_mmap::Handle> {
    let handle = mmap_get_attr_i64(obj, "_handle") as isize;
    if handle == 0 || host_mmap::is_invalid_handle_value(handle) {
        return None;
    }
    Some(handle as host_mmap::Handle)
}

/// Close the duplicated file handle `_handle` names, and mark it invalid so a
/// second close is a no-op.
#[cfg(windows)]
fn mmap_close_handle(obj: pyre_object::PyObjectRef) {
    if let Some(handle) = mmap_handle(obj) {
        host_mmap::close_handle(handle);
    }
    mmap_set_attr(
        obj,
        "_handle",
        pyre_object::w_int_new(host_mmap::INVALID_HANDLE as isize as i64),
    );
}

/// `rmmap.py:509-524 MMap.file_size` — the backing file's current size, which
/// diverges from the mapped length after `resize()`.  An anonymous map has no
/// file to stat, and fstat on the `-1` descriptor is the OSError that reports
/// it.
#[cfg(unix)]
fn mmap_file_size(obj: pyre_object::PyObjectRef) -> Result<i64, crate::PyError> {
    let fd = mmap_get_attr_i64(obj, "_fd") as libc::c_int;
    if fd < 0 {
        return Err(crate::PyError::os_error(
            "mmap: cannot find file size for anonymous map",
        ));
    }
    let mut st: libc::stat = unsafe { core::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st as *mut libc::stat) } != 0 {
        return Err(crate::PyError::os_error_with_errno(
            std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
            "mmap.size: fstat failed",
        ));
    }
    Ok(st.st_size as i64)
}

/// `rmmap.py:511-520` — `GetFileSize` on the handle the map owns.  An
/// anonymous map has no handle and reports the mapped length instead.
#[cfg(windows)]
fn mmap_file_size(obj: pyre_object::PyObjectRef) -> Result<i64, crate::PyError> {
    match mmap_handle(obj) {
        Some(handle) => host_mmap::get_file_len(handle).map_err(|e| mmap_io_err(e, "GetFileSize")),
        None => Ok(mmap_get_attr_i64(obj, "_len")),
    }
}

/// True when `obj` is an `mmap` instance.
#[cfg(any(unix, windows))]
pub(crate) fn is_mmap(obj: pyre_object::PyObjectRef) -> bool {
    match crate::typedef::r#type(obj) {
        Some(tp) => std::ptr::eq(tp.as_ptr(), mmap_type()),
        None => false,
    }
}

/// `_exports` — how many buffers are currently exported from the mapping.
/// The mapping is raw foreign memory, so unmapping it under a live view is a
/// use-after-free rather than a stale-but-owned read; `close` and `resize`
/// refuse while this is non-zero.
#[cfg(any(unix, windows))]
pub(crate) fn mmap_exports_incref(obj: pyre_object::PyObjectRef) {
    let n = mmap_get_attr_i64(obj, "_exports");
    mmap_set_attr(obj, "_exports", pyre_object::w_int_new(n + 1));
}

/// Paired with [`mmap_exports_incref`]; saturates at zero so a double release
/// cannot wrap the count and strand the mapping.
#[cfg(any(unix, windows))]
pub(crate) unsafe fn mmap_exports_decref(obj: pyre_object::PyObjectRef) {
    if !is_mmap(obj) {
        return;
    }
    let n = mmap_get_attr_i64(obj, "_exports");
    mmap_set_attr(obj, "_exports", pyre_object::w_int_new((n - 1).max(0)));
}

/// How many items `start:stop:step` selects.  `sys.maxsize` is a legal step,
/// so the count is derived in `i128`, where `stop - start + step` cannot wrap
/// into a negative length.
#[cfg(any(unix, windows))]
fn mmap_slice_len(start: i64, stop: i64, step: i64) -> i64 {
    let (span, stride) = if step > 0 {
        ((stop as i128 - start as i128).max(0), step as i128)
    } else {
        ((start as i128 - stop as i128).max(0), -(step as i128))
    };
    ((span + stride - 1) / stride) as i64
}

/// Reject unmapping while a view still points into the mapping.
#[cfg(any(unix, windows))]
fn mmap_check_exports(obj: pyre_object::PyObjectRef, message: &str) -> Result<(), crate::PyError> {
    if mmap_get_attr_i64(obj, "_exports") > 0 {
        return Err(crate::PyError::new(crate::PyErrorKind::BufferError, message));
    }
    Ok(())
}

/// `W_MMap.readbuf_w` / `writebuf_w` — expose the live mapping to the
/// object-space buffer protocol.  `None` means the object is not an mmap;
/// the inner error preserves the closed-mapping failure.
#[cfg(any(unix, windows))]
pub(crate) fn mmap_buffer_view(
    obj: pyre_object::PyObjectRef,
) -> Option<Result<(usize, usize, bool), crate::PyError>> {
    let w_type = crate::typedef::r#type(obj)?;
    if !std::ptr::eq(w_type.as_ptr(), mmap_type()) {
        return None;
    }
    Some(mmap_ptr(obj).map(|(ptr, len)| {
        let readonly = mmap_get_attr_i64(obj, "_access") == MMAP_ACCESS_READ;
        (ptr as usize, len, readonly)
    }))
}

#[cfg(any(unix, windows))]
fn mmap_get_attr_obj(obj: pyre_object::PyObjectRef, key: &str) -> pyre_object::PyObjectRef {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { pyre_object::w_dict_getitem_str(d, key) }.unwrap_or(pyre_object::PY_NULL)
}

// `interp_mmap.py:243 descr_iter` / `:256 descr_reversed` return a
// generator yielding the 1-byte slices `m[i:i+1]`.  pyre models that
// generator as a dedicated iterator object holding the source mmap, a
// cursor, and a step (`+1` forwards, `-1` for `reversed`).
#[cfg(any(unix, windows))]
static MMAP_ITER_TYPE_OBJ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

#[cfg(any(unix, windows))]
fn mmap_iterator_type() -> pyre_object::PyObjectRef {
    *MMAP_ITER_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("mmap_iterator", init_mmap_iterator_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as pyre_object::PyObjectRef
}

#[cfg(any(unix, windows))]
fn make_mmap_iterator(m: pyre_object::PyObjectRef, start: i64, step: i64) -> pyre_object::PyObjectRef {
    let it = pyre_object::w_instance_new(mmap_iterator_type());
    mmap_set_attr(it, "_m", m);
    mmap_set_attr(it, "_i", pyre_object::w_int_new(start));
    mmap_set_attr(it, "_step", pyre_object::w_int_new(step));
    it
}

#[cfg(any(unix, windows))]
fn init_mmap_iterator_type(ns: pyre_object::PyObjectRef) {
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__iter__",
        crate::make_builtin_function_with_arity(
            "__iter__",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    ) };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__next__",
        crate::make_builtin_function_with_arity(
            "__next__",
            |args| {
                let it = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let m = mmap_get_attr_obj(it, "_m");
                let i = mmap_get_attr_i64(it, "_i");
                let step = mmap_get_attr_i64(it, "_step");
                let (p, len) = mmap_ptr(m)?;
                if i < 0 || i >= len as i64 {
                    return Err(crate::PyError::stop_iteration());
                }
                let b = unsafe { *p.add(i as usize) };
                mmap_set_attr(it, "_i", pyre_object::w_int_new(i + step));
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[b]))
            },
            1,
        ),
    ) };
}

#[cfg(any(unix, windows))]
fn init_mmap_type(ns: pyre_object::PyObjectRef) {
    // `interp_mmap.py:341 __new__ = interp2app(mmap)` — the class call
    // `mmap.mmap(fileno, length, ...)` lands here.  args[0] is the type, the
    // rest are the constructor arguments; every one of them binds by keyword
    // too, so the signature-aware carrier resolves them into fixed slots
    // (`interp_mmap.py:333-335` / `:354-356` — the argument list is the one
    // real difference between the two platforms' constructors).
    // `Signature::new(argnames, varargname, kwargname, kwonlyargcount,
    // posonlyargcount)`: nothing is keyword-only, and only `cls` is
    // positional-only.
    #[cfg(unix)]
    let signature = crate::gateway::Signature::new(
        vec!["cls", "fileno", "length", "flags", "prot", "access", "offset"],
        None,
        None,
        0,
        1,
    );
    #[cfg(windows)]
    let signature = crate::gateway::Signature::new(
        vec!["cls", "fileno", "length", "tagname", "access", "offset"],
        None,
        None,
        0,
        1,
    );
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__new__",
        crate::typedef::make_new_descr_with_signature(
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "mmap() requires fileno + length",
                    ));
                }
                mmap_construct(&args[1..])
            },
            signature,
        ),
    ) };

    // close() — munmap and zero the pointer.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                mmap_close(obj)?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    ) };

    // `interp_mmap.py:391 closed = GetSetProperty(W_MMap.closed_get)` —
    // bare attribute access (`m.closed`) returns the bool directly via
    // descriptor lookup, not a bound method.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "closed",
        crate::typedef::make_getset_descriptor_named(
            crate::make_builtin_function_with_arity(
                "closed",
                |args| {
                    // GetSetProperty fget callbacks receive (descriptor_self,
                    // w_obj); the mmap instance is the second argument.
                    let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                    Ok(pyre_object::w_bool_from(
                        mmap_get_attr_i64(obj, "_ptr") == 0,
                    ))
                },
                2,
            ),
            "closed",
        ),
    ) };

    // `interp_mmap.py:98-103 descr_size` returns `mmap.file_size()` — the
    // underlying file's current size, not the mapped length.  The two diverge
    // after `resize()`.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "size",
        crate::make_builtin_function_with_arity(
            "size",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                if mmap_get_attr_i64(obj, "_ptr") == 0 {
                    return Err(crate::PyError::value_error("mmap closed or invalid"));
                }
                Ok(pyre_object::w_int_new(mmap_file_size(obj)?))
            },
            1,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    // `interp_mmap.py:42 readline` — read bytes from current pos until
    // the first '\n' (inclusive); if absent, read to end.  Mirrors
    // `rmmap.py:421-432`.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            let _ = p;
            mmap_registry_flush(mmap_get_attr_i64(obj, "_id") as u64, off, n)
                .map_err(|e| mmap_io_err(e, "msync"))?;
            Ok(pyre_object::w_none())
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            if start > end {
                return Ok(pyre_object::w_int_new(-1));
            }
            if needle.is_empty() {
                return Ok(pyre_object::w_int_new(start as i64));
            }
            if needle.len() > end - start {
                return Ok(pyre_object::w_int_new(-1));
            }
            let hay = unsafe { std::slice::from_raw_parts(p.add(start), end - start) };
            let pos = (0..=hay.len() - needle.len())
                .find(|&i| &hay[i..i + needle.len()] == needle)
                .map(|i| (start + i) as i64)
                .unwrap_or(-1);
            Ok(pyre_object::w_int_new(pos))
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            if start > end {
                return Ok(pyre_object::w_int_new(-1));
            }
            if needle.is_empty() {
                return Ok(pyre_object::w_int_new(end as i64));
            }
            if needle.len() > end - start {
                return Ok(pyre_object::w_int_new(-1));
            }
            let hay = unsafe { std::slice::from_raw_parts(p.add(start), end - start) };
            let pos = (0..=hay.len() - needle.len())
                .rev()
                .find(|&i| &hay[i..i + needle.len()] == needle)
                .map(|i| (start + i) as i64)
                .unwrap_or(-1);
            Ok(pyre_object::w_int_new(pos))
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    ) };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                mmap_close(obj)?;
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    // `interp_mmap.py:188 descr_getitem` — integer index returns a
    // single int byte; slice returns bytes (contiguous fast path for
    // step=1, stepped extraction otherwise).
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                        // `sys.maxsize` is a legal step, and a cursor that
                        // wrapped past it would index the mapping from a
                        // negative offset.
                        i = i.saturating_add(step);
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
    ) };

    // `interp_mmap.py:206 descr_setitem` — integer index writes a
    // single byte (0..256); slice writes a buffer whose length matches
    // the slice length.  Read-only mmaps raise TypeError.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                    let length = mmap_slice_len(start, stop, step);
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
                            i = i.saturating_add(step);
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
    ) };

    // `interp_mmap.py:243 descr_iter` — iterate the 1-byte slices
    // `m[i:i+1]` forwards.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__iter__",
        crate::make_builtin_function_with_arity(
            "__iter__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(make_mmap_iterator(obj, 0, 1))
            },
            1,
        ),
    ) };

    // `interp_mmap.py:256 descr_reversed` — iterate the 1-byte slices
    // `m[i:i+1]` from `len(m) - 1` down to `0`.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__reversed__",
        crate::make_builtin_function_with_arity(
            "__reversed__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let len = mmap_get_attr_i64(obj, "_len");
                Ok(make_mmap_iterator(obj, len - 1, -1))
            },
            1,
        ),
    ) };

    // `interp_mmap.py:372-374` — `madvise` is in `optional`, installed only
    // where `rmmap.has_madvise`.  Windows has no madvise(2), so the method is
    // absent from the type there.
    //
    // `interp_mmap.py:descr_madvise` — call madvise(addr+start, length,
    // advice).  Defaults: start=0, length=remaining bytes.
    #[cfg(all(unix, not(target_os = "redox")))]
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            let _ = p;
            mmap_registry_madvise(mmap_get_attr_i64(obj, "_id") as u64, start, length, option)
                .map_err(|e| mmap_io_err(e, "madvise"))?;
            Ok(pyre_object::w_none())
        }),
    ) };

    // `interp_mmap.py:descr_move` — copy `length` bytes from source
    // offset to dest offset within the mapping (memmove semantics).
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                // `rmmap.py:587` uses `memmove`, so the ranges may overlap.
                unsafe {
                    std::ptr::copy((p + src) as *const u8, (p + dest) as *mut u8, count);
                }
                Ok(pyre_object::w_none())
            },
            4,
        ),
    ) };

    // `interp_mmap.py:146 resize` → `rmmap.py:589-651`.  Read-only / copy
    // mappings reject with TypeError; the remap itself is per-platform
    // ([`mmap_resize_mapping`]).
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                mmap_check_exports(obj, "mmap can't resize with extant buffers exported.")?;
                let (p, old_len) = mmap_ptr(obj)?;
                let newsize = unsafe { pyre_object::w_int_get_value(args[1]) };
                if newsize < 0 {
                    return Err(crate::PyError::value_error("new_size must be positive"));
                }
                mmap_resize_mapping(obj, p, old_len, newsize as usize)?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    ) };

    // `interp_mmap.py:descr_repr` — `<mmap.mmap closed=False, access=...>`.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                    return Ok(pyre_object::w_str_new_managed("<mmap.mmap closed=True>"));
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
                // W_MMap.descr_repr returns `space.newtext(...)` for both
                // live and closed mappings.
                Ok(pyre_object::w_str_new_managed(&format!(
                    "<mmap.mmap closed=False, access={access_str}, length={len}, pos={pos}, offset={offset}>"
                )))
            },
            1,
        ),
    ) };
}

#[cfg(any(unix, windows))]
const MMAP_ACCESS_DEFAULT: i64 = 0;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_READ: i64 = 1;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_WRITE: i64 = 2;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_COPY: i64 = 3;

/// `rmmap.py:589-601` — ftruncate the backing fd (if any) to `offset +
/// newsize`, then remap.  host_env's `MappedFile` (memmap2) cannot mremap in
/// place, so the mapping is re-created at the new size and the registry entry
/// swapped: a file-backed map is re-mapped from the (ftruncated) fd, an
/// anonymous map is remade and the surviving bytes copied.  The new mapping
/// may land at a different address than an mremap would have, but that
/// address is never exposed to Python, so the observable result is unchanged.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn mmap_resize_mapping(
    obj: pyre_object::PyObjectRef,
    p: *mut u8,
    old_len: usize,
    newsize: usize,
) -> Result<(), crate::PyError> {
    let fd = mmap_get_attr_i64(obj, "_fd") as libc::c_int;
    let offset = mmap_get_attr_i64(obj, "_offset");
    let mapped = if fd >= 0 {
        let r = unsafe { libc::ftruncate(fd, (offset as libc::off_t) + newsize as libc::off_t) };
        if r != 0 {
            return Err(crate::PyError::os_error_with_errno(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                "ftruncate",
            ));
        }
        let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::borrow_raw(fd) };
        // `rmmap.py:729-745` resolves `access` plus the caller's `prot` to the
        // protection used by the original mapping.  Reuse that resolved mode;
        // `_access == _ACCESS_DEFAULT` alone does not preserve a PROT_READ map.
        let mode = mmap_access_mode(mmap_get_attr_i64(obj, "_mode"));
        let (dup_fd, mapped) = host_mmap::map_file(borrowed, offset, newsize, mode)
            .map_err(|e| mmap_io_err(e, "mmap"))?;
        drop(dup_fd);
        mapped
    } else {
        mmap_remake_anon(p, old_len, newsize)?
    };
    let (newptr, newlen) = mmap_registry_replace(
        mmap_get_attr_i64(obj, "_id") as u64,
        MappedObj::Mapped(mapped),
    );
    mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(newptr as usize as i64));
    mmap_set_attr(obj, "_len", pyre_object::w_int_new(newlen as i64));
    Ok(())
}

/// `rmmap.py:602-651` — unmap the view and close the mapping object, move the
/// file's EOF to `offset + newsize`, then map the file again.  Dropping the
/// old mapping first is required, not just tidy: `SetEndOfFile` fails with
/// ERROR_USER_MAPPED_FILE while any view of the file is open.  An anonymous
/// mapping has no file to extend, so it is remade and the surviving bytes
/// copied, as on Linux.
#[cfg(windows)]
fn mmap_resize_mapping(
    obj: pyre_object::PyObjectRef,
    p: *mut u8,
    old_len: usize,
    newsize: usize,
) -> Result<(), crate::PyError> {
    let id = mmap_get_attr_i64(obj, "_id") as u64;
    let tagname = mmap_get_attr_str(obj, "_tagname");
    let named = !tagname.is_empty();
    let handle = mmap_handle(obj);
    let old_bytes = (named && handle.is_none())
        .then(|| unsafe { std::slice::from_raw_parts(p, old_len) }.to_vec());

    let mapped = if let Some(handle) = handle {
        let offset = mmap_get_attr_i64(obj, "_offset");
        mmap_registry_remove(id);
        let mapped = host_mmap::extend_file(handle, offset + newsize as i64)
            .and_then(|()| {
                if named {
                    mmap_remap_named(obj, handle, newsize, &tagname, true)
                } else {
                    mmap_remap_handle(obj, handle, newsize)
                }
            });
        match mapped {
            Ok(mapped) => mapped,
            Err(e) => {
                // The view is already gone, so `_ptr` would dangle; put the
                // mapping back at its old size, and mark the object closed if
                // even that fails.
                let restored = if named {
                    mmap_remap_named(obj, handle, old_len, &tagname, false)
                } else {
                    mmap_remap_handle(obj, handle, old_len)
                };
                match restored {
                    Ok(old) => {
                        mmap_store_mapping(obj, id, old);
                    }
                    Err(_) => {
                        mmap_close_handle(obj);
                        mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(0));
                        mmap_set_attr(obj, "_len", pyre_object::w_int_new(0));
                        mmap_set_attr(obj, "_id", pyre_object::w_int_new(0));
                    }
                }
                return Err(mmap_io_err(e, "mmap"));
            }
        }
    } else if named {
        // `rmmap.py:602-646` has no tag-name precheck: it closes the old map
        // handle and recreates the mapping with `self.tagname`.  For a
        // pagefile-backed mapping there is no file EOF to move, but the same
        // close/recreate sequence applies.  ERROR_ALREADY_EXISTS comes from
        // CreateFileMapping when another live view still owns the name.
        mmap_registry_remove(id);
        let mapped = mmap_remap_named(
            obj,
            host_mmap::INVALID_HANDLE,
            newsize,
            &tagname,
            true,
        );
        match mapped {
            Ok(mut mapped) => {
                let old = old_bytes.as_deref().unwrap_or_default();
                let keep = old.len().min(newsize);
                mapped.as_mut_slice()[..keep].copy_from_slice(&old[..keep]);
                mapped
            }
            Err(e) => {
                match mmap_remap_named(
                    obj,
                    host_mmap::INVALID_HANDLE,
                    old_len,
                    &tagname,
                    false,
                ) {
                    Ok(mut old_mapping) => {
                        if let Some(old) = old_bytes.as_deref() {
                            old_mapping.as_mut_slice().copy_from_slice(old);
                        }
                        mmap_store_mapping(obj, id, old_mapping);
                    }
                    Err(_) => {
                        mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(0));
                        mmap_set_attr(obj, "_len", pyre_object::w_int_new(0));
                        mmap_set_attr(obj, "_id", pyre_object::w_int_new(0));
                    }
                }
                return Err(mmap_io_err(e, "CreateFileMapping"));
            }
        }
    } else {
        MappedObj::Mapped(mmap_remake_anon(p, old_len, newsize)?)
    };
    mmap_store_mapping(obj, id, mapped);
    Ok(())
}

/// `rmmap.py:640-645` — map the file the object's handle names, at the access
/// mode it was built with.
#[cfg(windows)]
fn mmap_remap_handle(
    obj: pyre_object::PyObjectRef,
    handle: host_mmap::Handle,
    size: usize,
) -> std::io::Result<MappedObj> {
    let offset = mmap_get_attr_i64(obj, "_offset");
    let access = mmap_access_mode(mmap_get_attr_i64(obj, "_access"));
    host_mmap::map_handle(handle, offset, size, access).map(MappedObj::Mapped)
}

/// Recreate the Windows mapping object with its original tag name, matching
/// `rmmap.py:634-645`.  `CreateFileMappingW` reports ERROR_ALREADY_EXISTS when
/// another mapping still owns the name; resize surfaces that OS error while
/// the recovery remap is allowed to reopen the existing object.
#[cfg(windows)]
fn mmap_remap_named(
    obj: pyre_object::PyObjectRef,
    handle: host_mmap::Handle,
    size: usize,
    tagname: &str,
    reject_existing: bool,
) -> std::io::Result<MappedObj> {
    let offset = mmap_get_attr_i64(obj, "_offset");
    let access = mmap_access_mode(mmap_get_attr_i64(obj, "_mode"));
    let named = host_mmap::create_named_mapping(handle, tagname, access, offset, size)?;
    const ERROR_ALREADY_EXISTS: i32 = 183;
    if reject_existing && host_mmap::last_error() as i32 == ERROR_ALREADY_EXISTS {
        drop(named);
        return Err(std::io::Error::from_raw_os_error(ERROR_ALREADY_EXISTS));
    }
    Ok(MappedObj::Named(named))
}

/// Publish a freshly created mapping as the object's live one.
#[cfg(windows)]
fn mmap_store_mapping(obj: pyre_object::PyObjectRef, id: u64, mapped: MappedObj) {
    let (newptr, newlen) = mmap_registry_replace(id, mapped);
    mmap_set_attr(obj, "_ptr", pyre_object::w_int_new(newptr as usize as i64));
    mmap_set_attr(obj, "_len", pyre_object::w_int_new(newlen as i64));
}

/// A resized anonymous mapping is a new mapping holding the bytes that fit.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
fn mmap_remake_anon(
    p: *mut u8,
    old_len: usize,
    newsize: usize,
) -> Result<host_mmap::MappedFile, crate::PyError> {
    let keep = old_len.min(newsize);
    let old = unsafe { std::slice::from_raw_parts(p, keep) }.to_vec();
    let mut mapped = host_mmap::map_anon(newsize).map_err(|e| mmap_io_err(e, "mmap"))?;
    mapped.as_mut_slice()[..keep].copy_from_slice(&old);
    Ok(mapped)
}

/// Platforms without mremap raise SystemError, matching PyPy's
/// RValueError→SystemError translation at `interp_mmap.py:155-157`.
#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn mmap_resize_mapping(
    _obj: pyre_object::PyObjectRef,
    _p: *mut u8,
    _old_len: usize,
    _newsize: usize,
) -> Result<(), crate::PyError> {
    Err(crate::PyError::new(
        crate::error::PyErrorKind::SystemError,
        "mmap: resizing not available--no mremap()",
    ))
}

/// The mapping mode an `ACCESS_*` argument asks for.  host_env expresses a
/// mapping as an `AccessMode` rather than the raw `flProtect` /
/// `dwDesiredAccess` pair `rmmap.py:904-914` derives.
#[cfg(any(unix, windows))]
fn mmap_access_mode(access: i64) -> host_mmap::AccessMode {
    match access {
        x if x == MMAP_ACCESS_READ => host_mmap::AccessMode::Read,
        x if x == MMAP_ACCESS_WRITE => host_mmap::AccessMode::Write,
        x if x == MMAP_ACCESS_COPY => host_mmap::AccessMode::Copy,
        _ => host_mmap::AccessMode::Default,
    }
}

/// A constructor argument that was actually supplied.  The signature-aware
/// gateway pads the array out to the whole parameter list and leaves
/// `PY_NULL` in the slots the call omitted.
#[cfg(any(unix, windows))]
fn mmap_arg(args: &[pyre_object::PyObjectRef], idx: usize) -> Option<pyre_object::PyObjectRef> {
    args.get(idx).copied().filter(|a| !a.is_null())
}

/// `@unwrap_spec(fileno=int, length=int, …)` — every numeric constructor
/// argument is an int or the call is rejected before anything is mapped.
#[cfg(any(unix, windows))]
fn mmap_check_int_args(
    args: &[pyre_object::PyObjectRef],
    slots: &[(usize, &str)],
) -> Result<(), crate::PyError> {
    for &(idx, label) in slots {
        if let Some(a) = mmap_arg(args, idx)
            && !unsafe { pyre_object::is_int(a) }
        {
            return Err(crate::PyError::type_error(format!(
                "mmap() {label} must be an integer"
            )));
        }
    }
    Ok(())
}

/// `interp_mmap.py:341-345 W_MMap.__init__` — park the mapping and record the
/// per-instance state every method reads back out of the instance dict.  The
/// caller adds the descriptor it owns (`_fd` on POSIX, `_handle` on Windows).
#[cfg(any(unix, windows))]
fn mmap_new_object(
    mapped: MappedObj,
    access: i64,
    mode: host_mmap::AccessMode,
    offset: i64,
) -> pyre_object::PyObjectRef {
    let (id, ptr, len) = mmap_registry_insert(mapped);
    let obj = pyre_object::w_instance_new(mmap_type());
    // `w_instance_new` returns a movable GC object.  Every integer allocation
    // below can collect, so keep the instance in a shadow-stack slot and reload
    // it only after each value has been allocated.
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.base();
    roots.pin_root(obj);
    for (key, value) in [
        ("_ptr", ptr as usize as i64),
        ("_len", len as i64),
        ("_id", id as i64),
        ("_pos", 0),
        ("_access", access),
        ("_mode", mode as i64),
        ("_offset", offset),
    ] {
        let w_value = pyre_object::w_int_new(value);
        mmap_set_attr(
            unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) },
            key,
            w_value,
        );
    }
    unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) }
}

// `interp_mmap.py:333-350 mmap(fileno, length, flags, prot, access, offset)`
// — `args` carries the constructor arguments starting at index 0; the
// `__new__` typecall wrapper drops the class from args[0] before invoking
// this helper.
#[cfg(unix)]
fn mmap_construct(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let (Some(w_fileno), Some(w_length)) = (mmap_arg(args, 0), mmap_arg(args, 1)) else {
        return Err(crate::PyError::type_error(
            "mmap() requires fileno + length",
        ));
    };
    mmap_check_int_args(
        args,
        &[
            (0, "fileno"),
            (1, "length"),
            (2, "flags"),
            (3, "prot"),
            (4, "access"),
            (5, "offset"),
        ],
    )?;
    let fd = (unsafe { pyre_object::w_int_get_value(w_fileno) }) as libc::c_int;
    let length_value = unsafe { pyre_object::w_int_get_value(w_length) };
    let flags_arg = mmap_arg(args, 2).map_or(host_mmap::MAP_SHARED, |a| {
        (unsafe { pyre_object::w_int_get_value(a) }) as libc::c_int
    });
    let prot_arg = mmap_arg(args, 3).map_or(host_mmap::PROT_READ | host_mmap::PROT_WRITE, |a| {
        (unsafe { pyre_object::w_int_get_value(a) }) as libc::c_int
    });
    let access = mmap_arg(args, 4).map_or(MMAP_ACCESS_DEFAULT, |a| unsafe {
        pyre_object::w_int_get_value(a)
    });
    let offset_value = mmap_arg(args, 5).map_or(0, |a| unsafe {
        pyre_object::w_int_get_value(a)
    });

    // `rmmap.py:718-727` performs these guards in this order and before the
    // size_t/off_t casts.
    if access != MMAP_ACCESS_DEFAULT
        && (flags_arg != host_mmap::MAP_SHARED
            || prot_arg != (host_mmap::PROT_READ | host_mmap::PROT_WRITE))
    {
        return Err(crate::PyError::value_error(
            "mmap can't specify both access and flags, prot.",
        ));
    }
    if length_value < 0 {
        return Err(crate::PyError::type_error(
            "memory mapped size must be positive",
        ));
    }
    if offset_value < 0 {
        return Err(crate::PyError::value_error("negative offset"));
    }

    let length = length_value as libc::size_t;
    let offset = offset_value as libc::off_t;
    let (flags, prot) = match access {
        x if x == MMAP_ACCESS_READ => (host_mmap::MAP_SHARED, host_mmap::PROT_READ),
        x if x == MMAP_ACCESS_WRITE => {
            (host_mmap::MAP_SHARED, host_mmap::PROT_READ | host_mmap::PROT_WRITE)
        }
        x if x == MMAP_ACCESS_COPY => {
            (host_mmap::MAP_PRIVATE, host_mmap::PROT_READ | host_mmap::PROT_WRITE)
        }
        _ => (flags_arg, prot_arg),
    };
    // fileno == -1 → anonymous mapping.  host_env expresses the mapping as an
    // `AccessMode` rather than raw prot/flags: a writable share maps as
    // Write, a writable private map (MAP_PRIVATE) as Copy, and a
    // non-writable map as Read.  This is exact for the ACCESS_* modes and the
    // default read/write map; exotic low-level prot bits (PROT_EXEC/PROT_NONE)
    // and placement flags (MAP_FIXED) collapse to the nearest mode.  `_access`
    // still records the caller's original access argument, so repr() and the
    // write-guard are unchanged.
    let real_fd = fd;
    let mode = if prot & host_mmap::PROT_WRITE != 0 {
        if flags & host_mmap::MAP_PRIVATE != 0 {
            host_mmap::AccessMode::Copy
        } else {
            host_mmap::AccessMode::Write
        }
    } else {
        host_mmap::AccessMode::Read
    };
    let mapped = if real_fd == -1 {
        host_mmap::map_anon(length).map_err(|e| mmap_io_err(e, "mmap"))?
    } else {
        let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::borrow_raw(real_fd) };
        let (dup_fd, mapped) = host_mmap::map_file(borrowed, offset, length, mode)
            .map_err(|e| mmap_io_err(e, "mmap"))?;
        // pyre keeps the caller's original fd for size()/resize(); the dup
        // host_env made isn't needed once the mapping exists (the mapping
        // survives fd close).
        drop(dup_fd);
        mapped
    };
    let obj = mmap_new_object(MappedObj::Mapped(mapped), access, mode, offset as i64);
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.base();
    roots.pin_root(obj);
    let w_fd = pyre_object::w_int_new(real_fd as i64);
    mmap_set_attr(
        unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) },
        "_fd",
        w_fd,
    );
    Ok(unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) })
}

// `interp_mmap.py:354-370 mmap(fileno, length, tagname, access, offset)` —
// the Windows constructor names a mapping object where POSIX passes
// flags/prot, and the mapping's protection comes from `access` alone
// (`rmmap.py:900-914`).
#[cfg(windows)]
fn mmap_construct(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let (Some(w_fileno), Some(w_length)) = (mmap_arg(args, 0), mmap_arg(args, 1)) else {
        return Err(crate::PyError::type_error(
            "mmap() requires fileno + length",
        ));
    };
    mmap_check_int_args(args, &[(0, "fileno"), (1, "length"), (3, "access"), (4, "offset")])?;
    let fileno = (unsafe { pyre_object::w_int_get_value(w_fileno) }) as i32;
    let length = unsafe { pyre_object::w_int_get_value(w_length) };
    // `rmmap.py:681-683 _check_map_size`.
    if length < 0 {
        return Err(crate::PyError::type_error(
            "memory mapped size must be positive",
        ));
    }
    let tagname = match mmap_arg(args, 2) {
        Some(a) if !unsafe { pyre_object::is_none(a) } => {
            if !unsafe { pyre_object::is_str(a) } {
                return Err(crate::PyError::type_error(format!(
                    "expected str or None for 'tagname', not {}",
                    crate::type_methods::arg_type_name(a)
                )));
            }
            // The name reaches Win32 as UTF-16, which host_env builds from a
            // `&str`, so a tag carrying a lone surrogate has no route through.
            let Some(tag) = (unsafe { pyre_object::w_str_get_value_opt(a) }) else {
                return Err(crate::PyError::value_error(
                    "mmap() tagname must not contain lone surrogates",
                ));
            };
            if tag.contains('\0') {
                return Err(crate::PyError::value_error("embedded null character"));
            }
            tag
        }
        _ => "",
    };
    let access = mmap_arg(args, 3).map_or(MMAP_ACCESS_DEFAULT, |a| unsafe {
        pyre_object::w_int_get_value(a)
    });
    if !(MMAP_ACCESS_DEFAULT..=MMAP_ACCESS_COPY).contains(&access) {
        return Err(crate::PyError::value_error("mmap invalid access parameter."));
    }
    let offset = mmap_arg(args, 4).map_or(0, |a| unsafe { pyre_object::w_int_get_value(a) });
    // `rmmap.py:897-898`.
    if offset < 0 {
        return Err(crate::PyError::value_error("negative offset"));
    }
    let mut map_size = length as usize;

    // `rmmap.py:916-918` — "assume -1 and 0 both mean invalid file descriptor
    // to 'anonymously' map memory".  The handle is duplicated so Python code
    // may close the file it came from while the mapping lives on; the guard
    // hands it back to the OS on every path that does not reach the object.
    let mut file_handle = None;
    if fileno != -1 && fileno != 0 {
        let fh = rustpython_host_env::nt::handle_from_fd(fileno);
        if host_mmap::is_invalid_handle_value(fh as isize) {
            return Err(crate::PyError::os_error_with_errno(
                libc::EBADF,
                "mmap: bad file descriptor",
            ));
        }
        let guard = MmapHandleGuard(
            host_mmap::duplicate_handle(fh).map_err(|e| mmap_io_err(e, "DuplicateHandle"))?,
        );
        // `rmmap.py:925-928` trusts map_size when the size cannot be read
        // (a non-seeking file); only a readable size constrains it.
        if let Ok(file_len) = host_mmap::get_file_len(fh) {
            if map_size == 0 {
                if file_len == 0 {
                    return Err(crate::PyError::value_error("cannot mmap an empty file"));
                }
                // PyPy rejects only offset > size (`rmmap.py:943-945`), unlike
                // CPython's offset >= size check (`mmapmodule.c:2087-2091`).
                if offset > file_len {
                    return Err(crate::PyError::value_error(
                        "mmap offset is greater than file size",
                    ));
                }
                let remaining = file_len - offset;
                map_size = remaining as usize;
                if map_size as i64 != remaining {
                    return Err(crate::PyError::value_error("mmap length is too large"));
                }
            } else {
                // PyPy rejects a mapping past EOF (`rmmap.py:949-950`).
                // CPython instead passes offset + map_size on to
                // CreateFileMapping (`mmapmodule.c:2100-2107`), so this is a
                // deliberate PyPy-over-CPython parity choice.
                let required = offset
                    .checked_add(map_size as i64)
                    .ok_or_else(|| crate::PyError::value_error("mmap length is too large"))?;
                if required > file_len {
                    return Err(crate::PyError::value_error(
                        "mmap length is greater than file size",
                    ));
                }
            }
        }
        file_handle = Some(guard);
    }

    let handle = file_handle
        .as_ref()
        .map_or(host_mmap::INVALID_HANDLE, |guard| guard.0);
    // A tag names a mapping object other processes can open by the same name,
    // which memmap2 cannot express, so those go straight through
    // `CreateFileMappingW`/`MapViewOfFile` (`rmmap.py:999-1004`).
    let (mapped, owned_handle) = if !tagname.is_empty() {
        let named = host_mmap::create_named_mapping(
            handle,
            tagname,
            mmap_access_mode(access),
            offset,
            map_size,
        )
        .map_err(|e| mmap_io_err(e, "CreateFileMapping"))?;
        // The mapping object holds its own reference to the file, but the
        // handle stays with the mmap so `size()` can still stat that file.
        (
            MappedObj::Named(named),
            file_handle.map_or(host_mmap::INVALID_HANDLE, |guard| guard.release()),
        )
    } else if let Some(guard) = file_handle {
        let handle = guard.release();
        let mapped = host_mmap::map_handle(handle, offset, map_size, mmap_access_mode(access))
            .map_err(|e| {
                host_mmap::close_handle(handle);
                mmap_io_err(e, "mmap")
            })?;
        (MappedObj::Mapped(mapped), handle)
    } else {
        let mapped = host_mmap::map_anon(map_size).map_err(|e| mmap_io_err(e, "mmap"))?;
        (MappedObj::Mapped(mapped), host_mmap::INVALID_HANDLE)
    };
    let mode = mmap_access_mode(access);
    let obj = mmap_new_object(mapped, access, mode, offset);
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.base();
    roots.pin_root(obj);
    let w_tagname = pyre_object::w_str_new_managed(&tagname);
    mmap_set_attr(
        unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) },
        "_tagname",
        w_tagname,
    );
    let w_handle = pyre_object::w_int_new(owned_handle as isize as i64);
    mmap_set_attr(
        unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) },
        "_handle",
        w_handle,
    );
    Ok(unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) })
}

/// Closes the duplicated file handle unless it is released into the finished
/// mmap object, so a constructor that fails after `DuplicateHandle` does not
/// keep the file open.
#[cfg(windows)]
struct MmapHandleGuard(host_mmap::Handle);

#[cfg(windows)]
impl MmapHandleGuard {
    fn release(mut self) -> host_mmap::Handle {
        let handle = self.0;
        self.0 = host_mmap::INVALID_HANDLE;
        handle
    }
}

#[cfg(windows)]
impl Drop for MmapHandleGuard {
    fn drop(&mut self) {
        if !host_mmap::is_invalid_handle_value(self.0 as isize) {
            host_mmap::close_handle(self.0);
        }
    }
}

/// `rmmap.py:57-113` — the mapping flags and advice values a POSIX `mmap(2)`
/// takes.  The portable subset sources from host_env's re-exports; the
/// platform-specific extras it does not re-export (MAP_FIXED, the Linux-only
/// MAP_* flags, PROT_NONE) stay on libc.  Windows has none of them: the
/// mapping's protection comes from `access` alone there, and its module
/// carries only the ACCESS_* and page constants.
#[cfg(unix)]
fn register_posix_constants(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(
        ns,
        "MAP_SHARED",
        pyre_object::w_int_new(host_mmap::MAP_SHARED as i64),
    );
    crate::module_ns_store(
        ns,
        "MAP_PRIVATE",
        pyre_object::w_int_new(host_mmap::MAP_PRIVATE as i64),
    );
    crate::module_ns_store(
        ns,
        "MAP_ANON",
        pyre_object::w_int_new(host_mmap::MAP_ANON as i64),
    );
    crate::module_ns_store(
        ns,
        "MAP_ANONYMOUS",
        pyre_object::w_int_new(host_mmap::MAP_ANONYMOUS as i64),
    );
    crate::module_ns_store(
        ns,
        "MAP_FIXED",
        pyre_object::w_int_new(libc::MAP_FIXED as i64),
    );
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        crate::module_ns_store(
            ns,
            "MAP_POPULATE",
            pyre_object::w_int_new(libc::MAP_POPULATE as i64),
        );
        crate::module_ns_store(
            ns,
            "MAP_STACK",
            pyre_object::w_int_new(libc::MAP_STACK as i64),
        );
        crate::module_ns_store(
            ns,
            "MAP_HUGETLB",
            pyre_object::w_int_new(libc::MAP_HUGETLB as i64),
        );
        crate::module_ns_store(
            ns,
            "MAP_NORESERVE",
            pyre_object::w_int_new(libc::MAP_NORESERVE as i64),
        );
        crate::module_ns_store(
            ns,
            "MAP_LOCKED",
            pyre_object::w_int_new(libc::MAP_LOCKED as i64),
        );
        crate::module_ns_store(
            ns,
            "MAP_NONBLOCK",
            pyre_object::w_int_new(libc::MAP_NONBLOCK as i64),
        );
    }
    crate::module_ns_store(
        ns,
        "PROT_READ",
        pyre_object::w_int_new(host_mmap::PROT_READ as i64),
    );
    crate::module_ns_store(
        ns,
        "PROT_WRITE",
        pyre_object::w_int_new(host_mmap::PROT_WRITE as i64),
    );
    crate::module_ns_store(
        ns,
        "PROT_EXEC",
        pyre_object::w_int_new(host_mmap::PROT_EXEC as i64),
    );
    crate::module_ns_store(
        ns,
        "PROT_NONE",
        pyre_object::w_int_new(libc::PROT_NONE as i64),
    );
    crate::module_ns_store(
        ns,
        "MADV_NORMAL",
        pyre_object::w_int_new(host_mmap::MADV_NORMAL as i64),
    );
    crate::module_ns_store(
        ns,
        "MADV_RANDOM",
        pyre_object::w_int_new(host_mmap::MADV_RANDOM as i64),
    );
    crate::module_ns_store(
        ns,
        "MADV_SEQUENTIAL",
        pyre_object::w_int_new(host_mmap::MADV_SEQUENTIAL as i64),
    );
    crate::module_ns_store(
        ns,
        "MADV_WILLNEED",
        pyre_object::w_int_new(host_mmap::MADV_WILLNEED as i64),
    );
    crate::module_ns_store(
        ns,
        "MADV_DONTNEED",
        pyre_object::w_int_new(host_mmap::MADV_DONTNEED as i64),
    );
}

pub fn register_module(ns: pyre_object::PyObjectRef) {
    #[cfg(any(unix, windows))]
    {
        // `interp_mmap.py:42 error = OSError` alias.
        let w_os_error = crate::builtins::lookup_exc_class("OSError")
            .expect("OSError must be installed before init_mmap");
        crate::module_ns_store(ns, "error", w_os_error);

        #[cfg(unix)]
        register_posix_constants(ns);

        crate::module_ns_store(
            ns,
            "ACCESS_DEFAULT",
            pyre_object::w_int_new(MMAP_ACCESS_DEFAULT),
        );
        crate::module_ns_store(ns, "ACCESS_READ", pyre_object::w_int_new(MMAP_ACCESS_READ));
        crate::module_ns_store(
            ns,
            "ACCESS_WRITE",
            pyre_object::w_int_new(MMAP_ACCESS_WRITE),
        );
        crate::module_ns_store(ns, "ACCESS_COPY", pyre_object::w_int_new(MMAP_ACCESS_COPY));

        // `rmmap.py:204-206` / `:229-243` — POSIX has one allocation unit, the
        // page size; Windows' mapping granularity is the coarser
        // `SYSTEM_INFO.dwAllocationGranularity`.
        crate::module_ns_store(
            ns,
            "PAGESIZE",
            pyre_object::w_int_new(rustpython_host_env::os::page_size() as i64),
        );
        crate::module_ns_store(
            ns,
            "ALLOCATIONGRANULARITY",
            pyre_object::w_int_new(rustpython_host_env::os::alloc_granularity() as i64),
        );

        // Register the type itself.
        crate::module_ns_store(ns, "mmap", mmap_type());
    }
}
