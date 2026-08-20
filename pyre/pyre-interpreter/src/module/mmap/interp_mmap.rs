//! mmap class + module-level helpers — PyPy: pypy/module/mmap/interp_mmap.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! `init_mmap` entry point has been renamed to `register_module` so that
//! moduledef.rs can call it directly; `init_mmap_type` remains private.


// ──────────────────────────────────────────────────────────────────────
// mmap module — PyPy: pypy/module/mmap/.
//
// `mmap.mmap(fileno, length, ...)` maps through `host_env::mmap`
// (memmap2-based, cross-platform), not raw libc, so the module works on
// POSIX and on Windows, where the constructor takes a `tagname` instead of
// flags/prot.  Like PyPy's `rmmap.MMap`, every Python object owns its
// mapping and the descriptor it duplicated — the fd on POSIX, the file
// handle on Windows (`rmmap.py`) — so close()/GC release exactly
// that object's native resources.
// ──────────────────────────────────────────────────────────────────────

#[cfg(any(unix, windows))]
use rustpython_host_env::mmap as host_mmap;

/// The live mapping one object owns.  A Windows `mmap(…, tagname=…)` goes
/// through `CreateFileMappingW`/`MapViewOfFile` (`rmmap.py:999-1004`) rather
/// than memmap2, so the two mapping flavours share one type.
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

    #[allow(dead_code)]
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

/// `rmmap.MMap`: the mapping plus the descriptor the object duplicated for it.
///
/// `mapped` is `None` only while a resize holds no mapping: Windows'
/// `SetEndOfFile` fails with ERROR_USER_MAPPED_FILE while any view of the file
/// is open, so the old view must go before the file grows.  `rmmap.py:602-651`
/// clears `self.data` across the same window.
#[cfg(any(unix, windows))]
struct NativeMMap {
    mapped: Option<MappedObj>,
    /// `rmmap.py`'s `_POSIX` branch — the descriptor the object duplicated at
    /// construction, kept so `resize()` can grow the file and so `close()`
    /// releases it.  A Windows mapping tracks `handle` instead.
    #[cfg(unix)]
    fd: Option<rustpython_host_env::crt_fd::Owned>,
    /// The `flags` the mapping was created with, as `mmap(2)` received them.
    /// `mode` cannot stand in for it: a `MAP_PRIVATE` mapping that is not
    /// writable resolves to the same `AccessMode::Read` as a shared one.
    /// `resize` needs the bit to reject expanding a shared anonymous mapping.
    #[cfg(unix)]
    #[allow(dead_code)]
    flags: libc::c_int,
    /// `rmmap.py:953-970` — the file handle the object duplicated at
    /// construction, or `INVALID_HANDLE` for a mapping backed by no file of
    /// its own.  Leaving it open would keep the file locked after `close()`.
    #[cfg(windows)]
    handle: host_mmap::Handle,
    /// `rmmap.py:986-987 m.tagname` — the name `CreateFileMappingW` created the
    /// mapping object under, empty for an untagged mapping.  `resize` recreates
    /// the mapping object under the same name.
    #[cfg(windows)]
    tagname: String,
    trackfd: bool,
}

#[cfg(windows)]
impl Drop for NativeMMap {
    fn drop(&mut self) {
        if !host_mmap::is_invalid_handle_value(self.handle as isize) {
            host_mmap::close_handle(self.handle);
            self.handle = host_mmap::INVALID_HANDLE;
        }
    }
}

/// PyPy `interp_mmap.py W_MMap`: cursor/access/offset/export state is stored
/// on the object and the low-level `rmmap.MMap` is owned by that same object.
/// The mapdict prefix is required because mmap is an acceptable base class.
#[cfg(any(unix, windows))]
#[crate::pyre_class("mmap.mmap", cpython_heaptype)]
#[derive(Default)]
pub struct W_MMap {
    pub map: usize,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    backend: *mut NativeMMap,
    pos: i64,
    access: i64,
    /// The protection the mapping was actually created with, which `access`
    /// alone does not determine: `_ACCESS_DEFAULT` resolves against the
    /// caller's `prot` (`rmmap.py`).  `resize` remaps with it, where
    /// `mremap` would have preserved it.  Held as the `AccessMode`
    /// discriminant, which shares its numbering with `MMAP_ACCESS_*`.
    mode: i64,
    offset: i64,
    exports: i64,
}

#[cfg(any(unix, windows))]
const _: () = assert!(
    std::mem::offset_of!(W_MMap, map)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, map),
    "W_MMap must keep W_ObjectObject's map offset"
);
#[cfg(any(unix, windows))]
const _: () = assert!(
    std::mem::offset_of!(W_MMap, storage)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, storage),
    "W_MMap must keep W_ObjectObject's storage offset"
);

#[cfg(any(unix, windows))]
fn mmap_this(obj: pyre_object::PyObjectRef) -> Result<&'static mut W_MMap, crate::PyError> {
    W_MMap::from_obj(obj).ok_or_else(|| crate::PyError::type_error("expected mmap object"))
}

#[cfg(any(unix, windows))]
fn mmap_native(obj: pyre_object::PyObjectRef) -> Result<&'static mut NativeMMap, crate::PyError> {
    let this = mmap_this(obj)?;
    if this.backend.is_null() {
        return Err(crate::PyError::value_error("mmap closed or invalid"));
    }
    Ok(unsafe { &mut *this.backend })
}

#[cfg(any(unix, windows))]
fn mmap_close_native(obj: pyre_object::PyObjectRef) {
    if let Some(this) = W_MMap::from_obj(obj) && !this.backend.is_null() {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
        this.pos = 0;
    }
}

/// The live mapping, or `EBADF` when the object no longer holds one.
#[cfg(any(unix, windows))]
fn mmap_mapped(obj: pyre_object::PyObjectRef) -> std::io::Result<&'static MappedObj> {
    mmap_native(obj)
        .ok()
        .and_then(|native| native.mapped.as_ref())
        .ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))
}

/// `rmmap.py flush` — `c_msync(self.getptr(offset), size, MS_SYNC)`.
/// The pointer goes to `msync` exactly as computed: Linux rejects a start that
/// is not page-aligned with EINVAL, and that refusal is observable
/// (`test_flush_return_value`).  `MappedFile::flush_range` rounds the start
/// down to a page boundary first, which turns the error into a success.
#[cfg(unix)]
fn mmap_flush(obj: pyre_object::PyObjectRef, offset: usize, size: usize) -> std::io::Result<()> {
    let start = unsafe { mmap_mapped(obj)?.as_ptr().add(offset) };
    if unsafe { libc::msync(start as *mut libc::c_void, size, libc::MS_SYNC) } == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
fn mmap_flush(obj: pyre_object::PyObjectRef, offset: usize, size: usize) -> std::io::Result<()> {
    mmap_mapped(obj)?.flush_range(offset, size)
}

#[cfg(all(unix, not(target_os = "redox")))]
fn mmap_madvise(
    obj: pyre_object::PyObjectRef,
    start: usize,
    length: usize,
    advice: i32,
) -> std::io::Result<()> {
    match mmap_mapped(obj)? {
        MappedObj::Mapped(m) => m.madvise_range(start, length, advice),
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
        let tp = crate::typedef::make_builtin_type_with_layout(
            "mmap.mmap",
            init_mmap_type,
            crate::typedef::w_object(),
            <W_MMap as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        // CPython 3.14 Modules/mmapmodule.c:mmap_exec uses
        // PyType_FromModuleAndSpec; mmap_object_type_spec is immutable.
        crate::typedef::mark_cpython_heap_type(tp, true);
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_MMap as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe { pyre_object::w_type_set_weakrefable(tp, true) };
        // A view dropped by the collector never reaches `__release_buffer__`,
        // so the buffer layer needs a way back here to drop the count.
        unsafe { pyre_object::buffer::set_external_release_hook(mmap_exports_decref) };
        tp as usize
    }) as pyre_object::PyObjectRef
}

#[cfg(any(unix, windows))]
fn mmap_get_attr_i64(obj: pyre_object::PyObjectRef, key: &str) -> i64 {
    if let Some(this) = W_MMap::from_obj(obj) {
        return match key {
            "_ptr" => (!this.backend.is_null())
                .then(|| unsafe { (&*this.backend).mapped.as_ref() })
                .flatten()
                .map_or(0, |m| m.as_ptr() as usize as i64),
            "_len" => (!this.backend.is_null())
                .then(|| unsafe { (&*this.backend).mapped.as_ref() })
                .flatten()
                .map_or(0, |m| m.len() as i64),
            "_pos" => this.pos,
            "_access" => this.access,
            "_mode" => this.mode,
            #[cfg(unix)]
            "_fd" => (!this.backend.is_null())
                .then(|| unsafe { (&*this.backend).fd.as_ref().map(|fd| fd.as_raw() as i64) })
                .flatten()
                .unwrap_or(-1),
            #[cfg(windows)]
            "_handle" => (!this.backend.is_null())
                .then(|| unsafe { (&*this.backend).handle as isize as i64 })
                .unwrap_or(host_mmap::INVALID_HANDLE as isize as i64),
            "_offset" => this.offset,
            "_exports" => this.exports,
            _ => 0,
        };
    }
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

#[cfg(any(unix, windows))]
fn mmap_set_attr(obj: pyre_object::PyObjectRef, key: &str, v: pyre_object::PyObjectRef) {
    if let Some(this) = W_MMap::from_obj(obj) {
        let value = if unsafe { pyre_object::is_int(v) } {
            unsafe { pyre_object::w_int_get_value(v) }
        } else {
            0
        };
        match key {
            "_pos" => this.pos = value,
            "_access" => this.access = value,
            "_mode" => this.mode = value,
            "_offset" => this.offset = value,
            "_exports" => this.exports = value,
            _ => {}
        }
        return;
    }
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
    let mapped = mmap_native(obj)?
        .mapped
        .as_ref()
        .ok_or_else(|| crate::PyError::value_error("mmap closed or invalid"))?;
    Ok((mapped.as_ptr() as *mut u8, mapped.len()))
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

/// `rmmap.py MMap.file_size` — the backing file's current size, which
/// diverges from the mapped length after `resize()`.  An anonymous map has no
/// file to stat, and fstat on the `-1` descriptor is the OSError that reports
/// it.
#[cfg(unix)]
fn mmap_file_size(obj: pyre_object::PyObjectRef) -> Result<i64, crate::PyError> {
    let fd = mmap_get_attr_i64(obj, "_fd") as libc::c_int;
    if fd < 0 {
        return Err(crate::PyError::os_error_with_errno(
            libc::EBADF,
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

/// Integer unwrapping may run arbitrary `__index__`; PyPy checks the mapping
/// again afterwards because that callback may have closed it (gh-103987).
#[cfg(any(unix, windows))]
fn mmap_index_w(
    obj: pyre_object::PyObjectRef,
    value: pyre_object::PyObjectRef,
) -> Result<i64, crate::PyError> {
    let index = crate::baseobjspace::int_w(crate::baseobjspace::space_index(value)?)?;
    let _ = mmap_native(obj)?;
    Ok(index)
}

/// Sweep-time counterpart of PyPy `W_MMap.__del__` / `rmmap.MMap.close`.
#[cfg(any(unix, windows))]
pub unsafe fn w_mmap_dealloc(obj: pyre_object::PyObjectRef) {
    mmap_close_native(obj);
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
    // PyPy's buffer exporters keep `_exports` as an untranslated Signed
    // field and mutate it directly (`bytearrayobject.py:68-80`).  In
    // particular releasebuffer can run from a GC finalizer, where allocating
    // a boxed Python integer would recursively enter the collector.
    if let Some(this) = W_MMap::from_obj(obj) {
        this.exports += 1;
    }
}

/// Paired with [`mmap_exports_incref`]; saturates at zero so a double release
/// cannot wrap the count and strand the mapping.
#[cfg(any(unix, windows))]
pub(crate) unsafe fn mmap_exports_decref(obj: pyre_object::PyObjectRef) {
    // Keep this allocation-free: memoryview destruction calls it while the
    // collector is already running.  This is the same raw-field decrement as
    // PyPy `W_BytearrayObject.bf_releasebuffer` / `ByteBuffer.releasebuffer`.
    if let Some(this) = W_MMap::from_obj(obj) {
        this.exports = (this.exports - 1).max(0);
    }
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

// `interp_mmap.py descr_iter` / `:256 descr_reversed` return a
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
    // `interp_mmap.py __new__ = interp2app(mmap)` — the class call
    // `mmap.mmap(fileno, length, ...)` lands here.  The common builtin
    // argument binder below supplies PyPy's named/defaulted gateway shape.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__new__",
        crate::typedef::make_new_descr(|args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "mmap() requires fileno + length",
                ));
            }
            mmap_construct(args[0], &args[1..])
        }),
    ) };

    // close() — munmap and zero the pointer.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let p = mmap_get_attr_i64(obj, "_ptr") as usize;
                if p != 0 {
                    mmap_check_exports(obj, "cannot close exported pointers exist")?;
                    mmap_close_native(obj);
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    ) };

    // `interp_mmap.py closed = GetSetProperty(W_MMap.closed_get)` —
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

    // `interp_mmap.py descr_size` returns `mmap.file_size()` —
    // the underlying file's current size via fstat, not the mapped
    // length.  The two diverge after `resize()`, and an anonymous mmap
    // (no fd) raises ValueError per rmmap.py:MMap.file_size.
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

    // CPython's mmap object advertises the seekable stream capability.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "seekable",
        crate::make_builtin_function_with_arity(
            "seekable",
            |_args| Ok(pyre_object::w_bool_from(true)),
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
            Ok(pyre_object::w_int_new(new_pos))
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
            // `interp_mmap.py read(num=-1)` — None or -1 reads to
            // end; positive value caps at remaining bytes.
            let requested = if args.len() >= 2 && !unsafe { pyre_object::is_none(args[1]) } {
                Some(mmap_index_w(obj, args[1])?)
            } else {
                None
            };
            let (p, len) = mmap_ptr(obj)?;
            let pos = mmap_get_attr_i64(obj, "_pos") as usize;
            let remaining = len.saturating_sub(pos);
            let n = if let Some(req) = requested {
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

    // `interp_mmap.py readline` — read bytes from current pos until
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
                let (p, len) = mmap_ptr(obj)?;
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
                let access = mmap_get_attr_i64(obj, "_access");
                if access == MMAP_ACCESS_READ {
                    return Err(crate::PyError::type_error("mmap is read-only"));
                }
                // `interp_mmap.py write_byte(byte=int)` —
                // `@unwrap_spec(byte=int)` rejects non-ints, then
                // `chr(byte)` raises ValueError on values outside
                // 0..256.
                let raw = mmap_index_w(obj, args[1])?;
                if !(0..=255).contains(&raw) {
                    return Err(crate::PyError::value_error(
                        "byte must be in range(0, 256)",
                    ));
                }
                let (p, len) = mmap_ptr(obj)?;
                let pos = mmap_get_attr_i64(obj, "_pos") as usize;
                if pos >= len {
                    return Err(crate::PyError::value_error("write_byte out of range"));
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
        // `interp_mmap.py flush(offset=0, size=0)` —
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
                return Err(crate::PyError::value_error("flush values out of range"));
            }
            let off = off_raw as usize;
            let raw_size = raw_size_raw as usize;
            if off > len {
                return Err(crate::PyError::value_error("flush values out of range"));
            }
            let n = if raw_size == 0 { len - off } else { raw_size };
            if off.checked_add(n).map(|s| s > len).unwrap_or(true) {
                return Err(crate::PyError::value_error("flush values out of range"));
            }
            let _ = p;
            mmap_flush(obj, off, n)
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
            // `interp_mmap.py find(w_tofind, w_start=None,
            // w_end=None)` defaults w_start to `self.mmap.pos` then
            // routes through rmmap.find which handles negative start /
            // end by adding `size` and clamping to 0.
            let cur = mmap_get_attr_i64(obj, "_pos") as usize;
            let start = if args.len() >= 3 {
                let s = mmap_index_w(obj, args[2])?;
                if s < 0 {
                    ((s + len as i64).max(0)) as usize
                } else {
                    (s as usize).min(len)
                }
            } else {
                cur
            };
            let end = if args.len() >= 4 {
                let e = mmap_index_w(obj, args[3])?;
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
            let hay = unsafe { std::slice::from_raw_parts(p.add(start), end - start) };
            if needle.len() > hay.len() {
                return Ok(pyre_object::w_int_new(-1));
            }
            let pos = (0..=hay.len().saturating_sub(needle.len()))
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
            // `interp_mmap.py rfind(w_tofind, w_start=None,
            // w_end=None)` defaults w_start to `self.mmap.pos`, not 0.
            // Negative args run through rmmap.find which adds `size`
            // and clamps to 0.
            let cur = mmap_get_attr_i64(obj, "_pos") as usize;
            let start = if args.len() >= 3 {
                let s = mmap_index_w(obj, args[2])?;
                if s < 0 {
                    ((s + len as i64).max(0)) as usize
                } else {
                    (s as usize).min(len)
                }
            } else {
                cur
            };
            let end = if args.len() >= 4 {
                let e = mmap_index_w(obj, args[3])?;
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
            let hay = unsafe { std::slice::from_raw_parts(p.add(start), end - start) };
            if needle.len() > hay.len() {
                return Ok(pyre_object::w_int_new(-1));
            }
            let pos = (0..=hay.len().saturating_sub(needle.len()))
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
                let p = mmap_get_attr_i64(obj, "_ptr") as usize;
                if p != 0 {
                    mmap_check_exports(obj, "cannot close exported pointers exist")?;
                    mmap_close_native(obj);
                }
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

    // `interp_mmap.py descr_getitem` — integer index returns a
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
                let (_, len) = mmap_ptr(obj)?;
                let len_i64 = len as i64;
                if unsafe { pyre_object::is_slice(index) } {
                    let (start, stop, step) =
                        unsafe { crate::baseobjspace::normalize_slice(index, len_i64)? };
                    let (p, _) = mmap_ptr(obj)?;
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
                        let Some(next) = i.checked_add(step) else {
                            break;
                        };
                        i = next;
                    }
                    return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&out));
                }
                let mut idx = mmap_index_w(obj, index)?;
                let (p, _) = mmap_ptr(obj)?;
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

    // `interp_mmap.py descr_setitem` — integer index writes a
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
                let (_, len) = mmap_ptr(obj)?;
                let len_i64 = len as i64;
                if unsafe { pyre_object::is_slice(index) } {
                    let (start, stop, step) =
                        unsafe { crate::baseobjspace::normalize_slice(index, len_i64)? };
                    let (p, _) = mmap_ptr(obj)?;
                    let length = if step > 0 && stop > start {
                        (1 + (i128::from(stop) - i128::from(start) - 1) / i128::from(step)) as i64
                    } else if step < 0 && start > stop {
                        (1 + (i128::from(start) - i128::from(stop) - 1) / -i128::from(step)) as i64
                    } else {
                        0
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
                            let Some(next) = i.checked_add(step) else {
                                break;
                            };
                            i = next;
                            k += 1;
                        }
                    }
                    return Ok(pyre_object::w_none());
                }
                let mut idx = mmap_index_w(obj, index)?;
                let (p, _) = mmap_ptr(obj)?;
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

    // `interp_mmap.py descr_iter` — iterate the 1-byte slices
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

    // `interp_mmap.py descr_reversed` — iterate the 1-byte slices
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

    // `interp_mmap.py` — `madvise` is registered from `optional[]`,
    // i.e. only where `rmmap` has the call.  Windows has no counterpart, so
    // the method is absent there rather than present and inert.
    //
    // `interp_mmap.py:descr_madvise` — call madvise(addr+start, length,
    // advice).  Defaults: start=0, length=remaining bytes.
    #[cfg(unix)]
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "madvise",
        crate::make_builtin_function("madvise", |args| {
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let (_, total) = mmap_ptr(obj)?;
            if args.len() < 2 {
                return Err(crate::PyError::type_error("madvise() requires option"));
            }
            let option = mmap_index_w(obj, args[1])? as i32;
            let start_raw = match args.get(2) {
                Some(&value) => mmap_index_w(obj, value)?,
                None => 0,
            };
            if start_raw < 0 || usize::try_from(start_raw).map_or(true, |start| start >= total) {
                return Err(crate::PyError::value_error("madvise start out of bounds"));
            }
            let start = start_raw as usize;
            let length_raw = match args.get(3) {
                Some(&value) => mmap_index_w(obj, value)?,
                None => (total - start) as i64,
            };
            if length_raw < 0 {
                return Err(crate::PyError::value_error("madvise length invalid"));
            }
            if i128::from(length_raw) + start as i128 > isize::MAX as i128 {
                return Err(crate::PyError::overflow_error("madvise length too large"));
            }
            let length = (length_raw as usize).min(total - start);
            #[cfg(not(target_os = "redox"))]
            {
                mmap_madvise(obj, start, length, option)
                .map_err(|e| mmap_io_err(e, "madvise"))?;
            }
            #[cfg(target_os = "redox")]
            {
                let _ = (length, option);
            }
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
                // `interp_mmap.py move(dest, src, count)` —
                // `@unwrap_spec(dest=int, src=int, count=int)` plus
                // `self.check_writeable()` upfront.  We require all
                // three args use the index protocol and reject ACCESS_READ.
                if mmap_get_attr_i64(obj, "_access") == MMAP_ACCESS_READ {
                    return Err(crate::PyError::type_error("mmap is read-only"));
                }
                let dest = mmap_index_w(obj, args[1])? as usize;
                let src = mmap_index_w(obj, args[2])? as usize;
                let count = mmap_index_w(obj, args[3])? as usize;
                let (p, total) = mmap_ptr(obj)?;
                if dest.saturating_add(count) > total || src.saturating_add(count) > total {
                    return Err(crate::PyError::value_error(
                        "source or destination out of range",
                    ));
                }
                unsafe {
                    std::ptr::copy(p.add(src), p.add(dest), count);
                }
                Ok(pyre_object::w_none())
            },
            4,
        ),
    ) };

    // `interp_mmap.py resize` → `rmmap.py`.  POSIX path:
    // ftruncate the backing fd (if any) to `offset + newsize`, then
    // mremap(MREMAP_MAYMOVE).  Platforms without mremap (e.g. macOS)
    // raise SystemError to match PyPy's RValueError→SystemError
    // translation at `interp_mmap.py:155-157`.  Read-only / copy
    // mappings reject with TypeError.
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
                let newsize = newsize as usize;
                if !mmap_native(obj)?.trackfd {
                    return Err(crate::PyError::value_error(
                        "mmap can't resize with trackfd=False.",
                    ));
                }
                mmap_resize_mapping(obj, p, old_len, newsize)?;
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
                // `interp_mmap.py descr_repr`: closed mmaps
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

/// `rmmap.py:589-601` — ftruncate the backing fd (if any) to `offset +
/// newsize`, then remap.  host_env's `MappedFile` (memmap2) cannot mremap in
/// place, so the mapping is re-created at the new size.  A file-backed map is
/// re-mapped from the (ftruncated) fd, an anonymous map is remade and the
/// surviving bytes copied.  The new mapping may land at a different address
/// than an mremap would have, but that address is never exposed to Python.
///
/// Re-creating the mapping does change one observable, though: `mremap` refuses
/// to grow a shared anonymous mapping on Linux (kernel bug 8691), while a
/// remake succeeds.  The guard below restores the refusal.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn mmap_resize_mapping(
    obj: pyre_object::PyObjectRef,
    p: *mut u8,
    old_len: usize,
    newsize: usize,
) -> Result<(), crate::PyError> {
    let fd = mmap_get_attr_i64(obj, "_fd") as libc::c_int;
    let offset = mmap_get_attr_i64(obj, "_offset");
    // `mmapmodule.c mmap_resize_method`, the `#ifdef __linux__` arm ahead of
    // the ftruncate:
    //   if (self->fd == -1 && !(self->flags & MAP_PRIVATE) && new_size > self->size)
    //       ValueError("mmap: can't expand a shared anonymous mapping on Linux")
    if fd < 0
        && mmap_native(obj)?.flags & host_mmap::MAP_PRIVATE == 0
        && newsize > old_len
    {
        return Err(crate::PyError::value_error(
            "mmap: can't expand a shared anonymous mapping on Linux",
        ));
    }
    let mapped = if fd >= 0 {
        let r = unsafe { libc::ftruncate(fd, (offset as libc::off_t) + newsize as libc::off_t) };
        if r != 0 {
            return Err(crate::PyError::os_error_with_errno(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                "ftruncate",
            ));
        }
        let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::borrow_raw(fd) };
        // `mremap` keeps the mapping's protection, so remap with the one the
        // original mapping resolved to (`rmmap.py:729-745`); `_access` alone
        // does not preserve a PROT_READ map.
        let mode = mmap_access_mode(mmap_get_attr_i64(obj, "_mode"));
        let (_, mapped) = host_mmap::map_file(borrowed, offset, newsize, mode)
            .map_err(|e| mmap_io_err(e, "mmap"))?;
        mapped
    } else {
        mmap_remake_anon(p, old_len, newsize)?
    };
    mmap_native(obj)?.mapped = Some(MappedObj::Mapped(mapped));
    Ok(())
}

/// `rmmap.py` — unmap the view and close the mapping object, move the
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
    let tagname = mmap_native(obj)?.tagname.clone();
    let named = !tagname.is_empty();
    let handle = mmap_handle(obj);
    let offset = mmap_get_attr_i64(obj, "_offset");
    let access = mmap_access_mode(mmap_get_attr_i64(obj, "_mode"));
    // A tagged mapping with no file behind it is pagefile-backed: there is no
    // EOF to move, so the bytes that survive the resize are carried by hand.
    let old_bytes = (named && handle.is_none())
        .then(|| unsafe { std::slice::from_raw_parts(p, old_len) }.to_vec());

    let mapped = if let Some(handle) = handle {
        // The view has to go before the file can grow, so `_ptr` reads 0 from
        // here until one of the two remaps below puts a mapping back.
        mmap_native(obj)?.mapped = None;
        let remap = |size: usize, reject_existing: bool| {
            mmap_remap_named(handle, offset, access, size, &tagname, reject_existing)
        };
        match host_mmap::extend_file(handle, offset + newsize as i64)
            .and_then(|()| remap(newsize, true))
        {
            Ok(mapped) => mapped,
            Err(e) => {
                // Put the mapping back at its old size; if even that fails the
                // object stays mapping-less, which reads as closed.
                if let Ok(old) = remap(old_len, false) {
                    mmap_native(obj)?.mapped = Some(old);
                }
                return Err(mmap_io_err(e, "mmap"));
            }
        }
    } else if named {
        // A pagefile-backed tagged mapping has no file EOF to move, but the
        // same close/recreate sequence applies.  ERROR_ALREADY_EXISTS comes
        // from `CreateFileMapping` when another live view still owns the name.
        mmap_native(obj)?.mapped = None;
        let remap = |size: usize, reject_existing: bool| {
            mmap_remap_named(
                host_mmap::INVALID_HANDLE,
                offset,
                access,
                size,
                &tagname,
                reject_existing,
            )
        };
        match remap(newsize, true) {
            Ok(mut mapped) => {
                let old = old_bytes.as_deref().unwrap_or_default();
                let keep = old.len().min(newsize);
                mapped.as_mut_slice()[..keep].copy_from_slice(&old[..keep]);
                mapped
            }
            Err(e) => {
                if let Ok(mut old_mapping) = remap(old_len, false) {
                    if let Some(old) = old_bytes.as_deref() {
                        old_mapping.as_mut_slice().copy_from_slice(old);
                    }
                    mmap_native(obj)?.mapped = Some(old_mapping);
                }
                return Err(mmap_io_err(e, "CreateFileMapping"));
            }
        }
    } else {
        MappedObj::Mapped(mmap_remake_anon(p, old_len, newsize)?)
    };
    mmap_native(obj)?.mapped = Some(mapped);
    Ok(())
}

/// Recreate the Windows mapping object under the name it already had, matching
/// `rmmap.py:635-636`, which passes `self.tagname` whether or not one was
/// given.  `CreateFileMappingW` reports ERROR_ALREADY_EXISTS when another
/// mapping still owns the name; a resize surfaces that OS error, while the
/// recovery remap is allowed to reopen the existing object.  An untagged
/// mapping cannot reach it — a zero-length name is not a name.
#[cfg(windows)]
fn mmap_remap_named(
    handle: host_mmap::Handle,
    offset: i64,
    access: host_mmap::AccessMode,
    size: usize,
    tagname: &str,
    reject_existing: bool,
) -> std::io::Result<MappedObj> {
    let named = host_mmap::create_named_mapping(handle, tagname, access, offset, size)?;
    const ERROR_ALREADY_EXISTS: i32 = 183;
    if reject_existing && host_mmap::last_error() as i32 == ERROR_ALREADY_EXISTS {
        drop(named);
        return Err(std::io::Error::from_raw_os_error(ERROR_ALREADY_EXISTS));
    }
    Ok(MappedObj::Named(named))
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
#[allow(dead_code)]
fn mmap_access_mode(access: i64) -> host_mmap::AccessMode {
    match access {
        x if x == MMAP_ACCESS_READ => host_mmap::AccessMode::Read,
        x if x == MMAP_ACCESS_WRITE => host_mmap::AccessMode::Write,
        x if x == MMAP_ACCESS_COPY => host_mmap::AccessMode::Copy,
        _ => host_mmap::AccessMode::Default,
    }
}

#[cfg(any(unix, windows))]
const MMAP_ACCESS_DEFAULT: i64 = 0;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_READ: i64 = 1;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_WRITE: i64 = 2;
#[cfg(any(unix, windows))]
const MMAP_ACCESS_COPY: i64 = 3;

/// `interp_mmap.py W_MMap.__init__` — hand the finished mapping to a
/// fresh instance of `cls`.  `mmap` is an acceptable base class, so the object
/// carries the mapdict prefix and the subclass tag rather than being allocated
/// against the builtin type directly.
#[cfg(any(unix, windows))]
fn mmap_new_object(
    cls: pyre_object::PyObjectRef,
    backend: NativeMMap,
    access: i64,
    mode: host_mmap::AccessMode,
    offset: i64,
) -> pyre_object::PyObjectRef {
    let backend = Box::into_raw(Box::new(backend));
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let obj = W_MMap::allocate_stable(W_MMap {
        ob: pyre_object::PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        map: 0,
        storage: std::ptr::null_mut(),
        backend,
        pos: 0,
        access,
        mode: mode as i64,
        offset,
        exports: 0,
    });
    crate::typedef::tag_subclass_instance(obj, unsafe {
        pyre_object::gc_roots::shadow_stack_get(cls_slot)
    })
}

// `interp_mmap.py:55-130 mmap_new` / CPython 3.14's `trackfd` addition.
#[cfg(unix)]
fn mmap_construct(
    cls: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let bound = crate::builtins::bind_builtin_kwargs(
        args,
        &["fileno", "length", "flags", "prot", "access", "offset", "trackfd"],
        &[true, true, false, false, false, false, false],
        "mmap",
    )?;
    let index_i64 = |obj: pyre_object::PyObjectRef, label: &str| {
        if obj.is_null() {
            return Err(crate::PyError::type_error(format!(
                "mmap() {label} must be an integer"
            )));
        }
        crate::baseobjspace::int_w(crate::baseobjspace::space_index(obj)?)
    };
    let fd_raw = index_i64(bound[0], "fileno")?;
    let fd = libc::c_int::try_from(fd_raw)
        .map_err(|_| crate::PyError::overflow_error("Python int too large to convert to C int"))?;
    let length_raw = index_i64(bound[1], "length")?;
    let flags_arg = if bound[2].is_null() {
        host_mmap::MAP_SHARED
    } else {
        index_i64(bound[2], "flags")? as libc::c_int
    };
    let prot_arg = if bound[3].is_null() {
        host_mmap::PROT_READ | host_mmap::PROT_WRITE
    } else {
        index_i64(bound[3], "prot")? as libc::c_int
    };
    let mut access = if bound[4].is_null() {
        MMAP_ACCESS_DEFAULT
    } else {
        index_i64(bound[4], "access")?
    };
    let offset = if bound[5].is_null() {
        0
    } else {
        index_i64(bound[5], "offset")?
    };
    let trackfd = if bound[6].is_null() {
        true
    } else {
        crate::baseobjspace::is_true(bound[6])?
    };
    // `rmmap.py:718-727` performs these guards in this order, and before the
    // size_t cast.
    if access != MMAP_ACCESS_DEFAULT
        && (flags_arg != host_mmap::MAP_SHARED
            || prot_arg != host_mmap::PROT_READ | host_mmap::PROT_WRITE)
    {
        return Err(crate::PyError::value_error(
            "mmap can't specify both access and flags, prot.",
        ));
    }
    // `rmmap.py _check_map_size`.
    if length_raw < 0 {
        return Err(crate::PyError::type_error(
            "memory mapped size must be positive",
        ));
    }
    if offset < 0 {
        return Err(crate::PyError::value_error("negative offset"));
    }
    let mut length = usize::try_from(length_raw)
        .map_err(|_| crate::PyError::overflow_error("memory mapped length must be positive"))?;
    let (flags, prot) = match access {
        x if x == MMAP_ACCESS_READ => (host_mmap::MAP_SHARED, host_mmap::PROT_READ),
        x if x == MMAP_ACCESS_WRITE => {
            (host_mmap::MAP_SHARED, host_mmap::PROT_READ | host_mmap::PROT_WRITE)
        }
        x if x == MMAP_ACCESS_COPY => {
            (host_mmap::MAP_PRIVATE, host_mmap::PROT_READ | host_mmap::PROT_WRITE)
        }
        x if x == MMAP_ACCESS_DEFAULT => {
            if prot_arg & host_mmap::PROT_WRITE != 0
                && prot_arg & host_mmap::PROT_READ == 0
            {
                access = MMAP_ACCESS_WRITE;
            } else if prot_arg & host_mmap::PROT_WRITE == 0 {
                access = MMAP_ACCESS_READ;
            }
            (flags_arg, prot_arg)
        }
        _ => return Err(crate::PyError::value_error("mmap invalid access parameter.")),
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
    if real_fd != -1 {
        let mut st: libc::stat = unsafe { core::mem::zeroed() };
        if unsafe { libc::fstat(real_fd, &mut st) } == 0
            && st.st_mode & libc::S_IFMT == libc::S_IFREG
        {
            let file_size = usize::try_from(st.st_size).unwrap_or(usize::MAX);
            let offset_usize = usize::try_from(offset).unwrap_or(usize::MAX);
            if length == 0 {
                if file_size == 0 {
                    return Err(crate::PyError::value_error("cannot mmap an empty file"));
                }
                // `rmmap.py:757-761` rejects only offset > size, which
                // leaves an offset landing exactly on EOF asking `mmap(2)` for
                // a zero-length mapping — EINVAL rather than the ValueError the
                // caller is told to expect.  `mmapmodule.c new_mmap_object`
                // refuses it as `if (offset >= status.st_size)` (`:1872`, and
                // `:2087` for the Windows block) — read at v3.14.6 in the
                // checkout at Z:/cpython; not executable on the Windows host
                // this was written on.
                if offset_usize >= file_size {
                    return Err(crate::PyError::value_error(
                        "mmap offset is greater than file size",
                    ));
                }
                length = file_size - offset_usize;
            } else if offset_usize.saturating_add(length) > file_size {
                return Err(crate::PyError::value_error(
                    "mmap length is greater than file size",
                ));
            }
        }
    }
    let (mapped, owned_fd) = if real_fd == -1 {
        (
            host_mmap::map_anon(length).map_err(|e| mmap_io_err(e, "mmap"))?,
            None,
        )
    } else {
        let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::borrow_raw(real_fd) };
        let (dup_fd, mapped) = host_mmap::map_file(borrowed, offset, length, mode)
            .map_err(|e| mmap_io_err(e, "mmap"))?;
        (mapped, trackfd.then_some(dup_fd))
    };
    Ok(mmap_new_object(
        cls,
        NativeMMap {
            mapped: Some(MappedObj::Mapped(mapped)),
            fd: owned_fd,
            flags,
            trackfd,
        },
        access,
        mode,
        offset,
    ))
}

// `interp_mmap.py mmap(fileno, length, tagname, access, offset)` —
// the Windows constructor names a mapping object where POSIX passes
// flags/prot, and the mapping's protection comes from `access` alone
// (`rmmap.py:900-914`).
#[cfg(windows)]
fn mmap_construct(
    cls: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let bound = crate::builtins::bind_builtin_kwargs(
        args,
        &["fileno", "length", "tagname", "access", "offset"],
        &[true, true, false, false, false],
        "mmap",
    )?;
    let index_i64 = |obj: pyre_object::PyObjectRef, label: &str| {
        if obj.is_null() {
            return Err(crate::PyError::type_error(format!(
                "mmap() {label} must be an integer"
            )));
        }
        crate::baseobjspace::int_w(crate::baseobjspace::space_index(obj)?)
    };
    let fileno = index_i64(bound[0], "fileno")? as i32;
    let length = index_i64(bound[1], "length")?;
    // `rmmap.py _check_map_size`.
    if length < 0 {
        return Err(crate::PyError::type_error(
            "memory mapped size must be positive",
        ));
    }
    let tagname = match bound[2] {
        a if a.is_null() || unsafe { pyre_object::is_none(a) } => "",
        a => {
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
    };
    let access = if bound[3].is_null() {
        MMAP_ACCESS_DEFAULT
    } else {
        index_i64(bound[3], "access")?
    };
    if !(MMAP_ACCESS_DEFAULT..=MMAP_ACCESS_COPY).contains(&access) {
        return Err(crate::PyError::value_error("mmap invalid access parameter."));
    }
    let offset = if bound[4].is_null() {
        0
    } else {
        index_i64(bound[4], "offset")?
    };
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
        // `rmmap.py` trusts map_size when the size cannot be read
        // (a non-seeking file); only a readable size constrains it.
        if let Ok(file_len) = host_mmap::get_file_len(fh) {
            if map_size == 0 {
                if file_len == 0 {
                    return Err(crate::PyError::value_error("cannot mmap an empty file"));
                }
                // `rmmap.py:943-945` rejects only offset > size, so an
                // offset landing exactly on EOF resolves to a zero-length
                // mapping and asks `MapViewOfFile` for an empty view, which
                // fails with a Win32 error instead.  `mmapmodule.c
                // new_mmap_object` refuses the offset one step earlier, `if
                // (offset >= size)`, in its Windows size block (`:2087`; the
                // POSIX one spells the same test at `:1872`) — read at v3.14.6
                // in the checkout at Z:/cpython.
                if offset >= file_len {
                    return Err(crate::PyError::value_error(
                        "mmap offset is greater than file size",
                    ));
                }
                let remaining = file_len - offset;
                map_size = remaining as usize;
                if map_size as i64 != remaining {
                    return Err(crate::PyError::value_error("mmap length is too large"));
                }
            }
            // A given length is not measured against the file at all here.
            // `CreateFileMapping` is asked for `offset + length` and grows the
            // file to it, so a length past EOF is how the mapping is extended
            // on this platform.  `rmmap.py` rejects it with "mmap
            // length is greater than file size" instead; the run fails on that
            // at `lib-python/3/test/test_mmap.py:202-209`, which calls
            // `mmap(fileno(), mapsize + 1)` and, on `sys.platform` starting
            // with "win", turns a ValueError into
            // `self.fail("Opening mmap with size+1 should work on Windows.")`.
            // `pypy/module/mmap/test/test_mmap.py:747-749` concedes the same
            // point from the other side, skipping its own
            // `raises(ValueError, ...)` under `os.name == "nt"` with the note
            // "this should work under windows".
        }
        file_handle = Some(guard);
    }

    let handle = file_handle
        .as_ref()
        .map_or(host_mmap::INVALID_HANDLE, |guard| guard.0);
    // `rmmap.py:998-1000` reaches every file-backed mapping through one
    // `CreateFileMapping(m.file_handle, NULL, flProtect, size_hi, size_lo,
    // m.tagname)`, where the maximum size is `offset + map_size` and the name
    // is whatever `tagname` holds — the empty string when none was given.  A
    // zero-length name is not a name: two mappings created with one do not
    // report ERROR_ALREADY_EXISTS to each other, so the untagged mapping needs
    // no second route.  memmap2 is the wrong one to give it, because it asks
    // for a maximum size of 0, which means "the file as it stands" and cannot
    // extend it, and because it rounds the offset down to the allocation
    // granularity instead of letting `MapViewOfFile` reject an unaligned one.
    let (mapped, owned_handle) = if file_handle.is_some() || !tagname.is_empty() {
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
    } else {
        let mapped = host_mmap::map_anon(map_size).map_err(|e| mmap_io_err(e, "mmap"))?;
        (MappedObj::Mapped(mapped), host_mmap::INVALID_HANDLE)
    };
    Ok(mmap_new_object(
        cls,
        NativeMMap {
            mapped: Some(mapped),
            handle: owned_handle,
            tagname: tagname.to_owned(),
            // No `trackfd` parameter on Windows; the object always owns the
            // handle it duplicated, so `resize()` is never disarmed.
            trackfd: true,
        },
        access,
        mmap_access_mode(access),
        offset,
    ))
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

/// `rmmap.py` — the mapping flags and advice values a POSIX `mmap(2)`
/// takes.  The portable subset sources from host_env's re-exports; the
/// platform-specific extras it does not re-export (MAP_FIXED, the Linux-only
/// MAP_* flags, PROT_NONE) stay on libc.  Windows has none of them: the
/// mapping's protection comes from `access` alone there, and its module
/// carries only the ACCESS_* and page constants.
#[cfg(unix)]
fn register_posix_constants(ns: pyre_object::PyObjectRef) {
    {
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
}
