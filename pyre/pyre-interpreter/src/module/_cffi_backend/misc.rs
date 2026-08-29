//! Raw-memory accessors and the object-to-C-integer conversions —
//! PyPy: `pypy/module/_cffi_backend/misc.py`.
//!
//! `misc.py` selects the C type to punch through by matching `size` against
//! `rffi.sizeof(TP)` over an unrolled list.  The same match is spelled here as
//! an ordinary `match` on the byte width, which is what those `rffi.sizeof`
//! constants evaluate to.

use crate::PyError;
use pyre_object::PyObjectRef;
use std::ffi::{c_char, c_double, c_int};

/// `misc.py`'s `long double` shims.  Every one of them is C because Rust has
/// no `long double` type at all.
unsafe extern "C" {
    pub fn pyre_cffi_sizeof_long_double() -> usize;
    pub fn pyre_cffi_alignof_long_double() -> usize;
    pub fn pyre_cffi_sizeof_wchar() -> usize;
    pub fn pyre_cffi_alignof_wchar() -> usize;
    pub fn pyre_cffi_sizeof_fast8() -> usize;
    pub fn pyre_cffi_alignof_fast8() -> usize;
    pub fn pyre_cffi_sizeof_fast16() -> usize;
    pub fn pyre_cffi_alignof_fast16() -> usize;
    pub fn pyre_cffi_sizeof_fast32() -> usize;
    pub fn pyre_cffi_alignof_fast32() -> usize;
    pub fn pyre_cffi_sizeof_fast64() -> usize;
    pub fn pyre_cffi_alignof_fast64() -> usize;
    pub fn pyre_cffi_read_long_double(p: *const c_char) -> c_double;
    pub fn pyre_cffi_write_long_double(p: *mut c_char, v: c_double);
    pub fn pyre_cffi_nonnull_long_double(p: *const c_char) -> c_int;
    pub fn pyre_cffi_str_long_double(p: *const c_char, out: *mut c_char, n: usize);
}

/// `sizeof(long double)`.
pub fn sizeof_long_double() -> i64 {
    unsafe { pyre_cffi_sizeof_long_double() as i64 }
}

/// `newtype.py alignment(rffi.LONGDOUBLE)`.
pub fn alignof_long_double() -> i64 {
    unsafe { pyre_cffi_alignof_long_double() as i64 }
}

/// The size and alignment of `wchar_t`, which is two bytes on Windows and
/// four where it is an `int`.
pub fn wchar_layout() -> (i64, i64) {
    unsafe {
        (
            pyre_cffi_sizeof_wchar() as i64,
            pyre_cffi_alignof_wchar() as i64,
        )
    }
}

/// The size and alignment of `int_fast<bits>_t`, which is a different C type
/// on every platform.  `uint_fast<bits>_t` shares the layout.
pub fn fast_int_layout(bits: u32) -> (i64, i64) {
    let (size, align) = unsafe {
        match bits {
            8 => (pyre_cffi_sizeof_fast8(), pyre_cffi_alignof_fast8()),
            16 => (pyre_cffi_sizeof_fast16(), pyre_cffi_alignof_fast16()),
            32 => (pyre_cffi_sizeof_fast32(), pyre_cffi_alignof_fast32()),
            _ => (pyre_cffi_sizeof_fast64(), pyre_cffi_alignof_fast64()),
        }
    };
    (size as i64, align as i64)
}

// ── raw reads and writes ────────────────────────────────────────────────

/// `misc.py read_raw_signed_data`.
///
/// # Safety
/// `target` must point at `size` readable bytes.
pub unsafe fn read_raw_signed_data(target: *const u8, size: i64) -> Result<i64, PyError> {
    unsafe {
        Ok(match size {
            1 => target.cast::<i8>().read_unaligned() as i64,
            2 => target.cast::<i16>().read_unaligned() as i64,
            4 => target.cast::<i32>().read_unaligned() as i64,
            8 => target.cast::<i64>().read_unaligned(),
            _ => return Err(bad_integer_size()),
        })
    }
}

/// `misc.py read_raw_unsigned_data`.
///
/// # Safety
/// `target` must point at `size` readable bytes.
pub unsafe fn read_raw_unsigned_data(target: *const u8, size: i64) -> Result<u64, PyError> {
    unsafe {
        Ok(match size {
            1 => u64::from(target.read()),
            2 => u64::from(target.cast::<u16>().read_unaligned()),
            4 => u64::from(target.cast::<u32>().read_unaligned()),
            8 => target.cast::<u64>().read_unaligned(),
            _ => return Err(bad_integer_size()),
        })
    }
}

/// `misc.py write_raw_signed_data`.
///
/// # Safety
/// `target` must point at `size` writable bytes.
pub unsafe fn write_raw_signed_data(
    target: *mut u8,
    source: i64,
    size: i64,
) -> Result<(), PyError> {
    unsafe {
        match size {
            1 => target.cast::<i8>().write_unaligned(source as i8),
            2 => target.cast::<i16>().write_unaligned(source as i16),
            4 => target.cast::<i32>().write_unaligned(source as i32),
            8 => target.cast::<i64>().write_unaligned(source),
            _ => return Err(bad_integer_size()),
        }
    }
    Ok(())
}

/// `misc.py write_raw_unsigned_data`.
///
/// # Safety
/// `target` must point at `size` writable bytes.
pub unsafe fn write_raw_unsigned_data(
    target: *mut u8,
    source: u64,
    size: i64,
) -> Result<(), PyError> {
    unsafe {
        match size {
            1 => target.write(source as u8),
            2 => target.cast::<u16>().write_unaligned(source as u16),
            4 => target.cast::<u32>().write_unaligned(source as u32),
            8 => target.cast::<u64>().write_unaligned(source),
            _ => return Err(bad_integer_size()),
        }
    }
    Ok(())
}

/// `misc.py read_raw_float_data`.
///
/// # Safety
/// `target` must point at `size` readable bytes.
pub unsafe fn read_raw_float_data(target: *const u8, size: i64) -> Result<f64, PyError> {
    unsafe {
        Ok(match size {
            4 => f64::from(target.cast::<f32>().read_unaligned()),
            8 => target.cast::<f64>().read_unaligned(),
            _ => return Err(bad_float_size()),
        })
    }
}

/// `misc.py write_raw_float_data`.
///
/// # Safety
/// `target` must point at `size` writable bytes.
pub unsafe fn write_raw_float_data(target: *mut u8, source: f64, size: i64) -> Result<(), PyError> {
    unsafe {
        match size {
            4 => target.cast::<f32>().write_unaligned(source as f32),
            8 => target.cast::<f64>().write_unaligned(source),
            _ => return Err(bad_float_size()),
        }
    }
    Ok(())
}

/// `misc.py read_raw_longdouble_data` narrowed to a `double`.
///
/// # Safety
/// `target` must point at `sizeof(long double)` readable bytes.
pub unsafe fn read_raw_longdouble_data(target: *const u8) -> f64 {
    unsafe { pyre_cffi_read_long_double(target.cast::<c_char>()) }
}

/// `misc.py write_raw_longdouble_data` widened from a `double`.
///
/// # Safety
/// `target` must point at `sizeof(long double)` writable bytes.
pub unsafe fn write_raw_longdouble_data(target: *mut u8, source: f64) {
    unsafe { pyre_cffi_write_long_double(target.cast::<c_char>(), source) }
}

/// `misc.py is_nonnull_longdouble`.  The comparison happens in the
/// `long double` domain, so it cannot be done on a narrowed copy.
///
/// # Safety
/// `target` must point at `sizeof(long double)` readable bytes.
pub unsafe fn is_nonnull_longdouble(target: *const u8) -> bool {
    unsafe { pyre_cffi_nonnull_long_double(target.cast::<c_char>()) != 0 }
}

/// `misc.py is_nonnull_float`.  True for a NaN, as the comment there notes.
///
/// # Safety
/// `target` must point at `size` readable bytes.
pub unsafe fn is_nonnull_float(target: *const u8, size: i64) -> Result<bool, PyError> {
    Ok(unsafe { read_raw_float_data(target, size)? } != 0.0)
}

/// `misc.py longdouble2str`.
///
/// # Safety
/// `target` must point at `sizeof(long double)` readable bytes.
pub unsafe fn longdouble2str(target: *const u8) -> String {
    // `%LE` of any representation C can name fits well inside this.
    let mut buffer = [0u8; 128];
    unsafe {
        pyre_cffi_str_long_double(
            target.cast::<c_char>(),
            buffer.as_mut_ptr().cast::<c_char>(),
            buffer.len(),
        );
    }
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

/// `misc.py signext` — sign-extend the low `size` bytes of `value`.
pub fn signext(value: i64, size: i64) -> i64 {
    match size {
        1 => value as i8 as i64,
        2 => value as i16 as i64,
        4 => value as i32 as i64,
        _ => value,
    }
}

fn bad_integer_size() -> PyError {
    PyError::new(crate::PyErrorKind::NotImplementedError, "bad integer size")
}

fn bad_float_size() -> PyError {
    PyError::new(crate::PyErrorKind::NotImplementedError, "bad float size")
}

// ── object → C integer ──────────────────────────────────────────────────

pub const NEG_MSG: &str = "can't convert negative number to unsigned";
pub const OVF_MSG: &str = "long too big to convert";

/// `misc.py _is_a_float`.
fn is_a_float(w_ob: PyObjectRef) -> bool {
    if let Some(cdata) = super::cdataobj::W_CData::from_obj(w_ob) {
        return super::ctypeobj::ctype_of(cdata).is_some_and(|ct| ct.is_float_family());
    }
    unsafe { pyre_object::pyobject::is_float(w_ob) }
}

/// True when `w_ob` is an `int` (or `bool`) whose value is already a machine
/// word — `space.int_w(w_ob, allow_conversion=False)`'s success case.
fn exact_int_w(w_ob: PyObjectRef) -> Option<Result<i64, PyError>> {
    unsafe {
        if pyre_object::pyobject::is_bool(w_ob) {
            return Some(Ok(i64::from(pyre_object::boolobject::w_bool_get_value(
                w_ob,
            ))));
        }
        if pyre_object::pyobject::is_int(w_ob) {
            return Some(Ok(pyre_object::intobject::w_int_get_value(w_ob)));
        }
        if pyre_object::pyobject::is_long(w_ob) {
            let big = pyre_object::longobject::w_long_get_value(w_ob);
            if pyre_object::longobject::jit_bigint_to_i64_fits(big) != 0 {
                return Some(Ok(pyre_object::longobject::jit_bigint_to_i64_value(big)));
            }
            return Some(Err(PyError::overflow_error(OVF_MSG)));
        }
    }
    None
}

/// `misc.py as_long`.  Accepts an `int`, and anything with `__index__`;
/// refuses a float.
pub fn as_long(w_ob: PyObjectRef) -> Result<i64, PyError> {
    match exact_int_w(w_ob) {
        Some(Ok(value)) => return Ok(value),
        // `as_long` lets an OverflowError through only after re-reading the
        // object with conversion allowed, which is where a bigint too large
        // for a word raises again.
        Some(Err(_)) => {}
        None => {
            if is_a_float(w_ob) {
                return Err(PyError::type_error("integer expected, got float"));
            }
        }
    }
    crate::baseobjspace::int_w(w_ob)
}

/// `misc.py as_long_long`.  Same as [`as_long`] on a 64-bit word.
pub fn as_long_long(w_ob: PyObjectRef) -> Result<i64, PyError> {
    as_long(w_ob)
}

/// `misc.py as_unsigned_long_long`.  `strict` reports an out-of-range value
/// as `OverflowError`; otherwise the value is masked and a float rounded
/// down, which is what an explicit `ffi.cast()` asks for.
pub fn as_unsigned_long_long(w_ob: PyObjectRef, strict: bool) -> Result<u64, PyError> {
    match exact_int_w(w_ob) {
        Some(Ok(value)) => {
            if strict && value < 0 {
                return Err(PyError::overflow_error(NEG_MSG));
            }
            return Ok(value as u64);
        }
        // A bigint wider than a word reaches the path below either way, which
        // reads it back through `space.int`.
        Some(Err(_)) => {}
        None => {
            if strict && is_a_float(w_ob) {
                return Err(PyError::type_error("integer expected, got float"));
            }
        }
    }
    let w_int = crate::baseobjspace::space_int(w_ob)?;
    if strict {
        // `toulonglong` signals a negative value as `ValueError` and a value
        // too wide for the word as `OverflowError`; both surface here as an
        // `OverflowError`, under their own messages.
        return crate::baseobjspace::uint_w(w_int).map_err(|err| {
            let message = if err.kind == crate::error::PyErrorKind::ValueError {
                NEG_MSG
            } else {
                OVF_MSG
            };
            PyError::overflow_error(message)
        });
    }
    Ok(bigint_mask_u64(w_int))
}

/// `misc.py as_unsigned_long`.  Identical to [`as_unsigned_long_long`] on a
/// target whose `unsigned long` is a machine word.
pub fn as_unsigned_long(w_ob: PyObjectRef, strict: bool) -> Result<u64, PyError> {
    as_unsigned_long_long(w_ob, strict)
}

/// `rbigint.ulonglongmask()` — the low 64 bits of an integer of any width.
fn bigint_mask_u64(w_int: PyObjectRef) -> u64 {
    unsafe {
        if pyre_object::pyobject::is_bool(w_int) {
            return u64::from(pyre_object::boolobject::w_bool_get_value(w_int));
        }
        if pyre_object::pyobject::is_int(w_int) {
            return pyre_object::intobject::w_int_get_value(w_int) as u64;
        }
        if pyre_object::pyobject::is_long(w_int) {
            return pyre_object::longobject::w_long_get_value(w_int).ulonglongmask();
        }
    }
    0
}

/// `misc.py object_as_bool` — an `int`, a `float`, or a cdata holding one.
pub fn object_as_bool(w_ob: PyObjectRef) -> Result<bool, PyError> {
    if let Some(value) = standard_object_as_bool(w_ob) {
        return Ok(value);
    }
    let is_cdata = super::cdataobj::W_CData::from_obj(w_ob).is_some();
    if is_cdata {
        let cdata = super::cdataobj::W_CData::from_obj(w_ob).unwrap();
        if let Some(ct) = super::ctypeobj::ctype_of(cdata)
            && ct.is_float_family()
        {
            let ptr = cdata.ptr;
            return if ct.kind == super::ctypeobj::KIND_PRIM_LONGDOUBLE {
                Ok(unsafe { is_nonnull_longdouble(ptr) })
            } else {
                unsafe { is_nonnull_float(ptr, ct.size) }
            };
        }
    }
    // `space.lookup(w_ob, '__float__')` — the class's slot, not the
    // instance's, so a cdata never takes the float branch.
    let has_float = !is_cdata
        && crate::typedef::r#type(w_ob).is_some_and(|w_type| unsafe {
            crate::baseobjspace::lookup_in_type_where(w_type.as_ptr(), "__float__").is_some()
        });
    let w_io = if has_float {
        pyre_object::w_float_new(crate::baseobjspace::float_w(w_ob)?)
    } else {
        crate::baseobjspace::space_int(w_ob)?
    };
    standard_object_as_bool(w_io).ok_or_else(|| PyError::type_error("integer/float expected"))
}

/// `misc.py _standard_object_as_bool`; `None` stands for its
/// `_NotStandardObject`.
fn standard_object_as_bool(w_ob: PyObjectRef) -> Option<bool> {
    unsafe {
        if pyre_object::pyobject::is_bool(w_ob) {
            return Some(pyre_object::boolobject::w_bool_get_value(w_ob));
        }
        if pyre_object::pyobject::is_int(w_ob) {
            return Some(pyre_object::intobject::w_int_get_value(w_ob) != 0);
        }
        if pyre_object::pyobject::is_long(w_ob) {
            // `space.bigint_w(w_ob).tobool()` — a value too wide for a word
            // is certainly not zero.
            return Some(pyre_object::longobject::w_long_get_value(w_ob).get_sign() != 0);
        }
        if pyre_object::pyobject::is_float(w_ob) {
            return Some(pyre_object::floatobject::w_float_get_value(w_ob) != 0.0);
        }
    }
    None
}

// ── the dynamic loader ──────────────────────────────────────────────────

/// `misc.py dlopen_w` — the name to report the library by, the loader handle,
/// and whether closing the library is this object's to do.
pub fn dlopen_w(w_filename: PyObjectRef, flags: i64) -> Result<(String, usize, bool), PyError> {
    use super::cdataobj::W_CData;
    use super::ctypeobj;

    if let Some(cdata) = W_CData::from_obj(w_filename) {
        // 'flags' is ignored in this case.
        let ct = ctypeobj::ctype_at(cdata.ctype)
            .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
        if ct.kind != ctypeobj::KIND_POINTER || !ct.has(ctypeobj::F_VOID_PTR) {
            return Err(PyError::type_error(format!(
                "dlopen() takes a file name or 'void *' handle, not '{}'",
                ct.name()
            )));
        }
        if cdata.ptr.is_null() {
            return Err(PyError::runtime_error("cannot call dlopen(NULL)"));
        }
        let fname = unsafe { ct.extra_repr(cdata.ptr)? };
        return Ok((fname, adopt_raw_handle(cdata.ptr)?, false));
    }
    let is_none = unsafe { pyre_object::pyobject::is_none(w_filename) };
    let fname = if is_none {
        "<None>".to_string()
    } else {
        crate::gateway::fsdecode_os_str_wtf8(&filename_of(w_filename)?).to_string()
    };
    let handle = open_library(w_filename, is_none, flags, &fname)?;
    Ok((fname, handle, true))
}

/// `space.fsencode_w` — a library name is a path, so it reaches the loader in
/// the filesystem's own units rather than through a lossy decode.
fn filename_of(w_filename: PyObjectRef) -> Result<std::ffi::OsString, PyError> {
    unsafe {
        if pyre_object::bytesobject::is_bytes(w_filename) {
            return Ok(crate::gateway::os_string_from_fs_bytes(
                pyre_object::bytesobject::w_bytes_data(w_filename),
            ));
        }
    }
    Ok(crate::gateway::os_string_from_fs_bytes(
        &crate::gateway::fsencode(w_filename)?,
    ))
}

/// `rffi.cast(DLLHANDLE, handle)` — the raw address becomes the key the rest
/// of this module looks symbols up under, which here means entering it in the
/// loader's own table.
#[cfg(all(feature = "host_env", unix))]
fn adopt_raw_handle(handle: *mut u8) -> Result<usize, PyError> {
    Ok(rustpython_host_env::ctypes::insert_raw_library_handle(
        handle.cast::<std::ffi::c_void>(),
    ))
}

/// A build without that table has no key to hand back, and answering `0` would
/// name a library whose every symbol is missing.
#[cfg(not(all(feature = "host_env", unix)))]
fn adopt_raw_handle(_handle: *mut u8) -> Result<usize, PyError> {
    Err(PyError::os_error(
        "dlopen() cannot adopt a raw 'void *' handle in this build",
    ))
}

/// `rdynload.dlopen`, whose mode defaults to resolving everything now.
#[cfg(all(feature = "host_env", unix))]
fn open_library(
    w_filename: PyObjectRef,
    is_none: bool,
    flags: i64,
    fname: &str,
) -> Result<usize, PyError> {
    let mut mode = flags as libc::c_int;
    if mode & (libc::RTLD_LAZY | libc::RTLD_NOW) == 0 {
        mode |= libc::RTLD_NOW;
    }
    if is_none {
        let ptr = rustpython_host_env::ctypes::dlopen_self(mode)
            .map_err(|e| PyError::os_error(format!("cannot load library {fname}: {e}")))?;
        return Ok(rustpython_host_env::ctypes::insert_raw_library_handle(ptr));
    }
    let name = filename_of(w_filename)?;
    rustpython_host_env::ctypes::open_library_with_mode(&name, mode).map_err(|e| {
        PyError::os_error(format!(
            "cannot load library {fname}: {}",
            crate::with_causes(&e)
        ))
    })
}

#[cfg(all(feature = "host_env", windows))]
fn open_library(
    w_filename: PyObjectRef,
    is_none: bool,
    _flags: i64,
    fname: &str,
) -> Result<usize, PyError> {
    if is_none {
        return Err(PyError::os_error("cannot use None"));
    }
    let name = filename_of(w_filename)?;
    rustpython_host_env::ctypes::open_library(&name).map_err(|e| {
        PyError::os_error(format!(
            "cannot load library {fname}: {}",
            crate::with_causes(&e)
        ))
    })
}

#[cfg(not(all(feature = "host_env", any(unix, windows))))]
fn open_library(
    _w_filename: PyObjectRef,
    _is_none: bool,
    _flags: i64,
    fname: &str,
) -> Result<usize, PyError> {
    Err(PyError::os_error(format!(
        "cannot load library {fname}: this build has no dynamic loader"
    )))
}
