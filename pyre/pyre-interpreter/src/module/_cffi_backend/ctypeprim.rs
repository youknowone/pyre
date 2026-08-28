//! The primitive ctypes — PyPy:
//! `pypy/module/_cffi_backend/ctypeprim.py`.
//!
//! `W_CTypePrimitive` and its seven subclasses are one Python type here (see
//! [`super::ctypeobj`]), so each method they override becomes a `match` on
//! [`W_CType::kind`].

use crate::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj::{self, W_CType};
use super::misc;

/// `W_CTypePrimitive.convert_to_object`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn convert_to_object(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    unsafe {
        match ct.kind {
            // `W_CTypePrimitiveChar.convert_to_object`.
            ctypeobj::KIND_PRIM_CHAR => {
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(
                    &[cdata.read()],
                ))
            }
            // `W_CTypePrimitiveUniChar.convert_to_object`.
            ctypeobj::KIND_PRIM_UNICHAR => {
                let value = misc::read_raw_unsigned_data(cdata, ct.size)? as u32;
                unichr(ct, value)
            }
            // `W_CTypePrimitiveSigned.convert_to_object`.
            ctypeobj::KIND_PRIM_SIGNED => Ok(pyre_object::w_int_new(misc::read_raw_signed_data(
                cdata, ct.size,
            )?)),
            // `W_CTypePrimitiveBool.convert_to_object`.
            ctypeobj::KIND_PRIM_BOOL => Ok(pyre_object::boolobject::w_bool_from(
                read_bool_0_or_1(cdata)? != 0,
            )),
            // `W_CTypePrimitiveUnsigned.convert_to_object`.
            ctypeobj::KIND_PRIM_UNSIGNED => {
                let value = misc::read_raw_unsigned_data(cdata, ct.size)?;
                Ok(unsigned_as_object(ct, value))
            }
            // `W_CTypePrimitiveFloat.convert_to_object`.
            ctypeobj::KIND_PRIM_FLOAT => Ok(pyre_object::w_float_new(misc::read_raw_float_data(
                cdata, ct.size,
            )?)),
            // `W_CTypePrimitiveLongDouble.convert_to_object` hands back a
            // fresh cdata, because the value does not fit a Python float.
            ctypeobj::KIND_PRIM_LONGDOUBLE => {
                let w_cdata = cdataobj::new_cdata_mem(instance_ctype(ct))?;
                let target = W_CData::from_obj(w_cdata)
                    .expect("new_cdata_mem returns a cdata")
                    .ptr;
                std::ptr::copy_nonoverlapping(cdata, target, ct.size as usize);
                Ok(w_cdata)
            }
            // `W_CTypePrimitiveComplex.convert_to_object`.
            ctypeobj::KIND_PRIM_COMPLEX => {
                let half = ct.size >> 1;
                let real = misc::read_raw_float_data(cdata, half)?;
                let imag = misc::read_raw_float_data(cdata.offset(half as isize), half)?;
                Ok(pyre_object::complexobject::w_complex_new(real, imag))
            }
            _ => Err(PyError::type_error(format!(
                "cannot return a cdata '{}'",
                ct.name()
            ))),
        }
    }
}

/// `W_CTypePrimitive.convert_from_object`.
///
/// # Safety
/// `cdata` must be writable for `ct.size` bytes.
pub unsafe fn convert_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    unsafe {
        match ct.kind {
            // `W_CTypePrimitiveChar.convert_from_object`.
            ctypeobj::KIND_PRIM_CHAR => {
                cdata.write(convert_to_char(ct, w_ob)?);
                Ok(())
            }
            // `W_CTypePrimitiveUniChar.convert_from_object`.
            ctypeobj::KIND_PRIM_UNICHAR => {
                let ordinal = convert_to_char_n_t(ct, w_ob)?;
                misc::write_raw_unsigned_data(cdata, u64::from(ordinal), ct.size)
            }
            // `W_CTypePrimitiveSigned.convert_from_object`.
            ctypeobj::KIND_PRIM_SIGNED => {
                // The conversion runs `__index__`, so the object the overflow
                // message names has to be read back out of its slot.
                let roots = pyre_object::gc_roots::push_roots();
                let ob_slot = roots.base();
                let _ = roots.pin_root(w_ob);
                let value = misc::as_long(roots.get(ob_slot))?;
                if ct.has(ctypeobj::F_VALUE_SMALLER_THAN_LONG)
                    && value != misc::signext(value, ct.size)
                {
                    return Err(overflow(ct, roots.get(ob_slot)));
                }
                misc::write_raw_signed_data(cdata, value, ct.size)
            }
            // `W_CTypePrimitiveBool` and `W_CTypePrimitiveUnsigned` share
            // `convert_from_object`; only the range differs.
            ctypeobj::KIND_PRIM_BOOL | ctypeobj::KIND_PRIM_UNSIGNED => {
                let roots = pyre_object::gc_roots::push_roots();
                let ob_slot = roots.base();
                let _ = roots.pin_root(w_ob);
                let value = misc::as_unsigned_long(roots.get(ob_slot), true)?;
                if ct.has(ctypeobj::F_VALUE_FITS_LONG) && value > vrange_max(ct) {
                    return Err(overflow(ct, roots.get(ob_slot)));
                }
                misc::write_raw_unsigned_data(cdata, value, ct.size)
            }
            // `W_CTypePrimitiveFloat.convert_from_object`.
            ctypeobj::KIND_PRIM_FLOAT => {
                let value = crate::baseobjspace::float_w(w_ob)?;
                misc::write_raw_float_data(cdata, value, ct.size)
            }
            // `W_CTypePrimitiveLongDouble.convert_from_object` — a long
            // double source is copied whole rather than narrowed.
            ctypeobj::KIND_PRIM_LONGDOUBLE => {
                if let Some(source) = W_CData::from_obj(w_ob)
                    && ctypeobj::ctype_at(source.ctype)
                        .is_some_and(|s| s.kind == ctypeobj::KIND_PRIM_LONGDOUBLE)
                {
                    std::ptr::copy_nonoverlapping(source.ptr, cdata, ct.size as usize);
                    return Ok(());
                }
                misc::write_raw_longdouble_data(cdata, crate::baseobjspace::float_w(w_ob)?);
                Ok(())
            }
            // `W_CTypePrimitiveComplex.convert_from_object`.
            ctypeobj::KIND_PRIM_COMPLEX => {
                let (real, imag) = unpack_complex(w_ob)?;
                let half = ct.size >> 1;
                misc::write_raw_float_data(cdata, real, half)?;
                misc::write_raw_float_data(cdata.offset(half as isize), imag, half)
            }
            _ => Err(PyError::type_error(format!(
                "cannot initialize cdata '{}'",
                ct.name()
            ))),
        }
    }
}

/// `W_CTypePrimitive.cast` and the two overrides of it.
pub fn cast(w_ctype: PyObjectRef, w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    match ct.kind {
        ctypeobj::KIND_PRIM_FLOAT | ctypeobj::KIND_PRIM_LONGDOUBLE => cast_float(w_ctype, w_ob),
        ctypeobj::KIND_PRIM_COMPLEX => cast_complex(w_ctype, w_ob),
        _ => cast_integer(w_ctype, w_ob),
    }
}

/// `W_CTypePrimitive.cast` — everything that ends up in an integer slot.
fn cast_integer(w_ctype: PyObjectRef, w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let value = if let Some(source) = W_CData::from_obj(w_ob)
        && ctypeobj::ctype_at(source.ctype).is_some_and(|it| it.is_ptr_or_array())
    {
        cast_result(ct, source.ptr as u64)
    } else if unsafe { pyre_object::bytesobject::is_bytes(w_ob) } {
        cast_result(ct, u64::from(cast_str(ct, w_ob)?))
    } else if unsafe { pyre_object::unicodeobject::is_str(w_ob) } {
        cast_result(ct, u64::from(cast_unicode(ct, w_ob)?))
    } else if ct.kind == ctypeobj::KIND_PRIM_BOOL {
        // `W_CTypePrimitiveBool._cast_generic`.
        u64::from(misc::object_as_bool(w_ob)?)
    } else {
        misc::as_unsigned_long_long(w_ob, false)?
    };
    let w_cdata = cdataobj::new_cdata_mem(w_ctype)?;
    let cdata = W_CData::from_obj(w_cdata).expect("new_cdata_mem returns a cdata");
    let ct = ctypeobj::ctype_arg(cdata.ctype)?;
    // `W_CTypePrimitive.cast` writes through `write_raw_integer_data`, which
    // a signed type spells as a signed store and everything else as unsigned;
    // the two are the same bits at this width.
    unsafe { misc::write_raw_unsigned_data(cdata.ptr, value, ct.size)? };
    Ok(w_cdata)
}

/// `W_CTypePrimitiveBool._cast_result` and the default above it.
fn cast_result(ct: &W_CType, value: u64) -> u64 {
    if ct.kind == ctypeobj::KIND_PRIM_BOOL {
        u64::from(value != 0)
    } else {
        value
    }
}

/// `W_CTypePrimitiveFloat.cast`, including the long-double override.
fn cast_float(w_ctype: PyObjectRef, w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    // `W_CTypePrimitiveLongDouble.cast` — a long double source keeps its
    // full precision instead of going through a `double`.
    if ct.kind == ctypeobj::KIND_PRIM_LONGDOUBLE
        && let Some(source) = W_CData::from_obj(w_ob)
        && ctypeobj::ctype_at(source.ctype)
            .is_some_and(|s| s.kind == ctypeobj::KIND_PRIM_LONGDOUBLE)
    {
        return unsafe { convert_to_object(ct, source.ptr) };
    }
    // `unwrap_primitive_cdata` boxes a cdata source, so what it hands back is
    // a fresh object nothing else roots while `float_w` dispatches.
    let roots = pyre_object::gc_roots::push_roots();
    let ob_slot = roots.base();
    let _ = roots.pin_root(unwrap_primitive_cdata(ct, w_ob)?);
    let w_ob = roots.get(ob_slot);
    let value = if unsafe { pyre_object::bytesobject::is_bytes(w_ob) } {
        f64::from(cast_str(ct, w_ob)?)
    } else if unsafe { pyre_object::unicodeobject::is_str(w_ob) } {
        f64::from(cast_unicode(ct, w_ob)?)
    } else {
        crate::baseobjspace::float_w(w_ob)?
    };
    let w_cdata = cdataobj::new_cdata_mem(w_ctype)?;
    let cdata = W_CData::from_obj(w_cdata).expect("new_cdata_mem returns a cdata");
    let ct = ctypeobj::ctype_arg(cdata.ctype)?;
    unsafe {
        if ct.kind == ctypeobj::KIND_PRIM_LONGDOUBLE {
            misc::write_raw_longdouble_data(cdata.ptr, value);
        } else {
            misc::write_raw_float_data(cdata.ptr, value, ct.size)?;
        }
    }
    Ok(w_cdata)
}

/// `W_CTypePrimitiveComplex.cast`.
fn cast_complex(w_ctype: PyObjectRef, w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let roots = pyre_object::gc_roots::push_roots();
    let ob_slot = roots.base();
    let _ = roots.pin_root(unwrap_primitive_cdata(ct, w_ob)?);
    let w_ob = roots.get(ob_slot);
    let (real, imag) = if unsafe { pyre_object::bytesobject::is_bytes(w_ob) } {
        (f64::from(cast_str(ct, w_ob)?), 0.0)
    } else if unsafe { pyre_object::unicodeobject::is_str(w_ob) } {
        (f64::from(cast_unicode(ct, w_ob)?), 0.0)
    } else {
        unpack_complex(w_ob)?
    };
    let w_cdata = cdataobj::new_cdata_mem(w_ctype)?;
    let cdata = W_CData::from_obj(w_cdata).expect("new_cdata_mem returns a cdata");
    let ct = ctypeobj::ctype_arg(cdata.ctype)?;
    let half = ct.size >> 1;
    unsafe {
        misc::write_raw_float_data(cdata.ptr, real, half)?;
        misc::write_raw_float_data(cdata.ptr.offset(half as isize), imag, half)?;
    }
    Ok(w_cdata)
}

/// The shared preamble of the float and complex casts: a cdata operand must
/// be primitive, and it is read as the value it holds.
fn unwrap_primitive_cdata(ct: &W_CType, w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let Some(source) = W_CData::from_obj(w_ob) else {
        return Ok(w_ob);
    };
    let source_ct = ctypeobj::ctype_at(source.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    if !source_ct.is_primitive() {
        return Err(PyError::type_error(format!(
            "cannot cast ctype '{}' to ctype '{}'",
            source_ct.name(),
            ct.name()
        )));
    }
    unsafe { convert_to_object(source_ct, source.ptr) }
}

/// `W_CTypePrimitive.cast_str`.
fn cast_str(ct: &W_CType, w_ob: PyObjectRef) -> Result<u8, PyError> {
    let s = unsafe { pyre_object::bytesobject::w_bytes_data(w_ob) };
    if s.len() != 1 {
        return Err(PyError::type_error(format!(
            "cannot cast string of length {} to ctype '{}'",
            s.len(),
            ct.name()
        )));
    }
    Ok(s[0])
}

/// `W_CTypePrimitive.cast_unicode`.
fn cast_unicode(ct: &W_CType, w_ob: PyObjectRef) -> Result<u32, PyError> {
    let value = unsafe { pyre_object::w_str_get_wtf8(w_ob) };
    let mut points = value.code_points();
    match (points.next(), points.next()) {
        (Some(point), None) => Ok(point.to_u32()),
        _ => Err(PyError::type_error(format!(
            "cannot cast unicode string of length {} to ctype '{}'",
            unsafe { pyre_object::w_str_len(w_ob) },
            ct.name()
        ))),
    }
}

/// `W_CTypePrimitive.cast_to_int` and the overrides of it.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn cast_to_int(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    unsafe {
        match ct.kind {
            // `W_CTypePrimitiveChar.cast_to_int`.
            ctypeobj::KIND_PRIM_CHAR => Ok(pyre_object::w_int_new(i64::from(cdata.read()))),
            // `W_CTypePrimitiveUniChar.cast_to_int`.
            ctypeobj::KIND_PRIM_UNICHAR => {
                if ct.has(ctypeobj::F_SIGNED_WCHAR) {
                    Ok(pyre_object::w_int_new(misc::read_raw_signed_data(
                        cdata, ct.size,
                    )?))
                } else {
                    Ok(unsigned_as_object(
                        ct,
                        misc::read_raw_unsigned_data(cdata, ct.size)?,
                    ))
                }
            }
            // `W_CTypePrimitiveBool.cast_to_int`.
            ctypeobj::KIND_PRIM_BOOL => {
                Ok(pyre_object::w_int_new(i64::from(read_bool_0_or_1(cdata)?)))
            }
            // `W_CTypePrimitiveFloat.cast_to_int` — `int(self.float(cdata))`.
            ctypeobj::KIND_PRIM_FLOAT | ctypeobj::KIND_PRIM_LONGDOUBLE => {
                let w_value = float(ct, cdata)?;
                crate::baseobjspace::space_int(w_value)
            }
            _ if ct.is_primitive() => convert_to_object(ct, cdata),
            _ => Err(PyError::type_error(format!(
                "int() not supported on cdata '{}'",
                ct.name()
            ))),
        }
    }
}

/// `W_CTypePrimitiveFloat.float` and the long-double override.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn float(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    unsafe {
        if ct.kind == ctypeobj::KIND_PRIM_LONGDOUBLE {
            return Ok(pyre_object::w_float_new(misc::read_raw_longdouble_data(
                cdata,
            )));
        }
        Ok(pyre_object::w_float_new(misc::read_raw_float_data(
            cdata, ct.size,
        )?))
    }
}

/// `W_CTypePrimitive.nonzero` and the float overrides of it.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn nonzero(ct: &W_CType, cdata: *const u8) -> Result<bool, PyError> {
    unsafe {
        match ct.kind {
            ctypeobj::KIND_PRIM_LONGDOUBLE => Ok(misc::is_nonnull_longdouble(cdata)),
            ctypeobj::KIND_PRIM_FLOAT => misc::is_nonnull_float(cdata, ct.size),
            ctypeobj::KIND_PRIM_COMPLEX => {
                let half = ct.size >> 1;
                Ok(misc::is_nonnull_float(cdata, half)?
                    | misc::is_nonnull_float(cdata.offset(half as isize), half)?)
            }
            _ => Ok(misc::read_raw_signed_data(cdata, ct.size)? != 0),
        }
    }
}

/// `W_CTypePrimitive.string` and the two overrides of it.
pub fn string(w_cdata: PyObjectRef, maxlen: i64) -> Result<PyObjectRef, PyError> {
    let cdata = cdataobj::cdata_arg(w_cdata)?;
    let ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    match ct.kind {
        // `W_CTypePrimitiveUniChar.string` — one character, as a `str`.
        ctypeobj::KIND_PRIM_UNICHAR => unsafe { convert_to_object(ct, cdata.ptr) },
        // `W_CTypePrimitiveBool.string` bypasses the size-1 case below.
        ctypeobj::KIND_PRIM_BOOL => Err(ctypeobj::unexpected_string_argument(ct)),
        _ if ct.size == 1 => Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[unsafe {
            cdata.ptr.read()
        }])),
        _ => {
            let _ = maxlen;
            Err(ctypeobj::unexpected_string_argument(ct))
        }
    }
}

/// `W_CTypePrimitive.pack_list_of_items` — the fast path that fills a whole
/// array from a list of Python numbers.  `false` means the caller falls back
/// to converting the items one at a time.
///
/// # Safety
/// `cdata` must be writable for `len(items) * ct.size` bytes.
pub unsafe fn pack_list_of_items(
    ct: &W_CType,
    cdata: *mut u8,
    items: &[PyObjectRef],
) -> Result<bool, PyError> {
    match ct.kind {
        ctypeobj::KIND_PRIM_SIGNED => {
            for (i, &w_item) in items.iter().enumerate() {
                let Some(value) = exact_int(w_item) else {
                    return Ok(false);
                };
                if ct.has(ctypeobj::F_VALUE_SMALLER_THAN_LONG)
                    && value != misc::signext(value, ct.size)
                {
                    return Err(overflow_value(ct, value));
                }
                unsafe {
                    misc::write_raw_signed_data(
                        cdata.offset(i as isize * ct.size as isize),
                        value,
                        ct.size,
                    )?;
                }
            }
            Ok(true)
        }
        ctypeobj::KIND_PRIM_FLOAT => {
            for (i, &w_item) in items.iter().enumerate() {
                let Some(value) = exact_float(w_item) else {
                    return Ok(false);
                };
                unsafe {
                    misc::write_raw_float_data(
                        cdata.offset(i as isize * ct.size as isize),
                        value,
                        ct.size,
                    )?;
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// `W_CTypePrimitive.unpack_list_of_int_items` /
/// `unpack_list_of_float_items` — the whole-array fast path of `unpack()`.
/// `None` means the caller converts the items one at a time.
///
/// # Safety
/// `ptr` must be readable for `length * ct.size` bytes.
pub unsafe fn unpack_list_of_items(
    ct: &W_CType,
    ptr: *const u8,
    length: i64,
) -> Result<Option<PyObjectRef>, PyError> {
    // Each item is pinned as it is produced: the next one's allocation is a
    // collection point, and a plain `Vec` of the earlier ones is not a root.
    let mut items = pyre_object::gc_roots::RootedItems::new();
    match ct.kind {
        ctypeobj::KIND_PRIM_SIGNED => {
            for i in 0..length {
                items.push(pyre_object::w_int_new(unsafe {
                    misc::read_raw_signed_data(ptr.offset((i * ct.size) as isize), ct.size)?
                }));
            }
        }
        // `W_CTypePrimitiveUnsigned.unpack_list_of_int_items` only takes the
        // fast path for a width that still fits a signed word.
        ctypeobj::KIND_PRIM_UNSIGNED if ct.has(ctypeobj::F_VALUE_FITS_LONG) => {
            for i in 0..length {
                items.push(pyre_object::w_int_new(unsafe {
                    misc::read_raw_unsigned_data(ptr.offset((i * ct.size) as isize), ct.size)?
                } as i64));
            }
        }
        ctypeobj::KIND_PRIM_FLOAT => {
            for i in 0..length {
                items.push(pyre_object::w_float_new(unsafe {
                    misc::read_raw_float_data(ptr.offset((i * ct.size) as isize), ct.size)?
                }));
            }
        }
        _ => return Ok(None),
    }
    Ok(Some(pyre_object::w_list_new(items.take())))
}

// ── helpers ─────────────────────────────────────────────────────────────

/// The `PyObjectRef` naming `ct` itself.  Every ctype is reachable from the
/// object it is stored on, so this is a pointer cast rather than a lookup.
fn instance_ctype(ct: &W_CType) -> PyObjectRef {
    (ct as *const W_CType)
        .cast_mut()
        .cast::<pyre_object::PyObject>()
}

/// `W_CTypePrimitiveUnsigned.convert_to_object`'s two arms: a width narrower
/// than a word is an ordinary `int`, and a full word may need a bigint.
fn unsigned_as_object(ct: &W_CType, value: u64) -> PyObjectRef {
    if ct.has(ctypeobj::F_VALUE_FITS_LONG) {
        return pyre_object::w_int_new(value as i64);
    }
    if value <= i64::MAX as u64 {
        return pyre_object::w_int_new(value as i64);
    }
    pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from_u128(u128::from(value)))
}

/// `W_CTypePrimitiveUnsigned._compute_vrange_max`, and
/// `W_CTypePrimitiveBool`'s override of it.
fn vrange_max(ct: &W_CType) -> u64 {
    if ct.kind == ctypeobj::KIND_PRIM_BOOL {
        return 1;
    }
    if ct.size >= 8 {
        return u64::MAX;
    }
    (1u64 << (ct.size * 8)) - 1
}

/// `W_CTypePrimitiveBool._read_bool_0_or_1`.
///
/// # Safety
/// `cdata` must be readable for one byte.
unsafe fn read_bool_0_or_1(cdata: *const u8) -> Result<u8, PyError> {
    let value = unsafe { cdata.read() };
    if value >= 2 {
        return Err(PyError::value_error(format!(
            "got a _Bool of value {value}, expected 0 or 1"
        )));
    }
    Ok(value)
}

/// `W_CTypePrimitiveUniChar.convert_to_object`'s code-point check.
fn unichr(ct: &W_CType, value: u32) -> Result<PyObjectRef, PyError> {
    if value > 0x10FFFF {
        let rendered = if ct.has(ctypeobj::F_SIGNED_WCHAR) {
            format!("{:#x}", value as i32)
        } else {
            format!("{value:#x}")
        };
        return Err(PyError::value_error(format!(
            "{} out of range for conversion to unicode: {rendered}",
            ct.name()
        )));
    }
    Ok(pyre_object::w_str_from_codepoint(value))
}

/// `W_CTypePrimitiveChar._convert_to_char`.
fn convert_to_char(ct: &W_CType, w_ob: PyObjectRef) -> Result<u8, PyError> {
    if unsafe { pyre_object::bytesobject::is_bytes(w_ob) } {
        let s = unsafe { pyre_object::bytesobject::w_bytes_data(w_ob) };
        if s.len() == 1 {
            return Ok(s[0]);
        }
    }
    if let Some(source) = W_CData::from_obj(w_ob)
        && ctypeobj::ctype_at(source.ctype).is_some_and(|s| s.kind == ctypeobj::KIND_PRIM_CHAR)
    {
        return Ok(unsafe { source.ptr.read() });
    }
    Err(ct.convert_error("string of length 1", w_ob))
}

/// `W_CTypePrimitiveUniChar._convert_to_charN_t`.
fn convert_to_char_n_t(ct: &W_CType, w_ob: PyObjectRef) -> Result<u32, PyError> {
    if unsafe { pyre_object::unicodeobject::is_str(w_ob) } {
        let value = unsafe { pyre_object::w_str_get_wtf8(w_ob) };
        let mut points = value.code_points();
        let (Some(point), None) = (points.next(), points.next()) else {
            return Err(ct.convert_error("single character", w_ob));
        };
        let ordinal = point.to_u32();
        if ct.size == 2 && ordinal > 0xFFFF {
            return Err(ct.convert_error("single character <= 0xFFFF", w_ob));
        }
        return Ok(ordinal);
    }
    if let Some(source) = W_CData::from_obj(w_ob)
        && let Some(source_ct) = ctypeobj::ctype_at(source.ctype)
        && source_ct.kind == ctypeobj::KIND_PRIM_UNICHAR
        && source_ct.size == ct.size
    {
        return Ok(unsafe { misc::read_raw_unsigned_data(source.ptr, ct.size)? } as u32);
    }
    Err(ct.convert_error("unicode string of length 1", w_ob))
}

/// `space.unpackcomplex`.
fn unpack_complex(w_ob: PyObjectRef) -> Result<(f64, f64), PyError> {
    if unsafe { pyre_object::pyobject::is_complex(w_ob) } {
        return Ok(unsafe {
            (
                pyre_object::complexobject::w_complex_get_real(w_ob),
                pyre_object::complexobject::w_complex_get_imag(w_ob),
            )
        });
    }
    Ok((crate::baseobjspace::float_w(w_ob)?, 0.0))
}

/// `W_CTypePrimitive._overflow`.
fn overflow(ct: &W_CType, w_ob: PyObjectRef) -> PyError {
    let rendered = crate::builtins::builtin_str(&[w_ob])
        .map(|w| unsafe { pyre_object::w_str_get_value(w) }.to_string())
        .unwrap_or_default();
    PyError::overflow_error(format!("integer {rendered} does not fit '{}'", ct.name()))
}

fn overflow_value(ct: &W_CType, value: i64) -> PyError {
    PyError::overflow_error(format!("integer {value} does not fit '{}'", ct.name()))
}

/// `space.listview_int`'s per-item test: an `int` already in a word.
fn exact_int(w_ob: PyObjectRef) -> Option<i64> {
    unsafe {
        if pyre_object::pyobject::is_bool(w_ob) {
            return None;
        }
        if pyre_object::pyobject::is_int(w_ob) {
            return Some(pyre_object::intobject::w_int_get_value(w_ob));
        }
    }
    None
}

/// `space.listview_float`'s per-item test.
fn exact_float(w_ob: PyObjectRef) -> Option<f64> {
    unsafe {
        if pyre_object::pyobject::is_float(w_ob) {
            return Some(pyre_object::floatobject::w_float_get_value(w_ob));
        }
    }
    None
}
