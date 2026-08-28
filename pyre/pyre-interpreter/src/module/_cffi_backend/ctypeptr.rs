//! Pointers and arrays — PyPy:
//! `pypy/module/_cffi_backend/ctypeptr.py` and `ctypearray.py`.
//!
//! `W_CTypePtrOrArray`, `W_CTypePtrBase`, `W_CTypePointer` and
//! `W_CTypeArray` all share `W_CType`'s typedef, so their overrides are the
//! `match` arms below rather than four classes.

use crate::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj::{self, W_CType};
use super::misc;

/// `W_CTypePtrBase.convert_to_object` — the pointer word at `cdata`, boxed.
///
/// # Safety
/// `cdata` must be readable for the width of a pointer.
pub unsafe fn pointer_convert_to_object(ct: &W_CType, cdata: *const u8) -> PyObjectRef {
    let target = unsafe { cdata.cast::<*mut u8>().read_unaligned() };
    cdataobj::new_cdata(target, ct.as_object())
}

/// `W_CTypeArray.convert_to_object` — the array itself, not a copy.  An
/// array of unknown length has no length to give the result, so it decays to
/// the pointer type, exactly as the comment there says.
///
/// # Safety
/// `cdata` must point into the array's storage.
pub unsafe fn array_convert_to_object(ct: &W_CType, cdata: *const u8) -> PyObjectRef {
    let w_ctype = if ct.length < 0 {
        ct.ctptr
    } else {
        ct.as_object()
    };
    cdataobj::new_cdata(cdata.cast_mut(), w_ctype)
}

/// `W_CTypePtrBase.convert_from_object`.
///
/// # Safety
/// `cdata` must be writable for the width of a pointer.
pub unsafe fn pointer_convert_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    let Some(source) = W_CData::from_obj(w_ob) else {
        return Err(ct.convert_error("cdata pointer", w_ob));
    };
    let mut other = ctypeobj::ctype_at(source.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    if other.kind == ctypeobj::KIND_ARRAY {
        other = ctypeobj::ctype_at(other.ctptr)
            .ok_or_else(|| PyError::system_error("array without a pointer type"))?;
    }
    if other.kind != ctypeobj::KIND_POINTER {
        return Err(ct.convert_error("compatible pointer", w_ob));
    }
    if !std::ptr::eq(ct as *const W_CType, other as *const W_CType) {
        if ct.has(ctypeobj::F_VOID_PTR) || other.has(ctypeobj::F_VOID_PTR) {
            // A cast from or to 'void *' is always allowed.
        } else if ct.has(ctypeobj::F_VOIDCHAR_PTR) || other.has(ctypeobj::F_VOIDCHAR_PTR) {
            // 'char *' is accepted either way for backward compatibility;
            // between two single-byte pointers it is silent, otherwise it
            // warns that the acceptance will end.
            if !(ct.has(ctypeobj::F_ONEBYTE_PTR) && other.has(ctypeobj::F_ONEBYTE_PTR)) {
                crate::warn::warn_category(
                    &format!(
                        "implicit cast from '{}' to '{}' will be forbidden in the future (check that the types are as you expect; use an explicit ffi.cast() if they are correct)",
                        other.name(),
                        ct.name()
                    ),
                    "UserWarning",
                    1,
                )?;
            }
        } else {
            return Err(ct.convert_error("compatible pointer", w_ob));
        }
    }
    unsafe { cdata.cast::<*mut u8>().write_unaligned(source.ptr) };
    Ok(())
}

/// `W_CTypeArray.convert_from_object`.
///
/// # Safety
/// `cdata` must be writable for the array's size.
pub unsafe fn array_convert_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    // The fast path: an array of exactly this type is a straight copy.
    if let Some(source) = W_CData::from_obj(w_ob)
        && source.ctype == ct.as_object()
    {
        let item = item_of(ct)?;
        let length = source.array_length()?;
        unsafe {
            std::ptr::copy_nonoverlapping(source.ptr, cdata, (item.size * length) as usize);
        }
        return Ok(());
    }
    unsafe { convert_array_from_object(ct, cdata, w_ob) }
}

/// `W_CTypePtrOrArray.convert_array_from_object`.
///
/// # Safety
/// `cdata` must be writable for as many items as `w_ob` supplies.
pub unsafe fn convert_array_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    let item = item_of(ct)?;
    if unsafe { pyre_object::pyobject::is_list(w_ob) || pyre_object::pyobject::is_tuple(w_ob) } {
        let items = crate::baseobjspace::unpackiterable(w_ob, -1)?;
        if !ct.within_bounds(items.len() as i64) {
            return Err(PyError::index_error(format!(
                "too many initializers for '{}' (got {})",
                ct.name(),
                items.len()
            )));
        }
        let roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&items);
        let items: Vec<_> = (0..items.len()).map(|i| roots.get(base + i)).collect();
        if unsafe { super::ctypeprim::pack_list_of_items(item, cdata, &items)? } {
            return Ok(());
        }
        for i in 0..items.len() {
            let element = unsafe { cdata.offset(i as isize * item.size as isize) };
            unsafe { ctypeobj::convert_from_object(item, element, roots.get(base + i))? };
        }
        return Ok(());
    }
    if ct.has(ctypeobj::F_ACCEPT_STR) {
        if !unsafe { pyre_object::bytesobject::is_bytes(w_ob) } {
            return Err(ct.convert_error("bytes or list or tuple", w_ob));
        }
        let s = unsafe { pyre_object::bytesobject::w_bytes_data(w_ob) };
        let n = s.len() as i64;
        if ct.length >= 0 && n > ct.length {
            return Err(PyError::index_error(format!(
                "initializer string is too long for '{}' (got {n} characters)",
                ct.name()
            )));
        }
        if item.kind == ctypeobj::KIND_PRIM_BOOL && s.iter().any(|&c| c > 1) {
            return Err(PyError::value_error(
                "an array of _Bool can only contain \\x00 or \\x01",
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), cdata, s.len());
            if n != ct.length {
                cdata.offset(n as isize).write(0);
            }
        }
        return Ok(());
    }
    if item.kind == ctypeobj::KIND_PRIM_UNICHAR {
        if !unsafe { pyre_object::unicodeobject::is_str(w_ob) } {
            return Err(ct.convert_error("unicode or list or tuple", w_ob));
        }
        let value = unsafe { pyre_object::w_str_get_wtf8(w_ob) };
        let points: Vec<u32> = value.code_points().map(|p| p.to_u32()).collect();
        let n = if item.size == 2 {
            points.iter().map(|&p| if p > 0xFFFF { 2 } else { 1 }).sum()
        } else {
            points.len() as i64
        };
        if ct.length >= 0 && n > ct.length {
            return Err(PyError::index_error(format!(
                "initializer unicode string is too long for '{}' (got {n} characters)",
                ct.name()
            )));
        }
        let mut at = 0isize;
        for point in points {
            unsafe {
                if item.size == 2 && point > 0xFFFF {
                    // A code point past the BMP is a surrogate pair in a
                    // `char16_t` array.
                    let unit = point - 0x10000;
                    misc::write_raw_unsigned_data(
                        cdata.offset(at * 2),
                        u64::from(0xD800 + (unit >> 10)),
                        2,
                    )?;
                    misc::write_raw_unsigned_data(
                        cdata.offset((at + 1) * 2),
                        u64::from(0xDC00 + (unit & 0x3FF)),
                        2,
                    )?;
                    at += 2;
                } else {
                    misc::write_raw_unsigned_data(
                        cdata.offset(at * item.size as isize),
                        u64::from(point),
                        item.size,
                    )?;
                    at += 1;
                }
            }
        }
        if n != ct.length {
            unsafe {
                misc::write_raw_unsigned_data(cdata.offset(at * item.size as isize), 0, item.size)?;
            }
        }
        return Ok(());
    }
    Err(ct.convert_error("list or tuple", w_ob))
}

/// `W_CTypePtrOrArray.cast`.
pub fn cast(w_ctype: PyObjectRef, w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if ct.size < 0 {
        return Err(PyError::type_error(format!(
            "cannot cast to '{}'",
            ct.name()
        )));
    }
    let value = if let Some(source) = W_CData::from_obj(w_ob)
        && ctypeobj::ctype_at(source.ctype).is_some_and(|it| it.is_ptr_or_array())
    {
        source.ptr
    } else {
        misc::as_unsigned_long(w_ob, false)? as usize as *mut u8
    };
    Ok(cdataobj::new_cdata(value, w_ctype))
}

/// `W_CTypePointer.newp`.
pub fn pointer_newp(w_ctype: PyObjectRef, w_init: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let item = item_of(ct)?;
    let mut datasize = item.size;
    if datasize < 0 {
        return Err(PyError::type_error(format!(
            "cannot instantiate ctype '{}' of unknown size",
            ct.name()
        )));
    }
    if ct.is_char_or_unichar_ptr_or_array() {
        // Room for the null character `newp` always adds.
        datasize *= 2;
    }
    let roots = pyre_object::gc_roots::push_roots();
    let init_slot = roots.base();
    let _ = roots.pin_root(w_init);
    let w_cdata = cdataobj::new_cdata_owning(w_ctype, datasize, -1)?;
    let cdata_slot = init_slot + 1;
    let _ = roots.pin_root(w_cdata);
    let w_init = roots.get(init_slot);
    if !unsafe { pyre_object::pyobject::is_none(w_init) } {
        let cdata = cdataobj::cdata_arg(roots.get(cdata_slot))?;
        let ct = ctypeobj::ctype_arg(cdata.ctype)?;
        let item = item_of(ct)?;
        unsafe { ctypeobj::convert_from_object(item, cdata.ptr, w_init)? };
    }
    Ok(roots.get(cdata_slot))
}

/// `W_CTypeArray.newp`.
pub fn array_newp(w_ctype: PyObjectRef, w_init: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let roots = pyre_object::gc_roots::push_roots();
    let init_slot = roots.base();
    let _ = roots.pin_root(w_init);
    let (w_init, datasize, length) = if ct.size < 0 {
        let (w_init, length) = new_array_length(ct, roots.get(init_slot))?;
        let item = item_of(ct)?;
        let datasize = length.checked_mul(item.size).ok_or_else(array_overflow)?;
        (w_init, datasize, length)
    } else {
        (roots.get(init_slot), ct.size, ct.length)
    };
    let init_slot2 = init_slot + 1;
    let _ = roots.pin_root(w_init);
    let w_cdata = cdataobj::new_cdata_owning(w_ctype, datasize, length)?;
    let cdata_slot = init_slot2 + 1;
    let _ = roots.pin_root(w_cdata);
    let w_init = roots.get(init_slot2);
    if !unsafe { pyre_object::pyobject::is_none(w_init) } {
        let cdata = cdataobj::cdata_arg(roots.get(cdata_slot))?;
        let ct = ctypeobj::ctype_arg(cdata.ctype)?;
        unsafe { array_convert_from_object(ct, cdata.ptr, w_init)? };
    }
    Ok(roots.get(cdata_slot))
}

/// `W_CTypeArray.get_new_array_length`.
fn new_array_length(ct: &W_CType, w_value: PyObjectRef) -> Result<(PyObjectRef, i64), PyError> {
    unsafe {
        if pyre_object::pyobject::is_list(w_value) || pyre_object::pyobject::is_tuple(w_value) {
            let length = crate::runtime_ops::sequence_len(w_value)? as i64;
            return Ok((w_value, length));
        }
        if pyre_object::bytesobject::is_bytes(w_value) {
            // A string initializer carries its own null terminator.
            let s = pyre_object::bytesobject::w_bytes_data(w_value);
            return Ok((w_value, s.len() as i64 + 1));
        }
        if pyre_object::unicodeobject::is_str(w_value) {
            let value = pyre_object::w_str_get_wtf8(w_value);
            let item = item_of(ct)?;
            let length: i64 = if item.size == 2 {
                value
                    .code_points()
                    .map(|p| if p.to_u32() > 0xFFFF { 2 } else { 1 })
                    .sum()
            } else {
                value.code_points().count() as i64
            };
            return Ok((w_value, length + 1));
        }
    }
    // `__index__` is arbitrary Python, so the name the TypeError reports is
    // read off the object before the conversion rather than after it.
    let got = crate::type_methods::arg_type_name(w_value);
    let length = crate::baseobjspace::index_int_w_preserve_negative(w_value).map_err(|e| {
        if e.kind == crate::PyErrorKind::TypeError {
            PyError::type_error(format!(
                "expected new array length or list/tuple/str, not {got}"
            ))
        } else {
            e
        }
    })?;
    if length < 0 {
        return Err(PyError::value_error("negative array length"));
    }
    Ok((pyre_object::w_none(), length))
}

fn array_overflow() -> PyError {
    PyError::overflow_error("array size would overflow a ssize_t")
}

/// `W_CTypePointer.add` and `W_CTypeArray.add`.
pub fn add(w_ctype: PyObjectRef, cdata: *mut u8, i: i64) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let item = item_of(ct)?;
    let mut item_size = item.size;
    if item_size < 0 {
        if ct.has(ctypeobj::F_VOID_PTR) {
            item_size = 1;
        } else {
            return Err(PyError::type_error(format!(
                "ctype '{}' points to items of unknown size",
                ct.name()
            )));
        }
    }
    let target = unsafe { cdata.offset((i * item_size) as isize) };
    // An array's arithmetic result is a pointer, not another array.
    let w_result_ctype = if ct.kind == ctypeobj::KIND_ARRAY {
        ct.ctptr
    } else {
        w_ctype
    };
    Ok(cdataobj::new_cdata(target, w_result_ctype))
}

/// `W_CTypePtrOrArray.string`.
pub fn string(w_cdata: PyObjectRef, maxlen: i64) -> Result<PyObjectRef, PyError> {
    let cdata = cdataobj::cdata_arg(w_cdata)?;
    let ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    let item = item_of(ct)?;
    if !item.is_primitive() || item.kind == ctypeobj::KIND_PRIM_BOOL {
        return Err(ctypeobj::unexpected_string_argument(ct));
    }
    if cdata.ptr.is_null() {
        let w_repr = crate::builtins::builtin_repr(&[w_cdata])?;
        return Err(PyError::runtime_error(format!(
            "cannot use string() on {}",
            unsafe { pyre_object::w_str_get_value(w_repr) }
        )));
    }
    let mut length = maxlen;
    if length < 0 && ct.kind == ctypeobj::KIND_ARRAY {
        length = cdata.array_length()?;
    }
    // A pointer to a one-byte type builds a `bytes` up to the first NUL.
    if item.size == 1 {
        let bytes = unsafe { read_until_nul(cdata.ptr, length) };
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(bytes));
    }
    if ct.is_unichar_ptr_or_array() {
        let measured = unsafe { measure_wide_length(cdata.ptr, item.size, length)? };
        return unpack_ptr(ct, item, cdata.ptr, measured);
    }
    Err(ctypeobj::unexpected_string_argument(ct))
}

/// `rffi.charp2str` / `charp2strn` — the bytes before the first NUL, and at
/// most `length` of them when one was given.
///
/// # Safety
/// `ptr` must be readable up to the first NUL, or for `length` bytes.
unsafe fn read_until_nul(ptr: *const u8, length: i64) -> &'static [u8] {
    let mut n = 0usize;
    let limit = if length < 0 {
        usize::MAX
    } else {
        length as usize
    };
    while n < limit && unsafe { ptr.add(n).read() } != 0 {
        n += 1;
    }
    unsafe { std::slice::from_raw_parts(ptr, n) }
}

/// `wchar_helper.measure_length_16` / `measure_length_32`.
///
/// # Safety
/// `ptr` must be readable up to the first zero unit, or for `length` units.
unsafe fn measure_wide_length(ptr: *const u8, unit_size: i64, length: i64) -> Result<i64, PyError> {
    let mut n = 0i64;
    while length < 0 || n < length {
        let unit = unsafe {
            misc::read_raw_unsigned_data(ptr.offset((n * unit_size) as isize), unit_size)?
        };
        if unit == 0 {
            break;
        }
        n += 1;
    }
    Ok(n)
}

/// `W_CType.unpack_ptr` and the primitive fast paths over it.
pub fn unpack_ptr(
    ct: &W_CType,
    item: &W_CType,
    ptr: *mut u8,
    length: i64,
) -> Result<PyObjectRef, PyError> {
    if item.size < 0 {
        return Err(PyError::value_error(format!(
            "'{}' points to items of unknown size",
            ct.name()
        )));
    }
    // `W_CTypePrimitiveChar.unpack_ptr` — a `bytes`, not a list.
    if item.kind == ctypeobj::KIND_PRIM_CHAR {
        let bytes = unsafe { std::slice::from_raw_parts(ptr, length.max(0) as usize) };
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(bytes));
    }
    // `W_CTypePrimitiveUniChar.unpack_ptr` — a `str`.
    if item.kind == ctypeobj::KIND_PRIM_UNICHAR {
        return unpack_wide_string(item, ptr, length);
    }
    if let Some(w_list) = unsafe { super::ctypeprim::unpack_list_of_items(item, ptr, length)? } {
        return Ok(w_list);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for i in 0..length {
        let element = unsafe { ptr.offset((i * item.size) as isize) };
        let _ = roots.pin_root(unsafe { ctypeobj::convert_to_object(item, element)? });
    }
    let items = (0..length as usize).map(|i| roots.get(base + i)).collect();
    Ok(pyre_object::w_list_new(items))
}

/// `wchar_helper.utf8_from_char16` / `utf8_from_char32`.
fn unpack_wide_string(item: &W_CType, ptr: *mut u8, length: i64) -> Result<PyObjectRef, PyError> {
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    let mut i = 0i64;
    while i < length {
        let mut value = unsafe {
            misc::read_raw_unsigned_data(ptr.offset((i * item.size) as isize), item.size)?
        } as u32;
        i += 1;
        // A `char16_t` array carries a surrogate pair for anything past the
        // BMP; a lone surrogate stays one, which WTF-8 can hold.
        if item.size == 2 && (0xD800..0xDC00).contains(&value) && i < length {
            let low = unsafe {
                misc::read_raw_unsigned_data(ptr.offset((i * item.size) as isize), item.size)?
            } as u32;
            if (0xDC00..0xE000).contains(&low) {
                value = 0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00);
                i += 1;
            }
        }
        let Some(point) = rustpython_wtf8::CodePoint::from_u32(value) else {
            return Err(PyError::value_error(format!(
                "{} out of range for conversion to unicode: {value:#x}",
                item.name()
            )));
        };
        out.push(point);
    }
    Ok(pyre_object::w_str_from_wtf8(out))
}

/// `W_CTypePtrOrArray.ctitem`, which every pointer and array has.
pub fn item_of(ct: &W_CType) -> Result<&'static mut W_CType, PyError> {
    ctypeobj::ctype_at(ct.ctitem)
        .ok_or_else(|| PyError::system_error("pointer or array without an item type"))
}
