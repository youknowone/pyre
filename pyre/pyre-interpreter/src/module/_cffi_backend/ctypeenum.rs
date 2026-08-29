//! Enums — PyPy: `pypy/module/_cffi_backend/ctypeenum.py`.
//!
//! `_Mixin_Enum` mixes into `W_CTypePrimitiveSigned` and
//! `W_CTypePrimitiveUnsigned`, so an enum ctype keeps their kind and adds
//! only the two enumerator maps; [`ctypeobj::F_ENUM`] is what selects the
//! mixin's overrides.

use crate::PyError;
use pyre_object::PyObjectRef;

use super::ctypeobj::{self, W_CType};
use super::misc;

/// `_Mixin_Enum._get_value` — a signed long for `W_CTypeEnumSigned`, an
/// unsigned one for `W_CTypeEnumUnsigned`, boxed the way the base primitive
/// would box it so it can key `enumvalues2erators`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
unsafe fn get_value(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    if ct.kind == ctypeobj::KIND_PRIM_SIGNED {
        return Ok(pyre_object::w_int_new(unsafe {
            misc::read_raw_signed_data(cdata, ct.size)?
        }));
    }
    let value = unsafe { misc::read_raw_unsigned_data(cdata, ct.size)? };
    Ok(super::ctypeprim::unsigned_as_object(ct, value))
}

/// The enumerator this value spells, if the enum has one.
fn enumerator_of(ct: &W_CType, w_value: PyObjectRef) -> Result<Option<String>, PyError> {
    if ct.enumvalues2erators.is_null() {
        return Ok(None);
    }
    match crate::baseobjspace::getitem(ct.enumvalues2erators, w_value) {
        Ok(w_name) => Ok(Some(crate::baseobjspace::text_w(w_name)?.to_string())),
        Err(e) if e.kind == crate::PyErrorKind::KeyError => Ok(None),
        Err(e) => Err(e),
    }
}

/// `_Mixin_Enum.extra_repr`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn extra_repr(ct: &W_CType, cdata: *const u8) -> Result<String, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(unsafe { get_value(ct, cdata)? });
    let text = value_str(roots.get(value_slot))?;
    Ok(match enumerator_of(ct, roots.get(value_slot))? {
        Some(name) => format!("{text}: {name}"),
        None => text,
    })
}

/// `_Mixin_Enum.string`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn string(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(unsafe { get_value(ct, cdata)? });
    Ok(pyre_object::w_str_new(
        &match enumerator_of(ct, roots.get(value_slot))? {
            Some(name) => name,
            None => value_str(roots.get(value_slot))?,
        },
    ))
}

/// `str(value)` for a boxed enum value, which may be wider than a machine
/// word when the base type is an unsigned 64-bit one.
fn value_str(w_value: PyObjectRef) -> Result<String, PyError> {
    let w_text = crate::builtins::builtin_str(&[w_value])?;
    Ok(unsafe { pyre_object::w_str_get_value(w_text) }.to_string())
}

/// `_Mixin_Enum._fget` builds its answer fresh each time, so neither map a
/// ctype holds is ever handed out for a caller to mutate.
pub fn copy_map(w_dict: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let w_copy = pyre_object::dictmultiobject::w_dict_new();
    if w_dict.is_null() {
        return Ok(w_copy);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let copy_slot = roots.base();
    let _ = roots.pin_root(w_copy);
    let dict_slot = copy_slot + 1;
    let _ = roots.pin_root(w_dict);
    let keys = crate::baseobjspace::fixedview(roots.get(dict_slot), -1)?;
    let keys_slot = pyre_object::gc_roots::shadow_stack_len();
    for &key in &keys {
        let _ = roots.pin_root(key);
    }
    for i in 0..keys.len() {
        let w_value = crate::baseobjspace::getitem(roots.get(dict_slot), roots.get(keys_slot + i))?;
        crate::baseobjspace::setitem(roots.get(copy_slot), roots.get(keys_slot + i), w_value)?;
    }
    Ok(roots.get(copy_slot))
}
