//! Codec entry points -- PyPy `cpyext/codecs.py`.
//!
//! `Encode` and `Decode` hand the object to the looked-up coder.
//! `codecs.py` calls `object.encode` / `object.decode` instead, so a
//! value with no such method never reaches the registry.

use super::object::{argument, call_method, result};
use super::pyobject::CPyObject;
use super::unicodeobject::utf8_text;
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char};

fn c_encoding(encoding: *const c_char) -> Result<String, ()> {
    if encoding.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return Err(());
    }
    let bytes = unsafe { CStr::from_ptr(encoding) }.to_bytes();
    match utf8_text(bytes) {
        Ok(text) => Ok(text.to_string()),
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            Err(())
        }
    }
}

/// `NULL` means the argument was omitted. A bad encoding is already recorded.
fn optional_text(text: *const c_char) -> Result<Option<PyObjectRef>, ()> {
    if text.is_null() {
        return Ok(None);
    }
    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    match utf8_text(bytes) {
        Ok(text) => Ok(Some(pyre_object::w_str_new_managed(text))),
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            Err(())
        }
    }
}

fn coder(encoding: &str, index: i64) -> Result<PyObjectRef, crate::PyError> {
    let info = crate::module::_codecs::lookup_codec_name(encoding)?;
    Ok(unsafe { pyre_object::w_tuple_getitem(info, index) }.unwrap_or_else(pyre_object::w_none))
}

fn incremental(encoding: *const c_char, errors: *const c_char, name: &str) -> *mut CPyObject {
    let Ok(encoding) = c_encoding(encoding) else {
        return std::ptr::null_mut();
    };
    let info = match crate::module::_codecs::lookup_codec_name(&encoding) {
        Ok(info) => info,
        Err(error) => return result(Err(error)),
    };
    let roots = pyre_object::gc_roots::push_roots();
    let info_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(info);
    let errors = match optional_text(errors) {
        Ok(None) => None,
        Ok(Some(text)) => {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(text);
            Some(pyre_object::gc_roots::shadow_stack_get(slot))
        }
        Err(()) => return std::ptr::null_mut(),
    };
    let info = pyre_object::gc_roots::shadow_stack_get(info_slot);
    let arguments: &[PyObjectRef] = match &errors {
        Some(text) => std::slice::from_ref(text),
        None => &[],
    };
    result(call_method(info, name, arguments))
}

/// `codecs.py PyCodec_IncrementalEncoder`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCodec_IncrementalEncoder(
    encoding: *const c_char,
    errors: *const c_char,
) -> *mut CPyObject {
    incremental(encoding, errors, "incrementalencoder")
}

/// `codecs.py PyCodec_IncrementalDecoder`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCodec_IncrementalDecoder(
    encoding: *const c_char,
    errors: *const c_char,
) -> *mut CPyObject {
    incremental(encoding, errors, "incrementaldecoder")
}

/// `codecs.py PyCodec_Encoder`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCodec_Encoder(encoding: *const c_char) -> *mut CPyObject {
    let Ok(encoding) = c_encoding(encoding) else {
        return std::ptr::null_mut();
    };
    result(coder(&encoding, 0))
}

/// `codecs.py PyCodec_Decoder`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCodec_Decoder(encoding: *const c_char) -> *mut CPyObject {
    let Ok(encoding) = c_encoding(encoding) else {
        return std::ptr::null_mut();
    };
    result(coder(&encoding, 1))
}

fn encode_or_decode(
    object: *mut CPyObject,
    encoding: *const c_char,
    errors: *const c_char,
    encode: bool,
) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let Ok(encoding) = c_encoding(encoding) else {
        return std::ptr::null_mut();
    };
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(object);
    let encoding_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(&encoding));
    let errors = match optional_text(errors) {
        Ok(None) => None,
        Ok(Some(text)) => {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(text);
            Some(pyre_object::gc_roots::shadow_stack_get(slot))
        }
        Err(()) => return std::ptr::null_mut(),
    };
    let live = pyre_object::gc_roots::shadow_stack_get;
    result(crate::module::_codecs::codec_encode_or_decode(
        live(object_slot),
        live(encoding_slot),
        errors,
        encode,
    ))
}

/// `codecs.py PyCodec_Encode`, through the looked-up coder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCodec_Encode(
    object: *mut CPyObject,
    encoding: *const c_char,
    errors: *const c_char,
) -> *mut CPyObject {
    encode_or_decode(object, encoding, errors, true)
}

/// `codecs.py PyCodec_Decode`, through the looked-up coder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCodec_Decode(
    object: *mut CPyObject,
    encoding: *const c_char,
    errors: *const c_char,
) -> *mut CPyObject {
    encode_or_decode(object, encoding, errors, false)
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyCodec_IncrementalEncoder as *const ());
    std::hint::black_box(PyCodec_IncrementalDecoder as *const ());
    std::hint::black_box(PyCodec_Encoder as *const ());
    std::hint::black_box(PyCodec_Decoder as *const ());
    std::hint::black_box(PyCodec_Encode as *const ());
    std::hint::black_box(PyCodec_Decode as *const ());
}
