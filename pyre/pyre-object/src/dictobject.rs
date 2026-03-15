//! W_DictObject — Python `dict` type (Phase 1: int keys only).
//!
//! Phase 1 uses a simple Vec<(i64, PyObjectRef)> for int-keyed dicts.
//! Full dict support (arbitrary hashable keys) will be added in Phase 2.
//! The object carries a stable cached length slot so truth/len tracing can
//! follow the same layout as the interpreter.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python dict object (Phase 1: int keys only).
///
/// Layout:
/// `[ob_type: *const PyType | entries: *mut Vec<(i64, PyObjectRef)> | len: usize]`
#[repr(C)]
pub struct W_DictObject {
    pub ob_header: PyObject,
    pub entries: *mut Vec<(i64, PyObjectRef)>,
    pub len: usize,
}

/// Field offset of `len` within `W_DictObject`, for JIT field access.
pub const DICT_LEN_OFFSET: usize = std::mem::offset_of!(W_DictObject, len);

/// Allocate a new empty W_DictObject.
pub fn w_dict_new() -> PyObjectRef {
    let obj = Box::new(W_DictObject {
        ob_header: PyObject {
            ob_type: &DICT_TYPE as *const PyType,
        },
        entries: Box::into_raw(Box::new(Vec::new())),
        len: 0,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Get a value by int key from a dict.
///
/// Returns None if the key is not found.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_getitem(obj: PyObjectRef, key: i64) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    for &(k, v) in entries {
        if k == key {
            return Some(v);
        }
    }
    None
}

/// Set a value by int key in a dict.
///
/// Overwrites existing entries with the same key.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_setitem(obj: PyObjectRef, key: i64, value: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *dict.entries;
    for entry in entries.iter_mut() {
        if entry.0 == key {
            entry.1 = value;
            return;
        }
    }
    entries.push((key, value));
    dict.len += 1;
}

/// Get the number of entries in a dict.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_len(obj: PyObjectRef) -> usize {
    let dict = &*(obj as *const W_DictObject);
    dict.len
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::{w_int_get_value, w_int_new};

    #[test]
    fn test_dict_create_and_access() {
        let dict = w_dict_new();
        unsafe {
            assert!(is_dict(dict));
            assert_eq!(w_dict_len(dict), 0);
            w_dict_setitem(dict, 1, w_int_new(100));
            assert_eq!(w_dict_len(dict), 1);
            let val = w_dict_getitem(dict, 1).unwrap();
            assert_eq!(w_int_get_value(val), 100);
        }
    }

    #[test]
    fn test_dict_overwrite() {
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem(dict, 1, w_int_new(10));
            w_dict_setitem(dict, 1, w_int_new(20));
            assert_eq!(w_dict_len(dict), 1);
            let val = w_dict_getitem(dict, 1).unwrap();
            assert_eq!(w_int_get_value(val), 20);
        }
    }

    #[test]
    fn test_dict_missing_key() {
        let dict = w_dict_new();
        unsafe {
            assert!(w_dict_getitem(dict, 42).is_none());
        }
    }

    #[test]
    fn test_dict_cached_len_matches_entries_growth() {
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem(dict, 1, w_int_new(10));
            w_dict_setitem(dict, 2, w_int_new(20));
            assert_eq!(w_dict_len(dict), 2);
            let entries = &*(*(dict as *const W_DictObject)).entries;
            assert_eq!(entries.len(), 2);
        }
    }
}
