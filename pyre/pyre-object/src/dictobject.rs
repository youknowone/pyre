//! W_DictObject — Python `dict` type.
//!
//! PyPy equivalent: pypy/objspace/std/dictobject.py
//!
//! Supports arbitrary PyObjectRef keys (int, str, etc.) with
//! equality comparison via pointer identity and type-specific checks.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python dict object.
///
/// Layout: `[ob_type | entries | len]`
///
/// Keys are PyObjectRef compared by dict_keys_equal.
/// PyPy uses multiple dict strategies; pyre uses a single Vec for simplicity.
#[repr(C)]
pub struct W_DictObject {
    pub ob_header: PyObject,
    pub entries: *mut Vec<(PyObjectRef, PyObjectRef)>,
    pub len: usize,
}

/// Field offset of `len` within `W_DictObject`, for JIT field access.
pub const DICT_LEN_OFFSET: usize = std::mem::offset_of!(W_DictObject, len);

/// Allocate a new empty dict.
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

/// Compare two dict keys for equality.
///
/// PyPy: uses space.eq_w which dispatches to type-specific comparison.
/// Simplified: pointer identity → int equality → str equality.
unsafe fn dict_keys_equal(a: PyObjectRef, b: PyObjectRef) -> bool {
    if std::ptr::eq(a, b) {
        return true;
    }
    if a.is_null() || b.is_null() {
        return false;
    }
    // Int keys
    if crate::is_int(a) && crate::is_int(b) {
        return crate::w_int_get_value(a) == crate::w_int_get_value(b);
    }
    // Str keys
    if crate::is_str(a) && crate::is_str(b) {
        return crate::w_str_get_value(a) == crate::w_str_get_value(b);
    }
    // Bool keys
    if crate::is_bool(a) && crate::is_bool(b) {
        return crate::w_bool_get_value(a) == crate::w_bool_get_value(b);
    }
    false
}

/// Get a value by PyObjectRef key.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_lookup(obj: PyObjectRef, key: PyObjectRef) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    for &(ref k, v) in entries {
        if dict_keys_equal(*k, key) {
            return Some(v);
        }
    }
    None
}

/// Set a value by PyObjectRef key.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject`.
pub unsafe fn w_dict_store(obj: PyObjectRef, key: PyObjectRef, value: PyObjectRef) {
    let dict = &mut *(obj as *mut W_DictObject);
    let entries = &mut *dict.entries;
    for entry in entries.iter_mut() {
        if dict_keys_equal(entry.0, key) {
            entry.1 = value;
            return;
        }
    }
    entries.push((key, value));
    dict.len += 1;
}

/// Get a value by int key (convenience wrapper).
pub unsafe fn w_dict_getitem(obj: PyObjectRef, key: i64) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    for &(ref k, v) in entries {
        if crate::is_int(*k) && crate::w_int_get_value(*k) == key {
            return Some(v);
        }
    }
    None
}

/// Set a value by int key (convenience wrapper).
pub unsafe fn w_dict_setitem(obj: PyObjectRef, key: i64, value: PyObjectRef) {
    w_dict_store(obj, crate::w_int_new(key), value)
}

/// Get a value by str key (convenience wrapper).
pub unsafe fn w_dict_getitem_str(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let dict = &*(obj as *const W_DictObject);
    let entries = &*dict.entries;
    for &(ref k, v) in entries {
        if crate::is_str(*k) && crate::w_str_get_value(*k) == key {
            return Some(v);
        }
    }
    None
}

/// Set a value by str key (convenience wrapper).
pub unsafe fn w_dict_setitem_str(obj: PyObjectRef, key: &str, value: PyObjectRef) {
    w_dict_store(obj, crate::w_str_new(key), value)
}

/// Get the number of entries.
pub unsafe fn w_dict_len(obj: PyObjectRef) -> usize {
    (*(obj as *const W_DictObject)).len
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::{w_int_get_value, w_int_new};

    #[test]
    fn test_dict_int_key() {
        let dict = w_dict_new();
        unsafe {
            assert!(is_dict(dict));
            w_dict_setitem(dict, 1, w_int_new(100));
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 1).unwrap()), 100);
        }
    }

    #[test]
    fn test_dict_str_key() {
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem_str(dict, "hello", w_int_new(42));
            assert_eq!(
                w_int_get_value(w_dict_getitem_str(dict, "hello").unwrap()),
                42
            );
            assert!(w_dict_getitem_str(dict, "world").is_none());
        }
    }

    #[test]
    fn test_dict_pyobj_key() {
        let dict = w_dict_new();
        let key = crate::w_str_new("test");
        unsafe {
            w_dict_store(dict, key, w_int_new(99));
            assert_eq!(w_int_get_value(w_dict_lookup(dict, key).unwrap()), 99);
        }
    }

    #[test]
    fn test_dict_overwrite() {
        let dict = w_dict_new();
        unsafe {
            w_dict_setitem(dict, 1, w_int_new(10));
            w_dict_setitem(dict, 1, w_int_new(20));
            assert_eq!(w_dict_len(dict), 1);
        }
    }
}
