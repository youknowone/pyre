//! W_StrObject -- Python `str` type backed by a heap-allocated String.
//!
//! Most string operations still go through residual helpers, but the object
//! carries a stable length slot so truth/len paths can follow the same layout
//! from both the interpreter and the tracer.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::pyobject::*;

/// Python string object.
///
/// Layout: `[ob_type: *const PyType | value: *mut String | len: usize]`
/// The `value` pointer owns a heap-allocated `String` (via `Box::into_raw`).
#[repr(C)]
pub struct W_StrObject {
    pub ob_header: PyObject,
    pub value: *mut String,
    pub len: usize,
}

/// Field offset of `value` within `W_StrObject`, for JIT field access.
pub const STR_VALUE_OFFSET: usize = std::mem::offset_of!(W_StrObject, value);
/// Field offset of `len` within `W_StrObject`, for JIT field access.
pub const STR_LEN_OFFSET: usize = std::mem::offset_of!(W_StrObject, len);

/// Allocate a new W_StrObject on the heap.
///
/// Phase 1: uses `Box::leak` for simplicity (objects are never freed).
/// The inner `String` is also `Box::into_raw`'d so it can be recovered.
pub fn w_str_new(s: &str) -> PyObjectRef {
    let inner = Box::into_raw(Box::new(s.to_string()));
    let obj = Box::new(W_StrObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
        },
        value: inner,
        len: s.len(),
    });
    Box::into_raw(obj) as PyObjectRef
}

thread_local! {
    /// String constant interning cache — single-threaded, no lock needed.
    /// RPython has no equivalent lock; string interning is handled by the
    /// translator at compile time, not at runtime.
    static STRING_CONSTANT_CACHE: RefCell<HashMap<String, usize>> =
        RefCell::new(HashMap::new());
}

/// Box a string constant into a heap Python str object.
pub fn box_str_constant(value: &str) -> PyObjectRef {
    STRING_CONSTANT_CACHE.with(|cache| {
        if let Some(&cached) = cache.borrow().get(value) {
            return cached as PyObjectRef;
        }
        let obj = w_str_new(value);
        cache.borrow_mut().insert(value.to_owned(), obj as usize);
        obj
    })
}

/// Extract the &str value from a known W_StrObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_StrObject`.
#[inline]
pub unsafe fn w_str_get_value(obj: PyObjectRef) -> &'static str {
    unsafe {
        let str_obj = obj as *const W_StrObject;
        &*(*str_obj).value
    }
}

/// Extract the cached string length from a known W_StrObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_StrObject`.
#[inline]
pub unsafe fn w_str_len(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_StrObject)).len }
}

/// Check if an object is a str.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_str(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &STR_TYPE) }
}

pub extern "C" fn jit_str_concat(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        let sa = w_str_get_value(a);
        let sb = w_str_get_value(b);
        let mut result = String::with_capacity(sa.len() + sb.len());
        result.push_str(sa);
        result.push_str(sb);
        w_str_new(&result) as i64
    }
}

pub extern "C" fn jit_str_repeat(s: i64, n: i64) -> i64 {
    let s = s as PyObjectRef;
    unsafe {
        let sv = w_str_get_value(s);
        let count = if n < 0 { 0 } else { n as usize };
        w_str_new(&sv.repeat(count)) as i64
    }
}

pub extern "C" fn jit_str_compare(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        let sa = w_str_get_value(a);
        let sb = w_str_get_value(b);
        match sa.cmp(sb) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

pub extern "C" fn jit_str_is_true(s: i64) -> i64 {
    let s = s as PyObjectRef;
    unsafe { (w_str_len(s) != 0) as i64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_str_create_and_read() {
        let obj = w_str_new("hello");
        unsafe {
            assert!(is_str(obj));
            assert!(!is_int(obj));
            assert_eq!(w_str_get_value(obj), "hello");
        }
    }

    #[test]
    fn test_str_empty() {
        let obj = w_str_new("");
        unsafe {
            assert!(is_str(obj));
            assert_eq!(w_str_get_value(obj), "");
        }
    }

    #[test]
    fn test_str_field_offset() {
        assert_eq!(STR_VALUE_OFFSET, 8); // after *const PyType (8 bytes on 64-bit)
        assert_eq!(STR_LEN_OFFSET, 16);
    }

    #[test]
    fn test_str_cached_len_matches_value() {
        let obj = w_str_new("hello");
        unsafe {
            assert_eq!(w_str_len(obj), 5);
            assert_eq!(w_str_get_value(obj).len(), 5);
        }
    }

    #[test]
    fn test_box_str_constant_reuses_same_object() {
        let a = box_str_constant("pyre");
        let b = box_str_constant("pyre");
        assert_eq!(a, b);
    }

    #[test]
    fn test_jit_string_helpers_share_str_semantics() {
        let a = w_str_new("ab");
        let b = w_str_new("cd");
        let cat = jit_str_concat(a as i64, b as i64) as PyObjectRef;
        let rep = jit_str_repeat(a as i64, 3) as PyObjectRef;
        unsafe {
            assert_eq!(w_str_get_value(cat), "abcd");
            assert_eq!(w_str_get_value(rep), "ababab");
            assert_eq!(jit_str_compare(a as i64, b as i64), -1);
            assert_eq!(jit_str_is_true(a as i64), 1);
            assert_eq!(jit_str_is_true(w_str_new("") as i64), 0);
        }
    }
}
