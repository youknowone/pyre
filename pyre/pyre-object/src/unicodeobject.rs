//! W_UnicodeObject -- Python `str` type backed by a heap-allocated WTF-8 buffer.
//!
//! Most string operations still go through residual helpers, but the object
//! carries a stable length slot so truth/len paths can follow the same layout
//! from both the interpreter and the tracer.
//!
//! The value buffer is a `Wtf8Buf` rather than a Rust `String`, mirroring
//! PyPy's `W_UnicodeObject._utf8` (`pypy/objspace/std/unicodeobject.py`),
//! which stores UTF-8 bytes that may carry encoded surrogates under
//! `allow_surrogates=True`.  WTF-8 is the same model: a superset of UTF-8
//! that can additionally represent lone surrogate code points.  Every
//! Rust `&str` is valid WTF-8, so the common (surrogate-free) path is
//! zero-cost via `Wtf8::as_str`.

use std::cell::RefCell;
use std::collections::HashMap;

use rustpython_wtf8::{Wtf8, Wtf8Buf};

use crate::pyobject::*;

/// Python string object.
///
/// Layout: `[ob_type | w_class | value:*mut Wtf8Buf | byte_len | len]`
/// `byte_len` is the WTF-8 byte count (RPython STR `rstr.py:1226
/// Array(Char)` parity — `llmodel.py:667 bh_strlen` reads this).
/// `len` is the codepoint count (RPython UNICODE parity —
/// `bh_unicodelen` reads this).  The `value` pointer owns a
/// heap-allocated `Wtf8Buf` (via `Box::into_raw`).
#[repr(C)]
pub struct W_UnicodeObject {
    pub ob_header: PyObject,
    pub value: *mut Wtf8Buf,
    pub byte_len: usize,
    pub len: usize,
}

/// Field offset of `value` within `W_UnicodeObject`, for JIT field access.
pub const UNICODE_VALUE_OFFSET: usize = std::mem::offset_of!(W_UnicodeObject, value);
/// Field offset of `byte_len` (UTF-8 byte count) for STR STRLEN parity.
pub const UNICODE_BYTE_LEN_OFFSET: usize = std::mem::offset_of!(W_UnicodeObject, byte_len);
/// Field offset of `len` (codepoint count) for UNICODE UNICODELEN parity.
pub const UNICODE_LEN_OFFSET: usize = std::mem::offset_of!(W_UnicodeObject, len);

/// GC type id assigned to `W_UnicodeObject` at JitDriver init time.
pub const W_UNICODE_GC_TYPE_ID: u32 = 34;

/// Fixed payload size (`framework.py:811`).
pub const W_UNICODE_OBJECT_SIZE: usize = std::mem::size_of::<W_UnicodeObject>();

impl crate::lltype::GcType for W_UnicodeObject {
    fn type_id() -> u32 {
        W_UNICODE_GC_TYPE_ID
    }
    const SIZE: usize = W_UNICODE_OBJECT_SIZE;
}

/// Allocate a new W_UnicodeObject on the heap.
///
/// Uses `Box::leak` for simplicity (objects are never freed).
/// The inner `Wtf8Buf` is also `Box::into_raw`'d so it can be recovered.
/// `from_string` takes ownership of the bytes with no copy or
/// re-validation (every `&str` is already valid WTF-8).
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py:139`), the
/// `box_str_constant` twin: the body runs a `str::chars` count plus an
/// unfused `malloc_typed` NewWithVtable (`W_UnicodeObject`), so the JIT
/// residualises the whole `&str -> W_UnicodeObject` construction to a
/// stable fnaddr instead of tracing it.  The `-> PyObjectRef` result is a
/// plain GCREF with no discriminant to erase.
#[majit_macros::dont_look_inside]
pub fn w_str_new(s: &str) -> PyObjectRef {
    let value = crate::lltype::malloc_raw(Wtf8Buf::from_string(s.to_string()));
    let byte_len = s.len();
    let char_len = s.chars().count();
    crate::lltype::malloc_typed(W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: get_instantiate(&STR_TYPE),
        },
        value,
        byte_len,
        len: char_len,
    }) as PyObjectRef
}

/// Allocate a new W_UnicodeObject from a WTF-8 buffer that may carry lone
/// surrogate code points (produced by surrogateescape / surrogatepass
/// decoding).  `byte_len` is the WTF-8 byte count, `len` the code point
/// count (which counts each surrogate as one code point).
pub fn w_str_from_wtf8(value: Wtf8Buf) -> PyObjectRef {
    let byte_len = value.len();
    let char_len = value.code_points().count();
    let value = crate::lltype::malloc_raw(value);
    crate::lltype::malloc_typed(W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class: get_instantiate(&STR_TYPE),
        },
        value,
        byte_len,
        len: char_len,
    }) as PyObjectRef
}

/// Allocate a `str` subclass instance through the stable GC allocator.
/// PyPy's `W_UnicodeObject` subclass is an ordinary GC object; exact strings
/// remain on pyre's existing immortal/interned path, while subclass identity
/// and app-level finalization require collector ownership.
pub fn w_str_subclass_from_wtf8(value: Wtf8Buf, w_class: PyObjectRef) -> PyObjectRef {
    let byte_len = value.len();
    let char_len = value.code_points().count();
    let value = crate::lltype::malloc_raw(value);
    let unicode = W_UnicodeObject {
        ob_header: PyObject {
            ob_type: &STR_TYPE as *const PyType,
            w_class,
        },
        value,
        byte_len,
        len: char_len,
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_UNICODE_GC_TYPE_ID, W_UNICODE_OBJECT_SIZE);
    let obj = if raw.is_null() {
        crate::lltype::malloc_typed(unicode) as PyObjectRef
    } else {
        unsafe {
            std::ptr::write(raw as *mut W_UnicodeObject, unicode);
            raw as PyObjectRef
        }
    };
    crate::gc_hook::maybe_register_finalizer(obj);
    obj
}

/// Lightweight GC destructor for the off-heap WTF-8 buffer owned by a
/// managed `W_UnicodeObject`.
pub unsafe fn unicode_object_destructor(obj_addr: usize) {
    let obj = unsafe { &mut *(obj_addr as *mut W_UnicodeObject) };
    if !obj.value.is_null() {
        unsafe { drop(Box::from_raw(obj.value)) };
        obj.value = std::ptr::null_mut();
    }
}

thread_local! {
    /// String constant interning cache — single-threaded, no lock needed.
    /// RPython has no equivalent lock; string interning is handled by the
    /// translator at compile time, not at runtime.  Keyed by WTF-8 so a
    /// surrogate-bearing constant (a `'\udcff'` literal) interns too.
    static STRING_CONSTANT_CACHE: RefCell<HashMap<Wtf8Buf, usize>> =
        RefCell::new(HashMap::new());
}

/// Box a string constant into a heap Python str object.
///
/// Reads the thread-local `STRING_CONSTANT_CACHE` the tracer cannot model;
/// the JIT residualises the call instead of tracing into it
/// (`@dont_look_inside`, `rlib/jit.py:139`), the `box_str`/`pin_root` twin.
#[majit_macros::dont_look_inside]
pub fn box_str_constant(value: &Wtf8) -> PyObjectRef {
    STRING_CONSTANT_CACHE.with(|cache| {
        if let Some(&cached) = cache.borrow().get(value) {
            return cached as PyObjectRef;
        }
        let obj = w_str_from_wtf8(value.to_owned());
        cache.borrow_mut().insert(value.to_owned(), obj as usize);
        obj
    })
}

/// Extract the &str value from a known W_UnicodeObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
///
/// # Panics
/// The backing buffer is WTF-8.  Surrogateescape / surrogatepass decoding
/// can produce surrogate-bearing strings, so a `&str` view is not
/// guaranteed.  Consumers that must tolerate lone surrogates read through
/// [`w_str_get_wtf8`]; `&str` consumers reach this accessor and panic on a
/// surrogate rather than silently corrupting.
#[inline]
pub unsafe fn w_str_get_value(obj: PyObjectRef) -> &'static str {
    unsafe {
        let str_obj = obj as *const W_UnicodeObject;
        (*(*str_obj).value)
            .as_str()
            .expect("w_str_get_value: backing Wtf8Buf is not valid UTF-8 (lone surrogate)")
    }
}

/// Borrow the WTF-8 view of a known W_UnicodeObject, surrogate-aware.
///
/// Unlike [`w_str_get_value`], this never panics on lone surrogates.
/// Callers that must handle surrogate-bearing strings (codec encode,
/// repr) read code points through this accessor.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_get_wtf8(obj: PyObjectRef) -> &'static Wtf8 {
    unsafe {
        let str_obj = obj as *const W_UnicodeObject;
        &*(*str_obj).value
    }
}

/// Borrow a known W_UnicodeObject as `&str`, or `None` when it carries a lone
/// surrogate (so the backing is not valid UTF-8).
///
/// String-keyed fast paths that store keys in a `&str`-keyed map use this
/// to skip surrogate keys and fall through to the generic object-keyed
/// path instead of panicking in [`w_str_get_value`].
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_get_value_opt(obj: PyObjectRef) -> Option<&'static str> {
    unsafe {
        let str_obj = obj as *const W_UnicodeObject;
        (*(*str_obj).value).as_str().ok()
    }
}

/// Extract the cached string length from a known W_UnicodeObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_UnicodeObject`.
#[inline]
pub unsafe fn w_str_len(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_UnicodeObject)).len }
}

/// Check if an object is a str.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_str(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &STR_TYPE) }
}

#[majit_macros::elidable]
pub extern "C" fn jit_str_concat(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        let sa = w_str_get_wtf8(a);
        let sb = w_str_get_wtf8(b);
        let mut result = Wtf8Buf::with_capacity(sa.len() + sb.len());
        result.push_wtf8(sa);
        result.push_wtf8(sb);
        w_str_from_wtf8(result) as i64
    }
}

#[majit_macros::elidable]
pub extern "C" fn jit_str_repeat(s: i64, n: i64) -> i64 {
    let s = s as PyObjectRef;
    unsafe {
        let sv = w_str_get_wtf8(s);
        let count = if n < 0 { 0 } else { n as usize };
        let mut result = Wtf8Buf::with_capacity(sv.len() * count);
        for _ in 0..count {
            result.push_wtf8(sv);
        }
        w_str_from_wtf8(result) as i64
    }
}

#[majit_macros::elidable]
pub extern "C" fn jit_str_compare(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        // WTF-8 byte order matches code point order, so the byte
        // comparison yields the same result as comparing code points.
        let sa = w_str_get_wtf8(a);
        let sb = w_str_get_wtf8(b);
        match sa.as_bytes().cmp(sb.as_bytes()) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

#[majit_macros::elidable]
pub extern "C" fn jit_str_is_true(s: i64) -> i64 {
    let s = s as PyObjectRef;
    unsafe { (w_str_len(s) != 0) as i64 }
}

/// `str(i)` over an unboxed integer: render `i` to its decimal
/// `W_UnicodeObject`.  The argument is a raw machine integer (the `'i'`
/// argcode operand), not a boxed object pointer.
///
/// `rint.py:rtype_str` / `rstr.py ll_int2dec` lower `str(int)` to a
/// `direct_call` of the decimal-render helper during rtyping, so the
/// blackhole never dispatches a bare `int_str` op.  Pyre keeps `str(x)`
/// as a graph-level `UnaryOp { op: "str" }`; `jtransform` lowers the
/// Int-operand form to a residual call here (the Ref-operand form is
/// identity, mirroring `ll_str` on a string).
#[majit_macros::elidable]
pub extern "C" fn jit_int_str(v: i64) -> i64 {
    w_str_new(&v.to_string()) as i64
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
        assert_eq!(UNICODE_VALUE_OFFSET, 16);
        assert_eq!(UNICODE_BYTE_LEN_OFFSET, 24);
        assert_eq!(UNICODE_LEN_OFFSET, 32);
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
    fn test_str_byte_len_vs_char_len() {
        let obj = w_str_new("café");
        unsafe {
            let str_obj = obj as *const W_UnicodeObject;
            assert_eq!((*str_obj).byte_len, 5); // UTF-8: c(1) a(1) f(1) é(2)
            assert_eq!((*str_obj).len, 4); // 4 codepoints
        }
    }

    #[test]
    fn test_box_str_constant_reuses_same_object() {
        let a = box_str_constant(Wtf8::new("pyre"));
        let b = box_str_constant(Wtf8::new("pyre"));
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

    #[test]
    fn test_jit_int_str_renders_decimal() {
        unsafe {
            assert_eq!(w_str_get_value(jit_int_str(0) as PyObjectRef), "0");
            assert_eq!(w_str_get_value(jit_int_str(123) as PyObjectRef), "123");
            assert_eq!(w_str_get_value(jit_int_str(-7) as PyObjectRef), "-7");
            assert_eq!(
                w_str_get_value(jit_int_str(i64::MIN) as PyObjectRef),
                "-9223372036854775808",
            );
        }
    }
}
