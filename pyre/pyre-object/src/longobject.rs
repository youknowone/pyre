//! W_LongObject -- arbitrary-precision integer backed by `BigInt`.
//!
//! Used when i64 overflow is detected in `W_IntObject` arithmetic.
//! The JIT never inlines bigint operations; `GuardClass(INT_TYPE)` rejects
//! `W_LongObject` and deoptimizes back to the interpreter.

use malachite_bigint::BigInt;

use crate::pyobject::*;

/// Arbitrary-precision integer object.
///
/// Layout: `[ob_type: *const PyType | value: *mut BigInt]`
/// The `value` pointer owns a heap-allocated `BigInt` (via `Box::into_raw`).
#[repr(C)]
pub struct W_LongObject {
    pub ob_header: PyObject,
    pub value: *mut BigInt,
}

// Safety: BigInt is Send+Sync and W_LongObject only stores a raw pointer
// that is effectively owned.
unsafe impl Send for W_LongObject {}
unsafe impl Sync for W_LongObject {}

/// Field offset of `value` within `W_LongObject`, for potential JIT field access.
pub const LONG_VALUE_OFFSET: usize = std::mem::offset_of!(W_LongObject, value);

/// GC type id assigned to `W_LongObject` at JitDriver init time.
pub const W_LONG_GC_TYPE_ID: u32 = 35;

/// Fixed payload size (`framework.py:811`).
pub const W_LONG_OBJECT_SIZE: usize = std::mem::size_of::<W_LongObject>();

impl crate::lltype::GcType for W_LongObject {
    fn type_id() -> u32 {
        W_LONG_GC_TYPE_ID
    }
    const SIZE: usize = W_LONG_OBJECT_SIZE;
}

/// Wrap an already heap-allocated `*mut BigInt` in a fresh W_LongObject
/// without copying the payload — the wrapper just stores `value`, it does not
/// take exclusive ownership. Pure-call CSE of the elidable `rbigint` helpers
/// can fold two ops to the same `*mut BigInt`, so one payload may back more
/// than one wrapper; that is sound only because payloads are never freed
/// (`malloc_raw` / `Box::leak`).
pub fn w_long_from_raw(value: *mut BigInt) -> PyObjectRef {
    // W_LongObject shares the `int` type with W_IntObject — the two only
    // differ in their storage layout, not their Python-level identity
    // (PyPy does the same via W_AbstractIntObject's typedef). Wire
    // `w_class` to INT_TYPE.instantiate so `type(x) is int` and
    // `isinstance(x, int)` both hold for long integers.
    crate::lltype::malloc_typed(W_LongObject {
        ob_header: PyObject {
            ob_type: &LONG_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        value,
    }) as PyObjectRef
}

/// Allocate a new W_LongObject on the heap from a `BigInt` value.
pub fn w_long_new(value: BigInt) -> PyObjectRef {
    w_long_from_raw(crate::lltype::malloc_raw(value))
}

/// Create a W_LongObject from an i64 value.
pub fn w_long_from_i64(v: i64) -> PyObjectRef {
    w_long_new(BigInt::from(v))
}

/// Box a bigint constant into a heap Python int object.
pub fn box_bigint_constant(value: &BigInt) -> PyObjectRef {
    w_long_new(value.clone())
}

/// `W_LongObject._fits_int()` — longobject.py:141 / rbigint.fits_int.
/// True if the value fits in a machine-word integer (i64 on 64-bit).
/// Used by `is_plain_int1` to accept long objects that are in the int range.
#[inline]
pub unsafe fn w_long_fits_int(obj: PyObjectRef) -> bool {
    unsafe {
        let big = w_long_get_value(obj);
        i64::try_from(big).is_ok()
    }
}

/// True when the W_LongObject's BigInt is zero. Divisor guard for the
/// can-raise floordiv/mod fast path (a zero divisor makes the payload helper
/// publish ZeroDivisionError, which the trait path defers to the generic
/// residual rather than triggering during tracing).
///
/// # Safety
/// `obj` must point to a valid `W_LongObject`.
#[inline]
pub unsafe fn w_long_is_zero(obj: PyObjectRef) -> bool {
    use malachite_bigint::Sign;
    unsafe { w_long_get_value(obj).sign() == Sign::NoSign }
}

/// Extract a reference to the BigInt value from a known W_LongObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_LongObject`.
#[inline]
pub unsafe fn w_long_get_value(obj: PyObjectRef) -> &'static BigInt {
    unsafe {
        let long_obj = obj as *const W_LongObject;
        &*(*long_obj).value
    }
}

/// `rbigint.fits_int()` (`rpython/rlib/rbigint.py:490`) — JIT-callable
/// wrapper. Returns 1 when the W_LongObject's BigInt fits in i64,
/// 0 otherwise. Used as the runtime fits_int guard before
/// `jit_w_long_toint`.
///
/// Unlike `rbigint.toint()`, upstream `fits_int()` is not marked
/// `@jit.elidable`, so keep this call cannot-raise but non-elidable.
pub extern "C" fn jit_w_long_fits_int(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    unsafe { w_long_fits_int(obj) as i64 }
}

/// `W_LongObject.toint()` (`pypy/objspace/std/longobject.py:138`) →
/// `rbigint.toint()` (`rpython/rlib/rbigint.py:465`, `@jit.elidable`).
/// Extract an i64 from a W_LongObject. RPython `toint` raises
/// `OverflowError` when the BigInt does not fit; the elidable
/// trace-time site emits a `fits_int` GUARD_TRUE first
/// (`pypy/objspace/std/listobject.py:2390 is_plain_int1` parity), so
/// the OverflowError path is unreachable in production. Pyre encodes
/// that unreachability as a panic. There is no `_int_w_unsafe` upstream —
/// this is the elidable `toint` after a `fits_int` guard.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_toint(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    unsafe {
        let big = w_long_get_value(obj);
        i64::try_from(big).unwrap_or_else(|_| {
            panic!("jit_w_long_toint: BigInt out of i64 range — fits_int guard violated")
        })
    }
}

/// `rbigint.add` (`rpython/rlib/rbigint.py:269`, `@jit.elidable`) — the
/// payload half of `W_LongObject._add` (`pypy/objspace/std/longobject.py:331`).
/// Both operands are guaranteed `W_LongObject` by a preceding
/// `GuardClass(LONG_TYPE)` on each, so the BigInt payloads are read
/// directly. Returns a freshly heap-allocated `*mut BigInt` (as i64) — the
/// arithmetic only, with no Python-object wrapper. `add` allocates a new
/// bigint, so its only failure mode is MemoryError: `EF_ELIDABLE_OR_MEMORYERROR`
/// (`call.py:294`, `cr == "mem"`). The value is still a pure function of the
/// operand payloads, so the optimizer may fold/CSE it; a trailing
/// `GuardNoException` covers the allocation. The result is an internal bigint
/// never exposed to Python `is`, so sharing one payload for two equal-input
/// adds is unobservable.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_add_raw(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        let sum = w_long_get_value(a) + w_long_get_value(b);
        crate::lltype::malloc_raw(sum) as i64
    }
}

/// `rbigint.sub` — the payload half of `W_LongObject._sub`
/// (`pypy/objspace/std/longobject.py`). Like [`jit_w_long_add_raw`] but
/// subtracts; allocates, so `EF_ELIDABLE_OR_MEMORYERROR`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_sub_raw(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe { crate::lltype::malloc_raw(w_long_get_value(a) - w_long_get_value(b)) as i64 }
}

/// `rbigint.mul` payload half of `W_LongObject._mul`. Allocates → `EF_ELIDABLE_OR_MEMORYERROR`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_mul_raw(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe { crate::lltype::malloc_raw(w_long_get_value(a) * w_long_get_value(b)) as i64 }
}

/// `rbigint.and_` payload half of `W_LongObject._and`. Allocates → `EF_ELIDABLE_OR_MEMORYERROR`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_and_raw(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe { crate::lltype::malloc_raw(w_long_get_value(a) & w_long_get_value(b)) as i64 }
}

/// `rbigint.or_` payload half of `W_LongObject._or`. Allocates → `EF_ELIDABLE_OR_MEMORYERROR`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_or_raw(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe { crate::lltype::malloc_raw(w_long_get_value(a) | w_long_get_value(b)) as i64 }
}

/// `rbigint.xor_` payload half of `W_LongObject._xor`. Allocates → `EF_ELIDABLE_OR_MEMORYERROR`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_xor_raw(a: i64, b: i64) -> i64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe { crate::lltype::malloc_raw(w_long_get_value(a) ^ w_long_get_value(b)) as i64 }
}

/// `rbigint` comparison payload for `W_LongObject` — returns the sign of
/// `a <=> b` as `-1` / `0` / `1`. RPython exposes the comparison as six methods
/// (`lt`/`le`/`eq`/`ne`/`gt`/`ge`, the latter built as `other.lt(self)`
/// wrappers, `rbigint.py:573/664`); Rust's total `Ord::cmp` collapses them into
/// one three-way result, and the caller recovers each relation with a plain
/// `int_<cmp>(sign, 0)` (e.g. `a < b` ⟺ `sign < 0`, `a == b` ⟺ `sign == 0`).
/// A comparison neither allocates nor raises, so this is
/// `EF_ELIDABLE_CANNOT_RAISE` and the fast path records `CallPure*` with NO
/// trailing guard.
#[majit_macros::elidable_cannot_raise]
pub extern "C" fn jit_w_long_cmp(a: i64, b: i64) -> i64 {
    use core::cmp::Ordering;
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        match w_long_get_value(a).cmp(w_long_get_value(b)) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }
}

/// `bigint_result` — wrap the bigint produced by [`jit_w_long_add_raw`] in a
/// Python int, demoting to `W_IntObject` when it fits in i64, otherwise
/// reusing the `*mut BigInt` payload in a fresh `W_LongObject`. This is the
/// `W_LongObject(...)` wrapper allocation that upstream keeps a residual `NEW`
/// outside the elidable `rbigint.add` (the int fast path boxes the same way,
/// via the `dont_look_inside` `jit_w_int_new`). Marked `dont_look_inside`, not
/// elidable, so the wrapper object is never pure-CSE'd and each add yields a
/// distinct boxed result, matching `W_LongObject(op(...))`.
///
/// The i64-range demotion to `W_IntObject` is pyre's two-class `int`
/// representation (small-int fast object + bigint object); PyPy's default
/// `newlong` (`longobject.py:495`, `withsmalllong=False`) keeps a
/// `W_LongObject`. Both denote the same `int` value — this is a representation
/// choice spanning every int path, not specific to this helper.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_bigint_result_box(num: i64) -> i64 {
    let num = num as *mut BigInt;
    unsafe {
        match i64::try_from(&*num) {
            Ok(v) => crate::intobject::w_int_new(v) as usize as i64,
            Err(_) => w_long_from_raw(num) as usize as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_create_and_read() {
        let obj = w_long_new(BigInt::from(42));
        unsafe {
            assert!(is_long(obj));
            assert!(!is_int(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(42));
        }
    }

    #[test]
    fn test_long_from_i64() {
        let obj = w_long_from_i64(i64::MAX);
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(i64::MAX));
        }
    }

    #[test]
    fn test_long_large_value() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big.clone());
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), big);
        }
    }

    #[test]
    fn test_long_field_offset() {
        assert_eq!(LONG_VALUE_OFFSET, 16);
    }

    #[test]
    fn test_long_type_name_is_int() {
        // Python users see "int" for both W_IntObject and W_LongObject
        assert_eq!(LONG_TYPE.name, "int");
    }

    #[test]
    fn test_jit_w_long_fits_int_in_range() {
        let obj = w_long_from_i64(123);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
        let obj = w_long_from_i64(i64::MAX);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
        let obj = w_long_from_i64(i64::MIN);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
    }

    #[test]
    fn test_jit_w_long_fits_int_out_of_range() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big);
        assert_eq!(jit_w_long_fits_int(obj as i64), 0);
        let big = BigInt::from(i64::MIN) - BigInt::from(1);
        let obj = w_long_new(big);
        assert_eq!(jit_w_long_fits_int(obj as i64), 0);
    }

    #[test]
    fn test_jit_w_long_toint_extracts_i64() {
        let obj = w_long_from_i64(42);
        assert_eq!(jit_w_long_toint(obj as i64), 42);
        let obj = w_long_from_i64(i64::MAX);
        assert_eq!(jit_w_long_toint(obj as i64), i64::MAX);
        let obj = w_long_from_i64(i64::MIN);
        assert_eq!(jit_w_long_toint(obj as i64), i64::MIN);
    }

    #[test]
    fn test_jit_w_long_add_raw_payload() {
        // The elidable half returns a bare `*mut BigInt` carrying the sum,
        // with no Python-object wrapper.
        let a = w_long_new(BigInt::from(i64::MAX));
        let b = w_long_new(BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a as i64, b as i64) as *mut BigInt;
        unsafe {
            assert_eq!(*raw, BigInt::from(i64::MAX) * 2);
        }
    }

    #[test]
    fn test_jit_w_long_binop_raw_payloads() {
        // sub/mul/and/or/xor raw helpers mirror jit_w_long_add_raw: bare
        // `*mut BigInt` carrying the arithmetic result, no Python wrapper.
        let x = BigInt::from(i64::MAX) + BigInt::from(7);
        let y = BigInt::from(i64::MAX) - BigInt::from(3);
        let a = w_long_new(x.clone());
        let b = w_long_new(y.clone());
        unsafe {
            let sub = jit_w_long_sub_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*sub, &x - &y);
            let mul = jit_w_long_mul_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*mul, &x * &y);
            let and = jit_w_long_and_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*and, &x & &y);
            let or = jit_w_long_or_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*or, &x | &y);
            let xor = jit_w_long_xor_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*xor, &x ^ &y);
        }
    }

    #[test]
    fn test_jit_bigint_result_box_keeps_long_out_of_range() {
        // Sum out of i64 range boxes as W_LongObject, reusing the payload.
        let a = w_long_new(BigInt::from(i64::MAX));
        let b = w_long_new(BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a as i64, b as i64);
        let r = jit_bigint_result_box(raw) as PyObjectRef;
        unsafe {
            assert!(is_long(r));
            assert_eq!(*w_long_get_value(r), BigInt::from(i64::MAX) * 2);
        }
    }

    #[test]
    fn test_jit_bigint_result_box_demotes_to_int_when_fits() {
        // `bigint_result` parity: a sum that fits in i64 demotes to W_IntObject
        // (so a later GuardClass(LONG_TYPE) on the result correctly side-exits).
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_long_new(BigInt::from(-1) - BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a as i64, b as i64);
        let r = jit_bigint_result_box(raw) as PyObjectRef;
        unsafe {
            assert!(is_int(r));
            assert!(!is_long(r));
            assert_eq!(crate::intobject::w_int_get_value(r), 0);
        }
    }
}
