//! W_IntObject — Python `int` type backed by i64.
//!
//! Phase 1 uses a fixed i64 representation. BigInt support (like PyPy's
//! `W_LongObject`) will be added in Phase 4.

use std::sync::LazyLock;

use crate::pyobject::*;

/// Python integer object.
///
/// Layout: `[ob_type: *const PyType | intval: i64]`
/// The JIT reads `intval` via `GetfieldGcI` at offset 8 (after the type pointer).
#[repr(C)]
pub struct W_IntObject {
    pub ob_header: PyObject,
    pub intval: i64,
}

/// Field offset of `intval` within `W_IntObject`, for JIT field access.
pub const INT_INTVAL_OFFSET: usize = std::mem::offset_of!(W_IntObject, intval);

// ── Small-int cache ──────────────────────────────────────────────────
//
// Pre-allocate W_IntObject for values -5..=256 (262 entries, matching
// CPython's NSMALLPOSINTS/NSMALLNEGINTS). This eliminates heap allocation
// for the most common integer values — constants, loop counters, and
// small arithmetic results.

pub const SMALL_INT_MIN: i64 = -5;
pub const SMALL_INT_MAX: i64 = 256;
pub const W_INT_OBJECT_SIZE: usize = std::mem::size_of::<W_IntObject>();

static SMALL_INTS: LazyLock<Vec<W_IntObject>> = LazyLock::new(|| {
    (SMALL_INT_MIN..=SMALL_INT_MAX)
        .map(|v| W_IntObject {
            ob_header: PyObject {
                ob_type: &INT_TYPE as *const PyType,
            },
            intval: v,
        })
        .collect()
});

/// Create or retrieve a W_IntObject for the given value.
///
/// Values in -5..=256 are returned from a pre-allocated cache (no heap
/// allocation). Values outside this range are heap-allocated via Box::leak.
#[inline]
pub fn w_int_new(value: i64) -> PyObjectRef {
    if value >= SMALL_INT_MIN && value <= SMALL_INT_MAX {
        let idx = (value - SMALL_INT_MIN) as usize;
        (&SMALL_INTS[idx] as *const W_IntObject).cast_mut() as PyObjectRef
    } else {
        let obj = Box::new(W_IntObject {
            ob_header: PyObject {
                ob_type: &INT_TYPE as *const PyType,
            },
            intval: value,
        });
        Box::into_raw(obj) as PyObjectRef
    }
}

/// Create a W_IntObject bypassing the small-int cache.
///
/// Used for int subclass instances that need unique object identity
/// (so per-object attributes in ATTR_TABLE don't collide).
pub fn w_int_new_unique(value: i64) -> PyObjectRef {
    let obj = Box::new(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
        },
        intval: value,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Return the address of INT_TYPE for JIT type-id validation.
#[inline]
pub fn w_int_type_id() -> usize {
    &INT_TYPE as *const PyType as usize
}

/// Extract the i64 value from a known W_IntObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_IntObject`.
#[inline]
pub unsafe fn w_int_get_value(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_IntObject)).intval }
}

pub extern "C" fn jit_w_int_new(value: i64) -> i64 {
    w_int_new(value) as i64
}

#[inline]
pub fn w_int_small_cached(value: i64) -> bool {
    (SMALL_INT_MIN..=SMALL_INT_MAX).contains(&value)
}

#[inline]
pub fn w_int_small_cache_base_ptr() -> PyObjectRef {
    SMALL_INTS.as_ptr().cast_mut() as PyObjectRef
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_create_and_read() {
        let obj = w_int_new(42);
        unsafe {
            assert!(is_int(obj));
            assert!(!is_bool(obj));
            assert_eq!(w_int_get_value(obj), 42);
        }
    }

    #[test]
    fn test_int_negative() {
        let obj = w_int_new(-7);
        unsafe {
            assert_eq!(w_int_get_value(obj), -7);
        }
    }

    #[test]
    fn test_int_field_offset() {
        assert_eq!(INT_INTVAL_OFFSET, 8); // after *const PyType (8 bytes on 64-bit)
    }

    #[test]
    fn test_small_int_cache_returns_same_pointer() {
        let a = w_int_new(42);
        let b = w_int_new(42);
        assert_eq!(a, b, "cached ints should return the same pointer");
    }

    #[test]
    fn test_small_int_cache_boundary() {
        // Values at cache boundaries
        let low = w_int_new(SMALL_INT_MIN);
        let high = w_int_new(SMALL_INT_MAX);
        unsafe {
            assert_eq!(w_int_get_value(low), SMALL_INT_MIN);
            assert_eq!(w_int_get_value(high), SMALL_INT_MAX);
        }
        // Same pointer on second call
        assert_eq!(low, w_int_new(SMALL_INT_MIN));
        assert_eq!(high, w_int_new(SMALL_INT_MAX));
    }

    #[test]
    fn test_large_int_not_cached() {
        let a = w_int_new(1000);
        let b = w_int_new(1000);
        assert_ne!(a, b, "large ints should be different heap allocations");
        unsafe {
            assert_eq!(w_int_get_value(a), 1000);
            assert_eq!(w_int_get_value(b), 1000);
            drop(Box::from_raw(a as *mut W_IntObject));
            drop(Box::from_raw(b as *mut W_IntObject));
        }
    }
}
