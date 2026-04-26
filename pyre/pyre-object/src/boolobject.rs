//! W_BoolObject — Python `bool` type.
//!
//! Booleans are a subtype of int in Python. In pyre, they use a separate
//! struct for JIT type specialization via `GuardClass`.

use crate::pyobject::*;

/// Python boolean object.
#[repr(C)]
pub struct W_BoolObject {
    pub ob_header: PyObject,
    pub boolval: bool,
}

/// Field offset of `boolval` within `W_BoolObject`, for JIT field access.
pub const BOOL_BOOLVAL_OFFSET: usize = std::mem::offset_of!(W_BoolObject, boolval);

/// Fixed payload size (`framework.py:811`).
pub const W_BOOL_OBJECT_SIZE: usize = std::mem::size_of::<W_BoolObject>();

impl crate::lltype::GcType for W_BoolObject {
    /// Mirrors `pyre_jit_trace::descr::W_BOOL_GC_TYPE_ID`. Re-stating the
    /// constant here would re-introduce the cross-crate dependency the
    /// crate split was meant to avoid; the JIT init asserts the registered
    /// id matches the descr constant, so any drift surfaces there.
    ///
    /// Note: there are no `malloc_typed::<W_BoolObject>` callers — every
    /// bool flows through the `TRUE_SINGLETON` / `FALSE_SINGLETON`
    /// statics via [`w_bool_from`]. The impl is kept as a consistency
    /// anchor for the GC registration's `debug_assert_eq!(_, SIZE)` so
    /// the singleton struct layout cannot drift from the registered
    /// type info.
    const TYPE_ID: u32 = 5;
    const SIZE: usize = W_BOOL_OBJECT_SIZE;
}

/// Extract the bool value from a known W_BoolObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_BoolObject`.
#[inline]
pub unsafe fn w_bool_get_value(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_BoolObject)).boolval }
}

// ── Bool singletons ──────────────────────────────────────────────────
//
// pypy/objspace/std/objspace.py:61 installs `space.w_True` /
// `space.w_False` as singletons; every PyPy `space.newbool(value)`
// call (pypy/interpreter/baseobjspace.py:893 `newbool`) returns one of
// the two pre-allocated objects. pyre mirrors the singleton model with
// two `static W_BoolObject` instances and routes all callers through
// [`w_bool_from`].

static TRUE_SINGLETON: W_BoolObject = W_BoolObject {
    ob_header: PyObject {
        ob_type: &BOOL_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
    boolval: true,
};

static FALSE_SINGLETON: W_BoolObject = W_BoolObject {
    ob_header: PyObject {
        ob_type: &BOOL_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
    boolval: false,
};

/// Get a boolean PyObjectRef from a bool value.
///
/// Returns a pointer to a pre-allocated static singleton,
/// avoiding heap allocation on every comparison/branch.
#[inline]
pub fn w_bool_from(value: bool) -> PyObjectRef {
    if value {
        (&TRUE_SINGLETON as *const W_BoolObject).cast_mut() as PyObjectRef
    } else {
        (&FALSE_SINGLETON as *const W_BoolObject).cast_mut() as PyObjectRef
    }
}

// ── W_BoolObject.descr_and/or/xor (boolobject.py:54-76) ──────────────
//
// PyPy _make_bitwise_binop:
//     def descr_binop(self, space, w_other):
//         if not isinstance(w_other, W_BoolObject):
//             return int_op(self, space, w_other)
//         a = bool(self.intval)
//         b = bool(w_other.intval)
//         return space.newbool(op(a, b))
//
// The `isinstance(self, W_BoolObject)` dispatch happens on the caller
// side (space.and_) — these helpers assume both operands are bool.

/// boolobject.py:74 descr_and — both operands W_BoolObject.
#[inline]
pub unsafe fn bool_descr_and(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_bool_from(w_bool_get_value(a) & w_bool_get_value(b)) }
}

/// boolobject.py:75 descr_or — both operands W_BoolObject.
#[inline]
pub unsafe fn bool_descr_or(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_bool_from(w_bool_get_value(a) | w_bool_get_value(b)) }
}

/// boolobject.py:76 descr_xor — both operands W_BoolObject.
#[inline]
pub unsafe fn bool_descr_xor(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_bool_from(w_bool_get_value(a) ^ w_bool_get_value(b)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_true() {
        let obj = w_bool_from(true);
        unsafe {
            assert!(is_bool(obj));
            // bool is a subclass of int, so is_int(bool) is true.
            assert!(is_int(obj));
            assert!(w_bool_get_value(obj));
        }
    }

    #[test]
    fn test_bool_false() {
        let obj = w_bool_from(false);
        unsafe {
            assert!(!w_bool_get_value(obj));
        }
    }

    /// `w_bool_from` returns one of the two static singletons —
    /// every call with the same value yields the same address.
    /// pypy/objspace/std/objspace.py:61 installs `space.w_True` /
    /// `space.w_False` with the same identity invariant.
    #[test]
    fn test_bool_singleton_identity() {
        let a = w_bool_from(true);
        let b = w_bool_from(true);
        let c = w_bool_from(false);
        let d = w_bool_from(false);
        assert!(std::ptr::eq(a, b), "w_bool_from(true) is not a singleton");
        assert!(std::ptr::eq(c, d), "w_bool_from(false) is not a singleton");
        assert!(!std::ptr::eq(a, c));
    }
}
