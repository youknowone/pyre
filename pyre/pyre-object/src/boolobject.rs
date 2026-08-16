//! W_BoolObject — Python `bool` type.
//!
//! `W_BoolObject` is a subclass of `W_IntObject` (`boolobject.py:30`), so a
//! bool holds the same `intval: i64` field at the same offset as an int
//! (0 for `False`, 1 for `True`). The two are distinguished only by the
//! `&BOOL_TYPE` vtable, which lets `GuardClass` specialize on the concrete
//! class while every `intval` field read stays layout-compatible with
//! `W_IntObject`.

use crate::pyobject::*;

/// Python boolean object.
#[repr(C)]
pub struct W_BoolObject {
    pub ob_header: PyObject,
    pub intval: i64,
}

/// Field offset of `intval` within `W_BoolObject`, for JIT field access.
/// Layout-identical to `INT_INTVAL_OFFSET` (`bool` inherits `intval`).
pub const BOOL_INTVAL_OFFSET: usize = std::mem::offset_of!(W_BoolObject, intval);

const _: () = {
    assert!(BOOL_INTVAL_OFFSET == crate::intobject::INT_INTVAL_OFFSET);
};

/// Fixed payload size (`framework.py:811`).
pub const W_BOOL_OBJECT_SIZE: usize = std::mem::size_of::<W_BoolObject>();

impl crate::lltype::GcType for W_BoolObject {
    /// Mirrors `pyre_jit_trace::descr::W_BOOL_GC_TYPE_ID`. Re-stating the
    /// constant here would re-introduce the cross-crate dependency the
    /// crate split was meant to avoid; the JIT init asserts the registered
    /// id matches the descr constant, so any drift surfaces there.
    ///
    /// Every bool flows through the two process-global prebuilt allocations
    /// owned by [`w_bool_from`].
    fn type_id() -> u32 {
        5
    }
    const SIZE: usize = W_BOOL_OBJECT_SIZE;
}

/// Extract the bool value from a known W_BoolObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_BoolObject`.
#[inline]
pub unsafe fn w_bool_get_value(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_BoolObject)).intval != 0 }
}

// ── Bool singletons ──────────────────────────────────────────────────
//
// pypy/objspace/std/objspace.py:61 installs `space.w_True` /
// `space.w_False` as singletons; every PyPy `space.newbool(value)`
// call (pypy/interpreter/baseobjspace.py:893 `newbool`) returns one of
// the two pre-allocated objects. pyre mirrors the singleton model with
// two process-global prebuilt objects and routes all callers through
// [`w_bool_from`]. Their host allocations carry the same GC header as an
// RPython translated prebuilt object.

static TRUE_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static FALSE_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

#[majit_macros::dont_look_inside]
fn bool_singleton(slot: &'static std::sync::OnceLock<usize>, intval: i64) -> PyObjectRef {
    *slot.get_or_init(|| {
        crate::lltype::malloc_typed_immortal(W_BoolObject {
            ob_header: PyObject {
                ob_type: &BOOL_TYPE as *const PyType,
                w_class: std::ptr::null_mut(),
            },
            intval,
        }) as usize
    }) as PyObjectRef
}

/// Get a boolean PyObjectRef from a bool value.
///
/// Returns a pointer to a pre-allocated static singleton,
/// avoiding heap allocation on every comparison/branch.
/// Both singletons are immortal and their addresses never change after the
/// first materialisation, so the result depends only on `value` and the
/// call cannot raise.
#[majit_macros::elidable_cannot_raise]
#[inline]
pub fn w_bool_from(value: bool) -> *mut PyObject {
    if value {
        bool_singleton(&TRUE_SINGLETON, 1)
    } else {
        bool_singleton(&FALSE_SINGLETON, 0)
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
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bool_descr_and(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_bool_from(w_bool_get_value(a) & w_bool_get_value(b)) }
}

/// boolobject.py:75 descr_or — both operands W_BoolObject.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bool_descr_or(a: PyObjectRef, b: PyObjectRef) -> PyObjectRef {
    unsafe { w_bool_from(w_bool_get_value(a) | w_bool_get_value(b)) }
}

/// boolobject.py:76 descr_xor — both operands W_BoolObject.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
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

    /// `w_bool_from` returns one of the two prebuilt singletons —
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
        unsafe {
            let hdr = majit_gc::header::header_of(a as usize);
            assert_eq!((*hdr).type_id(), 5);
            assert!((*hdr).has_flag(majit_gc::flags::NO_HEAP_PTRS));
        }
    }
}
