//! W_InstanceObject — instance of a user-defined class.
//!
//! PyPy equivalent: pypy/objspace/std/objectobject.py → W_ObjectObject
//!
//! An instance holds a pointer to its W_TypeObject (class) in `ob_header.w_class`.
//! Per-instance attributes are stored in the thread-local ATTR_TABLE
//! side table, matching PyPy's instance __dict__.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python instance object.
///
/// Layout: `[ob_type | w_class]` — same as PyObject header.
///
/// - `ob_type`: always &INSTANCE_TYPE (for is_instance() checks)
/// - `w_class`: pointer to the W_TypeObject this is an instance of
///
/// The Python class is stored in `ob_header.w_class`, shared with all
/// other object types. RPython stores this in `typeptr` (rclass.py).
#[repr(C)]
pub struct W_InstanceObject {
    pub ob_header: PyObject,
}

/// `W_InstanceObject` is the bare `[ob_header]` instance for user-
/// defined classes. Its size matches `size_of::<PyObject>()` and it
/// shares the GC type id of `INSTANCE_TYPE` (the `object` root —
/// `OBJECT_GC_TYPE_ID = 0`). Registering it as a separate id would
/// duplicate the root of the inheritance hierarchy and break the
/// preorder-range invariants in `subclass_range`. Allocation goes
/// through the untyped [`malloc`] for now; a future pass that
/// teaches `malloc_typed` about parent-id aliasing can re-enable a
/// typed entry point.
///
/// Fixed payload size (`framework.py:811`).
pub const W_INSTANCE_OBJECT_SIZE: usize = std::mem::size_of::<W_InstanceObject>();

/// Allocate a new instance of a user-defined class.
///
/// PyPy equivalent: object.__new__(space, w_type) → allocate_instance
pub fn w_instance_new(w_type: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the `lltype::malloc` call below. `w_type` is a `W_TypeObject`
    // (`pyre-object::typeobject` GC type id 33) — user-defined types
    // are allocated through `malloc_typed`, so the typeptr is a live
    // GC reference across the instance allocation. The `is_in_nursery`
    // filter in the walker (`majit-gc/src/collector.rs:764`) keeps the
    // built-in static `PyType` case (e.g. `INT_TYPE`) untouched.
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_type);

    crate::lltype::malloc(W_InstanceObject {
        ob_header: PyObject {
            ob_type: &INSTANCE_TYPE as *const PyType,
            w_class: w_type,
        },
    }) as PyObjectRef
}

/// Get the class (W_TypeObject) of an instance.
pub unsafe fn w_instance_get_type(obj: PyObjectRef) -> PyObjectRef {
    (*obj).w_class
}

/// Check if an object is an instance of a user-defined class.
#[inline]
pub unsafe fn is_instance(obj: PyObjectRef) -> bool {
    py_type_check(obj, &INSTANCE_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_create_and_check() {
        // Use a sentinel as the "type"
        let fake_type = PY_NULL;
        let obj = w_instance_new(fake_type);
        unsafe {
            assert!(is_instance(obj));
            assert!(!is_int(obj));
            assert!(!crate::typeobject::is_type(obj));
            assert_eq!(w_instance_get_type(obj), fake_type);
        }
    }
}
