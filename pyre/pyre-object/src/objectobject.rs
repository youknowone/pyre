//! W_ObjectObject — instance of a user-defined class.
//!
//! PyPy equivalent: pypy/objspace/std/objectobject.py → W_ObjectObject
//!
//! An instance holds a pointer to its W_TypeObject (class) in `ob_header.w_class`.
//! Per-instance attributes live in the mapdict `map`+`storage` pair
//! (`mapdict.py:907-910`), matching PyPy's instance attribute layout.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python instance object.
///
/// Layout: `[ob_type | w_class | map | storage]`.
///
/// - `ob_type`: always &INSTANCE_TYPE (for is_instance() checks)
/// - `w_class`: pointer to the W_TypeObject this is an instance of
/// - `map`: the attribute map (`AbstractAttribute` chain) — the
///   `self.map` of `MapdictStorageMixin` (`mapdict.py:907`)
/// - `storage`: the per-instance attribute storage list — the
///   `self.storage` of `MapdictStorageMixin` (`mapdict.py:910`)
///
/// The Python class is stored in `ob_header.w_class`, shared with all
/// other object types. RPython stores this in `typeptr` (rclass.py).
///
/// `map` is an erased `*const MapNode` (the objspace map node layer in
/// `objspace::std::mapdict` lives in the `pyre-interpreter` crate, which
/// `pyre-object` must not depend on; the interpreter side casts it back).
/// This mirrors the `W_DictObject.dstorage: *mut u8` erasure. A null
/// `map`/`storage` is the `_mapdict_init_empty` state with `storage =
/// None` (`mapdict.py:908-910`); the real terminator is installed by the
/// mapdict layer on first attribute access.
#[repr(C)]
pub struct W_ObjectObject {
    pub ob_header: PyObject,
    /// `self.map` (`mapdict.py:907`); erased `*const MapNode`.
    pub map: *const u8,
    /// `self.storage` (`mapdict.py:910`) — a `Ptr(GcArray(OBJECTPTR))` block of
    /// attribute values (`ItemsBlock`, tagged `W_MAPDICT_STORAGE_GC_TYPE_ID`).
    /// null = `None`, the `_mapdict_init_empty` empty state (`mapdict.py:910`).
    /// The block is a mixed boxed/unboxed array; the mapdict layer
    /// (`pyre-interpreter`) reads/writes it through `crate::object_array`
    /// helpers, and `object_object_custom_trace` walks its boxed slots.
    pub storage: *mut crate::object_array::ItemsBlock,
}

/// Fixed payload size of the `[ob_header | map | storage]` instance
/// payload (`framework.py:811`).
pub const W_OBJECT_OBJECT_SIZE: usize = std::mem::size_of::<W_ObjectObject>();

/// GC type id for the `W_ObjectObject` Rust struct. `pyre-jit::eval`
/// registers it through `object_subclass_with_custom_trace` with
/// `W_OBJECT_OBJECT_SIZE` + an `object_object_custom_trace` that
/// traces the off-heap `storage` value slots, so a collection keeps an
/// instance's attribute values reachable and reclaims dead instances.
///
/// This GC header id is a separate axis from the class-identity
/// preorder id `INSTANCE_TYPE` carries for `subclass_range`
/// (`gctypelayout` `get_type_id`/`fixedsize` vs `rclass`
/// `OBJECT.subclassrange_{min,max}`): the collector reads the header id
/// to find size + custom trace, while isinstance reads the `ob_type`
/// vtable. `INSTANCE_TYPE` therefore stays mapped to `object_tid`
/// (`OBJECT_GC_TYPE_ID = 0`) and this id is reachable only through the
/// GC header stamped by [`w_instance_new`].
pub const W_OBJECT_OBJECT_GC_TYPE_ID: u32 = 53;

/// Allocate a new instance of a user-defined class.
///
/// PyPy equivalent: object.__new__(space, w_type) → allocate_instance
pub fn w_instance_new(w_type: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the allocation below. `w_type` is a `W_TypeObject`
    // (`pyre-object::typeobject` GC type id 33) — user-defined types
    // are stable old-gen GC objects, so the pinned typeptr remains a live,
    // non-moving GC reference across the instance allocation. The
    // `is_in_nursery` filter in the walker (`majit-gc/src/collector.rs:764`)
    // keeps the built-in static `PyType` case (e.g. `INT_TYPE`) untouched.
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_type);

    let obj = alloc_instance_object(W_ObjectObject {
        ob_header: PyObject {
            ob_type: &INSTANCE_TYPE as *const PyType,
            w_class: w_type,
        },
        // `_mapdict_init_empty` (`mapdict.py:908-910`): `storage = None`.
        // The map terminator lives in the `pyre-interpreter` mapdict
        // layer and is installed there on first attribute access; a null
        // map is the not-yet-initialized empty state.
        map: std::ptr::null(),
        storage: std::ptr::null_mut(),
    });
    // objspace.py `allocate_instance`: types with `hasuserdel` register the
    // fresh instance on `space.finalizer_queue` immediately after allocation.
    crate::gc_hook::maybe_register_finalizer(obj);
    obj
}

/// Allocate a `W_ObjectObject` through the GC. The header is stamped
/// with [`W_OBJECT_OBJECT_GC_TYPE_ID`] so `object_object_custom_trace`
/// roots the `storage` value slots and dead instances are reclaimed.
/// Falls back to the leaking `lltype::malloc` `Box` when no GC hook is
/// installed (single-crate tests / pre-init snapshot tools).
///
/// PRE-EXISTING-ADAPTATION: PyPy instances live in the movable nursery
/// (`rclass`/`gctypelayout` standard `GcStruct`). Pyre uses the stable
/// (non-moving) old-gen allocator instead of `try_gc_alloc`, because the
/// JIT trace GC-safepoint gcmap does not yet forward a transient
/// instance ref held across an in-trace minor collection — a movable
/// instance read from e.g. `objs[i % 3]` and carried into a method-call
/// guard reads a stale (relocated) pointer out of the deadframe and
/// SIGSEGVs (`synth/inheritance_dispatch`; the interpreter path is fine,
/// `PYRE_NO_JIT=1` and stable allocation both pass). Convergence path:
/// extend the trace GC-safepoint liveness/gcmap (the `op_live`
/// subsystem) to cover transient Ref slots, then switch this call back
/// to `try_gc_alloc` for the movable nursery.
fn alloc_instance_object(value: W_ObjectObject) -> PyObjectRef {
    let raw =
        crate::gc_hook::try_gc_alloc_stable_raw(W_OBJECT_OBJECT_GC_TYPE_ID, W_OBJECT_OBJECT_SIZE);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut W_ObjectObject, value);
            raw as PyObjectRef
        }
    } else {
        crate::lltype::malloc(value) as PyObjectRef
    }
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
