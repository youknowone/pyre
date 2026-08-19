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
    /// `self.map` (`mapdict.py:907`) — the interned map node's address held as
    /// a raw word, not a pointer.  Map nodes are interned, shared per type and
    /// never freed, so the GC neither owns nor traces this slot
    /// (`object_object_custom_trace` walks only `storage`).  A pointer-typed
    /// field would make the translator lower every read of it to a `Ref`,
    /// which is what the JIT reserves for GC references; the word spelling
    /// keeps the one field on one kind.  Cast to `MapRef` at each use.
    pub map: usize,
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
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn w_instance_new(w_type: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the allocation below. `w_type` is a `W_TypeObject`
    // (`pyre-object::typeobject` GC type id 33) — user-defined types
    // are stable old-gen GC objects, so the pinned typeptr remains a live,
    // non-moving GC reference across the instance allocation. The
    // `is_in_nursery` filter in the walker (`majit-gc/src/collector.rs`)
    // keeps the built-in static `PyType` case (e.g. `INT_TYPE`) untouched.
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_type);

    let obj = alloc_instance_object(W_ObjectObject {
        ob_header: PyObject {
            ob_type: &INSTANCE_TYPE as *const PyType,
            w_class: w_type,
        },
        // `mapdict.py:758-761 user_setup` → `_mapdict_init_empty(
        // w_subtype.terminator)` (`mapdict.py:908-910`): the instance map is
        // the owning type's terminator from construction, and `storage = None`.
        //
        // Reading it here rather than installing it on first attribute access
        // is what makes `_get_mapdict_map`'s `jit.promote(self.map)` promotable.
        // A deferred install leaves every fresh instance at zero until the
        // first access, so the promoted map guard the JIT bakes — recorded
        // AFTER that install, hence naming the terminator — cannot hold on the
        // next iteration's fresh instance. It then fails on every pass through
        // a loop that constructs an object and touches an attribute, and each
        // failure is a full deopt.
        //
        // Zero stays legal: `pyre-object` cannot build a terminator (it lives
        // in the interpreter's mapdict layer and `pyre-object` must not depend
        // on it), so a type whose terminator has not been created yet still
        // gets one from `ensure_mapdict_initialized` on first access — which
        // also stores it on the type, so every later instance is eager.
        //
        // The `is_type` test is what makes reading the field safe: `w_type` is
        // a `W_TypeObject` on every interpreter path, but the allocator is also
        // driven with a sentinel that has no type layout behind it, and the
        // terminator field would be read off whatever that address points at.
        map: unsafe {
            if crate::typeobject::is_type(w_type) {
                crate::typeobject::w_type_get_terminator(w_type) as usize
            } else {
                0
            }
        },
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
/// (`rclass`/`gctypelayout` standard `GcStruct`). Pyre allocates them
/// through the stable (non-moving) old-gen allocator instead.
///
/// This note used to record a SIGSEGV as the reason — a movable instance
/// read from `objs[i % 3]` and carried into a method-call guard reading a
/// stale pointer out of the deadframe, reproduced by
/// `synth/inheritance_dispatch` — and named "extend the trace GC-safepoint
/// gcmap to cover transient Ref slots" as the convergence path. That crash
/// no longer reproduces: with this call switched to `try_gc_alloc`,
/// `inheritance_dispatch` passes at the default, 256 KB and 128 KB nursery
/// sizes, the whole `check.py` corpus passes on both backends, and seven
/// GC-sensitive fixtures produce byte-identical output under a 128 KB
/// nursery.
///
/// The switch is not made because it measures nothing. Allocation is about
/// a twentieth of what instantiating an object costs — the rest is call
/// dispatch — and the host-side "nursery" allocator is non-collecting and
/// falls back to old-gen once the nursery is full
/// (`majit-backend-dynasm/src/runner.rs` `dynasm_alloc_nursery_typed`), so
/// the flip only reorders "try nursery, else old-gen" against "always
/// old-gen": minor and major collection counts and peak RSS come out
/// unchanged, and every non-microbenchmark fixture moves within noise.
/// Real convergence needs host-side allocation that may collect, which in
/// turn needs every allocation site to root the raw pointers it holds.
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
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_instance_get_type(obj: PyObjectRef) -> PyObjectRef {
    (*obj).w_class
}

/// Check if an object is an instance of a user-defined class.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
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
