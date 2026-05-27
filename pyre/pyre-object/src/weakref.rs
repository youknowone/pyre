//! Host-side WEAKREF allocator + dereference.
//!
//! Thin shim over `majit_gc::weakref::Weakref` for pyre-object callers
//! (typeobject.weak_subclasses, future W_Weakref / WeakrefLifeline).
//! `pyre-object` cannot depend on `majit-gc`, so this module mirrors
//! the upstream WEAKREF layout (gctypelayout.py:587
//! `WEAKREF = lltype.GcStruct("weakref", ("weakptr", llmemory.Address))`)
//! and relies on the runtime GC type registration in `pyre-jit::eval`
//! plus the GC's invalidate_*_weakrefs hooks to keep the slot
//! coherent across collections.

use crate::gc_hook::try_gc_alloc;
use crate::pyobject::*;

/// GC type id for the WEAKREF GcStruct. Registered by
/// `pyre-jit::eval::init` after `W_INT_MUTABLE_CELL` and before the
/// per-exception kind loop. A `debug_assert_eq!` in the registration
/// site pins the id to this value so callers can treat it as a
/// constant.
pub const WEAKREF_GC_TYPE_ID: u32 = 51;

/// `WEAKREF = lltype.GcStruct("weakref", ("weakptr", llmemory.Address))`
/// (gctypelayout.py:587). Single `weakptr` slot at offset 0 — the
/// majit-gc collector reads / writes this slot during
/// `invalidate_young_weakrefs` / `invalidate_old_weakrefs`.
#[repr(C)]
pub struct Weakref {
    pub weakptr: PyObjectRef,
}

/// `sizeof_weakref = llmemory.sizeof(WEAKREF)` (gctypelayout.py:589).
pub const SIZEOF_WEAKREF: usize = std::mem::size_of::<Weakref>();

/// Allocate a new WEAKREF GcStruct via the active GC and initialise
/// its single `weakptr` slot to `target`. Returns null when no GC
/// hook is installed on this thread (test environments that did not
/// wire `pyre-jit`).
///
/// # Safety
///
/// Caller must ensure `target` outlives the weakref or accept that a
/// subsequent collection will null the slot.
pub unsafe fn w_weakref_new(target: PyObjectRef) -> *mut Weakref {
    let Some(payload) = try_gc_alloc(WEAKREF_GC_TYPE_ID, SIZEOF_WEAKREF) else {
        return std::ptr::null_mut();
    };
    if payload.is_null() {
        return std::ptr::null_mut();
    }
    let wref = payload as *mut Weakref;
    unsafe { (*wref).weakptr = target };
    wref
}

/// `ll_weakref_deref(wref)` (gctypelayout.py:594-596). Reads the
/// `weakptr` slot. Returns null when the GC has already invalidated
/// the target during a minor / major cycle (incminimark.py:3068-3079
/// / :3116-3122).
///
/// # Safety
///
/// `wref` must point to a live WEAKREF GcStruct.
pub unsafe fn w_weakref_deref(wref: *const Weakref) -> PyObjectRef {
    if wref.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*wref).weakptr }
}

// ── W_GcWeakref wrapper ──────────────────────────────────────────────
//
// pyre's `interp_weakref.rs` simulates PyPy's W_WeakrefBase /
// WeakrefLifeline subclasses on top of `W_InstanceObject` + ATTR_*
// instance-dict slots (TODO: bring to parity). Instance-dict slots
// can only hold `PyObjectRef`, not a raw `*mut Weakref`, so this
// tiny W_Root wraps the rweakref pointer for storage in those slots.
//
// A faithful port would replace the W_InstanceObject simulation with
// typed W_Root subclasses carrying inline `*mut Weakref` fields (the
// shape PyPy's W_Weakref / WeakrefLifeline use). That refactor is
// out of scope here; this wrapper restores correct weak semantics
// without touching the simulation layer.

/// Internal type tag — used by `py_type_check` to recognise a
/// `W_GcWeakref` PyObject when it surfaces through a generic slot.
pub static GC_WEAKREF_TYPE: PyType = new_pytype("__GcWeakref");

/// GC type id assigned to `W_GcWeakref` — slot 52, immediately after
/// `WEAKREF_GC_TYPE_ID=51`.
pub const W_GC_WEAKREF_GC_TYPE_ID: u32 = 52;

#[repr(C)]
pub struct W_GcWeakref {
    pub ob_header: PyObject,
    /// Strong pointer to a `Weakref` GcStruct. The GC traces this slot
    /// (see `W_GC_WEAKREF_GC_PTR_OFFSETS`) so the Weakref struct itself
    /// survives across collections; the `weakptr` slot inside the
    /// Weakref is invalidated separately by
    /// `invalidate_young_weakrefs` / `invalidate_old_weakrefs`.
    pub inner: *mut Weakref,
}

pub const W_GC_WEAKREF_OBJECT_SIZE: usize = std::mem::size_of::<W_GcWeakref>();

/// Byte offset of the inline `*mut Weakref` field the GC must trace
/// (as a strong GcRef) during minor / major collection. Mirrors the
/// `W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS` convention on celldict.rs:120.
pub const W_GC_WEAKREF_GC_PTR_OFFSETS: [usize; 1] = [std::mem::offset_of!(W_GcWeakref, inner)];

impl crate::lltype::GcType for W_GcWeakref {
    fn type_id() -> u32 {
        W_GC_WEAKREF_GC_TYPE_ID
    }
    const SIZE: usize = W_GC_WEAKREF_OBJECT_SIZE;
}

/// Allocate a `W_GcWeakref` wrapping a fresh rweakref to `target`.
/// Returns null when no GC hook is installed (test environments that
/// did not wire `pyre-jit`) or when `target` itself is null.
pub fn w_gc_weakref_new(target: PyObjectRef) -> PyObjectRef {
    if target.is_null() {
        return std::ptr::null_mut();
    }
    let inner = unsafe { w_weakref_new(target) };
    if inner.is_null() {
        return std::ptr::null_mut();
    }
    crate::lltype::malloc_typed(W_GcWeakref {
        ob_header: PyObject {
            ob_type: &GC_WEAKREF_TYPE as *const PyType,
            w_class: get_instantiate(&GC_WEAKREF_TYPE),
        },
        inner,
    }) as PyObjectRef
}

/// `isinstance(obj, W_GcWeakref)` predicate.
///
/// # Safety
///
/// `obj` must be a valid (possibly null) PyObjectRef.
#[inline]
pub unsafe fn is_gc_weakref(obj: PyObjectRef) -> bool {
    !obj.is_null() && unsafe { py_type_check(obj, &GC_WEAKREF_TYPE) }
}

/// Dereference a `W_GcWeakref` slot. Returns the original target if
/// still alive, or null after the GC invalidated the underlying
/// rweakref. Returns null for null / non-W_GcWeakref inputs so callers
/// can use the same code path for "uninitialised slot" / "dead
/// referent".
///
/// # Safety
///
/// `obj` must be a valid (possibly null) PyObjectRef.
pub unsafe fn w_gc_weakref_deref(obj: PyObjectRef) -> PyObjectRef {
    if !unsafe { is_gc_weakref(obj) } {
        return std::ptr::null_mut();
    }
    let wref = unsafe { (*(obj as *const W_GcWeakref)).inner };
    unsafe { w_weakref_deref(wref) }
}

/// Allocate a W_GcWeakref for `target`, falling back to a strong
/// PyObjectRef when no GC hook is installed (unit-test environments
/// that did not wire `pyre-jit`). The strong-ref fallback restores
/// the historical instance-dict-slot behavior for tests while
/// production paths get real weak semantics.
///
/// Pair with `w_gc_weakref_or_strong_deref` on the reader side.
pub fn w_gc_weakref_new_or_strong(target: PyObjectRef) -> PyObjectRef {
    let wrapped = w_gc_weakref_new(target);
    if wrapped.is_null() { target } else { wrapped }
}

/// Read a slot written by `w_gc_weakref_new_or_strong`. When the slot
/// holds a W_GcWeakref, deref through the GC weakref. Otherwise treat
/// the slot itself as a strong PyObjectRef (the no-GC fallback path).
///
/// # Safety
///
/// `slot` must be a valid (possibly null) PyObjectRef.
pub unsafe fn w_gc_weakref_or_strong_deref(slot: PyObjectRef) -> PyObjectRef {
    if slot.is_null() {
        return std::ptr::null_mut();
    }
    if unsafe { is_gc_weakref(slot) } {
        return unsafe { w_gc_weakref_deref(slot) };
    }
    slot
}
