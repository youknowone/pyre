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

impl crate::lltype::GcType for Weakref {
    fn type_id() -> u32 {
        WEAKREF_GC_TYPE_ID
    }
    const SIZE: usize = SIZEOF_WEAKREF;
}

/// Allocate a new WEAKREF GcStruct via the active GC and initialise its single
/// `weakptr` slot to `target`. Always returns a non-null, usable weakref:
/// `weakref.ref(...)` never fails in PyPy, and under
/// `translation.rweakref=False` it is implemented as a strong reference
/// (rweakref.py:11-16). When no GC hook is installed yet (pre-build bootstrap,
/// e.g. a module-level `class B(A)` evaluated before the JIT GC is wired) or the
/// GC reports OOM, fall back to a Box-immortal `Weakref`: a never-collected slot
/// whose `weakptr` stays valid — exactly the rweakref-off strong-ref mode.
///
/// # Safety
///
/// Caller must ensure `target` outlives the weakref or accept that a
/// subsequent collection will null the slot (GC-allocated path only; the
/// Box-immortal bootstrap slot is never cleared).
pub unsafe fn w_weakref_new(target: PyObjectRef) -> *mut Weakref {
    if let Some(payload) = try_gc_alloc(WEAKREF_GC_TYPE_ID, SIZEOF_WEAKREF) {
        if payload.is_null() {
            // GC OOM — fall through to the immortal bootstrap below.
        } else {
            let wref = payload as *mut Weakref;
            unsafe { (*wref).weakptr = target };
            return wref;
        }
    }
    crate::lltype::malloc_typed(Weakref { weakptr: target })
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

// ── GcWeakrefBox wrapper ──────────────────────────────────────────────
//
// pyre's `interp__weakref.rs` simulates PyPy's W_WeakrefBase /
// WeakrefLifeline subclasses on top of `W_ObjectObject` + ATTR_*
// instance-dict slots (TODO: bring to parity). Instance-dict slots
// can only hold `PyObjectRef`, not a raw `*mut Weakref`, so this
// tiny internal PyObject wraps the rweakref pointer for storage in those slots.
//
// A faithful port would replace the W_ObjectObject simulation with
// typed W_Root subclasses carrying inline `*mut Weakref` fields (the
// shape PyPy's W_Weakref / WeakrefLifeline use). That refactor is
// out of scope here; this wrapper restores correct weak semantics
// without touching the simulation layer.

/// Internal type tag — used by `py_type_check` to recognise a
/// `GcWeakrefBox` PyObject when it surfaces through a generic slot.
pub static GC_WEAKREF_BOX_TYPE: PyType = new_pytype("__GcWeakrefBox");

/// GC type id assigned to `GcWeakrefBox` — slot 52, immediately after
/// `WEAKREF_GC_TYPE_ID=51`.
pub const GC_WEAKREF_BOX_GC_TYPE_ID: u32 = 52;

#[repr(C)]
pub struct GcWeakrefBox {
    pub ob_header: PyObject,
    /// Strong pointer to a `Weakref` GcStruct. The GC traces this slot
    /// (see `GC_WEAKREF_BOX_GC_PTR_OFFSETS`) so the Weakref struct itself
    /// survives across collections; the `weakptr` slot inside the
    /// Weakref is invalidated separately by
    /// `invalidate_young_weakrefs` / `invalidate_old_weakrefs`.
    pub inner: *mut Weakref,
}

pub const GC_WEAKREF_BOX_OBJECT_SIZE: usize = std::mem::size_of::<GcWeakrefBox>();

/// Byte offset of the inline `*mut Weakref` field the GC must trace
/// (as a strong GcRef) during minor / major collection. Mirrors the
/// `W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS` convention on celldict.rs:120.
pub const GC_WEAKREF_BOX_GC_PTR_OFFSETS: [usize; 1] = [std::mem::offset_of!(GcWeakrefBox, inner)];

impl crate::lltype::GcType for GcWeakrefBox {
    fn type_id() -> u32 {
        GC_WEAKREF_BOX_GC_TYPE_ID
    }
    const SIZE: usize = GC_WEAKREF_BOX_OBJECT_SIZE;
}

thread_local! {
    /// Every `GcWeakrefBox` ever allocated. The box is `malloc_typed`
    /// (immortal, off-GC) so the collector never traces into it, which
    /// would leave its `inner` `*mut Weakref` slot un-relocated /
    /// un-retained across a collection — the boxed Weakref would be swept
    /// (or moved without updating `inner`) and `w_gc_weakref_box_deref`
    /// would read a dangling slot. Walking these `inner` slots as roots
    /// (see [`walk_gc_weakref_box_inner_roots`]) keeps each boxed Weakref
    /// alive and relocates the slot in place, exactly as the signal
    /// handler-table walker does for its immortal dict.
    static WEAKREF_BOXES: std::cell::RefCell<Vec<*mut GcWeakrefBox>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Allocate a `GcWeakrefBox` wrapping a fresh rweakref to `target`.
/// Returns null when no GC hook is installed (test environments that
/// did not wire `pyre-jit`) or when `target` itself is null.
pub fn w_gc_weakref_box_new(target: PyObjectRef) -> PyObjectRef {
    if target.is_null() {
        return std::ptr::null_mut();
    }
    let inner = unsafe { w_weakref_new(target) };
    if inner.is_null() {
        return std::ptr::null_mut();
    }
    let boxed = crate::lltype::malloc_typed(GcWeakrefBox {
        ob_header: PyObject {
            ob_type: &GC_WEAKREF_BOX_TYPE as *const PyType,
            w_class: get_instantiate(&GC_WEAKREF_BOX_TYPE),
        },
        inner,
    });
    WEAKREF_BOXES.with(|b| b.borrow_mut().push(boxed));
    boxed as PyObjectRef
}

/// Visit each immortal box's `inner` Weakref pointer as a strong GC root.
/// The collector keeps the Weakref alive and rewrites the slot after a
/// relocation; its `weakptr` is still cleared independently by
/// `invalidate_young_weakrefs` / `invalidate_old_weakrefs` when the
/// weakly-referenced target dies, so weak semantics are preserved while
/// the box's `inner` slot stays coherent.
pub fn walk_gc_weakref_box_inner_roots(mut visitor: impl FnMut(&mut PyObjectRef)) {
    WEAKREF_BOXES.with(|b| {
        for &boxed in b.borrow().iter() {
            if boxed.is_null() {
                continue;
            }
            let inner_slot = unsafe { std::ptr::addr_of_mut!((*boxed).inner) } as *mut PyObjectRef;
            visitor(unsafe { &mut *inner_slot });
        }
    });
}

/// `isinstance(obj, GcWeakrefBox)` predicate.
///
/// # Safety
///
/// `obj` must be a valid (possibly null) PyObjectRef.
#[inline]
pub unsafe fn is_gc_weakref_box(obj: PyObjectRef) -> bool {
    !obj.is_null() && unsafe { py_type_check(obj, &GC_WEAKREF_BOX_TYPE) }
}

/// Dereference a `GcWeakrefBox` slot. Returns the original target if
/// still alive, or null after the GC invalidated the underlying
/// rweakref. Returns null for null / non-GcWeakrefBox inputs so callers
/// can use the same code path for "uninitialised slot" / "dead
/// referent".
///
/// # Safety
///
/// `obj` must be a valid (possibly null) PyObjectRef.
pub unsafe fn w_gc_weakref_box_deref(obj: PyObjectRef) -> PyObjectRef {
    if !unsafe { is_gc_weakref_box(obj) } {
        return std::ptr::null_mut();
    }
    let wref = unsafe { (*(obj as *const GcWeakrefBox)).inner };
    unsafe { w_weakref_deref(wref) }
}

/// Allocate a GcWeakrefBox for `target`, falling back to a strong
/// PyObjectRef when no GC hook is installed (unit-test environments
/// that did not wire `pyre-jit`). The strong-ref fallback restores
/// the historical instance-dict-slot behavior for tests while
/// production paths get real weak semantics.
///
/// Pair with `w_gc_weakref_box_or_strong_deref` on the reader side.
pub fn w_gc_weakref_box_new_or_strong(target: PyObjectRef) -> PyObjectRef {
    let wrapped = w_gc_weakref_box_new(target);
    if wrapped.is_null() { target } else { wrapped }
}

/// Read a slot written by `w_gc_weakref_box_new_or_strong`. When the slot
/// holds a GcWeakrefBox, deref through the GC weakref. Otherwise treat
/// the slot itself as a strong PyObjectRef (the no-GC fallback path).
///
/// # Safety
///
/// `slot` must be a valid (possibly null) PyObjectRef.
pub unsafe fn w_gc_weakref_box_or_strong_deref(slot: PyObjectRef) -> PyObjectRef {
    if slot.is_null() {
        return std::ptr::null_mut();
    }
    if unsafe { is_gc_weakref_box(slot) } {
        return unsafe { w_gc_weakref_box_deref(slot) };
    }
    slot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_weakref_new_pre_gc_returns_strong_immortal_ref() {
        // No GC hook is wired in a bare pyre-object unit test, so try_gc_alloc
        // returns None. w_weakref_new must still hand back a usable, non-null
        // weakref whose deref yields the target — PyPy's weakref.ref never
        // fails (a strong reference under translation.rweakref=False). Before
        // the bootstrap fallback this returned null, which a module-level
        // `class B(A)` recorded into the base's weak_subclasses, dropping the
        // subclass from mutated()/get_subclasses().
        let target = 0xdead_beef_usize as PyObjectRef;
        let wref = unsafe { w_weakref_new(target) };
        assert!(!wref.is_null());
        assert_eq!(unsafe { w_weakref_deref(wref) }, target);
    }
}
