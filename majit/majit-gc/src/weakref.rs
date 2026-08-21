//! Weak reference primitive.
//!
//! Mirrors `rpython/memory/gctypelayout.py` — the WEAKREF
//! GcStruct that the framework GC special-cases at minor and major
//! collection time. RPython:
//!
//! ```python
//! WEAKREF = lltype.GcStruct("weakref", ("weakptr", llmemory.Address))
//! WEAKREFPTR = lltype.Ptr(WEAKREF)
//! sizeof_weakref = llmemory.sizeof(WEAKREF)
//! empty_weakref = lltype.malloc(WEAKREF, immortal=True)
//! empty_weakref.weakptr = llmemory.NULL
//! weakptr_offset = llmemory.offsetof(WEAKREF, "weakptr")
//!
//! def ll_weakref_deref(wref):
//!     wref = llmemory.cast_weakrefptr_to_ptr(WEAKREFPTR, wref)
//!     return wref.weakptr
//! ```
//!
//! The payload is a single `weakptr` slot — a raw `GcRef`-shaped
//! address that the collector **does not** follow when discovering
//! live objects. Instead, the collector walks `T_IS_WEAKREF`-tagged
//! objects during its own cycle and invalidates `weakptr` (sets it to
//! null) for any target it didn't otherwise reach. The minor and major
//! collection cycles wire that side; this module is the data layout
//! the cycles operate on.

use majit_ir::GcRef;

/// `WEAKREF = lltype.GcStruct("weakref", ("weakptr", llmemory.Address))`
/// (gctypelayout.py:587).
///
/// `#[repr(C)]` so `weakptr` sits at offset 0 of the payload, matching
/// the `weakptr_offset = llmemory.offsetof(WEAKREF, "weakptr")` the
/// framework reads (gctypelayout.py:592).
#[repr(C)]
pub struct Weakref {
    pub weakptr: GcRef,
}

/// `sizeof_weakref = llmemory.sizeof(WEAKREF)` (gctypelayout.py).
pub const SIZEOF_WEAKREF: usize = std::mem::size_of::<Weakref>();

/// `weakptr_offset = llmemory.offsetof(WEAKREF, "weakptr")`
/// (gctypelayout.py:592). The framework GC reads / writes the weakptr
/// slot at this offset off the WEAKREF payload base, and the JIT
/// lowering of `weakref_deref` emits a single load at this offset.
pub const WEAKPTR_OFFSET: usize = std::mem::offset_of!(Weakref, weakptr);

/// `ll_weakref_deref(wref)` (gctypelayout.py).
///
/// Reads the `weakptr` slot of a WEAKREF struct. Returns the target
/// `GcRef`, or a null `GcRef` if the target has been invalidated by
/// the collector.
///
/// # Safety
///
/// `wref` must point to a WEAKREF struct (i.e. its `TypeInfo` was
/// created via `TypeInfo::weakref()`).
pub unsafe fn ll_weakref_deref(wref: *const Weakref) -> GcRef {
    unsafe { (*wref).weakptr }
}
