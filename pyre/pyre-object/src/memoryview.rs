//! Native `memoryview` object — `W_MemoryView`
//! (`pypy/objspace/std/memoryobject.py`) over a [`BufferView`]
//! (`pypy/interpreter/buffer.py`) over a byte-level
//! [`Buffer`](crate::buffer::Buffer) (`rpython/rlib/buffer.py`).
//!
//! `W_MemoryView` holds an off-heap `*const BufferView`; the view carries the
//! geometry plus the backing `Buffer`, so the three layers stay distinct
//! rather than collapsed into one struct.  Byte access reads/writes the LIVE
//! backing object through the view's `Buffer` — no copy — so a view observes
//! later mutations of its source and writes through to it.
//!
//! Because the refs the collector must keep alive (the backing exporter, the
//! `.obj` exporter, the format / shape / strides objects) live *inside* the
//! off-heap view rather than as inline `PyObjectRef` fields, the macro-derived
//! `ptr_offsets` cannot reach them; the JIT driver registers
//! `memoryview_object_custom_trace` (`pyre-jit/src/eval.rs`) to walk the view,
//! mirroring `W_ListObject` / `W_TupleObject`'s off-block trace.

use crate::bufferview::BufferView;
use crate::pyobject::*;
use pyre_macros::pyre_class;

/// A `memoryview` — a view over a byte backing, behind an off-heap
/// [`BufferView`].
#[pyre_class("memoryview", static_name = "MEMORYVIEW")]
pub struct W_MemoryView {
    /// `self.view` (`memoryobject.py`).  Never null for a live memoryview; the
    /// geometry and backing live here, off the GC heap, reached by the custom
    /// trace.
    pub view: *const BufferView,
    /// Flips on `release()` / context-manager exit.
    pub released: bool,
}

/// Allocate a [`BufferView`] off the GC heap (a stationary `Box`, like a
/// list's `ItemsBlock`).  The collector reaches its refs through
/// `memoryview_object_custom_trace` and never moves the box.
pub fn bufferview_alloc(view: BufferView) -> *const BufferView {
    Box::into_raw(Box::new(view)) as *const BufferView
}

/// Allocate the `W_MemoryView` header in GC old-gen with no view attached
/// yet (mirrors `w_set_new` allocating an empty body).
///
/// Old-gen (`try_gc_alloc_stable`, non-moving mark-sweep) so the object
/// carries `TRACK_YOUNG_PTRS` and `memoryview_object_custom_trace` runs on a
/// minor collection once the header sits in the remembered set — without
/// that, the managed `format` / `shape` / `strides` objects reachable only
/// through the view's off-heap box would be swept.  Allocating the header
/// *before* the ref-bearing box keeps no half-built box alive across the
/// collection this call may trigger: the caller re-reads its pinned refs
/// from the shadow stack (post-relocation) and attaches the box with
/// [`w_memoryview_set_view`].  Falls back to `malloc_typed` when no GC hook
/// is installed (unit tests).
pub fn w_memoryview_alloc_header(released: bool) -> PyObjectRef {
    let payload = W_MemoryView {
        ob: PyObject {
            ob_type: &MEMORYVIEW_TYPE as *const PyType,
            w_class: crate::pyobject::get_instantiate(&MEMORYVIEW_TYPE),
        },
        view: std::ptr::null(),
        released,
    };
    match crate::gc_hook::try_gc_alloc_stable(
        <W_MemoryView as crate::lltype::GcType>::type_id(),
        <W_MemoryView as crate::lltype::GcType>::SIZE,
    )
    .filter(|p| !p.is_null())
    {
        Some(raw) => {
            unsafe { std::ptr::write(raw as *mut W_MemoryView, payload) };
            raw as PyObjectRef
        }
        None => crate::lltype::malloc_typed(payload) as PyObjectRef,
    }
}

/// Attach the off-heap view to a header from [`w_memoryview_alloc_header`]
/// and fire the GC write barrier so the old-gen `W_MemoryView` enters the
/// remembered set; `memoryview_object_custom_trace` then forwards the view's
/// refs on the next minor collection.  Mirrors `set_write_barrier` after a
/// store into a set's off-heap items.
///
/// # Safety
/// `mv` must point to a `W_MemoryView` from [`w_memoryview_alloc_header`] and
/// `view` to a live [`bufferview_alloc`] box whose `PyObjectRef`s are already
/// the post-collection (relocated) pointers.
pub unsafe fn w_memoryview_set_view(mv: PyObjectRef, view: *const BufferView) {
    unsafe {
        (*(mv as *mut W_MemoryView)).view = view;
    }
    crate::gc_hook::try_gc_write_barrier(mv as *mut u8);
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_w_memoryview(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &MEMORYVIEW_TYPE) }
}

/// The off-heap view backing `obj`.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView` whose `view` is live.  Once
/// `w_memoryview_set_released` drops the box and nulls `view`, this
/// dereferences a null pointer — every accessor must gate on
/// `w_memoryview_released` first (as the interpreter methods do).
#[inline]
pub unsafe fn w_memoryview_view(obj: PyObjectRef) -> &'static BufferView {
    unsafe { &*(*(obj as *const W_MemoryView)).view }
}

macro_rules! mv_view_obj {
    ($name:ident, $accessor:ident) => {
        /// # Safety
        /// `obj` must point to a valid `W_MemoryView`.
        #[inline]
        pub unsafe fn $name(obj: PyObjectRef) -> PyObjectRef {
            unsafe { w_memoryview_view(obj).$accessor() }
        }
    };
}
macro_rules! mv_view_scalar {
    ($name:ident, $accessor:ident, $ty:ty) => {
        /// # Safety
        /// `obj` must point to a valid `W_MemoryView`.
        #[inline]
        pub unsafe fn $name(obj: PyObjectRef) -> $ty {
            unsafe { w_memoryview_view(obj).$accessor() }
        }
    };
}

/// The exporter actually read/written (bytes / bytearray / array.array) — the
/// `.obj` of the root storage.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_backing(obj: PyObjectRef) -> PyObjectRef {
    unsafe { w_memoryview_view(obj).backing().w_obj() }
}

mv_view_obj!(w_memoryview_obj, w_obj);
mv_view_obj!(w_memoryview_format, w_format);
mv_view_obj!(w_memoryview_shape, w_shape);
mv_view_obj!(w_memoryview_strides, w_strides);
mv_view_scalar!(w_memoryview_itemsize, itemsize, i64);
mv_view_scalar!(w_memoryview_ndim, ndim, i64);
mv_view_scalar!(w_memoryview_offset, offset, i64);
mv_view_scalar!(w_memoryview_length, length, i64);
mv_view_scalar!(w_memoryview_readonly, readonly, bool);

/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_released(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_MemoryView)).released }
}

/// Release the view: drop the off-heap `BufferView` box (reclaiming any
/// nested `Buffer::Sub` boxes through `Box`'s recursive drop glue) and null
/// `view`, then flip `released`.  Mirrors `descr_release` clearing
/// `self.view = None` (`memoryobject.py`) so the backing / view graph is
/// dropped eagerly on release rather than lingering until the header is
/// GC-collected.  Idempotent: a second call finds `view` already null.  The
/// `released` flag is an inline scalar, so no write barrier is needed (a
/// barrier guards `PyObjectRef` stores only).
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_set_released(obj: PyObjectRef) {
    unsafe {
        let mv = obj as *mut W_MemoryView;
        let view_ptr = (*mv).view as *mut BufferView;
        if !view_ptr.is_null() {
            drop(Box::from_raw(view_ptr));
            (*mv).view = std::ptr::null();
        }
        (*mv).released = true;
    }
}

/// `strides[0]` — the signed byte step between consecutive elements of a
/// 1-D view.  A contiguous view has `strides[0] == itemsize`; a strided
/// slice (`m[::2]`, `m[::-1]`) carries `parent_stride * step`, possibly
/// negative.  Falls back to `itemsize` when the tuple is unexpectedly empty.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_stride0(obj: PyObjectRef) -> i64 {
    unsafe {
        let view = w_memoryview_view(obj);
        match crate::tupleobject::w_tuple_getitem(view.w_strides(), 0) {
            Some(s) => crate::intobject::w_int_get_value(s),
            None => view.itemsize(),
        }
    }
}
