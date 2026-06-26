//! Native `memoryview` object — a flattened port of PyPy's
//! `W_MemoryView` (`pypy/objspace/std/memoryobject.py`) over its
//! interp-level `BufferView` (`pypy/interpreter/buffer.py`).
//!
//! PyPy splits the view across two layers (`W_MemoryView` → `BufferView`
//! → byte-level `Buffer`) for RPython's type system; at runtime the
//! behaviour is a single view over a byte backing, so this collapses both
//! layers into one `#[pyre_class]` struct (the same blessed structural
//! adaptation as the other native collapses).  The struct holds only
//! `PyObjectRef` + scalar fields so the GC tracer reaches every reference
//! through the macro-derived `ptr_offsets`; shape/strides ride as Python
//! tuple objects rather than inline `Vec`s.
//!
//! Byte access reads/writes the LIVE backing object (`w_backing`) through
//! the existing accessors — no copy — so a view observes later mutations
//! to its source and writes through to it, fixing the dict-stub's
//! detached-`to_vec` staleness bug.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// A `memoryview` over a contiguous byte backing.
///
/// `w_obj` is the original exporter (the `.obj` property); `w_backing` is
/// the object actually read/written (bytes / bytearray / array.array) —
/// for a plain view the two coincide, but a chained cast/slice keeps
/// `w_backing` pinned to the root storage while `w_obj` still reports the
/// exporter.  `offset`/`length` bound the live byte window inside
/// `w_backing`; `w_shape`/`w_strides` are `tuple[int]` (1-D: `(count,)` /
/// `(itemsize,)`).  `released` flips on `release()` / context-manager exit.
#[pyre_class("memoryview", static_name = "MEMORYVIEW")]
pub struct W_MemoryView {
    pub w_obj: PyObjectRef,
    pub w_backing: PyObjectRef,
    pub w_format: PyObjectRef,
    pub w_shape: PyObjectRef,
    pub w_strides: PyObjectRef,
    pub itemsize: i64,
    pub ndim: i64,
    pub offset: i64,
    pub length: i64,
    pub readonly: bool,
    pub released: bool,
}

/// Allocate a `W_MemoryView` from already-acquired view parameters.  Every
/// `PyObjectRef` is pinned before `allocate`, which can trigger a
/// collection that would otherwise move/free a ref held only in a Rust
/// local (mirrors `w_range_new`).
#[allow(clippy::too_many_arguments)]
pub fn w_memoryview_alloc(
    w_obj: PyObjectRef,
    w_backing: PyObjectRef,
    w_format: PyObjectRef,
    w_shape: PyObjectRef,
    w_strides: PyObjectRef,
    itemsize: i64,
    ndim: i64,
    offset: i64,
    length: i64,
    readonly: bool,
    released: bool,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_obj);
    crate::gc_roots::pin_root(w_backing);
    crate::gc_roots::pin_root(w_format);
    crate::gc_roots::pin_root(w_shape);
    crate::gc_roots::pin_root(w_strides);
    W_MemoryView::allocate(W_MemoryView {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_obj,
        w_backing,
        w_format,
        w_shape,
        w_strides,
        itemsize,
        ndim,
        offset,
        length,
        readonly,
        released,
    })
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_w_memoryview(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &MEMORYVIEW_TYPE) }
}

macro_rules! mv_obj_accessor {
    ($name:ident, $field:ident) => {
        /// # Safety
        /// `obj` must point to a valid `W_MemoryView`.
        #[inline]
        pub unsafe fn $name(obj: PyObjectRef) -> PyObjectRef {
            unsafe { (*(obj as *const W_MemoryView)).$field }
        }
    };
}
macro_rules! mv_scalar_accessor {
    ($name:ident, $field:ident, $ty:ty) => {
        /// # Safety
        /// `obj` must point to a valid `W_MemoryView`.
        #[inline]
        pub unsafe fn $name(obj: PyObjectRef) -> $ty {
            unsafe { (*(obj as *const W_MemoryView)).$field }
        }
    };
}

mv_obj_accessor!(w_memoryview_obj, w_obj);
mv_obj_accessor!(w_memoryview_backing, w_backing);
mv_obj_accessor!(w_memoryview_format, w_format);
mv_obj_accessor!(w_memoryview_shape, w_shape);
mv_obj_accessor!(w_memoryview_strides, w_strides);
mv_scalar_accessor!(w_memoryview_itemsize, itemsize, i64);
mv_scalar_accessor!(w_memoryview_ndim, ndim, i64);
mv_scalar_accessor!(w_memoryview_offset, offset, i64);
mv_scalar_accessor!(w_memoryview_length, length, i64);
mv_scalar_accessor!(w_memoryview_readonly, readonly, bool);
mv_scalar_accessor!(w_memoryview_released, released, bool);

/// Mark the view released.  `released` is an inline scalar, so no write
/// barrier is needed (a barrier guards `PyObjectRef` stores only).
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_set_released(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_MemoryView)).released = true;
    }
}

/// `strides[0]` — the signed byte step between consecutive elements of a
/// 1-D view.  A contiguous view has `strides[0] == itemsize`; a strided
/// slice (`m[::2]`, `m[::-1]`) carries `parent_stride * step`, possibly
/// negative.  Falls back to `itemsize` when the tuple is unexpectedly empty.
///
/// Byte gathering that honours this stride lives in the interpreter crate
/// (`builtins::memoryview_gather_bytes`) because a subclass-safe backing
/// read needs `isinstance_w`, which pyre-object must not depend on.
///
/// # Safety
/// `obj` must point to a valid `W_MemoryView`.
#[inline]
pub unsafe fn w_memoryview_stride0(obj: PyObjectRef) -> i64 {
    unsafe {
        let strides = (*(obj as *const W_MemoryView)).w_strides;
        match crate::tupleobject::w_tuple_getitem(strides, 0) {
            Some(s) => crate::intobject::w_int_get_value(s),
            None => (*(obj as *const W_MemoryView)).itemsize,
        }
    }
}
