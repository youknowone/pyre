//! View layer for the buffer protocol — the pyre analogue of
//! `pypy/interpreter/buffer.py`'s `BufferView`.  A `BufferView` carries the
//! geometry (offset / shape / strides / format / itemsize) over a byte-level
//! [`Buffer`] and gathers the live logical bytes in C order, honouring a
//! strided or N-D layout, without detaching a copy of the backing.
//!
//! `memoryview`'s `W_MemoryView` holds one of these off the GC heap; the GC
//! reaches the refs inside (the backing exporter, the `.obj` exporter, and
//! the format / shape / strides objects) through `W_MemoryView`'s custom
//! trace.  The format / shape / strides ride as their Python objects so the
//! `W_MemoryView` accessors stay pure reads; lowering them to native Rust
//! `str` / `Vec` (the `SimpleView` / `RawBufferView` subclass split) is a
//! later slice.

use crate::buffer::Buffer;
use crate::pyobject::PyObjectRef;

/// `_copy_base` — push one `isz`-wide element at byte offset `base`, dropping
/// it when the address falls outside the backing (a reversed / strided slice
/// past the end), so the gather never panics.
fn copy_base(full: &[u8], base: i64, isz: usize, out: &mut Vec<u8>) {
    if isz > 0 && base >= 0 && base as usize + isz <= full.len() {
        let b = base as usize;
        out.extend_from_slice(&full[b..b + isz]);
    }
}

/// `_copy_rec` — recursive C-order copy of dimension `idim`.  The innermost
/// dimension walks `shape[ndim-1]` elements by `strides[ndim-1]`; an outer
/// dimension recurses `shape[idim]` times, advancing `off` by `strides[idim]`.
fn copy_rec(
    full: &[u8],
    shape: &[i64],
    strides: &[i64],
    ndim: i64,
    idim: i64,
    mut off: i64,
    isz: usize,
    out: &mut Vec<u8>,
) {
    let dimshape = shape.get(idim as usize).copied().unwrap_or(0);
    let dimstride = strides.get(idim as usize).copied().unwrap_or(0);
    if idim == ndim - 1 {
        if dimstride == 0 {
            return;
        }
        for _ in 0..dimshape {
            copy_base(full, off, isz, out);
            off += dimstride;
        }
    } else {
        for _ in 0..dimshape {
            copy_rec(full, shape, strides, ndim, idim + 1, off, isz, out);
            off += dimstride;
        }
    }
}

/// Read a `tuple[int]` (shape or strides) into a native vector.
///
/// # Safety
/// `t` must point to a valid tuple of ints.
unsafe fn read_dims(t: PyObjectRef) -> Vec<i64> {
    unsafe {
        let n = crate::tupleobject::w_tuple_len(t);
        (0..n)
            .map(|i| {
                crate::tupleobject::w_tuple_getitem(t, i as i64)
                    .map(|w| crate::intobject::w_int_get_value(w))
                    .unwrap_or(0)
            })
            .collect()
    }
}

/// A view of a [`Buffer`]'s bytes with offset / shape / stride geometry and a
/// buffer-protocol format.
///
/// PyPy splits this into a class hierarchy — `SimpleView` / `RawBufferView`
/// (1-D), `BufferSlice` (strided), `BufferView1D` / `BufferViewND` (cast) —
/// each carrying only the state it needs and deriving the rest.  The single
/// [`Strided`](BufferView::Strided) variant is the general case that holds
/// every geometry field explicitly; the specialised variants peel off in
/// later slices, each routing its own constructor and deriving geometry
/// GC-safely.
pub enum BufferView {
    /// General strided / N-dimensional view over the root [`Buffer`], carrying
    /// absolute `offset` / `shape` / `strides` geometry.
    Strided {
        /// Byte storage actually read / written (the root exporter's buffer).
        backing: Buffer,
        /// The exporter reported by `memoryview.obj` — coincides with the
        /// backing for a plain view, but a chained cast / slice keeps the root
        /// storage in `backing` while `w_obj` still reports the original
        /// exporter.
        w_obj: PyObjectRef,
        /// Format string object (`memoryview.format`).
        w_format: PyObjectRef,
        /// Shape tuple (`memoryview.shape`).
        w_shape: PyObjectRef,
        /// Strides tuple (`memoryview.strides`).
        w_strides: PyObjectRef,
        itemsize: i64,
        ndim: i64,
        offset: i64,
        length: i64,
        readonly: bool,
    },
    /// `SimpleView` (`pypy/interpreter/buffer.py:270`) — a plain contiguous
    /// 1-D byte view (`bytes` / `bytearray`).  Format `'B'`, itemsize 1,
    /// ndim 1, offset 0, shape `[length]`, strides `[1]` are all derived;
    /// `readonly` comes from the backing, so nothing but the exporter refs is
    /// stored.
    Simple {
        backing: Buffer,
        w_obj: PyObjectRef,
        length: i64,
    },
    /// `RawBufferView` (`buffer.py:231`) — a typed contiguous 1-D view
    /// (`array.array`).  Format / itemsize are explicit; ndim 1, offset 0,
    /// shape `[length / itemsize]` (`[0]` when empty), strides `[itemsize]`
    /// derive; `readonly` comes from the backing.
    Raw {
        backing: Buffer,
        w_obj: PyObjectRef,
        w_fmt: PyObjectRef,
        itemsize: i64,
        length: i64,
    },
}

impl BufferView {
    /// The backing byte storage (the root exporter's [`Buffer`]).
    #[inline]
    pub fn backing(&self) -> &Buffer {
        match self {
            BufferView::Strided { backing, .. }
            | BufferView::Simple { backing, .. }
            | BufferView::Raw { backing, .. } => backing,
        }
    }
    /// The exporter reported by `memoryview.obj`.
    #[inline]
    pub fn w_obj(&self) -> PyObjectRef {
        match self {
            BufferView::Strided { w_obj, .. }
            | BufferView::Simple { w_obj, .. }
            | BufferView::Raw { w_obj, .. } => *w_obj,
        }
    }
    /// The element format string (`getformat`), read natively — the callers
    /// that need a Python `str` wrap a fresh one at the `descr` boundary.  A
    /// `Simple` view derives `'B'`; a `Raw` view reads its explicit format.
    ///
    /// # Safety
    /// The view's stored format object must be a live `str`.
    #[inline]
    pub unsafe fn format_str(&self) -> &'static str {
        unsafe {
            match self {
                BufferView::Strided { w_format, .. } => crate::w_str_get_value(*w_format),
                BufferView::Simple { .. } => "B",
                BufferView::Raw { w_fmt, .. } => crate::w_str_get_value(*w_fmt),
            }
        }
    }
    /// The per-dimension element counts (`getshape`) as native extents.  A
    /// `Simple` view is `[length]`; a `Raw` view is `[length / itemsize]`
    /// (`[0]` when empty, `buffer.py:254`).
    ///
    /// # Safety
    /// The view's stored shape object must be a live tuple of ints.
    #[inline]
    pub unsafe fn native_shape(&self) -> Vec<i64> {
        unsafe {
            match self {
                BufferView::Strided { w_shape, .. } => read_dims(*w_shape),
                BufferView::Simple { length, .. } => vec![*length],
                BufferView::Raw {
                    itemsize, length, ..
                } => {
                    if *length == 0 {
                        vec![0]
                    } else {
                        vec![*length / *itemsize]
                    }
                }
            }
        }
    }
    /// The per-dimension byte steps (`getstrides`) as native extents.  A
    /// `Simple` view is `[1]`; a `Raw` view is `[itemsize]`.
    ///
    /// # Safety
    /// The view's stored strides object must be a live tuple of ints.
    #[inline]
    pub unsafe fn native_strides(&self) -> Vec<i64> {
        unsafe {
            match self {
                BufferView::Strided { w_strides, .. } => read_dims(*w_strides),
                BufferView::Simple { .. } => vec![1],
                BufferView::Raw { itemsize, .. } => vec![*itemsize],
            }
        }
    }
    /// `strides[0]` — the signed byte step between consecutive elements of a
    /// 1-D view, falling back to `itemsize` when the strides are empty.
    ///
    /// # Safety
    /// The view's stored strides object must be a live tuple of ints.
    #[inline]
    pub unsafe fn stride0(&self) -> i64 {
        unsafe {
            match self {
                BufferView::Strided {
                    w_strides,
                    itemsize,
                    ..
                } => crate::tupleobject::w_tuple_getitem(*w_strides, 0)
                    .map(|s| crate::intobject::w_int_get_value(s))
                    .unwrap_or(*itemsize),
                BufferView::Simple { .. } => 1,
                BufferView::Raw { itemsize, .. } => *itemsize,
            }
        }
    }
    #[inline]
    pub fn itemsize(&self) -> i64 {
        match self {
            BufferView::Strided { itemsize, .. } | BufferView::Raw { itemsize, .. } => *itemsize,
            BufferView::Simple { .. } => 1,
        }
    }
    #[inline]
    pub fn ndim(&self) -> i64 {
        match self {
            BufferView::Strided { ndim, .. } => *ndim,
            BufferView::Simple { .. } | BufferView::Raw { .. } => 1,
        }
    }
    #[inline]
    pub fn offset(&self) -> i64 {
        match self {
            BufferView::Strided { offset, .. } => *offset,
            BufferView::Simple { .. } | BufferView::Raw { .. } => 0,
        }
    }
    #[inline]
    pub fn length(&self) -> i64 {
        match self {
            BufferView::Strided { length, .. }
            | BufferView::Simple { length, .. }
            | BufferView::Raw { length, .. } => *length,
        }
    }
    #[inline]
    pub fn readonly(&self) -> bool {
        match self {
            BufferView::Strided { readonly, .. } => *readonly,
            BufferView::Simple { backing, .. } | BufferView::Raw { backing, .. } => {
                backing.readonly()
            }
        }
    }

    /// The LIVE logical bytes of the view (`buffer.py as_str`), read from the
    /// backing object's own storage — no detached copy — so the view observes
    /// later mutation of a bytearray / array source.  Honours offset / shape /
    /// strides so a strided slice (`m[::2]`, `m[::-1]`) or an N-D view gathers
    /// the right elements in C order.
    ///
    /// # Safety
    /// The backing [`Buffer`]'s `w_obj` must point to a live object of its
    /// tagged kind.
    pub unsafe fn gather(&self) -> Vec<u8> {
        unsafe {
            let itemsize = self.itemsize();
            let ndim = self.ndim();
            let offset = self.offset();
            let full = self.backing().as_bytes();
            let isz = itemsize.max(0) as usize;
            if ndim == 0 {
                let mut out = Vec::with_capacity(isz);
                copy_base(full, offset, isz, &mut out);
                return out;
            }
            let shape = self.native_shape();
            let strides = self.native_strides();
            let count = if itemsize > 0 {
                self.length() / itemsize
            } else {
                0
            };
            let mut out = Vec::with_capacity(count.max(0) as usize * isz);
            copy_rec(full, &shape, &strides, ndim, 0, offset, isz, &mut out);
            out
        }
    }
}
