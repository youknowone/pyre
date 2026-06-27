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
pub struct BufferView {
    /// Byte storage actually read / written (the root exporter's buffer).
    pub backing: Buffer,
    /// The exporter reported by `memoryview.obj` — coincides with the backing
    /// for a plain view, but a chained cast / slice keeps the root storage in
    /// `backing` while `w_obj` still reports the original exporter.
    pub w_obj: PyObjectRef,
    /// Format string object (`memoryview.format`).
    pub w_format: PyObjectRef,
    /// Shape tuple (`memoryview.shape`).
    pub w_shape: PyObjectRef,
    /// Strides tuple (`memoryview.strides`).
    pub w_strides: PyObjectRef,
    pub itemsize: i64,
    pub ndim: i64,
    pub offset: i64,
    pub length: i64,
    pub readonly: bool,
}

impl BufferView {
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
            let full = self.backing.as_bytes();
            let isz = self.itemsize.max(0) as usize;
            if self.ndim == 0 {
                let mut out = Vec::with_capacity(isz);
                copy_base(full, self.offset, isz, &mut out);
                return out;
            }
            let shape = read_dims(self.w_shape);
            let strides = read_dims(self.w_strides);
            let count = if self.itemsize > 0 {
                self.length / self.itemsize
            } else {
                0
            };
            let mut out = Vec::with_capacity(count.max(0) as usize * isz);
            copy_rec(full, &shape, &strides, self.ndim, 0, self.offset, isz, &mut out);
            out
        }
    }
}
