//! View layer for the buffer protocol — the pyre analogue of
//! `pypy/interpreter/buffer.py`'s `BufferView` hierarchy.  A `BufferView`
//! carries the geometry (offset / shape / strides) over a byte-level
//! [`Buffer`] and gathers the live logical bytes in C order, honouring a
//! strided or N-D layout, without detaching a copy of the backing.
//!
//! `memoryview`'s `W_MemoryView` sits on top of this; the format-aware
//! element pack/unpack (`value_from_bytes` / `bytes_from_value`) is added in
//! a later slice.

use crate::buffer::Buffer;

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

/// A view of a [`Buffer`]'s bytes with offset / shape / stride geometry.
pub enum BufferView {
    /// Contiguous single-byte 1-D view (`SimpleView`): the logical bytes are
    /// a direct `offset..offset+length` window of the backing.
    Simple {
        data: Buffer,
        offset: i64,
        length: i64,
    },
    /// Format / shape / stride-aware view (`RawBufferView`): the logical
    /// bytes are gathered in C order across `shape` by `strides`, so a
    /// strided slice (`m[::2]`, `m[::-1]`) or an N-D view reads the right
    /// elements.
    Raw {
        data: Buffer,
        itemsize: i64,
        ndim: i64,
        offset: i64,
        length: i64,
        shape: Vec<i64>,
        strides: Vec<i64>,
    },
}

impl BufferView {
    /// The LIVE logical bytes of the view (`buffer.py as_str`), read from the
    /// backing object's own storage — no detached copy — so the view observes
    /// later mutation of a bytearray / array source.
    ///
    /// # Safety
    /// The backing [`Buffer`]'s `w_obj` must point to a live object of its
    /// tagged kind.
    pub unsafe fn gather(&self) -> Vec<u8> {
        unsafe {
            match self {
                BufferView::Simple {
                    data,
                    offset,
                    length,
                } => {
                    let full = data.as_bytes();
                    let mut out = Vec::with_capacity((*length).max(0) as usize);
                    copy_rec(full, &[*length], &[1], 1, 0, *offset, 1, &mut out);
                    out
                }
                BufferView::Raw {
                    data,
                    itemsize,
                    ndim,
                    offset,
                    length,
                    shape,
                    strides,
                } => {
                    let full = data.as_bytes();
                    let isz = (*itemsize).max(0) as usize;
                    if *ndim == 0 {
                        let mut out = Vec::with_capacity(isz);
                        copy_base(full, *offset, isz, &mut out);
                        return out;
                    }
                    let count = if *itemsize > 0 { *length / *itemsize } else { 0 };
                    let mut out = Vec::with_capacity(count.max(0) as usize * isz);
                    copy_rec(full, shape, strides, *ndim, 0, *offset, isz, &mut out);
                    out
                }
            }
        }
    }
}
