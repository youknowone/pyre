//! Byte-storage layer for the buffer protocol — the pyre analogue of
//! `rpython/rlib/buffer.py`'s `Buffer` hierarchy.
//!
//! Each variant tags the concrete exporter, so a byte read dispatches to that
//! exporter's own storage accessor and a `bytes`/`bytearray`/`array` *subclass*
//! is read through its own fields.  The concrete kind is decided once, at
//! construction time, by the objspace-level `buffer_w` — which lives in the
//! interpreter crate because picking the variant needs `isinstance_w`, a
//! dependency pyre-object must not take.  The `memoryview` `BufferView` /
//! `W_MemoryView` layers sit on top of this.

use crate::pyobject::PyObjectRef;

/// Flat byte storage behind a buffer-protocol exporter.
pub enum Buffer {
    /// `bytes` — read-only (`StringBuffer`).
    String { w_obj: PyObjectRef },
    /// `bytearray` — mutable (`ByteBuffer`).
    Byte { w_obj: PyObjectRef },
    /// `array.array` — its raw element bytes.
    Array { w_obj: PyObjectRef },
    /// A `[offset, offset+size)` window over another `Buffer` (`SubBuffer`,
    /// `rpython/rlib/buffer.py:389`).  Sub-buffers never nest — see [`sub`].
    ///
    /// [`sub`]: Buffer::sub
    Sub {
        parent: Box<Buffer>,
        offset: usize,
        size: usize,
    },
}

impl Buffer {
    /// `SubBuffer(parent, offset, size)` (`rpython/rlib/buffer.py:389`).  A
    /// `Sub` over a `Sub` is collapsed to a single window over the inner
    /// buffer (`buffer.py:397` — "don't nest them"): the offsets sum and the
    /// size clamps to the inner window, so the wrapper depth never exceeds 1.
    pub fn sub(parent: Buffer, offset: usize, size: usize) -> Buffer {
        match parent {
            Buffer::Sub {
                parent: inner,
                offset: inner_off,
                size: inner_size,
            } => {
                let at_most = inner_size.saturating_sub(offset);
                Buffer::Sub {
                    parent: inner,
                    offset: inner_off + offset,
                    size: size.min(at_most),
                }
            }
            other => Buffer::Sub {
                parent: Box::new(other),
                offset,
                size,
            },
        }
    }

    /// The root exporter object whose storage this buffer reads/writes; a
    /// `Sub` reports its parent's exporter (`SubBuffer` has no `.obj` of its
    /// own).
    #[inline]
    pub fn w_obj(&self) -> PyObjectRef {
        match self {
            Buffer::String { w_obj } | Buffer::Byte { w_obj } | Buffer::Array { w_obj } => *w_obj,
            Buffer::Sub { parent, .. } => parent.w_obj(),
        }
    }

    /// The byte storage this buffer exposes (`getlength` is its `.len()`).  A
    /// `Sub` is the `[offset, offset+size)` window of its parent, clamped to
    /// the parent's live length (`SubBuffer.getlength`, `buffer.py:413`).
    ///
    /// # Safety
    /// The variant's `w_obj` must point to a live object of the tagged kind.
    #[inline]
    pub unsafe fn as_bytes(&self) -> &'static [u8] {
        unsafe {
            match self {
                Buffer::String { w_obj } => crate::bytesobject::w_bytes_data(*w_obj),
                Buffer::Byte { w_obj } => crate::bytearrayobject::w_bytearray_data(*w_obj),
                Buffer::Array { w_obj } => crate::interp_array::w_array_bytes(*w_obj),
                Buffer::Sub {
                    parent,
                    offset,
                    size,
                } => {
                    let full = parent.as_bytes();
                    let off = (*offset).min(full.len());
                    let end = off.saturating_add(*size).min(full.len());
                    &full[off..end]
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `w_obj` / `sub` never dereference the exporter, so a fake address is a
    // valid stand-in for the geometry-only tests below.
    fn fake(addr: usize) -> PyObjectRef {
        addr as PyObjectRef
    }

    #[test]
    fn sub_wraps_a_leaf_buffer() {
        match Buffer::sub(
            Buffer::String {
                w_obj: fake(0x1000),
            },
            2,
            5,
        ) {
            Buffer::Sub {
                parent,
                offset,
                size,
            } => {
                assert_eq!((offset, size), (2, 5));
                assert!(matches!(*parent, Buffer::String { .. }));
            }
            _ => panic!("expected Sub"),
        }
    }

    #[test]
    fn sub_over_sub_collapses_to_depth_one() {
        // `SubBuffer.__init__` (buffer.py:397): the offsets sum and the parent
        // is the inner buffer, so the wrapper never nests.
        let leaf = Buffer::Byte {
            w_obj: fake(0x2000),
        };
        let nested = Buffer::sub(Buffer::sub(leaf, 2, 5), 1, 3);
        match nested {
            Buffer::Sub {
                parent,
                offset,
                size,
            } => {
                assert_eq!((offset, size), (3, 3)); // 2+1, min(3, 5-1)
                assert!(matches!(*parent, Buffer::Byte { .. }));
                assert_eq!(parent.w_obj(), fake(0x2000));
            }
            _ => panic!("expected collapsed Sub"),
        }
    }

    #[test]
    fn sub_over_sub_clamps_size_to_inner_window() {
        let nested = Buffer::sub(
            Buffer::sub(
                Buffer::Array {
                    w_obj: fake(0x3000),
                },
                4,
                6,
            ),
            2,
            100,
        );
        match nested {
            Buffer::Sub { offset, size, .. } => assert_eq!((offset, size), (6, 4)), // 4+2, 6-2
            _ => panic!("expected Sub"),
        }
    }

    #[test]
    fn w_obj_recurses_through_sub_to_the_root_exporter() {
        let s = Buffer::sub(
            Buffer::Array {
                w_obj: fake(0x4000),
            },
            1,
            2,
        );
        assert_eq!(s.w_obj(), fake(0x4000));
    }
}
