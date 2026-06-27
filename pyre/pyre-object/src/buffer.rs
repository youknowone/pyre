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
}

impl Buffer {
    /// The exporter object whose storage this buffer reads/writes.
    #[inline]
    pub fn w_obj(&self) -> PyObjectRef {
        match self {
            Buffer::String { w_obj } | Buffer::Byte { w_obj } | Buffer::Array { w_obj } => *w_obj,
        }
    }

    /// The full byte storage of the exporter (`getlength` is its `.len()`).
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
            }
        }
    }
}
