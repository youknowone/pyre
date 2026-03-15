use crate::intutils::IntBound;
/// Abstract information attached to operations during optimization.
///
/// Translated from rpython/jit/metainterp/optimizeopt/info.py.
/// Each operation can have associated analysis info (e.g., known integer bounds,
/// pointer info, virtual object state).
use majit_ir::{DescrRef, GcRef, OpRef, Value};

/// Information about an operation's result, attached during optimization.
#[derive(Clone, Debug)]
pub enum OpInfo {
    /// No information known.
    Unknown,
    /// Known constant value.
    Constant(Value),
    /// Known integer bounds.
    IntBound(IntBound),
    /// Pointer info (non-null, known class, virtual, etc.).
    Ptr(PtrInfo),
}

impl OpInfo {
    pub fn is_constant(&self) -> bool {
        matches!(self, OpInfo::Constant(_))
    }

    pub fn get_constant(&self) -> Option<&Value> {
        match self {
            OpInfo::Constant(v) => Some(v),
            _ => None,
        }
    }

    pub fn get_int_bound(&self) -> Option<&IntBound> {
        match self {
            OpInfo::IntBound(b) => Some(b),
            _ => None,
        }
    }
}

/// Information about a pointer value.
///
/// Mirrors rpython/jit/metainterp/optimizeopt/info.py PtrInfo hierarchy.
#[derive(Clone, Debug)]
pub enum PtrInfo {
    /// Known to be non-null, nothing else.
    NonNull,
    /// Known constant pointer.
    Constant(GcRef),
    /// Known class (type) of the object.
    KnownClass {
        /// The class pointer.
        class_ptr: GcRef,
        /// Whether this is also known non-null.
        is_nonnull: bool,
    },
    /// Virtual object (allocation removed by the optimizer).
    Virtual(VirtualInfo),
    /// Virtual array.
    VirtualArray(VirtualArrayInfo),
    /// Virtual struct (no vtable).
    VirtualStruct(VirtualStructInfo),
    /// Virtual array of structs (interior field access).
    VirtualArrayStruct(VirtualArrayStructInfo),
    /// Virtual raw buffer.
    VirtualRawBuffer(VirtualRawBufferInfo),
    /// Virtualizable object (interpreter frame).
    ///
    /// Unlike virtual objects (allocation eliminated), a virtualizable
    /// already exists on the heap. Its fields are tracked so that
    /// setfield/getfield ops can be eliminated — field values live in
    /// registers instead of memory.
    ///
    /// On "force" (escape), only the field writes are emitted
    /// (no allocation, since the object already exists).
    Virtualizable(VirtualizableFieldState),
}

/// A virtual object whose allocation has been removed.
///
/// Fields are tracked as OpRefs to the operations that produce their values.
#[derive(Clone, Debug)]
pub struct VirtualInfo {
    /// The size descriptor of this object.
    pub descr: DescrRef,
    /// Known class (if any).
    pub known_class: Option<GcRef>,
    /// Field values: (field_descr_index, value_opref).
    pub fields: Vec<(u32, OpRef)>,
}

/// A virtual array.
#[derive(Clone, Debug)]
pub struct VirtualArrayInfo {
    /// The array descriptor.
    pub descr: DescrRef,
    /// Element values.
    pub items: Vec<OpRef>,
}

/// A virtual struct (no vtable).
#[derive(Clone, Debug)]
pub struct VirtualStructInfo {
    /// The size descriptor.
    pub descr: DescrRef,
    /// Field values: (field_index, value, optional original field descriptor).
    pub fields: Vec<(u32, OpRef)>,
    /// Original field descriptors keyed by field_index, used for force.
    pub field_descrs: Vec<(u32, DescrRef)>,
}

/// A virtual array of structs (interior field access pattern).
///
/// Mirrors RPython's VArrayStructInfo where each array element
/// is a fixed-size struct with named fields. Used for RPython arrays
/// with complex item types (e.g., hash table entries with key+value fields).
#[derive(Clone, Debug)]
pub struct VirtualArrayStructInfo {
    /// The array descriptor.
    pub descr: DescrRef,
    /// Per-element fields: outer Vec = elements, inner Vec = (field_descr_index, value_opref).
    pub element_fields: Vec<Vec<(u32, OpRef)>>,
}

/// A virtual raw memory buffer.
///
/// Mirrors RPython's VRawBufferInfo for virtualized raw_malloc allocations.
/// Tracks writes to byte offsets within the buffer. Entries are kept sorted
/// by offset and must never overlap (matching RPython's RawBuffer invariant).
#[derive(Clone, Debug)]
pub struct VirtualRawBufferInfo {
    /// Size of the buffer in bytes.
    pub size: usize,
    /// Values stored at byte offsets: (offset, length, value_opref).
    ///
    /// Sorted by offset. Invariant: `entries[i].0 + entries[i].1 <= entries[i+1].0`
    /// (no overlapping writes).
    pub entries: Vec<(usize, usize, OpRef)>,
}

/// Error returned when a raw buffer operation violates invariants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawBufferError {
    /// A write overlaps with an existing write.
    OverlappingWrite {
        new_offset: usize,
        new_length: usize,
        existing_offset: usize,
        existing_length: usize,
    },
    /// A read from an offset that was never written.
    UninitializedRead { offset: usize, length: usize },
    /// A read whose length/offset doesn't match the write at that offset.
    IncompatibleRead {
        offset: usize,
        read_length: usize,
        write_length: usize,
    },
}

impl VirtualRawBufferInfo {
    /// Write a value at `(offset, length)`. Maintains sorted order by offset.
    ///
    /// If a write already exists at the same offset with the same length,
    /// updates the value. Returns `Err` on overlapping writes or
    /// incompatible length at the same offset.
    pub fn write_value(
        &mut self,
        offset: usize,
        length: usize,
        value: OpRef,
    ) -> Result<(), RawBufferError> {
        let mut insert_pos = 0;
        for (i, &(wo, wl, _)) in self.entries.iter().enumerate() {
            if wo == offset {
                if wl != length {
                    return Err(RawBufferError::OverlappingWrite {
                        new_offset: offset,
                        new_length: length,
                        existing_offset: wo,
                        existing_length: wl,
                    });
                }
                // Same offset and length: update in place.
                self.entries[i].2 = value;
                return Ok(());
            } else if wo > offset {
                break;
            }
            insert_pos = i + 1;
        }
        // Check overlap with next entry.
        if insert_pos < self.entries.len() {
            let (next_off, _, _) = self.entries[insert_pos];
            if offset + length > next_off {
                return Err(RawBufferError::OverlappingWrite {
                    new_offset: offset,
                    new_length: length,
                    existing_offset: next_off,
                    existing_length: self.entries[insert_pos].1,
                });
            }
        }
        // Check overlap with previous entry.
        if insert_pos > 0 {
            let (prev_off, prev_len, _) = self.entries[insert_pos - 1];
            if prev_off + prev_len > offset {
                return Err(RawBufferError::OverlappingWrite {
                    new_offset: offset,
                    new_length: length,
                    existing_offset: prev_off,
                    existing_length: prev_len,
                });
            }
        }
        self.entries.insert(insert_pos, (offset, length, value));
        Ok(())
    }

    /// Read the value at `(offset, length)`.
    ///
    /// Returns `Err(UninitializedRead)` if no write exists at that offset,
    /// or `Err(IncompatibleRead)` if the length doesn't match.
    pub fn read_value(&self, offset: usize, length: usize) -> Result<OpRef, RawBufferError> {
        for &(wo, wl, val) in &self.entries {
            if wo == offset {
                if wl != length {
                    return Err(RawBufferError::IncompatibleRead {
                        offset,
                        read_length: length,
                        write_length: wl,
                    });
                }
                return Ok(val);
            }
        }
        Err(RawBufferError::UninitializedRead { offset, length })
    }

    /// Check if a read at `(offset, size)` is fully covered by previous writes.
    ///
    /// Every byte in `[offset, offset+size)` must fall within at least one
    /// existing write region.
    pub fn is_read_fully_covered(&self, offset: usize, size: usize) -> bool {
        (0..size).all(|i| {
            let byte = offset + i;
            self.entries
                .iter()
                .any(|&(wo, wl, _)| byte >= wo && byte < wo + wl)
        })
    }

    /// Find the index of an existing write that is completely overwritten
    /// by a new write at `(offset, size)`.
    ///
    /// Returns the index of the first entry fully contained within
    /// `[offset, offset+size)`.
    pub fn find_overwritten_write(&self, offset: usize, size: usize) -> Option<usize> {
        self.entries
            .iter()
            .position(|&(wo, wl, _)| offset <= wo && offset + size >= wo + wl)
    }
}

/// Tracked field state for a virtualizable object (interpreter frame).
///
/// Mirrors RPython's virtualizable handling in the optimizer:
/// the frame already exists on the heap, but during JIT execution its
/// fields are kept in registers. The optimizer tracks the current value
/// of each field so that redundant setfield/getfield ops are eliminated.
///
/// When the virtualizable is "forced" (escapes to non-JIT code), field
/// values are written back to the heap via SETFIELD_RAW ops.
#[derive(Clone, Debug)]
pub struct VirtualizableFieldState {
    /// Tracked static field values: (field_descr_index, current_value_opref).
    /// Indices correspond to VirtualizableInfo::static_fields order.
    pub fields: Vec<(u32, OpRef)>,
    /// Tracked array field values: (array_field_index, element_values).
    /// Indices correspond to VirtualizableInfo::array_fields order.
    pub arrays: Vec<(u32, Vec<OpRef>)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(size: usize) -> VirtualRawBufferInfo {
        VirtualRawBufferInfo {
            size,
            entries: Vec::new(),
        }
    }

    #[test]
    fn rawbuffer_write_and_read() {
        let mut buf = make_buf(32);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        buf.write_value(8, 4, OpRef(20)).unwrap();
        buf.write_value(16, 8, OpRef(30)).unwrap();

        assert_eq!(buf.read_value(0, 8).unwrap(), OpRef(10));
        assert_eq!(buf.read_value(8, 4).unwrap(), OpRef(20));
        assert_eq!(buf.read_value(16, 8).unwrap(), OpRef(30));
    }

    #[test]
    fn rawbuffer_update_same_offset() {
        let mut buf = make_buf(16);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        buf.write_value(0, 8, OpRef(99)).unwrap();

        assert_eq!(buf.read_value(0, 8).unwrap(), OpRef(99));
        assert_eq!(buf.entries.len(), 1);
    }

    #[test]
    fn rawbuffer_overlap_next() {
        let mut buf = make_buf(32);
        buf.write_value(8, 8, OpRef(10)).unwrap();
        // Write at offset 4 with length 8 overlaps [8, 16)
        let err = buf.write_value(4, 8, OpRef(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_overlap_prev() {
        let mut buf = make_buf(32);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        // Write at offset 4 overlaps with [0, 8)
        let err = buf.write_value(4, 4, OpRef(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_incompatible_length_at_same_offset() {
        let mut buf = make_buf(16);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        let err = buf.write_value(0, 4, OpRef(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_uninitialized_read() {
        let buf = make_buf(16);
        let err = buf.read_value(0, 8).unwrap_err();
        assert_eq!(
            err,
            RawBufferError::UninitializedRead {
                offset: 0,
                length: 8
            }
        );
    }

    #[test]
    fn rawbuffer_incompatible_read_length() {
        let mut buf = make_buf(16);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        let err = buf.read_value(0, 4).unwrap_err();
        assert_eq!(
            err,
            RawBufferError::IncompatibleRead {
                offset: 0,
                read_length: 4,
                write_length: 8,
            }
        );
    }

    #[test]
    fn rawbuffer_read_fully_covered() {
        let mut buf = make_buf(32);
        buf.write_value(0, 8, OpRef(10)).unwrap();
        buf.write_value(8, 8, OpRef(20)).unwrap();

        // [0, 16) is fully covered by [0,8) + [8,16)
        assert!(buf.is_read_fully_covered(0, 16));
        // [0, 8) covered
        assert!(buf.is_read_fully_covered(0, 8));
        // [4, 8) falls within [0, 8)
        assert!(buf.is_read_fully_covered(4, 4));
    }

    #[test]
    fn rawbuffer_read_partially_covered_fails() {
        let mut buf = make_buf(32);
        buf.write_value(0, 4, OpRef(10)).unwrap();
        buf.write_value(8, 4, OpRef(20)).unwrap();

        // Bytes 4..8 are not covered by any write
        assert!(!buf.is_read_fully_covered(0, 8));
        // Byte 16 was never written
        assert!(!buf.is_read_fully_covered(16, 4));
    }

    #[test]
    fn rawbuffer_overwritten_write_detected() {
        let mut buf = make_buf(32);
        buf.write_value(4, 4, OpRef(10)).unwrap();
        buf.write_value(12, 4, OpRef(20)).unwrap();

        // A write [4, 12) fully contains [4, 8)
        assert_eq!(buf.find_overwritten_write(4, 8), Some(0));
        // A write [0, 16) fully contains [4, 8)
        assert_eq!(buf.find_overwritten_write(0, 16), Some(0));
        // A write [12, 20) fully contains [12, 16)
        assert_eq!(buf.find_overwritten_write(12, 8), Some(1));
        // A write [0, 4) does not contain any existing entry
        assert_eq!(buf.find_overwritten_write(0, 4), None);
    }

    #[test]
    fn rawbuffer_sorted_insertion() {
        let mut buf = make_buf(32);
        buf.write_value(16, 4, OpRef(30)).unwrap();
        buf.write_value(0, 4, OpRef(10)).unwrap();
        buf.write_value(8, 4, OpRef(20)).unwrap();

        // Entries should be sorted by offset
        assert_eq!(buf.entries[0].0, 0);
        assert_eq!(buf.entries[1].0, 8);
        assert_eq!(buf.entries[2].0, 16);
    }
}
