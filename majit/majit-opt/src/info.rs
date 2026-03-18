use crate::intutils::IntBound;
/// Abstract information attached to operations during optimization.
///
/// Translated from rpython/jit/metainterp/optimizeopt/info.py.
/// Each operation can have associated analysis info (e.g., known integer bounds,
/// pointer info, virtual object state).
use majit_ir::{DescrRef, GcRef, OpRef, Value};

/// Information about an operation's result, attached during optimization.
///
/// info.py: AbstractInfo hierarchy — the base class for all optimization info.
#[derive(Clone, Debug)]
pub enum OpInfo {
    /// No information known.
    Unknown,
    /// Known constant value (integer or pointer).
    Constant(Value),
    /// Known integer bounds.
    IntBound(IntBound),
    /// Pointer info (non-null, known class, virtual, etc.).
    Ptr(PtrInfo),
    /// Known constant float value.
    /// info.py: FloatConstInfo — tracks float constants separately
    /// because they need special boxing on 32-bit platforms.
    FloatConst(f64),
}

impl OpInfo {
    pub fn is_constant(&self) -> bool {
        matches!(
            self,
            OpInfo::Constant(_) | OpInfo::FloatConst(_) | OpInfo::Ptr(PtrInfo::Constant(_))
        )
    }

    pub fn get_constant(&self) -> Option<&Value> {
        match self {
            OpInfo::Constant(v) => Some(v),
            _ => None,
        }
    }

    /// Get the constant float value if this is a FloatConst.
    pub fn get_constant_float(&self) -> Option<f64> {
        match self {
            OpInfo::FloatConst(f) => Some(*f),
            OpInfo::Constant(Value::Float(f)) => Some(*f),
            _ => None,
        }
    }

    pub fn get_int_bound(&self) -> Option<&IntBound> {
        match self {
            OpInfo::IntBound(b) => Some(b),
            _ => None,
        }
    }

    /// Whether this info is known non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            OpInfo::Ptr(ptr) => ptr.is_nonnull(),
            OpInfo::Constant(Value::Int(v)) => *v != 0,
            _ => false,
        }
    }

    /// Whether this info represents a virtual (allocation-removed) object.
    /// info.py: is_virtual()
    pub fn is_virtual(&self) -> bool {
        matches!(self, OpInfo::Ptr(ptr) if ptr.is_virtual())
    }

    /// Get the PtrInfo if present.
    pub fn get_ptr_info(&self) -> Option<&PtrInfo> {
        match self {
            OpInfo::Ptr(p) => Some(p),
            _ => None,
        }
    }
}

/// Information about a pointer value.
///
/// info.py: PtrInfo hierarchy:
///   NonNullPtrInfo → AbstractVirtualPtrInfo → {InstancePtrInfo, StructPtrInfo,
///   ArrayPtrInfo, ArrayStructInfo, RawBufferPtrInfo, RawStructPtrInfo, RawSlicePtrInfo}
///   ConstPtrInfo
#[derive(Clone, Debug)]
pub enum PtrInfo {
    /// Known to be non-null, nothing else.
    /// info.py: NonNullPtrInfo
    NonNull,
    /// Known constant pointer.
    /// info.py: ConstPtrInfo
    Constant(GcRef),
    /// Known class (type) of the object.
    /// info.py: NonNullPtrInfo with _known_class set
    KnownClass { class_ptr: GcRef, is_nonnull: bool },
    /// Virtual object (allocation removed by the optimizer).
    /// info.py: InstancePtrInfo
    Virtual(VirtualInfo),
    /// Virtual array.
    /// info.py: ArrayPtrInfo
    VirtualArray(VirtualArrayInfo),
    /// Virtual struct (no vtable).
    /// info.py: StructPtrInfo
    VirtualStruct(VirtualStructInfo),
    /// Virtual array of structs (interior field access).
    /// info.py: ArrayStructInfo
    VirtualArrayStruct(VirtualArrayStructInfo),
    /// Virtual raw buffer.
    /// info.py: RawBufferPtrInfo
    VirtualRawBuffer(VirtualRawBufferInfo),
    /// Virtualizable object (interpreter frame).
    Virtualizable(VirtualizableFieldState),
}

impl PtrInfo {
    // ── Constructors (info.py: factory methods) ──

    /// Create a NonNull PtrInfo.
    pub fn nonnull() -> Self {
        PtrInfo::NonNull
    }

    /// Create a Constant PtrInfo.
    pub fn constant(gcref: GcRef) -> Self {
        PtrInfo::Constant(gcref)
    }

    /// Create a KnownClass PtrInfo.
    pub fn known_class(class_ptr: GcRef, is_nonnull: bool) -> Self {
        PtrInfo::KnownClass {
            class_ptr,
            is_nonnull,
        }
    }

    /// Create a Virtual PtrInfo (allocation removed).
    pub fn virtual_obj(descr: DescrRef, known_class: Option<GcRef>) -> Self {
        PtrInfo::Virtual(VirtualInfo {
            descr,
            known_class,
            fields: Vec::new(),
            field_descrs: Vec::new(),
        })
    }

    /// Create a VirtualArray PtrInfo.
    pub fn virtual_array(descr: DescrRef, length: usize) -> Self {
        PtrInfo::VirtualArray(VirtualArrayInfo {
            descr,
            items: vec![OpRef::NONE; length],
        })
    }

    /// Create a VirtualStruct PtrInfo.
    pub fn virtual_struct(descr: DescrRef) -> Self {
        PtrInfo::VirtualStruct(VirtualStructInfo {
            descr,
            fields: Vec::new(),
            field_descrs: Vec::new(),
        })
    }

    // ── Query methods ──

    /// Whether this pointer is known to be non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            PtrInfo::NonNull => true,
            PtrInfo::Constant(gcref) => !gcref.is_null(),
            PtrInfo::KnownClass { is_nonnull, .. } => *is_nonnull,
            PtrInfo::Virtual(_)
            | PtrInfo::VirtualArray(_)
            | PtrInfo::VirtualStruct(_)
            | PtrInfo::VirtualArrayStruct(_)
            | PtrInfo::VirtualRawBuffer(_)
            | PtrInfo::Virtualizable(_) => true,
        }
    }

    /// Whether this pointer is a virtual (allocation removed).
    /// info.py: is_virtual()
    pub fn is_virtual(&self) -> bool {
        matches!(
            self,
            PtrInfo::Virtual(_)
                | PtrInfo::VirtualArray(_)
                | PtrInfo::VirtualStruct(_)
                | PtrInfo::VirtualArrayStruct(_)
                | PtrInfo::VirtualRawBuffer(_)
                | PtrInfo::Virtualizable(_)
        )
    }

    /// Whether this is a constant pointer.
    /// info.py: isinstance(info, ConstPtrInfo)
    pub fn is_constant(&self) -> bool {
        matches!(self, PtrInfo::Constant(_))
    }

    /// Get the known class, if any.
    /// info.py: get_known_class_or_none()
    pub fn get_known_class(&self) -> Option<&GcRef> {
        match self {
            PtrInfo::KnownClass { class_ptr, .. } => Some(class_ptr),
            PtrInfo::Virtual(v) => v.known_class.as_ref(),
            _ => None,
        }
    }

    /// Get constant GcRef value if this is a constant pointer.
    pub fn get_constant_ref(&self) -> Option<&GcRef> {
        match self {
            PtrInfo::Constant(r) => Some(r),
            _ => None,
        }
    }

    /// Get the string length from a constant string pointer.
    /// info.py: getstrlen() on ConstPtrInfo
    pub fn getstrlen(&self) -> Option<usize> {
        // Only meaningful for constant string objects.
        // In RPython, this reads the string header to get the length.
        // In our implementation, GcRef doesn't carry string metadata,
        // so we return None. This can be overridden when GcRef is enriched.
        None
    }

    /// Get the string hash from a constant string pointer.
    /// info.py: getstrhash() on ConstPtrInfo
    pub fn getstrhash(&self) -> Option<i64> {
        None
    }

    /// Count the number of fields/items in this virtual object.
    /// info.py: _get_num_items() / num_fields
    pub fn num_fields(&self) -> usize {
        match self {
            PtrInfo::Virtual(v) => v.fields.len(),
            PtrInfo::VirtualArray(v) => v.items.len(),
            PtrInfo::VirtualStruct(v) => v.fields.len(),
            PtrInfo::VirtualArrayStruct(v) => v.element_fields.len(),
            PtrInfo::VirtualRawBuffer(v) => v.entries.len(),
            _ => 0,
        }
    }

    /// Enumerate all OpRef values stored in this virtual's fields/items.
    /// info.py: visitor_walk_recursive — walks all fields of a virtual.
    pub fn visitor_walk_recursive(&self) -> Vec<OpRef> {
        match self {
            PtrInfo::Virtual(v) => v.fields.iter().map(|(_, r)| *r).collect(),
            PtrInfo::VirtualArray(v) => v.items.clone(),
            PtrInfo::VirtualStruct(v) => v.fields.iter().map(|(_, r)| *r).collect(),
            PtrInfo::VirtualArrayStruct(v) => v
                .element_fields
                .iter()
                .flat_map(|fields| fields.iter().map(|(_, r)| *r))
                .collect(),
            PtrInfo::VirtualRawBuffer(v) => v.entries.iter().map(|(_, _, r)| *r).collect(),
            PtrInfo::Virtualizable(v) => {
                let mut refs: Vec<OpRef> = v.fields.iter().map(|(_, r)| *r).collect();
                for (_, items) in &v.arrays {
                    refs.extend(items.iter().copied());
                }
                refs
            }
            _ => Vec::new(),
        }
    }

    /// info.py: force_at_the_end_of_preamble(op, optforce, rec)
    /// Force a virtual object at the end of the preamble iteration.
    /// This is called when loop peeling discovers that a virtual must
    /// be materialized before the loop body begins.
    ///
    /// Returns the materialized OpRef, or None if not virtual.
    pub fn force_at_the_end_of_preamble(&self) -> bool {
        self.is_virtual()
    }

    /// info.py: make_guards(op, short_boxes, optimizer)
    /// Generate guard operations to verify this pointer info.
    /// Returns a list of opcodes and expected values for guards.
    pub fn make_guards(&self) -> Vec<majit_ir::OpCode> {
        match self {
            PtrInfo::NonNull => vec![majit_ir::OpCode::GuardNonnull],
            PtrInfo::KnownClass { .. } => vec![majit_ir::OpCode::GuardNonnullClass],
            PtrInfo::Constant(_) => vec![majit_ir::OpCode::GuardValue],
            _ => Vec::new(),
        }
    }

    /// info.py: get_descr() — get the size/type descriptor for virtual objects.
    pub fn get_descr(&self) -> Option<&DescrRef> {
        match self {
            PtrInfo::Virtual(v) => Some(&v.descr),
            PtrInfo::VirtualArray(v) => Some(&v.descr),
            PtrInfo::VirtualStruct(v) => Some(&v.descr),
            PtrInfo::VirtualArrayStruct(v) => Some(&v.descr),
            _ => None,
        }
    }

    /// info.py: setfield(field_descr, value) — set a field on a virtual object.
    pub fn set_field(&mut self, field_idx: u32, value: OpRef) {
        match self {
            PtrInfo::Virtual(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = value;
                        return;
                    }
                }
                v.fields.push((field_idx, value));
            }
            PtrInfo::VirtualStruct(v) => {
                for entry in &mut v.fields {
                    if entry.0 == field_idx {
                        entry.1 = value;
                        return;
                    }
                }
                v.fields.push((field_idx, value));
            }
            _ => {}
        }
    }

    /// info.py: getfield(field_descr) — get a field from a virtual object.
    pub fn get_field(&self, field_idx: u32) -> Option<OpRef> {
        match self {
            PtrInfo::Virtual(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| *v),
            PtrInfo::VirtualStruct(v) => v
                .fields
                .iter()
                .find(|(k, _)| *k == field_idx)
                .map(|(_, v)| *v),
            _ => None,
        }
    }

    /// info.py: setitem(index, value) — set an item in a virtual array.
    pub fn set_item(&mut self, index: usize, value: OpRef) {
        if let PtrInfo::VirtualArray(v) = self {
            if index < v.items.len() {
                v.items[index] = value;
            }
        }
    }

    /// info.py: getitem(index) — get an item from a virtual array.
    pub fn get_item(&self, index: usize) -> Option<OpRef> {
        if let PtrInfo::VirtualArray(v) = self {
            v.items.get(index).copied()
        } else {
            None
        }
    }

    /// Copy fields from this virtual info to another.
    /// info.py: copy_fields_to_const()
    pub fn copy_fields_to(&self, other: &mut PtrInfo) {
        match (self, other) {
            (PtrInfo::Virtual(src), PtrInfo::Virtual(dst)) => {
                dst.fields = src.fields.clone();
                dst.field_descrs = src.field_descrs.clone();
            }
            (PtrInfo::VirtualStruct(src), PtrInfo::VirtualStruct(dst)) => {
                dst.fields = src.fields.clone();
                dst.field_descrs = src.field_descrs.clone();
            }
            (PtrInfo::VirtualArray(src), PtrInfo::VirtualArray(dst)) => {
                dst.items = src.items.clone();
            }
            _ => {}
        }
    }
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
    /// Original field descriptors, preserving offset/size/type info for forcing.
    pub field_descrs: Vec<(u32, DescrRef)>,
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
    /// Original field descriptors: (field_descr_index, original_descr).
    /// Used to emit correct SetfieldRaw ops when forcing.
    pub field_descrs: Vec<(u32, DescrRef)>,
    /// Tracked array field values: (array_field_index, element_values).
    /// Indices correspond to VirtualizableInfo::array_fields order.
    pub arrays: Vec<(u32, Vec<OpRef>)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Descr, OpCode, Value};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr;
    impl Descr for TestDescr {}

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

    #[test]
    fn test_ptr_info_factories() {
        let nonnull = PtrInfo::nonnull();
        assert!(nonnull.is_nonnull());
        assert!(!nonnull.is_virtual());

        let constant = PtrInfo::constant(GcRef(0x1000));
        assert!(constant.is_nonnull());
        assert!(constant.is_constant());

        let kc = PtrInfo::known_class(GcRef(0x2000), true);
        assert!(kc.is_nonnull());
        assert!(kc.get_known_class().is_some());
    }

    #[test]
    fn test_ptr_info_virtual_factories() {
        let descr: DescrRef = Arc::new(TestDescr);

        let virtual_obj = PtrInfo::virtual_obj(descr.clone(), Some(GcRef(0x3000)));
        assert!(virtual_obj.is_virtual());
        assert!(virtual_obj.is_nonnull());
        assert!(virtual_obj.get_descr().is_some());

        let virtual_arr = PtrInfo::virtual_array(descr.clone(), 5);
        assert!(virtual_arr.is_virtual());
        assert_eq!(virtual_arr.num_fields(), 5);

        let virtual_struct = PtrInfo::virtual_struct(descr);
        assert!(virtual_struct.is_virtual());
    }

    #[test]
    fn test_ptr_info_set_get_field() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);

        assert_eq!(info.get_field(0), None);
        info.set_field(0, OpRef(10));
        assert_eq!(info.get_field(0), Some(OpRef(10)));
        info.set_field(0, OpRef(20)); // overwrite
        assert_eq!(info.get_field(0), Some(OpRef(20)));
        info.set_field(1, OpRef(30));
        assert_eq!(info.get_field(1), Some(OpRef(30)));
    }

    #[test]
    fn test_ptr_info_set_get_item() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_array(descr, 3);

        assert_eq!(info.get_item(0), Some(OpRef::NONE)); // initialized to NONE
        info.set_item(0, OpRef(10));
        assert_eq!(info.get_item(0), Some(OpRef(10)));
        info.set_item(2, OpRef(30));
        assert_eq!(info.get_item(2), Some(OpRef(30)));
        assert_eq!(info.get_item(5), None); // out of bounds
    }

    #[test]
    fn test_ptr_info_make_guards() {
        let nonnull = PtrInfo::nonnull();
        let guards = nonnull.make_guards();
        assert!(guards.contains(&OpCode::GuardNonnull));

        let constant = PtrInfo::constant(GcRef(0x1000));
        let guards = constant.make_guards();
        assert!(guards.contains(&OpCode::GuardValue));

        let kc = PtrInfo::known_class(GcRef(0x2000), true);
        let guards = kc.make_guards();
        assert!(guards.contains(&OpCode::GuardNonnullClass));
    }

    #[test]
    fn test_ptr_info_visitor_walk() {
        let descr: DescrRef = Arc::new(TestDescr);
        let mut info = PtrInfo::virtual_obj(descr, None);
        info.set_field(0, OpRef(10));
        info.set_field(1, OpRef(20));
        let refs = info.visitor_walk_recursive();
        assert_eq!(refs, vec![OpRef(10), OpRef(20)]);
    }

    #[test]
    fn test_opinfo_is_nonnull() {
        assert!(!OpInfo::Unknown.is_nonnull());
        assert!(OpInfo::Constant(Value::Int(42)).is_nonnull());
        assert!(!OpInfo::Constant(Value::Int(0)).is_nonnull());
        assert!(OpInfo::Ptr(PtrInfo::NonNull).is_nonnull());
    }

    #[test]
    fn test_opinfo_float_const() {
        let info = OpInfo::FloatConst(3.14);
        assert!(info.is_constant());
        assert_eq!(info.get_constant_float(), Some(3.14));
    }
}
