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
    /// Field values.
    pub fields: Vec<(u32, OpRef)>,
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
/// Tracks writes to byte offsets within the buffer.
#[derive(Clone, Debug)]
pub struct VirtualRawBufferInfo {
    /// Size of the buffer in bytes.
    pub size: usize,
    /// Values stored at byte offsets: (offset, value_opref).
    pub entries: Vec<(usize, OpRef)>,
}
