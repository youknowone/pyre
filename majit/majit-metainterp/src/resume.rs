//! Resume data: encodes the mapping from guard fail_args to full interpreter state.
//!
//! When a guard fails, the JIT needs to reconstruct the interpreter's full state
//! (program counter, local variables, stack contents) from the values stored in
//! the DeadFrame. Resume data provides this mapping.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/resume.py`.

use std::collections::HashMap;

use majit_backend::{
    ExitFrameLayout, ExitPendingFieldLayout, ExitRecoveryLayout, ExitValueSourceLayout,
    ExitVirtualLayout,
};
use majit_ir::{GcRef, Type};

// ═══════════════════════════════════════════════════════════════
// RPython resume.py:96-139 — structural port (i16 tags).
// ═══════════════════════════════════════════════════════════════

// resume.py:96-97
#[derive(Debug)]
pub struct TagOverflow;

// resume.py:99-104
pub fn tag(value: i32, tagbits: u8) -> Result<i16, TagOverflow> {
    debug_assert!(tagbits <= 3);
    let sx = value >> 13;
    if sx != 0 && sx != -1 {
        return Err(TagOverflow);
    }
    Ok(((value << 2) | tagbits as i32) as i16)
}

// resume.py:106-109
pub fn untag(value: i16) -> (i32, u8) {
    let widened = value as i32;
    let tagbits = (widened & TAGMASK as i32) as u8;
    (widened >> 2, tagbits)
}

// resume.py:111-113
#[inline]
pub fn tagged_eq(x: i16, y: i16) -> bool {
    (x as i32) == (y as i32)
}

// resume.py:115-121
pub fn tagged_list_eq(tl1: &[i16], tl2: &[i16]) -> bool {
    if tl1.len() != tl2.len() {
        return false;
    }
    tl1.iter().zip(tl2.iter()).all(|(&a, &b)| tagged_eq(a, b))
}

// resume.py:123-132
pub const TAGCONST: u8 = 0;
pub const TAGINT: u8 = 1;
pub const TAGBOX: u8 = 2;
pub const TAGVIRTUAL: u8 = 3;
const TAGMASK: u8 = 3;

pub const UNASSIGNED: i16 = ((-1i32 << 13) << 2 | TAGBOX as i32) as i16;
pub const UNASSIGNEDVIRTUAL: i16 = ((-1i32 << 13) << 2 | TAGVIRTUAL as i32) as i16;
pub const NULLREF: i16 = ((-1i32 << 2) | TAGCONST as i32) as i16;
pub const UNINITIALIZED_TAG: i16 = ((-2i32 << 2) | TAGCONST as i32) as i16;
pub const TAG_CONST_OFFSET: i32 = 0;

/// Vec-backed livebox map: OpRef → i16 tag.
/// Replaces HashMap<u32, i16> with O(1) Vec indexing.
/// Sentinel i16::MIN means "not present".
pub struct LiveboxMap {
    /// Op results (OpRef < CONST_BASE).
    results: Vec<i16>,
    /// Constants (OpRef >= CONST_BASE), offset by CONST_BASE.
    constants: Vec<i16>,
}

const LIVEBOX_ABSENT: i16 = i16::MIN;

impl LiveboxMap {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            constants: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn get(&self, key: u32) -> Option<i16> {
        let (vec, idx) = if key >= majit_ir::OpRef::CONST_BASE {
            (
                &self.constants,
                (key - majit_ir::OpRef::CONST_BASE) as usize,
            )
        } else {
            (&self.results, key as usize)
        };
        if idx < vec.len() {
            let v = vec[idx];
            if v != LIVEBOX_ABSENT { Some(v) } else { None }
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: u32, value: i16) {
        let (vec, idx) = if key >= majit_ir::OpRef::CONST_BASE {
            (
                &mut self.constants,
                (key - majit_ir::OpRef::CONST_BASE) as usize,
            )
        } else {
            (&mut self.results, key as usize)
        };
        if idx >= vec.len() {
            vec.resize(idx + 1, LIVEBOX_ABSENT);
        }
        vec[idx] = value;
    }

    #[inline(always)]
    pub fn contains_key(&self, key: u32) -> bool {
        self.get(key).is_some()
    }

    /// Iterate over all (key, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, i16)> + '_ {
        self.results
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != LIVEBOX_ABSENT)
            .map(|(i, v)| (i as u32, *v))
            .chain(
                self.constants
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v != LIVEBOX_ABSENT)
                    .map(|(i, v)| (i as u32 + majit_ir::OpRef::CONST_BASE, *v)),
            )
    }
}

// resume.py:134-139
pub struct NumberingState {
    pub writer: crate::resumecode::Writer,
    pub liveboxes: LiveboxMap,
    pub num_boxes: i32,
    pub num_virtuals: i32,
    /// RPython Box.type parity: type of each TAGBOX livebox, captured at
    /// numbering time when env.get_type() is called.
    pub livebox_types: std::collections::HashMap<u32, majit_ir::Type>,
}

impl NumberingState {
    pub fn new(size: usize) -> Self {
        NumberingState {
            writer: crate::resumecode::Writer::new(size),
            liveboxes: LiveboxMap::new(),
            num_boxes: 0,
            num_virtuals: 0,
            livebox_types: std::collections::HashMap::new(),
        }
    }
    pub fn append_short(&mut self, item: i16) {
        self.writer.append_short(item as i32);
    }
    pub fn append_int(&mut self, item: i32) {
        self.writer.append_int(item);
    }
    pub fn patch_current_size(&mut self, index: usize) {
        self.writer.patch_current_size(index);
    }
    pub fn create_numbering(&self) -> Vec<u8> {
        self.writer.create_numbering()
    }
}

/// RPython snapshot: the state captured at a guard point.
/// Corresponds to RPython's SnapshotIterator output:
/// snapshot_iter.vable_array, snapshot_iter.vref_array, snapshot_iter.framestack.
///
/// NOTE: RPython does not have this struct. It uses `trace.get_snapshot_iter(position)`
/// which returns a lazy iterator over the trace's snapshot data (opencoder.py).
/// We use an eager struct because pyre's tracing records fail_args directly
/// on guard ops rather than using RPython's snapshot log format.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Virtualizable field boxes (resume.py:234-241).
    pub vable_array: Vec<majit_ir::OpRef>,
    /// Virtualref pairs (resume.py:243-247). Length must be even.
    pub vref_array: Vec<majit_ir::OpRef>,
    /// Frame chain (resume.py:249-253). Multiple frames for inlined calls.
    pub framestack: Vec<SnapshotFrame>,
}

/// One frame in a snapshot's frame chain.
#[derive(Debug, Clone)]
pub struct SnapshotFrame {
    /// Index into the jitcode table (resume.py:250 jitcode_index).
    pub jitcode_index: i32,
    /// Bytecode program counter (resume.py:250 pc).
    pub pc: i32,
    /// Live boxes for this frame's registers (resume.py:253).
    pub boxes: Vec<majit_ir::OpRef>,
}

impl Snapshot {
    /// Create a simple single-frame snapshot (pyre common case).
    pub fn single_frame(pc: i32, boxes: Vec<majit_ir::OpRef>) -> Self {
        Snapshot {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            framestack: vec![SnapshotFrame {
                jitcode_index: 0,
                pc,
                boxes,
            }],
        }
    }

    /// Create a multi-frame snapshot from (jitcode_index, pc, boxes) tuples.
    pub fn multi_frame(frames: Vec<(i32, i32, Vec<majit_ir::OpRef>)>) -> Self {
        Snapshot {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            framestack: frames
                .into_iter()
                .map(|(jitcode_index, pc, boxes)| SnapshotFrame {
                    jitcode_index,
                    pc,
                    boxes,
                })
                .collect(),
        }
    }

    /// Estimated encoded size for NumberingState capacity hint.
    pub fn estimated_size(&self) -> usize {
        let frame_size: usize = self.framestack.iter().map(|f| f.boxes.len() + 2).sum();
        self.vable_array.len() + self.vref_array.len() + frame_size + 4
    }
}

/// Re-export BoxEnv from majit-ir.
pub use majit_ir::BoxEnv;

/// Simple BoxEnv implementation backed by constant/type HashMaps.
/// Used in tests and for simple snapshot numbering.
pub struct SimpleBoxEnv {
    pub constants: HashMap<u32, (i64, majit_ir::Type)>,
    pub replacements: HashMap<u32, majit_ir::OpRef>,
    pub types: HashMap<u32, majit_ir::Type>,
    pub virtuals: std::collections::HashSet<u32>,
    pub virtual_fields: HashMap<u32, majit_ir::VirtualFieldsInfo>,
}

impl SimpleBoxEnv {
    pub fn new() -> Self {
        SimpleBoxEnv {
            constants: HashMap::new(),
            replacements: HashMap::new(),
            types: HashMap::new(),
            virtuals: std::collections::HashSet::new(),
            virtual_fields: HashMap::new(),
        }
    }
}

impl BoxEnv for SimpleBoxEnv {
    fn get_box_replacement(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
        self.replacements.get(&opref.0).copied().unwrap_or(opref)
    }
    fn is_const(&self, opref: majit_ir::OpRef) -> bool {
        self.constants.contains_key(&opref.0)
    }
    fn get_const(&self, opref: majit_ir::OpRef) -> (i64, majit_ir::Type) {
        self.constants
            .get(&opref.0)
            .copied()
            .unwrap_or((0, majit_ir::Type::Int))
    }
    fn get_type(&self, opref: majit_ir::OpRef) -> majit_ir::Type {
        self.types
            .get(&opref.0)
            .copied()
            .unwrap_or(majit_ir::Type::Int)
    }
    fn is_virtual_ref(&self, opref: majit_ir::OpRef) -> bool {
        self.virtuals.contains(&opref.0)
    }
    fn is_virtual_raw(&self, opref: majit_ir::OpRef) -> bool {
        self.virtuals.contains(&opref.0)
    }
    fn get_virtual_fields(&self, opref: majit_ir::OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        self.virtual_fields.get(&opref.0).cloned()
    }
}

// resume.py:123-132 — tag constants (i64 widened for rd_numb encoding).
// Same values as the i16 TAGCONST/TAGINT/TAGBOX/TAGVIRTUAL above.
const TAGMASK_I64: i64 = TAGMASK as i64;
const ENCODED_UNINITIALIZED: i64 = -2;
const ENCODED_UNAVAILABLE: i64 = -3;

// Two low bits are reserved for the tag.
const INLINE_TAGGED_MIN: i64 = -(1_i64 << 61);
const INLINE_TAGGED_MAX: i64 = (1_i64 << 61) - 1;

/// resume.py: ResumeGuardDescr storage fields.
///
/// `rd_numb` is a flat encoded numbering section (resume.py:466):
/// 1. items_resume_section (total rd_numb length)
/// 2. count (number of liveboxes, resume.py:921)
/// 3. number of frames
/// 4. per-frame `(pc, slot_count, slot_sources...)`
///
/// Fields match RPython's `ResumeGuardDescr`:
/// - `rd_numb`: encoded numbering (resume.py:466)
/// - `rd_consts`: shared constant pool (resume.py:467)
/// - `rd_virtuals`: live VirtualInfo objects (compile.py:858)
/// - `rd_pendingfields`: pending field writes (resume.py:468)
#[derive(Debug, Clone)]
pub struct EncodedResumeData {
    /// resume.py:466 storage.rd_numb — flat encoded numbering section.
    pub rd_numb: Vec<i64>,
    /// resume.py:467 storage.rd_consts — shared constant pool.
    pub rd_consts: Vec<i64>,
    /// resume.py:468 storage.rd_pendingfields — pending field writes.
    pub rd_pendingfields: Vec<EncodedPendingFieldWrite>,
    /// compile.py:858 storage.rd_virtuals — live VirtualInfo objects.
    pub rd_virtuals: Vec<VirtualInfo>,
    /// resume.py:411 liveboxes — compact TAGBOX(n) → original FailArg index.
    /// In RPython, liveboxes[n] is the Box object that was assigned TAGBOX(n).
    /// Here, liveboxes[n] is the original deadframe slot index.
    pub liveboxes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecodedResumeLayout {
    vable_array: Vec<ResumeValueSource>,
    frames: Vec<FrameInfo>,
    virtuals: Vec<VirtualInfo>,
    pending_fields: Vec<PendingFieldInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeValueKind {
    FailArg,
    Constant,
    Virtual,
    Uninitialized,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeValueLayoutSummary {
    pub kind: ResumeValueKind,
    pub fail_arg_index: usize,
    pub raw_fail_arg_position: Option<usize>,
    pub constant: Option<i64>,
    pub virtual_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeFrameLayoutSummary {
    pub trace_id: Option<u64>,
    pub header_pc: Option<u64>,
    pub source_guard: Option<(u64, u32)>,
    pub pc: u64,
    pub slot_sources: Vec<ResumeValueKind>,
    pub slot_layouts: Vec<ResumeValueLayoutSummary>,
    pub slot_types: Option<Vec<Type>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeVirtualKind {
    Object,
    Struct,
    Array,
    ArrayStruct,
    RawBuffer,
    /// resume.py: VStrPlainInfo — virtual plain string.
    StrPlain,
    /// resume.py: VStrConcatInfo — virtual concatenated string.
    StrConcat,
    /// resume.py: VStrSliceInfo — virtual string slice.
    StrSlice,
    /// resume.py: VUniPlainInfo — virtual plain unicode string.
    UniPlain,
    /// resume.py: VUniConcatInfo — virtual concatenated unicode.
    UniConcat,
    /// resume.py: VUniSliceInfo — virtual unicode slice.
    UniSlice,
}

#[derive(Debug, Clone)]
pub enum ResumeVirtualLayoutSummary {
    Object {
        /// resume.py:615 self.descr — live SizeDescr, preserved across summary round-trip.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        /// info.py:318 _known_class — vtable pointer.
        known_class: Option<i64>,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    Struct {
        /// resume.py:631 self.typedescr — live SizeDescr, preserved across summary round-trip.
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    Array {
        descr_index: u32,
        items: Vec<ResumeValueLayoutSummary>,
    },
    ArrayStruct {
        descr_index: u32,
        /// Per-field type within each element: 0=ref, 1=int, 2=float.
        field_types: Vec<u8>,
        element_fields: Vec<Vec<(u32, ResumeValueLayoutSummary)>>,
    },
    RawBuffer {
        size: usize,
        entries: Vec<(usize, usize, ResumeValueLayoutSummary)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingFieldLayoutSummary {
    pub descr_index: u32,
    pub item_index: Option<usize>,
    pub is_array_item: bool,
    pub target_kind: ResumeValueKind,
    pub value_kind: ResumeValueKind,
    pub target: ResumeValueLayoutSummary,
    pub value: ResumeValueLayoutSummary,
}

#[derive(Debug, Clone)]
pub struct ResumeLayoutSummary {
    pub num_frames: usize,
    pub frame_pcs: Vec<u64>,
    pub frame_slot_counts: Vec<usize>,
    pub frame_layouts: Vec<ResumeFrameLayoutSummary>,
    pub num_virtuals: usize,
    pub virtual_kinds: Vec<ResumeVirtualKind>,
    pub virtual_layouts: Vec<ResumeVirtualLayoutSummary>,

    pub pending_field_count: usize,
    pub pending_field_layouts: Vec<PendingFieldLayoutSummary>,
    pub const_pool_size: usize,
}

impl ResumeValueLayoutSummary {
    pub(crate) fn raw_fail_arg_position(&self) -> usize {
        if let Some(position) = self.raw_fail_arg_position {
            return position;
        }
        self.fail_arg_index
    }

    fn to_resume_source(&self) -> ResumeValueSource {
        match self.kind {
            ResumeValueKind::FailArg => ResumeValueSource::FailArg(self.raw_fail_arg_position()),
            ResumeValueKind::Constant => {
                ResumeValueSource::Constant(self.constant.expect("missing constant value"))
            }
            ResumeValueKind::Virtual => {
                ResumeValueSource::Virtual(self.virtual_index.expect("missing virtual index"))
            }
            ResumeValueKind::Uninitialized => ResumeValueSource::Uninitialized,
            ResumeValueKind::Unavailable => ResumeValueSource::Unavailable,
        }
    }

    fn to_exit_source(&self, virtual_offset: usize) -> ExitValueSourceLayout {
        match self.kind {
            ResumeValueKind::FailArg => {
                ExitValueSourceLayout::ExitValue(self.raw_fail_arg_position())
            }
            ResumeValueKind::Constant => {
                ExitValueSourceLayout::Constant(self.constant.expect("missing constant value"))
            }
            ResumeValueKind::Virtual => ExitValueSourceLayout::Virtual(
                self.virtual_index.expect("missing virtual index") + virtual_offset,
            ),
            ResumeValueKind::Uninitialized => ExitValueSourceLayout::Uninitialized,
            ResumeValueKind::Unavailable => ExitValueSourceLayout::Unavailable,
        }
    }
}

impl ResumeFrameLayoutSummary {
    fn to_frame_info(&self) -> FrameInfo {
        FrameInfo {
            pc: self.pc,
            slot_map: self
                .slot_layouts
                .iter()
                .map(|slot| slot.to_resume_source())
                .collect(),
        }
    }

    fn to_exit_frame_layout(&self, virtual_offset: usize) -> ExitFrameLayout {
        ExitFrameLayout {
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            source_guard: self.source_guard,
            pc: self.pc,
            slots: self
                .slot_layouts
                .iter()
                .map(|slot| slot.to_exit_source(virtual_offset))
                .collect(),
            slot_types: self.slot_types.clone(),
        }
    }

    /// Build a `ResumeFrameLayoutSummary` from a backend-origin `ExitFrameLayout`.
    ///
    /// Each `ExitValueSourceLayout` slot is converted to the corresponding
    /// `ResumeValueLayoutSummary`, preserving slot types when present.
    pub fn from_exit_frame_layout(exit_frame: &ExitFrameLayout) -> Self {
        let slot_layouts: Vec<ResumeValueLayoutSummary> = exit_frame
            .slots
            .iter()
            .map(ResumeValueLayoutSummary::from_exit_value_source)
            .collect();
        let slot_sources: Vec<ResumeValueKind> = slot_layouts.iter().map(|s| s.kind).collect();

        Self {
            trace_id: exit_frame.trace_id,
            header_pc: exit_frame.header_pc,
            source_guard: exit_frame.source_guard,
            pc: exit_frame.pc,
            slot_sources,
            slot_layouts,
            slot_types: exit_frame.slot_types.clone(),
        }
    }
}

impl ResumeValueLayoutSummary {
    /// Build a `ResumeValueLayoutSummary` from a backend-origin `ExitValueSourceLayout`.
    pub fn from_exit_value_source(source: &ExitValueSourceLayout) -> Self {
        match source {
            ExitValueSourceLayout::ExitValue(index) => Self {
                kind: ResumeValueKind::FailArg,
                fail_arg_index: *index,
                raw_fail_arg_position: Some(*index),
                constant: None,
                virtual_index: None,
            },
            ExitValueSourceLayout::Constant(value) => Self {
                kind: ResumeValueKind::Constant,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: Some(*value),
                virtual_index: None,
            },
            ExitValueSourceLayout::Virtual(index) => Self {
                kind: ResumeValueKind::Virtual,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: Some(*index),
            },
            ExitValueSourceLayout::Uninitialized => Self {
                kind: ResumeValueKind::Uninitialized,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: None,
            },
            ExitValueSourceLayout::Unavailable => Self {
                kind: ResumeValueKind::Unavailable,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: None,
            },
        }
    }
}

impl ResumeVirtualLayoutSummary {
    fn to_virtual_info(&self) -> VirtualInfo {
        match self {
            ResumeVirtualLayoutSummary::Object {
                descr,
                type_id,
                descr_index,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => VirtualInfo::VirtualObj {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_resume_source()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Struct {
                typedescr,
                type_id,
                descr_index,
                fields,
                fielddescrs,
                descr_size,
            } => VirtualInfo::VStruct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_resume_source()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Array { descr_index, items } => VirtualInfo::VArray {
                descr_index: *descr_index,
                items: items
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                descr_index,
                element_fields,
                ..
            } => VirtualInfo::VArrayStruct {
                descr_index: *descr_index,
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| (*field_descr, source.to_resume_source()))
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer { size, entries } => VirtualInfo::VRawBuffer {
                size: *size,
                entries: entries
                    .iter()
                    .map(|(offset, size_in_bytes, source)| {
                        (*offset, *size_in_bytes, source.to_resume_source())
                    })
                    .collect(),
            },
        }
    }

    fn to_exit_virtual_layout(&self, virtual_offset: usize) -> ExitVirtualLayout {
        match self {
            ResumeVirtualLayoutSummary::Object {
                descr,
                type_id,
                descr_index,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => ExitVirtualLayout::Object {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_exit_source(virtual_offset)))
                    .collect(),
                target_slot: None,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Struct {
                typedescr,
                type_id,
                descr_index,
                fields,
                fielddescrs,
                descr_size,
            } => ExitVirtualLayout::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (*field_descr, source.to_exit_source(virtual_offset))
                    })
                    .collect(),
                target_slot: None,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Array { descr_index, items } => ExitVirtualLayout::Array {
                descr_index: *descr_index,
                clear: false,
                kind: 0, // default ref
                items: items
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                descr_index,
                field_types,
                element_fields,
            } => ExitVirtualLayout::ArrayStruct {
                descr_index: *descr_index,
                field_types: field_types.clone(),
                item_size: 0,
                field_offsets: vec![],
                field_sizes: vec![],
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| {
                                (*field_descr, source.to_exit_source(virtual_offset))
                            })
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer { size, entries } => {
                ExitVirtualLayout::RawBuffer {
                    size: *size,
                    entries: entries
                        .iter()
                        .map(|(offset, size_in_bytes, source)| {
                            (
                                *offset,
                                *size_in_bytes,
                                source.to_exit_source(virtual_offset),
                            )
                        })
                        .collect(),
                }
            }
        }
    }
}

impl PendingFieldLayoutSummary {
    fn to_pending_field_info(&self) -> PendingFieldInfo {
        PendingFieldInfo {
            descr_index: self.descr_index,
            target: self.target.to_resume_source(),
            value: self.value.to_resume_source(),
            item_index: self.item_index,
        }
    }

    fn to_exit_pending_field_layout(&self, virtual_offset: usize) -> ExitPendingFieldLayout {
        ExitPendingFieldLayout {
            descr_index: self.descr_index,
            item_index: self.item_index,
            is_array_item: self.is_array_item,
            target: self.target.to_exit_source(virtual_offset),
            value: self.value.to_exit_source(virtual_offset),
            field_offset: 0,
            field_size: 0,
            field_type: majit_ir::Type::Int,
        }
    }
}

impl ResumeLayoutSummary {
    pub fn to_resume_data(&self) -> ResumeData {
        ResumeData {
            vable_array: Vec::new(),
            frames: self
                .frame_layouts
                .iter()
                .map(|frame| frame.to_frame_info())
                .collect(),
            virtuals: self
                .virtual_layouts
                .iter()
                .map(|virt| virt.to_virtual_info())
                .collect(),
            pending_fields: self
                .pending_field_layouts
                .iter()
                .map(|pending| pending.to_pending_field_info())
                .collect(),
        }
    }

    pub fn to_exit_recovery_layout(&self) -> ExitRecoveryLayout {
        self.to_exit_recovery_layout_with_caller_prefix(None)
    }

    pub fn to_exit_recovery_layout_with_caller_prefix(
        &self,
        caller_prefix: Option<&ExitRecoveryLayout>,
    ) -> ExitRecoveryLayout {
        if self.frame_layouts.is_empty() {
            return caller_prefix.cloned().unwrap_or(ExitRecoveryLayout {
                frames: Vec::new(),
                virtual_layouts: Vec::new(),
                pending_field_layouts: Vec::new(),
            });
        }

        let prefix_frame_count = caller_prefix
            .map(|layout| layout.frames.len().saturating_sub(self.frame_layouts.len()))
            .unwrap_or(0);
        let preserve_prefix = prefix_frame_count > 0;

        let mut frames = caller_prefix
            .map(|layout| layout.frames[..prefix_frame_count].to_vec())
            .unwrap_or_default();
        // RPython parity: rd_virtuals is stored once on the guard descriptor
        // and never replaced (compile.py:866, resume.py:492). Always preserve
        // caller_prefix's virtual_layouts — they originate from
        // build_guard_metadata and must not be overwritten.
        let mut virtual_layouts = caller_prefix
            .map(|layout| layout.virtual_layouts.clone())
            .unwrap_or_default();
        let mut pending_field_layouts = if preserve_prefix {
            caller_prefix
                .map(|layout| layout.pending_field_layouts.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let virtual_offset = virtual_layouts.len();

        frames.extend(
            self.frame_layouts
                .iter()
                .map(|frame| frame.to_exit_frame_layout(virtual_offset)),
        );
        virtual_layouts.extend(
            self.virtual_layouts
                .iter()
                .map(|virt| virt.to_exit_virtual_layout(virtual_offset)),
        );
        pending_field_layouts.extend(
            self.pending_field_layouts
                .iter()
                .map(|pending| pending.to_exit_pending_field_layout(virtual_offset)),
        );

        ExitRecoveryLayout {
            frames,
            virtual_layouts,
            pending_field_layouts,
        }
    }

    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let resume_data = self.to_resume_data();
        let virtuals = resume_data.materialize_virtuals(fail_values);
        let pending_fields =
            ResumeData::resolve_pending_field_writes(&resume_data.pending_fields, fail_values);
        let frames = self
            .frame_layouts
            .iter()
            .map(|frame| ReconstructedFrame {
                trace_id: frame.trace_id,
                header_pc: frame.header_pc,
                source_guard: frame.source_guard,
                pc: frame.pc,
                jitcode_index: 0,
                slot_types: frame.slot_types.clone(),
                values: frame
                    .slot_layouts
                    .iter()
                    .map(|slot| {
                        ResumeData::resolve_frame_slot_source(&slot.to_resume_source(), fail_values)
                    })
                    .collect(),
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals,
            pending_fields,
        }
    }

    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    pub fn reconstruct_frame(
        &self,
        frame_index: usize,
        fail_values: &[i64],
    ) -> Option<ReconstructedFrame> {
        let frame = self.frame_layouts.get(frame_index)?;
        Some(ReconstructedFrame {
            trace_id: frame.trace_id,
            header_pc: frame.header_pc,
            source_guard: frame.source_guard,
            pc: frame.pc,
            jitcode_index: 0,
            slot_types: frame.slot_types.clone(),
            values: frame
                .slot_layouts
                .iter()
                .map(|slot| {
                    ResumeData::resolve_frame_slot_source(&slot.to_resume_source(), fail_values)
                })
                .collect(),
        })
    }

    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        self.to_resume_data().materialize_virtuals(fail_values)
    }

    pub fn resolve_pending_field_writes(
        &self,
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        let resume_data = self.to_resume_data();
        ResumeData::resolve_pending_field_writes(&resume_data.pending_fields, fail_values)
    }
}

fn can_inline_tagged(value: i64) -> bool {
    (INLINE_TAGGED_MIN..=INLINE_TAGGED_MAX).contains(&value)
}

/// resume.py:161-188 getconst + resume.py:199-226 _number_boxes
///
/// Encode a ResumeValueSource as an i64 tagged value for rd_numb.
/// Assigns compact sequential TAGBOX numbers via liveboxes/box_map,
/// matching RPython's `_number_boxes` dedup assignment.
fn encode_tagged_source(
    source: &ResumeValueSource,
    rd_consts: &mut Vec<i64>,
    const_indices: &mut HashMap<i64, usize>,
    liveboxes: &mut Vec<usize>,
    box_map: &mut HashMap<usize, usize>,
) -> i64 {
    match source {
        // resume.py:214-224: new box → liveboxes[box] = tag(num_boxes, TAGBOX)
        ResumeValueSource::FailArg(index) => {
            let compact = *box_map.entry(*index).or_insert_with(|| {
                let n = liveboxes.len();
                liveboxes.push(*index);
                n
            });
            tag_i64(encode_len(compact), TAGBOX)
        }
        // resume.py:209: isinstance(box, Const) → self.getconst(box)
        ResumeValueSource::Constant(value) if can_inline_tagged(*value) => {
            // resume.py:163-167: try tag(val, TAGINT)
            tag_i64(*value, TAGINT)
        }
        ResumeValueSource::Constant(value) => {
            // resume.py:168-172: large int → _newconst → tag(index, TAGCONST)
            let next_index = rd_consts.len();
            let index = *const_indices.entry(*value).or_insert_with(|| {
                rd_consts.push(*value);
                next_index
            });
            tag_i64(encode_len(index), TAGCONST)
        }
        // resume.py:219-221: virtual → tag(num_virtuals, TAGVIRTUAL)
        ResumeValueSource::Virtual(index) => tag_i64(encode_len(*index), TAGVIRTUAL),
        ResumeValueSource::Uninitialized => tag_i64(ENCODED_UNINITIALIZED, TAGCONST),
        ResumeValueSource::Unavailable => tag_i64(ENCODED_UNAVAILABLE, TAGCONST),
    }
}

/// resume.py:99-104 tag() — i64 widened variant for rd_numb encoding.
fn tag_i64(value: i64, tagbits: u8) -> i64 {
    debug_assert!(tagbits <= 3);
    debug_assert!(
        can_inline_tagged(value),
        "tagged resume value {value} exceeds inline range"
    );
    (value << 2) | tagbits as i64
}

/// resume.py:106-109 untag() — i64 widened variant for rd_numb decoding.
fn untag_i64(encoded: i64) -> (i64, u8) {
    ((encoded >> 2), (encoded & TAGMASK_I64) as u8)
}

fn encode_len(value: usize) -> i64 {
    i64::try_from(value).expect("resume length exceeds i64")
}

fn decode_len(value: i64) -> usize {
    usize::try_from(value).expect("negative or oversized resume length")
}

fn encode_u64(value: u64) -> i64 {
    value as i64
}

fn decode_u64(value: i64) -> u64 {
    value as u64
}

/// Describes how to reconstruct a single frame in the interpreter's call stack.
///
/// Each frame has a bytecode position (pc) and a set of named/indexed slots
/// that map to tagged resume sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameInfo {
    /// Bytecode position (program counter) for this frame.
    pub pc: u64,
    /// Mapping from slot index to a tagged resume source.
    pub slot_map: Vec<FrameSlotSource>,
}

/// Complete resume data for a guard exit point.
///
/// Contains enough information to reconstruct the full interpreter state
/// from the values stored in a DeadFrame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeData {
    /// resume.py: snapshot_iter.vable_array / virtualizable_boxes
    pub vable_array: Vec<ResumeValueSource>,
    /// Stack of frames, outermost first.
    /// For a simple non-inlined trace, this has exactly one entry.
    pub frames: Vec<FrameInfo>,
    /// Virtual object descriptions for virtualized state.
    /// Each entry maps a fail_arg position to a virtual object that needs
    /// to be materialized when resuming.
    pub virtuals: Vec<VirtualInfo>,
    /// Deferred heap writes that must be replayed when resuming.
    ///
    /// Mirrors RPython's `rd_pendingfields`, which applies writes after
    /// virtuals and boxes have been reconstructed.
    pub pending_fields: Vec<PendingFieldInfo>,
}

/// Tagged source for a value that must be reconstructed on resume.
///
/// This is the majit equivalent of the tagged numbering used by
/// `rpython/jit/metainterp/resume.py`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeValueSource {
    /// Value comes from the deadframe fail-args array.
    FailArg(usize),
    /// Value is a compile-time constant.
    Constant(i64),
    /// Value is a virtual object that must be materialized on resume.
    Virtual(usize),
    /// Value exists conceptually but is still uninitialized.
    ///
    /// Mirrors RPython's `UNINITIALIZED` tag used for string/unicode content.
    Uninitialized,
    /// Slot is not live at this guard.
    Unavailable,
}

impl ResumeValueSource {
    pub fn kind(&self) -> ResumeValueKind {
        match self {
            ResumeValueSource::FailArg(_) => ResumeValueKind::FailArg,
            ResumeValueSource::Constant(_) => ResumeValueKind::Constant,
            ResumeValueSource::Virtual(_) => ResumeValueKind::Virtual,
            ResumeValueSource::Uninitialized => ResumeValueKind::Uninitialized,
            ResumeValueSource::Unavailable => ResumeValueKind::Unavailable,
        }
    }

    pub fn layout_summary(&self) -> ResumeValueLayoutSummary {
        match self {
            ResumeValueSource::FailArg(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::FailArg,
                fail_arg_index: *index,
                raw_fail_arg_position: Some(*index),
                constant: None,
                virtual_index: None,
            },
            ResumeValueSource::Constant(value) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Constant,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: Some(*value),
                virtual_index: None,
            },
            ResumeValueSource::Virtual(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Virtual,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: Some(*index),
            },
            ResumeValueSource::Uninitialized => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Uninitialized,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: None,
            },
            ResumeValueSource::Unavailable => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Unavailable,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: None,
            },
        }
    }
}

/// Source for a resumed frame slot.
pub type FrameSlotSource = ResumeValueSource;

/// Description of a virtual object that needs materialization on resume.
///
/// Mirrors RPython's AbstractVirtualInfo hierarchy:
/// - VirtualInfo (NEW_WITH_VTABLE)
/// - VStructInfo (NEW / plain struct)
/// - VArrayInfoClear / VArrayInfoNotClear (NEW_ARRAY)
/// - VArrayStructInfo (array of structs with interior fields)
/// - VRawBufferInfo (raw memory buffer)
#[derive(Debug, Clone)]
pub enum VirtualInfo {
    /// resume.py:612 VirtualInfo(descr, fielddescrs).
    VirtualObj {
        /// resume.py:615 self.descr — live SizeDescr.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        /// info.py:318 _known_class — vtable pointer.
        known_class: Option<i64>,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py:628 VStructInfo(typedescr, fielddescrs).
    VStruct {
        /// resume.py:631 self.typedescr — the full SizeDescr.
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    /// Virtual array (from NEW_ARRAY).
    VArray {
        /// Array descriptor index.
        descr_index: u32,
        /// Element values.
        items: Vec<VirtualFieldSource>,
    },
    /// Virtual array of structs (from arrays with interior field access).
    ///
    /// Each element is a struct with multiple fields. Mirrors RPython's
    /// VArrayStructInfo where each array slot contains a fixed-size struct.
    VArrayStruct {
        /// Array descriptor index.
        descr_index: u32,
        /// Per-element fields: outer Vec = elements, inner Vec = (field_index, source).
        element_fields: Vec<Vec<(u32, VirtualFieldSource)>>,
    },
    /// Virtual raw memory buffer (from raw_malloc).
    VRawBuffer {
        /// Size of the buffer in bytes.
        size: usize,
        /// Values stored at byte offsets: (offset, size_in_bytes, source).
        entries: Vec<(usize, usize, VirtualFieldSource)>,
    },
    /// resume.py: VRawSliceInfo — a slice into a virtual raw buffer.
    VRawSlice {
        /// Offset from the parent raw buffer.
        offset: i64,
        /// Source of the parent buffer.
        parent: VirtualFieldSource,
    },
    /// resume.py: VStrPlainInfo — virtual string (known characters).
    VStrPlain {
        /// Character values (as OpRef sources).
        chars: Vec<VirtualFieldSource>,
    },
    /// resume.py: VStrConcatInfo — virtual string concat (left + right).
    VStrConcat {
        left: Box<VirtualFieldSource>,
        right: Box<VirtualFieldSource>,
    },
    /// resume.py: VStrSliceInfo — virtual string slice.
    VStrSlice {
        source: Box<VirtualFieldSource>,
        start: Box<VirtualFieldSource>,
        length: Box<VirtualFieldSource>,
    },
    /// resume.py: VUniPlainInfo — virtual unicode string.
    VUniPlain { chars: Vec<VirtualFieldSource> },
    /// resume.py: VUniConcatInfo — virtual unicode concat.
    VUniConcat {
        left: Box<VirtualFieldSource>,
        right: Box<VirtualFieldSource>,
    },
    /// resume.py: VUniSliceInfo — virtual unicode slice.
    VUniSlice {
        source: Box<VirtualFieldSource>,
        start: Box<VirtualFieldSource>,
        length: Box<VirtualFieldSource>,
    },
}

// PartialEq/Eq: compare by data fields, skip descr/typedescr (Arc<dyn Descr>).
impl PartialEq for VirtualInfo {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                VirtualInfo::VirtualObj {
                    descr: None,
                    type_id: a1,
                    descr_index: a2,
                    fields: a3,
                    fielddescrs: a4,
                    descr_size: a5,
                    ..
                },
                VirtualInfo::VirtualObj {
                    descr: None,
                    type_id: b1,
                    descr_index: b2,
                    fields: b3,
                    fielddescrs: b4,
                    descr_size: b5,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5,
            (
                VirtualInfo::VStruct {
                    type_id: a1,
                    descr_index: a2,
                    fields: a3,
                    fielddescrs: a4,
                    descr_size: a5,
                    ..
                },
                VirtualInfo::VStruct {
                    type_id: b1,
                    descr_index: b2,
                    fields: b3,
                    fielddescrs: b4,
                    descr_size: b5,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5,
            (
                VirtualInfo::VArray {
                    descr_index: a1,
                    items: a2,
                },
                VirtualInfo::VArray {
                    descr_index: b1,
                    items: b2,
                },
            ) => a1 == b1 && a2 == b2,
            _ => false,
        }
    }
}
impl Eq for VirtualInfo {}

impl VirtualInfo {
    /// Iterate over all field sources in this virtual.
    /// resume.py: visitor_walk_recursive walks all box references in a virtual.
    pub fn field_sources(&self) -> Vec<&VirtualFieldSource> {
        match self {
            VirtualInfo::VirtualObj { fields, .. } | VirtualInfo::VStruct { fields, .. } => {
                fields.iter().map(|(_, src)| src).collect()
            }
            VirtualInfo::VArray { items, .. } => items.iter().collect(),
            VirtualInfo::VArrayStruct { element_fields, .. } => element_fields
                .iter()
                .flat_map(|el| el.iter().map(|(_, src)| src))
                .collect(),
            VirtualInfo::VRawBuffer { entries, .. } => {
                entries.iter().map(|(_, _, src)| src).collect()
            }
            VirtualInfo::VRawSlice { parent, .. } => vec![parent],
            VirtualInfo::VStrPlain { chars } | VirtualInfo::VUniPlain { chars } => {
                chars.iter().collect()
            }
            VirtualInfo::VStrConcat { left, right } | VirtualInfo::VUniConcat { left, right } => {
                vec![left.as_ref(), right.as_ref()]
            }
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
            }
            | VirtualInfo::VUniSlice {
                source,
                start,
                length,
            } => vec![source.as_ref(), start.as_ref(), length.as_ref()],
        }
    }

    pub fn kind(&self) -> ResumeVirtualKind {
        match self {
            VirtualInfo::VirtualObj { .. } => ResumeVirtualKind::Object,
            VirtualInfo::VStruct { .. } => ResumeVirtualKind::Struct,
            VirtualInfo::VArray { .. } => ResumeVirtualKind::Array,
            VirtualInfo::VArrayStruct { .. } => ResumeVirtualKind::ArrayStruct,
            VirtualInfo::VRawBuffer { .. } | VirtualInfo::VRawSlice { .. } => {
                ResumeVirtualKind::RawBuffer
            }
            VirtualInfo::VStrPlain { .. }
            | VirtualInfo::VStrConcat { .. }
            | VirtualInfo::VStrSlice { .. }
            | VirtualInfo::VUniPlain { .. }
            | VirtualInfo::VUniConcat { .. }
            | VirtualInfo::VUniSlice { .. } => ResumeVirtualKind::Struct,
        }
    }

    pub fn layout_summary(&self) -> ResumeVirtualLayoutSummary {
        match self {
            VirtualInfo::VirtualObj {
                descr,
                type_id,
                descr_index,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => ResumeVirtualLayoutSummary::Object {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.layout_summary()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            VirtualInfo::VStruct {
                typedescr,
                type_id,
                descr_index,
                fields,
                fielddescrs,
                descr_size,
            } => ResumeVirtualLayoutSummary::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.layout_summary()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            VirtualInfo::VArray { descr_index, items } => ResumeVirtualLayoutSummary::Array {
                descr_index: *descr_index,
                items: items.iter().map(|source| source.layout_summary()).collect(),
            },
            VirtualInfo::VArrayStruct {
                descr_index,
                element_fields,
            } => {
                let fpe = element_fields.first().map(|ef| ef.len()).unwrap_or(0);
                ResumeVirtualLayoutSummary::ArrayStruct {
                    descr_index: *descr_index,
                    field_types: vec![0u8; fpe],
                    element_fields: element_fields
                        .iter()
                        .map(|fields| {
                            fields
                                .iter()
                                .map(|(field_descr, source)| {
                                    (*field_descr, source.layout_summary())
                                })
                                .collect()
                        })
                        .collect(),
                }
            }
            VirtualInfo::VRawBuffer { size, entries } => ResumeVirtualLayoutSummary::RawBuffer {
                size: *size,
                entries: entries
                    .iter()
                    .map(|(offset, size_in_bytes, source)| {
                        (*offset, *size_in_bytes, source.layout_summary())
                    })
                    .collect(),
            },
            // String/unicode virtual infos — represented as structs
            // with synthetic field indices for reconstruction.
            VirtualInfo::VRawSlice { .. }
            | VirtualInfo::VStrPlain { .. }
            | VirtualInfo::VStrConcat { .. }
            | VirtualInfo::VStrSlice { .. }
            | VirtualInfo::VUniPlain { .. }
            | VirtualInfo::VUniConcat { .. }
            | VirtualInfo::VUniSlice { .. } => ResumeVirtualLayoutSummary::Struct {
                typedescr: None,
                type_id: 0,
                descr_index: 0,
                fields: vec![],
                fielddescrs: vec![],
                descr_size: 0,
            },
        }
    }
}

/// Source of a virtual object's field value.
pub type VirtualFieldSource = ResumeValueSource;

/// Convert a tagged fieldnum (i16, resume.py encoding) to a VirtualFieldSource.
///
/// resume.py:1552-1596 decode_int/decode_ref: tagged values encode where
/// each field value comes from at resume time.
///
/// `consts` is the rd_consts array. `count` is the number of fail_args
/// (used for negative TAGBOX indices). Both come from the containing
/// ResumeGuardDescr / EncodedResumeData.
pub fn tagged_to_source(tagged: i16, consts: &[i64], count: i32) -> VirtualFieldSource {
    if tagged_eq(tagged, UNASSIGNED) {
        return ResumeValueSource::Unavailable;
    }
    if tagged_eq(tagged, UNINITIALIZED_TAG) {
        return ResumeValueSource::Uninitialized;
    }
    if tagged_eq(tagged, NULLREF) {
        return ResumeValueSource::Constant(0);
    }
    let (num, tag_bits) = untag(tagged);
    match tag_bits {
        TAGCONST => {
            let idx = (num - TAG_CONST_OFFSET) as usize;
            if idx < consts.len() {
                ResumeValueSource::Constant(consts[idx])
            } else {
                ResumeValueSource::Constant(0)
            }
        }
        TAGINT => ResumeValueSource::Constant(num as i64),
        TAGBOX => {
            let mut idx = num;
            if idx < 0 {
                idx += count;
            }
            ResumeValueSource::FailArg(idx as usize)
        }
        TAGVIRTUAL => ResumeValueSource::Virtual(num as usize),
        _ => ResumeValueSource::Unavailable,
    }
}

/// Convert an `RdVirtualInfo` (IR-level, from compile.rs/pyjitpl.rs)
/// to a `VirtualInfo` (resume-level, used by ResumeDataDirectReader).
///
/// `consts` and `count` are needed to decode tagged fieldnums.
pub fn rd_virtual_to_virtual_info(
    rd: &majit_ir::RdVirtualInfo,
    consts: &[i64],
    count: i32,
) -> VirtualInfo {
    match rd {
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            type_id,
            descr_index,
            known_class,
            fielddescrs,
            fieldnums,
            descr_size,
        } => {
            let fields = fielddescrs
                .iter()
                .zip(fieldnums.iter())
                .map(|(fd, &tagged)| (fd.index, tagged_to_source(tagged, consts, count)))
                .collect();
            VirtualInfo::VirtualObj {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                known_class: *known_class,
                fields,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            }
        }
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            type_id,
            descr_index,
            fielddescrs,
            fieldnums,
            descr_size,
        } => {
            let fields = fielddescrs
                .iter()
                .zip(fieldnums.iter())
                .map(|(fd, &tagged)| (fd.index, tagged_to_source(tagged, consts, count)))
                .collect();
            VirtualInfo::VStruct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
                fields,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            }
        }
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            descr_index,
            fieldnums,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            descr_index,
            fieldnums,
            ..
        } => {
            let items = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count))
                .collect();
            VirtualInfo::VArray {
                descr_index: *descr_index,
                items,
            }
        }
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            descr_index,
            size,
            fielddescr_indices,
            fieldnums,
            ..
        } => {
            let num_fields = fielddescr_indices.len().max(1);
            let mut element_fields = Vec::new();
            for chunk in fieldnums.chunks(num_fields) {
                let elem: Vec<(u32, VirtualFieldSource)> = fielddescr_indices
                    .iter()
                    .zip(chunk.iter())
                    .map(|(&fd_idx, &tagged)| (fd_idx, tagged_to_source(tagged, consts, count)))
                    .collect();
                element_fields.push(elem);
            }
            if element_fields.is_empty() {
                for _ in 0..*size {
                    element_fields.push(vec![]);
                }
            }
            VirtualInfo::VArrayStruct {
                descr_index: *descr_index,
                element_fields,
            }
        }
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            size,
            offsets,
            entry_sizes,
            fieldnums,
        } => {
            let entries = offsets
                .iter()
                .zip(entry_sizes.iter())
                .zip(fieldnums.iter())
                .map(|((&off, &sz), &tagged)| (off, sz, tagged_to_source(tagged, consts, count)))
                .collect();
            VirtualInfo::VRawBuffer {
                size: *size,
                entries,
            }
        }
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            let parent = fieldnums
                .first()
                .map(|&tagged| tagged_to_source(tagged, consts, count))
                .unwrap_or(ResumeValueSource::Unavailable);
            VirtualInfo::VRawSlice {
                offset: *offset as i64,
                parent,
            }
        }
        majit_ir::RdVirtualInfo::Empty => VirtualInfo::VirtualObj {
            descr: None,
            type_id: 0,
            descr_index: 0,
            known_class: None,
            fields: vec![],
            fielddescrs: vec![],
            descr_size: 0,
        },
    }
}

/// Deferred heap write to replay during resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingFieldInfo {
    /// Descriptor index identifying the target field or array descriptor.
    pub descr_index: u32,
    /// Source of the object/array pointer to update.
    pub target: ResumeValueSource,
    /// Source of the value to write.
    pub value: ResumeValueSource,
    /// Array item index. `None` means a plain field write.
    pub item_index: Option<usize>,
}

impl PendingFieldInfo {
    pub fn layout_summary(&self) -> PendingFieldLayoutSummary {
        PendingFieldLayoutSummary {
            descr_index: self.descr_index,
            item_index: self.item_index,
            is_array_item: self.item_index.is_some(),
            target_kind: self.target.kind(),
            value_kind: self.value.kind(),
            target: self.target.layout_summary(),
            value: self.value.layout_summary(),
        }
    }
}

/// Concrete pending heap write reconstructed from resume data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPendingFieldWrite {
    /// Descriptor index identifying the field or array descriptor.
    pub descr_index: u32,
    /// Concrete object/array pointer.
    pub target: MaterializedValue,
    /// Concrete value to write.
    pub value: MaterializedValue,
    /// Array item index. `None` means a plain field write.
    pub item_index: Option<usize>,
}

/// Encoded pending field write stored alongside an encoded resume snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPendingFieldWrite {
    pub descr_index: u32,
    pub target: i64,
    pub value: i64,
    pub item_index: Option<usize>,
}

impl EncodedResumeData {
    pub fn encode(rd: &ResumeData) -> Self {
        Self::from_semantic(
            &rd.vable_array,
            &rd.frames,
            &rd.virtuals,
            &rd.pending_fields,
        )
    }

    /// resume.py:231-267 number + resume.py:380-468 finish
    ///
    /// Walks all frames via _number_boxes, assigning compact sequential
    /// TAGBOX numbers to unique liveboxes (resume.py:199-226).
    fn from_semantic(
        vable_array: &[ResumeValueSource],
        frames: &[FrameInfo],
        virtuals: &[VirtualInfo],
        pending_fields: &[PendingFieldInfo],
    ) -> Self {
        let mut rd_numb = Vec::new();
        let mut rd_consts = Vec::new();
        let mut const_indices = HashMap::new();
        // resume.py:138 numb_state.liveboxes — compact TAGBOX numbering state.
        // liveboxes[compact_n] = original FailArg index.
        let mut liveboxes: Vec<usize> = Vec::new();
        let mut box_map: HashMap<usize, usize> = HashMap::new();

        // resume.py:234-235: reserve slots for items_resume_section and count.
        rd_numb.push(0); // [0] = items_resume_section (patched later)
        rd_numb.push(0); // [1] = count (patched later)
        rd_numb.push(encode_len(vable_array.len()));
        for source in vable_array {
            let tagged = encode_tagged_source(
                source,
                &mut rd_consts,
                &mut const_indices,
                &mut liveboxes,
                &mut box_map,
            );
            rd_numb.push(tagged);
        }
        rd_numb.push(encode_len(0));
        rd_numb.push(encode_len(frames.len()));

        // resume.py:249-258: per-frame encoding via _number_boxes
        for frame in frames {
            rd_numb.push(encode_u64(frame.pc));
            rd_numb.push(encode_len(frame.slot_map.len()));
            // resume.py:253 _number_boxes(snapshot_iter, iter_array(snapshot), numb_state)
            for source in &frame.slot_map {
                let tagged = encode_tagged_source(
                    source,
                    &mut rd_consts,
                    &mut const_indices,
                    &mut liveboxes,
                    &mut box_map,
                );
                rd_numb.push(tagged);
            }
        }

        // compile.py:858 rd_virtuals — stored as live objects, not serialized.
        let rd_virtuals = virtuals.to_vec();

        // resume.py:412-418: visitor_walk_recursive — register virtual field boxes.
        for vinfo in &rd_virtuals {
            for source in vinfo.field_sources() {
                if let ResumeValueSource::FailArg(index) = source {
                    box_map.entry(*index).or_insert_with(|| {
                        let n = liveboxes.len();
                        liveboxes.push(*index);
                        n
                    });
                }
            }
        }

        // resume.py:420-430: walk pending fields — register + encode.
        let rd_pendingfields: Vec<_> = pending_fields
            .iter()
            .map(|pending| EncodedPendingFieldWrite {
                descr_index: pending.descr_index,
                target: encode_tagged_source(
                    &pending.target,
                    &mut rd_consts,
                    &mut const_indices,
                    &mut liveboxes,
                    &mut box_map,
                ),
                value: encode_tagged_source(
                    &pending.value,
                    &mut rd_consts,
                    &mut const_indices,
                    &mut liveboxes,
                    &mut box_map,
                ),
                item_index: pending.item_index,
            })
            .collect();

        // resume.py:260: numb_state.patch_current_size(0) → items_resume_section
        rd_numb[0] = encode_len(rd_numb.len());
        // resume.py:464: numb_state.patch(1, len(liveboxes)) → count
        rd_numb[1] = encode_len(liveboxes.len());

        EncodedResumeData {
            rd_numb,
            rd_consts,
            rd_pendingfields,
            rd_virtuals,
            liveboxes,
        }
    }

    /// resume.py:916-923 AbstractResumeDataReader._init — decode rd_numb.
    fn decode_layout(&self) -> DecodedResumeLayout {
        let mut cursor = 0usize;
        // resume.py:919 items_resume_section
        let items_resume_section = self.next_word(&mut cursor);
        assert_eq!(
            decode_len(items_resume_section),
            self.rd_numb.len(),
            "resume item count mismatch"
        );
        // resume.py:921 self.count — number of liveboxes in the deadframe.
        let _count = decode_len(self.next_word(&mut cursor));

        let vable_count = decode_len(self.next_word(&mut cursor));
        let mut vable_array = Vec::with_capacity(vable_count);
        for _ in 0..vable_count {
            vable_array.push(self.decode_box(self.next_word(&mut cursor)));
        }
        let vref_count = decode_len(self.next_word(&mut cursor));
        for _ in 0..(vref_count * 2) {
            let _ = self.decode_box(self.next_word(&mut cursor));
        }
        let num_frames = decode_len(self.next_word(&mut cursor));
        let mut frames = Vec::with_capacity(num_frames);
        for _ in 0..num_frames {
            let pc = decode_u64(self.next_word(&mut cursor));
            let slot_count = decode_len(self.next_word(&mut cursor));
            let mut slot_map = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                slot_map.push(self.decode_box(self.next_word(&mut cursor)));
            }
            frames.push(FrameInfo { pc, slot_map });
        }

        // compile.py:858 rd_virtuals — live objects, not deserialized from rd_numb.
        let virtuals = self.rd_virtuals.clone();

        assert_eq!(
            cursor,
            self.rd_numb.len(),
            "resume decoder left trailing data"
        );
        // resume.py:926 _prepare_pendingfields
        let pending_fields = self
            .rd_pendingfields
            .iter()
            .map(|pending| PendingFieldInfo {
                descr_index: pending.descr_index,
                target: self.decode_box(pending.target),
                value: self.decode_box(pending.value),
                item_index: pending.item_index,
            })
            .collect();
        DecodedResumeLayout {
            vable_array,
            frames,
            virtuals,
            pending_fields,
        }
    }

    /// resume.py:919 resumecodereader.next_item()
    fn next_word(&self, cursor: &mut usize) -> i64 {
        let word = self
            .rd_numb
            .get(*cursor)
            .copied()
            .expect("truncated encoded resume data");
        *cursor += 1;
        word
    }

    /// resume.py:1240-1270 decode_box — decode a tagged value from rd_numb.
    fn decode_box(&self, encoded: i64) -> ResumeValueSource {
        let (value, tag) = untag_i64(encoded);
        match tag {
            TAGINT => ResumeValueSource::Constant(value),
            // resume.py:1261 self.liveboxes[num] — compact TAGBOX → original FailArg.
            TAGBOX => {
                let compact_idx = decode_len(value);
                let original_idx = self.liveboxes[compact_idx];
                ResumeValueSource::FailArg(original_idx)
            }
            TAGVIRTUAL => ResumeValueSource::Virtual(decode_len(value)),
            TAGCONST => match value {
                ENCODED_UNINITIALIZED => ResumeValueSource::Uninitialized,
                ENCODED_UNAVAILABLE => ResumeValueSource::Unavailable,
                index if index >= 0 => ResumeValueSource::Constant(
                    *self
                        .rd_consts
                        .get(decode_len(index))
                        .expect("resume const pool index out of bounds"),
                ),
                other => panic!("unknown CONST-tagged resume sentinel {other}"),
            },
            other => panic!("unknown resume tag {other}"),
        }
    }

    /// Decode this encoded snapshot back into a `ResumeData`.
    pub fn decode(&self) -> ResumeData {
        let layout = self.decode_layout();
        ResumeData {
            vable_array: layout.vable_array,
            frames: layout.frames,
            virtuals: layout.virtuals,
            pending_fields: layout.pending_fields,
        }
    }

    /// Return a compact summary of this snapshot's frame/jitframe layout.
    pub fn layout_summary(&self) -> ResumeLayoutSummary {
        let layout = self.decode_layout();
        ResumeLayoutSummary {
            num_frames: layout.frames.len(),
            frame_pcs: layout.frames.iter().map(|frame| frame.pc).collect(),
            frame_slot_counts: layout
                .frames
                .iter()
                .map(|frame| frame.slot_map.len())
                .collect(),
            frame_layouts: layout
                .frames
                .iter()
                .map(|frame| ResumeFrameLayoutSummary {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: frame.pc,
                    slot_sources: frame.slot_map.iter().map(ResumeValueSource::kind).collect(),
                    slot_layouts: frame
                        .slot_map
                        .iter()
                        .map(|source| source.layout_summary())
                        .collect(),
                    slot_types: None,
                })
                .collect(),
            num_virtuals: layout.virtuals.len(),
            virtual_kinds: layout.virtuals.iter().map(VirtualInfo::kind).collect(),
            virtual_layouts: layout
                .virtuals
                .iter()
                .map(|virt| virt.layout_summary())
                .collect(),

            pending_field_count: layout.pending_fields.len(),
            pending_field_layouts: layout
                .pending_fields
                .iter()
                .map(|pending| pending.layout_summary())
                .collect(),
            const_pool_size: self.rd_consts.len(),
        }
    }

    /// Reconstruct the full interpreter state directly from the encoded snapshot.
    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let layout = self.decode_layout();
        let virtuals = ResumeData::materialize_virtuals_from_infos(&layout.virtuals, fail_values);
        let pending_fields =
            ResumeData::resolve_pending_field_writes(&layout.pending_fields, fail_values);
        let frames = layout
            .frames
            .iter()
            .map(|frame| ReconstructedFrame {
                trace_id: None,
                header_pc: None,
                source_guard: None,
                pc: frame.pc,
                jitcode_index: 0,
                slot_types: None,
                values: frame
                    .slot_map
                    .iter()
                    .map(|slot| ResumeData::resolve_frame_slot_source(slot, fail_values))
                    .collect(),
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals,
            pending_fields,
        }
    }

    /// Reconstruct only the interpreter frames from the encoded snapshot.
    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    /// Materialize virtual objects referenced by this encoded snapshot.
    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        let layout = self.decode_layout();
        ResumeData::materialize_virtuals_from_infos(&layout.virtuals, fail_values)
    }

    /// Resolve pending heap writes referenced by this encoded snapshot.
    pub fn resolve_pending_field_writes(
        &self,
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        let layout = self.decode_layout();
        ResumeData::resolve_pending_field_writes(&layout.pending_fields, fail_values)
    }
}

impl ResumeData {
    /// Create a simple ResumeData for a single-frame trace.
    pub fn simple(pc: u64, num_slots: usize) -> Self {
        let slot_map: Vec<FrameSlotSource> = (0..num_slots).map(FrameSlotSource::FailArg).collect();
        ResumeData {
            vable_array: Vec::new(),
            frames: vec![FrameInfo { pc, slot_map }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    /// Encode this resume snapshot into a compact RPython-style numbering.
    pub fn encode(&self) -> EncodedResumeData {
        EncodedResumeData::from_semantic(
            &self.vable_array,
            &self.frames,
            &self.virtuals,
            &self.pending_fields,
        )
    }

    fn decode_layout(&self) -> DecodedResumeLayout {
        self.encode().decode_layout()
    }

    /// Reconstruct the full resume state from fail_args data.
    ///
    /// `fail_values` are the concrete values extracted from the DeadFrame.
    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let decoded = self.decode_layout();
        let materialized_virtuals =
            Self::materialize_virtuals_from_infos(&decoded.virtuals, fail_values);
        let frames = decoded
            .frames
            .iter()
            .map(|frame| {
                let values = frame
                    .slot_map
                    .iter()
                    .map(|slot| Self::resolve_frame_slot_source(slot, fail_values))
                    .collect();
                ReconstructedFrame {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: frame.pc,
                    jitcode_index: 0,
                    slot_types: None,
                    values,
                }
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals: materialized_virtuals,
            pending_fields: Self::resolve_pending_field_writes(
                &decoded.pending_fields,
                fail_values,
            ),
        }
    }

    /// Reconstruct frame slots from fail_args data.
    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    /// Materialize virtual objects from resume data.
    ///
    /// When a guard fails and some fail_args slots hold virtual objects
    /// (objects that were never allocated during optimized execution),
    /// this method resolves each VirtualInfo into a `MaterializedVirtual`.
    ///
    /// `fail_values` are the concrete i64 values from the DeadFrame.
    ///
    /// Mirrors RPython's `ResumeDataVirtualAdder.finish()` → `virtual_materialize()`.
    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        let decoded = self.decode_layout();
        Self::materialize_virtuals_from_infos(&decoded.virtuals, fail_values)
    }

    fn materialize_virtuals_from_infos(
        virtuals: &[VirtualInfo],
        fail_values: &[i64],
    ) -> Vec<MaterializedVirtual> {
        let mut result = Vec::with_capacity(virtuals.len());
        // First pass: create empty materialized shells (handles forward references)
        for vinfo in virtuals {
            result.push(MaterializedVirtual::from_info(vinfo));
        }

        // Second pass: resolve field sources
        for (i, vinfo) in virtuals.iter().enumerate() {
            result[i].resolve_fields(vinfo, fail_values);
        }

        result
    }

    /// Resolve pending heap writes into concrete values.
    pub fn resolve_pending_field_writes(
        pending_fields: &[PendingFieldInfo],
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        pending_fields
            .iter()
            .map(|pending| ResolvedPendingFieldWrite {
                descr_index: pending.descr_index,
                target: Self::resolve_materialized_source(&pending.target, fail_values),
                value: Self::resolve_materialized_source(&pending.value, fail_values),
                item_index: pending.item_index,
            })
            .collect()
    }

    /// Resolve a single VirtualFieldSource to a concrete i64 value.
    ///
    /// For `Virtual(idx)` references, returns 0 as a placeholder —
    /// the actual object address is only known after allocation by the
    /// interpreter's allocator.
    pub fn resolve_field_source(source: &VirtualFieldSource, fail_values: &[i64]) -> i64 {
        match Self::resolve_materialized_source(source, fail_values) {
            MaterializedValue::Value(value) => value,
            MaterializedValue::VirtualRef(_) => 0,
        }
    }

    pub fn resolve_materialized_source(
        source: &VirtualFieldSource,
        fail_values: &[i64],
    ) -> MaterializedValue {
        match source {
            ResumeValueSource::FailArg(idx) => {
                MaterializedValue::Value(fail_values.get(*idx).copied().unwrap_or(0))
            }
            ResumeValueSource::Constant(val) => MaterializedValue::Value(*val),
            ResumeValueSource::Virtual(idx) => MaterializedValue::VirtualRef(*idx),
            ResumeValueSource::Uninitialized | ResumeValueSource::Unavailable => {
                MaterializedValue::Value(0)
            }
        }
    }

    /// Resolve a single frame-slot source into a reconstructed value.
    pub fn resolve_frame_slot_source(
        source: &FrameSlotSource,
        fail_values: &[i64],
    ) -> ReconstructedValue {
        match source {
            ResumeValueSource::FailArg(idx) => {
                ReconstructedValue::Value(fail_values.get(*idx).copied().unwrap_or(0))
            }
            ResumeValueSource::Constant(val) => ReconstructedValue::Value(*val),
            ResumeValueSource::Virtual(idx) => ReconstructedValue::Virtual(*idx),
            ResumeValueSource::Uninitialized => ReconstructedValue::Uninitialized,
            ResumeValueSource::Unavailable => ReconstructedValue::Unavailable,
        }
    }
}

/// A reconstructed interpreter frame from resume data.
#[derive(Debug, Clone)]
pub struct ReconstructedFrame {
    /// Compiled trace identifier for this frame, when known.
    pub trace_id: Option<u64>,
    /// Trace header pc associated with this frame, when known.
    pub header_pc: Option<u64>,
    /// Source guard this frame's trace is attached to, when known.
    pub source_guard: Option<(u64, u32)>,
    /// Program counter for this frame.
    pub pc: u64,
    /// resume.py:1051: jitcode index for CodeObject lookup.
    pub jitcode_index: i32,
    /// Typed layout of the reconstructed slots, when known.
    pub slot_types: Option<Vec<Type>>,
    /// Reconstructed values for each slot.
    pub values: Vec<ReconstructedValue>,
}

impl ReconstructedFrame {
    /// Lossy conversion: extract integer values, dropping virtual/unavailable info.
    pub fn lossy_values(&self) -> Vec<i64> {
        self.values
            .iter()
            .map(ReconstructedValue::lossy_i64)
            .collect()
    }
}

/// Full reconstructed state for a guard recovery.
#[derive(Debug, Clone)]
pub struct ReconstructedState {
    /// Reconstructed interpreter frames, outermost first.
    pub frames: Vec<ReconstructedFrame>,
    /// Materialized virtual objects referenced by frame slots.
    pub virtuals: Vec<MaterializedVirtual>,
    /// Deferred heap writes that the interpreter must replay after reconstruction.
    pub pending_fields: Vec<ResolvedPendingFieldWrite>,
}

/// Reconstructed slot value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructedValue {
    /// Concrete raw value, including ints, refs, and float bits.
    Value(i64),
    /// Reference to a materialized virtual in `ReconstructedState.virtuals`.
    Virtual(usize),
    /// Slot exists but remains uninitialized.
    Uninitialized,
    /// Slot is dead/unavailable at this guard.
    Unavailable,
}

impl ReconstructedValue {
    /// Lossy conversion used by the current integer-only compatibility layer.
    pub fn lossy_i64(&self) -> i64 {
        match self {
            ReconstructedValue::Value(value) => *value,
            ReconstructedValue::Virtual(_)
            | ReconstructedValue::Uninitialized
            | ReconstructedValue::Unavailable => 0,
        }
    }
}

/// A materialized virtual object, ready for the interpreter to allocate.
///
/// After a guard failure, virtual objects must be allocated on the heap
/// and their fields populated from the DeadFrame values. This struct
/// holds the resolved field values for a single virtual object.
///
/// Mirrors RPython's `_materialize_virtual()` in resume.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterializedValue {
    Value(i64),
    VirtualRef(usize),
}

impl MaterializedValue {
    pub fn resolve_with_refs(&self, materialized_refs: &[Option<GcRef>]) -> Option<i64> {
        match self {
            MaterializedValue::Value(value) => Some(*value),
            MaterializedValue::VirtualRef(index) => materialized_refs
                .get(*index)
                .copied()
                .flatten()
                .map(|gc_ref| gc_ref.as_usize() as i64),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterializedVirtual {
    /// Object with vtable.
    Obj {
        type_id: u32,
        descr_index: u32,
        /// (field_descr_index, concrete_value).
        fields: Vec<(u32, MaterializedValue)>,
    },
    /// Plain struct.
    Struct {
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, MaterializedValue)>,
    },
    /// Array.
    Array {
        descr_index: u32,
        items: Vec<MaterializedValue>,
    },
    /// Array of structs.
    ArrayStruct {
        descr_index: u32,
        /// Per-element: Vec<(field_index, value)>.
        elements: Vec<Vec<(u32, MaterializedValue)>>,
    },
    /// Raw buffer.
    RawBuffer {
        size: usize,
        /// (offset, size_in_bytes, value).
        entries: Vec<(usize, usize, MaterializedValue)>,
    },
}

impl MaterializedVirtual {
    /// Create an empty shell from a VirtualInfo (forward-reference safe).
    fn from_info(info: &VirtualInfo) -> Self {
        match info {
            VirtualInfo::VirtualObj {
                type_id,
                descr_index,
                ..
            } => MaterializedVirtual::Obj {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: Vec::new(),
            },
            VirtualInfo::VStruct {
                type_id,
                descr_index,
                ..
            } => MaterializedVirtual::Struct {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: Vec::new(),
            },
            VirtualInfo::VArray { descr_index, items } => MaterializedVirtual::Array {
                descr_index: *descr_index,
                items: vec![MaterializedValue::Value(0); items.len()],
            },
            VirtualInfo::VArrayStruct {
                descr_index,
                element_fields,
            } => MaterializedVirtual::ArrayStruct {
                descr_index: *descr_index,
                elements: vec![Vec::new(); element_fields.len()],
            },
            VirtualInfo::VRawBuffer { size, .. } => MaterializedVirtual::RawBuffer {
                size: *size,
                entries: Vec::new(),
            },
            VirtualInfo::VRawSlice { .. }
            | VirtualInfo::VStrPlain { .. }
            | VirtualInfo::VStrConcat { .. }
            | VirtualInfo::VStrSlice { .. }
            | VirtualInfo::VUniPlain { .. }
            | VirtualInfo::VUniConcat { .. }
            | VirtualInfo::VUniSlice { .. } => MaterializedVirtual::Struct {
                type_id: 0,
                descr_index: 0,
                fields: Vec::new(),
            },
        }
    }

    /// Resolve fields from fail_values.
    fn resolve_fields(&mut self, info: &VirtualInfo, fail_values: &[i64]) {
        match (self, info) {
            (
                MaterializedVirtual::Obj { fields, .. },
                VirtualInfo::VirtualObj {
                    fields: src_fields, ..
                },
            )
            | (
                MaterializedVirtual::Struct { fields, .. },
                VirtualInfo::VStruct {
                    fields: src_fields, ..
                },
            ) => {
                *fields = src_fields
                    .iter()
                    .map(|(idx, src)| {
                        (
                            *idx,
                            ResumeData::resolve_materialized_source(src, fail_values),
                        )
                    })
                    .collect();
            }
            (
                MaterializedVirtual::Array { items, .. },
                VirtualInfo::VArray {
                    items: src_items, ..
                },
            ) => {
                *items = src_items
                    .iter()
                    .map(|src| ResumeData::resolve_materialized_source(src, fail_values))
                    .collect();
            }
            (
                MaterializedVirtual::ArrayStruct { elements, .. },
                VirtualInfo::VArrayStruct {
                    element_fields: src_elems,
                    ..
                },
            ) => {
                *elements = src_elems
                    .iter()
                    .map(|elem_fields| {
                        elem_fields
                            .iter()
                            .map(|(idx, src)| {
                                (
                                    *idx,
                                    ResumeData::resolve_materialized_source(src, fail_values),
                                )
                            })
                            .collect()
                    })
                    .collect();
            }
            (
                MaterializedVirtual::RawBuffer { entries, .. },
                VirtualInfo::VRawBuffer {
                    entries: src_entries,
                    ..
                },
            ) => {
                *entries = src_entries
                    .iter()
                    .map(|(off, sz, src)| {
                        (
                            *off,
                            *sz,
                            ResumeData::resolve_materialized_source(src, fail_values),
                        )
                    })
                    .collect();
            }
            _ => {} // type mismatch — should not happen
        }
    }

    pub fn resolve_with_refs(
        &self,
        materialized_refs: &[Option<GcRef>],
    ) -> Option<MaterializedVirtual> {
        match self {
            MaterializedVirtual::Obj {
                type_id,
                descr_index,
                fields,
            } => Some(MaterializedVirtual::Obj {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(idx, value)| {
                        Some((
                            *idx,
                            MaterializedValue::Value(value.resolve_with_refs(materialized_refs)?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::Struct {
                type_id,
                descr_index,
                fields,
            } => Some(MaterializedVirtual::Struct {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(idx, value)| {
                        Some((
                            *idx,
                            MaterializedValue::Value(value.resolve_with_refs(materialized_refs)?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::Array { descr_index, items } => Some(MaterializedVirtual::Array {
                descr_index: *descr_index,
                items: items
                    .iter()
                    .map(|value| {
                        Some(MaterializedValue::Value(
                            value.resolve_with_refs(materialized_refs)?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::ArrayStruct {
                descr_index,
                elements,
            } => Some(MaterializedVirtual::ArrayStruct {
                descr_index: *descr_index,
                elements: elements
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(idx, value)| {
                                Some((
                                    *idx,
                                    MaterializedValue::Value(
                                        value.resolve_with_refs(materialized_refs)?,
                                    ),
                                ))
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::RawBuffer { size, entries } => {
                Some(MaterializedVirtual::RawBuffer {
                    size: *size,
                    entries: entries
                        .iter()
                        .map(|(offset, size_in_bytes, value)| {
                            Some((
                                *offset,
                                *size_in_bytes,
                                MaterializedValue::Value(
                                    value.resolve_with_refs(materialized_refs)?,
                                ),
                            ))
                        })
                        .collect::<Option<Vec<_>>>()?,
                })
            }
        }
    }
}

/// Builder for constructing ResumeData during trace compilation.
///
/// resume.py:298-493 ResumeDataVirtualAdder.finish() is implemented
/// across two functions in majit:
/// - `store_final_boxes_in_guard` (mod.rs) — numbering + rd_numb/rd_consts
/// - `store_final_boxes_in_guard` (optimizer.rs) — virtual expansion + rd_virtuals
pub struct ResumeDataVirtualAdder {
    vable_array: Vec<ResumeValueSource>,
    frames: Vec<FrameInfoBuilder>,
    virtuals: Vec<VirtualInfo>,
    pending_fields: Vec<PendingFieldInfo>,
}

struct FrameInfoBuilder {
    pc: u64,
    slot_map: Vec<FrameSlotSource>,
}

impl ResumeDataVirtualAdder {
    /// Create a new builder.
    pub fn new() -> Self {
        ResumeDataVirtualAdder {
            vable_array: Vec::new(),
            frames: Vec::new(),
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    pub fn set_vable_array(&mut self, values: Vec<ResumeValueSource>) {
        self.vable_array = values;
    }

    /// Push a new frame onto the stack.
    pub fn push_frame(&mut self, pc: u64) {
        self.frames.push(FrameInfoBuilder {
            pc,
            slot_map: Vec::new(),
        });
    }

    /// Map a slot in the current frame to a fail_arg index.
    pub fn map_slot(&mut self, slot_idx: usize, fail_arg_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::FailArg(fail_arg_idx));
    }

    /// Set a slot in the current frame to a tagged source.
    pub fn set_slot_source(&mut self, slot_idx: usize, source: FrameSlotSource) {
        let frame = self.frames.last_mut().expect("no frame pushed");
        while frame.slot_map.len() <= slot_idx {
            frame.slot_map.push(FrameSlotSource::Unavailable);
        }
        frame.slot_map[slot_idx] = source;
    }

    /// Set a frame slot to a compile-time constant.
    pub fn set_slot_constant(&mut self, slot_idx: usize, value: i64) {
        self.set_slot_source(slot_idx, FrameSlotSource::Constant(value));
    }

    /// Set a frame slot to reference a virtual object.
    pub fn set_slot_virtual(&mut self, slot_idx: usize, virtual_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Virtual(virtual_idx));
    }

    /// Mark a frame slot as present but uninitialized.
    pub fn set_slot_uninitialized(&mut self, slot_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Uninitialized);
    }

    /// Mark a frame slot as dead/unavailable.
    pub fn set_slot_unavailable(&mut self, slot_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Unavailable);
    }

    /// Add a virtual object description. Returns the index in the virtuals array.
    pub fn add_virtual(&mut self, info: VirtualInfo) -> usize {
        let idx = self.virtuals.len();
        self.virtuals.push(info);
        idx
    }

    /// Convenience: add a virtual object (NEW_WITH_VTABLE).
    pub fn add_virtual_obj(
        &mut self,
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        known_class: Option<i64>,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    ) -> usize {
        self.add_virtual(VirtualInfo::VirtualObj {
            descr,
            type_id,
            descr_index,
            known_class,
            fields,
            fielddescrs,
            descr_size,
        })
    }

    /// Convenience: add a virtual struct (NEW).
    pub fn add_virtual_struct(
        &mut self,
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    ) -> usize {
        self.add_virtual(VirtualInfo::VStruct {
            typedescr,
            type_id,
            descr_index,
            fields,
            fielddescrs,
            descr_size,
        })
    }

    /// Convenience: add a virtual array (NEW_ARRAY).
    pub fn add_virtual_array(&mut self, descr_index: u32, items: Vec<VirtualFieldSource>) -> usize {
        self.add_virtual(VirtualInfo::VArray { descr_index, items })
    }

    /// Convenience: add a virtual array of structs.
    pub fn add_virtual_array_struct(
        &mut self,
        descr_index: u32,
        element_fields: Vec<Vec<(u32, VirtualFieldSource)>>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VArrayStruct {
            descr_index,
            element_fields,
        })
    }

    /// Convenience: add a virtual raw buffer.
    pub fn add_virtual_raw_buffer(
        &mut self,
        size: usize,
        entries: Vec<(usize, usize, VirtualFieldSource)>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VRawBuffer { size, entries })
    }

    /// Add a deferred field write to replay on resume.
    pub fn add_pending_field_write(
        &mut self,
        descr_index: u32,
        target: ResumeValueSource,
        value: ResumeValueSource,
    ) {
        self.pending_fields.push(PendingFieldInfo {
            descr_index,
            target,
            value,
            item_index: None,
        });
    }

    /// Add a deferred array item write to replay on resume.
    pub fn add_pending_arrayitem_write(
        &mut self,
        descr_index: u32,
        target: ResumeValueSource,
        item_index: usize,
        value: ResumeValueSource,
    ) {
        self.pending_fields.push(PendingFieldInfo {
            descr_index,
            target,
            value,
            item_index: Some(item_index),
        });
    }

    /// resume.py: visit_vrawslice(offset) — add virtual raw slice.
    pub fn add_virtual_raw_slice(&mut self, offset: i64, parent: VirtualFieldSource) -> usize {
        self.add_virtual(VirtualInfo::VRawSlice { offset, parent })
    }

    /// resume.py: visit_vstrplain() — add virtual plain string.
    pub fn add_virtual_str_plain(&mut self, chars: Vec<VirtualFieldSource>) -> usize {
        self.add_virtual(VirtualInfo::VStrPlain { chars })
    }

    /// resume.py: visit_vstrconcat() — add virtual string concat.
    pub fn add_virtual_str_concat(
        &mut self,
        left: VirtualFieldSource,
        right: VirtualFieldSource,
    ) -> usize {
        self.add_virtual(VirtualInfo::VStrConcat {
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// resume.py: visit_vstrslice() — add virtual string slice.
    pub fn add_virtual_str_slice(
        &mut self,
        source: VirtualFieldSource,
        start: VirtualFieldSource,
        length: VirtualFieldSource,
    ) -> usize {
        self.add_virtual(VirtualInfo::VStrSlice {
            source: Box::new(source),
            start: Box::new(start),
            length: Box::new(length),
        })
    }

    /// resume.py: visit_vuniplain() — add virtual unicode string.
    pub fn add_virtual_uni_plain(&mut self, chars: Vec<VirtualFieldSource>) -> usize {
        self.add_virtual(VirtualInfo::VUniPlain { chars })
    }

    /// resume.py: visit_vuniconcat() — add virtual unicode concat.
    pub fn add_virtual_uni_concat(
        &mut self,
        left: VirtualFieldSource,
        right: VirtualFieldSource,
    ) -> usize {
        self.add_virtual(VirtualInfo::VUniConcat {
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// resume.py: visit_vunislice() — add virtual unicode slice.
    pub fn add_virtual_uni_slice(
        &mut self,
        source: VirtualFieldSource,
        start: VirtualFieldSource,
        length: VirtualFieldSource,
    ) -> usize {
        self.add_virtual(VirtualInfo::VUniSlice {
            source: Box::new(source),
            start: Box::new(start),
            length: Box::new(length),
        })
    }

    /// Build the final ResumeData.
    pub fn build(self) -> ResumeData {
        ResumeData {
            vable_array: self.vable_array,
            frames: self
                .frames
                .into_iter()
                .map(|f| FrameInfo {
                    pc: f.pc,
                    slot_map: f.slot_map,
                })
                .collect(),
            virtuals: self.virtuals,
            pending_fields: self.pending_fields,
        }
    }
}

impl Default for ResumeDataVirtualAdder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fail-arg compression ─────────────────────────────────────────────

/// Shared resume data storage that deduplicates common snapshot sections
/// across multiple guards in the same trace.
///
/// RPython's `ResumeDataLoopMemo` shares constant pools and frame sections
/// across guards. We use a shared `ResumeEncoder` state so that the same
/// large constant only appears once in the pool.
/// RPython resume.py:142 ResumeDataLoopMemo.
/// Shared constant pool + box numbering cache across all guards in a loop.
///
/// NOTE: RPython's ResumeDataLoopMemo also stores `metainterp_sd` and `cpu`
/// (for box allocation during rebuild). pyre doesn't need these: the BoxEnv
/// trait provides box access. The canonical decoder lives in
/// `majit_ir::resumedata::rebuild_from_numbering`.
/// RPython's nvirtuals/nvholes/nvreused stats are kept for monitoring.
pub struct ResumeDataLoopMemo {
    /// resume.py:147 — shared constant pool.
    /// RPython stores Const objects (with type INT/REF/FLOAT).
    /// We store (value, type) pairs to preserve type information.
    consts: Vec<(i64, majit_ir::Type)>,
    /// resume.py:148 — large integers (outside TAGINT range) → tagged const.
    large_ints: HashMap<i64, i16>,
    /// resume.py:149 — ref pointers → tagged const.
    refs: HashMap<i64, i16>,
    /// resume.py:147 self.consts — i64 constant pool for encode_shared.
    /// Becomes storage.rd_consts (resume.py:467).
    rd_consts_pool: Vec<i64>,
    /// resume.py:155 self.large_ints — dedup index for rd_consts_pool.
    const_indices: HashMap<i64, usize>,
    /// resume.py:150-151 — cached box/virtual numbering.
    pub cached_boxes: HashMap<u32, i32>,
    pub cached_virtuals: HashMap<u32, i32>,
    /// resume.py:153-155 — statistics.
    pub nvirtuals: usize,
    pub nvholes: usize,
    pub nvreused: usize,
}

impl ResumeDataLoopMemo {
    pub fn new() -> Self {
        ResumeDataLoopMemo {
            consts: Vec::new(),
            large_ints: HashMap::new(),
            refs: HashMap::new(),
            rd_consts_pool: Vec::new(),
            const_indices: HashMap::new(),
            cached_boxes: HashMap::new(),
            cached_virtuals: HashMap::new(),
            nvirtuals: 0,
            nvholes: 0,
            nvreused: 0,
        }
    }

    /// resume.py:157-183 getconst(const) — tag a constant value.
    /// Unified entry point matching RPython's getconst(const) which
    /// dispatches on const.type (INT, REF, FLOAT).
    pub fn getconst(&mut self, val: i64, tp: majit_ir::Type) -> i16 {
        match tp {
            majit_ir::Type::Int => self.getconst_int(val),
            majit_ir::Type::Ref => self.getconst_ref(val),
            majit_ir::Type::Float => self.getconst_float(val),
            majit_ir::Type::Void => self.newconst(val, tp),
        }
    }

    /// resume.py:158-172 getconst for INT type.
    pub fn getconst_int(&mut self, val: i64) -> i16 {
        // Try inline TAGINT (-8191..8190 in RPython's i16 range).
        let shifted = val >> 13;
        if shifted == 0 || shifted == -1 {
            return ((val << 2) | TAGINT as i64) as i16;
        }
        // Large int: check cache.
        if let Some(&tagged) = self.large_ints.get(&val) {
            return tagged;
        }
        let tagged = self.newconst(val, majit_ir::Type::Int);
        self.large_ints.insert(val, tagged);
        tagged
    }

    /// resume.py:173-182 getconst for REF type.
    pub fn getconst_ref(&mut self, val: i64) -> i16 {
        if val == 0 {
            return NULLREF;
        }
        if let Some(&tagged) = self.refs.get(&val) {
            return tagged;
        }
        let tagged = self.newconst(val, majit_ir::Type::Ref);
        self.refs.insert(val, tagged);
        tagged
    }

    /// resume.py:183 getconst fallback for FLOAT type.
    pub fn getconst_float(&mut self, val: i64) -> i16 {
        // FLOAT constants always go to the pool (no inline encoding).
        // RPython: return self._newconst(const)
        self.newconst(val, majit_ir::Type::Float)
    }

    /// resume.py:185 _newconst — add to consts pool, return TAGCONST-tagged.
    fn newconst(&mut self, val: i64, tp: majit_ir::Type) -> i16 {
        let index = self.consts.len() as i32 + TAG_CONST_OFFSET;
        self.consts.push((val, tp));
        ((index << 2) | TAGCONST as i32) as i16
    }

    /// resume.py:264 assign_number_to_box — returns a negative number.
    /// resume.py:264-273 assign_number_to_box(box, boxes).
    ///
    /// RPython version mutates `boxes` list:
    /// - cached: `boxes[-num - 1] = box`
    /// - new: `boxes.append(box); num = -len(boxes)`
    pub fn assign_number_to_box(&mut self, box_id: u32, boxes: &mut Vec<u32>) -> i32 {
        if let Some(&num) = self.cached_boxes.get(&box_id) {
            // resume.py:268: boxes[-num - 1] = box
            let idx = (-num - 1) as usize;
            if idx < boxes.len() {
                boxes[idx] = box_id;
            }
            return num;
        }
        // resume.py:270-271: boxes.append(box); num = -len(boxes)
        boxes.push(box_id);
        let num = -(boxes.len() as i32);
        self.cached_boxes.insert(box_id, num);
        num
    }

    /// resume.py:264-273 variant for `_number_virtuals`: boxes is `Vec<Option<u32>>`.
    /// RPython's `new_liveboxes = [None] * memo.num_cached_boxes()`.
    pub fn assign_number_to_box_opt(&mut self, box_id: u32, boxes: &mut Vec<Option<u32>>) -> i32 {
        if let Some(&num) = self.cached_boxes.get(&box_id) {
            let idx = (-num - 1) as usize;
            if idx < boxes.len() {
                boxes[idx] = Some(box_id);
            }
            return num;
        }
        boxes.push(Some(box_id));
        let num = -(boxes.len() as i32);
        self.cached_boxes.insert(box_id, num);
        num
    }

    /// resume.py:278 assign_number_to_virtual — returns a negative number.
    pub fn assign_number_to_virtual(&mut self, box_id: u32) -> i32 {
        if let Some(&num) = self.cached_virtuals.get(&box_id) {
            return num;
        }
        let num = -(self.cached_virtuals.len() as i32) - 1;
        self.cached_virtuals.insert(box_id, num);
        num
    }

    /// resume.py:286 clear_box_virtual_numbers.
    pub fn clear_box_virtual_numbers(&mut self) {
        self.cached_boxes.clear();
        self.cached_virtuals.clear();
    }

    /// Access the shared constant pool (value, type) pairs.
    pub fn consts(&self) -> &[(i64, majit_ir::Type)] {
        &self.consts
    }

    /// Access constant values only (for decode compatibility).
    pub fn const_values(&self) -> Vec<i64> {
        self.consts.iter().map(|&(v, _)| v).collect()
    }

    /// Take ownership of the shared constant pool.
    pub fn take_consts(&mut self) -> Vec<(i64, majit_ir::Type)> {
        std::mem::take(&mut self.consts)
    }

    /// resume.py:370-374 register_box — add a non-const, non-seen box to
    /// new_liveboxes with UNASSIGNED or UNASSIGNEDVIRTUAL tag.
    fn register_box_in_new_liveboxes(
        &self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &mut LiveboxMap,
        new_liveboxes_order: &mut Vec<u32>,
    ) {
        if opref.is_none()
            || env.is_const(opref)
            || liveboxes_from_env.contains_key(opref.0)
            || new_liveboxes.contains_key(opref.0)
        {
            return;
        }
        // resume.py:212-216: check if field is virtual
        let is_virtual = match env.get_type(opref) {
            majit_ir::Type::Ref => env.is_virtual_ref(opref),
            majit_ir::Type::Int => env.is_virtual_raw(opref),
            _ => false,
        };
        let t = if is_virtual {
            UNASSIGNEDVIRTUAL
        } else {
            UNASSIGNED
        };
        new_liveboxes.insert(opref.0, t);
        new_liveboxes_order.push(opref.0);
    }

    /// resume.py:560-568 _gettagged — resolve an OpRef to its tagged number.
    /// Looks up in liveboxes_from_env first, then new_liveboxes, then constant.
    fn _gettagged(
        &mut self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &LiveboxMap,
    ) -> i16 {
        if opref.is_none() {
            return UNINITIALIZED_TAG;
        }
        // resume.py:563-564: isinstance(box, Const) → getconst
        if env.is_const(opref) {
            let (val, tp) = env.get_const(opref);
            return self.getconst(val, tp);
        }
        // resume.py:566-567: liveboxes_from_env → existing tag
        if let Some(tagged) = liveboxes_from_env.get(opref.0) {
            return tagged;
        }
        if let Some(tagged) = new_liveboxes.get(opref.0) {
            // Resolve UNASSIGNED to real cached number
            if tagged_eq(tagged, UNASSIGNED) {
                if let Some(&num) = self.cached_boxes.get(&opref.0) {
                    return tag(num, TAGBOX).unwrap_or(UNASSIGNED);
                }
            }
            if tagged_eq(tagged, UNASSIGNEDVIRTUAL) {
                if let Some(&num) = self.cached_virtuals.get(&opref.0) {
                    return tag(num, TAGVIRTUAL).unwrap_or(UNASSIGNEDVIRTUAL);
                }
            }
            return tagged;
        }
        UNASSIGNED
    }

    /// resume.py:192-226 _number_boxes — tag each box in a snapshot section.
    ///
    /// Exact port of RPython's `_number_boxes(self, iter, iterator, numb_state)`.
    ///
    /// `env` provides box access matching RPython's box operations:
    /// - `get_box_replacement(opref)` → forwarded OpRef (resume.py:202)
    /// - `is_const(opref)` → isinstance(box, Const) (resume.py:204)
    /// - `get_const(opref)` → (value, type) for constants
    /// - `get_type(opref)` → box.type ('i', 'r', 'f') (resume.py:211,214)
    /// - `is_virtual_ref(opref)` → getptrinfo(box).is_virtual() (resume.py:212-213)
    /// - `is_virtual_raw(opref)` → getrawptrinfo(box).is_virtual() (resume.py:215-216)
    /// See `majit_ir::resumedata::ResumeDataLoopMemo::_number_boxes` for docs.
    pub fn _number_boxes(
        &mut self,
        boxes: &[majit_ir::OpRef],
        numb_state: &mut NumberingState,
        env: &dyn BoxEnv,
    ) -> Result<(), TagOverflow> {
        for &raw_opref in boxes {
            if raw_opref.is_none() {
                numb_state.append_short(NULLREF);
                continue;
            }
            let opref = env.get_box_replacement(raw_opref);
            if opref.is_none() {
                numb_state.append_short(NULLREF);
                continue;
            }
            // resume.py:204: isinstance(box, Const) → getconst
            if env.is_const(opref) {
                let (val, tp) = env.get_const(opref);
                let tagged = self.getconst(val, tp);
                numb_state.append_short(tagged);
                continue;
            }
            // resume.py:206-208: liveboxes
            if let Some(tagged) = numb_state.liveboxes.get(opref.0) {
                numb_state.append_short(tagged);
                continue;
            }
            let box_type = env.get_type(opref);
            let is_virtual = match box_type {
                majit_ir::Type::Ref => env.is_virtual_ref(opref),
                majit_ir::Type::Int => env.is_virtual_raw(opref),
                _ => false,
            };
            let tagged = if is_virtual {
                let t = tag(numb_state.num_virtuals, TAGVIRTUAL)?;
                numb_state.num_virtuals += 1;
                t
            } else {
                numb_state.livebox_types.insert(opref.0, box_type);
                let t = tag(numb_state.num_boxes, TAGBOX)?;
                numb_state.num_boxes += 1;
                t
            };
            numb_state.liveboxes.insert(opref.0, tagged);
            numb_state.append_short(tagged);
        }
        Ok(())
    }

    /// resume.py:228-256 number() — serialize a guard's full snapshot.
    ///
    /// Output format (in NumberingState):
    /// ```text
    /// [0]  size (patched later)
    /// [1]  number of failargs (patched later)
    /// [2]  vable_array_length  (0 if no virtualizable)
    ///      [tagged boxes for vable_array]
    /// [n]  vref_array_length   (0 if no virtualrefs)
    ///      [tagged boxes for vref_array]
    /// [m]  frame0_pc frame0_slots...
    /// [m+] frame1_pc frame1_slots...
    /// ...
    /// ```
    ///
    /// `frames` is a list of (pc, fail_args_slice) for each frame.
    /// In pyre (single frame), this is typically one frame.
    /// resume.py:228-256 number() — serialize a guard's full snapshot.
    ///
    /// Exact port of RPython's `number(self, position, trace, ...)`.
    ///
    /// `snapshot` describes the guard's state:
    /// - `vable_array`: virtualizable field boxes
    /// - `vref_array`: virtualref pairs
    /// - `framestack`: list of (jitcode_index, pc, boxes) per frame
    ///
    /// `env` implements BoxEnv for box operations.
    ///
    /// Returns `Err(TagOverflow)` if any box index exceeds the tag range.
    /// RPython: raises TagOverflow → caller does compile.giveup().
    ///
    /// NOTE: Slot 1 (number of failargs) is left as 0 here.
    /// RPython patches it later in ResumeDataVirtualAdder.finish()
    /// (resume.py:433). Callers must call
    /// `numb_state.writer.patch(1, num_liveboxes)` after finish().
    pub fn number(
        &mut self,
        snapshot: &Snapshot,
        env: &dyn BoxEnv,
    ) -> Result<NumberingState, TagOverflow> {
        let size_hint = snapshot.estimated_size();
        let mut numb_state = NumberingState::new(size_hint);

        // resume.py:231-232: patch later
        numb_state.append_int(0); // slot 0: size of resume section
        numb_state.append_int(0); // slot 1: number of failargs (patched by finish())

        // resume.py:234-241: virtualizable array
        numb_state.append_int(snapshot.vable_array.len() as i32);
        self._number_boxes(&snapshot.vable_array, &mut numb_state, env)?;

        // resume.py:243-247: virtualref array
        let vref_len = snapshot.vref_array.len();
        debug_assert!(vref_len & 1 == 0, "vref_array length must be even");
        numb_state.append_int((vref_len >> 1) as i32);
        self._number_boxes(&snapshot.vref_array, &mut numb_state, env)?;

        // resume.py:249-253: frame chain.
        // Per-frame: jitcode_index, pc, [tagged_values...].
        // RPython uses jitcode.get_live_vars_info(pc) at decode time
        // to know how many tagged values each frame has.
        for frame in &snapshot.framestack {
            numb_state.append_int(frame.jitcode_index);
            numb_state.append_int(frame.pc);
            self._number_boxes(&frame.boxes, &mut numb_state, env)?;
        }

        // resume.py:254: patch total size
        numb_state.patch_current_size(0);

        Ok(numb_state)
    }

    /// resume.py:389-452 ResumeDataVirtualAdder.finish() — exact port.
    ///
    /// `numb_state`: output of `number()`
    /// `env`: BoxEnv for resolving box properties (constants, types).
    ///   Virtual fields are discovered via `env.get_virtual_fields()`,
    ///   matching RPython's `visitor_walk_recursive` callback pattern.
    /// `pending_setfields`: resume.py:428-442 register_box + visitor_walk_recursive,
    ///   resume.py:520-558 _add_pending_fields tagging.
    ///   target_tagged/value_tagged are filled in-place.
    /// `optimizer_knowledge`: bridgeopt.py:63 serialize_optimizer_knowledge.
    ///   Heap field triples and known-class info for bridge compilation.
    ///
    /// Returns `(rd_numb, rd_consts, rd_virtuals, liveboxes, livebox_types)`.
    pub fn finish(
        &mut self,
        mut numb_state: NumberingState,
        env: &dyn majit_ir::BoxEnv,
        pending_setfields: &mut [majit_ir::GuardPendingFieldEntry],
        optimizer_knowledge: Option<&OptimizerKnowledgeForResume>,
    ) -> (
        Vec<u8>,
        Vec<(i64, majit_ir::Type)>,
        Vec<majit_ir::RdVirtualInfo>,
        Vec<majit_ir::OpRef>,
        std::collections::HashMap<u32, majit_ir::Type>,
    ) {
        let num_env_virtuals = numb_state.num_virtuals;

        // resume.py:410-426: split liveboxes_from_env into TAGBOX/TAGVIRTUAL
        let mut liveboxes: Vec<Option<majit_ir::OpRef>> = vec![None; numb_state.num_boxes as usize];

        // resume.py:413: self.vfieldboxes collected by virtual walk
        // resume.py:408: self.liveboxes — newly discovered boxes from field walk
        let mut new_liveboxes = LiveboxMap::new();
        // Insertion-order tracking for _number_virtuals (RPython dict is ordered).
        let mut new_liveboxes_order: Vec<u32> = Vec::new();

        // resume.py:414-426: iterate liveboxes_from_env, discover virtual fields.
        // RPython iterates in insertion order; we sort by tag for determinism.
        let mut sorted_liveboxes: Vec<(u32, i16)> = numb_state.liveboxes.iter().collect();
        sorted_liveboxes.sort_by_key(|&(_, tagged)| {
            let (val, tagbits) = untag(tagged);
            (tagbits, val)
        });

        // Collect virtual fields discovered via env.get_virtual_fields()
        // (resume.py:419-426 visitor_walk_recursive pattern).
        let mut virtual_fields: HashMap<u32, majit_ir::VirtualFieldsInfo> = HashMap::new();

        // resume.py:419-426: visitor_walk_recursive — worklist for nested virtuals.
        let mut virtual_worklist: Vec<u32> = Vec::new();

        for &(opref_id, tagged) in &sorted_liveboxes {
            let (i, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                if (i as usize) < liveboxes.len() {
                    liveboxes[i as usize] = Some(majit_ir::OpRef(opref_id));
                }
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                virtual_worklist.push(opref_id);
            }
        }

        // Worklist-based recursive virtual discovery (RPython visitor_walk_recursive).
        // Process each virtual: register its field boxes, and if any field is
        // itself a virtual, add it to the worklist for later processing.
        let mut worklist_idx = 0;
        while worklist_idx < virtual_worklist.len() {
            let opref_id = virtual_worklist[worklist_idx];
            worklist_idx += 1;

            if virtual_fields.contains_key(&opref_id) {
                continue; // already_seen_virtual
            }
            if let Some(vf) = env.get_virtual_fields(majit_ir::OpRef(opref_id)) {
                // resume.py:362-368: register_virtual_fields
                for &field_opref in &vf.field_oprefs {
                    // resume.py:370-374: register_box
                    self.register_box_in_new_liveboxes(
                        field_opref,
                        env,
                        &numb_state.liveboxes,
                        &mut new_liveboxes,
                        &mut new_liveboxes_order,
                    );
                    // If field is a virtual, add to worklist for recursive processing
                    let resolved = env.get_box_replacement(field_opref);
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved.0)
                        && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                    {
                        // Assign TAGVIRTUAL to nested virtual
                        if numb_state.liveboxes.get(resolved.0).is_none()
                            && new_liveboxes.get(resolved.0).is_none()
                        {
                            new_liveboxes.insert(resolved.0, UNASSIGNEDVIRTUAL);
                            new_liveboxes_order.push(resolved.0);
                        }
                        virtual_worklist.push(resolved.0);
                    }
                }
                virtual_fields.insert(opref_id, vf);
            }
        }

        // resume.py:428-442: process pending_setfields — register_box on
        // target and value, then visitor_walk_recursive on virtual fieldbox.
        for pf in pending_setfields.iter() {
            let box_opref = env.get_box_replacement(pf.target);
            let fieldbox = env.get_box_replacement(pf.value);
            // resume.py:438-439: self.register_box(box); self.register_box(fieldbox)
            self.register_box_in_new_liveboxes(
                box_opref,
                env,
                &numb_state.liveboxes,
                &mut new_liveboxes,
                &mut new_liveboxes_order,
            );
            self.register_box_in_new_liveboxes(
                fieldbox,
                env,
                &numb_state.liveboxes,
                &mut new_liveboxes,
                &mut new_liveboxes_order,
            );
            // resume.py:440-442: info.visitor_walk_recursive(fieldbox, self)
            if let Some(vf) = env.get_virtual_fields(fieldbox) {
                for &field_opref in &vf.field_oprefs {
                    self.register_box_in_new_liveboxes(
                        field_opref,
                        env,
                        &numb_state.liveboxes,
                        &mut new_liveboxes,
                        &mut new_liveboxes_order,
                    );
                    // Nested virtual discovery (same as main worklist above)
                    let resolved = env.get_box_replacement(field_opref);
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved.0)
                        && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                    {
                        if numb_state.liveboxes.get(resolved.0).is_none()
                            && new_liveboxes.get(resolved.0).is_none()
                        {
                            new_liveboxes.insert(resolved.0, UNASSIGNEDVIRTUAL);
                            new_liveboxes_order.push(resolved.0);
                        }
                        virtual_worklist.push(resolved.0);
                    }
                }
                virtual_fields.insert(fieldbox.0, vf);
            }
        }

        // resume.py:440-442 parity: drain worklist for nested virtuals
        // discovered from pending_setfields. RPython's visitor_walk_recursive
        // recursively processes all levels; our worklist pattern resumes here.
        while worklist_idx < virtual_worklist.len() {
            let opref_id = virtual_worklist[worklist_idx];
            worklist_idx += 1;
            if virtual_fields.contains_key(&opref_id) {
                continue;
            }
            if let Some(vf) = env.get_virtual_fields(majit_ir::OpRef(opref_id)) {
                for &field_opref in &vf.field_oprefs {
                    self.register_box_in_new_liveboxes(
                        field_opref,
                        env,
                        &numb_state.liveboxes,
                        &mut new_liveboxes,
                        &mut new_liveboxes_order,
                    );
                    let resolved = env.get_box_replacement(field_opref);
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved.0)
                        && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                    {
                        if numb_state.liveboxes.get(resolved.0).is_none()
                            && new_liveboxes.get(resolved.0).is_none()
                        {
                            new_liveboxes.insert(resolved.0, UNASSIGNEDVIRTUAL);
                            new_liveboxes_order.push(resolved.0);
                        }
                        virtual_worklist.push(resolved.0);
                    }
                }
                virtual_fields.insert(opref_id, vf);
            }
        }

        // resume.py:454-509: _number_virtuals
        let mut new_boxes_list: Vec<Option<u32>> = vec![None; self.cached_boxes.len()];
        let mut count = 0;
        // Iterate in insertion order (RPython dict iteration = insertion order).
        let keys: Vec<(u32, i16)> = new_liveboxes_order
            .iter()
            .filter_map(|&k| new_liveboxes.get(k).map(|v| (k, v)))
            .collect();
        for (opref_id, tagged) in keys {
            let (_, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                // resume.py:472-473: index = assign_number_to_box; liveboxes[box] = tag(index, TAGBOX)
                let index = self.assign_number_to_box_opt(opref_id, &mut new_boxes_list);
                if let Ok(t) = tag(index, TAGBOX) {
                    new_liveboxes.insert(opref_id, t);
                }
                count += 1;
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                if tagged_eq(tagged, UNASSIGNEDVIRTUAL) {
                    // resume.py:479-480: index = assign_number_to_virtual; liveboxes[box] = tag(index, TAGVIRTUAL)
                    let index = self.assign_number_to_virtual(opref_id);
                    if let Ok(t) = tag(index, TAGVIRTUAL) {
                        new_liveboxes.insert(opref_id, t);
                    }
                }
            }
        }
        // resume.py:483-484: new_liveboxes.reverse(); liveboxes.extend(new_liveboxes)
        new_boxes_list.reverse();
        for box_id in &new_boxes_list {
            liveboxes.push(box_id.map(majit_ir::OpRef));
        }
        let nholes = new_boxes_list.len() - count;

        // resume.py:488-506: create rd_virtuals
        // resume.py:500-501: make_virtual_info(info, fieldnums) via BoxEnv dispatch
        let mut rd_virtuals: Vec<majit_ir::RdVirtualInfo> = Vec::new();
        if !virtual_fields.is_empty() {
            let length = num_env_virtuals as usize + self.cached_virtuals.len();
            rd_virtuals.resize(length, majit_ir::RdVirtualInfo::Empty);
            self.nvirtuals += length;
            self.nvholes += length - virtual_fields.len();

            for (&opref_id, vf) in &virtual_fields {
                // resume.py:496: num, _ = untag(self.liveboxes[virtualbox])
                // Check both numb_state.liveboxes (env virtuals) and
                // new_liveboxes (nested virtuals discovered via worklist).
                let tagged = numb_state
                    .liveboxes
                    .get(opref_id)
                    .or_else(|| new_liveboxes.get(opref_id))
                    .unwrap_or(UNASSIGNEDVIRTUAL);
                let (num, _) = untag(tagged);
                // RPython uses Python negative indexing: virtuals[-1] = virtuals[len-1].
                // Negative nums come from assign_number_to_virtual for nested virtuals.
                let num_idx = if num >= 0 {
                    num as usize
                } else {
                    (rd_virtuals.len() as i32 + num) as usize
                };
                if num_idx < rd_virtuals.len() {
                    // resume.py:500: fieldnums = [self._gettagged(box) for box in fieldboxes]
                    let fieldnums: Vec<i16> = vf
                        .field_oprefs
                        .iter()
                        .map(|&opref| {
                            // resume.py:560-568: _gettagged
                            if opref.is_none() {
                                return UNINITIALIZED_TAG;
                            }
                            // resume.py:563-564: isinstance(box, Const)
                            if env.is_const(opref) {
                                let (val, tp) = env.get_const(opref);
                                return self.getconst(val, tp);
                            }
                            // resume.py:566-567: liveboxes_from_env
                            if let Some(t) = numb_state.liveboxes.get(opref.0) {
                                return t;
                            }
                            // Then check new_liveboxes
                            if let Some(t) = new_liveboxes.get(opref.0) {
                                // If still UNASSIGNED, get the real tag from cached_boxes
                                if tagged_eq(t, UNASSIGNED) {
                                    if let Some(&num) = self.cached_boxes.get(&opref.0) {
                                        return tag(num, TAGBOX).unwrap_or(UNASSIGNED);
                                    }
                                }
                                if tagged_eq(t, UNASSIGNEDVIRTUAL) {
                                    if let Some(&num) = self.cached_virtuals.get(&opref.0) {
                                        return tag(num, TAGVIRTUAL).unwrap_or(UNASSIGNEDVIRTUAL);
                                    }
                                }
                                return t;
                            }
                            UNASSIGNED
                        })
                        .collect();
                    // resume.py:501: vinfo = self.make_virtual_info(info, fieldnums)
                    if let Some(rd_virt) =
                        env.make_rd_virtual_info(majit_ir::OpRef(opref_id), fieldnums)
                    {
                        rd_virtuals[num_idx] = rd_virt;
                    }
                }
            }
        }

        // resume.py:508-509: _invalidation_needed heuristic
        if nholes > 0 && liveboxes.len() > majit_ir::FAILARGS_LIMIT / 2 {
            if nholes > liveboxes.len() / 3 {
                self.clear_box_virtual_numbers();
            }
        }

        // resume.py:445, 520-558: _add_pending_fields(pending_setfields)
        // Tag target and value in-place on GuardPendingFieldEntry.
        for pf in pending_setfields.iter_mut() {
            let target = env.get_box_replacement(pf.target);
            let value = env.get_box_replacement(pf.value);
            pf.target_tagged = self._gettagged(target, env, &numb_state.liveboxes, &new_liveboxes);
            pf.value_tagged = self._gettagged(value, env, &numb_state.liveboxes, &new_liveboxes);
        }

        // resume.py:447: numb_state.patch(1, len(liveboxes))
        numb_state.writer.patch(1, liveboxes.len() as i32);

        // resume.py:449: _add_optimizer_sections(numb_state, ...)
        // bridgeopt.py:63-122: serialize_optimizer_knowledge
        if let Some(knowledge) = optimizer_knowledge {
            // Known classes bitfield (bridgeopt.py:74-88)
            let mut bitfield: i32 = 0;
            let mut shifts = 0;
            for livebox in &liveboxes {
                if let Some(opref) = livebox {
                    if env.get_type(*opref) != majit_ir::Type::Ref {
                        continue;
                    }
                    bitfield <<= 1;
                    if knowledge.known_classes.contains(&opref.0) {
                        bitfield |= 1;
                    }
                    shifts += 1;
                    if shifts == 6 {
                        numb_state.append_int(bitfield);
                        bitfield = 0;
                        shifts = 0;
                    }
                }
            }
            if shifts > 0 {
                numb_state.append_int(bitfield << (6 - shifts));
            }

            // Heap field triples (bridgeopt.py:90-108)
            numb_state.append_int(knowledge.heap_fields.len() as i32);
            for &(obj, descr_idx, val) in &knowledge.heap_fields {
                numb_state.writer.append_short(self._gettagged(
                    obj,
                    env,
                    &numb_state.liveboxes,
                    &new_liveboxes,
                ) as i32);
                numb_state.append_int(descr_idx as i32);
                numb_state.writer.append_short(self._gettagged(
                    val,
                    env,
                    &numb_state.liveboxes,
                    &new_liveboxes,
                ) as i32);
            }
            numb_state.append_int(0); // array items (empty)

            // Loop-invariant results (bridgeopt.py:113-122)
            numb_state.append_int(knowledge.loopinvariant_results.len() as i32);
            for &(const_ptr, result) in &knowledge.loopinvariant_results {
                numb_state
                    .writer
                    .append_short(self.getconst_int(const_ptr) as i32);
                numb_state.writer.append_short(self._gettagged(
                    result,
                    env,
                    &numb_state.liveboxes,
                    &new_liveboxes,
                ) as i32);
            }
        }

        // resume.py:450-451: storage.rd_numb, storage.rd_consts
        let rd_numb = numb_state.create_numbering();
        let rd_consts = self.consts.clone();

        // Resolve each livebox through the forwarding chain so the backend
        // sees the final concrete OpRef (not an optimizer-internal alias).
        let ordered_liveboxes: Vec<majit_ir::OpRef> = liveboxes
            .into_iter()
            .map(|opt| {
                opt.map(|opref| env.get_box_replacement(opref))
                    .unwrap_or(majit_ir::OpRef::NONE)
            })
            .collect();

        // Merge livebox_types: numbering-time types + types for boxes
        // discovered during virtual field walking.
        let mut all_livebox_types = numb_state.livebox_types;
        for &opref in &ordered_liveboxes {
            if !opref.is_none() && !all_livebox_types.contains_key(&opref.0) {
                all_livebox_types.insert(opref.0, env.get_type(opref));
            }
        }
        (
            rd_numb,
            rd_consts,
            rd_virtuals,
            ordered_liveboxes,
            all_livebox_types,
        )
    }

    /// resume.py:452-468 finish (on ResumeDataVirtualAdder) — encode with shared pool.
    pub fn encode_shared(&mut self, rd: &ResumeData) -> EncodedResumeData {
        let mut rd_numb = Vec::new();
        // resume.py:138 compact TAGBOX numbering state.
        let mut liveboxes: Vec<usize> = Vec::new();
        let mut box_map: HashMap<usize, usize> = HashMap::new();

        // resume.py:234-235: reserve slots
        rd_numb.push(0); // [0] = items_resume_section
        rd_numb.push(0); // [1] = count
        rd_numb.push(encode_len(rd.vable_array.len()));
        for source in &rd.vable_array {
            let tagged = encode_tagged_source(
                source,
                &mut self.rd_consts_pool,
                &mut self.const_indices,
                &mut liveboxes,
                &mut box_map,
            );
            rd_numb.push(tagged);
        }
        rd_numb.push(encode_len(0));
        rd_numb.push(encode_len(rd.frames.len()));

        // resume.py:249-258: per-frame via _number_boxes
        for frame in &rd.frames {
            rd_numb.push(encode_u64(frame.pc));
            rd_numb.push(encode_len(frame.slot_map.len()));
            for source in &frame.slot_map {
                let tagged = encode_tagged_source(
                    source,
                    &mut self.rd_consts_pool,
                    &mut self.const_indices,
                    &mut liveboxes,
                    &mut box_map,
                );
                rd_numb.push(tagged);
            }
        }

        let rd_virtuals = rd.virtuals.clone();

        // resume.py:412-418: register virtual field boxes.
        for vinfo in &rd_virtuals {
            for source in vinfo.field_sources() {
                if let ResumeValueSource::FailArg(index) = source {
                    box_map.entry(*index).or_insert_with(|| {
                        let n = liveboxes.len();
                        liveboxes.push(*index);
                        n
                    });
                }
            }
        }

        // resume.py:420-430: walk pending fields — register + encode.
        let rd_pendingfields: Vec<_> = rd
            .pending_fields
            .iter()
            .map(|pending| EncodedPendingFieldWrite {
                descr_index: pending.descr_index,
                target: encode_tagged_source(
                    &pending.target,
                    &mut self.rd_consts_pool,
                    &mut self.const_indices,
                    &mut liveboxes,
                    &mut box_map,
                ),
                value: encode_tagged_source(
                    &pending.value,
                    &mut self.rd_consts_pool,
                    &mut self.const_indices,
                    &mut liveboxes,
                    &mut box_map,
                ),
                item_index: pending.item_index,
            })
            .collect();

        // resume.py:260 patch_current_size, resume.py:464 patch count
        rd_numb[0] = encode_len(rd_numb.len());
        rd_numb[1] = encode_len(liveboxes.len());

        EncodedResumeData {
            rd_numb,
            rd_consts: self.rd_consts_pool.clone(),
            rd_pendingfields,
            rd_virtuals,
            liveboxes,
        }
    }

    /// Number of entries in the shared constant pool.
    pub fn num_shared_consts(&self) -> usize {
        self.consts.len()
    }
}

impl Default for ResumeDataLoopMemo {
    fn default() -> Self {
        Self::new()
    }
}

/// resume.py: AbstractResumeDataReader — reads resume data to
/// reconstruct interpreter state after a guard failure.
///
/// Two concrete implementations in RPython:
/// - ResumeDataBoxReader: creates boxes (for blackhole interpreter)
/// - ResumeDataDirectReader: reads values directly (for fast path)
pub struct ResumeDataReader<'a> {
    /// The resume data to read from.
    resume_data: &'a ResumeData,
    /// Fail argument values from the guard failure.
    fail_values: &'a [i64],
    /// Materialized virtuals (lazily populated).
    virtuals: Vec<Option<i64>>,
}

impl<'a> ResumeDataReader<'a> {
    /// resume.py: AbstractResumeDataReader.__init__
    pub fn new(resume_data: &'a ResumeData, fail_values: &'a [i64]) -> Self {
        let num_virtuals = resume_data.virtuals.len();
        ResumeDataReader {
            resume_data,
            fail_values,
            virtuals: vec![None; num_virtuals],
        }
    }

    /// resume.py: _decode_box — decode a tagged value reference.
    pub fn decode_frame_slot(&self, source: &FrameSlotSource) -> i64 {
        self.decode_value(source)
    }

    /// Decode a ResumeValueSource to a concrete value.
    pub fn decode_value(&self, source: &ResumeValueSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(idx) => self.fail_values.get(*idx).copied().unwrap_or(0),
            ResumeValueSource::Constant(val) => *val,
            ResumeValueSource::Virtual(vidx) => {
                self.virtuals.get(*vidx).copied().flatten().unwrap_or(0)
            }
            ResumeValueSource::Uninitialized | ResumeValueSource::Unavailable => 0,
        }
    }

    /// resume.py: consume_boxes — read all frame slots for one frame.
    pub fn read_frame_slots(&self, frame_idx: usize) -> Vec<i64> {
        if frame_idx >= self.resume_data.frames.len() {
            return vec![];
        }
        let frame = &self.resume_data.frames[frame_idx];
        frame
            .slot_map
            .iter()
            .map(|source| self.decode_frame_slot(source))
            .collect()
    }

    /// Number of frames in the resume data.
    pub fn num_frames(&self) -> usize {
        self.resume_data.frames.len()
    }

    /// PC for a given frame.
    pub fn frame_pc(&self, frame_idx: usize) -> u64 {
        self.resume_data
            .frames
            .get(frame_idx)
            .map(|f| f.pc)
            .unwrap_or(0)
    }
}

/// resume.py:576-728 VirtualInfo parity.
/// Describes a virtual object's fields for materialization.
/// RPython uses a class hierarchy (VirtualInfo, VStructInfo, VArrayInfoClear, etc.).
/// We use a single struct with tagged field values.
#[derive(Debug, Clone, Default)]
pub struct VirtualFieldValues {
    /// Descriptor (type/class) for the virtual object.
    pub descr: Option<majit_ir::DescrRef>,
    /// Known class pointer (ob_type for NewWithVtable).
    pub known_class: Option<i64>,
    /// Tagged field values (i16 tags referencing consts/boxes/other virtuals).
    pub fieldnums: Vec<i16>,
}

/// resume.py:554-557 — tagged pending field entry.
/// RPython stores (lldescr, num, fieldnum, itemindex) where num and fieldnum
/// are tagged references into the numbering system.
#[derive(Debug, Clone)]
pub struct TaggedPendingField {
    pub descr_index: u32,
    pub item_index: i32,
    /// Tagged reference to target box (from _gettagged).
    pub num: i16,
    /// Tagged reference to value box (from _gettagged).
    pub fieldnum: i16,
}

/// bridgeopt.py:63 — optimizer knowledge for resume data encoding.
/// Passed into finish() for _add_optimizer_sections.
pub struct OptimizerKnowledgeForResume {
    /// OpRef IDs with known class.
    pub known_classes: std::collections::HashSet<u32>,
    /// (obj_opref, descr_index, val_opref) heap field triples.
    pub heap_fields: Vec<(majit_ir::OpRef, u32, majit_ir::OpRef)>,
    /// (const_func_ptr, result_opref) loop-invariant call results.
    pub loopinvariant_results: Vec<(i64, majit_ir::OpRef)>,
}

// VirtualFieldInfo removed: replaced by majit_ir::VirtualFieldsInfo.
// finish() now discovers virtual fields via env.get_virtual_fields().

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};

    #[test]
    fn test_simple_resume_data() {
        let rd = ResumeData::simple(42, 3);
        let fail_values = vec![10, 20, 30];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].pc, 42);
        assert_eq!(
            frames[0].values,
            vec![
                ReconstructedValue::Value(10),
                ReconstructedValue::Value(20),
                ReconstructedValue::Value(30),
            ]
        );
    }

    #[test]
    fn test_resume_data_with_gaps() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 100,
                slot_map: vec![
                    FrameSlotSource::FailArg(2),
                    FrameSlotSource::Unavailable,
                    FrameSlotSource::FailArg(0),
                ],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };
        let fail_values = vec![10, 20, 30];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(
            frames[0].values,
            vec![
                ReconstructedValue::Value(30),
                ReconstructedValue::Unavailable,
                ReconstructedValue::Value(10),
            ]
        );
        assert_eq!(frames[0].lossy_values(), vec![30, 0, 10]);
    }

    #[test]
    fn test_multi_frame_resume() {
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    pc: 10,
                    slot_map: vec![FrameSlotSource::FailArg(0), FrameSlotSource::FailArg(1)],
                },
                FrameInfo {
                    pc: 20,
                    slot_map: vec![FrameSlotSource::FailArg(2), FrameSlotSource::FailArg(3)],
                },
            ],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };
        let fail_values = vec![1, 2, 3, 4];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].pc, 10);
        assert_eq!(
            frames[0].values,
            vec![ReconstructedValue::Value(1), ReconstructedValue::Value(2)]
        );
        assert_eq!(frames[1].pc, 20);
        assert_eq!(
            frames[1].values,
            vec![ReconstructedValue::Value(3), ReconstructedValue::Value(4)]
        );
    }

    #[test]
    fn test_builder() {
        let mut builder = ResumeDataVirtualAdder::new();
        builder.push_frame(42);
        builder.map_slot(0, 0);
        builder.map_slot(2, 1); // gap at slot 1
        let rd = builder.build();

        assert_eq!(rd.frames.len(), 1);
        assert_eq!(rd.frames[0].pc, 42);
        assert_eq!(
            rd.frames[0].slot_map,
            vec![
                FrameSlotSource::FailArg(0),
                FrameSlotSource::Unavailable,
                FrameSlotSource::FailArg(1),
            ]
        );
    }

    #[test]
    fn test_rpython_style_tagged_sources() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 55,
                slot_map: vec![
                    FrameSlotSource::FailArg(3),
                    FrameSlotSource::Constant(7),
                    FrameSlotSource::Virtual(1),
                    FrameSlotSource::Uninitialized,
                    FrameSlotSource::Unavailable,
                ],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };

        let encoded = rd.encode();
        assert_eq!(encoded.rd_numb[0] as usize, encoded.rd_numb.len());

        // Header: [size, count, num_frames, pc, slot_count, ...slots]
        let slot_words = &encoded.rd_numb[5..10];
        assert_eq!(untag_i64(slot_words[0]), (0, TAGBOX));
        assert_eq!(untag_i64(slot_words[1]), (7, TAGINT));
        assert_eq!(untag_i64(slot_words[2]), (1, TAGVIRTUAL));
        assert_eq!(untag_i64(slot_words[3]), (ENCODED_UNINITIALIZED, TAGCONST));
        assert_eq!(untag_i64(slot_words[4]), (ENCODED_UNAVAILABLE, TAGCONST));
    }

    #[test]
    fn test_encoded_resume_roundtrip_preserves_semantics() {
        let large_const = 1_i64 << 62;
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    pc: 10,
                    slot_map: vec![
                        FrameSlotSource::FailArg(0),
                        FrameSlotSource::Constant(large_const),
                        FrameSlotSource::Virtual(0),
                    ],
                },
                FrameInfo {
                    pc: 20,
                    slot_map: vec![FrameSlotSource::Unavailable, FrameSlotSource::Uninitialized],
                },
            ],
            virtuals: vec![
                VirtualInfo::VirtualObj {
                    descr: None,
                    type_id: 1,
                    descr_index: 7,
                    fields: vec![
                        (0, VirtualFieldSource::FailArg(1)),
                        (1, VirtualFieldSource::Constant(large_const)),
                    ],
                },
                VirtualInfo::VRawBuffer {
                    size: 8,
                    entries: vec![(0, 8, VirtualFieldSource::Virtual(0))],
                },
            ],
            pending_fields: Vec::new(),
        };

        let encoded = rd.encode();
        let decoded = encoded.decode_layout();
        assert_eq!(decoded.frames, rd.frames);
        assert_eq!(decoded.virtuals, rd.virtuals);
        assert_eq!(encoded.rd_consts, vec![large_const]);
    }

    #[test]
    fn test_encoded_resume_dedups_large_constants() {
        let large_const = (1_i64 << 62) + 123;
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 99,
                slot_map: vec![
                    FrameSlotSource::Constant(large_const),
                    FrameSlotSource::Constant(large_const),
                ],
            }],
            virtuals: vec![VirtualInfo::VArray {
                descr_index: 4,
                items: vec![
                    VirtualFieldSource::Constant(large_const),
                    VirtualFieldSource::Constant(large_const),
                ],
            }],
            pending_fields: Vec::new(),
        };

        let encoded = rd.encode();
        assert_eq!(encoded.rd_consts, vec![large_const]);
    }

    #[test]
    fn test_reconstruct_state_keeps_constants_and_virtual_slots() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 77,
                slot_map: vec![
                    FrameSlotSource::FailArg(0),
                    FrameSlotSource::Constant(42),
                    FrameSlotSource::Virtual(0),
                    FrameSlotSource::Uninitialized,
                    FrameSlotSource::Unavailable,
                ],
            }],
            virtuals: vec![VirtualInfo::VStruct {
                typedescr: None,
                type_id: 1,
                descr_index: 10,
                fields: vec![(0, VirtualFieldSource::FailArg(1))],
                fielddescrs: vec![],
                descr_size: 0,
            }],
            pending_fields: Vec::new(),
        };
        let state = rd.reconstruct_state(&[7, 99]);
        assert_eq!(state.frames.len(), 1);
        assert_eq!(
            state.frames[0].values,
            vec![
                ReconstructedValue::Value(7),
                ReconstructedValue::Value(42),
                ReconstructedValue::Virtual(0),
                ReconstructedValue::Uninitialized,
                ReconstructedValue::Unavailable,
            ]
        );
        assert_eq!(state.frames[0].lossy_values(), vec![7, 42, 0, 0, 0]);
        assert_eq!(state.virtuals.len(), 1);
        match &state.virtuals[0] {
            MaterializedVirtual::Struct { fields, .. } => {
                assert_eq!(fields, &vec![(0, MaterializedValue::Value(99))]);
            }
            other => panic!("expected Struct, got {other:?}"),
        }
        assert!(state.pending_fields.is_empty());
    }

    #[test]
    fn test_pending_fields_roundtrip_and_reconstruction() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 88,
                slot_map: vec![FrameSlotSource::FailArg(0), FrameSlotSource::FailArg(1)],
            }],
            virtuals: Vec::new(),
            pending_fields: vec![
                PendingFieldInfo {
                    descr_index: 7,
                    target: ResumeValueSource::FailArg(0),
                    value: ResumeValueSource::Constant(55),
                    item_index: None,
                },
                PendingFieldInfo {
                    descr_index: 8,
                    target: ResumeValueSource::FailArg(1),
                    value: ResumeValueSource::FailArg(2),
                    item_index: Some(3),
                },
            ],
        };

        let encoded = rd.encode();
        assert_eq!(encoded.rd_pendingfields.len(), 2);
        let decoded = encoded.decode();
        assert_eq!(decoded, rd);

        let state = encoded.reconstruct_state(&[101, 202, 303]);
        assert_eq!(
            state.pending_fields,
            vec![
                ResolvedPendingFieldWrite {
                    descr_index: 7,
                    target: MaterializedValue::Value(101),
                    value: MaterializedValue::Value(55),
                    item_index: None,
                },
                ResolvedPendingFieldWrite {
                    descr_index: 8,
                    target: MaterializedValue::Value(202),
                    value: MaterializedValue::Value(303),
                    item_index: Some(3),
                },
            ]
        );
    }

    #[test]
    fn test_materialize_virtual_obj() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![],
            }],
            virtuals: vec![VirtualInfo::VirtualObj {
                descr: None,
                type_id: 42,
                descr_index: 1,
                fields: vec![
                    (0, VirtualFieldSource::FailArg(0)),
                    (1, VirtualFieldSource::Constant(99)),
                ],
            }],
            pending_fields: Vec::new(),
        };
        let fail_values = vec![10, 20, 30];
        let materialized = rd.materialize_virtuals(&fail_values);
        assert_eq!(materialized.len(), 1);
        match &materialized[0] {
            MaterializedVirtual::Obj {
                type_id, fields, ..
            } => {
                assert_eq!(*type_id, 42);
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0], (0, MaterializedValue::Value(10))); // from FailArg(0)
                assert_eq!(fields[1], (1, MaterializedValue::Value(99))); // constant
            }
            _ => panic!("expected Obj"),
        }
    }

    #[test]
    fn test_materialize_virtual_array() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![],
            }],
            virtuals: vec![VirtualInfo::VArray {
                descr_index: 5,
                items: vec![
                    VirtualFieldSource::FailArg(0),
                    VirtualFieldSource::FailArg(1),
                    VirtualFieldSource::Constant(42),
                ],
            }],
            pending_fields: Vec::new(),
        };
        let fail_values = vec![100, 200];
        let materialized = rd.materialize_virtuals(&fail_values);
        assert_eq!(materialized.len(), 1);
        match &materialized[0] {
            MaterializedVirtual::Array { items, descr_index } => {
                assert_eq!(*descr_index, 5);
                assert_eq!(
                    *items,
                    vec![
                        MaterializedValue::Value(100),
                        MaterializedValue::Value(200),
                        MaterializedValue::Value(42),
                    ]
                );
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_materialize_nested_virtuals() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![],
            }],
            virtuals: vec![
                // Inner virtual (index 0)
                VirtualInfo::VStruct {
                    typedescr: None,
                    type_id: 1,
                    descr_index: 10,
                    fields: vec![(0, VirtualFieldSource::FailArg(0))],
                    fielddescrs: vec![],
                    descr_size: 0,
                },
                // Outer virtual (index 1), references inner via Virtual(0)
                VirtualInfo::VirtualObj {
                    descr: None,
                    type_id: 2,
                    descr_index: 20,
                    fields: vec![
                        (0, VirtualFieldSource::Virtual(0)),
                        (1, VirtualFieldSource::FailArg(1)),
                    ],
                },
            ],
            pending_fields: Vec::new(),
        };
        let fail_values = vec![10, 20];
        let materialized = rd.materialize_virtuals(&fail_values);
        assert_eq!(materialized.len(), 2);

        // Inner struct should be materialized
        match &materialized[0] {
            MaterializedVirtual::Struct {
                type_id, fields, ..
            } => {
                assert_eq!(*type_id, 1);
                assert_eq!(fields[0], (0, MaterializedValue::Value(10)));
            }
            _ => panic!("expected Struct"),
        }

        // Outer obj should reference inner via placeholder
        match &materialized[1] {
            MaterializedVirtual::Obj {
                type_id, fields, ..
            } => {
                assert_eq!(*type_id, 2);
                assert_eq!(fields[0], (0, MaterializedValue::VirtualRef(0)));
                assert_eq!(fields[1], (1, MaterializedValue::Value(20)));
            }
            _ => panic!("expected Obj"),
        }
    }

    #[test]
    fn test_materialize_array_struct() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![],
            }],
            virtuals: vec![VirtualInfo::VArrayStruct {
                descr_index: 3,
                element_fields: vec![
                    vec![
                        (0, VirtualFieldSource::FailArg(0)),
                        (1, VirtualFieldSource::FailArg(1)),
                    ],
                    vec![
                        (0, VirtualFieldSource::Constant(99)),
                        (1, VirtualFieldSource::FailArg(2)),
                    ],
                ],
            }],
            pending_fields: Vec::new(),
        };
        let fail_values = vec![10, 20, 30];
        let materialized = rd.materialize_virtuals(&fail_values);
        match &materialized[0] {
            MaterializedVirtual::ArrayStruct { elements, .. } => {
                assert_eq!(elements.len(), 2);
                assert_eq!(
                    elements[0],
                    vec![
                        (0, MaterializedValue::Value(10)),
                        (1, MaterializedValue::Value(20)),
                    ]
                );
                assert_eq!(
                    elements[1],
                    vec![
                        (0, MaterializedValue::Value(99)),
                        (1, MaterializedValue::Value(30)),
                    ]
                );
            }
            _ => panic!("expected ArrayStruct"),
        }
    }

    #[test]
    fn test_materialize_raw_buffer() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![],
            }],
            virtuals: vec![VirtualInfo::VRawBuffer {
                size: 32,
                entries: vec![
                    (0, 8, VirtualFieldSource::FailArg(0)),
                    (8, 4, VirtualFieldSource::Constant(0xFF)),
                ],
            }],
            pending_fields: Vec::new(),
        };
        let fail_values = vec![0xCAFE];
        let materialized = rd.materialize_virtuals(&fail_values);
        match &materialized[0] {
            MaterializedVirtual::RawBuffer { size, entries } => {
                assert_eq!(*size, 32);
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0], (0, 8, MaterializedValue::Value(0xCAFE)));
                assert_eq!(entries[1], (8, 4, MaterializedValue::Value(0xFF)));
            }
            _ => panic!("expected RawBuffer"),
        }
    }

    #[test]
    fn test_builder_add_virtual_convenience() {
        let mut builder = ResumeDataVirtualAdder::new();
        builder.push_frame(0);
        let v0 = builder.add_virtual_obj(
            None,
            1,
            10,
            None,
            vec![(0, VirtualFieldSource::FailArg(0))],
            vec![],
            0,
        );
        let v1 = builder.add_virtual_array(
            20,
            vec![
                VirtualFieldSource::FailArg(1),
                VirtualFieldSource::Virtual(v0),
            ],
        );
        let v2 = builder.add_virtual_raw_buffer(16, vec![(0, 8, VirtualFieldSource::Constant(42))]);
        assert_eq!(v0, 0);
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);

        let rd = builder.build();
        assert_eq!(rd.virtuals.len(), 3);
    }

    // ── Encoding / Decoding tests ──

    #[test]
    fn test_encode_decode_simple() {
        let rd = ResumeData::simple(42, 3);
        let encoded = EncodedResumeData::encode(&rd);

        let decoded = encoded.decode();
        assert_eq!(decoded.frames.len(), 1);
        assert_eq!(decoded.frames[0].pc, 42);
        assert_eq!(decoded.frames[0].slot_map.len(), 3);
        assert_eq!(decoded.frames[0].slot_map[0], FrameSlotSource::FailArg(0));
        assert_eq!(decoded.frames[0].slot_map[1], FrameSlotSource::FailArg(1));
        assert_eq!(decoded.frames[0].slot_map[2], FrameSlotSource::FailArg(2));
    }

    #[test]
    fn test_encode_decode_mixed_sources() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 100,
                slot_map: vec![
                    FrameSlotSource::FailArg(0),
                    FrameSlotSource::Constant(42),
                    FrameSlotSource::Virtual(0),
                    FrameSlotSource::Uninitialized,
                    FrameSlotSource::Unavailable,
                ],
            }],
            virtuals: vec![VirtualInfo::VStruct {
                typedescr: None,
                type_id: 1,
                descr_index: 10,
                fields: vec![(0, VirtualFieldSource::FailArg(1))],
                fielddescrs: vec![],
                descr_size: 0,
            }],
            pending_fields: Vec::new(),
        };

        let encoded = EncodedResumeData::encode(&rd);
        let decoded = encoded.decode();
        assert_eq!(decoded, rd);
    }

    #[test]
    fn test_encode_decode_multi_frame() {
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    pc: 10,
                    slot_map: vec![FrameSlotSource::FailArg(0), FrameSlotSource::Constant(99)],
                },
                FrameInfo {
                    pc: 20,
                    slot_map: vec![FrameSlotSource::FailArg(1), FrameSlotSource::FailArg(2)],
                },
            ],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };
        let encoded = EncodedResumeData::encode(&rd);
        let decoded = encoded.decode();
        assert_eq!(decoded, rd);
    }

    #[test]
    fn test_encoded_resume_compacts_sparse_fail_args_and_projects_raw_values() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 33,
                slot_map: vec![
                    FrameSlotSource::FailArg(7),
                    FrameSlotSource::Constant(11),
                    FrameSlotSource::FailArg(2),
                ],
            }],
            virtuals: vec![VirtualInfo::VStruct {
                typedescr: None,
                type_id: 1,
                descr_index: 9,
                fields: vec![(0, VirtualFieldSource::FailArg(7))],
                fielddescrs: vec![],
                descr_size: 0,
            }],
            pending_fields: vec![PendingFieldInfo {
                descr_index: 12,
                target: ResumeValueSource::FailArg(2),
                value: ResumeValueSource::FailArg(7),
                item_index: Some(4),
            }],
        };

        let encoded = EncodedResumeData::encode(&rd);
        assert_eq!(encoded.decode(), rd);

        let raw_fail_values = vec![100, 101, 102, 103, 104, 105, 106, 107];

        let reconstructed = encoded.reconstruct_state(&raw_fail_values);
        assert_eq!(
            reconstructed.frames[0].values,
            vec![
                ReconstructedValue::Value(107),
                ReconstructedValue::Value(11),
                ReconstructedValue::Value(102),
            ]
        );
        assert_eq!(
            reconstructed.pending_fields,
            vec![ResolvedPendingFieldWrite {
                descr_index: 12,
                target: MaterializedValue::Value(102),
                value: MaterializedValue::Value(107),
                item_index: Some(4),
            }]
        );
    }

    #[test]
    fn test_encode_decode_all_virtual_types() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![FrameSlotSource::Virtual(0), FrameSlotSource::Virtual(1)],
            }],
            virtuals: vec![
                VirtualInfo::VirtualObj {
                    descr: None,
                    type_id: 42,
                    descr_index: 1,
                    fields: vec![
                        (0, VirtualFieldSource::FailArg(0)),
                        (1, VirtualFieldSource::Constant(99)),
                    ],
                },
                VirtualInfo::VArray {
                    descr_index: 5,
                    items: vec![
                        VirtualFieldSource::FailArg(1),
                        VirtualFieldSource::Constant(7),
                    ],
                },
            ],
            pending_fields: Vec::new(),
        };
        let encoded = EncodedResumeData::encode(&rd);
        let decoded = encoded.decode();
        assert_eq!(decoded, rd);
    }

    #[test]
    fn test_encode_decode_array_struct_and_raw_buffer() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![FrameSlotSource::Virtual(0), FrameSlotSource::Virtual(1)],
            }],
            virtuals: vec![
                VirtualInfo::VArrayStruct {
                    descr_index: 3,
                    element_fields: vec![
                        vec![
                            (0, VirtualFieldSource::FailArg(0)),
                            (1, VirtualFieldSource::Constant(11)),
                        ],
                        vec![(0, VirtualFieldSource::FailArg(1))],
                    ],
                },
                VirtualInfo::VRawBuffer {
                    size: 32,
                    entries: vec![
                        (0, 8, VirtualFieldSource::FailArg(2)),
                        (8, 4, VirtualFieldSource::Constant(0xFF)),
                    ],
                },
            ],
            pending_fields: Vec::new(),
        };
        let encoded = EncodedResumeData::encode(&rd);
        let decoded = encoded.decode();
        assert_eq!(decoded, rd);
    }

    #[test]
    fn test_encode_inline_vs_pool_constants() {
        // Small constant: should be inlined with TAGINT
        let rd_small = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![FrameSlotSource::Constant(42)],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };
        let enc_small = EncodedResumeData::encode(&rd_small);
        assert!(enc_small.rd_consts.is_empty()); // small const inlined
        let dec_small = enc_small.decode();
        assert_eq!(dec_small, rd_small);
    }

    #[test]
    fn test_encode_no_failargs() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 0,
                slot_map: vec![FrameSlotSource::Constant(1), FrameSlotSource::Unavailable],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };
        let enc = EncodedResumeData::encode(&rd);
        let dec = enc.decode();
        assert_eq!(dec, rd);
    }

    #[test]
    fn test_encode_decode_roundtrip_reconstruction() {
        // Verify that encoding → decoding → reconstruction gives the same result
        let rd = ResumeData {
            frames: vec![FrameInfo {
                pc: 77,
                slot_map: vec![
                    FrameSlotSource::FailArg(0),
                    FrameSlotSource::Constant(42),
                    FrameSlotSource::Virtual(0),
                ],
            }],
            virtuals: vec![VirtualInfo::VStruct {
                typedescr: None,
                type_id: 1,
                descr_index: 10,
                fields: vec![(0, VirtualFieldSource::FailArg(1))],
                fielddescrs: vec![],
                descr_size: 0,
            }],
            pending_fields: Vec::new(),
        };
        let fail_values = vec![7, 99];

        let original_state = rd.reconstruct_state(&fail_values);
        let encoded = EncodedResumeData::encode(&rd);
        let decoded = encoded.decode();
        let roundtrip_state = decoded.reconstruct_state(&fail_values);

        assert_eq!(original_state.frames.len(), roundtrip_state.frames.len());
        assert_eq!(original_state.frames[0].pc, roundtrip_state.frames[0].pc);
        assert_eq!(
            original_state.frames[0].values,
            roundtrip_state.frames[0].values
        );
    }

    // ── ResumeDataLoopMemo tests ──

    #[test]
    fn test_loop_memo_shares_constant_pool() {
        let mut memo = ResumeDataLoopMemo::new();

        let rd1 = ResumeData {
            frames: vec![FrameInfo {
                pc: 10,
                slot_map: vec![FrameSlotSource::FailArg(0), FrameSlotSource::Constant(42)],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };
        let enc1 = memo.encode_shared(&rd1);
        let pool_size_1 = memo.num_shared_consts();

        // Second guard re-uses the same constant 42
        let rd2 = ResumeData {
            frames: vec![FrameInfo {
                pc: 20,
                slot_map: vec![
                    FrameSlotSource::FailArg(0),
                    FrameSlotSource::Constant(42),
                    FrameSlotSource::Constant(99),
                ],
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };
        let enc2 = memo.encode_shared(&rd2);
        let pool_size_2 = memo.num_shared_consts();

        // 42 is small enough to be inlined, so pool should be empty
        // unless it exceeds the inline range. For small values, both
        // guards use TAGINT, so the pool stays at 0.
        assert_eq!(pool_size_1, 0);
        assert_eq!(pool_size_2, 0);

        // Roundtrip both
        let dec1 = enc1.decode();
        let dec2 = enc2.decode();
        assert_eq!(dec1, rd1);
        assert_eq!(dec2, rd2);
    }

    #[test]
    fn test_layout_summary_reports_slot_virtual_and_pending_write_kinds() {
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    pc: 11,
                    slot_map: vec![
                        FrameSlotSource::FailArg(3),
                        FrameSlotSource::Constant(77),
                        FrameSlotSource::Virtual(0),
                    ],
                },
                FrameInfo {
                    pc: 22,
                    slot_map: vec![FrameSlotSource::Unavailable, FrameSlotSource::Uninitialized],
                },
            ],
            virtuals: vec![VirtualInfo::VArray {
                descr_index: 7,
                items: vec![VirtualFieldSource::FailArg(3)],
            }],
            pending_fields: vec![PendingFieldInfo {
                descr_index: 9,
                target: ResumeValueSource::FailArg(1),
                value: ResumeValueSource::Constant(88),
                item_index: Some(2),
            }],
        };

        let summary = rd.encode().layout_summary();
        assert_eq!(summary.num_frames, 2);
        assert_eq!(summary.frame_pcs, vec![11, 22]);
        assert_eq!(summary.frame_slot_counts, vec![3, 2]);
        assert_eq!(
            summary.frame_layouts,
            vec![
                ResumeFrameLayoutSummary {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 11,
                    slot_sources: vec![
                        ResumeValueKind::FailArg,
                        ResumeValueKind::Constant,
                        ResumeValueKind::Virtual,
                    ],
                    slot_layouts: vec![
                        ResumeValueLayoutSummary {
                            kind: ResumeValueKind::FailArg,
                            fail_arg_index: 0,
                            raw_fail_arg_position: Some(3),
                            constant: None,
                            virtual_index: None,
                        },
                        ResumeValueLayoutSummary {
                            kind: ResumeValueKind::Constant,
                            fail_arg_index: 0,
                            raw_fail_arg_position: None,
                            constant: Some(77),
                            virtual_index: None,
                        },
                        ResumeValueLayoutSummary {
                            kind: ResumeValueKind::Virtual,
                            fail_arg_index: 0,
                            raw_fail_arg_position: None,
                            constant: None,
                            virtual_index: Some(0),
                        },
                    ],
                    slot_types: None,
                },
                ResumeFrameLayoutSummary {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 22,
                    slot_sources: vec![
                        ResumeValueKind::Unavailable,
                        ResumeValueKind::Uninitialized,
                    ],
                    slot_layouts: vec![
                        ResumeValueLayoutSummary {
                            kind: ResumeValueKind::Unavailable,
                            fail_arg_index: 0,
                            raw_fail_arg_position: None,
                            constant: None,
                            virtual_index: None,
                        },
                        ResumeValueLayoutSummary {
                            kind: ResumeValueKind::Uninitialized,
                            fail_arg_index: 0,
                            raw_fail_arg_position: None,
                            constant: None,
                            virtual_index: None,
                        },
                    ],
                    slot_types: None,
                },
            ]
        );
        assert_eq!(summary.virtual_kinds, vec![ResumeVirtualKind::Array]);
        assert_eq!(
            summary.virtual_layouts,
            vec![ResumeVirtualLayoutSummary::Array {
                descr_index: 7,
                items: vec![ResumeValueLayoutSummary {
                    kind: ResumeValueKind::FailArg,
                    fail_arg_index: 0,
                    raw_fail_arg_position: Some(3),
                    constant: None,
                    virtual_index: None,
                }],
            }]
        );
        assert_eq!(
            summary.pending_field_layouts,
            vec![PendingFieldLayoutSummary {
                descr_index: 9,
                item_index: Some(2),
                is_array_item: true,
                target_kind: ResumeValueKind::FailArg,
                value_kind: ResumeValueKind::Constant,
                target: ResumeValueLayoutSummary {
                    kind: ResumeValueKind::FailArg,
                    fail_arg_index: 1,
                    raw_fail_arg_position: Some(1),
                    constant: None,
                    virtual_index: None,
                },
                value: ResumeValueLayoutSummary {
                    kind: ResumeValueKind::Constant,
                    fail_arg_index: 0,
                    raw_fail_arg_position: None,
                    constant: Some(88),
                    virtual_index: None,
                },
            }]
        );
        assert_eq!(summary.pending_field_count, 1);
    }

    #[test]
    fn test_layout_summary_roundtrips_into_reconstruction_helpers() {
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    pc: 55,
                    slot_map: vec![
                        FrameSlotSource::Virtual(0),
                        FrameSlotSource::FailArg(4),
                        FrameSlotSource::Constant(99),
                    ],
                },
                FrameInfo {
                    pc: 77,
                    slot_map: vec![FrameSlotSource::Unavailable, FrameSlotSource::Virtual(1)],
                },
            ],
            virtuals: vec![
                VirtualInfo::VStruct {
                    typedescr: None,
                    type_id: 3,
                    descr_index: 10,
                    fields: vec![
                        (0, VirtualFieldSource::FailArg(2)),
                        (1, VirtualFieldSource::Constant(123)),
                    ],
                    fielddescrs: vec![],
                    descr_size: 0,
                },
                VirtualInfo::VArray {
                    descr_index: 11,
                    items: vec![
                        VirtualFieldSource::Virtual(0),
                        VirtualFieldSource::FailArg(1),
                    ],
                },
            ],
            pending_fields: vec![
                PendingFieldInfo {
                    descr_index: 44,
                    target: ResumeValueSource::Virtual(0),
                    value: ResumeValueSource::FailArg(4),
                    item_index: None,
                },
                PendingFieldInfo {
                    descr_index: 45,
                    target: ResumeValueSource::FailArg(0),
                    value: ResumeValueSource::Virtual(1),
                    item_index: Some(2),
                },
            ],
        };
        let encoded = rd.encode();
        let summary = encoded.layout_summary();
        let fail_values = vec![101, 202, 303, 404, 505];

        assert_eq!(summary.to_resume_data(), encoded.decode());

        let expected = encoded.reconstruct_state(&fail_values);
        let reconstructed = summary.reconstruct_state(&fail_values);

        assert_eq!(reconstructed.frames.len(), expected.frames.len());
        for (actual, expected) in reconstructed.frames.iter().zip(expected.frames.iter()) {
            assert_eq!(actual.pc, expected.pc);
            assert_eq!(actual.values, expected.values);
        }
        assert_eq!(reconstructed.virtuals, expected.virtuals);
        assert_eq!(reconstructed.pending_fields, expected.pending_fields);
        assert_eq!(
            summary.materialize_virtuals(&fail_values),
            expected.virtuals
        );
        assert_eq!(
            summary.resolve_pending_field_writes(&fail_values),
            expected.pending_fields
        );
    }

    #[test]
    fn test_layout_summary_reconstruction_preserves_frame_metadata() {
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    pc: 55,
                    slot_map: vec![FrameSlotSource::FailArg(0), FrameSlotSource::Constant(99)],
                },
                FrameInfo {
                    pc: 77,
                    slot_map: vec![FrameSlotSource::FailArg(1)],
                },
            ],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        };
        let mut summary = rd.encode().layout_summary();
        summary.frame_layouts[0].trace_id = Some(1001);
        summary.frame_layouts[0].header_pc = Some(5001);
        summary.frame_layouts[0].source_guard = Some((9001, 1));
        summary.frame_layouts[0].slot_types = Some(vec![Type::Int, Type::Int]);
        summary.frame_layouts[1].trace_id = Some(1002);
        summary.frame_layouts[1].header_pc = Some(5002);
        summary.frame_layouts[1].source_guard = Some((9002, 2));
        summary.frame_layouts[1].slot_types = Some(vec![Type::Ref]);

        let reconstructed = summary.reconstruct_state(&[11, 22]);
        assert_eq!(reconstructed.frames.len(), 2);
        assert_eq!(reconstructed.frames[0].trace_id, Some(1001));
        assert_eq!(reconstructed.frames[0].header_pc, Some(5001));
        assert_eq!(reconstructed.frames[0].source_guard, Some((9001, 1)));
        assert_eq!(
            reconstructed.frames[0].slot_types,
            Some(vec![Type::Int, Type::Int])
        );
        assert_eq!(reconstructed.frames[1].trace_id, Some(1002));
        assert_eq!(reconstructed.frames[1].header_pc, Some(5002));
        assert_eq!(reconstructed.frames[1].source_guard, Some((9002, 2)));
        assert_eq!(reconstructed.frames[1].slot_types, Some(vec![Type::Ref]));

        let frame = summary
            .reconstruct_frame(1, &[11, 22])
            .expect("frame metadata should reconstruct");
        assert_eq!(frame.trace_id, Some(1002));
        assert_eq!(frame.header_pc, Some(5002));
        assert_eq!(frame.source_guard, Some((9002, 2)));
        assert_eq!(frame.slot_types, Some(vec![Type::Ref]));
        assert_eq!(frame.lossy_values(), vec![22]);
    }

    #[test]
    fn test_layout_summary_to_exit_recovery_layout_preserves_caller_prefix_and_shifts_virtuals() {
        let summary = ResumeLayoutSummary {
            num_frames: 1,
            frame_pcs: vec![77],
            frame_slot_counts: vec![1],
            frame_layouts: vec![ResumeFrameLayoutSummary {
                trace_id: Some(22),
                header_pc: Some(222),
                source_guard: Some((21, 5)),
                pc: 77,
                slot_sources: vec![ResumeValueKind::Virtual],
                slot_layouts: vec![ResumeValueLayoutSummary {
                    kind: ResumeValueKind::Virtual,
                    fail_arg_index: 0,
                    raw_fail_arg_position: None,
                    constant: None,
                    virtual_index: Some(0),
                }],
                slot_types: Some(vec![Type::Ref]),
            }],
            num_virtuals: 1,
            virtual_kinds: vec![ResumeVirtualKind::Array],
            virtual_layouts: vec![ResumeVirtualLayoutSummary::Array {
                descr_index: 9,
                items: vec![ResumeValueLayoutSummary {
                    kind: ResumeValueKind::FailArg,
                    fail_arg_index: 0,
                    raw_fail_arg_position: Some(1),
                    constant: None,
                    virtual_index: None,
                }],
            }],
            pending_field_count: 1,
            pending_field_layouts: vec![PendingFieldLayoutSummary {
                descr_index: 11,
                item_index: Some(2),
                is_array_item: true,
                target_kind: ResumeValueKind::Virtual,
                value_kind: ResumeValueKind::FailArg,
                target: ResumeValueLayoutSummary {
                    kind: ResumeValueKind::Virtual,
                    fail_arg_index: 0,
                    raw_fail_arg_position: None,
                    constant: None,
                    virtual_index: Some(0),
                },
                value: ResumeValueLayoutSummary {
                    kind: ResumeValueKind::FailArg,
                    fail_arg_index: 0,
                    raw_fail_arg_position: Some(1),
                    constant: None,
                    virtual_index: None,
                },
            }],
            const_pool_size: 0,
        };
        let caller_prefix = ExitRecoveryLayout {
            frames: vec![
                ExitFrameLayout {
                    trace_id: Some(10),
                    header_pc: Some(100),
                    source_guard: Some((9, 1)),
                    pc: 55,
                    slots: vec![ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
                ExitFrameLayout {
                    trace_id: Some(11),
                    header_pc: Some(110),
                    source_guard: Some((10, 2)),
                    pc: 66,
                    slots: vec![ExitValueSourceLayout::ExitValue(1)],
                    slot_types: Some(vec![Type::Ref]),
                },
            ],
            virtual_layouts: vec![ExitVirtualLayout::Array {
                descr_index: 3,
                items: vec![ExitValueSourceLayout::Constant(9)],
            }],
            pending_field_layouts: vec![ExitPendingFieldLayout {
                descr_index: 4,
                item_index: None,
                is_array_item: false,
                target: ExitValueSourceLayout::ExitValue(0),
                value: ExitValueSourceLayout::Constant(1),
                field_offset: 0,
                field_size: 0,
                field_type: majit_ir::Type::Int,
            }],
        };

        let recovery = summary.to_exit_recovery_layout_with_caller_prefix(Some(&caller_prefix));
        assert_eq!(recovery.frames.len(), 2);
        assert_eq!(recovery.frames[0], caller_prefix.frames[0]);
        assert_eq!(recovery.frames[1].trace_id, Some(22));
        assert_eq!(recovery.frames[1].header_pc, Some(222));
        assert_eq!(recovery.frames[1].source_guard, Some((21, 5)));
        assert_eq!(recovery.frames[1].pc, 77);
        assert_eq!(recovery.frames[1].slot_types, Some(vec![Type::Ref]));
        assert_eq!(
            recovery.frames[1].slots,
            vec![ExitValueSourceLayout::Virtual(1)]
        );
        assert_eq!(recovery.virtual_layouts.len(), 2);
        assert_eq!(
            recovery.virtual_layouts[0],
            caller_prefix.virtual_layouts[0]
        );
        assert_eq!(
            recovery.virtual_layouts[1],
            ExitVirtualLayout::Array {
                descr_index: 9,
                items: vec![ExitValueSourceLayout::ExitValue(1)],
            }
        );
        assert_eq!(recovery.pending_field_layouts.len(), 2);
        assert_eq!(
            recovery.pending_field_layouts[1],
            ExitPendingFieldLayout {
                descr_index: 11,
                item_index: Some(2),
                is_array_item: true,
                target: ExitValueSourceLayout::Virtual(1),
                value: ExitValueSourceLayout::ExitValue(1),
                field_offset: 0,
                field_size: 0,
                field_type: majit_ir::Type::Int,
            }
        );
    }

    #[test]
    fn test_memo_number_simple() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.constants.insert(10001, (42i64, majit_ir::Type::Int));
        let snapshot = Snapshot::single_frame(8, vec![OpRef(10001), OpRef(1), OpRef(2)]);
        let numb_state = memo.number(&snapshot, &env).unwrap();
        // Should have: [size, num_failargs, 0(vable), 0(vref), 0(jitcode), 8(pc), tagged...]
        let items = crate::resumecode::unpack_all(&numb_state.create_numbering());
        // items[0] = total size
        assert!(items[0] > 0);
        // items[1] = num_failargs: 0 (not patched yet — RPython patches in finish())
        // After finish: patch(1, numb_state.liveboxes.len()) would set to 2.
        assert_eq!(items[1], 0);
        // items[2] = vable_array_length = 0
        assert_eq!(items[2], 0);
        // items[3] = vref_array_length = 0
        assert_eq!(items[3], 0);
        // items[4] = jitcode_index = 0
        assert_eq!(items[4], 0);
        // items[5] = pc = 8
        assert_eq!(items[5], 8);
        // items[6] = OpRef(10001) tagged as TAGINT(42) since 42 fits in 13 bits
        let (val, tagbits) = untag(items[6] as i16);
        assert_eq!(tagbits, TAGINT);
        assert_eq!(val, 42);
        // items[7] = OpRef(1) tagged as TAGBOX(0) — first live box
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[8] = OpRef(2) tagged as TAGBOX(1) — second live box
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_number_rebuild_roundtrip() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.constants.insert(10001, (42i64, majit_ir::Type::Int));
        let snapshot = Snapshot::single_frame(8, vec![OpRef(10001), OpRef(1), OpRef(2)]);
        let mut numb_state = memo.number(&snapshot, &env).unwrap();
        // RPython: ResumeDataVirtualAdder.finish() patches slot 1 with num_boxes.
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, memo.consts(), None);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(rebuilt_frames[0].pc, 8);
        assert_eq!(rebuilt_frames[0].values.len(), 3);
        assert_eq!(rebuilt_frames[0].values[0], RebuiltValue::Int(42));
        assert_eq!(rebuilt_frames[0].values[1], RebuiltValue::Box(0));
        assert_eq!(rebuilt_frames[0].values[2], RebuiltValue::Box(1));
    }

    #[test]
    fn test_number_rebuild_with_virtual() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2); // OpRef(2) is virtual (Ref type)
        env.types.insert(2, majit_ir::Type::Ref);
        let snapshot = Snapshot::single_frame(10, vec![OpRef(1), OpRef(2), OpRef(3)]);
        let mut numb_state = memo.number(&snapshot, &env).unwrap();
        // RPython: finish() patches with len(newboxes) which is num_boxes
        // (not liveboxes which includes virtuals).
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, memo.consts(), None);
        assert_eq!(num_failargs, 2); // OpRef(1) and OpRef(3) are boxes
        assert_eq!(rebuilt_frames[0].values.len(), 3);
        assert_eq!(rebuilt_frames[0].values[0], RebuiltValue::Box(0));
        assert_eq!(rebuilt_frames[0].values[1], RebuiltValue::Virtual(0));
        assert_eq!(rebuilt_frames[0].values[2], RebuiltValue::Box(1));
    }

    #[test]
    fn test_memo_number_with_virtual() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2);
        env.types.insert(2, majit_ir::Type::Ref);
        let snapshot = Snapshot::single_frame(10, vec![OpRef(1), OpRef(2), OpRef(3)]);
        let numb_state = memo.number(&snapshot, &env).unwrap();
        let items = crate::resumecode::unpack_all(&numb_state.create_numbering());
        // items[1] = num_failargs: 0 (not patched — RPython patches in finish())
        assert_eq!(items[1], 0);
        // items[6] = OpRef(1) → TAGBOX(0)
        let (val, tagbits) = untag(items[6] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[7] = OpRef(2) → TAGVIRTUAL(0)
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGVIRTUAL);
        assert_eq!(val, 0);
        // items[8] = OpRef(3) → TAGBOX(1)
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_multi_frame_snapshot() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.constants.insert(10000, (99i64, majit_ir::Type::Int));

        let snapshot = Snapshot {
            vable_array: vec![],
            vref_array: vec![],
            framestack: vec![
                SnapshotFrame {
                    jitcode_index: 0,
                    pc: 10,
                    boxes: vec![OpRef(1), OpRef(10000)],
                },
                SnapshotFrame {
                    jitcode_index: 1,
                    pc: 20,
                    boxes: vec![OpRef(2), OpRef(3)],
                },
            ],
        };

        let mut numb_state = memo.number(&snapshot, &env).unwrap();
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        // Multi-frame encoding: no box_count, RPython parity.
        let items = crate::resumecode::unpack_all(&rd_numb);
        assert_eq!(items[1], 3); // num_failargs: 3 boxes patched
        // Frame 0: items[4]=jitcode(0), items[5]=pc(10), items[6..7]=tagged
        assert_eq!(items[4], 0);
        assert_eq!(items[5], 10);
        // Frame 1: items[8]=jitcode(1), items[9]=pc(20), items[10..11]=tagged
        assert_eq!(items[8], 1);
        assert_eq!(items[9], 20);

        // Roundtrip with liveness-based closure.
        let rd_consts: Vec<(i64, majit_ir::Type)> = memo.consts().to_vec();
        let frame_count = |jitcode_index: i32, _pc: i32| -> usize {
            match jitcode_index {
                0 => 2, // Frame 0 has 2 boxes
                1 => 2, // Frame 1 has 2 boxes
                _ => 0,
            }
        };
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, Some(&frame_count));
        assert_eq!(num_failargs, 3);
        assert_eq!(rebuilt_frames.len(), 2);
        assert_eq!(rebuilt_frames[0].jitcode_index, 0);
        assert_eq!(rebuilt_frames[0].pc, 10);
        assert_eq!(rebuilt_frames[0].values.len(), 2);
        assert_eq!(rebuilt_frames[1].jitcode_index, 1);
        assert_eq!(rebuilt_frames[1].pc, 20);
        assert_eq!(rebuilt_frames[1].values.len(), 2);
    }

    #[test]
    fn test_finish_produces_rd_numb_and_liveboxes() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.constants.insert(10001, (42i64, majit_ir::Type::Int));
        env.virtuals.insert(2);
        env.types.insert(2, majit_ir::Type::Ref);

        let snapshot = Snapshot::single_frame(8, vec![OpRef(10001), OpRef(1), OpRef(2), OpRef(3)]);
        let numb_state = memo.number(&snapshot, &env).unwrap();
        let (rd_numb, rd_consts, _rd_virtuals, liveboxes, _livebox_types) =
            memo.finish(numb_state, &env, &mut [], None);

        // liveboxes should contain only TAGBOX entries: OpRef(1) and OpRef(3)
        assert_eq!(liveboxes.len(), 2);
        assert_eq!(liveboxes[0], OpRef(1)); // box #0
        assert_eq!(liveboxes[1], OpRef(3)); // box #1

        // rd_numb should be valid
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, None);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(rebuilt_frames[0].values[0], RebuiltValue::Int(42));
        assert_eq!(rebuilt_frames[0].values[1], RebuiltValue::Box(0));
        assert_eq!(rebuilt_frames[0].values[2], RebuiltValue::Virtual(0));
        assert_eq!(rebuilt_frames[0].values[3], RebuiltValue::Box(1));
    }
}

// ═══════════════════════════════════════════════════════════════
// resume.py:901-1039 AbstractResumeDataReader
// resume.py:1354-1601 ResumeDataDirectReader
//
// Direct reader that decodes resume data and fills blackhole
// interpreter registers with concrete values from the deadframe.
// ═══════════════════════════════════════════════════════════════

use crate::blackhole::BlackholeInterpreter;
use crate::resumecode::Reader;

/// RPython virtualref_info interface for resume data consumption.
///
/// Corresponds to `metainterp_sd.virtualref_info` (VirtualRefInfo).
pub trait VRefInfo {
    /// resume.py:1397 vrefinfo.continue_tracing(vref, virtual)
    fn continue_tracing(&self, vref: i64, virtual_ref: i64);
}

/// RPython virtualizable_info interface for resume data consumption.
///
/// Corresponds to `jitdriver_sd.virtualizable_info` (VirtualizableInfo).
pub trait VirtualizableInfo {
    /// resume.py:1406 vinfo.get_total_size(virtualizable)
    fn get_total_size(&self, virtualizable: i64) -> usize;

    /// resume.py:1407 vinfo.reset_token_gcref(virtualizable)
    fn reset_token_gcref(&self, virtualizable: i64);

    /// resume.py:1408 vinfo.write_from_resume_data_partial(virtualizable, self)
    ///
    /// Read fields from the resume reader and write them into the virtualizable.
    fn write_from_resume_data_partial(
        &self,
        virtualizable: i64,
        reader: &mut ResumeDataDirectReader,
    );
}

/// RPython greenfield_info interface for resume data consumption.
///
/// Corresponds to `jitdriver_sd.greenfield_info`.
pub trait GreenfieldInfo {}

/// resume.py:1354 ResumeDataDirectReader
///
/// Reads encoded resume data (rd_numb) and fills blackhole interpreter
/// registers directly from the deadframe's fail_args values.
///
/// Combines AbstractResumeDataReader (resume.py:901) mixin with
/// ResumeDataDirectReader (resume.py:1354) concrete class.
pub struct ResumeDataDirectReader<'a> {
    // AbstractResumeDataReader fields (resume.py:909-922)
    /// resume.py:918 resumecodereader
    pub resumecodereader: Reader<'a>,
    /// resume.py:919 items_resume_section — total items in resume section
    pub items_resume_section: i32,
    /// resume.py:921 count — number of failargs
    pub count: i32,
    /// resume.py:922 consts — constant pool from rd_consts
    pub consts: &'a [i64],

    // ResumeDataDirectReader fields (resume.py:1364-1367)
    /// resume.py:1366 deadframe — raw fail_args values
    pub deadframe: &'a [i64],
    /// pyre flat-deadframe adaptation: original type of each deadframe slot.
    /// RPython's CPU exposes typed getters (get_ref_value/get_int_value/...);
    /// pyre passes a flat raw slice and needs slot kinds to emulate
    /// load_box_from_cpu(kind) for TAGBOX decode.
    pub deadframe_types: Option<&'a [majit_ir::Type]>,

    // resume.py:1358 resume_after_guard_not_forced
    //   0: not a GUARD_NOT_FORCED
    //   1: in handle_async_forcing
    //   2: resuming from the GUARD_NOT_FORCED
    pub resume_after_guard_not_forced: u8,

    // resume.py:909 rd_virtuals
    rd_virtuals: Option<&'a [VirtualInfo]>,

    // resume.py:910 virtuals_cache — lazy-allocated virtual objects
    virtuals_cache_ptr: Vec<i64>,
    virtuals_cache_int: Vec<i64>,

    /// resume.py:1367 — CPU allocation backend.
    /// RPython uses self.cpu (from metainterp_sd.cpu) for allocate_with_vtable etc.
    allocator: &'a dyn BlackholeAllocator,

    /// resume.py:1404: virtualizable pointer read by consume_vable_info.
    /// Stored so the caller (blackhole_from_resumedata) can access it
    /// after consume_vref_and_vable completes.
    pub virtualizable_ptr: i64,
}

/// resume.py:1433-1456 CPU allocation interface for virtual materialization.
///
/// ResumeDataDirectReader calls these methods when TAGVIRTUAL values
/// need to be lazily allocated during decode_ref/decode_int.
pub trait BlackholeAllocator {
    /// resume.py:1437 allocate_with_vtable
    fn allocate_with_vtable(&self, descr_index: u32) -> i64 {
        let _ = descr_index;
        0
    }
    /// resume.py:1441 allocate_struct
    fn allocate_struct(&self, descr_index: u32) -> i64 {
        let _ = descr_index;
        0
    }
    /// resume.py:1444 allocate_array
    fn allocate_array(&self, length: usize, descr_index: u32, clear: bool) -> i64 {
        let _ = (length, descr_index, clear);
        0
    }
    /// resume.py:1509 setfield
    fn setfield(&self, struct_ptr: i64, field_descr: u32, value: i64) {
        let _ = (struct_ptr, field_descr, value);
    }
    /// resume.py:1531 setarrayitem_int
    fn setarrayitem_int(&self, array: i64, index: usize, value: i64, descr: u32) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1535 setarrayitem_ref
    fn setarrayitem_ref(&self, array: i64, index: usize, value: i64, descr: u32) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1539 setarrayitem_float
    fn setarrayitem_float(&self, array: i64, index: usize, value: i64, descr: u32) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1452 allocate_raw_buffer
    fn allocate_raw_buffer(&self, size: usize) -> i64 {
        let _ = size;
        0
    }
    /// resume.py:1543 setrawbuffer_item
    fn setrawbuffer_item(&self, buffer: i64, offset: usize, value: i64, descr: u32) {
        let _ = (buffer, offset, value, descr);
    }
    /// resume.py:1509-1528 setfield — write field value at known offset.
    fn setfield_typed(
        &self,
        struct_ptr: i64,
        value: i64,
        descr: u32,
        field_offset: usize,
        field_size: usize,
    ) {
        let _ = (field_offset, field_size);
        self.setfield(struct_ptr, descr, value);
    }
    /// resume.py:1009-1015 setarrayitem dispatch by descr type.
    fn setarrayitem_typed(&self, array: i64, index: usize, value: i64, descr: u32) {
        self.setarrayitem_int(array, index, value, descr);
    }
    /// Pyre-specific: box a raw int to a PyObject ref.
    ///
    /// RPython equivalent: cpu.get_ref_value always returns GCREF because
    /// the jitframe stores typed values. Pyre's deadframe is untyped i64;
    /// when a slot typed as Int is read through decode_ref, this method
    /// wraps it into a valid GCREF (W_IntObject).
    fn box_int(&self, value: i64) -> i64 {
        value // default: return raw value (override in pyre allocator)
    }
    /// Pyre-specific: box raw float bits to a PyObject ref.
    fn box_float(&self, value: i64) -> i64 {
        value
    }
}

/// Default no-op allocator.
pub struct NullAllocator;
impl BlackholeAllocator for NullAllocator {}

impl VirtualInfo {
    /// resume.py:576 kind attribute — REF for object/struct/array/string,
    /// INT for raw buffers.
    pub fn is_about_raw(&self) -> bool {
        matches!(
            self,
            VirtualInfo::VRawBuffer { .. } | VirtualInfo::VRawSlice { .. }
        )
    }

    /// resume.py:618/634/650 allocate(decoder, index)
    ///
    /// Allocate a virtual object and fill in its fields from the decoder.
    /// Sets virtuals_cache_ptr[index] before filling fields (for recursive refs).
    pub fn allocate(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64 {
        match self {
            VirtualInfo::VirtualObj {
                type_id, fields, ..
            } => {
                let obj = allocator.allocate_with_vtable(*type_id);
                decoder.virtuals_cache_ptr[index] = obj;
                for (field_descr, source) in fields {
                    let value = decoder.decode_field_source(source);
                    allocator.setfield(obj, *field_descr, value);
                }
                obj
            }
            VirtualInfo::VStruct {
                type_id, fields, ..
            } => {
                // resume.py:632 allocate_struct(descr)
                // Use type_id for pyre's GC allocation dispatch.
                let obj = allocator.allocate_struct(*type_id);
                decoder.virtuals_cache_ptr[index] = obj;
                for (field_descr, source) in fields {
                    let value = decoder.decode_field_source(source);
                    allocator.setfield(obj, *field_descr, value);
                }
                obj
            }
            VirtualInfo::VArray { descr_index, items } => {
                let length = items.len();
                let array = allocator.allocate_array(length, *descr_index, true);
                decoder.virtuals_cache_ptr[index] = array;
                for (i, source) in items.iter().enumerate() {
                    let value = decoder.decode_field_source(source);
                    // resume.py:656-676 — dispatch by arraydescr type
                    allocator.setarrayitem_typed(array, i, value, *descr_index);
                }
                array
            }
            _ => {
                decoder.virtuals_cache_ptr[index] = 0;
                0
            }
        }
    }

    /// resume.py:701 VRawBufferInfo.allocate_int / VRawSliceInfo.allocate_int
    pub fn allocate_int(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64 {
        match self {
            VirtualInfo::VRawBuffer { size, entries, .. } => {
                // resume.py:703
                let buffer = allocator.allocate_raw_buffer(*size);
                // resume.py:704
                decoder.virtuals_cache_int[index] = buffer;
                // resume.py:705-708
                for (offset, _size_bytes, source) in entries {
                    let value = decoder.decode_field_source(source);
                    allocator.setrawbuffer_item(buffer, *offset, value, 0);
                }
                buffer
            }
            VirtualInfo::VRawSlice { offset, parent } => {
                // resume.py:723-725 — parent is an INT virtual (raw buffer)
                let parent_val = decoder.decode_field_source_int(parent);
                let result = parent_val + *offset;
                decoder.virtuals_cache_int[index] = result;
                result
            }
            _ => panic!("allocate_int called on non-raw virtual"),
        }
    }
}

impl<'a> ResumeDataDirectReader<'a> {
    /// resume.py:1364 __init__
    pub fn new(
        rd_numb: &'a [u8],
        rd_consts: &'a [i64],
        deadframe: &'a [i64],
        deadframe_types: Option<&'a [majit_ir::Type]>,
        all_virtuals: Option<(Vec<i64>, Vec<i64>)>,
        allocator: &'a dyn BlackholeAllocator,
    ) -> Self {
        // resume.py:915-922 _init
        let mut resumecodereader = Reader::new(rd_numb);
        let items_resume_section = resumecodereader.next_item();
        let count = resumecodereader.next_item();

        // resume.py:1368-1376
        let (resume_after_guard_not_forced, virtuals_cache_ptr, virtuals_cache_int) =
            if let Some((ptrs, ints)) = all_virtuals {
                // resume.py:1373-1374: special case for GUARD_NOT_FORCED
                (2, ptrs, ints)
            } else {
                (0, Vec::new(), Vec::new())
            };

        ResumeDataDirectReader {
            resumecodereader,
            items_resume_section,
            count,
            consts: rd_consts,
            deadframe,
            deadframe_types,
            resume_after_guard_not_forced,
            rd_virtuals: None,
            virtuals_cache_ptr,
            virtuals_cache_int,
            allocator,
            virtualizable_ptr: 0,
        }
    }

    /// resume.py:924 _prepare — init virtuals and pending fields.
    pub fn prepare(
        &mut self,
        rd_virtuals: Option<&'a [VirtualInfo]>,
        rd_pendingfields: Option<&[PendingFieldInfo]>,
        rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    ) {
        // resume.py:925
        self.prepare_virtuals(rd_virtuals);
        // resume.py:926
        if rd_pendingfields.is_some() {
            self.prepare_pendingfields(rd_pendingfields);
        } else if let Some(guard_pf) = rd_guard_pendingfields {
            self.prepare_guard_pendingfields(guard_pf);
        }
    }

    /// resume.py:993 _prepare_pendingfields
    fn prepare_pendingfields(&mut self, pendingfields: Option<&[PendingFieldInfo]>) {
        let Some(pendingfields) = pendingfields else {
            return;
        };
        // resume.py:995-1007
        for pf in pendingfields {
            // resume.py:1002: struct = self.decode_ref(num)
            let struct_ptr = match &pf.target {
                ResumeValueSource::FailArg(idx) => self.deadframe[*idx],
                ResumeValueSource::Constant(val) => *val,
                ResumeValueSource::Virtual(idx) => self.getvirtual_ptr(*idx),
                _ => 0,
            };
            // resume.py:1003-1007
            match pf.item_index {
                None => {
                    // resume.py:1004-1005: self.setfield(struct, fieldnum, descr)
                    let value = match &pf.value {
                        ResumeValueSource::FailArg(idx) => self.deadframe[*idx],
                        ResumeValueSource::Constant(val) => *val,
                        ResumeValueSource::Virtual(idx) => self.getvirtual_ptr(*idx),
                        _ => 0,
                    };
                    self.allocator
                        .setfield_typed(struct_ptr, value, pf.descr_index, 0, 0);
                }
                Some(item_index) => {
                    // resume.py:1007: self.setarrayitem(struct, itemindex, fieldnum, descr)
                    let value = match &pf.value {
                        ResumeValueSource::FailArg(idx) => self.deadframe[*idx],
                        ResumeValueSource::Constant(val) => *val,
                        ResumeValueSource::Virtual(idx) => self.getvirtual_ptr(*idx),
                        _ => 0,
                    };
                    // resume.py:1009-1015 setarrayitem: dispatch by descr type
                    self.allocator.setarrayitem_typed(
                        struct_ptr,
                        item_index,
                        value,
                        pf.descr_index,
                    );
                }
            }
        }
    }

    /// resume.py:993 _prepare_pendingfields — variant for GuardPendingFieldEntry.
    ///
    /// RPython encodes pendingfield target/value as tagged values (TAGBOX/
    /// TAGCONST/TAGVIRTUAL) via _gettagged in _add_pending_fields
    /// (resume.py:548-549). At restore time, decode_ref(num) resolves
    /// the tagged value to a concrete pointer.
    ///
    /// When target_tagged/value_tagged are not UNASSIGNED, we use the
    /// RPython tagged path (decode_ref/decode_int). When they are
    /// UNASSIGNED (not yet tagged), we use field_offset for direct
    /// memory writes via the allocator.
    fn prepare_guard_pendingfields(&mut self, pendingfields: &[majit_ir::GuardPendingFieldEntry]) {
        for pf in pendingfields {
            // resume.py:1002-1007
            let (struct_ptr, value) = if pf.target_tagged != UNASSIGNED
                && pf.value_tagged != UNASSIGNED
            {
                // RPython tagged path: decode_ref(num) for target,
                // then dispatch by field type for value.
                // resume.py:1002: struct = self.decode_ref(num)
                let s = self.decode_ref(pf.target_tagged);
                // resume.py:1004-1007: setfield/setarrayitem dispatch by descr type
                let v = match pf.field_type {
                    majit_ir::Type::Ref => self.decode_ref(pf.value_tagged),
                    majit_ir::Type::Float => self.decode_float(pf.value_tagged),
                    _ => self.decode_int(pf.value_tagged),
                };
                (s, v)
            } else {
                // Direct memory write via field_offset.
                // The Cranelift backend materializes pending fields as
                // raw stores at known offsets, so we delegate to the
                // allocator which knows the concrete memory layout.
                let s = if !pf.target.is_none() && (pf.target.0 as usize) < self.deadframe.len() {
                    self.deadframe[pf.target.0 as usize]
                } else {
                    0
                };
                let v = if !pf.value.is_none() && (pf.value.0 as usize) < self.deadframe.len() {
                    self.deadframe[pf.value.0 as usize]
                } else {
                    0
                };
                (s, v)
            };

            if pf.item_index < 0 {
                // resume.py:1005: self.setfield(struct, fieldnum, descr)
                self.allocator.setfield_typed(
                    struct_ptr,
                    value,
                    pf.descr_index,
                    pf.field_offset,
                    pf.field_size,
                );
            } else {
                // resume.py:1007: self.setarrayitem(struct, itemindex, fieldnum, descr)
                self.allocator.setarrayitem_typed(
                    struct_ptr,
                    pf.item_index as usize,
                    value,
                    pf.descr_index,
                );
            }
        }
    }

    /// resume.py:1378 handling_async_forcing
    pub fn handling_async_forcing(&mut self) {
        self.resume_after_guard_not_forced = 1;
    }

    // ---- AbstractResumeDataReader methods (resume.py:928-1038) ----

    /// resume.py:928 read_jitcode_pos_pc
    pub fn read_jitcode_pos_pc(&mut self) -> (i32, i32) {
        let jitcode_pos = self.resumecodereader.next_item();
        let pc = self.resumecodereader.next_item();
        (jitcode_pos, pc)
    }

    /// resume.py:933 next_int
    pub fn next_int(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_int(tagged)
    }

    /// resume.py:936 next_ref
    pub fn next_ref(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_ref(tagged)
    }

    /// resume.py:939 next_float
    pub fn next_float(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_float(tagged)
    }

    /// resume.py:1410-1421 load_next_value_of_type
    pub fn next_value_of_type(&mut self, tp: majit_ir::Type) -> i64 {
        match tp {
            majit_ir::Type::Int => self.next_int(),
            majit_ir::Type::Ref => self.next_ref(),
            majit_ir::Type::Float => self.next_float(),
            _ => self.next_int(),
        }
    }

    /// resume.py:942 done_reading
    pub fn done_reading(&self) -> bool {
        self.resumecodereader.items_read >= self.items_resume_section as usize
    }

    /// resume.py:945 getvirtual_ptr
    ///
    /// Returns the index'th virtual, building it lazily if needed.
    /// Note that this may be called recursively; that's why the
    /// allocate() methods must fill in the cache as soon as they
    /// have the object, before they fill its fields.
    pub fn getvirtual_ptr(&mut self, index: usize) -> i64 {
        // resume.py:950: assert self.virtuals_cache is not None
        assert!(
            !self.virtuals_cache_ptr.is_empty(),
            "getvirtual_ptr: virtuals_cache is empty (rd_virtuals not prepared)"
        );
        // resume.py:951-952
        let v = self.virtuals_cache_ptr[index];
        if v != 0 {
            return v;
        }
        // resume.py:953-955: lazy allocation
        assert!(self.rd_virtuals.is_some(), "rd_virtuals is None");
        // Safety: rd_virtuals is an immutable slice reference that we need to
        // read while mutating virtuals_cache through self. The slice data is
        // never modified by allocate(), only the cache vectors are written.
        let rd_virtuals_ptr = self.rd_virtuals.unwrap().as_ptr();
        let rd_virtuals_len = self.rd_virtuals.unwrap().len();
        let vinfo = unsafe { &*rd_virtuals_ptr.add(index) };
        debug_assert!(index < rd_virtuals_len);
        let allocator = self.allocator as *const dyn BlackholeAllocator;
        let v = vinfo.allocate(self, index, unsafe { &*allocator });
        debug_assert_eq!(v, self.virtuals_cache_ptr[index], "resume.py: bad cache");
        v
    }

    /// resume.py:958 getvirtual_int
    pub fn getvirtual_int(&mut self, index: usize) -> i64 {
        // resume.py:959: assert self.virtuals_cache is not None
        assert!(
            !self.virtuals_cache_int.is_empty(),
            "getvirtual_int: virtuals_cache is empty (rd_virtuals not prepared)"
        );
        // resume.py:960-961
        let v = self.virtuals_cache_int[index];
        if v != 0 {
            return v;
        }
        // resume.py:962-966
        assert!(self.rd_virtuals.is_some(), "rd_virtuals is None");
        let rd_virtuals_ptr = self.rd_virtuals.unwrap().as_ptr();
        let vinfo = unsafe { &*rd_virtuals_ptr.add(index) };
        assert!(vinfo.is_about_raw(), "getvirtual_int: not a raw virtual");
        let allocator = self.allocator as *const dyn BlackholeAllocator;
        let v = vinfo.allocate_int(self, index, unsafe { &*allocator });
        debug_assert_eq!(v, self.virtuals_cache_int[index], "resume.py: bad cache");
        v
    }

    /// resume.py:969 force_all_virtuals
    pub fn force_all_virtuals(&mut self) -> (&[i64], &[i64]) {
        if let Some(rd_virtuals) = self.rd_virtuals {
            for i in 0..rd_virtuals.len() {
                let rd_virtual = &rd_virtuals[i];
                if rd_virtual.is_about_raw() {
                    // resume.py:977: kind == INT
                    self.getvirtual_int(i);
                } else {
                    // resume.py:976: kind == REF
                    self.getvirtual_ptr(i);
                }
            }
        }
        (&self.virtuals_cache_ptr, &self.virtuals_cache_int)
    }

    /// resume.py:983 _prepare_virtuals
    fn prepare_virtuals(&mut self, virtuals: Option<&'a [VirtualInfo]>) {
        if let Some(v) = virtuals {
            self.rd_virtuals = Some(v);
            // resume.py:990-991
            self.virtuals_cache_ptr = vec![0; v.len()];
            self.virtuals_cache_int = vec![0; v.len()];
        }
    }

    // ---- ResumeDataDirectReader methods (resume.py:1380-1601) ----

    /// resume.py:1381 consume_one_section
    ///
    /// Read one resume frame section and fill the blackhole interpreter's
    /// registers. Uses liveness info from the jitcode to determine which
    /// registers of each type (int/ref/float) are live at the current PC.
    ///
    /// RPython:
    ///   self.blackholeinterp = blackholeinterp
    ///   info = blackholeinterp.get_current_position_info()
    ///   self._prepare_next_section(info)
    pub fn consume_one_section(&mut self, bh: &mut BlackholeInterpreter) {
        // resume.py:1382-1384
        if let Some(info) = bh.get_current_position_info().cloned() {
            // resume.py:1017-1026 _prepare_next_section
            // jitcode.py:146-167 enumerate_vars(info, all_liveness,
            //     callback_i, callback_r, callback_f, unique_id)

            // resume.py:1028-1030 _callback_i
            for &reg_idx in &info.live_i_regs {
                if self.done_reading() {
                    break;
                }
                let value = self.next_int();
                // resume.py:1590-1591 write_an_int
                bh.setarg_i(reg_idx as usize, value);
            }
            // resume.py:1032-1034 _callback_r
            for &reg_idx in &info.live_r_regs {
                if self.done_reading() {
                    break;
                }
                let value = self.next_ref();
                // resume.py:1593-1594 write_a_ref
                bh.setarg_r(reg_idx as usize, value);
            }
            // resume.py:1036-1038 _callback_f
            for &reg_idx in &info.live_f_regs {
                if self.done_reading() {
                    break;
                }
                let value = self.next_float();
                // resume.py:1596-1597 write_a_float
                bh.setarg_f(reg_idx as usize, value);
            }
        } else {
            // No liveness info at this PC. Fall back to reading all
            // registers sequentially — handles jitcodes generated
            // without precise liveness (e.g. the proc-macro path).
            let num_i = bh.jitcode.num_regs_i() as usize;
            let num_r = bh.jitcode.num_regs_r() as usize;
            let num_f = bh.jitcode.num_regs_f() as usize;
            for i in 0..num_i {
                if self.done_reading() {
                    break;
                }
                bh.setarg_i(i, self.next_int());
            }
            for i in 0..num_r {
                if self.done_reading() {
                    break;
                }
                bh.setarg_r(i, self.next_ref());
            }
            for i in 0..num_f {
                if self.done_reading() {
                    break;
                }
                bh.setarg_f(i, self.next_float());
            }
        }
    }

    /// resume.py:1386 consume_virtualref_info
    pub fn consume_virtualref_info(&mut self, vrefinfo: Option<&dyn VRefInfo>) {
        // resume.py:1389
        let size = self.resumecodereader.next_item();
        // resume.py:1390-1391
        if vrefinfo.is_none() || size == 0 {
            // resume.py:1391: assert size == 0
            assert!(
                size == 0,
                "consume_virtualref_info: vrefinfo is None but size={size} != 0"
            );
            return;
        }
        let vrefinfo = vrefinfo.unwrap();
        // resume.py:1393-1397
        for _i in 0..size {
            let virtual_val = self.next_ref();
            let vref = self.next_ref();
            // resume.py:1397
            vrefinfo.continue_tracing(vref, virtual_val);
        }
    }

    /// resume.py:1399 consume_vable_info
    pub fn consume_vable_info(&mut self, vinfo: &dyn VirtualizableInfo, vable_size: i32) {
        // resume.py:1403
        assert!(vable_size > 0);
        // resume.py:1404
        let virtualizable = self.next_ref();
        self.virtualizable_ptr = virtualizable;
        // resume.py:1406
        let expected = vinfo.get_total_size(virtualizable) as i32;
        // resume.py:1406: assert vinfo.get_total_size(virtualizable) == vable_size - 1
        assert_eq!(
            expected,
            vable_size - 1,
            "consume_vable_info: vable size mismatch: expected={} actual={}",
            expected,
            vable_size - 1,
        );
        // resume.py:1407
        vinfo.reset_token_gcref(virtualizable);
        // resume.py:1408
        vinfo.write_from_resume_data_partial(virtualizable, self);
    }

    /// resume.py:1424 consume_vref_and_vable
    pub fn consume_vref_and_vable(
        &mut self,
        vrefinfo: Option<&dyn VRefInfo>,
        vinfo: Option<&dyn VirtualizableInfo>,
        ginfo: Option<&dyn GreenfieldInfo>,
    ) {
        // resume.py:1425
        let vable_size = self.resumecodereader.next_item();

        if self.resume_after_guard_not_forced != 2 {
            // resume.py:1427-1428
            if let Some(vi) = vinfo {
                if vable_size > 0 {
                    self.consume_vable_info(vi, vable_size);
                }
            }
            // resume.py:1429-1430
            if ginfo.is_some() {
                let _ginfo_item = self.resumecodereader.next_item();
            }
            // resume.py:1431
            self.consume_virtualref_info(vrefinfo);
        } else {
            // resume.py:1433-1435
            self.resumecodereader.jump(vable_size as usize);
            let vref_size = self.resumecodereader.next_item();
            self.resumecodereader.jump(vref_size as usize * 2);
        }
    }

    /// resume.py:1552 decode_int
    pub fn decode_int(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1555
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx]
            }
            TAGINT => {
                // resume.py:1557
                num as i64
            }
            TAGVIRTUAL => {
                // resume.py:1559
                self.getvirtual_int(num as usize)
            }
            TAGBOX => {
                // resume.py:1561-1564
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                self.deadframe[idx as usize]
            }
            _ => unreachable!("bad tag: {tag}"),
        }
    }

    /// resume.py:1566 decode_ref
    pub fn decode_ref(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1569-1571
                if tagged_eq(tagged, NULLREF) {
                    return 0; // ConstPtr.value (null pointer)
                }
                // resume.py:1571
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx]
            }
            TAGVIRTUAL => {
                // resume.py:1573
                self.getvirtual_ptr(num as usize)
            }
            TAGBOX => {
                // resume.py:1575-1578
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                let value = self.deadframe[idx as usize];
                match self
                    .deadframe_types
                    .and_then(|tys| tys.get(idx as usize))
                    .copied()
                    .unwrap_or(majit_ir::Type::Ref)
                {
                    majit_ir::Type::Ref => value,
                    majit_ir::Type::Int => self.allocator.box_int(value),
                    majit_ir::Type::Float => self.allocator.box_float(value),
                    majit_ir::Type::Void => value,
                }
            }
            TAGINT => {
                // pyre parity: all values are in ref registers (no typed
                // register files). Optimizer may unbox Ref→Int, producing
                // TAGINT in snapshot numbering for a ref-register slot.
                // Box the int back to a PyObject because ref registers
                // store pointers. The allocator handles boxing.
                self.allocator.box_int(num as i64)
            }
            _ => {
                panic!("decode_ref: unexpected tag {tag}")
            }
        }
    }

    /// resume.py:1580 decode_float
    pub fn decode_float(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1583
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx]
            }
            TAGBOX => {
                // resume.py:1585-1588
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                self.deadframe[idx as usize]
            }
            _ => {
                // resume.py:1580 — only TAGCONST and TAGBOX valid for floats
                panic!("decode_float: unexpected tag {tag}")
            }
        }
    }

    /// Decode a VirtualFieldSource as a REF value (resume.py:1566 decode_ref).
    ///
    /// Virtual sources go through getvirtual_ptr (REF virtuals).
    pub fn decode_field_source(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe[*index],
            ResumeValueSource::Constant(value) => *value,
            ResumeValueSource::Virtual(index) => self.getvirtual_ptr(*index),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// Decode a VirtualFieldSource as an INT value (resume.py:1552 decode_int).
    ///
    /// Virtual sources go through getvirtual_int (INT/raw virtuals).
    pub fn decode_field_source_int(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe[*index],
            ResumeValueSource::Constant(value) => *value,
            ResumeValueSource::Virtual(index) => self.getvirtual_int(*index),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// resume.py:1599 int_add_const
    pub fn int_add_const(&self, base: i64, offset: i64) -> i64 {
        base + offset
    }
}

/// resume.py:1312 blackhole_from_resumedata
///
/// Build a chain of BlackholeInterpreters from encoded resume data.
/// Returns the topmost (innermost) interpreter.
///
/// `resolve_jitcode` corresponds to RPython's `jitcodes[jitcode_pos]` lookup.
/// In RPython, `metainterp_sd.jitcodes` is a pre-compiled array.
/// In majit, this is typically backed by jitcode_factory or a pre-built table.
pub fn blackhole_from_resumedata<'a>(
    builder: &mut crate::blackhole::BlackholeInterpBuilder,
    resolve_jitcode: &dyn Fn(i32, i32) -> Option<(crate::jitcode::JitCode, usize)>,
    rd_numb: &'a [u8],
    rd_consts: &'a [i64],
    deadframe: &'a [i64],
    deadframe_types: Option<&'a [majit_ir::Type]>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_pendingfields: Option<&[PendingFieldInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    vrefinfo: Option<&dyn VRefInfo>,
    vinfo: Option<&dyn VirtualizableInfo>,
    ginfo: Option<&dyn GreenfieldInfo>,
    allocator: &'a dyn BlackholeAllocator,
) -> Option<(BlackholeInterpreter, i64)> {
    // resume.py:1317-1321
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        deadframe,
        deadframe_types,
        None,
        allocator,
    );

    // resume.py:1324
    resumereader.prepare(rd_virtuals, rd_pendingfields, rd_guard_pendingfields);

    // resume.py:1325
    resumereader.consume_vref_and_vable(vrefinfo, vinfo, ginfo);

    // resume.py:1404: virtualizable pointer read by consume_vable_info.
    let virtualizable_ptr = resumereader.virtualizable_ptr;

    // resume.py:1332-1343
    // Build chain bottom-up: first frame acquired is the outermost.
    let mut curbh: Option<Box<BlackholeInterpreter>> = None;

    while !resumereader.done_reading() {
        // resume.py:1334-1336
        let mut nextbh = builder.acquire_interp();
        nextbh.nextblackholeinterp = curbh;

        // resume.py:1338-1340
        let (jitcode_pos, pc) = resumereader.read_jitcode_pos_pc();
        // resume.py:1339: jitcode = jitcodes[jitcode_pos]
        // resolve_jitcode returns (jitcode, actual_pc) where actual_pc
        // is the JitCode byte offset (may differ from the rd_numb pc
        // when rd_numb encodes Python PCs).
        let (jitcode, actual_pc) = resolve_jitcode(jitcode_pos, pc)?;
        nextbh.setposition(jitcode, actual_pc);

        // resume.py:1341
        resumereader.consume_one_section(&mut nextbh);

        // resume.py:1342 — handle_rvmprof_enter (not applicable in majit)

        curbh = Some(Box::new(nextbh));
    }

    curbh.map(|b| (*b, virtualizable_ptr))
}

/// resume.py:1345 force_from_resumedata
///
/// Force all virtuals from resume data without running a blackhole.
/// Used for GUARD_NOT_FORCED handling.
/// Returns (virtuals_cache_ptr, virtuals_cache_int) — RPython VirtualCache parity.
pub fn force_from_resumedata<'a>(
    rd_numb: &'a [u8],
    rd_consts: &'a [i64],
    deadframe: &'a [i64],
    deadframe_types: Option<&'a [majit_ir::Type]>,
    vrefinfo: Option<&dyn VRefInfo>,
    vinfo: Option<&dyn VirtualizableInfo>,
    ginfo: Option<&dyn GreenfieldInfo>,
    allocator: &'a dyn BlackholeAllocator,
) -> (Vec<i64>, Vec<i64>) {
    // resume.py:1347-1348
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        deadframe,
        deadframe_types,
        None,
        allocator,
    );
    resumereader.handling_async_forcing();
    // resume.py:1350
    resumereader.consume_vref_and_vable(vrefinfo, vinfo, ginfo);
    // resume.py:1351: return resumereader.force_all_virtuals()
    let (ptrs, ints) = resumereader.force_all_virtuals();
    (ptrs.to_vec(), ints.to_vec())
}
