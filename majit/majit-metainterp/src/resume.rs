//! Resume data: encodes the mapping from guard fail_args to full interpreter state.
//!
//! When a guard fails, the JIT needs to reconstruct the interpreter's full state
//! (program counter, local variables, stack contents) from the values stored in
//! the DeadFrame. Resume data provides this mapping.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/resume.py`.

use std::collections::HashMap;

use majit_codegen::{
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

// resume.py:134-139
pub struct NumberingState {
    pub writer: crate::resumecode::Writer,
    pub liveboxes: HashMap<u32, i16>,
    pub num_boxes: i32,
    pub num_virtuals: i32,
}

impl NumberingState {
    pub fn new(size: usize) -> Self {
        NumberingState {
            writer: crate::resumecode::Writer::new(size),
            liveboxes: HashMap::new(),
            num_boxes: 0,
            num_virtuals: 0,
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
}

impl SimpleBoxEnv {
    pub fn new() -> Self {
        SimpleBoxEnv {
            constants: HashMap::new(),
            replacements: HashMap::new(),
            types: HashMap::new(),
            virtuals: std::collections::HashSet::new(),
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
}

// ═══════════════════════════════════════════════════════════════
// Legacy i64 tags — used by existing EncodedResumeData code.
// Will be replaced as Phases 3-5 migrate to i16 tags above.
// ═══════════════════════════════════════════════════════════════
const LEGACY_TAGMASK: i64 = 0b11;
const LEGACY_TAGCONST: i64 = 0;
const LEGACY_TAGINT: i64 = 1;
const LEGACY_TAGBOX: i64 = 2;
const LEGACY_TAGVIRTUAL: i64 = 3;
const ENCODED_UNINITIALIZED: i64 = -2;
const ENCODED_UNAVAILABLE: i64 = -3;

// Two low bits are reserved for the tag.
const INLINE_TAGGED_MIN: i64 = -(1_i64 << 61);
const INLINE_TAGGED_MAX: i64 = (1_i64 << 61) - 1;

/// Compact resume snapshot using RPython-style tagged numbering.
///
/// `code` is a flat encoded section containing:
/// 1. total item count
/// 2. number of fail args required by the snapshot
/// 3. number of frames
/// 4. per-frame `(pc, slot_count, slot_sources...)`
/// 5. number of virtuals
/// 6. per-virtual encoded metadata and sources
///
/// The semantic `ResumeData` view is still exposed separately, but
/// reconstruction goes through this encoded representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedResumeData {
    /// Flat encoded numbering section.
    pub code: Vec<i64>,
    /// Shared constant pool for CONST-tagged entries.
    pub consts: Vec<i64>,
    /// Number of compact fail-arg slots referenced by the encoded section.
    pub num_fail_args: usize,
    /// Mapping from compact fail-arg numbering back to the raw guard fail-arg slots.
    pub fail_arg_positions: Vec<usize>,
    /// Pending field/array writes that must be replayed on resume.
    pub pending_fields: Vec<EncodedPendingFieldWrite>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecodedResumeLayout {
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
    pub fail_arg_index: Option<usize>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeVirtualLayoutSummary {
    Object {
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
    },
    Struct {
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
    },
    Array {
        descr_index: u32,
        items: Vec<ResumeValueLayoutSummary>,
    },
    ArrayStruct {
        descr_index: u32,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeLayoutSummary {
    pub num_frames: usize,
    pub frame_pcs: Vec<u64>,
    pub frame_slot_counts: Vec<usize>,
    pub frame_layouts: Vec<ResumeFrameLayoutSummary>,
    pub num_virtuals: usize,
    pub virtual_kinds: Vec<ResumeVirtualKind>,
    pub virtual_layouts: Vec<ResumeVirtualLayoutSummary>,
    pub num_fail_args: usize,
    pub fail_arg_positions: Vec<usize>,
    pub pending_field_count: usize,
    pub pending_field_layouts: Vec<PendingFieldLayoutSummary>,
    pub const_pool_size: usize,
}

impl ResumeValueLayoutSummary {
    pub(crate) fn raw_fail_arg_position(&self, fail_arg_positions: &[usize]) -> usize {
        if let Some(position) = self.raw_fail_arg_position {
            return position;
        }
        let compact_index = self
            .fail_arg_index
            .expect("resume layout missing fail-arg index for FailArg source");
        *fail_arg_positions
            .get(compact_index)
            .expect("resume layout compact fail-arg index out of bounds")
    }

    fn to_resume_source(&self, fail_arg_positions: &[usize]) -> ResumeValueSource {
        match self.kind {
            ResumeValueKind::FailArg => {
                ResumeValueSource::FailArg(self.raw_fail_arg_position(fail_arg_positions))
            }
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

    fn to_exit_source(
        &self,
        fail_arg_positions: &[usize],
        virtual_offset: usize,
    ) -> ExitValueSourceLayout {
        match self.kind {
            ResumeValueKind::FailArg => {
                ExitValueSourceLayout::ExitValue(self.raw_fail_arg_position(fail_arg_positions))
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
    fn to_frame_info(&self, fail_arg_positions: &[usize]) -> FrameInfo {
        FrameInfo {
            pc: self.pc,
            slot_map: self
                .slot_layouts
                .iter()
                .map(|slot| slot.to_resume_source(fail_arg_positions))
                .collect(),
        }
    }

    fn to_exit_frame_layout(
        &self,
        fail_arg_positions: &[usize],
        virtual_offset: usize,
    ) -> ExitFrameLayout {
        ExitFrameLayout {
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            source_guard: self.source_guard,
            pc: self.pc,
            slots: self
                .slot_layouts
                .iter()
                .map(|slot| slot.to_exit_source(fail_arg_positions, virtual_offset))
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
                fail_arg_index: Some(*index),
                raw_fail_arg_position: Some(*index),
                constant: None,
                virtual_index: None,
            },
            ExitValueSourceLayout::Constant(value) => Self {
                kind: ResumeValueKind::Constant,
                fail_arg_index: None,
                raw_fail_arg_position: None,
                constant: Some(*value),
                virtual_index: None,
            },
            ExitValueSourceLayout::Virtual(index) => Self {
                kind: ResumeValueKind::Virtual,
                fail_arg_index: None,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: Some(*index),
            },
            ExitValueSourceLayout::Uninitialized => Self {
                kind: ResumeValueKind::Uninitialized,
                fail_arg_index: None,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: None,
            },
            ExitValueSourceLayout::Unavailable => Self {
                kind: ResumeValueKind::Unavailable,
                fail_arg_index: None,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: None,
            },
        }
    }
}

impl ResumeVirtualLayoutSummary {
    fn to_virtual_info(&self, fail_arg_positions: &[usize]) -> VirtualInfo {
        match self {
            ResumeVirtualLayoutSummary::Object {
                type_id,
                descr_index,
                fields,
            } => VirtualInfo::VirtualObj {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (*field_descr, source.to_resume_source(fail_arg_positions))
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::Struct {
                type_id,
                descr_index,
                fields,
            } => VirtualInfo::VStruct {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (*field_descr, source.to_resume_source(fail_arg_positions))
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::Array { descr_index, items } => VirtualInfo::VArray {
                descr_index: *descr_index,
                items: items
                    .iter()
                    .map(|source| source.to_resume_source(fail_arg_positions))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                descr_index,
                element_fields,
            } => VirtualInfo::VArrayStruct {
                descr_index: *descr_index,
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| {
                                (*field_descr, source.to_resume_source(fail_arg_positions))
                            })
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer { size, entries } => VirtualInfo::VRawBuffer {
                size: *size,
                entries: entries
                    .iter()
                    .map(|(offset, size_in_bytes, source)| {
                        (
                            *offset,
                            *size_in_bytes,
                            source.to_resume_source(fail_arg_positions),
                        )
                    })
                    .collect(),
            },
        }
    }

    fn to_exit_virtual_layout(
        &self,
        fail_arg_positions: &[usize],
        virtual_offset: usize,
    ) -> ExitVirtualLayout {
        match self {
            ResumeVirtualLayoutSummary::Object {
                type_id,
                descr_index,
                fields,
            } => ExitVirtualLayout::Object {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (
                            *field_descr,
                            source.to_exit_source(fail_arg_positions, virtual_offset),
                        )
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::Struct {
                type_id,
                descr_index,
                fields,
            } => ExitVirtualLayout::Struct {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (
                            *field_descr,
                            source.to_exit_source(fail_arg_positions, virtual_offset),
                        )
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::Array { descr_index, items } => ExitVirtualLayout::Array {
                descr_index: *descr_index,
                items: items
                    .iter()
                    .map(|source| source.to_exit_source(fail_arg_positions, virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                descr_index,
                element_fields,
            } => ExitVirtualLayout::ArrayStruct {
                descr_index: *descr_index,
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| {
                                (
                                    *field_descr,
                                    source.to_exit_source(fail_arg_positions, virtual_offset),
                                )
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
                                source.to_exit_source(fail_arg_positions, virtual_offset),
                            )
                        })
                        .collect(),
                }
            }
        }
    }
}

impl PendingFieldLayoutSummary {
    fn to_pending_field_info(&self, fail_arg_positions: &[usize]) -> PendingFieldInfo {
        PendingFieldInfo {
            descr_index: self.descr_index,
            target: self.target.to_resume_source(fail_arg_positions),
            value: self.value.to_resume_source(fail_arg_positions),
            item_index: self.item_index,
        }
    }

    fn to_exit_pending_field_layout(
        &self,
        fail_arg_positions: &[usize],
        virtual_offset: usize,
    ) -> ExitPendingFieldLayout {
        ExitPendingFieldLayout {
            descr_index: self.descr_index,
            item_index: self.item_index,
            is_array_item: self.is_array_item,
            target: self
                .target
                .to_exit_source(fail_arg_positions, virtual_offset),
            value: self
                .value
                .to_exit_source(fail_arg_positions, virtual_offset),
            field_offset: 0,
            field_size: 0,
            field_type: majit_ir::Type::Int,
        }
    }
}

impl ResumeLayoutSummary {
    pub fn to_resume_data(&self) -> ResumeData {
        ResumeData {
            frames: self
                .frame_layouts
                .iter()
                .map(|frame| frame.to_frame_info(&self.fail_arg_positions))
                .collect(),
            virtuals: self
                .virtual_layouts
                .iter()
                .map(|virt| virt.to_virtual_info(&self.fail_arg_positions))
                .collect(),
            pending_fields: self
                .pending_field_layouts
                .iter()
                .map(|pending| pending.to_pending_field_info(&self.fail_arg_positions))
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
        let mut virtual_layouts = if preserve_prefix {
            caller_prefix
                .map(|layout| layout.virtual_layouts.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
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
                .map(|frame| frame.to_exit_frame_layout(&self.fail_arg_positions, virtual_offset)),
        );
        virtual_layouts.extend(
            self.virtual_layouts
                .iter()
                .map(|virt| virt.to_exit_virtual_layout(&self.fail_arg_positions, virtual_offset)),
        );
        pending_field_layouts.extend(self.pending_field_layouts.iter().map(|pending| {
            pending.to_exit_pending_field_layout(&self.fail_arg_positions, virtual_offset)
        }));

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
                slot_types: frame.slot_types.clone(),
                values: frame
                    .slot_layouts
                    .iter()
                    .map(|slot| {
                        ResumeData::resolve_frame_slot_source(
                            &slot.to_resume_source(&self.fail_arg_positions),
                            fail_values,
                        )
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
            slot_types: frame.slot_types.clone(),
            values: frame
                .slot_layouts
                .iter()
                .map(|slot| {
                    ResumeData::resolve_frame_slot_source(
                        &slot.to_resume_source(&self.fail_arg_positions),
                        fail_values,
                    )
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

    pub fn compact_fail_values(&self, raw_fail_values: &[i64]) -> Vec<i64> {
        self.fail_arg_positions
            .iter()
            .map(|&raw_index| raw_fail_values.get(raw_index).copied().unwrap_or(0))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncodedVirtualKind {
    VirtualObj = 0,
    VStruct = 1,
    VArray = 2,
    VArrayStruct = 3,
    VRawBuffer = 4,
}

impl EncodedVirtualKind {
    fn from_word(word: i64) -> Self {
        match word {
            0 => EncodedVirtualKind::VirtualObj,
            1 => EncodedVirtualKind::VStruct,
            2 => EncodedVirtualKind::VArray,
            3 => EncodedVirtualKind::VArrayStruct,
            4 => EncodedVirtualKind::VRawBuffer,
            other => panic!("unknown encoded virtual kind {other}"),
        }
    }
}

fn can_inline_tagged(value: i64) -> bool {
    (INLINE_TAGGED_MIN..=INLINE_TAGGED_MAX).contains(&value)
}

/// Legacy i64 tag — used by existing EncodedResumeData code.
fn tag_value(value: i64, tagbits: i64) -> i64 {
    debug_assert!((LEGACY_TAGCONST..=LEGACY_TAGVIRTUAL).contains(&tagbits));
    debug_assert!(
        can_inline_tagged(value),
        "tagged resume value {value} exceeds inline range"
    );
    (value << 2) | tagbits
}

/// Legacy i64 untag — used by existing EncodedResumeData code.
fn untag_value(encoded: i64) -> (i64, i64) {
    (encoded >> 2, encoded & LEGACY_TAGMASK)
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

    pub fn layout_summary(&self, fail_arg_positions: &[usize]) -> ResumeValueLayoutSummary {
        match self {
            ResumeValueSource::FailArg(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::FailArg,
                fail_arg_index: fail_arg_positions.iter().position(|pos| pos == index),
                raw_fail_arg_position: Some(*index),
                constant: None,
                virtual_index: None,
            },
            ResumeValueSource::Constant(value) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Constant,
                fail_arg_index: None,
                raw_fail_arg_position: None,
                constant: Some(*value),
                virtual_index: None,
            },
            ResumeValueSource::Virtual(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Virtual,
                fail_arg_index: None,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: Some(*index),
            },
            ResumeValueSource::Uninitialized => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Uninitialized,
                fail_arg_index: None,
                raw_fail_arg_position: None,
                constant: None,
                virtual_index: None,
            },
            ResumeValueSource::Unavailable => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Unavailable,
                fail_arg_index: None,
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
/// - VArrayInfo (NEW_ARRAY)
/// - VArrayStructInfo (array of structs with interior fields)
/// - VRawBufferInfo (raw memory buffer)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VirtualInfo {
    /// Virtual object with vtable (from NEW_WITH_VTABLE).
    VirtualObj {
        /// Type ID (vtable pointer or class identifier).
        type_id: u32,
        /// Descriptor index for the allocation.
        descr_index: u32,
        /// Field values: (field_descr_index, source).
        fields: Vec<(u32, VirtualFieldSource)>,
    },
    /// Virtual struct without vtable (from NEW).
    VStruct {
        /// Type ID.
        type_id: u32,
        /// Descriptor index.
        descr_index: u32,
        /// Field values: (field_descr_index, source).
        fields: Vec<(u32, VirtualFieldSource)>,
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

impl VirtualInfo {
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

    pub fn layout_summary(&self, fail_arg_positions: &[usize]) -> ResumeVirtualLayoutSummary {
        match self {
            VirtualInfo::VirtualObj {
                type_id,
                descr_index,
                fields,
            } => ResumeVirtualLayoutSummary::Object {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (*field_descr, source.layout_summary(fail_arg_positions))
                    })
                    .collect(),
            },
            VirtualInfo::VStruct {
                type_id,
                descr_index,
                fields,
            } => ResumeVirtualLayoutSummary::Struct {
                type_id: *type_id,
                descr_index: *descr_index,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (*field_descr, source.layout_summary(fail_arg_positions))
                    })
                    .collect(),
            },
            VirtualInfo::VArray { descr_index, items } => ResumeVirtualLayoutSummary::Array {
                descr_index: *descr_index,
                items: items
                    .iter()
                    .map(|source| source.layout_summary(fail_arg_positions))
                    .collect(),
            },
            VirtualInfo::VArrayStruct {
                descr_index,
                element_fields,
            } => ResumeVirtualLayoutSummary::ArrayStruct {
                descr_index: *descr_index,
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| {
                                (*field_descr, source.layout_summary(fail_arg_positions))
                            })
                            .collect()
                    })
                    .collect(),
            },
            VirtualInfo::VRawBuffer { size, entries } => ResumeVirtualLayoutSummary::RawBuffer {
                size: *size,
                entries: entries
                    .iter()
                    .map(|(offset, size_in_bytes, source)| {
                        (
                            *offset,
                            *size_in_bytes,
                            source.layout_summary(fail_arg_positions),
                        )
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
                type_id: 0,
                descr_index: 0,
                fields: vec![],
            },
        }
    }
}

/// Source of a virtual object's field value.
pub type VirtualFieldSource = ResumeValueSource;

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
    pub fn layout_summary(&self, fail_arg_positions: &[usize]) -> PendingFieldLayoutSummary {
        PendingFieldLayoutSummary {
            descr_index: self.descr_index,
            item_index: self.item_index,
            is_array_item: self.item_index.is_some(),
            target_kind: self.target.kind(),
            value_kind: self.value.kind(),
            target: self.target.layout_summary(fail_arg_positions),
            value: self.value.layout_summary(fail_arg_positions),
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
        Self::from_semantic(&rd.frames, &rd.virtuals, &rd.pending_fields)
    }

    fn from_semantic(
        frames: &[FrameInfo],
        virtuals: &[VirtualInfo],
        pending_fields: &[PendingFieldInfo],
    ) -> Self {
        let mut code = vec![0, 0, encode_len(frames.len())];
        let mut consts = Vec::new();
        let mut const_indices = HashMap::new();
        let mut num_fail_args = 0usize;
        let mut fail_arg_positions = Vec::new();
        let mut fail_arg_indices = HashMap::new();
        for frame in frames {
            code.push(encode_u64(frame.pc));
            code.push(encode_len(frame.slot_map.len()));
            for source in &frame.slot_map {
                let encoded = Self::encode_source(
                    source,
                    &mut consts,
                    &mut const_indices,
                    &mut num_fail_args,
                    &mut fail_arg_positions,
                    &mut fail_arg_indices,
                );
                code.push(encoded);
            }
        }
        code.push(encode_len(virtuals.len()));
        for virtual_info in virtuals {
            Self::encode_virtual(
                virtual_info,
                &mut code,
                &mut consts,
                &mut const_indices,
                &mut num_fail_args,
                &mut fail_arg_positions,
                &mut fail_arg_indices,
            );
        }
        let pending_fields = pending_fields
            .iter()
            .map(|pending| EncodedPendingFieldWrite {
                descr_index: pending.descr_index,
                target: Self::encode_source(
                    &pending.target,
                    &mut consts,
                    &mut const_indices,
                    &mut num_fail_args,
                    &mut fail_arg_positions,
                    &mut fail_arg_indices,
                ),
                value: Self::encode_source(
                    &pending.value,
                    &mut consts,
                    &mut const_indices,
                    &mut num_fail_args,
                    &mut fail_arg_positions,
                    &mut fail_arg_indices,
                ),
                item_index: pending.item_index,
            })
            .collect();
        code[0] = encode_len(code.len());
        code[1] = encode_len(num_fail_args);
        EncodedResumeData {
            code,
            consts,
            num_fail_args,
            fail_arg_positions,
            pending_fields,
        }
    }

    fn encode_source(
        source: &ResumeValueSource,
        consts: &mut Vec<i64>,
        const_indices: &mut HashMap<i64, usize>,
        num_fail_args: &mut usize,
        fail_arg_positions: &mut Vec<usize>,
        fail_arg_indices: &mut HashMap<usize, usize>,
    ) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => {
                let compact_index = *fail_arg_indices.entry(*index).or_insert_with(|| {
                    let next_index = fail_arg_positions.len();
                    fail_arg_positions.push(*index);
                    next_index
                });
                *num_fail_args = fail_arg_positions.len();
                tag_value(encode_len(compact_index), LEGACY_TAGBOX)
            }
            ResumeValueSource::Constant(value) if can_inline_tagged(*value) => {
                tag_value(*value, LEGACY_TAGINT)
            }
            ResumeValueSource::Constant(value) => {
                let next_index = consts.len();
                let index = *const_indices.entry(*value).or_insert_with(|| {
                    consts.push(*value);
                    next_index
                });
                tag_value(encode_len(index), LEGACY_TAGCONST)
            }
            ResumeValueSource::Virtual(index) => tag_value(encode_len(*index), LEGACY_TAGVIRTUAL),
            ResumeValueSource::Uninitialized => tag_value(ENCODED_UNINITIALIZED, LEGACY_TAGCONST),
            ResumeValueSource::Unavailable => tag_value(ENCODED_UNAVAILABLE, LEGACY_TAGCONST),
        }
    }

    fn encode_virtual(
        virtual_info: &VirtualInfo,
        code: &mut Vec<i64>,
        consts: &mut Vec<i64>,
        const_indices: &mut HashMap<i64, usize>,
        num_fail_args: &mut usize,
        fail_arg_positions: &mut Vec<usize>,
        fail_arg_indices: &mut HashMap<usize, usize>,
    ) {
        match virtual_info {
            VirtualInfo::VirtualObj {
                type_id,
                descr_index,
                fields,
            } => {
                code.push(EncodedVirtualKind::VirtualObj as i64);
                code.push(i64::from(*type_id));
                code.push(i64::from(*descr_index));
                code.push(encode_len(fields.len()));
                for (field_index, source) in fields {
                    code.push(i64::from(*field_index));
                    code.push(Self::encode_source(
                        source,
                        consts,
                        const_indices,
                        num_fail_args,
                        fail_arg_positions,
                        fail_arg_indices,
                    ));
                }
            }
            VirtualInfo::VStruct {
                type_id,
                descr_index,
                fields,
            } => {
                code.push(EncodedVirtualKind::VStruct as i64);
                code.push(i64::from(*type_id));
                code.push(i64::from(*descr_index));
                code.push(encode_len(fields.len()));
                for (field_index, source) in fields {
                    code.push(i64::from(*field_index));
                    code.push(Self::encode_source(
                        source,
                        consts,
                        const_indices,
                        num_fail_args,
                        fail_arg_positions,
                        fail_arg_indices,
                    ));
                }
            }
            VirtualInfo::VArray { descr_index, items } => {
                code.push(EncodedVirtualKind::VArray as i64);
                code.push(i64::from(*descr_index));
                code.push(encode_len(items.len()));
                for source in items {
                    code.push(Self::encode_source(
                        source,
                        consts,
                        const_indices,
                        num_fail_args,
                        fail_arg_positions,
                        fail_arg_indices,
                    ));
                }
            }
            VirtualInfo::VArrayStruct {
                descr_index,
                element_fields,
            } => {
                code.push(EncodedVirtualKind::VArrayStruct as i64);
                code.push(i64::from(*descr_index));
                code.push(encode_len(element_fields.len()));
                for element in element_fields {
                    code.push(encode_len(element.len()));
                    for (field_index, source) in element {
                        code.push(i64::from(*field_index));
                        code.push(Self::encode_source(
                            source,
                            consts,
                            const_indices,
                            num_fail_args,
                            fail_arg_positions,
                            fail_arg_indices,
                        ));
                    }
                }
            }
            VirtualInfo::VRawBuffer { size, entries } => {
                code.push(EncodedVirtualKind::VRawBuffer as i64);
                code.push(encode_len(*size));
                code.push(encode_len(entries.len()));
                for (offset, size_in_bytes, source) in entries {
                    code.push(encode_len(*offset));
                    code.push(encode_len(*size_in_bytes));
                    code.push(Self::encode_source(
                        source,
                        consts,
                        const_indices,
                        num_fail_args,
                        fail_arg_positions,
                        fail_arg_indices,
                    ));
                }
            }
            // String/unicode/raw-slice virtuals: encode as empty struct
            // (reconstruction uses the VirtualInfo directly, not encoded form)
            VirtualInfo::VRawSlice { .. }
            | VirtualInfo::VStrPlain { .. }
            | VirtualInfo::VStrConcat { .. }
            | VirtualInfo::VStrSlice { .. }
            | VirtualInfo::VUniPlain { .. }
            | VirtualInfo::VUniConcat { .. }
            | VirtualInfo::VUniSlice { .. } => {
                code.push(EncodedVirtualKind::VStruct as i64);
                code.push(0); // type_id
                code.push(0); // descr_index
                code.push(0); // num fields
            }
        }
    }

    fn decode_layout(&self) -> DecodedResumeLayout {
        let mut cursor = 0usize;
        let encoded_items = self.next_word(&mut cursor);
        assert_eq!(
            decode_len(encoded_items),
            self.code.len(),
            "resume item count mismatch"
        );
        let encoded_fail_args = self.next_word(&mut cursor);
        assert_eq!(
            decode_len(encoded_fail_args),
            self.num_fail_args,
            "resume fail-arg count mismatch"
        );
        assert_eq!(
            self.fail_arg_positions.len(),
            self.num_fail_args,
            "resume fail-arg position map mismatch"
        );

        let num_frames = decode_len(self.next_word(&mut cursor));
        let mut frames = Vec::with_capacity(num_frames);
        for _ in 0..num_frames {
            let pc = decode_u64(self.next_word(&mut cursor));
            let slot_count = decode_len(self.next_word(&mut cursor));
            let mut slot_map = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                slot_map.push(self.decode_source(self.next_word(&mut cursor)));
            }
            frames.push(FrameInfo { pc, slot_map });
        }

        let num_virtuals = decode_len(self.next_word(&mut cursor));
        let mut virtuals = Vec::with_capacity(num_virtuals);
        for _ in 0..num_virtuals {
            virtuals.push(self.decode_virtual(&mut cursor));
        }

        assert_eq!(cursor, self.code.len(), "resume decoder left trailing data");
        let pending_fields = self
            .pending_fields
            .iter()
            .map(|pending| PendingFieldInfo {
                descr_index: pending.descr_index,
                target: self.decode_source(pending.target),
                value: self.decode_source(pending.value),
                item_index: pending.item_index,
            })
            .collect();
        DecodedResumeLayout {
            frames,
            virtuals,
            pending_fields,
        }
    }

    fn next_word(&self, cursor: &mut usize) -> i64 {
        let word = self
            .code
            .get(*cursor)
            .copied()
            .expect("truncated encoded resume data");
        *cursor += 1;
        word
    }

    fn decode_source(&self, encoded: i64) -> ResumeValueSource {
        let (value, tag) = untag_value(encoded);
        match tag {
            LEGACY_TAGINT => ResumeValueSource::Constant(value),
            LEGACY_TAGBOX => {
                let compact_index = decode_len(value);
                let raw_index = *self
                    .fail_arg_positions
                    .get(compact_index)
                    .expect("resume fail-arg position out of bounds");
                ResumeValueSource::FailArg(raw_index)
            }
            LEGACY_TAGVIRTUAL => ResumeValueSource::Virtual(decode_len(value)),
            LEGACY_TAGCONST => match value {
                ENCODED_UNINITIALIZED => ResumeValueSource::Uninitialized,
                ENCODED_UNAVAILABLE => ResumeValueSource::Unavailable,
                index if index >= 0 => ResumeValueSource::Constant(
                    *self
                        .consts
                        .get(decode_len(index))
                        .expect("resume const pool index out of bounds"),
                ),
                other => panic!("unknown CONST-tagged resume sentinel {other}"),
            },
            other => panic!("unknown resume tag {other}"),
        }
    }

    fn decode_virtual(&self, cursor: &mut usize) -> VirtualInfo {
        let kind = EncodedVirtualKind::from_word(self.next_word(cursor));
        match kind {
            EncodedVirtualKind::VirtualObj => {
                let type_id = u32::try_from(self.next_word(cursor)).expect("negative type_id");
                let descr_index =
                    u32::try_from(self.next_word(cursor)).expect("negative descr_index");
                let field_count = decode_len(self.next_word(cursor));
                let mut fields = Vec::with_capacity(field_count);
                for _ in 0..field_count {
                    let field_index =
                        u32::try_from(self.next_word(cursor)).expect("negative field index");
                    let source = self.decode_source(self.next_word(cursor));
                    fields.push((field_index, source));
                }
                VirtualInfo::VirtualObj {
                    type_id,
                    descr_index,
                    fields,
                }
            }
            EncodedVirtualKind::VStruct => {
                let type_id = u32::try_from(self.next_word(cursor)).expect("negative type_id");
                let descr_index =
                    u32::try_from(self.next_word(cursor)).expect("negative descr_index");
                let field_count = decode_len(self.next_word(cursor));
                let mut fields = Vec::with_capacity(field_count);
                for _ in 0..field_count {
                    let field_index =
                        u32::try_from(self.next_word(cursor)).expect("negative field index");
                    let source = self.decode_source(self.next_word(cursor));
                    fields.push((field_index, source));
                }
                VirtualInfo::VStruct {
                    type_id,
                    descr_index,
                    fields,
                }
            }
            EncodedVirtualKind::VArray => {
                let descr_index =
                    u32::try_from(self.next_word(cursor)).expect("negative descr_index");
                let item_count = decode_len(self.next_word(cursor));
                let mut items = Vec::with_capacity(item_count);
                for _ in 0..item_count {
                    items.push(self.decode_source(self.next_word(cursor)));
                }
                VirtualInfo::VArray { descr_index, items }
            }
            EncodedVirtualKind::VArrayStruct => {
                let descr_index =
                    u32::try_from(self.next_word(cursor)).expect("negative descr_index");
                let element_count = decode_len(self.next_word(cursor));
                let mut element_fields = Vec::with_capacity(element_count);
                for _ in 0..element_count {
                    let field_count = decode_len(self.next_word(cursor));
                    let mut fields = Vec::with_capacity(field_count);
                    for _ in 0..field_count {
                        let field_index =
                            u32::try_from(self.next_word(cursor)).expect("negative field index");
                        let source = self.decode_source(self.next_word(cursor));
                        fields.push((field_index, source));
                    }
                    element_fields.push(fields);
                }
                VirtualInfo::VArrayStruct {
                    descr_index,
                    element_fields,
                }
            }
            EncodedVirtualKind::VRawBuffer => {
                let size = decode_len(self.next_word(cursor));
                let entry_count = decode_len(self.next_word(cursor));
                let mut entries = Vec::with_capacity(entry_count);
                for _ in 0..entry_count {
                    let offset = decode_len(self.next_word(cursor));
                    let size_in_bytes = decode_len(self.next_word(cursor));
                    let source = self.decode_source(self.next_word(cursor));
                    entries.push((offset, size_in_bytes, source));
                }
                VirtualInfo::VRawBuffer { size, entries }
            }
        }
    }

    /// Decode this encoded snapshot back into a `ResumeData`.
    pub fn decode(&self) -> ResumeData {
        let layout = self.decode_layout();
        ResumeData {
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
                        .map(|source| source.layout_summary(&self.fail_arg_positions))
                        .collect(),
                    slot_types: None,
                })
                .collect(),
            num_virtuals: layout.virtuals.len(),
            virtual_kinds: layout.virtuals.iter().map(VirtualInfo::kind).collect(),
            virtual_layouts: layout
                .virtuals
                .iter()
                .map(|virt| virt.layout_summary(&self.fail_arg_positions))
                .collect(),
            num_fail_args: self.num_fail_args,
            fail_arg_positions: self.fail_arg_positions.clone(),
            pending_field_count: layout.pending_fields.len(),
            pending_field_layouts: layout
                .pending_fields
                .iter()
                .map(|pending| pending.layout_summary(&self.fail_arg_positions))
                .collect(),
            const_pool_size: self.consts.len(),
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

    /// Project raw backend fail values down to the compact numbering used by this snapshot.
    pub fn compact_fail_values(&self, raw_fail_values: &[i64]) -> Vec<i64> {
        self.fail_arg_positions
            .iter()
            .map(|&raw_index| raw_fail_values.get(raw_index).copied().unwrap_or(0))
            .collect()
    }
}

impl ResumeData {
    /// Create a simple ResumeData for a single-frame trace.
    pub fn simple(pc: u64, num_slots: usize) -> Self {
        let slot_map: Vec<FrameSlotSource> = (0..num_slots).map(FrameSlotSource::FailArg).collect();
        ResumeData {
            frames: vec![FrameInfo { pc, slot_map }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    /// Encode this resume snapshot into a compact RPython-style numbering.
    pub fn encode(&self) -> EncodedResumeData {
        EncodedResumeData::from_semantic(&self.frames, &self.virtuals, &self.pending_fields)
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
    /// Typed layout of the reconstructed slots, when known.
    pub slot_types: Option<Vec<Type>>,
    /// Reconstructed values for each slot.
    pub values: Vec<ReconstructedValue>,
}

impl ReconstructedFrame {
    /// Lossy conversion for legacy integer-only callers.
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
/// TODO(RPython parity): This is a legacy builder that creates `ResumeData`
/// (the old format). RPython's `ResumeDataVirtualAdder` (resume.py:298-493)
/// is fundamentally different:
///
/// 1. Takes `(optimizer, descr, guard_op, trace, memo)` as __init__ args
/// 2. `finish(pendingfields)`:
///    - Calls `memo.number()` to create NumberingState
///    - Walks virtual objects via `visitor_walk_recursive()`
///    - Collects virtual fieldnums into `rd_virtuals` array
///    - Patches slot 1 with `len(liveboxes)` (num_failargs)
///    - Stores `rd_numb = numb_state.create_numbering()`
///    - Stores `rd_consts = memo.consts`
///    - Returns `newboxes` (the final fail_args list)
///
/// Porting `finish()` requires:
/// - Access to optimizer's PtrInfo (for virtual walking)
/// - VirtualVisitor implementation (for fieldnums collection)
/// - Integration with `store_final_boxes_in_guard()` in optimizer.rs
pub struct ResumeDataVirtualAdder {
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
            frames: Vec::new(),
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
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
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VirtualObj {
            type_id,
            descr_index,
            fields,
        })
    }

    /// Convenience: add a virtual struct (NEW).
    pub fn add_virtual_struct(
        &mut self,
        type_id: u32,
        descr_index: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VStruct {
            type_id,
            descr_index,
            fields,
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
/// trait provides box access, and rebuild_from_numbering is a pure decoder.
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
    /// Legacy untyped constant pool for encode_shared.
    legacy_consts: Vec<i64>,
    /// Unified index for encode_shared (legacy path).
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
            legacy_consts: Vec::new(),
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

    /// resume.py:560-568 _gettagged — resolve an OpRef to its tagged number.
    /// Looks up in liveboxes_from_env first, then new_liveboxes, then constant.
    fn _gettagged(
        &mut self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &HashMap<u32, i16>,
        new_liveboxes: &HashMap<u32, i16>,
    ) -> i16 {
        if opref.is_none() {
            return UNINITIALIZED_TAG;
        }
        if env.is_const(opref) {
            let (val, tp) = env.get_const(opref);
            return self.getconst(val, tp);
        }
        if let Some(&tagged) = liveboxes_from_env.get(&opref.0) {
            return tagged;
        }
        if let Some(&tagged) = new_liveboxes.get(&opref.0) {
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
    pub fn _number_boxes(
        &mut self,
        boxes: &[majit_ir::OpRef],
        numb_state: &mut NumberingState,
        env: &dyn BoxEnv,
    ) -> Result<(), TagOverflow> {
        for &raw_opref in boxes {
            // resume.py:201-202: box = iter.get(item); box = box.get_box_replacement()
            let opref = env.get_box_replacement(raw_opref);

            // resume.py:204-205: isinstance(box, Const) → self.getconst(box)
            if env.is_const(opref) {
                let (val, tp) = env.get_const(opref);
                let tagged = self.getconst(val, tp);
                numb_state.append_short(tagged);
                continue;
            }
            // resume.py:207-208: liveboxes[box] (already seen)
            if let Some(&tagged) = numb_state.liveboxes.get(&opref.0) {
                numb_state.append_short(tagged);
                continue;
            }
            // resume.py:210-216: check virtual by type
            let is_virtual = match env.get_type(opref) {
                // resume.py:211-213: if box.type == 'r': getptrinfo(box).is_virtual()
                majit_ir::Type::Ref => env.is_virtual_ref(opref),
                // resume.py:214-216: if box.type == 'i': getrawptrinfo(box).is_virtual()
                majit_ir::Type::Int => env.is_virtual_raw(opref),
                _ => false,
            };
            // resume.py:217-223: tag as TAGVIRTUAL or TAGBOX
            let tagged = if is_virtual {
                let t = tag(numb_state.num_virtuals, TAGVIRTUAL)?;
                numb_state.num_virtuals += 1;
                t
            } else {
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

        // resume.py:249-253: frame chain
        // NOTE: RPython does NOT encode slot_count per frame. It uses
        // jitcode.position_info (runtime descriptor) to know how many
        // registers each jitcode frame has. Since pyre doesn't have
        // the RPython jitcode infrastructure, we encode slot_count
        // inline so rebuild_from_numbering can split frames.
        for frame in &snapshot.framestack {
            numb_state.append_int(frame.jitcode_index);
            numb_state.append_int(frame.pc);
            numb_state.append_int(frame.boxes.len() as i32);
            self._number_boxes(&frame.boxes, &mut numb_state, env)?;
        }

        // resume.py:254: patch total size
        numb_state.patch_current_size(0);

        Ok(numb_state)
    }

    /// resume.py:389-452 ResumeDataVirtualAdder.finish() — exact port.
    ///
    /// `numb_state`: output of `number()`
    /// `env`: BoxEnv for resolving box properties (constants, types)
    /// `virtual_fields`: maps opref_id → (descr, known_class, field_oprefs)
    ///   from the optimizer's PtrInfo walk. RPython discovers these via
    ///   `visitor_walk_recursive`; in majit the optimizer provides them.
    /// `pending_setfields`: resume.py:520-558 _add_pending_fields.
    ///   Each entry is (descr_index, item_index, target_opref, value_opref).
    /// `optimizer_knowledge`: bridgeopt.py:63 serialize_optimizer_knowledge.
    ///   Heap field triples and known-class info for bridge compilation.
    ///
    /// Returns `(rd_numb, rd_consts, rd_virtuals, rd_pendingfields, liveboxes)`.
    pub fn finish(
        &mut self,
        mut numb_state: NumberingState,
        env: &dyn majit_ir::BoxEnv,
        virtual_fields: &HashMap<u32, VirtualFieldInfo>,
        pending_setfields: &[(u32, i32, majit_ir::OpRef, majit_ir::OpRef)],
        optimizer_knowledge: Option<&OptimizerKnowledgeForResume>,
    ) -> (
        Vec<u8>,
        Vec<(i64, majit_ir::Type)>,
        Vec<VirtualFieldValues>,
        Vec<TaggedPendingField>,
        Vec<majit_ir::OpRef>,
    ) {
        let num_env_virtuals = numb_state.num_virtuals;

        // resume.py:410-426: split liveboxes_from_env into TAGBOX/TAGVIRTUAL
        let mut liveboxes: Vec<Option<majit_ir::OpRef>> = vec![None; numb_state.num_boxes as usize];

        // resume.py:413: self.vfieldboxes collected by virtual walk
        // resume.py:408: self.liveboxes — newly discovered boxes from field walk
        let mut new_liveboxes: HashMap<u32, i16> = HashMap::new();

        // Iterate in deterministic order (sorted by opref_id) to ensure
        // consistent virtual field numbering across runs.
        // RPython uses dict iteration which is insertion-ordered in the
        // implementation; Rust HashMap is not ordered.
        let mut sorted_liveboxes: Vec<(u32, i16)> =
            numb_state.liveboxes.iter().map(|(&k, &v)| (k, v)).collect();
        sorted_liveboxes.sort_by_key(|&(k, _)| k);

        for &(opref_id, tagged) in &sorted_liveboxes {
            let (i, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                if (i as usize) < liveboxes.len() {
                    liveboxes[i as usize] = Some(majit_ir::OpRef(opref_id));
                }
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                if let Some(vf) = virtual_fields.get(&opref_id) {
                    // resume.py:362-368: register_virtual_fields
                    for &field_opref in &vf.field_oprefs {
                        // resume.py:370-374: register_box
                        if !field_opref.is_none()
                            && !env.is_const(field_opref)
                            && !numb_state.liveboxes.contains_key(&field_opref.0)
                            && !new_liveboxes.contains_key(&field_opref.0)
                        {
                            // resume.py:212-216: check if field is virtual
                            let is_virtual = match env.get_type(field_opref) {
                                majit_ir::Type::Ref => env.is_virtual_ref(field_opref),
                                majit_ir::Type::Int => env.is_virtual_raw(field_opref),
                                _ => false,
                            };
                            if is_virtual {
                                new_liveboxes.insert(field_opref.0, UNASSIGNEDVIRTUAL);
                            } else {
                                new_liveboxes.insert(field_opref.0, UNASSIGNED);
                            }
                        }
                    }
                }
            }
        }

        // resume.py:454-509: _number_virtuals
        let mut new_boxes_list: Vec<Option<u32>> = vec![None; self.cached_boxes.len()];
        let mut count = 0;
        // Collect and sort keys for deterministic ordering.
        let mut keys: Vec<(u32, i16)> = new_liveboxes.iter().map(|(&k, &v)| (k, v)).collect();
        keys.sort_by_key(|&(k, _)| k);
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
        let mut rd_virtuals = Vec::new();
        if !virtual_fields.is_empty() {
            let length = num_env_virtuals as usize + self.cached_virtuals.len();
            rd_virtuals.resize_with(length, VirtualFieldValues::default);
            self.nvirtuals += length;
            self.nvholes += length - virtual_fields.len();

            for (&opref_id, vf) in virtual_fields {
                // resume.py:496: num, _ = untag(self.liveboxes[virtualbox])
                let tagged = numb_state
                    .liveboxes
                    .get(&opref_id)
                    .copied()
                    .unwrap_or(UNASSIGNEDVIRTUAL);
                let (num, _) = untag(tagged);
                if num >= 0 && (num as usize) < rd_virtuals.len() {
                    // resume.py:500: fieldnums = [self._gettagged(box) for box in fieldboxes]
                    let fieldnums: Vec<i16> = vf
                        .field_oprefs
                        .iter()
                        .map(|&opref| {
                            // resume.py:560-568: _gettagged
                            if opref.is_none() {
                                return UNINITIALIZED_TAG;
                            }
                            if env.is_const(opref) {
                                let (val, tp) = env.get_const(opref);
                                return self.getconst(val, tp);
                            }
                            // Check liveboxes_from_env first
                            if let Some(&t) = numb_state.liveboxes.get(&opref.0) {
                                return t;
                            }
                            // Then check new_liveboxes
                            if let Some(&t) = new_liveboxes.get(&opref.0) {
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
                    rd_virtuals[num as usize] = VirtualFieldValues {
                        descr: vf.descr.clone(),
                        known_class: vf.known_class,
                        fieldnums,
                    };
                }
            }
        }

        // resume.py:508-509: _invalidation_needed heuristic
        if nholes > 0 && liveboxes.len() > majit_ir::FAILARGS_LIMIT / 2 {
            if nholes > liveboxes.len() / 3 {
                self.clear_box_virtual_numbers();
            }
        }

        // resume.py:445: _add_pending_fields(pending_setfields)
        let mut rd_pendingfields_tagged = Vec::new();
        for &(descr_index, item_index, target, value) in pending_setfields {
            let num = self._gettagged(target, env, &numb_state.liveboxes, &new_liveboxes);
            let fieldnum = self._gettagged(value, env, &numb_state.liveboxes, &new_liveboxes);
            rd_pendingfields_tagged.push(TaggedPendingField {
                descr_index,
                item_index,
                num,
                fieldnum,
            });
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

        (
            rd_numb,
            rd_consts,
            rd_virtuals,
            rd_pendingfields_tagged,
            ordered_liveboxes,
        )
    }

    /// Encode a `ResumeData` using the shared constant pool.
    ///
    /// Returns a compact `EncodedResumeData` whose `consts` pool is shared
    /// across all guards encoded through this memo.
    pub fn encode_shared(&mut self, rd: &ResumeData) -> EncodedResumeData {
        let mut code = vec![0, 0, encode_len(rd.frames.len())];
        let mut num_fail_args = 0usize;
        let mut fail_arg_positions = Vec::new();
        let mut fail_arg_indices = HashMap::new();

        for frame in &rd.frames {
            code.push(encode_u64(frame.pc));
            code.push(encode_len(frame.slot_map.len()));
            for source in &frame.slot_map {
                let encoded = EncodedResumeData::encode_source(
                    source,
                    &mut self.legacy_consts,
                    &mut self.const_indices,
                    &mut num_fail_args,
                    &mut fail_arg_positions,
                    &mut fail_arg_indices,
                );
                code.push(encoded);
            }
        }
        code.push(encode_len(rd.virtuals.len()));
        for vinfo in &rd.virtuals {
            EncodedResumeData::encode_virtual(
                vinfo,
                &mut code,
                &mut self.legacy_consts,
                &mut self.const_indices,
                &mut num_fail_args,
                &mut fail_arg_positions,
                &mut fail_arg_indices,
            );
        }
        let pending_fields = rd
            .pending_fields
            .iter()
            .map(|pending| EncodedPendingFieldWrite {
                descr_index: pending.descr_index,
                target: EncodedResumeData::encode_source(
                    &pending.target,
                    &mut self.legacy_consts,
                    &mut self.const_indices,
                    &mut num_fail_args,
                    &mut fail_arg_positions,
                    &mut fail_arg_indices,
                ),
                value: EncodedResumeData::encode_source(
                    &pending.value,
                    &mut self.legacy_consts,
                    &mut self.const_indices,
                    &mut num_fail_args,
                    &mut fail_arg_positions,
                    &mut fail_arg_indices,
                ),
                item_index: pending.item_index,
            })
            .collect();
        code[0] = encode_len(code.len());
        code[1] = encode_len(num_fail_args);

        EncodedResumeData {
            code,
            consts: self.legacy_consts.clone(),
            num_fail_args,
            fail_arg_positions,
            pending_fields,
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

// ═══════════════════════════════════════════════════════════════
// resume.py:1042-1080 rebuild_from_resumedata — tagged numbering
// deserialization.
// ═══════════════════════════════════════════════════════════════

/// Result of rebuilding interpreter state from resume data.
#[derive(Debug)]
pub struct RebuiltFrame {
    pub jitcode_index: i32,
    pub pc: i32,
    pub values: Vec<RebuiltValue>,
}

/// resume.py:576-728 VirtualInfo parity.
/// Describes a virtual object's fields for materialization.
/// RPython uses a class hierarchy (VirtualInfo, VStructInfo, VArrayInfo, etc.).
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

/// Input to ResumeDataLoopMemo.finish() — virtual field info from the optimizer.
///
/// resume.py:359-368 — `register_virtual_fields` equivalent.
/// The optimizer walks PtrInfo::Virtual/VirtualStruct to discover fields,
/// then passes the results here. RPython discovers these via
/// `info.visitor_walk_recursive(box, self)` callback pattern.
#[derive(Debug, Clone)]
pub struct VirtualFieldInfo {
    /// Type descriptor for the virtual object.
    pub descr: Option<majit_ir::DescrRef>,
    /// Known class pointer (for NewWithVtable).
    pub known_class: Option<i64>,
    /// Resolved field OpRefs (after get_box_replacement).
    pub field_oprefs: Vec<majit_ir::OpRef>,
}

/// A single value decoded from tagged resume numbering.
#[derive(Debug, Clone, PartialEq)]
pub enum RebuiltValue {
    /// TAGINT: inline integer constant.
    Int(i32),
    /// TAGCONST: value from constant pool (value, type).
    Const(i64, majit_ir::Type),
    /// TAGBOX: live value from fail_args[index].
    Box(usize),
    /// TAGVIRTUAL: virtual object (index into rd_virtuals).
    Virtual(usize),
    /// Unassigned slot.
    Unassigned,
}

/// Decode a single tagged value from resume numbering.
/// resume.py:1552-1588 decode_int/decode_ref/decode_float parity.
fn decode_tagged(
    tagged: i16,
    num_failargs: i32,
    rd_consts: &[(i64, majit_ir::Type)],
) -> RebuiltValue {
    let (val, tagbits) = untag(tagged);
    match tagbits {
        TAGINT => RebuiltValue::Int(val),
        TAGCONST => {
            if tagged == NULLREF {
                RebuiltValue::Const(0, majit_ir::Type::Ref)
            } else if tagged == UNINITIALIZED_TAG {
                RebuiltValue::Unassigned
            } else {
                let idx = (val - TAG_CONST_OFFSET) as usize;
                let (c, tp) = rd_consts
                    .get(idx)
                    .copied()
                    .unwrap_or((0, majit_ir::Type::Int));
                RebuiltValue::Const(c, tp)
            }
        }
        TAGBOX => {
            let index = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            RebuiltValue::Box(index)
        }
        TAGVIRTUAL => RebuiltValue::Virtual(val as usize),
        _ => RebuiltValue::Unassigned,
    }
}

/// resume.py:1042-1057 rebuild_from_resumedata parity.
///
/// Decode a numbering (produced by `ResumeDataLoopMemo::number()`)
/// back into frame state. `rd_consts` is the shared constant pool.
///
/// NOTE: RPython's rebuild_from_resumedata is an OO builder that
/// creates MetaInterp frames and writes to blackhole registers
/// in-place (via ResumeDataBoxReader). This is a pure functional
/// decoder that returns RebuiltFrame structs. The caller materializes
/// concrete interpreter state from the decoded values.
///
/// TODO(ResumeDataVirtualAdder): TAGVIRTUAL values are returned as
/// `RebuiltValue::Virtual(index)`. RPython's decode_ref/decode_int
/// calls `getvirtual_ptr(num)` / `getvirtual_int(num)` which lazily
/// allocates the virtual object from rd_virtuals[num]. Once
/// ResumeDataVirtualAdder is ported and rd_virtuals is populated,
/// add a `rd_virtuals` parameter and implement materialization.
pub fn rebuild_from_numbering(
    rd_numb: &[u8],
    rd_consts: &[(i64, majit_ir::Type)],
) -> (i32, Vec<RebuiltFrame>) {
    let mut reader = crate::resumecode::Reader::new(rd_numb);

    let total_size = reader.next_item();
    let num_failargs = reader.next_item();

    // Virtualizable array (skip).
    let vable_len = reader.next_item();
    if vable_len > 0 {
        reader.jump(vable_len as usize);
    }

    // Virtualref array (skip).
    let vref_len = reader.next_item();
    if vref_len > 0 {
        reader.jump((vref_len * 2) as usize);
    }

    // Frames.
    // resume.py:1049-1055: read frames until done.
    // Each frame has: jitcode_index, pc, slot_count, [tagged values × slot_count].
    // slot_count is our extension (RPython uses jitcode.position_info).
    let mut frames = Vec::new();
    while reader.has_more() {
        let jitcode_index = reader.next_item();
        let pc = reader.next_item();
        let slot_count = reader.next_item() as usize;
        let mut values = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            let tagged = reader.next_item() as i16;
            values.push(decode_tagged(tagged, num_failargs, rd_consts));
        }
        frames.push(RebuiltFrame {
            jitcode_index,
            pc,
            values,
        });
    }

    let _ = total_size;
    (num_failargs, frames)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(encoded.code[0] as usize, encoded.code.len());
        assert_eq!(encoded.num_fail_args, 1);
        assert_eq!(encoded.fail_arg_positions, vec![3]);

        // Header: [size, fail_arg_count, num_frames, pc, slot_count, ...slots]
        let slot_words = &encoded.code[5..10];
        assert_eq!(untag_value(slot_words[0]), (0, LEGACY_TAGBOX));
        assert_eq!(untag_value(slot_words[1]), (7, LEGACY_TAGINT));
        assert_eq!(untag_value(slot_words[2]), (1, LEGACY_TAGVIRTUAL));
        assert_eq!(
            untag_value(slot_words[3]),
            (ENCODED_UNINITIALIZED, LEGACY_TAGCONST)
        );
        assert_eq!(
            untag_value(slot_words[4]),
            (ENCODED_UNAVAILABLE, LEGACY_TAGCONST)
        );
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
        assert_eq!(encoded.consts, vec![large_const]);
        assert_eq!(encoded.num_fail_args, 2);
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
        assert_eq!(encoded.consts, vec![large_const]);
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
                type_id: 1,
                descr_index: 10,
                fields: vec![(0, VirtualFieldSource::FailArg(1))],
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
        assert_eq!(encoded.pending_fields.len(), 2);
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
                    type_id: 1,
                    descr_index: 10,
                    fields: vec![(0, VirtualFieldSource::FailArg(0))],
                },
                // Outer virtual (index 1), references inner via Virtual(0)
                VirtualInfo::VirtualObj {
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
        let v0 = builder.add_virtual_obj(1, 10, vec![(0, VirtualFieldSource::FailArg(0))]);
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
        assert_eq!(encoded.num_fail_args, 3);

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
                type_id: 1,
                descr_index: 10,
                fields: vec![(0, VirtualFieldSource::FailArg(1))],
            }],
            pending_fields: Vec::new(),
        };

        let encoded = EncodedResumeData::encode(&rd);
        assert_eq!(encoded.num_fail_args, 2); // max fail_arg is 1, so 2 needed
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
        assert_eq!(encoded.num_fail_args, 3);
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
                type_id: 1,
                descr_index: 9,
                fields: vec![(0, VirtualFieldSource::FailArg(7))],
            }],
            pending_fields: vec![PendingFieldInfo {
                descr_index: 12,
                target: ResumeValueSource::FailArg(2),
                value: ResumeValueSource::FailArg(7),
                item_index: Some(4),
            }],
        };

        let encoded = EncodedResumeData::encode(&rd);
        assert_eq!(encoded.num_fail_args, 2);
        assert_eq!(encoded.fail_arg_positions, vec![7, 2]);
        assert_eq!(encoded.decode(), rd);

        let raw_fail_values = vec![100, 101, 102, 103, 104, 105, 106, 107];
        assert_eq!(
            encoded.compact_fail_values(&raw_fail_values),
            vec![107, 102]
        );

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
        assert!(enc_small.consts.is_empty()); // small const inlined
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
        assert_eq!(enc.num_fail_args, 0);
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
                type_id: 1,
                descr_index: 10,
                fields: vec![(0, VirtualFieldSource::FailArg(1))],
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
                            fail_arg_index: Some(0),
                            raw_fail_arg_position: Some(3),
                            constant: None,
                            virtual_index: None,
                        },
                        ResumeValueLayoutSummary {
                            kind: ResumeValueKind::Constant,
                            fail_arg_index: None,
                            raw_fail_arg_position: None,
                            constant: Some(77),
                            virtual_index: None,
                        },
                        ResumeValueLayoutSummary {
                            kind: ResumeValueKind::Virtual,
                            fail_arg_index: None,
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
                            fail_arg_index: None,
                            raw_fail_arg_position: None,
                            constant: None,
                            virtual_index: None,
                        },
                        ResumeValueLayoutSummary {
                            kind: ResumeValueKind::Uninitialized,
                            fail_arg_index: None,
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
                    fail_arg_index: Some(0),
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
                    fail_arg_index: Some(1),
                    raw_fail_arg_position: Some(1),
                    constant: None,
                    virtual_index: None,
                },
                value: ResumeValueLayoutSummary {
                    kind: ResumeValueKind::Constant,
                    fail_arg_index: None,
                    raw_fail_arg_position: None,
                    constant: Some(88),
                    virtual_index: None,
                },
            }]
        );
        assert_eq!(summary.num_fail_args, 2);
        assert_eq!(summary.fail_arg_positions, vec![3, 1]);
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
                    type_id: 3,
                    descr_index: 10,
                    fields: vec![
                        (0, VirtualFieldSource::FailArg(2)),
                        (1, VirtualFieldSource::Constant(123)),
                    ],
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
        assert_eq!(
            summary.compact_fail_values(&fail_values),
            encoded.compact_fail_values(&fail_values)
        );

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
                    fail_arg_index: None,
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
                    fail_arg_index: Some(0),
                    raw_fail_arg_position: Some(1),
                    constant: None,
                    virtual_index: None,
                }],
            }],
            num_fail_args: 1,
            fail_arg_positions: vec![1],
            pending_field_count: 1,
            pending_field_layouts: vec![PendingFieldLayoutSummary {
                descr_index: 11,
                item_index: Some(2),
                is_array_item: true,
                target_kind: ResumeValueKind::Virtual,
                value_kind: ResumeValueKind::FailArg,
                target: ResumeValueLayoutSummary {
                    kind: ResumeValueKind::Virtual,
                    fail_arg_index: None,
                    raw_fail_arg_position: None,
                    constant: None,
                    virtual_index: Some(0),
                },
                value: ResumeValueLayoutSummary {
                    kind: ResumeValueKind::FailArg,
                    fail_arg_index: Some(0),
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
        // items[6] = slot_count = 3
        assert_eq!(items[6], 3);
        // items[7] = OpRef(10001) tagged as TAGINT(42) since 42 fits in 13 bits
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGINT);
        assert_eq!(val, 42);
        // items[8] = OpRef(1) tagged as TAGBOX(0) — first live box
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[9] = OpRef(2) tagged as TAGBOX(1) — second live box
        let (val, tagbits) = untag(items[9] as i16);
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

        let (num_failargs, rebuilt_frames) = rebuild_from_numbering(&rd_numb, memo.consts());
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

        let (num_failargs, rebuilt_frames) = rebuild_from_numbering(&rd_numb, memo.consts());
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
        // items[6] = slot_count = 3
        assert_eq!(items[6], 3);
        // items[7] = OpRef(1) → TAGBOX(0)
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[8] = OpRef(2) → TAGVIRTUAL(0)
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGVIRTUAL);
        assert_eq!(val, 0);
        // items[9] = OpRef(3) → TAGBOX(1)
        let (val, tagbits) = untag(items[9] as i16);
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

        let (num_failargs, rebuilt_frames) = rebuild_from_numbering(&rd_numb, memo.consts());
        assert_eq!(num_failargs, 3); // OpRef(1), OpRef(2), OpRef(3) are boxes
        assert_eq!(rebuilt_frames.len(), 2);
        // Frame 0
        assert_eq!(rebuilt_frames[0].jitcode_index, 0);
        assert_eq!(rebuilt_frames[0].pc, 10);
        // Frame 1
        assert_eq!(rebuilt_frames[1].jitcode_index, 1);
        assert_eq!(rebuilt_frames[1].pc, 20);
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
        let empty_vfields = HashMap::new();
        let (rd_numb, rd_consts, _rd_virtuals, _rd_pf, liveboxes) =
            memo.finish(numb_state, &env, &empty_vfields, &[], None);

        // liveboxes should contain only TAGBOX entries: OpRef(1) and OpRef(3)
        assert_eq!(liveboxes.len(), 2);
        assert_eq!(liveboxes[0], OpRef(1)); // box #0
        assert_eq!(liveboxes[1], OpRef(3)); // box #1

        // rd_numb should be valid
        let (num_failargs, rebuilt_frames) = rebuild_from_numbering(&rd_numb, &rd_consts);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(rebuilt_frames[0].values[0], RebuiltValue::Int(42));
        assert_eq!(rebuilt_frames[0].values[1], RebuiltValue::Box(0));
        assert_eq!(rebuilt_frames[0].values[2], RebuiltValue::Virtual(0));
        assert_eq!(rebuilt_frames[0].values[3], RebuiltValue::Box(1));
    }
}
