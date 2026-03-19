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

const TAG_MASK: i64 = 0b11;
const TAG_CONST: i64 = 0;
const TAG_INT: i64 = 1;
const TAG_FAILARG: i64 = 2;
const TAG_VIRTUAL: i64 = 3;

// RPython uses CONST-tagged sentinels for special states like UNINITIALIZED.
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

fn tag_value(value: i64, tag: i64) -> i64 {
    assert!((TAG_CONST..=TAG_VIRTUAL).contains(&tag));
    assert!(
        can_inline_tagged(value),
        "tagged resume value {value} exceeds inline range"
    );
    (value << 2) | tag
}

fn untag_value(encoded: i64) -> (i64, i64) {
    (encoded >> 2, encoded & TAG_MASK)
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
    VUniPlain {
        chars: Vec<VirtualFieldSource>,
    },
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
                tag_value(encode_len(compact_index), TAG_FAILARG)
            }
            ResumeValueSource::Constant(value) if can_inline_tagged(*value) => {
                tag_value(*value, TAG_INT)
            }
            ResumeValueSource::Constant(value) => {
                let next_index = consts.len();
                let index = *const_indices.entry(*value).or_insert_with(|| {
                    consts.push(*value);
                    next_index
                });
                tag_value(encode_len(index), TAG_CONST)
            }
            ResumeValueSource::Virtual(index) => tag_value(encode_len(*index), TAG_VIRTUAL),
            ResumeValueSource::Uninitialized => tag_value(ENCODED_UNINITIALIZED, TAG_CONST),
            ResumeValueSource::Unavailable => tag_value(ENCODED_UNAVAILABLE, TAG_CONST),
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
            TAG_INT => ResumeValueSource::Constant(value),
            TAG_FAILARG => {
                let compact_index = decode_len(value);
                let raw_index = *self
                    .fail_arg_positions
                    .get(compact_index)
                    .expect("resume fail-arg position out of bounds");
                ResumeValueSource::FailArg(raw_index)
            }
            TAG_VIRTUAL => ResumeValueSource::Virtual(decode_len(value)),
            TAG_CONST => match value {
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
pub struct ResumeDataLoopMemo {
    /// Shared constant pool across all guards in this loop.
    consts: Vec<i64>,
    const_indices: HashMap<i64, usize>,
}

impl ResumeDataLoopMemo {
    pub fn new() -> Self {
        ResumeDataLoopMemo {
            consts: Vec::new(),
            const_indices: HashMap::new(),
        }
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
                    &mut self.consts,
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
                &mut self.consts,
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
                    &mut self.consts,
                    &mut self.const_indices,
                    &mut num_fail_args,
                    &mut fail_arg_positions,
                    &mut fail_arg_indices,
                ),
                value: EncodedResumeData::encode_source(
                    &pending.value,
                    &mut self.consts,
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
            consts: self.consts.clone(),
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
            ResumeValueSource::FailArg(idx) => {
                self.fail_values.get(*idx).copied().unwrap_or(0)
            }
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
        assert_eq!(untag_value(slot_words[0]), (0, TAG_FAILARG));
        assert_eq!(untag_value(slot_words[1]), (7, TAG_INT));
        assert_eq!(untag_value(slot_words[2]), (1, TAG_VIRTUAL));
        assert_eq!(
            untag_value(slot_words[3]),
            (ENCODED_UNINITIALIZED, TAG_CONST)
        );
        assert_eq!(untag_value(slot_words[4]), (ENCODED_UNAVAILABLE, TAG_CONST));
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
        // Small constant: should be inlined with TAG_INT
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
        // guards use TAG_INT, so the pool stays at 0.
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
            }
        );
    }
}
