//! Resume data decoding primitives.
//!
//! resume.py tag layout + `rebuild_from_numbering` (decode-side) shared by
//! `majit-metainterp` (guard failure) and `pyre-jit-trace` (trace-side
//! recovery). The encoder-side (`Snapshot`, `NumberingState`,
//! `ResumeDataLoopMemo`) lives in `majit-metainterp/src/resume.rs` to stay
//! colocated with the metainterp context that drives it.

use crate::resumecode;
use crate::{Const, Type};

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global frame_value_count callback for RPython-parity multi-frame decode.
/// resume.py:1049: consume_boxes(f.get_current_position_info(), ...) uses
/// per-jitcode liveness at the decode site. pyre registers this callback
/// from pyre-jit-trace::state so that all callers of rebuild_from_numbering
/// (including those in majit-metainterp and majit-backend-cranelift) can
/// split multi-frame sections correctly.
static FRAME_VALUE_COUNT_FN: AtomicUsize = AtomicUsize::new(0);

/// Register the global frame_value_count callback.
///
/// Signature: `(jitcode_index, pc) -> count`.
pub fn set_frame_value_count_fn(f: fn(i32, i32) -> usize) {
    FRAME_VALUE_COUNT_FN.store(f as usize, Ordering::Relaxed);
}

/// Get the registered frame_value_count callback (if any).
pub fn get_frame_value_count_fn() -> Option<fn(i32, i32) -> usize> {
    let p = FRAME_VALUE_COUNT_FN.load(Ordering::Relaxed);
    if p == 0 {
        None
    } else {
        Some(unsafe { std::mem::transmute(p) })
    }
}

// resume.py:123-132 — tag constants
pub const TAGCONST: u8 = 0;
pub const TAGINT: u8 = 1;
pub const TAGBOX: u8 = 2;
pub const TAGVIRTUAL: u8 = 3;

/// After-residual-call marker folded into a snapshot frame's pc word.
///
/// The same Python PC serves two resume semantics: a normal guard at a
/// `residual_call` opcode re-executes from the call start
/// (`pc_map[pc]`), while an after-residual-call guard (`GUARD_EXCEPTION`
/// / `GUARD_NO_EXCEPTION` / `GUARD_NOT_FORCED` / `GUARD_ALWAYS_FAILS`)
/// must resume at the call's own post-call `-live-`/`catch_exception`
/// (`PyJitCode.after_residual_call_resume_pc`).  RPython keeps
/// `frame.pc` at the post-call jitcode position
/// (`pyjitpl.py:2610-2624 capture_resumedata(resumepc=-1)`); pyre stores
/// the Python PC and translates at decode time, so this high bit on the
/// rd_numb pc word carries the distinction without a format change.
///
/// Stealing bit 14 narrows the storable Python PC from `append_int`'s
/// full `i16` range down to `[0, 1 << 14)`: a pc `>= 1 << 14` sets bit 14
/// on its own and `decode_resume_pc` would mis-read it as marked.
/// RPython has no such cap at this layer — it stores the JitCode pc
/// directly (`pyjitpl.py:2610-2624`), needing no marker bit.  This is a
/// PRE-EXISTING-ADAPTATION of pyre's pc_map indirection (resume.rs:259-263
/// stores the Python PC, not the jitcode offset); the convergence path is
/// to store the jitcode pc directly and drop both pc_map and this marker.
/// Until then every resume pc is bounds-checked against this constant at
/// capture — marked pcs through `encode_after_residual_call_pc`, unmarked
/// pcs at their `build_framestack_snapshot` / `capture_resumedata` sites —
/// so an out-of-range pc fails loudly at trace time instead of corrupting
/// decode at resume time.
pub const AFTER_RESIDUAL_CALL_PC_FLAG: i32 = 1 << 14;

/// Sentinel for a snapshot frame's `jitcode_pc` word: no direct JitCode
/// resume coordinate. Such a frame is unrepresentable and must decline before
/// resume data is published.
///
/// Each per-frame header carries a `jitcode_pc` word after `pc`. A branch guard whose
/// not-taken arm keeps operand-stack temps the JitCode virtualized cannot
/// be described by the merge-target Python pc: its kept temps are only
/// live at the *guard* JitCode position, not at the resume Python opcode
/// boundary.  For such guards the capture stores the guard's JitCode byte
/// offset here (a non-negative value), and both the box-count liveness
/// query and the blackhole `setposition` resolve directly off it,
/// bypassing `pc_map` — the direct `setposition(jitcode, pc)` shape
/// (`pyjitpl.py:2610-2624`) without the pc round-trip.  The Python `pc`
/// word stays populated for interpreter re-entry / `last_instr`.
pub const NO_JITCODE_PC: i32 = -1;

/// #73: a depth-0 branch guard's resume word may carry the walk's
/// arm-independent `-live-` BEFORE anchor (`orgpc`) plus the GuardTrue/False
/// flavor, tagged into the free negative space of the `jitcode_pc` word
/// (offsets are >=0, NO_JITCODE_PC is -1). Decode expands it to the block-head
/// resume marker. `orgpc` must be <= BRANCH_ORGPC_MAX to stay in i16 range.
pub const BRANCH_ORGPC_MAX: usize = 16382;

/// Tag `orgpc` (+ 1 flavor bit) into the free negative space `[-32767, -2]` of
/// the `jitcode_pc` word. `orgpc` must be `<= BRANCH_ORGPC_MAX`.
pub const fn encode_branch_orgpc(orgpc: usize, flavor_guard_true: bool) -> i32 {
    // orgpc in [0, BRANCH_ORGPC_MAX], 1 flavor bit -> [-32767, -2]
    -2 - (((orgpc as i32) << 1) | (flavor_guard_true as i32))
}

/// Inverse of [`encode_branch_orgpc`]. Returns `Some((orgpc, flavor))` for a
/// tagged word (`<= -2`), `None` for offsets (`>= 0`) and NO_JITCODE_PC (`-1`).
pub const fn decode_branch_orgpc(word: i32) -> Option<(usize, bool)> {
    if word <= -2 {
        let v = (-2 - word) as usize;
        Some((v >> 1, (v & 1) == 1))
    } else {
        None
    }
}

/// Fold the after-residual-call marker into a snapshot frame pc word.
/// `pc` must be a non-negative Python PC `< AFTER_RESIDUAL_CALL_PC_FLAG`.
/// The bound is enforced in release builds, not only under `debug_assert`,
/// because a stored pc `>= 1 << 14` is silently mis-decoded as marked.
pub fn encode_after_residual_call_pc(pc: i32) -> i32 {
    assert!(
        pc >= 0 && pc < AFTER_RESIDUAL_CALL_PC_FLAG,
        "after-residual-call pc {pc} out of range for marker bit"
    );
    pc | AFTER_RESIDUAL_CALL_PC_FLAG
}

/// Split a snapshot frame pc word into `(python_pc, after_residual_call)`.
/// A negative word (sentinel / invalid pc) passes through unflagged so the
/// existing `pc < 0` guards keep rejecting it.
pub fn decode_resume_pc(raw_pc: i32) -> (i32, bool) {
    if raw_pc >= 0 && raw_pc & AFTER_RESIDUAL_CALL_PC_FLAG != 0 {
        (raw_pc & !AFTER_RESIDUAL_CALL_PC_FLAG, true)
    } else {
        (raw_pc, false)
    }
}

/// `resume.py` tag discriminator used by `ResumeValueSource` /
/// `ResumeValueLayoutSummary` to record where a guard-time value
/// comes from.  Moved here from `majit-metainterp::resume` so it can
/// be referenced from `majit-backend` along with the rest of the
/// Phase C-1 unified-descr type migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeValueKind {
    FailArg,
    Constant,
    Virtual,
    Uninitialized,
    Unavailable,
}

/// Summary of a `ResumeValueSource` in serialization-friendly form.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade)
/// alongside `ResumeValueKind` so backend-side resume readers can
/// build summaries without depending on the metainterp crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeValueLayoutSummary {
    pub kind: ResumeValueKind,
    pub fail_arg_index: usize,
    pub raw_fail_arg_position: Option<usize>,
    /// RPython parity: `Const.value` raw bits (getint/getref_base/getfloatstorage
    /// all project to `i64`).
    pub constant: Option<i64>,
    /// RPython parity: `Const.type` — paired with `constant` so the summary
    /// round-trips back into a typed `ResumeValueSource::Constant(Const)`.
    pub constant_type: Option<Type>,
    pub virtual_index: Option<usize>,
}

/// `resume.py` virtual-info class discriminator used by `VirtualInfo`
/// to record the variant kind without unfolding the enum.  Moved here
/// from `majit-metainterp::resume` (Phase C-1 cascade) so the
/// `ResumeData` chain can live in a backend-accessible crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeVirtualKind {
    Object,
    Struct,
    Array,
    ArrayStruct,
    RawBuffer,
    /// resume.py:763 VStrPlainInfo — virtual plain string.
    StrPlain,
    /// resume.py:781 VStrConcatInfo — virtual concatenated string.
    StrConcat,
    /// resume.py:801 VStrSliceInfo — virtual string slice.
    StrSlice,
    /// resume.py:817 VUniPlainInfo — virtual plain unicode string.
    UniPlain,
    /// resume.py:836 VUniConcatInfo — virtual concatenated unicode.
    UniConcat,
    /// resume.py:856 VUniSliceInfo — virtual unicode slice.
    UniSlice,
}

/// Per-frame layout summary used by `ResumeLayoutSummary`.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade) —
/// all field types (`Type`, `ResumeValueKind`, `ResumeValueLayoutSummary`)
/// live in `majit-ir`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeFrameLayoutSummary {
    pub trace_id: Option<u64>,
    pub header_pc: Option<u64>,
    pub source_guard: Option<(u64, u32)>,
    /// resume.py:250 jitcode_index — index into metainterp_sd.jitcodes[].
    pub jitcode_index: i32,
    pub pc: u64,
    pub slot_sources: Vec<ResumeValueKind>,
    pub slot_layouts: Vec<ResumeValueLayoutSummary>,
    pub slot_types: Option<Vec<Type>>,
}

/// Serialization-friendly summary of a `PendingFieldInfo`.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade);
/// dependencies (`DescrRef`, `ResumeValueKind`, `ResumeValueLayoutSummary`)
/// are all in `majit-ir`.
#[derive(Debug, Clone)]
pub struct PendingFieldLayoutSummary {
    /// `resume.py:88 lldescr` — identity-compared via `Arc::ptr_eq`
    /// (`history.py:125`).
    pub descr: Option<crate::DescrRef>,
    pub item_index: Option<usize>,
    pub is_array_item: bool,
    pub target_kind: ResumeValueKind,
    pub value_kind: ResumeValueKind,
    pub target: ResumeValueLayoutSummary,
    pub value: ResumeValueLayoutSummary,
}

impl PartialEq for PendingFieldLayoutSummary {
    fn eq(&self, other: &Self) -> bool {
        opt_descr_arc_ptr_eq(&self.descr, &other.descr)
            && self.item_index == other.item_index
            && self.is_array_item == other.is_array_item
            && self.target_kind == other.target_kind
            && self.value_kind == other.value_kind
            && self.target == other.target
            && self.value == other.value
    }
}
impl Eq for PendingFieldLayoutSummary {}

/// `history.py:125 id(descr)` parity: `Option<DescrRef>` identity
/// comparison via `Arc::ptr_eq`.
#[inline]
pub fn opt_descr_arc_ptr_eq(a: &Option<crate::DescrRef>, b: &Option<crate::DescrRef>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    }
}

/// Serialization-friendly summary of a `VirtualInfo`.
///
/// Moved here from `majit-metainterp::resume` (Phase C-1 cascade).
/// Carries the same shape as `VirtualInfo` but with
/// `ResumeValueLayoutSummary` instead of `ResumeValueSource` field
/// values — enables round-trip through backend-side resume layout
/// builders without depending on `majit-metainterp`.
#[derive(Debug, Clone)]
pub enum ResumeVirtualLayoutSummary {
    Object {
        /// resume.py:615 self.descr — live SizeDescr, preserved across summary round-trip.
        descr: Option<crate::DescrRef>,
        type_id: u32,
        /// info.py:318 _known_class — vtable pointer.
        known_class: Option<i64>,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
        fielddescrs: Vec<crate::FieldDescrInfo>,
        descr_size: usize,
    },
    Struct {
        /// resume.py:631 self.typedescr — live SizeDescr, preserved across summary round-trip.
        typedescr: Option<crate::DescrRef>,
        type_id: u32,
        fields: Vec<(u32, ResumeValueLayoutSummary)>,
        fielddescrs: Vec<crate::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py:643-684 AbstractVArrayInfo
    Array {
        /// resume.py:646: self.arraydescr
        arraydescr: Option<crate::DescrRef>,
        /// resume.py:680-683: VArrayInfoClear.clear=True / VArrayInfoNotClear.clear=False
        clear: bool,
        items: Vec<ResumeValueLayoutSummary>,
    },
    /// resume.py:736 VArrayStructInfo(arraydescr, size, fielddescrs)
    ArrayStruct {
        /// resume.py:739: self.arraydescr
        arraydescr: Option<crate::DescrRef>,
        /// resume.py:740: self.fielddescrs
        fielddescrs: Vec<crate::DescrRef>,
        element_fields: Vec<Vec<(u32, ResumeValueLayoutSummary)>>,
    },
    RawBuffer {
        /// resume.py:694: self.func
        func: i64,
        size: usize,
        /// resume.py:695: self.offsets — signed (rawbuffer.py:14).
        offsets: Vec<i64>,
        /// resume.py:697: self.descrs
        descrs: Vec<crate::ArrayDescrInfo>,
        /// resume.py:693: fieldnums (decoded)
        values: Vec<ResumeValueLayoutSummary>,
    },
    /// `resume.py: VRawSliceInfo` — a slice into a virtual raw buffer.
    RawSlice {
        offset: i64,
        parent: ResumeValueLayoutSummary,
    },
    /// `resume.py:763 VStrPlainInfo` — virtual string (known characters).
    StrPlain {
        chars: Vec<ResumeValueLayoutSummary>,
    },
    /// `resume.py:781 VStrConcatInfo` — virtual string concat. OS_STR_CONCAT
    /// funcptr is resolved at materialization via
    /// `callinfocollection.funcptr_for_oopspec(...)` (resume.py:1467-1468),
    /// not stored on the summary.
    StrConcat {
        left: ResumeValueLayoutSummary,
        right: ResumeValueLayoutSummary,
    },
    /// `resume.py:801 VStrSliceInfo` — virtual slice of a larger string.
    /// OS_STR_SLICE funcptr resolved via callinfocollection at
    /// materialization (resume.py:1477-1478).
    StrSlice {
        source: ResumeValueLayoutSummary,
        start: ResumeValueLayoutSummary,
        length: ResumeValueLayoutSummary,
    },
    /// `resume.py:817 VUniPlainInfo` — unicode counterpart.
    UniPlain {
        chars: Vec<ResumeValueLayoutSummary>,
    },
    /// `resume.py:836 VUniConcatInfo` — unicode counterpart.
    /// OS_UNI_CONCAT funcptr resolved via callinfocollection
    /// (resume.py:1494-1495).
    UniConcat {
        left: ResumeValueLayoutSummary,
        right: ResumeValueLayoutSummary,
    },
    /// `resume.py:856 VUniSliceInfo` — unicode counterpart.
    /// OS_UNI_SLICE funcptr resolved via callinfocollection
    /// (resume.py:1504-1505).
    UniSlice {
        source: ResumeValueLayoutSummary,
        start: ResumeValueLayoutSummary,
        length: ResumeValueLayoutSummary,
    },
}
const TAGMASK: u8 = 3;

pub const UNASSIGNED: i16 = ((-1i32 << 13) << 2 | TAGBOX as i32) as i16;
pub const NULLREF: i16 = ((-1i32 << 2) | TAGCONST as i32) as i16;
pub const UNINITIALIZED_TAG: i16 = ((-2i32 << 2) | TAGCONST as i32) as i16;
pub const TAG_CONST_OFFSET: i32 = 0;

/// resume.py:106-109
pub fn untag(value: i16) -> (i32, u8) {
    let widened = value as i32;
    let tagbits = (widened & TAGMASK as i32) as u8;
    (widened >> 2, tagbits)
}

// ── Decoding (for guard failure recovery) ──

/// Decoded value from rd_numb.
#[derive(Debug, Clone, PartialEq)]
pub enum RebuiltValue {
    /// TAGBOX(n, kind): value from deadframe slot n with the kind that
    /// the parent guard's `fail_arg_types[n]` recorded at numbering time.
    /// resume.py:1245 `decode_box(num, kind)` parity — RPython resolves
    /// the kind at decode time from the callback (i/r/f) that the encoder
    /// dispatched through `enumerate_vars(info, liveness_info, ...)`. The
    /// box's `.type` matches that kind. majit stores the kind on the
    /// variant so consumers don't need a parallel `fail_arg_types` lookup.
    Box(usize, Type),
    /// TAGCONST / TAGINT: compile-time constant. Carries a `Const`
    /// (Int/Float/Ref), matching RPython's `decode_box` which returns a
    /// `ConstInt`/`ConstFloat`/`ConstPtr` regardless of whether the value
    /// came from the inline TAGINT encoding or the TAGCONST pool
    /// (resume.py:1250-1270).
    Const(crate::Const),
    /// TAGVIRTUAL(n): virtual object index n.
    Virtual(usize),
    /// Uninitialized/unassigned slot.
    Unassigned,
}

/// Decoded frame from rd_numb.
#[derive(Debug, Clone)]
pub struct RebuiltFrame {
    pub jitcode_index: i32,
    /// resume.py:250 `pc` — the JitCode byte offset.
    pub pc: i32,
    pub values: Vec<RebuiltValue>,
}

/// Decode one tagged resume value outside the numbered frame stream.
///
/// Pending fields carry their target and value through the same
/// TAGBOX/TAGCONST/TAGVIRTUAL encoding as frame slots
/// (`resume.py:_add_pending_fields`).  Bridge setup uses this entry point so
/// its consume-side view is identical to `rebuild_from_numbering`.
pub fn decode_tagged_value(
    tagged: i16,
    num_failargs: i32,
    rd_consts: &[Const],
    fail_arg_types: &[Type],
    num_virtuals: usize,
) -> RebuiltValue {
    let (val, tagbits) = untag(tagged);
    match tagbits {
        // resume.py:1257 ConstInt(num) — TAGINT always produces an int box.
        TAGINT => RebuiltValue::Const(crate::Const::Int(val as i64)),
        TAGCONST => {
            if tagged == NULLREF {
                // history.py:361 CONST_NULL = ConstPtr(null).
                RebuiltValue::Const(crate::Const::Ref(crate::GcRef::NULL))
            } else if tagged == UNINITIALIZED_TAG {
                RebuiltValue::Unassigned
            } else {
                let idx = (val - TAG_CONST_OFFSET) as usize;
                // resume.py:1555/1571/1583 self.consts[num - TAG_CONST_OFFSET]
                // — the Const carries its type with it.
                let c = rd_consts.get(idx).copied().unwrap_or(Const::Int(0));
                RebuiltValue::Const(c)
            }
        }
        TAGBOX => {
            let index = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            // resume.py:1245 decode_box(num, kind) parity — pull the
            // box's `.type` from the parent guard's fail_arg_types,
            // which `_number_boxes` populated when encoding via
            // `env.get_type(opref)`. resume.py:1264 asserts
            // `box.type == kind`; a missing entry means the
            // fail_arg_types vector is shorter than the encoded box
            // count (encoder/decoder disagreement), which must fail
            // loudly rather than silently recover as Int.
            let kind = *fail_arg_types.get(index).unwrap_or_else(|| {
                panic!(
                    "resume decode_box: fail_arg_types[{index}] missing (len {}); \
                     encoder/decoder failarg-count disagreement (resume.py:1245)",
                    fail_arg_types.len(),
                )
            });
            RebuiltValue::Box(index, kind)
        }
        TAGVIRTUAL => {
            // resume.py:278-284 assign_number_to_virtual numbers nested
            // virtuals negatively; resume.py:951-954 getvirtual_ptr resolves
            // them via Python negative list indexing into rd_virtuals.
            let index = if val < 0 {
                (num_virtuals as i32 + val) as usize
            } else {
                val as usize
            };
            RebuiltValue::Virtual(index)
        }
        _ => RebuiltValue::Unassigned,
    }
}

/// Decode rd_numb back into vable/vref values and per-frame tagged values.
///
/// resume.py:249-253, resume.py:1049-1055: RPython encodes frames as
/// `jitcode_index, pc, [tagged_values...]` and uses jitcode liveness
/// (`get_current_position_info`) at the decode site to know how many
/// values each frame has.
///
/// `frame_value_count`: when `Some(f)`, `f(jitcode_index, pc)`
/// returns the number of tagged values for that frame (RPython parity:
/// liveness-driven decode). When `None`, all remaining items after
/// `(jitcode_index, pc)` are consumed as a single frame (backward-compat for
/// callers that only ever see single-frame data).
///
/// `fail_arg_types`: parent guard's per-failarg type vector. resume.py:1245
/// `decode_box(num, kind)` parity — TAGBOX values use this to fill in their
/// kind so the resulting `RebuiltValue::Box` carries its own type and
/// downstream consumers don't need a parallel side channel.
pub fn rebuild_from_numbering(
    rd_numb: &[u8],
    rd_consts: &[Const],
    fail_arg_types: &[Type],
    frame_value_count: Option<&dyn Fn(i32, i32) -> usize>,
    num_virtuals: usize,
) -> (i32, Vec<RebuiltValue>, Vec<RebuiltValue>, Vec<RebuiltFrame>) {
    let mut reader = resumecode::Reader::new(rd_numb);

    let total_size = reader.next_item();
    let num_failargs = reader.next_item();

    // resume.py:1045: consume_vref_and_vable_boxes — virtualizable array.
    let vable_len = reader.next_item();
    let mut vable_values = Vec::new();
    for _ in 0..vable_len {
        if !reader.has_more() {
            break;
        }
        let tagged = reader.next_item() as i16;
        vable_values.push(decode_tagged_value(
            tagged,
            num_failargs,
            rd_consts,
            fail_arg_types,
            num_virtuals,
        ));
    }

    // resume.py:1045: virtualref array (pairs).
    let vref_len = reader.next_item();
    let mut vref_values = Vec::new();
    for _ in 0..(vref_len * 2) {
        if !reader.has_more() {
            break;
        }
        let tagged = reader.next_item() as i16;
        vref_values.push(decode_tagged_value(
            tagged,
            num_failargs,
            rd_consts,
            fail_arg_types,
            num_virtuals,
        ));
    }

    // resume.py:1049-1055: frame section — jitcode_index, pc, [tagged_values...].
    // RPython uses consume_one_section → enumerate_vars(liveness) to split frames.
    let mut frames = Vec::new();
    while reader.items_read < total_size as usize && reader.has_more() {
        let jitcode_index = reader.next_item();
        let pc = if reader.has_more() && reader.items_read < total_size as usize {
            reader.next_item()
        } else {
            0
        };
        let box_count = if let Some(f) = &frame_value_count {
            // RPython parity: liveness-driven frame boundary.
            f(jitcode_index, pc)
        } else {
            // Single-frame fallback: consume all remaining items.
            (total_size as usize).saturating_sub(reader.items_read)
        };
        let mut values = Vec::with_capacity(box_count);
        for _ in 0..box_count {
            if !reader.has_more() {
                break;
            }
            let tagged = reader.next_item() as i16;
            values.push(decode_tagged_value(
                tagged,
                num_failargs,
                rd_consts,
                fail_arg_types,
                num_virtuals,
            ));
        }
        frames.push(RebuiltFrame {
            jitcode_index,
            pc,
            values,
        });
    }
    (num_failargs, vable_values, vref_values, frames)
}

#[cfg(test)]
mod after_residual_call_pc_tests {
    use super::{AFTER_RESIDUAL_CALL_PC_FLAG, decode_resume_pc, encode_after_residual_call_pc};

    #[test]
    fn marker_roundtrips_in_range() {
        // Every Python PC below the marker bit round-trips intact as a
        // marked resume pc.
        for pc in [0, 1, 100, AFTER_RESIDUAL_CALL_PC_FLAG - 1] {
            let encoded = encode_after_residual_call_pc(pc);
            assert_eq!(decode_resume_pc(encoded), (pc, true));
        }
    }

    #[test]
    fn unmarked_pc_below_flag_decodes_unmarked() {
        // An unmarked resume pc that leaves bit 14 free decodes as itself.
        for pc in [0, 1, 100, AFTER_RESIDUAL_CALL_PC_FLAG - 1] {
            assert_eq!(decode_resume_pc(pc), (pc, false));
        }
    }

    #[test]
    fn negative_sentinel_passes_through_unmarked() {
        // resume.rs uses negative pcs as sentinels; they must not read as
        // marked (decode_resume_pc:74-83).
        assert_eq!(decode_resume_pc(-1), (-1, false));
    }

    #[test]
    #[should_panic(expected = "out of range for marker bit")]
    fn encode_rejects_pc_at_flag_boundary() {
        // A pc >= 1 << 14 cannot carry the marker; encoding must reject it
        // in release builds too (the bound is `assert!`, not debug-only)
        // rather than silently aliasing a smaller pc on decode.
        let _ = encode_after_residual_call_pc(AFTER_RESIDUAL_CALL_PC_FLAG);
    }

    #[test]
    fn raw_pc_at_flag_aliases_a_marked_pc() {
        // Documents WHY the capture-site asserts bound every *unmarked* pc:
        // a raw pc with bit 14 set is indistinguishable from a marked pc at
        // decode, so an unmarked stored pc >= the flag would be mis-read.
        assert_eq!(decode_resume_pc(AFTER_RESIDUAL_CALL_PC_FLAG), (0, true));
    }
}

#[cfg(test)]
mod branch_orgpc_tag_tests {
    use super::{BRANCH_ORGPC_MAX, NO_JITCODE_PC, decode_branch_orgpc, encode_branch_orgpc};

    #[test]
    fn roundtrips_across_orgpc_and_flavor() {
        // #73 S3.5: every (orgpc, flavor) in range tags into the negative space
        // and decodes back exactly, including the two endpoints.
        for orgpc in [
            0usize,
            1,
            2,
            7,
            100,
            8191,
            BRANCH_ORGPC_MAX - 1,
            BRANCH_ORGPC_MAX,
        ] {
            for flavor in [false, true] {
                let word = encode_branch_orgpc(orgpc, flavor);
                assert!(word <= -2, "tagged word {word} must land in negative space");
                assert_eq!(decode_branch_orgpc(word), Some((orgpc, flavor)));
            }
        }
    }

    #[test]
    fn tagged_words_stay_in_i16_range() {
        // The word is serialized by `append_int`, which asserts i16 range.
        for orgpc in [0usize, BRANCH_ORGPC_MAX] {
            for flavor in [false, true] {
                let word = encode_branch_orgpc(orgpc, flavor);
                assert!(
                    word >= i16::MIN as i32 && word <= i16::MAX as i32,
                    "tagged word {word} out of i16 range"
                );
            }
        }
    }

    #[test]
    fn non_tagged_words_decode_none() {
        // Offsets (>=0) and NO_JITCODE_PC (-1) are NOT tagged: decode returns
        // None so the expand at decode is a pure no-op for them.
        for word in [0, 5, 32767, NO_JITCODE_PC] {
            assert_eq!(decode_branch_orgpc(word), None);
        }
    }
}
