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
    pub pc: i32,
    pub values: Vec<RebuiltValue>,
}

fn decode_tagged(
    tagged: i16,
    num_failargs: i32,
    rd_consts: &[Const],
    fail_arg_types: &[Type],
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
            // `box.type == kind`; a mismatch here means the
            // fail_arg_types vector is shorter than expected (encoder/
            // decoder count disagreement).
            let kind = fail_arg_types.get(index).copied().unwrap_or_else(|| {
                panic!(
                    "decode_tagged: TAGBOX index {} out of fail_arg_types (len {})",
                    index,
                    fail_arg_types.len()
                )
            });
            RebuiltValue::Box(index, kind)
        }
        TAGVIRTUAL => RebuiltValue::Virtual(val as usize),
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
/// `frame_value_count`: when `Some(f)`, `f(jitcode_index, pc)` returns
/// the number of tagged values for that frame (RPython parity: liveness-
/// driven decode). When `None`, all remaining items after `(jitcode_index,
/// pc)` are consumed as a single frame (backward-compat for callers that
/// only ever see single-frame data).
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
        vable_values.push(decode_tagged(
            tagged,
            num_failargs,
            rd_consts,
            fail_arg_types,
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
        vref_values.push(decode_tagged(
            tagged,
            num_failargs,
            rd_consts,
            fail_arg_types,
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
            values.push(decode_tagged(
                tagged,
                num_failargs,
                rd_consts,
                fail_arg_types,
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
