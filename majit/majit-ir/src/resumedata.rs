//! Resume data numbering primitives.
//!
//! Direct port of rpython/jit/metainterp/resume.py tag/numbering infrastructure.
//! Shared between majit-opt (store_final_boxes_in_guard) and majit-meta (rebuild).

use std::collections::HashMap;

use crate::resumecode;
use crate::{BoxEnv, OpRef, Type};

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

#[derive(Debug)]
pub struct TagOverflow;

/// resume.py:96-100
pub fn tag(value: i32, tagbits: u8) -> Result<i16, TagOverflow> {
    debug_assert!(tagbits <= 3);
    let sx = value >> 13;
    if sx != 0 && sx != -1 {
        return Err(TagOverflow);
    }
    Ok(((value << 2) | tagbits as i32) as i16)
}

/// resume.py:106-109
pub fn untag(value: i16) -> (i32, u8) {
    let widened = value as i32;
    let tagbits = (widened & TAGMASK as i32) as u8;
    (widened >> 2, tagbits)
}

// ── Snapshot types for numbering ──

/// One frame in a snapshot's frame chain.
/// resume.py: Snapshot → framestack entries.
#[derive(Debug, Clone)]
pub struct SnapshotFrame {
    pub jitcode_index: i32,
    pub pc: i32,
    pub boxes: Vec<OpRef>,
}

/// Full snapshot for numbering.
/// resume.py:234-253 layout.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub vable_array: Vec<OpRef>,
    pub vref_array: Vec<OpRef>,
    pub framestack: Vec<SnapshotFrame>,
}

impl Snapshot {
    pub fn estimated_size(&self) -> usize {
        let mut n = 4; // header ints
        n += self.vable_array.len();
        n += self.vref_array.len();
        for f in &self.framestack {
            n += 3 + f.boxes.len(); // jitcode_index + pc + slot_count + boxes
        }
        n
    }

    /// Create a single-frame snapshot from a flat list of OpRefs.
    pub fn single_frame(pc: i32, boxes: Vec<OpRef>) -> Self {
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

    /// Create a multi-frame snapshot from per-frame (jitcode_index, pc, boxes) tuples.
    /// Used when a guard fires inside an inlined callee and the snapshot
    /// contains [callee_section..., caller_section...].
    pub fn multi_frame(frames: Vec<(i32, i32, Vec<OpRef>)>) -> Self {
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
}

// ── NumberingState ──

/// resume.py:70-98 NumberingState equivalent.
pub struct NumberingState {
    pub writer: resumecode::Writer,
    /// Maps OpRef.0 → tagged value (TAGBOX/TAGCONST/TAGVIRTUAL).
    pub liveboxes: HashMap<u32, i16>,
    pub num_boxes: i32,
    pub num_virtuals: i32,
}

impl NumberingState {
    pub fn new(size: usize) -> Self {
        NumberingState {
            writer: resumecode::Writer::new(size),
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

    pub fn patch(&mut self, index: usize, item: i32) {
        self.writer.patch(index, item);
    }

    pub fn create_numbering(&self) -> Vec<u8> {
        self.writer.create_numbering()
    }
}

// ── ResumeDataLoopMemo (numbering part) ──

/// resume.py:145-165 ResumeDataLoopMemo — shared constant pool + numbering.
pub struct ResumeDataLoopMemo {
    /// resume.py:147 — constant pool (value, type).
    consts: Vec<(i64, Type)>,
    /// resume.py:148 — large integers → tagged const.
    large_ints: HashMap<i64, i16>,
    /// resume.py:149 — ref pointers → tagged const.
    refs: HashMap<i64, i16>,
}

impl ResumeDataLoopMemo {
    pub fn new() -> Self {
        ResumeDataLoopMemo {
            consts: Vec::new(),
            large_ints: HashMap::new(),
            refs: HashMap::new(),
        }
    }

    pub fn consts(&self) -> &[(i64, Type)] {
        &self.consts
    }

    fn newconst(&mut self, val: i64, tp: Type) -> i16 {
        let idx = self.consts.len() as i32 + TAG_CONST_OFFSET;
        self.consts.push((val, tp));
        tag(idx, TAGCONST).unwrap_or(NULLREF)
    }

    /// resume.py:172-181
    fn getconst_int(&mut self, val: i64) -> i16 {
        // Small ints → TAGINT
        let small = val as i32;
        if small as i64 == val {
            if let Ok(t) = tag(small, TAGINT) {
                return t;
            }
        }
        if let Some(&t) = self.large_ints.get(&val) {
            return t;
        }
        let t = self.newconst(val, Type::Int);
        self.large_ints.insert(val, t);
        t
    }

    fn getconst_ref(&mut self, val: i64) -> i16 {
        // resume.py:174-176: null ref → NULLREF
        if val == 0 {
            return NULLREF;
        }
        if let Some(&t) = self.refs.get(&val) {
            return t;
        }
        let t = self.newconst(val, Type::Ref);
        self.refs.insert(val, t);
        t
    }

    fn getconst_float(&mut self, val: i64) -> i16 {
        self.newconst(val, Type::Float)
    }

    /// resume.py:186-192
    pub fn getconst(&mut self, val: i64, tp: Type) -> i16 {
        match tp {
            Type::Int => self.getconst_int(val),
            Type::Ref => self.getconst_ref(val),
            Type::Float => self.getconst_float(val),
            Type::Void => self.newconst(val, tp),
        }
    }

    /// resume.py:196-223 _number_boxes — tag each box in the snapshot.
    ///
    /// KNOWN DEVIATION: RPython's read_boxes/wrap() (virtualizable.py:86)
    /// creates fresh Box objects for virtualizable_boxes with separate
    /// Python identity from frame register boxes. _number_boxes
    /// (resume.py:207) keys by box identity after get_box_replacement(),
    /// so fresh vable boxes MAY get separate TAGBOX indices if their
    /// forwarding chains diverge from the frame register boxes.
    ///
    /// In pyre, OpRefs are integers shared between vable and frame sections.
    /// Dedup always occurs. This is a CONFIRMED deviation: when two vable
    /// slots share a TAGBOX with a frame register slot, recovery gives all
    /// three the same deadframe value, corrupting distinct vable slots.
    /// (Observed: fib_loop locals[1] and locals[2] get identical pointers.)
    ///
    /// Consequence: synchronize_virtualizable (vable→PyFrame writeback) is
    /// BLOCKED — only frame-section recovery (liveness-based) is safe.
    /// Fix requires fresh OpRef identity for vable (SameAs-based encoding
    /// at trace recording time, with backend exit block recompilation).
    pub fn _number_boxes(
        &mut self,
        boxes: &[OpRef],
        numb_state: &mut NumberingState,
        env: &dyn BoxEnv,
    ) -> Result<(), TagOverflow> {
        for &raw_opref in boxes {
            // resume.py:561-562 _gettagged: box is None → UNINITIALIZED
            if raw_opref.is_none() {
                numb_state.append_short(UNINITIALIZED_TAG);
                continue;
            }

            // resume.py:201-202
            let opref = env.get_box_replacement(raw_opref);

            // resume.py:561-562: forwarded-away box → UNINITIALIZED
            if opref.is_none() {
                numb_state.append_short(UNINITIALIZED_TAG);
                continue;
            }

            // resume.py:204-205: isinstance(box, Const)
            if env.is_const(opref) {
                let (val, tp) = env.get_const(opref);
                let tagged = self.getconst(val, tp);
                numb_state.append_short(tagged);
                continue;
            }
            // resume.py:207-208: already seen
            if let Some(&tagged) = numb_state.liveboxes.get(&opref.0) {
                numb_state.append_short(tagged);
                continue;
            }
            // resume.py:210-216: virtual check
            let is_virtual = match env.get_type(opref) {
                Type::Ref => env.is_virtual_ref(opref),
                Type::Int => env.is_virtual_raw(opref),
                _ => false,
            };
            // resume.py:217-223: tag
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

    /// resume.py:228-256 number() — serialize a snapshot into NumberingState.
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
        // Per-frame: jitcode_index, pc, box_count, [tagged_values...].
        // box_count is majit-specific — RPython omits it and uses
        // jitcode.get_live_vars_info(pc) at decode time instead.
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
}

// ── Decoding (for guard failure recovery) ──

/// Decoded value from rd_numb.
#[derive(Debug, Clone, PartialEq)]
pub enum RebuiltValue {
    /// TAGBOX(n): value from deadframe slot n.
    Box(usize),
    /// TAGCONST: compile-time constant with type.
    Const(i64, Type),
    /// TAGINT: small integer encoded inline.
    Int(i32),
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

fn decode_tagged(tagged: i16, num_failargs: i32, rd_consts: &[(i64, Type)]) -> RebuiltValue {
    let (val, tagbits) = untag(tagged);
    match tagbits {
        TAGINT => RebuiltValue::Int(val),
        TAGCONST => {
            if tagged == NULLREF {
                RebuiltValue::Const(0, Type::Ref)
            } else if tagged == UNINITIALIZED_TAG {
                RebuiltValue::Unassigned
            } else {
                let idx = (val - TAG_CONST_OFFSET) as usize;
                let (c, tp) = rd_consts.get(idx).copied().unwrap_or((0, Type::Int));
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
pub fn rebuild_from_numbering(
    rd_numb: &[u8],
    rd_consts: &[(i64, Type)],
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
        vable_values.push(decode_tagged(tagged, num_failargs, rd_consts));
    }

    // resume.py:1045: virtualref array (pairs).
    let vref_len = reader.next_item();
    let mut vref_values = Vec::new();
    for _ in 0..(vref_len * 2) {
        if !reader.has_more() {
            break;
        }
        let tagged = reader.next_item() as i16;
        vref_values.push(decode_tagged(tagged, num_failargs, rd_consts));
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
            values.push(decode_tagged(tagged, num_failargs, rd_consts));
        }
        frames.push(RebuiltFrame {
            jitcode_index,
            pc,
            values,
        });
    }
    (num_failargs, vable_values, vref_values, frames)
}
