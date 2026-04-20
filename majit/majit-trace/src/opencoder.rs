/// Compact binary encoding for traces.
///
/// Translates the concept from rpython/jit/metainterp/opencoder.py —
/// a compact binary format for serializing and deserializing traces.
///
/// Uses LEB128 variable-length integer encoding for compactness.
use std::collections::HashMap;

use majit_ir::{InputArg, OPCODE_COUNT, Op, OpCode, OpRef, Type};

use crate::history::TreeLoop;

/// Encode a u64 as a variable-length integer (LEB128).
pub fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

/// Decode a varint from a byte slice. Returns (value, bytes_consumed).
///
/// # Panics
///
/// Panics if the buffer is truncated (no terminating byte with high bit clear).
pub fn decode_varint(buf: &[u8]) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in buf.iter().enumerate() {
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return (value, i + 1);
        }
        shift += 7;
    }
    panic!("truncated varint");
}

fn type_to_u8(tp: Type) -> u8 {
    match tp {
        Type::Int => 0,
        Type::Ref => 1,
        Type::Float => 2,
        Type::Void => 3,
    }
}

fn u8_to_type(v: u8) -> Type {
    match v {
        0 => Type::Int,
        1 => Type::Ref,
        2 => Type::Float,
        3 => Type::Void,
        _ => panic!("invalid type byte: {v}"),
    }
}

fn u16_to_opcode(v: u16) -> OpCode {
    assert!(
        (v as usize) < OPCODE_COUNT,
        "invalid opcode discriminant: {v}"
    );
    // SAFETY: OpCode is #[repr(u16)] and we checked the discriminant is in range.
    unsafe { std::mem::transmute(v) }
}

/// Encode a Trace into a compact byte buffer.
pub fn encode_trace(trace: &TreeLoop) -> Vec<u8> {
    let mut buf = Vec::new();

    // Encode input args count and types
    encode_varint(&mut buf, trace.inputargs.len() as u64);
    for arg in &trace.inputargs {
        buf.push(type_to_u8(arg.tp));
    }

    // Encode ops count
    encode_varint(&mut buf, trace.ops.len() as u64);

    // Encode each op
    for op in &trace.ops {
        encode_varint(&mut buf, op.opcode as u16 as u64);
        encode_varint(&mut buf, op.args.len() as u64);
        for arg in &op.args {
            encode_varint(&mut buf, arg.0 as u64);
        }
        // opencoder.py:702-707 _encode_descr: 0 = none, else descr_index + 1.
        // RPython uses descr.get_descr_index() (global all_descrs index),
        // NOT descr.index() (parent-local slot).
        if let Some(ref descr) = op.descr {
            let descr_index = descr.get_descr_index();
            if descr_index != -1 {
                encode_varint(&mut buf, (descr_index as u64) + 1);
            } else {
                buf.push(1); // has descriptor but no global index
            }
        } else {
            buf.push(0);
        }
        // Encode fail_args presence and count.
        if let Some(ref fa) = op.fail_args {
            encode_varint(&mut buf, fa.len() as u64 + 1); // +1 to distinguish from 0=none
            for arg in fa.iter() {
                encode_varint(&mut buf, arg.0 as u64);
            }
        } else {
            buf.push(0);
        }
    }

    buf
}

/// Decode a Trace from a compact byte buffer.
///
/// Note: descriptor references are not preserved — ops with descriptors
/// in the original trace will have `descr: None` after decoding, but
/// the has-descriptor flag is still decoded (and could be used to
/// reconstruct descriptors from a separate table).
pub fn decode_trace(buf: &[u8]) -> TreeLoop {
    let mut pos = 0;

    // Decode input args
    let (num_inputargs, n) = decode_varint(&buf[pos..]);
    pos += n;
    let num_inputargs = num_inputargs as usize;

    let mut inputargs = Vec::with_capacity(num_inputargs);
    for i in 0..num_inputargs {
        let tp = u8_to_type(buf[pos]);
        pos += 1;
        inputargs.push(InputArg::from_type(tp, i as u32));
    }

    // Decode ops
    let (num_ops, n) = decode_varint(&buf[pos..]);
    pos += n;
    let num_ops = num_ops as usize;

    let mut ops = Vec::with_capacity(num_ops);
    for _ in 0..num_ops {
        let (opcode_raw, n) = decode_varint(&buf[pos..]);
        pos += n;
        let opcode = u16_to_opcode(opcode_raw as u16);

        let (num_args, n) = decode_varint(&buf[pos..]);
        pos += n;
        let num_args = num_args as usize;

        let mut args = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            let (arg_ref, n) = decode_varint(&buf[pos..]);
            pos += n;
            args.push(OpRef(arg_ref as u32));
        }

        // Decode descriptor index
        let (descr_marker, n) = decode_varint(&buf[pos..]);
        pos += n;
        let _descr_index = if descr_marker > 0 {
            Some((descr_marker - 1) as u32)
        } else {
            None
        };

        // Decode fail_args
        let (fa_marker, n) = decode_varint(&buf[pos..]);
        pos += n;
        let fail_args = if fa_marker > 0 {
            let fa_len = (fa_marker - 1) as usize;
            let mut fa = Vec::with_capacity(fa_len);
            for _ in 0..fa_len {
                let (arg_ref, n) = decode_varint(&buf[pos..]);
                pos += n;
                fa.push(OpRef(arg_ref as u32));
            }
            Some(fa)
        } else {
            None
        };

        let mut op = Op::new(opcode, &args);
        if let Some(fa) = fail_args {
            op.fail_args = Some(fa.into());
        }
        ops.push(op);
    }

    TreeLoop::new(inputargs, ops)
}

// ── Signed varint encoding (opencoder.py:59-89) ──
//
// Literal port of rpython/jit/metainterp/opencoder.py encode_varint_signed
// / decode_varint_signed. Either 2 bytes or 4 bytes, chosen by range:
//
//   2 bytes: value in [-2^14, 2^14-1]  (bit 7 of byte 0 == 0)
//     byte 0: low 7 bits of value; bit 7 = 0
//     byte 1: bits 7..14 of value; bit 7 of byte 1 acts as sign bit
//
//   4 bytes: value in [MIN_VALUE, MAX_VALUE] = [-2^30, 2^30-1]
//     byte 0: low 7 bits of value; bit 7 = 1 (flag)
//     byte 1: bits 7..14 of value
//     byte 2: bits 15..22 of value
//     byte 3: bits 23..30 of value; bit 7 of byte 3 is sign bit
//
// Decoding sign-extends from bit (shift) using the top bit of the last
// written byte.
//
// This replaces the pre-Phase-B zigzag LEB128 encoder so the pyre wire
// format binary-matches RPython.

/// opencoder.py:59-73 encode_varint_signed.
pub fn encode_varint_signed(buf: &mut Vec<u8>, value: i64) {
    debug_assert!(
        MIN_VALUE <= value && value <= MAX_VALUE,
        "encode_varint_signed out of range: {value}"
    );
    let mut v = value;
    let flag: u8 = if !(-(1 << 14)..(1 << 14)).contains(&v) {
        0x80
    } else {
        0
    };
    buf.push(((v & 0b0111_1111) as u8) | flag);
    v >>= 7;
    buf.push((v & 0xff) as u8);
    if flag != 0 {
        v >>= 8;
        buf.push((v & 0xff) as u8);
        v >>= 8;
        buf.push((v & 0xff) as u8);
    }
}

/// opencoder.py:75-89 decode_varint_signed. Returns (value, bytes_consumed).
pub fn decode_varint_signed(buf: &[u8]) -> (i64, usize) {
    let byte0 = buf[0];
    let byte1 = buf[1];
    let mut res: i64 = ((byte0 & 0b0111_1111) as i64) | ((byte1 as i64) << 7);
    let mut shift: u32 = 15;
    let mut index: usize = 2;
    let mut lastbyte = byte1;
    if byte0 & 0b1000_0000 != 0 {
        let byte2 = buf[index];
        let byte3 = buf[index + 1];
        res |= ((byte2 as i64) << 15) | ((byte3 as i64) << 23);
        shift = 31;
        index += 2;
        lastbyte = byte3;
    }
    // sign-extend: top bit of the last written byte is the sign bit.
    if lastbyte & 0b1000_0000 != 0 {
        res |= -1i64 << shift;
    }
    (res, index)
}

// ── Snapshot management (opencoder.py) ──

/// Tag constants for snapshot encoding.
/// opencoder.py: TAGINT, TAGCONSTPTR, TAGCONSTOTHER, TAGBOX
pub const TAGINT: u8 = 0; // small integer constant
pub const TAGCONSTPTR: u8 = 1; // GC pointer constant
pub const TAGCONSTOTHER: u8 = 2; // big int or float constant
pub const TAGBOX: u8 = 3; // reference to a traced value (OpRef)

/// Number of tag bits.
pub const TAG_SHIFT: u32 = 2;
/// Mask for tag extraction.
pub const TAG_MASK: u8 = (1 << TAG_SHIFT) - 1;
/// RPython compatibility constants.
pub const TAGMASK: u8 = TAG_MASK;
pub const TAGSHIFT: u32 = TAG_SHIFT;

/// Initial trace buffer size.
pub const INIT_SIZE: usize = 4096;

/// RPython compatibility numeric limits for signed varints in this model.
pub const MIN_VALUE: i64 = -(1 << 30);
pub const MAX_VALUE: i64 = (1 << 30) - 1;

/// RPython compatibility bounds for small int tagging.
pub const SMALL_INT_START: i64 = -(1 << 28);
pub const SMALL_INT_STOP: i64 = 1 << 28;

/// RPython compatibility sentinel values for snapshot linked lists.
pub const SNAPSHOT_PREV_NEEDS_PATCHING: i32 = -3;
pub const SNAPSHOT_PREV_NONE: i32 = -2;
pub const SNAPSHOT_PREV_COMES_NEXT: i32 = -1;

/// opencoder.py:91-100 skip_varint_signed.
///
/// Each varint is 2 bytes if bit 7 of byte 0 is clear, 4 bytes if set.
/// `skip` is the number of varints to skip over; returns the new cursor.
#[inline]
pub fn skip_varint_signed(buf: &[u8], mut index: usize, skip: usize) -> usize {
    assert!(skip > 0);
    let mut remaining = skip;
    loop {
        let byte = buf[index];
        if byte & 0b1000_0000 != 0 {
            index += 2;
        }
        index += 2;
        remaining -= 1;
        if remaining == 0 {
            return index;
        }
    }
}

#[inline]
pub fn varint_only_decode(buf: &[u8], index: usize, skip: usize) -> i64 {
    let start = if skip > 0 {
        skip_varint_signed(buf, index, skip)
    } else {
        index
    };
    decode_varint_signed(&buf[start..]).0
}

#[inline]
pub fn combine_uint(index1: u32, index2: u32) -> u32 {
    (index1 << 16) | index2
}

#[inline]
pub fn unpack_uint(packed: u32) -> (u32, u32) {
    ((packed >> 16) & 0xffff, packed & 0xffff)
}

/// Encode a tagged value for snapshot storage.
/// opencoder.py: tag(kind, value)
pub fn tag(kind: u8, value: u32) -> u32 {
    (value << TAG_SHIFT) | kind as u32
}

/// Decode a tagged value from snapshot storage.
/// opencoder.py: untag(tagged) -> (kind, value)
pub fn untag(tagged: u32) -> (u8, u32) {
    let kind = (tagged & TAG_MASK as u32) as u8;
    let value = tagged >> TAG_SHIFT;
    (kind, value)
}

/// A snapshot captures the interpreter state at a guard/exit point.
/// opencoder.py: Snapshot
///
/// When a guard fails, the JIT needs to reconstruct the interpreter
/// state (frames, local variables) from the snapshot to resume
/// execution in the interpreter.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// Encoded values (tagged): local variables and virtual object fields.
    pub values: Vec<u32>,
    /// Index of the previous snapshot in the chain (for call stack).
    pub prev: Option<usize>,
    /// Jitcode index identifying which interpreter function this frame belongs to.
    pub jitcode_index: u32,
    /// Program counter (bytecode offset) within the jitcode.
    pub pc: u32,
}

/// Top-level snapshot (includes vable/vref tracking).
/// opencoder.py: TopSnapshot
#[derive(Clone, Debug)]
pub struct TopSnapshot {
    /// The snapshot for the topmost frame.
    pub snapshot: Snapshot,
    /// Index into snapshot data for virtualizable array.
    pub vable_array_index: Option<usize>,
    /// Index into snapshot data for virtual ref array.
    pub vref_array_index: Option<usize>,
}

/// opencoder.py: SnapshotIterator — iterates snapshot values
/// in bottom-up frame order (inner frame first).
pub struct SnapshotIterator<'a> {
    storage: &'a SnapshotStorage,
    current_snapshot_idx: Option<usize>,
}

impl<'a> SnapshotIterator<'a> {
    pub fn new(storage: &'a SnapshotStorage, top_snapshot_idx: usize) -> Self {
        let start = if top_snapshot_idx < storage.top_snapshots.len() {
            Some(storage.snapshots.len() - 1) // start from innermost
        } else {
            None
        };
        SnapshotIterator {
            storage,
            current_snapshot_idx: start,
        }
    }

    /// Get the current snapshot frame.
    pub fn current(&self) -> Option<&'a Snapshot> {
        self.current_snapshot_idx
            .and_then(|idx| self.storage.snapshots.get(idx))
    }

    /// Move to the outer (caller) frame.
    pub fn next_frame(&mut self) -> Option<&'a Snapshot> {
        let idx = self.current_snapshot_idx?;
        let snap = self.storage.snapshots.get(idx)?;
        self.current_snapshot_idx = snap.prev;
        self.current()
    }

    /// Decode a tagged value from the snapshot.
    pub fn decode_value(&self, tagged: u32) -> DecodedSnapshotValue {
        let (kind, value) = untag(tagged);
        match kind {
            TAGINT => DecodedSnapshotValue::SmallInt(value as i64),
            TAGCONSTPTR => DecodedSnapshotValue::ConstPtr(
                self.storage
                    .const_refs
                    .get(value as usize)
                    .copied()
                    .unwrap_or(0),
            ),
            TAGCONSTOTHER => DecodedSnapshotValue::ConstOther(value),
            TAGBOX => DecodedSnapshotValue::Box(majit_ir::OpRef(value)),
            _ => DecodedSnapshotValue::SmallInt(0),
        }
    }
}

/// Decoded value from a snapshot.
#[derive(Clone, Debug)]
pub enum DecodedSnapshotValue {
    /// Small integer constant (fits in tag bits).
    SmallInt(i64),
    /// GC pointer constant (index into const_refs pool).
    ConstPtr(u64),
    /// Other constant (big int or float, index into pool).
    ConstOther(u32),
    /// Reference to a traced value (OpRef).
    Box(majit_ir::OpRef),
}

/// opencoder.py: TopDownSnapshotIterator — iterates snapshots
/// from outermost to innermost frame (top-down order).
/// Used by resume data construction which processes frames from
/// the outermost caller to the innermost callee.
pub struct TopDownSnapshotIterator<'a> {
    storage: &'a SnapshotStorage,
    snapshot_index: usize,
    vable_array_index: Option<usize>,
    vref_array_index: Option<usize>,
    /// Frames collected in bottom-up order, then reversed.
    frames: Vec<usize>,
    pos: usize,
}

impl<'a> TopDownSnapshotIterator<'a> {
    /// Create a top-down iterator starting from a top snapshot.
    pub fn new(storage: &'a SnapshotStorage, top_idx: usize) -> Self {
        let mut frames = Vec::new();
        let mut vable_array_index = None;
        let mut vref_array_index = None;
        if top_idx < storage.top_snapshots.len() {
            vable_array_index = storage.top_snapshots[top_idx].vable_array_index;
            vref_array_index = storage.top_snapshots[top_idx].vref_array_index;
            // Walk the prev chain to collect all frames bottom-up
            let first_snap_idx = storage
                .top_snapshots
                .get(top_idx)
                .and_then(|top| top.snapshot.prev)
                .unwrap_or(0);
            let mut idx = Some(first_snap_idx);
            while let Some(i) = idx {
                if i < storage.snapshots.len() {
                    frames.push(i);
                    idx = storage.snapshots[i].prev;
                } else {
                    break;
                }
            }
            // Reverse for top-down order (outermost first)
            frames.reverse();
        }
        TopDownSnapshotIterator {
            storage,
            snapshot_index: top_idx,
            vable_array_index,
            vref_array_index,
            frames,
            pos: 0,
        }
    }

    fn snapshot_values(&self, index: usize) -> &'a [u32] {
        self.storage
            .snapshots
            .get(index)
            .map(|snapshot| snapshot.values.as_slice())
            .unwrap_or(&[])
    }

    /// opencoder.py: iter_vable_array()
    pub fn iter_vable_array(&self) -> BoxArrayIter<'a> {
        match self.vable_array_index {
            Some(index) => BoxArrayIter::new(self.snapshot_values(index)),
            None => BoxArrayIter::empty(),
        }
    }

    /// opencoder.py: iter_vref_array()
    pub fn iter_vref_array(&self) -> BoxArrayIter<'a> {
        match self.vref_array_index {
            Some(index) => BoxArrayIter::new(self.snapshot_values(index)),
            None => BoxArrayIter::empty(),
        }
    }

    /// opencoder.py: iter_array(snapshot_index)
    pub fn iter_array(&self, snapshot_index: usize) -> BoxArrayIter<'a> {
        BoxArrayIter::new(self.snapshot_values(snapshot_index))
    }

    /// opencoder.py: length(snapshot_index)
    pub fn length(&self, snapshot_index: usize) -> usize {
        self.snapshot_values(snapshot_index).len()
    }

    /// opencoder.py: prev(snapshot_index)
    pub fn prev(&mut self, snapshot_index: usize) -> i32 {
        self.storage
            .snapshots
            .get(snapshot_index)
            .and_then(|snapshot| snapshot.prev)
            .map_or(SNAPSHOT_PREV_NONE, |prev| prev as i32)
    }

    /// opencoder.py: unpack_jitcode_pc(snapshot_index)
    pub fn unpack_jitcode_pc(&self, snapshot_index: usize) -> (u32, u32) {
        self.storage
            .snapshots
            .get(snapshot_index)
            .map(|snapshot| (snapshot.jitcode_index, snapshot.pc))
            .unwrap_or((u32::MAX, u32::MAX))
    }

    /// opencoder.py: is_empty_snapshot(snapshot_index)
    pub fn is_empty_snapshot(&self, snapshot_index: usize) -> bool {
        self.storage
            .snapshots
            .get(snapshot_index)
            .is_none_or(|snapshot| snapshot.values.is_empty())
    }

    /// opencoder.py: decode_snapshot_int()
    pub fn decode_snapshot_int(&mut self) -> i64 {
        self.snapshot_index as i64
    }

    /// Get the next frame (outermost to innermost).
    pub fn next_frame(&mut self) -> Option<&'a Snapshot> {
        if self.pos < self.frames.len() {
            let idx = self.frames[self.pos];
            self.pos += 1;
            self.storage.snapshots.get(idx)
        } else {
            None
        }
    }

    /// Number of frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Whether all frames have been consumed.
    pub fn done(&self) -> bool {
        self.pos >= self.frames.len()
    }
}

impl<'a> Iterator for TopDownSnapshotIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.frames.len() {
            let idx = self.frames[self.pos];
            self.pos += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Snapshot storage for a trace.
/// opencoder.py: Trace._snapshot_data, _snapshot_array_data
///
/// gcreftracer.py parity: const_refs entries are GC references stored as
/// raw u64. They are rooted on the shadow stack so GC can update them
/// when objects move. refresh_from_gc() copies updated values back.
#[derive(Debug)]
pub struct SnapshotStorage {
    /// All snapshots in this trace, indexed by position.
    pub snapshots: Vec<Snapshot>,
    /// Top snapshots corresponding to guard operations.
    pub top_snapshots: Vec<TopSnapshot>,
    /// Constant pool: GC references (pointers).
    /// opencoder.py: Trace._refs
    pub const_refs: Vec<u64>,
    /// Constant pool: big integers (>28-bit).
    pub const_bigints: Vec<i64>,
    /// Constant pool: float values.
    pub const_floats: Vec<f64>,
    /// (const_refs index, shadow stack index) for each rooted entry.
    /// gcreftracer.py parity: GC's walk_roots updates shadow stack;
    /// refresh_from_gc copies values back to const_refs.
    rooted_ref_indices: Vec<(usize, usize)>,
    /// Shadow stack depth at creation. release_roots pops to here.
    shadow_stack_base: usize,
}

impl Default for SnapshotStorage {
    fn default() -> Self {
        SnapshotStorage {
            snapshots: Vec::new(),
            top_snapshots: Vec::new(),
            const_refs: Vec::new(),
            const_bigints: Vec::new(),
            const_floats: Vec::new(),
            rooted_ref_indices: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
    }
}

impl SnapshotStorage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a snapshot and return its index.
    pub fn add_snapshot(&mut self, snapshot: Snapshot) -> usize {
        let idx = self.snapshots.len();
        self.snapshots.push(snapshot);
        idx
    }

    /// Add a top snapshot (for a guard).
    pub fn add_top_snapshot(&mut self, top: TopSnapshot) -> usize {
        let idx = self.top_snapshots.len();
        self.top_snapshots.push(top);
        idx
    }

    /// Add a GC reference constant and return its pool index.
    /// gcreftracer.py parity: non-null refs are rooted on shadow stack.
    pub fn add_const_ref(&mut self, ptr: u64) -> u32 {
        let cr_idx = self.const_refs.len();
        self.const_refs.push(ptr);
        if ptr != 0 {
            let ss_idx = majit_gc::shadow_stack::push(majit_ir::GcRef(ptr as usize));
            self.rooted_ref_indices.push((cr_idx, ss_idx));
        }
        cr_idx as u32
    }

    /// Add a big integer constant and return its pool index.
    pub fn add_const_bigint(&mut self, value: i64) -> u32 {
        let idx = self.const_bigints.len() as u32;
        self.const_bigints.push(value);
        idx
    }

    /// Add a float constant and return its pool index.
    pub fn add_const_float(&mut self, value: f64) -> u32 {
        let idx = self.const_floats.len() as u32;
        self.const_floats.push(value);
        idx
    }

    /// Total number of snapshots.
    pub fn num_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    /// Update const_refs from shadow stack — GC may have moved objects.
    /// gcreftracer.py:gcrefs_trace parity.
    pub fn refresh_from_gc(&mut self) {
        for &(cr_idx, ss_idx) in &self.rooted_ref_indices {
            self.const_refs[cr_idx] = majit_gc::shadow_stack::get(ss_idx).0 as u64;
        }
    }

    /// Release shadow stack roots.
    fn release_roots(&mut self) {
        if !self.rooted_ref_indices.is_empty() {
            majit_gc::shadow_stack::pop_to(self.shadow_stack_base);
            self.rooted_ref_indices.clear();
        }
    }
}

impl Drop for SnapshotStorage {
    fn drop(&mut self) {
        self.release_roots();
    }
}

// ── Trace Iterator ──
// opencoder.py:249-406 TraceIterator(BaseTrace).
//
// Literal port of RPython's TraceIterator. Each call to `next()` produces
// a *fresh* operation whose `pos` is a freshly allocated OpRef (from the
// majit-specific `_fresh` counter) and whose args have been translated
// through `_cache` from raw trace positions to the per-iteration fresh
// OpRefs.
//
// In RPython every visited op is a freshly allocated `cls()` ResOperation
// Python object whose identity is `is`. Two iterators over the same trace
// therefore produce DIFFERENT box objects at the same `_index` slot — the
// `_index` field walks `_cache` in raw trace position coordinates; the
// freshness comes from `cls()`. majit has no separate identity from
// position, so we split the RPython `_index` into two fields:
//
//   * `_index`  — raw trace position coordinate, matching RPython's
//                 `self._cache[self._index] = res; self._index += 1`. Used
//                 by `replace_last_cached`/`kill_cache_at` which key on
//                 `_cache` slots.
//   * `_fresh`  — majit-specific fresh OpRef allocation counter, seeded
//                 from the `start_fresh` constructor argument. Phase 1
//                 starts at `num_inputs`; Phase 2 starts at Phase 1's
//                 `_fresh` high water so the two iterations produce
//                 disjoint OpRef ranges.
//
// RPython keeps these two roles fused because Python object identity
// makes `_index`'s dual use safe.

/// opencoder.py:249 class TraceIterator(BaseTrace).
pub struct TraceIterator<'a> {
    /// opencoder.py:252 self.trace
    pub trace: &'a [majit_ir::Op],
    /// opencoder.py:255 self._cache: per-iterator map from raw trace
    /// position to fresh box (OpRef) materialized for this iteration.
    /// In RPython this is `[None] * trace._index`; here `Vec<Option<OpRef>>`.
    pub _cache: Vec<Option<OpRef>>,
    /// opencoder.py:259-262 self.inputargs: fresh inputarg boxes for this
    /// iteration. Each is an `OpRef` allocated from the iterator's
    /// `_fresh` counter at construction time.
    pub inputargs: Vec<OpRef>,
    /// opencoder.py:268 self.start
    pub start: usize,
    /// opencoder.py:269 self.pos
    pub pos: usize,
    /// opencoder.py:270 self._count
    pub _count: u32,
    /// opencoder.py:271 self._index: raw trace position coordinate of the
    /// next `_cache` slot to write. RPython uses this as both the cache
    /// slot and the implicit fresh-ResOp position; majit uses it ONLY for
    /// the cache slot (see `_fresh` for the OpRef allocation counter).
    /// Advanced on every non-void op processed by `next()`.
    pub _index: u32,
    /// opencoder.py:272 self.start_index — raw trace position coordinate
    /// of the iteration start.
    pub start_index: u32,
    /// opencoder.py:273 self.end
    pub end: usize,
    /// majit-specific fresh OpRef allocation counter. Seeded from the
    /// `start_fresh` constructor argument. Advances by one per non-void
    /// op in `next()` and per inputarg in `new()`. Two iterations over
    /// the same trace produce disjoint OpRef ranges by passing different
    /// `start_fresh` values.
    pub _fresh: u32,
}

impl<'a> TraceIterator<'a> {
    /// opencoder.py:250-273 TraceIterator.__init__.
    ///
    /// `force_inputargs` corresponds to the same RPython parameter
    /// (cut-trace path); when None, the iterator pre-seeds the cache with
    /// the trace's own inputargs at positions `[0, num_inputargs)`.
    /// `start_fresh` is majit-specific: it seeds the fresh OpRef counter
    /// so two iterations over the same trace can produce disjoint OpRef
    /// ranges. RPython does not need this because each iteration's
    /// `cls()` allocation produces distinct Python objects.
    pub fn new(
        trace: &'a [majit_ir::Op],
        start: usize,
        end: usize,
        force_inputargs: Option<&[OpRef]>,
        num_inputargs: usize,
        start_fresh: u32,
    ) -> Self {
        // self._cache = [None] * trace._index
        // The iterator's cache must be large enough to hold any raw trace
        // position we may encounter. RPython sizes it from `trace._index`
        // (the encoder's monotonic op-result counter); here the equivalent
        // is the maximum raw position seen in `trace[start..end]` plus one.
        let max_pos = trace[start..end]
            .iter()
            .flat_map(|op| {
                std::iter::once(op.pos)
                    .chain(op.args.iter().copied())
                    .chain(op.fail_args.iter().flat_map(|fa| fa.iter().copied()))
            })
            .filter(|opref| !opref.is_none() && !opref.is_constant())
            .map(|opref| opref.0)
            .max()
            .unwrap_or(0);
        let cache_size = ((max_pos as usize) + 1).max(num_inputargs);
        let mut _cache: Vec<Option<OpRef>> = vec![None; cache_size];
        let mut _fresh = start_fresh;
        let inputargs: Vec<OpRef>;
        if let Some(force) = force_inputargs {
            // self.inputargs = [rop.inputarg_from_tp(arg.type) for
            //                   arg in force_inputargs]
            // for i, arg in enumerate(force_inputargs):
            //     self._cache[arg.get_position()] = self.inputargs[i]
            inputargs = force
                .iter()
                .map(|_| {
                    let r = OpRef(_fresh);
                    _fresh += 1;
                    r
                })
                .collect();
            for (i, &arg) in force.iter().enumerate() {
                let p = arg.0 as usize;
                if p >= _cache.len() {
                    _cache.resize(p + 1, None);
                }
                _cache[p] = Some(inputargs[i]);
            }
        } else {
            // self.inputargs = [rop.inputarg_from_tp(arg.type) for
            //                   arg in self.trace.inputargs]
            // for i, arg in enumerate(self.inputargs):
            //     self._cache[self.trace.inputargs[i].get_position()] = arg
            inputargs = (0..num_inputargs)
                .map(|_| {
                    let r = OpRef(_fresh);
                    _fresh += 1;
                    r
                })
                .collect();
            for i in 0..num_inputargs {
                _cache[i] = Some(inputargs[i]);
            }
        }
        TraceIterator {
            trace,
            _cache,
            inputargs,
            start,
            pos: start,
            _count: start as u32,
            _index: start as u32,
            start_index: start as u32,
            end,
            _fresh,
        }
    }

    /// opencoder.py:286-289 _get(self, i).
    fn _get(&self, i: usize) -> OpRef {
        let res = self._cache[i];
        debug_assert!(res.is_some(), "TraceIterator._get cache miss at {i}");
        res.unwrap()
    }

    /// opencoder.py:291-292 done().
    pub fn done(&self) -> bool {
        self.pos >= self.end
    }

    /// opencoder.py:321-335 _untag(tagged).
    ///
    /// In RPython this dispatches on the tag (TAGBOX/TAGINT/TAGCONSTPTR/
    /// TAGCONSTOTHER). majit's OpRef carries the tag implicitly: constants
    /// have `OpRef >= CONST_BASE`, NONE is `u32::MAX`. Both pass through
    /// unchanged; only TAGBOX-equivalent OpRefs go through `_get`.
    fn _untag(&self, opref: OpRef) -> OpRef {
        if opref.is_none() || opref.is_constant() {
            opref
        } else {
            self._get(opref.0 as usize)
        }
    }

    /// opencoder.py:278-280 kill_cache_at(pos).
    pub fn kill_cache_at(&mut self, pos: usize) {
        if pos != 0 {
            self._cache[pos] = None;
        }
    }

    /// opencoder.py:282-284 replace_last_cached(oldbox, box).
    /// RPython writes `_cache[_index - 1]` where `_index` walks `_cache`
    /// densely in ops-count coordinates. In majit the key is the raw
    /// trace position (which is monotonic for non-void ops in a
    /// well-formed trace), so `_index - 1` lands on the same cache slot
    /// the previous `next()` call wrote.
    pub fn replace_last_cached(&mut self, oldbox: OpRef, new_box: OpRef) {
        let last_idx = (self._index - 1) as usize;
        debug_assert_eq!(self._cache[last_idx], Some(oldbox));
        self._cache[last_idx] = Some(new_box);
    }

    /// opencoder.py:362-406 next() — produce the next operation as a fresh
    /// box (`cls()` in RPython) with translated args.
    pub fn next(&mut self) -> Option<majit_ir::Op> {
        if self.done() {
            return None;
        }
        let src = &self.trace[self.pos];
        self.pos += 1;
        let mut res = src.clone();
        // for i in range(argnum):
        //     res.setarg(i, self._untag(self._next()))
        for arg in res.args.iter_mut() {
            *arg = self._untag(*arg);
        }
        if let Some(ref mut fa) = res.fail_args {
            for arg in fa.iter_mut() {
                *arg = self._untag(*arg);
            }
        }
        // RPython opencoder.py:399-401:
        //     res = ResOperation(opnum, args, descr)   # fresh cls() object
        //     if res.type != 'v':
        //         self._cache[self._index] = res
        //         self._index += 1
        //
        // RPython allocates a fresh `cls()` ResOperation Python object on
        // EVERY visited op (void or non-void); only non-void results land
        // in `_cache`, but the freshness is unconditional. majit's
        // analogue of "fresh cls()" is the `_fresh` OpRef counter, so we
        // must advance `_fresh` for void ops too — otherwise a non-void
        // op processed after a void op gets a fresh OpRef that collides
        // with the void op's raw `pos`. With this in place,
        // `start_fresh = 0` over a recorder-emitted trace produces a
        // bit-identical sequence of OpRefs (every op `i` gets
        // `OpRef(num_inputs + i)`, matching `recorder.record_op`'s
        // monotonic `op_count`).
        let is_void_result = src.pos.is_none() || src.opcode.result_type() == majit_ir::Type::Void;
        if !is_void_result {
            let orig = src.pos.0 as usize;
            if orig >= self._cache.len() {
                self._cache.resize(orig + 1, None);
            }
            // opencoder.py:399-401 always allocates a fresh ResOperation
            // instance and then overwrites `_cache[self._index]` with it.
            // Reusing an existing cache entry collapses distinct box
            // identities when a raw op slot collides with an inputarg or
            // earlier phase-local box.
            let fresh = OpRef(self._fresh);
            self._fresh += 1;
            self._cache[orig] = Some(fresh);
            res.pos = fresh;
            // RPython `_index` parity: advance past the cache slot we
            // just wrote. In RPython this happens via `_index += 1`
            // because `_index` == cache slot index; in majit the cache
            // key is the raw trace position, so we assign `orig + 1`
            // directly to keep `_cache[_index - 1]` (`replace_last_cached`)
            // pointing at the slot we just wrote.
            self._index = orig as u32 + 1;
        } else if !src.pos.is_none() {
            // Void op carrying a raw trace position: still allocate a
            // fresh OpRef so the `_fresh` counter stays in lockstep with
            // the raw trace position counter. The op is not cached
            // (RPython doesn't cache void results either), so later args
            // never reference it.
            let f = OpRef(self._fresh);
            self._fresh += 1;
            res.pos = f;
        } else {
            res.pos = src.pos;
        }
        // self._count += 1
        self._count += 1;
        Some(res)
    }
}

/// opencoder.py: CutTrace — supports cutting traces at breakpoints
/// for bridge compilation.
#[derive(Clone, Debug)]
pub struct CutTrace {
    /// Position at which the trace was cut.
    pub cut_at: usize,
    /// Number of input args at the cut point.
    pub num_inputargs: usize,
}

impl CutTrace {
    pub fn new(cut_at: usize, num_inputargs: usize) -> Self {
        CutTrace {
            cut_at,
            num_inputargs,
        }
    }
}

/// opencoder.py: BoxArrayIter — iterates encoded box arrays
/// in snapshot data, decoding tagged values.
pub struct BoxArrayIter<'a> {
    values: &'a [u32],
    pos: usize,
}

impl<'a> BoxArrayIter<'a> {
    pub fn empty() -> Self {
        Self::new(&[])
    }

    pub fn new(values: &'a [u32]) -> Self {
        BoxArrayIter { values, pos: 0 }
    }

    /// Get the next tagged value, or None if exhausted.
    pub fn next_value(&mut self) -> Option<u32> {
        if self.pos < self.values.len() {
            let val = self.values[self.pos];
            self.pos += 1;
            Some(val)
        } else {
            None
        }
    }

    /// Decode the next tagged value.
    pub fn next_decoded(&mut self) -> Option<(u8, u32)> {
        self.next_value().map(untag)
    }

    /// Remaining values.
    pub fn remaining(&self) -> usize {
        self.values.len() - self.pos
    }

    /// Whether exhausted.
    pub fn done(&self) -> bool {
        self.pos >= self.values.len()
    }
}

impl<'a> Iterator for BoxArrayIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_value()
    }
}

// ── Trace Recording API (opencoder.py: Trace class) ──

/// opencoder.py: Trace — compact trace recording buffer.
///
/// Literal port of `rpython/jit/metainterp/opencoder.py::Trace`. The state
/// shape matches RPython 1:1 so the wire format, snapshot chain, and
/// liverange computations can all track RPython behavior. Field names are
/// preserved with a leading underscore where RPython uses one, to make
/// line-by-line porting unambiguous.
///
/// NOTE: methods `record_op*`, `capture_resumedata`, `create_top_snapshot`,
/// `_encode_descr`, `tag_overflow_imminent`, and `tracing_done` are
/// rebuilt across Phase B2–B11 in small sub-phase commits. B1 lands only
/// the state fields and the lowest-level `append_byte` / `append_int` /
/// `_double_ops` / `length` / `cut_point` / `cut_at` helpers that RPython
/// puts next to the struct definition (opencoder.py:468-578).
pub type Trace = TraceRecordBuffer;
pub struct TraceRecordBuffer {
    /// opencoder.py:473 self._ops — pre-allocated byte buffer that records
    /// the operation stream. `_pos` walks forward over it; `_double_ops`
    /// doubles the buffer when `_pos + 4 > len`. The length-of-trace is
    /// `_pos`, NOT `_ops.len()` (opencoder.py:564 length()).
    pub _ops: Vec<u8>,
    /// opencoder.py:475 self._pos — next write position into `_ops`.
    pub _pos: usize,
    /// opencoder.py:497 self._count — total count of ops recorded, seeded
    /// to `max_num_inputargs`. Advances for every op including voids.
    pub _count: u32,
    /// opencoder.py:498 self._index — position of resulting resops, seeded
    /// to `max_num_inputargs`. Advances only for non-void ops (the ones
    /// that produce a box-valued result).
    pub _index: u32,
    /// opencoder.py:499 self._start — frozen baseline equal to
    /// `max_num_inputargs`; iterator construction uses this as the start
    /// cursor (see `get_iter`).
    pub _start: u32,
    /// opencoder.py:496 self.max_num_inputargs — cap on the number of
    /// input-arg boxes. Inputarg positions must stay under this value.
    pub max_num_inputargs: u32,
    /// opencoder.py:491,504 self.inputargs — the actual inputarg boxes
    /// after `set_inputargs` is called. Empty until `set_inputargs` runs.
    pub inputargs: Vec<InputArg>,
    /// opencoder.py:481 self._descrs — `_descrs[0]` is always None; new
    /// descrs append. Indexed by `descr_index - all_descrs_len - 1` when
    /// the descr has no global `get_descr_index` (opencoder.py:702-707).
    pub _descrs: Vec<Option<majit_ir::DescrRef>>,
    /// opencoder.py:482 self._refs — `_refs[0]` is always nullptr (index
    /// 0 reserved for the 0-length snapshot array). Non-null GC refs
    /// append and are GC-rooted by the snapshot storage subsystem.
    pub _refs: Vec<u64>,
    /// opencoder.py:483 self._refs_dict — caches addr → index into
    /// `_refs`. Cleared by `tracing_done`.
    pub _refs_dict: HashMap<u64, u32>,
    /// opencoder.py:484 self._bigints — constant pool for big ints
    /// (> SMALL_INT_STOP). Indexed via `(idx << 1)` in TAGCONSTOTHER
    /// (bit 0 = 0 means bigint).
    pub _bigints: Vec<i64>,
    /// opencoder.py:485 self._bigints_dict — caches value → index.
    /// Cleared by `tracing_done`.
    pub _bigints_dict: HashMap<i64, u32>,
    /// opencoder.py:486 self._floats — constant pool for floats. Indexed
    /// via `(idx << 1) | 1` in TAGCONSTOTHER (bit 0 = 1 means float).
    pub _floats: Vec<u64>,
    /// opencoder.py:487 self._snapshot_data — byte chain encoding the
    /// per-guard snapshot records (jitcode_index, pc, array_index,
    /// prev) in RPython `encode_varint_signed` format.
    pub _snapshot_data: Vec<u8>,
    /// opencoder.py:488 self._snapshot_array_data — byte chain encoding
    /// the box arrays referenced by snapshots. Index 0 is reserved for
    /// the empty (length 0) array — `append_snapshot_array_data_int(0)`
    /// is called in the constructor (opencoder.py:489).
    pub _snapshot_array_data: Vec<u8>,
    /// opencoder.py:478 self._total_snapshots — monotonic snapshot
    /// counter; bumped in `create_snapshot` / `create_top_snapshot`.
    pub _total_snapshots: u32,
    /// opencoder.py:501 self.tag_overflow — set when `append_int` or
    /// `append_snapshot_*_int` receives a value outside [MIN_VALUE,
    /// MAX_VALUE]. Consulted by `tracing_done` to decide whether to
    /// raise `SwitchToBlackhole(ABORT_TOO_LONG)`.
    pub tag_overflow: bool,
    /// opencoder.py:476-477,479-480 instrumentation counters. `nodict`
    /// tracks `ConstPtrJitCode` entries that hit the opencoder-index
    /// fast path (dict-bypass).
    pub _consts_bigint: u32,
    pub _consts_float: u32,
    pub _consts_ptr: u32,
    pub _consts_ptr_nodict: u32,

    /// Dead-range memoization cache. opencoder.py:469 and :873-883 —
    /// `_deadranges = (-1, None)` means "not computed". Stored as
    /// `Option<(u32, Vec<usize>)>` where the u32 is the `_count` snapshot
    /// at cache time.
    pub _deadranges: Option<(u32, Vec<usize>)>,
}

impl TraceRecordBuffer {
    /// opencoder.py:471-501 Trace.__init__(max_num_inputargs, metainterp_sd).
    ///
    /// The `metainterp_sd` argument in RPython carries `all_descrs` for
    /// descriptor index arithmetic. pyre's DescrRef already holds a
    /// global index via `get_descr_index`, so the `all_descrs_len`
    /// offset used by `_encode_descr` / `TraceIterator` is supplied per
    /// call site rather than stored on the trace.
    pub fn new(max_num_inputargs: u32) -> Self {
        let mut t = TraceRecordBuffer {
            _ops: vec![0u8; INIT_SIZE],
            _pos: 0,
            _count: max_num_inputargs,
            _index: max_num_inputargs,
            _start: max_num_inputargs,
            max_num_inputargs,
            inputargs: Vec::new(),
            // opencoder.py:481 — `_descrs = [None]` so index 0 is reserved.
            _descrs: vec![None],
            // opencoder.py:482 — `_refs = [lltype.nullptr(GCREF.TO)]` so
            // index 0 is the null reference / empty-array sentinel.
            _refs: vec![0u64],
            _refs_dict: HashMap::new(),
            _bigints: Vec::new(),
            _bigints_dict: HashMap::new(),
            _floats: Vec::new(),
            _snapshot_data: Vec::new(),
            _snapshot_array_data: Vec::new(),
            _total_snapshots: 0,
            tag_overflow: false,
            _consts_bigint: 0,
            _consts_float: 0,
            _consts_ptr: 0,
            _consts_ptr_nodict: 0,
            _deadranges: None,
        };
        // opencoder.py:489 — `append_snapshot_array_data_int(0)` so all
        // zero-length arrays share index 0. Append via the signed-varint
        // encoder directly: Phase B2 will route this through a typed
        // helper, but the byte pattern must land regardless.
        t.append_snapshot_array_data_int(0);
        t
    }

    /// opencoder.py:503-508 set_inputargs(inputargs).
    ///
    /// Records the actual inputarg boxes. Positions must be < max_num_inputargs
    /// (checked in debug mode only, mirroring RPython's `we_are_translated`
    /// guard on the check).
    pub fn set_inputargs(&mut self, inputargs: Vec<InputArg>) {
        debug_assert!(
            inputargs
                .iter()
                .all(|ia| (ia.index as u32) < self.max_num_inputargs),
            "inputarg position exceeds max_num_inputargs"
        );
        self.inputargs = inputargs;
    }

    /// opencoder.py:510-511 _double_ops — double the byte buffer when
    /// `_pos` gets close to the end.
    fn _double_ops(&mut self) {
        let new_len = self._ops.len().max(1) * 2;
        self._ops.resize(new_len, 0);
    }

    /// opencoder.py:513-518 append_byte(c) — write a single byte and
    /// advance `_pos`. Doubles the buffer if needed.
    pub fn append_byte(&mut self, c: u8) {
        if self._pos >= self._ops.len() {
            self._double_ops();
        }
        self._ops[self._pos] = c;
        self._pos += 1;
    }

    /// opencoder.py:520-541 append_int(i). Writes a signed varint into
    /// `_ops` at `_pos` using the same 2-or-4 byte layout as
    /// `encode_varint_signed`, but inlined so the buffer-doubling check
    /// happens once (RPython checks `_pos + 4 > len(_ops)` then writes in
    /// place). Out-of-range values trip `tag_overflow` and encode 0.
    pub fn append_int(&mut self, i: i64) {
        let mut v = i;
        if !(MIN_VALUE..=MAX_VALUE).contains(&v) {
            self.tag_overflow = true;
            v = 0;
        }
        if self._pos + 4 > self._ops.len() {
            self._double_ops();
        }
        let flag: u8 = if !(-(1 << 14)..(1 << 14)).contains(&v) {
            0x80
        } else {
            0
        };
        self._ops[self._pos] = ((v & 0b0111_1111) as u8) | flag;
        self._pos += 1;
        v >>= 7;
        self._ops[self._pos] = (v & 0xff) as u8;
        self._pos += 1;
        if flag != 0 {
            v >>= 8;
            self._ops[self._pos] = (v & 0xff) as u8;
            self._pos += 1;
            v >>= 8;
            self._ops[self._pos] = (v & 0xff) as u8;
            self._pos += 1;
        }
    }

    /// opencoder.py:543-544 tag_overflow_imminent — returns true once the
    /// trace stream has grown past 80% of MAX_VALUE. Phase B11 will wire
    /// this up to the warmstate soft-abort plumbing.
    pub fn tag_overflow_imminent(&self) -> bool {
        (self._pos as i64) > (MAX_VALUE as f64 * 0.8) as i64
    }

    /// opencoder.py:564-565 length — the encoded trace length is `_pos`,
    /// NOT `_ops.len()` (the buffer is pre-allocated with padding).
    pub fn length(&self) -> usize {
        self._pos
    }

    /// opencoder.py:567-568 cut_point — snapshot `(pos, count, index,
    /// snapshot_data_len, snapshot_array_data_len)`. Returned as a tuple
    /// so `cut_at` can restore all five fields atomically.
    pub fn cut_point(&self) -> (usize, u32, u32, usize, usize) {
        (
            self._pos,
            self._count,
            self._index,
            self._snapshot_data.len(),
            self._snapshot_array_data.len(),
        )
    }

    /// opencoder.py:570-575 cut_at(end) — restore the first three fields
    /// of the cut_point tuple. The snapshot lengths are intentionally
    /// NOT truncated in RPython (they grow monotonically for a single
    /// trace; bridge compilation handles snapshot reuse separately).
    pub fn cut_at(&mut self, end: (usize, u32, u32, usize, usize)) {
        self._pos = end.0;
        self._count = end.1;
        self._index = end.2;
        // `end.3`, `end.4` are the snapshot data lengths; RPython ignores
        // them in `cut_at` — keep parity.
    }

    /// Placeholder for B2/B3: the `append_int` / `record_op*` /
    /// `capture_resumedata` / `_encode_descr` / `_encode` API lands in
    /// subsequent sub-phases. The field layout below is what those
    /// methods need to operate on.

    // ── Legacy-compat shims (to be removed as later sub-phases land) ──
    //
    // These keep existing call sites in this crate (chiefly
    // `get_live_ranges` and its tests) compiling while Phase B1 lands
    // the new field shape. Subsequent sub-phases replace them with
    // strict RPython ports.

    /// Legacy compatibility: count of recorded ops in the op stream.
    /// Prior to B6 this tracked every recorded op including voids;
    /// Phase B6 rewires the semantics to `_index`-based non-void count.
    /// For now, report `_index - _start` so existing callers see the
    /// number of *box-producing* ops.
    pub fn num_ops(&self) -> usize {
        (self._index - self._start) as usize
    }

    /// Placeholder: tracing_done is rewritten in Phase B11.
    pub fn tracing_done(&mut self) -> bool {
        // Clear the dicts per opencoder.py:550-551.
        self._bigints_dict.clear();
        self._refs_dict.clear();
        !self.tag_overflow
    }

    // ── Box encoding (opencoder.py:603-640 _encode + _cached_const_*) ──

    /// opencoder.py:603-608 _encode for ConstInt that fits in SMALL_INT
    /// range — return `tag(TAGINT, value)`.
    ///
    /// Values outside [SMALL_INT_START, SMALL_INT_STOP) must route
    /// through `_encode_bigint` because the signed varint can't pack
    /// them in the TAGINT namespace.
    pub fn _encode_smallint(value: i64) -> i64 {
        debug_assert!(
            (SMALL_INT_START..SMALL_INT_STOP).contains(&value),
            "value {value} out of SMALL_INT range — use _encode_bigint"
        );
        tag(TAGINT, value as u32) as i64
    }

    /// opencoder.py:609-622 _encode for ConstInt that does NOT fit in
    /// SMALL_INT range — interns into `_bigints` (dedup via
    /// `_bigints_dict`) and returns `tag(TAGCONSTOTHER, (idx << 1))`.
    /// Bit 0 of `idx << 1` is 0 → bigint; bit 0 == 1 → float.
    pub fn _encode_bigint(&mut self, value: i64) -> i64 {
        self._consts_bigint += 1;
        let v = if let Some(&idx) = self._bigints_dict.get(&value) {
            idx as u64
        } else {
            let idx = self._bigints.len() as u64;
            self._bigints.push(value);
            self._bigints_dict.insert(value, idx as u32);
            idx
        };
        tag(TAGCONSTOTHER, (v << 1) as u32) as i64
    }

    /// opencoder.py:623-628 _encode for ConstFloat — no caching; appends
    /// to `_floats` and returns `tag(TAGCONSTOTHER, (idx << 1) | 1)`.
    /// Bit 0 == 1 flags the float path on decode.
    pub fn _encode_float(&mut self, raw: u64) -> i64 {
        self._consts_float += 1;
        let idx = self._floats.len() as u32;
        self._floats.push(raw);
        tag(TAGCONSTOTHER, (idx << 1) | 1) as i64
    }

    /// opencoder.py:583-601 _cached_const_ptr + :629-632 _encode for
    /// ConstPtr — dedup via `_refs_dict` (by address), push to `_refs`,
    /// return `tag(TAGCONSTPTR, idx)`. Index 0 is reserved for nullptr
    /// (seeded by the constructor).
    pub fn _encode_ptr(&mut self, addr: u64) -> i64 {
        self._consts_ptr += 1;
        if addr == 0 {
            return tag(TAGCONSTPTR, 0) as i64;
        }
        let v = if let Some(&idx) = self._refs_dict.get(&addr) {
            idx
        } else {
            let idx = self._refs.len() as u32;
            self._refs.push(addr);
            self._refs_dict.insert(addr, idx);
            idx
        };
        tag(TAGCONSTPTR, v) as i64
    }

    /// opencoder.py:633-638 _encode for AbstractResOp — boxes that live
    /// in the trace's own op stream. Position is the box's `_index`
    /// slot from record time.
    pub fn _encode_box_position(position: u32) -> i64 {
        tag(TAGBOX, position) as i64
    }

    // ── Snapshot writers (opencoder.py:712-817) ──

    /// opencoder.py:728-733 new_array(lgt).
    ///
    /// Length-0 arrays always return index 0 (the pre-seeded empty
    /// entry from the constructor). Non-zero lengths write the length
    /// as the first varint at the tail of `_snapshot_array_data` and
    /// return the offset before the write; callers then write the
    /// array contents via `_add_box_to_storage`.
    pub fn new_array(&mut self, lgt: usize) -> i64 {
        if lgt == 0 {
            return 0;
        }
        let res = self._snapshot_array_data.len() as i64;
        self.append_snapshot_array_data_int(lgt as i64);
        res
    }

    /// opencoder.py:735-736 _add_box_to_storage(tagged_box).
    ///
    /// Appends a pre-encoded tagged box value to `_snapshot_array_data`.
    /// The caller computes the tag via the `_encode_*` helpers above.
    pub fn _add_box_to_storage(&mut self, tagged: i64) {
        self.append_snapshot_array_data_int(tagged);
    }

    /// opencoder.py:712-716 _list_of_boxes(boxes). Writes an array
    /// header then each box, returns the array index (0 for empty).
    pub fn _list_of_boxes(&mut self, boxes: &[i64]) -> i64 {
        let res = self.new_array(boxes.len());
        for &b in boxes {
            self._add_box_to_storage(b);
        }
        res
    }

    /// opencoder.py:718-726 _list_of_boxes_virtualizable(boxes).
    ///
    /// The virtualizable lives at the end of the locals_cells_stack in
    /// the tracer's view, but the snapshot encoding wants it at the
    /// FRONT so the resume reader can pull it out before the local
    /// frame slots. This reorders `[a, b, c, vable]` to `[vable, a, b, c]`.
    pub fn _list_of_boxes_virtualizable(&mut self, boxes: &[i64]) -> i64 {
        if boxes.is_empty() {
            return self.new_array(0);
        }
        let res = self.new_array(boxes.len());
        // boxes[-1] first (the virtualizable), then boxes[:-1].
        self._add_box_to_storage(*boxes.last().unwrap());
        for &b in &boxes[..boxes.len() - 1] {
            self._add_box_to_storage(b);
        }
        res
    }

    /// opencoder.py:750-765 _encode_snapshot(index, pc, array, is_last).
    ///
    /// Writes jitcode_index + pc + array_index + prev-sentinel into
    /// `_snapshot_data`. `prev` is either `SNAPSHOT_PREV_NONE` for the
    /// last snapshot in a chain (end of framestack) or
    /// `SNAPSHOT_PREV_NEEDS_PATCHING` for internal snapshots — those
    /// get patched later via `snapshot_add_prev`.
    ///
    /// Returns the snapshot_index (offset into `_snapshot_data` before
    /// the write).
    pub fn _encode_snapshot(&mut self, index: i64, pc: i64, array: i64, is_last: bool) -> i64 {
        let res = self._snapshot_data.len() as i64;
        self.append_snapshot_data_int(index);
        self.append_snapshot_data_int(pc);
        self.append_snapshot_data_int(array);
        let prev = if is_last {
            SNAPSHOT_PREV_NONE as i64
        } else {
            SNAPSHOT_PREV_NEEDS_PATCHING as i64
        };
        self.append_snapshot_data_int(prev);
        res
    }

    /// opencoder.py:812-817 snapshot_add_prev(prev).
    ///
    /// Overwrites the trailing prev-sentinel slot in `_snapshot_data`
    /// (written as `SNAPSHOT_PREV_NEEDS_PATCHING` by the previous
    /// `_encode_snapshot`) with the actual prev index. Called from
    /// `create_snapshot` (SNAPSHOT_PREV_COMES_NEXT) and
    /// `_ensure_parent_resumedata` (the real parent snapshot index).
    ///
    /// INVARIANT: the last 2 bytes of `_snapshot_data` must be the
    /// signed-varint encoding of `SNAPSHOT_PREV_NEEDS_PATCHING` (-3).
    /// `-3` fits in 2 bytes (RPython literally asserts the bytes
    /// `'}' '\xff'` at opencoder.py:813-814). We assert the same.
    pub fn snapshot_add_prev(&mut self, prev: i32) {
        debug_assert!(self._snapshot_data.len() >= 2, "snapshot_data too short");
        let n = self._snapshot_data.len();
        debug_assert_eq!(
            self._snapshot_data[n - 2],
            b'}',
            "prev slot byte 0 not \\x7d (NEEDS_PATCHING marker)"
        );
        debug_assert_eq!(
            self._snapshot_data[n - 1],
            0xffu8,
            "prev slot byte 1 not \\xff (NEEDS_PATCHING marker)"
        );
        self._snapshot_data.pop();
        self._snapshot_data.pop();
        self.append_snapshot_data_int(prev as i64);
    }

    // ── Snapshot chain entry points (opencoder.py:767-843) ──

    /// opencoder.py:767-785 create_top_snapshot(frame, vable_boxes,
    /// vref_boxes, after_residual_call, is_last).
    ///
    /// The caller has already computed the active-boxes array via
    /// `new_array` + `_add_box_to_storage` calls (that's the `array`
    /// parameter here — RPython does this inline through the frame's
    /// `get_list_of_active_boxes` callback). vable_boxes / vref_boxes
    /// are pre-encoded tagged values; this writes their arrays into
    /// `_snapshot_array_data`, then the top snapshot record into
    /// `_snapshot_data`, and finally patches the preceding guard's
    /// descr slot to the snapshot index.
    pub fn create_top_snapshot(
        &mut self,
        jitcode_index: i64,
        pc: i64,
        array: i64,
        vable_boxes: &[i64],
        vref_boxes: &[i64],
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        let s = self._snapshot_data.len() as i64;
        let vable_array = self._list_of_boxes_virtualizable(vable_boxes);
        let vref_array = self._list_of_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        self._encode_snapshot(jitcode_index, pc, array, is_last);
        // Patch the guard's trailing descr slot to the snapshot index.
        self.patch_last_guard_descr_slot(s);
        s
    }

    /// opencoder.py:787-804 create_empty_top_snapshot(vable_boxes, vref_boxes).
    ///
    /// No frame data — used when tracing starts at a guard before any
    /// frame has been entered. Writes a snapshot with jitcode_index=-1,
    /// pc=0, an empty box array, and SNAPSHOT_PREV_NONE.
    pub fn create_empty_top_snapshot(&mut self, vable_boxes: &[i64], vref_boxes: &[i64]) -> i64 {
        self._total_snapshots += 1;
        let s = self._snapshot_data.len() as i64;
        let empty_array = self._list_of_boxes(&[]);
        let vable_array = self._list_of_boxes_virtualizable(vable_boxes);
        let vref_array = self._list_of_boxes(vref_boxes);
        self.append_snapshot_data_int(vable_array);
        self.append_snapshot_data_int(vref_array);
        self._encode_snapshot(-1, 0, empty_array, true);
        self.patch_last_guard_descr_slot(s);
        s
    }

    /// opencoder.py:806-810 create_snapshot(frame, is_last).
    ///
    /// Called while walking the framestack toward the outermost frame
    /// to capture parent resumedata. Writes a `SNAPSHOT_PREV_COMES_NEXT`
    /// sentinel on the PREVIOUS snapshot (via `snapshot_add_prev`),
    /// then encodes this snapshot with `SNAPSHOT_PREV_NEEDS_PATCHING`
    /// (to be resolved on the NEXT iteration).
    pub fn create_snapshot(
        &mut self,
        jitcode_index: i64,
        pc: i64,
        array: i64,
        is_last: bool,
    ) -> i64 {
        self._total_snapshots += 1;
        self.snapshot_add_prev(SNAPSHOT_PREV_COMES_NEXT);
        self._encode_snapshot(jitcode_index, pc, array, is_last)
    }

    /// opencoder.py:781-784 `_pos -= 2; self.append_int(s)` — patch the
    /// guard's trailing descr slot to the snapshot index.
    ///
    /// INVARIANT: guards record their trailing descr via `append_int(0)`
    /// (see `_op_end` + opencoder.py:653-657). `0` always encodes as
    /// exactly 2 bytes in the RPython signed-varint layout (both bytes
    /// are `\x00`), so `capture_resumedata` / `create_top_snapshot` can
    /// safely rewind `_pos -= 2` and overwrite the slot with the actual
    /// snapshot_index via a fresh `append_int(s)`. The new value may
    /// occupy 2 *or* 4 bytes; `_ops` is pre-grown by `_double_ops` so
    /// both sizes fit.
    ///
    /// RPython inlines this sequence in `create_top_snapshot` and
    /// `create_snapshot`; pyre factors it out so Phase B7's snapshot
    /// code is shorter and the assertion that the slot was indeed a
    /// guard 0-placeholder is always executed.
    pub fn patch_last_guard_descr_slot(&mut self, snapshot_index: i64) {
        debug_assert!(
            self._pos >= 2,
            "patch_last_guard_descr_slot called with _pos < 2"
        );
        debug_assert_eq!(
            self._ops[self._pos - 2],
            0u8,
            "guard descr placeholder byte 0 was not \\x00"
        );
        debug_assert_eq!(
            self._ops[self._pos - 1],
            0u8,
            "guard descr placeholder byte 1 was not \\x00"
        );
        self._pos -= 2;
        self.append_int(snapshot_index);
    }

    /// opencoder.py:702-707 _encode_descr(descr).
    ///
    /// Two-tier descr numbering mirroring RPython:
    ///   * Global descrs — `descr.get_descr_index() >= 0`. Encoded as
    ///     `global_index + 1`. Index `[1, all_descrs_len + 1)`.
    ///   * Local descrs — `get_descr_index() == -1`. Appended to
    ///     `self._descrs` (which starts with a `None` sentinel at [0]
    ///     so index 0 means "no descr") and encoded as
    ///     `all_descrs_len + len(_descrs) - 1 + 1`. RPython stores the
    ///     ref so the TraceIterator can look it up with
    ///     `_descrs[descr_index - all_descr_len - 1]`.
    ///
    /// `all_descrs_len` is the global descriptor table length at
    /// encode time — the caller provides it so the Trace type does not
    /// need a back-reference to metainterp state (RPython stores
    /// `metainterp_sd` directly; pyre passes it per-call).
    pub fn _encode_descr(&mut self, descr: &majit_ir::DescrRef, all_descrs_len: u32) -> i64 {
        let descr_index = descr.get_descr_index();
        if descr_index >= 0 {
            return (descr_index as i64) + 1;
        }
        self._descrs.push(Some(descr.clone()));
        // _descrs[0] is the sentinel None; new descrs append from index 1.
        // RPython returns `len(_descrs) - 1 + all_descrs_len + 1`. `len - 1`
        // because index 0 is reserved; we want the local-only offset.
        let local_index = (self._descrs.len() as i64) - 1;
        local_index + (all_descrs_len as i64) + 1
    }

    /// opencoder.py:642-650 _op_start(opnum, num_argboxes).
    ///
    /// Writes the opnum byte. For fixed-arity ops the decoder reads the
    /// arity from `OpCode::arity()`; for variadic ops (arity == None)
    /// the writer records the actual `num_argboxes` count inline as a
    /// signed varint. Returns the old `_pos` so `_op_end` can patch the
    /// trailing descr int if the opnum turns out to be a guard that gets
    /// its snapshot index filled in later.
    pub fn _op_start(&mut self, opcode: OpCode, num_argboxes: usize) -> usize {
        let old_pos = self._pos;
        let opnum = opcode.as_u16();
        debug_assert!(opnum <= 0xFF, "opnum {opnum} exceeds 1 byte");
        self.append_byte(opnum as u8);
        match opcode.arity() {
            None => {
                self.append_int(num_argboxes as i64);
            }
            Some(expected) => {
                debug_assert_eq!(
                    num_argboxes, expected as usize,
                    "fixed-arity mismatch for {:?}: expected {expected}, got {num_argboxes}",
                    opcode
                );
            }
        }
        old_pos
    }

    /// opencoder.py:652-662 _op_end(opnum, descr, old_pos).
    ///
    /// Writes the trailing descr int if the opcode carries one
    /// (`has_descr()`). `descr_index` is the already-encoded descr slot
    /// from `_encode_descr` (Phase B4); guards pass 0 to get a
    /// placeholder that `capture_resumedata` patches to the snapshot
    /// index (Phase B5/B7). Bumps `_count` unconditionally and `_index`
    /// only for non-void ops (opencoder.py:661).
    pub fn _op_end(&mut self, opcode: OpCode, descr_index: i64, _old_pos: usize) {
        if opcode.has_descr() {
            self.append_int(descr_index);
        }
        self._count += 1;
        if opcode.result_type() != Type::Void {
            self._index += 1;
        }
    }

    /// opencoder.py:664-670 record_op(opnum, argboxes, descr=None).
    ///
    /// Returns the pre-bump `_index` value — the box position that THIS
    /// op's result will occupy if it is non-void (opencoder.py:665 `pos
    /// = self._index`). Args are pre-encoded tagged values from
    /// `_encode(box)` (lands in Phase B5); for now callers that don't
    /// yet call `_encode` can build tags via the free `tag` helper.
    /// `descr_index` is the `_encode_descr` output (Phase B4), or 0 for
    /// no-descr / guard-placeholder paths.
    pub fn record_op(&mut self, opcode: OpCode, argboxes: &[i64], descr_index: i64) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, argboxes.len());
        for &box_tag in argboxes {
            self.append_int(box_tag);
        }
        self._op_end(opcode, descr_index, old_pos);
        pos
    }

    /// opencoder.py:672-676 record_op0(opnum, descr=None).
    pub fn record_op0(&mut self, opcode: OpCode, descr_index: i64) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 0);
        self._op_end(opcode, descr_index, old_pos);
        pos
    }

    /// opencoder.py:678-683 record_op1(opnum, argbox1, descr=None).
    pub fn record_op1(&mut self, opcode: OpCode, arg0: i64, descr_index: i64) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 1);
        self.append_int(arg0);
        self._op_end(opcode, descr_index, old_pos);
        pos
    }

    /// opencoder.py:685-691 record_op2(opnum, argbox1, argbox2, descr=None).
    pub fn record_op2(&mut self, opcode: OpCode, arg0: i64, arg1: i64, descr_index: i64) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 2);
        self.append_int(arg0);
        self.append_int(arg1);
        self._op_end(opcode, descr_index, old_pos);
        pos
    }

    /// opencoder.py:693-700 record_op3(opnum, argbox1, argbox2, argbox3, descr=None).
    pub fn record_op3(
        &mut self,
        opcode: OpCode,
        arg0: i64,
        arg1: i64,
        arg2: i64,
        descr_index: i64,
    ) -> u32 {
        let pos = self._index;
        let old_pos = self._op_start(opcode, 3);
        self.append_int(arg0);
        self.append_int(arg1);
        self.append_int(arg2);
        self._op_end(opcode, descr_index, old_pos);
        pos
    }

    /// opencoder.py:738-742 append_snapshot_array_data_int(i).
    /// Writes a signed varint into `_snapshot_array_data`; values outside
    /// [MIN_VALUE, MAX_VALUE] trip `tag_overflow` (and encode 0 to keep
    /// the stream parseable).
    ///
    /// Phase B1 keeps this on the zigzag-based `encode_varint_signed`
    /// currently in this file so the constructor's reserve-index-0 call
    /// succeeds. Phase B2 replaces the encoder with RPython's 2/4-byte
    /// format, which will make this method fully parity-compliant.
    pub fn append_snapshot_array_data_int(&mut self, i: i64) {
        if !(MIN_VALUE..=MAX_VALUE).contains(&i) {
            self.tag_overflow = true;
            encode_varint_signed(&mut self._snapshot_array_data, 0);
        } else {
            encode_varint_signed(&mut self._snapshot_array_data, i);
        }
    }

    /// opencoder.py:744-748 append_snapshot_data_int(i).
    pub fn append_snapshot_data_int(&mut self, i: i64) {
        if !(MIN_VALUE..=MAX_VALUE).contains(&i) {
            self.tag_overflow = true;
            encode_varint_signed(&mut self._snapshot_data, 0);
        } else {
            encode_varint_signed(&mut self._snapshot_data, i);
        }
    }

    /// opencoder.py:852-858 Trace.get_live_ranges().
    ///
    /// Returns a list where `liveranges[v]` is the trace step at which the
    /// value with raw trace position `v` is last used. Mirrors RPython's
    /// `TraceIterator.next_element_update_live_range` (opencoder.py:339-360).
    ///
    /// `index_start` is the equivalent of RPython's `t._count` at the
    /// moment `get_live_ranges` enters the iteration loop
    /// (opencoder.py:855 `index = t._count`). For a full Trace this is
    /// `Trace._start = max_num_inputargs` (opencoder.py:268-270, set by
    /// `TraceIterator.__init__`). For a `CutTrace` it is the cut's saved
    /// `count` (opencoder.py:421-425), which can be larger than
    /// `num_inputargs` because the cut starts in the middle of a trace.
    /// Callers that operate on a freshly recorded full trace pass
    /// `self.num_inputargs`; callers that have a cut/slice MUST pass the
    /// equivalent `count` so that the sequential index numbering picks up
    /// where the parent left off, matching `CutTrace.get_iter`'s
    /// `iter._count = self.count` override (opencoder.py:421).
    ///
    /// 1. The array is sized to cover the entire raw box space
    ///    (`[0] * self._index`). Constants live in a separate namespace —
    ///    in majit they sit above the `CONST_BIT` boundary at value
    ///    `>= 1<<31` — and must NOT contribute to the array length, or a
    ///    single constant reference would inflate the allocation by 2 GiB.
    ///    The size is computed from `op.pos` after filtering constants and
    ///    `OpRef::NONE` sentinels. Note: in majit this slot space is
    ///    LARGER than RPython's `_index`, because majit's recorder
    ///    consumes a sequential OpRef for every op including voids
    ///    (recorder.rs:163-203). Per-op slots that belong to void ops sit
    ///    in the array unused, which is harmless for live/dead range
    ///    consumers — they only read slots they reference via arg.0.
    /// 2. The forward walk writes `liveranges[v] = index` for every TAGBOX
    ///    argument (`!is_constant() && !is_none()`) — TAGINT / TAGCONSTPTR /
    ///    TAGCONSTOTHER refs are skipped, just like opencoder.py:347-349.
    ///    `index` is the RPython sequential counter, NOT `op.pos`. RPython
    ///    only increments it for non-void ops (opencoder.py:357-358); void
    ///    ops keep using the previous index. majit's recorder consumes a
    ///    sequential OpRef for every op including void ones, so `op.pos`
    ///    advances on every op and would skew live-range step values
    ///    across runs of consecutive guards / SetfieldGc. Tracking a
    ///    separate `index` counter restores parity.
    /// 3. Each non-void op also self-defines its slot
    ///    (`liveranges[index] = index`, opencoder.py:351-352). majit
    ///    addresses the slot by `op.pos.0` (because that's how arg
    ///    references find it) but stores the sequential `index` value to
    ///    stay coordinate-consistent with the arg writes above. Without
    ///    this self-definition the dead-range computation collapses any
    ///    value that is produced but never read into position 0, which is
    ///    wrong for the register allocator's free-slot logic.
    pub fn get_live_ranges(&self, ops: &[majit_ir::Op], index_start: usize) -> Vec<usize> {
        use majit_ir::Type;
        // opencoder.py:854 — `[0] * self._index`. _index = the full raw
        // box space (inputargs + every recorded op slot). majit assigns a
        // sequential OpRef to every recorded op via `op_count`, so the
        // recorder's own counter is the most accurate ceiling. Fall back
        // to scanning op.pos for callers that pass a slice that no longer
        // matches the recorder state.
        let computed_max = ops
            .iter()
            .filter(|op| !op.pos.is_none() && !op.pos.is_constant())
            .map(|op| op.pos.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let total = self
            .num_ops()
            .max(computed_max)
            .max(self.max_num_inputargs as usize);
        let mut liveranges = vec![0usize; total];
        // opencoder.py:855 — `index = t._count`. Equivalent to
        // `iter._count = start` (opencoder.py:270) for a full Trace, and
        // to `iter._count = self.count` (opencoder.py:421) for a CutTrace.
        // The caller passes whichever value matches the slice it owns.
        let mut index = index_start;
        for op in ops {
            if op.pos.is_none() || op.pos.is_constant() {
                continue;
            }
            // opencoder.py:347-349 — TAGBOX args set liveranges[v] = index.
            // Skip constants (TAGINT/TAGCONSTPTR/TAGCONSTOTHER) and the
            // OpRef::NONE sentinel (no live data behind it).
            for &arg in &op.args {
                if arg.is_none() || arg.is_constant() {
                    continue;
                }
                let v = arg.0 as usize;
                if v < liveranges.len() {
                    liveranges[v] = index;
                }
            }
            if let Some(ref fa) = op.fail_args {
                for &arg in fa.iter() {
                    if arg.is_none() || arg.is_constant() {
                        continue;
                    }
                    let v = arg.0 as usize;
                    if v < liveranges.len() {
                        liveranges[v] = index;
                    }
                }
            }
            // opencoder.py:351-352 — non-void ops self-define. RPython's
            // `liveranges[index] = index` writes at slot=index because
            // RPython's index space and box position space coincide; in
            // majit the slot is addressed by `op.pos.0` (the value other
            // ops will reference via arg.0) while the value is the
            // sequential index, matching the arg-write semantics above.
            if op.result_type() != Type::Void {
                let v = op.pos.0 as usize;
                if v < liveranges.len() {
                    liveranges[v] = index;
                }
                // opencoder.py:357-358 — only non-void ops advance index.
                index += 1;
            }
        }
        liveranges
    }

    /// Encoded trace byte length (equivalent to RPython `length()`).
    pub fn data_len(&self) -> usize {
        self._pos
    }

    /// opencoder.py: get_dead_ranges()
    ///
    /// Compute dead ranges: for each op index x, the values that are
    /// known to be dead before x. Used by the register allocator to
    /// know when to free registers.
    ///
    /// `index_start` is forwarded to `get_live_ranges`; see that
    /// function's doc for the `t._count` semantics. Full-trace callers
    /// pass `self.num_inputargs`; cut/slice callers pass the cut's
    /// `count` so the sequential index numbering matches the parent
    /// trace's continuation point.
    pub fn get_dead_ranges(&self, ops: &[majit_ir::Op], index_start: usize) -> Vec<usize> {
        let live_ranges = self.get_live_ranges(ops, index_start);
        let mut dead_ranges = vec![0usize; live_ranges.len() + 2];
        for (i, &last_use) in live_ranges.iter().enumerate() {
            if last_use > 0 && last_use + 1 < dead_ranges.len() {
                // Value i dies after position last_use.
                // Record it in dead_ranges[last_use + 1].
                let mut pos = last_use + 1;
                while pos < dead_ranges.len() && dead_ranges[pos] != 0 {
                    pos += 1;
                }
                if pos < dead_ranges.len() {
                    dead_ranges[pos] = i;
                }
            }
        }
        dead_ranges
    }

    /// RPython-compatible: unpack tagged box stream into raw pairs.
    pub fn unpack(&self, encoded: &[u32]) -> Vec<(u8, u32)> {
        encoded.iter().copied().map(untag).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        let values = [0u64, 1, 127, 128, 255, 256, 65535, 0xFFFF_FFFF, u64::MAX];
        for &val in &values {
            let mut buf = Vec::new();
            encode_varint(&mut buf, val);
            let (decoded, consumed) = decode_varint(&buf);
            assert_eq!(decoded, val, "roundtrip failed for {val}");
            assert_eq!(consumed, buf.len());
        }
    }

    #[test]
    fn test_varint_small_values() {
        // Values 0..=127 should encode to a single byte.
        for val in 0..=127u64 {
            let mut buf = Vec::new();
            encode_varint(&mut buf, val);
            assert_eq!(buf.len(), 1, "value {val} should be 1 byte");
            assert_eq!(buf[0], val as u8);
        }
    }

    #[test]
    fn test_varint_128_is_two_bytes() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 128);
        assert_eq!(buf.len(), 2);
        let (decoded, consumed) = decode_varint(&buf);
        assert_eq!(decoded, 128);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_empty_trace_roundtrip() {
        let trace = TreeLoop::new(vec![], vec![]);
        let encoded = encode_trace(&trace);
        let decoded = decode_trace(&encoded);
        assert_eq!(decoded.num_inputargs(), 0);
        assert_eq!(decoded.num_ops(), 0);
    }

    #[test]
    fn test_trace_roundtrip() {
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(2)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);

        let encoded = encode_trace(&trace);
        let decoded = decode_trace(&encoded);

        assert_eq!(decoded.num_inputargs(), 3);
        assert_eq!(decoded.inputargs[0].tp, Type::Int);
        assert_eq!(decoded.inputargs[0].index, 0);
        assert_eq!(decoded.inputargs[1].tp, Type::Ref);
        assert_eq!(decoded.inputargs[1].index, 1);
        assert_eq!(decoded.inputargs[2].tp, Type::Float);
        assert_eq!(decoded.inputargs[2].index, 2);

        assert_eq!(decoded.num_ops(), 3);
        assert_eq!(decoded.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(decoded.ops[0].args.as_slice(), &[OpRef(0), OpRef(0)]);
        assert_eq!(decoded.ops[1].opcode, OpCode::GuardTrue);
        assert_eq!(decoded.ops[1].args.as_slice(), &[OpRef(1)]);
        assert_eq!(decoded.ops[2].opcode, OpCode::Jump);
        assert_eq!(decoded.ops[2].args.as_slice(), &[OpRef(2)]);

        assert!(decoded.is_loop());
    }

    #[test]
    fn test_trace_with_many_ops() {
        let inputargs = vec![InputArg::new_int(0)];
        let mut ops = Vec::new();

        // Build a chain of IntAdd ops
        for i in 0..100u32 {
            ops.push(Op::new(OpCode::IntAdd, &[OpRef(i), OpRef(0)]));
        }
        ops.push(Op::new(OpCode::Jump, &[OpRef(100)]));

        let trace = TreeLoop::new(inputargs, ops);
        let encoded = encode_trace(&trace);
        let decoded = decode_trace(&encoded);

        assert_eq!(decoded.num_ops(), 101);
        for i in 0..100 {
            assert_eq!(decoded.ops[i].opcode, OpCode::IntAdd);
            assert_eq!(decoded.ops[i].args[0], OpRef(i as u32));
            assert_eq!(decoded.ops[i].args[1], OpRef(0));
        }
        assert_eq!(decoded.ops[100].opcode, OpCode::Jump);
        assert!(decoded.is_loop());
    }

    #[test]
    fn test_trace_with_finish() {
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::new(OpCode::Finish, &[OpRef(1)]),
        ];
        let trace = TreeLoop::new(inputargs, ops);
        let encoded = encode_trace(&trace);
        let decoded = decode_trace(&encoded);

        assert!(decoded.is_finished());
        assert!(!decoded.is_loop());
    }

    #[test]
    fn test_trace_preserves_descr_flag() {
        // Verify that the has-descr byte is encoded (even though we don't
        // reconstruct the descriptor on decode).
        let ops = vec![Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)])];
        let trace = TreeLoop::new(vec![InputArg::new_int(0)], ops);

        let encoded = encode_trace(&trace);
        // The last byte of the encoded op should be the descr flag (0 = no descr)
        assert_eq!(*encoded.last().unwrap(), 0);
    }

    #[test]
    fn test_varint_multiple_in_buffer() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 42);
        encode_varint(&mut buf, 12345);
        encode_varint(&mut buf, 0);

        let (v1, n1) = decode_varint(&buf);
        assert_eq!(v1, 42);
        let (v2, n2) = decode_varint(&buf[n1..]);
        assert_eq!(v2, 12345);
        let (v3, _n3) = decode_varint(&buf[n1 + n2..]);
        assert_eq!(v3, 0);
    }

    #[test]
    fn test_signed_varint_roundtrip() {
        // opencoder.py enforces MIN_VALUE=-2^30, MAX_VALUE=2^30-1. Test
        // the 2-byte path (values in [-2^14, 2^14-1]) and the 4-byte path
        // (outside it). `i64::MAX` / `i64::MIN` would trip the
        // debug_assert — they are out of the encoder's documented range.
        let values = [
            0i64,
            1,
            -1,
            127,
            -128,
            1000,
            -1000,
            // 2-byte boundaries (stay in ±2^14 range)
            (1 << 14) - 1,
            -(1 << 14),
            // 4-byte range
            (1 << 14),
            -(1 << 14) - 1,
            (1 << 20),
            -(1 << 20),
            MIN_VALUE,
            MAX_VALUE,
        ];
        for &val in &values {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            let (decoded, _consumed) = decode_varint_signed(&buf);
            assert_eq!(decoded, val, "signed varint roundtrip failed for {val}");
        }
    }

    /// Phase B2: encoder must match RPython's 2-byte / 4-byte layout
    /// bit-exactly for snapshot and op stream compatibility. Verify
    /// small values stay 2 bytes and big values expand to 4 bytes.
    #[test]
    fn test_signed_varint_wire_format_b2() {
        // 2-byte values: flag bit (bit 7 of byte 0) MUST be clear.
        for &val in &[0i64, 1, -1, (1 << 14) - 1, -(1 << 14)] {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            assert_eq!(buf.len(), 2, "value {val} should be 2 bytes");
            assert_eq!(buf[0] & 0x80, 0, "flag bit set for 2-byte value {val}");
        }
        // 4-byte values: flag bit MUST be set.
        for &val in &[1i64 << 14, -(1i64 << 14) - 1, MAX_VALUE, MIN_VALUE] {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            assert_eq!(buf.len(), 4, "value {val} should be 4 bytes");
            assert_eq!(buf[0] & 0x80, 0x80, "flag bit clear for 4-byte value {val}");
        }
    }

    #[test]
    fn test_tag_untag_roundtrip() {
        for kind in [TAGINT, TAGCONSTPTR, TAGCONSTOTHER, TAGBOX] {
            for value in [0u32, 1, 42, 255, 1000] {
                let tagged = tag(kind, value);
                let (k, v) = untag(tagged);
                assert_eq!(k, kind, "tag kind mismatch");
                assert_eq!(v, value, "tag value mismatch");
            }
        }
    }

    #[test]
    fn test_snapshot_storage() {
        let mut storage = SnapshotStorage::new();
        let snap = Snapshot {
            values: vec![tag(TAGINT, 42), tag(TAGBOX, 0)],
            prev: None,
            jitcode_index: 0,
            pc: 10,
        };
        let idx = storage.add_snapshot(snap);
        assert_eq!(idx, 0);
        assert_eq!(storage.num_snapshots(), 1);

        let const_idx = storage.add_const_ref(0x1000);
        assert_eq!(const_idx, 0);
        let float_idx = storage.add_const_float(3.14);
        assert_eq!(float_idx, 0);
    }

    /// Helper: build an Op with a specific raw trace position.
    fn op_at(pos: u32, opcode: majit_ir::OpCode, args: &[OpRef]) -> majit_ir::Op {
        let mut op = majit_ir::Op::new(opcode, args);
        op.pos = OpRef(pos);
        op
    }

    /// rpython/jit/metainterp/test/test_opencoder.py::test_liveranges parity.
    /// 3 inputargs + NEW_WITH_VTABLE (= p0 at pos 3) + GUARD_TRUE (void at pos
    /// 4) referencing all four boxes via fail_args. The final liverange of
    /// every box should be the GUARD_TRUE position so that nothing dies
    /// before the guard.
    #[test]
    fn test_get_live_ranges_parity() {
        let mut buf = TraceRecordBuffer::new(3);
        // Mirror RPython: p0 = NEW_WITH_VTABLE at raw position 3, then a
        // GUARD_TRUE that pulls i0/i1/i2/p0 into its fail_args (the
        // capture_resumedata equivalent for the guard's snapshot refs).
        let p0 = op_at(3, majit_ir::OpCode::NewWithVtable, &[]);
        let mut guard = op_at(4, majit_ir::OpCode::GuardTrue, &[OpRef(0)]);
        guard.fail_args = Some(smallvec::smallvec![OpRef(1), OpRef(2), OpRef(3)]);
        let ops = vec![p0, guard];
        // op_count covers inputargs + recorded ops in record order.
        // Pretend the recorder allocated positions 0..5 (3 inputargs + 2 ops).
        // opencoder.py:498 _index = max_num_inputargs + non_void_count; here
        // NewWithVtable is non-void (+1), GuardTrue is void (+0), so _index=4.
        buf._index = 4;
        let total = 5;
        // Full-trace caller passes max_num_inputargs as t._count.
        let lr = buf.get_live_ranges(&ops, buf.max_num_inputargs as usize);
        // total >= max(op.pos)+1 = 5, total >= num_inputargs = 3 → length 5.
        assert_eq!(lr.len(), total);
        // i0/i1/i2/p0 last referenced by GUARD_TRUE at position 4.
        assert_eq!(lr[0], 4);
        assert_eq!(lr[1], 4);
        assert_eq!(lr[2], 4);
        assert_eq!(lr[3], 4);
        // GUARD_TRUE is void → no self-def at its own slot.
        assert_eq!(lr[4], 0);
    }

    /// Constants must NOT inflate the liveranges array. A single constant
    /// in majit's CONST_BIT namespace has value `>= 1<<31`; treating it as a
    /// raw box index would request a 16 GiB allocation. The function must
    /// skip TAGINT/TAGCONSTPTR refs entirely (opencoder.py:347-349).
    #[test]
    fn test_get_live_ranges_skips_constants() {
        let buf = TraceRecordBuffer::new(2);
        // i0 is OpRef(0), i1 is OpRef(1). v2 = i0 + const(42).
        let const_ref = OpRef::from_const(0);
        assert!(const_ref.is_constant());
        assert!(const_ref.0 >= 1u32 << 31);
        let add = op_at(2, majit_ir::OpCode::IntAdd, &[OpRef(0), const_ref]);
        let mut guard = op_at(3, majit_ir::OpCode::GuardTrue, &[OpRef(1)]);
        guard.fail_args = Some(smallvec::smallvec![const_ref, OpRef(2)]);
        let ops = vec![add, guard];
        let lr = buf.get_live_ranges(&ops, buf.max_num_inputargs as usize);
        // Length should be at most max(op.pos)+1 = 4 — never the constant
        // value (>= 1<<31). Anything bigger means the constant leaked into
        // the size calculation.
        assert!(
            lr.len() <= 16,
            "liveranges grew to {} — constants leaked into size calc",
            lr.len()
        );
        // i0 used by IntAdd at pos 2.
        assert_eq!(lr[0], 2);
        // i1 used by GuardTrue at pos 3.
        assert_eq!(lr[1], 3);
        // v2 (IntAdd result) used by GuardTrue's fail_args at pos 3, AND
        // self-defined at pos 2. Last write wins → 3.
        assert_eq!(lr[2], 3);
    }

    /// Non-void ops must self-define their slot so values that are produced
    /// but never read still have a real liverange entry. Without this the
    /// dead-range computation collapses dead values into position 0.
    #[test]
    fn test_get_live_ranges_self_defines_unused_results() {
        let buf = TraceRecordBuffer::new(1);
        // v1 = IntAdd(i0, i0) — value-producing, never used.
        let add = op_at(1, majit_ir::OpCode::IntAdd, &[OpRef(0), OpRef(0)]);
        let ops = vec![add];
        let lr = buf.get_live_ranges(&ops, buf.max_num_inputargs as usize);
        // i0 last referenced at IntAdd's position.
        assert_eq!(lr[0], 1);
        // v1 self-defines at pos 1 even though no later op reads it.
        assert_eq!(lr[1], 1);
    }

    /// opencoder.py:357-358 — only non-void ops advance the index counter.
    /// Consecutive void ops between two non-void ops must keep the same
    /// index, otherwise majit's `op.pos`-driven counter (which advances
    /// for every recorded op including guards/SetfieldGc) skews live-range
    /// step values whenever real traces hit guard chains.
    #[test]
    fn test_get_live_ranges_consecutive_voids_share_index() {
        let buf = TraceRecordBuffer::new(2);
        // Trace shape:
        //   pos 0, 1 = inputargs i0, i1
        //   pos 2    = IntAdd(i0, i1)        (non-void)
        //   pos 3    = GuardTrue(v2)         (void)
        //   pos 4    = GuardNoException()    (void, references v2 via fail_args)
        //   pos 5    = IntMul(i0, i1)        (non-void)
        let add = op_at(2, majit_ir::OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        let g1 = op_at(3, majit_ir::OpCode::GuardTrue, &[OpRef(2)]);
        let mut g2 = op_at(4, majit_ir::OpCode::GuardNoException, &[]);
        g2.fail_args = Some(smallvec::smallvec![OpRef(2)]);
        let mul = op_at(5, majit_ir::OpCode::IntMul, &[OpRef(0), OpRef(1)]);
        let ops = vec![add, g1, g2, mul];
        let lr = buf.get_live_ranges(&ops, buf.max_num_inputargs as usize);

        // Sequential index walk (start = num_inputargs = 2):
        //   IntAdd     index=2 → liveranges[2]=2,                       index → 3
        //   GuardTrue  index=3 → liveranges[2]=3,                       index → 3
        //   GuardNoExc index=3 → liveranges[2]=3 (via fail_args),       index → 3
        //   IntMul     index=3 → liveranges[0]=3, liveranges[1]=3,
        //                        liveranges[5]=3,                       index → 4
        //
        // The two void ops between IntAdd and IntMul share index 3 with
        // IntMul. Under the old `cur_index = op.pos` formula liveranges[2]
        // would have been bumped to 4 (GuardNoException's pos) instead of
        // staying at 3, and the IntMul self-def would land at 5 instead
        // of 3 — diverging from RPython parity.
        assert_eq!(lr[0], 3);
        assert_eq!(lr[1], 3);
        assert_eq!(lr[2], 3);
        // Void slots stay at 0 (no self-def for guards).
        assert_eq!(lr[3], 0);
        assert_eq!(lr[4], 0);
        // IntMul self-def written at the IntMul's slot (op.pos = 5) with
        // value = sequential index (3). Slot space is majit pos; values
        // are sequential RPython index — matching the arg-write semantics.
        assert_eq!(lr[5], 3);
    }

    /// opencoder.py:421 — `CutTrace.get_iter` overrides `iter._count =
    /// self.count` so the sequential index numbering picks up where the
    /// parent trace left off. The same slice fed through
    /// `get_live_ranges` with `index_start = num_inputargs` would assign
    /// indices starting at num_inputargs; with `index_start = cut_count`
    /// the indices start at cut_count. Verify both are reachable via the
    /// new parameter.
    #[test]
    fn test_get_live_ranges_honours_cut_trace_count() {
        let buf = TraceRecordBuffer::new(2);
        // pos 0, 1 = inputargs (live, but unreferenced in this slice)
        // pos 2    = IntAdd(i0, i1)        (non-void)
        // pos 3    = IntMul(v2, i0)        (non-void, references v2)
        let add = op_at(2, majit_ir::OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        let mul = op_at(3, majit_ir::OpCode::IntMul, &[OpRef(2), OpRef(0)]);
        let ops = vec![add, mul];

        // Full-trace caller: index starts at num_inputargs = 2.
        let lr_full = buf.get_live_ranges(&ops, buf.max_num_inputargs as usize);
        // IntAdd at index=2 → liveranges[0]=2, liveranges[1]=2,
        //                     liveranges[2]=2,                 index → 3
        // IntMul at index=3 → liveranges[2]=3, liveranges[0]=3,
        //                     liveranges[3]=3,                 index → 4
        assert_eq!(lr_full[0], 3);
        assert_eq!(lr_full[1], 2);
        assert_eq!(lr_full[2], 3);
        assert_eq!(lr_full[3], 3);

        // CutTrace caller: imagine the parent left off at sequential
        // count = 17 (e.g. 17 prior non-void ops). Pass that as
        // index_start; the live-range step values shift accordingly.
        let lr_cut = buf.get_live_ranges(&ops, 17);
        // IntAdd at index=17 → liveranges[1]=17, liveranges[2]=17,
        //                      liveranges[0]=17,                 index → 18
        // IntMul at index=18 → liveranges[2]=18, liveranges[0]=18,
        //                      liveranges[3]=18,                 index → 19
        assert_eq!(lr_cut[0], 18);
        assert_eq!(lr_cut[1], 17);
        assert_eq!(lr_cut[2], 18);
        assert_eq!(lr_cut[3], 18);
    }

    #[test]
    fn test_trace_iterator_next_basic() {
        // opencoder.py:362-406 TraceIterator.next() parity:
        //
        // Phase-1 iteration over a 2-inputarg trace producing two IntAdd
        // results. Inputargs occupy positions [0, 2); the first non-void
        // result is allocated at OpRef(2) (= start_fresh = num_inputargs).
        let ops = vec![
            op_at(2, majit_ir::OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            op_at(3, majit_ir::OpCode::IntAdd, &[OpRef(2), OpRef(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[OpRef(3)]),
        ];
        let num_inputargs = 2;

        // Phase 1 / legacy layout: start_fresh = 0 → inputargs allocated
        // at [OpRef(0), OpRef(1)], op results follow at [OpRef(2), …).
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, num_inputargs, 0);
        assert_eq!(iter.inputargs, vec![OpRef(0), OpRef(1)]);
        assert_eq!(iter._cache[0], Some(OpRef(0)));
        assert_eq!(iter._cache[1], Some(OpRef(1)));

        // First IntAdd: result at fresh OpRef(2), args translated via cache.
        let r1 = iter.next().unwrap();
        assert_eq!(r1.pos, OpRef(2));
        assert_eq!(r1.args[0], OpRef(0));
        assert_eq!(r1.args[1], OpRef(1));

        // Second IntAdd: result at fresh OpRef(3); first arg references the
        // previous result (raw pos 2 → cached OpRef(2)).
        let r2 = iter.next().unwrap();
        assert_eq!(r2.pos, OpRef(3));
        assert_eq!(r2.args[0], OpRef(2));
        assert_eq!(r2.args[1], OpRef(0));

        // Finish is void: pos unchanged, arg cached → OpRef(3).
        let finish = iter.next().unwrap();
        assert_eq!(finish.args[0], OpRef(3));
        assert!(iter.done());
    }

    #[test]
    fn test_trace_iterator_phase1_phase2_disjoint() {
        // opencoder.py:362-406 TraceIterator.next() identity parity:
        //
        // RPython distinguishes Phase 1 boxes from Phase 2 boxes by Python
        // `is`-identity even when the two iterators visit the same source
        // trace position. majit lacks separate identity from position, so
        // each phase passes a different `start_fresh` and the iterator
        // produces disjoint OpRefs by construction.
        let ops = vec![
            op_at(2, majit_ir::OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            op_at(3, majit_ir::OpCode::IntAdd, &[OpRef(2), OpRef(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[OpRef(3)]),
        ];
        let num_inputargs = 2;

        // Phase 1: start_fresh = 0 reproduces the legacy positional layout
        // (inputargs at [0, num_inputargs), op results at [num_inputargs, …)).
        let mut p1 = TraceIterator::new(&ops, 0, ops.len(), None, num_inputargs, 0);
        let mut p1_ops = Vec::new();
        while let Some(op) = p1.next() {
            p1_ops.push(op);
        }
        let p1_high_water = p1._fresh;
        assert_eq!(p1_high_water, 4);
        // RPython `_index` walks the `_cache` slots monotonically; after
        // the 2 non-void ops it should point just past the last written
        // raw trace position (3 + 1 = 4).
        assert_eq!(p1._index, 4);

        // Phase 2: start_fresh = phase1 high water → disjoint namespace.
        let mut p2 = TraceIterator::new(&ops, 0, ops.len(), None, num_inputargs, p1_high_water);
        let mut p2_ops = Vec::new();
        while let Some(op) = p2.next() {
            p2_ops.push(op);
        }
        assert_eq!(p2.inputargs, vec![OpRef(4), OpRef(5)]);
        assert_eq!(p2_ops[0].pos, OpRef(6));
        assert_eq!(p2_ops[0].args[0], OpRef(4));
        assert_eq!(p2_ops[0].args[1], OpRef(5));
        assert_eq!(p2_ops[1].pos, OpRef(7));
        assert_eq!(p2_ops[1].args[0], OpRef(6));
        assert_eq!(p2_ops[1].args[1], OpRef(4));
        assert_eq!(p2_ops[2].args[0], OpRef(7));

        // No Phase 1 OpRef equals any Phase 2 OpRef.
        let p1_set: std::collections::HashSet<u32> = p1_ops
            .iter()
            .map(|op| op.pos.0)
            .filter(|&p| p != u32::MAX)
            .collect();
        let p2_set: std::collections::HashSet<u32> = p2_ops
            .iter()
            .map(|op| op.pos.0)
            .filter(|&p| p != u32::MAX)
            .collect();
        assert!(p1_set.is_disjoint(&p2_set));
    }

    #[test]
    fn test_trace_iterator_constants_passthrough() {
        // opencoder.py:321-335 _untag(): TAGINT/TAGCONSTPTR/TAGCONSTOTHER
        // pass through unchanged. In majit, constant OpRefs (is_constant
        // high-bit marker) must NOT be remapped through `_cache`.
        let const_ref = OpRef::from_const(5);
        let ops = vec![op_at(1, majit_ir::OpCode::IntAdd, &[OpRef(0), const_ref])];
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, 1, 0);
        let r = iter.next().unwrap();
        assert_eq!(r.pos, OpRef(1));
        assert_eq!(r.args[0], OpRef(0));
        assert_eq!(r.args[1], const_ref);
    }

    #[test]
    fn test_trace_iterator_replace_last_cached_shifted_phase() {
        // opencoder.py:282-284 replace_last_cached parity. RPython walks
        // `_cache` densely by ops-count (`_index`), so `_cache[_index - 1]`
        // always targets the slot just written by `next()`. majit's
        // `_index` is separated from the fresh OpRef counter — in a
        // shifted phase (Phase 2 with `start_fresh` >> 0) the cache key
        // is still the raw trace position, so `_index` must stay in raw
        // trace position coordinates.
        let ops = vec![
            op_at(1, majit_ir::OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            op_at(2, majit_ir::OpCode::IntAdd, &[OpRef(1), OpRef(0)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[OpRef(2)]),
        ];
        // Shifted phase: start_fresh = 100 so any confusion between
        // _index (raw trace position) and _fresh (fresh OpRef counter)
        // surfaces as an out-of-bounds panic or stale cache slot.
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, 1, 100);
        let r1 = iter.next().unwrap();
        // After writing raw pos 1, _index should be 2 (next slot).
        assert_eq!(iter._index, 2);
        assert_eq!(r1.pos, OpRef(101));
        // replace_last_cached(oldbox=OpRef(101), new_box=OpRef(999))
        // must target _cache[_index - 1] = _cache[1], where the last
        // write placed OpRef(101).
        iter.replace_last_cached(OpRef(101), OpRef(999));
        assert_eq!(iter._cache[1], Some(OpRef(999)));

        // The next op references raw pos 1 as its first arg, so the
        // replaced value should flow through _untag → _get.
        let r2 = iter.next().unwrap();
        assert_eq!(r2.args[0], OpRef(999));
        assert_eq!(r2.args[1], OpRef(100));
        // After writing raw pos 2, _index advances to 3.
        assert_eq!(iter._index, 3);
    }

    #[test]
    fn test_trace_iterator_fail_args_remapped() {
        // opencoder.py:362-406 next() routes guard fail_args through the
        // same _untag/_get path as regular args.
        let mut guard = op_at(2, majit_ir::OpCode::GuardTrue, &[OpRef(1)]);
        guard.fail_args = Some(vec![OpRef(0), OpRef(1)].into());
        let ops = vec![
            op_at(1, majit_ir::OpCode::IntEq, &[OpRef(0), OpRef(0)]),
            guard,
        ];
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, 1, 10);
        let r1 = iter.next().unwrap();
        assert_eq!(r1.pos, OpRef(11));
        assert_eq!(r1.args[0], OpRef(10));
        assert_eq!(r1.args[1], OpRef(10));
        let r2 = iter.next().unwrap();
        assert_eq!(r2.args[0], OpRef(11));
        let fa = r2.fail_args.as_ref().unwrap();
        assert_eq!(fa[0], OpRef(10));
        assert_eq!(fa[1], OpRef(11));
    }

    #[test]
    fn test_trace_iterator_overwrites_inputarg_cache_slot_with_fresh_result() {
        // RPython next() allocates a fresh result box even when the raw trace
        // position collides with an already-seeded inputarg cache slot.
        let ops = vec![
            op_at(3, majit_ir::OpCode::GetfieldRawI, &[OpRef(0)]),
            op_at(1, majit_ir::OpCode::GetarrayitemGcR, &[OpRef(3), OpRef(1)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[OpRef(1)]),
        ];
        let mut iter = TraceIterator::new(&ops, 0, ops.len(), None, 4, 100);

        assert_eq!(
            iter.inputargs,
            vec![OpRef(100), OpRef(101), OpRef(102), OpRef(103)]
        );
        assert_eq!(iter._cache[1], Some(OpRef(101)));

        let op0 = iter.next().unwrap();
        assert_eq!(op0.pos, OpRef(104));
        assert_eq!(op0.args.as_slice(), &[OpRef(100)]);

        let op1 = iter.next().unwrap();
        assert_eq!(op1.args.as_slice(), &[OpRef(104), OpRef(101)]);
        assert_eq!(op1.pos, OpRef(105));
        assert_eq!(iter._cache[1], Some(OpRef(105)));

        let finish = iter.next().unwrap();
        assert_eq!(finish.args.as_slice(), &[OpRef(105)]);
    }

    #[test]
    fn test_box_array_iter() {
        let values = vec![tag(TAGINT, 42), tag(TAGBOX, 10), tag(TAGCONSTPTR, 0)];
        let mut iter = BoxArrayIter::new(&values);
        assert_eq!(iter.remaining(), 3);

        let (k, v) = iter.next_decoded().unwrap();
        assert_eq!(k, TAGINT);
        assert_eq!(v, 42);

        let (k, v) = iter.next_decoded().unwrap();
        assert_eq!(k, TAGBOX);
        assert_eq!(v, 10);

        assert_eq!(iter.remaining(), 1);
        assert!(!iter.done());
        iter.next_value();
        assert!(iter.done());
    }

    // Phase B1 intentionally drops `test_trace_record_buffer` and
    // `test_trace_record_snapshot`: they exercised the pre-Phase-B
    // legacy record_op / capture_resumedata / snapshots API that
    // diverged from rpython/jit/metainterp/opencoder.py and has been
    // removed. Phase B3 (wire format), B5 (guard 0-placeholder), B7
    // (capture_resumedata + create_top_snapshot), and Phase C
    // (RPython test_opencoder.py ports) replace them with parity
    // tests that exercise the RPython-shaped API.

    #[test]
    fn test_trace_record_buffer_initial_state_b1() {
        // Phase B1 smoke test: `Trace::new(max_num_inputargs)` seeds the
        // counters per opencoder.py:497-499 (all three = max_num_inputargs)
        // and reserves snapshot_array_data index 0 for the empty-length
        // array (opencoder.py:489 + :728-733).
        let buf = TraceRecordBuffer::new(3);
        assert_eq!(buf.max_num_inputargs, 3);
        assert_eq!(buf._count, 3);
        assert_eq!(buf._index, 3);
        assert_eq!(buf._start, 3);
        assert_eq!(buf._pos, 0);
        assert_eq!(buf.length(), 0);
        assert!(buf.inputargs.is_empty());
        assert!(!buf.tag_overflow);
        // _descrs starts with one None entry so index 0 means "no descr".
        assert_eq!(buf._descrs.len(), 1);
        assert!(buf._descrs[0].is_none());
        // _refs starts with nullptr so index 0 means "null".
        assert_eq!(buf._refs, vec![0u64]);
        // _snapshot_array_data has one signed-varint-encoded 0 recorded.
        assert!(!buf._snapshot_array_data.is_empty());
    }

    /// Phase B9: cut_point/cut_at shape matches opencoder.py:567-575.
    /// `cut_point` returns the 5-tuple (_pos, _count, _index,
    /// len(_snapshot_data), len(_snapshot_array_data)). `cut_at`
    /// restores _pos, _count, _index; the snapshot-chain lengths in
    /// slots 3/4 are observed but NOT rewound (they grow monotonically
    /// in RPython; bridge compilation reuses earlier snapshots).
    #[test]
    fn test_cut_point_five_tuple_b9() {
        let mut buf = TraceRecordBuffer::new(1);
        let before = buf.cut_point();
        assert_eq!(before.0, buf._pos);
        assert_eq!(before.1, buf._count);
        assert_eq!(before.2, buf._index);
        assert_eq!(before.3, buf._snapshot_data.len());
        assert_eq!(before.4, buf._snapshot_array_data.len());

        // Advance all five independently.
        buf.record_op1(OpCode::GuardTrue, tag(TAGBOX, 0) as i64, 0);
        let array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(7));
        let _ = buf.create_top_snapshot(1, 10, array, &[], &[], true);
        let after = buf.cut_point();
        assert_ne!(after.0, before.0);
        assert_ne!(after.1, before.1); // _count bumped by guard.
        assert_eq!(after.2, before.2); // _index did NOT bump (guard is void).
        assert_ne!(after.3, before.3); // _snapshot_data grew.
        assert_ne!(after.4, before.4); // _snapshot_array_data grew.

        // cut_at restores only the first three fields.
        buf.cut_at(before);
        assert_eq!(buf._pos, before.0);
        assert_eq!(buf._count, before.1);
        assert_eq!(buf._index, before.2);
    }

    /// Phase B7 smoke test: record a guard, then call
    /// `create_top_snapshot` with small box arrays. Verify:
    /// - The guard's trailing 0-placeholder is patched to the snapshot
    ///   index returned by `create_top_snapshot`.
    /// - `_snapshot_data` grew, `_snapshot_array_data` grew for
    ///   non-empty arrays, `_total_snapshots` bumped.
    /// - Length-0 arrays collapse to index 0 (the pre-seeded slot).
    #[test]
    fn test_create_top_snapshot_b7() {
        let mut buf = TraceRecordBuffer::new(1);
        // Record a guard so we have a 0-placeholder to patch.
        buf.record_op1(OpCode::GuardTrue, tag(TAGBOX, 0) as i64, 0);
        let pos_after_guard = buf._pos;
        let total_before = buf._total_snapshots;
        let sd_len_before = buf._snapshot_data.len();

        // Build the active-boxes array via new_array + _add_box_to_storage,
        // mimicking how the frame's get_list_of_active_boxes drives it.
        let array = buf.new_array(2);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(1));
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(2));

        // No vable / vref boxes.
        let s = buf.create_top_snapshot(0, 0, array, &[], &[], true);
        assert_eq!(s as usize, sd_len_before);
        assert_eq!(buf._total_snapshots, total_before + 1);
        assert!(buf._snapshot_data.len() > sd_len_before);

        // The guard's trailing 2 zero bytes should have been replaced
        // with `s` (which fits in 2 bytes since it's 0 here).
        assert_eq!(buf._pos, pos_after_guard);
        let (decoded, _) = decode_varint_signed(&buf._ops[pos_after_guard - 2..]);
        assert_eq!(decoded, s, "guard descr slot not patched to snapshot index");
    }

    /// Phase B7 smoke test: create_snapshot patches the previous
    /// snapshot's NEEDS_PATCHING sentinel with SNAPSHOT_PREV_COMES_NEXT
    /// (-1) and then writes a new snapshot with its own
    /// NEEDS_PATCHING sentinel at the end.
    #[test]
    fn test_create_snapshot_prev_chain_b7() {
        let mut buf = TraceRecordBuffer::new(1);
        buf.record_op1(OpCode::GuardTrue, tag(TAGBOX, 0) as i64, 0);

        // First: a top snapshot with is_last=false so it leaves a
        // NEEDS_PATCHING sentinel for the next create_snapshot to chain
        // onto.
        let top_array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(7));
        let _top_idx = buf.create_top_snapshot(1, 10, top_array, &[], &[], false);
        // The snapshot_data now ends with NEEDS_PATCHING (bytes 0x7d 0xff).
        let n = buf._snapshot_data.len();
        assert_eq!(buf._snapshot_data[n - 2], 0x7d);
        assert_eq!(buf._snapshot_data[n - 1], 0xff);

        // Now append a parent snapshot: create_snapshot patches the
        // trailing sentinel to SNAPSHOT_PREV_COMES_NEXT and writes the
        // new snapshot record.
        let parent_array = buf.new_array(1);
        buf._add_box_to_storage(TraceRecordBuffer::_encode_smallint(9));
        let _parent_idx = buf.create_snapshot(0, 5, parent_array, true);
        // The top snapshot's prev slot is now COMES_NEXT (-1 encodes
        // as 0x7f 0xff in the signed-varint 2-byte layout).
        // Decoding at the original offset should yield -1.
        let (decoded, _) = decode_varint_signed(&buf._snapshot_data[n - 2..n]);
        assert_eq!(decoded, SNAPSHOT_PREV_COMES_NEXT as i64);
    }

    /// Phase B6 parity: `_count` advances on EVERY recorded op, `_index`
    /// advances ONLY for ops whose `opclasses[opnum].type != 'v'`
    /// (opencoder.py:660-662). Interleave non-void (IntAdd) and void
    /// (GuardTrue) ops and verify the two counters progress according
    /// to the RPython rule.
    #[test]
    fn test_void_op_index_policy_b6() {
        let mut buf = TraceRecordBuffer::new(2);
        // Initial state: _start = _count = _index = max_num_inputargs = 2.
        assert_eq!(buf._count, 2);
        assert_eq!(buf._index, 2);

        // IntAdd (non-void Int result) — bumps both counters.
        let p1 = buf.record_op2(
            OpCode::IntAdd,
            tag(TAGBOX, 0) as i64,
            tag(TAGBOX, 1) as i64,
            0,
        );
        assert_eq!(p1, 2, "record_op returns pre-bump _index");
        assert_eq!(buf._count, 3);
        assert_eq!(buf._index, 3);

        // GuardTrue (void) — bumps _count only.
        let p2 = buf.record_op1(OpCode::GuardTrue, tag(TAGBOX, 2) as i64, 0);
        assert_eq!(
            p2, 3,
            "guard returns pre-bump _index (same as last non-void)"
        );
        assert_eq!(buf._count, 4);
        assert_eq!(buf._index, 3, "void must NOT bump _index");

        // IntSub (non-void) — bumps both again. Note _index picks up
        // from 3, not from _count=4, so the next non-void result
        // position is 3.
        let p3 = buf.record_op2(
            OpCode::IntSub,
            tag(TAGBOX, 0) as i64,
            tag(TAGBOX, 2) as i64,
            0,
        );
        assert_eq!(p3, 3, "non-void after void uses the void's pre-bump _index");
        assert_eq!(buf._count, 5);
        assert_eq!(buf._index, 4);
    }

    /// Phase B5 smoke test: guard records trailing descr as 0 (two
    /// \\x00 bytes in the 2-byte signed-varint form).
    /// `patch_last_guard_descr_slot` rewinds 2 bytes and replaces them
    /// with the actual snapshot index via append_int — works for both
    /// 2-byte and 4-byte encodings of the patched value.
    #[test]
    fn test_guard_descr_placeholder_patch_b5() {
        // Phase B3's record_op1 writes opnum + arg + descr-int. For a
        // guard with has_descr + descr_index=0, the trailing 2 bytes
        // are the zero-placeholder.
        let mut buf = TraceRecordBuffer::new(1);
        buf.record_op1(OpCode::GuardTrue, tag(TAGBOX, 0) as i64, 0);
        let pos_before_patch = buf._pos;
        assert_eq!(
            buf._ops[pos_before_patch - 2],
            0u8,
            "guard descr placeholder missing"
        );
        assert_eq!(buf._ops[pos_before_patch - 1], 0u8);

        // Patch to a small snapshot index (fits in 2 bytes).
        buf.patch_last_guard_descr_slot(42);
        assert_eq!(
            buf._pos, pos_before_patch,
            "2-byte patch should keep _pos at the same offset"
        );
        // Verify we can decode it back.
        let tail_start = pos_before_patch - 2;
        let (decoded, consumed) = decode_varint_signed(&buf._ops[tail_start..]);
        assert_eq!(decoded, 42);
        assert_eq!(consumed, 2);

        // Patch to a large snapshot index (requires 4 bytes).
        let mut buf2 = TraceRecordBuffer::new(1);
        buf2.record_op1(OpCode::GuardTrue, tag(TAGBOX, 0) as i64, 0);
        let pos_before = buf2._pos;
        buf2.patch_last_guard_descr_slot(MAX_VALUE);
        assert_eq!(
            buf2._pos,
            pos_before + 2,
            "4-byte patch should grow _pos by 2"
        );
    }

    /// Phase B4 smoke test: _encode_descr returns `global_index + 1`
    /// for descrs with a global index, else appends to _descrs and
    /// returns `all_descrs_len + len(_descrs) - 1 + 1`.
    #[test]
    fn test_encode_descr_b4() {
        use std::sync::Arc;

        #[derive(Debug)]
        struct D {
            idx: i32,
        }
        impl majit_ir::Descr for D {
            fn index(&self) -> u32 {
                0
            }
            fn get_descr_index(&self) -> i32 {
                self.idx
            }
        }

        let mut buf = TraceRecordBuffer::new(0);
        // Global descrs return `get_descr_index() + 1`.
        let d_global: majit_ir::DescrRef = Arc::new(D { idx: 17 });
        assert_eq!(buf._encode_descr(&d_global, 100), 18);
        assert_eq!(buf._descrs.len(), 1, "global descr must not append");

        // Local descrs (get_descr_index == -1) append and return
        // `all_descrs_len + len(_descrs) - 1 + 1`.
        let d_local: majit_ir::DescrRef = Arc::new(D { idx: -1 });
        // first local append: _descrs goes 1 → 2; encoded = 100 + 2.
        assert_eq!(buf._encode_descr(&d_local, 100), 102);
        assert_eq!(buf._descrs.len(), 2);
        // second local append: _descrs goes 2 → 3; encoded = 100 + 3.
        let d_local2: majit_ir::DescrRef = Arc::new(D { idx: -1 });
        assert_eq!(buf._encode_descr(&d_local2, 100), 103);
        assert_eq!(buf._descrs.len(), 3);
    }

    /// Phase B3 smoke test: fixed-arity record_op* does NOT write a
    /// count varint (opencoder.py:642-650), only the opnum byte + args.
    /// Variadic record_op writes opnum + count + args. _count bumps
    /// once per call; _index bumps only for non-void results.
    #[test]
    fn test_record_op_wire_format_b3() {
        let mut buf = TraceRecordBuffer::new(2);
        let start_index = buf._index;
        let start_count = buf._count;
        // IntAdd is fixed-arity 2, result_type Int → non-void.
        // Args passed as RPython-encoded tagged values.
        let pos = buf.record_op2(
            OpCode::IntAdd,
            tag(TAGBOX, 0) as i64,
            tag(TAGBOX, 1) as i64,
            0, // no descr
        );
        assert_eq!(pos, start_index, "record_op returns pre-bump _index");
        assert_eq!(buf._count, start_count + 1);
        assert_eq!(buf._index, start_index + 1, "non-void should bump _index");
        // opnum byte at [0], then 2 varints for args. IntAdd has no descr.
        assert_eq!(buf._ops[0], OpCode::IntAdd.as_u16() as u8);
        // 2 args × 2-byte varint each = 4 bytes after opnum.
        assert_eq!(buf._pos, 1 + 2 + 2, "opnum + 2 varint-2 args");

        // A void op must not bump _index but must bump _count.
        let void_start_index = buf._index;
        let void_start_count = buf._count;
        // GuardTrue is fixed-arity 1, result_type Void, has_descr.
        let _ = buf.record_op1(OpCode::GuardTrue, tag(TAGBOX, 0) as i64, 0);
        assert_eq!(buf._index, void_start_index, "void should NOT bump _index");
        assert_eq!(buf._count, void_start_count + 1);
    }

    #[test]
    fn test_trace_cut_point_roundtrip_b1() {
        // opencoder.py:567-575 cut_point / cut_at tuple shape:
        // (_pos, _count, _index, len(_snapshot_data), len(_snapshot_array_data)).
        let mut buf = TraceRecordBuffer::new(2);
        let p0 = buf.cut_point();
        // append_byte advances _pos; append_snapshot_*_int extends the chains.
        buf.append_byte(0x42);
        buf.append_snapshot_data_int(123);
        buf.append_snapshot_array_data_int(42);
        let p1 = buf.cut_point();
        assert_ne!(p0, p1);
        buf.cut_at(p0);
        assert_eq!(buf._pos, p0.0);
        assert_eq!(buf._count, p0.1);
        assert_eq!(buf._index, p0.2);
        // RPython's cut_at intentionally does NOT rewind the snapshot chains
        // (they grow monotonically). Only _pos/_count/_index come back.
    }
}
