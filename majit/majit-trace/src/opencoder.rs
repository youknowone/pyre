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
        // Encode descriptor: 0 = none, else descr_index + 1.
        if let Some(ref descr) = op.descr {
            let idx = descr.index();
            if idx != u32::MAX {
                encode_varint(&mut buf, (idx as u64) + 1);
            } else {
                buf.push(1); // has descriptor but no index
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

// ── Signed varint encoding (opencoder.py) ──

/// Encode a signed i64 as a varint (zigzag encoding).
/// opencoder.py: encode_varint_signed
pub fn encode_varint_signed(buf: &mut Vec<u8>, value: i64) {
    // Zigzag: map negative to odd, positive to even
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint(buf, zigzag);
}

/// Decode a signed varint (zigzag encoding). Returns (value, bytes_consumed).
/// opencoder.py: decode_varint_signed
pub fn decode_varint_signed(buf: &[u8]) -> (i64, usize) {
    let (zigzag, consumed) = decode_varint(buf);
    let value = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
    (value, consumed)
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

#[inline]
pub fn skip_varint_signed(buf: &[u8], mut index: usize, skip: usize) -> usize {
    assert!(skip > 0);
    for _ in 0..skip {
        let byte = buf[index];
        if byte & 0x80 != 0 {
            index += 2;
        } else {
            index += 1;
        }
    }
    index
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
            let _top = &storage.top_snapshots[top_snapshot_idx];
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

    /// Clone the storage data without rooting — for consumption after
    /// the trace is finalized. Refreshes from GC first.
    /// RPython: Trace._refs is consumed as a plain list after tracing_done().
    pub fn snapshot_unrooted(&mut self) -> SnapshotStorage {
        self.refresh_from_gc();
        SnapshotStorage {
            snapshots: self.snapshots.clone(),
            top_snapshots: self.top_snapshots.clone(),
            const_refs: self.const_refs.clone(),
            const_bigints: self.const_bigints.clone(),
            const_floats: self.const_floats.clone(),
            // Unrooted copy — no shadow stack tracking.
            rooted_ref_indices: Vec::new(),
            shadow_stack_base: majit_gc::shadow_stack::depth(),
        }
    }
}

impl Drop for SnapshotStorage {
    fn drop(&mut self) {
        self.release_roots();
    }
}

// ── Trace Iterator (opencoder.py: TraceIterator) ──

/// opencoder.py: TraceIterator — iterates operations in a trace
/// with dead range tracking.
///
/// The iterator walks through encoded trace operations and maintains
/// live/dead range information for each value (OpRef). This enables
/// the optimizer to know which values are still in use at each point.
pub struct TraceIterator<'a> {
    /// The operations being iterated.
    ops: &'a [majit_ir::Op],
    /// Current position in the operation list.
    pos: usize,
    /// Live range end for each OpRef: maps OpRef → last use position.
    /// opencoder.py: _deadranges
    live_range_end: Vec<usize>,
    /// Whether live ranges have been computed.
    ranges_computed: bool,
}

impl<'a> TraceIterator<'a> {
    /// Create a new TraceIterator over the given operations.
    pub fn new(ops: &'a [majit_ir::Op]) -> Self {
        TraceIterator {
            ops,
            pos: 0,
            live_range_end: Vec::new(),
            ranges_computed: false,
        }
    }

    /// Get the next operation, or None if exhausted.
    pub fn next_op(&mut self) -> Option<&'a majit_ir::Op> {
        if self.pos < self.ops.len() {
            let op = &self.ops[self.pos];
            self.pos += 1;
            Some(op)
        } else {
            None
        }
    }

    /// Current position in the operation list.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Whether all operations have been consumed.
    pub fn done(&self) -> bool {
        self.pos >= self.ops.len()
    }

    /// Total number of operations.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// opencoder.py: get_dead_ranges() — compute live ranges for all values.
    /// Returns a reference to the live_range_end map.
    pub fn compute_dead_ranges(&mut self) -> &[usize] {
        if self.ranges_computed {
            return &self.live_range_end;
        }

        // Find the maximum OpRef used
        let max_ref = self
            .ops
            .iter()
            .flat_map(|op| {
                op.args
                    .iter()
                    .chain(op.fail_args.iter().flat_map(|fa| fa.iter()))
                    .map(|r| r.0 as usize)
            })
            .max()
            .unwrap_or(0);

        self.live_range_end = vec![0; max_ref + 1];

        // Walk backwards: last use of each OpRef is its dead range end.
        for (i, op) in self.ops.iter().enumerate().rev() {
            for arg in &op.args {
                let idx = arg.0 as usize;
                if idx < self.live_range_end.len() && self.live_range_end[idx] == 0 {
                    self.live_range_end[idx] = i;
                }
            }
            if let Some(ref fa) = op.fail_args {
                for arg in fa.iter() {
                    let idx = arg.0 as usize;
                    if idx < self.live_range_end.len() && self.live_range_end[idx] == 0 {
                        self.live_range_end[idx] = i;
                    }
                }
            }
        }

        self.ranges_computed = true;
        &self.live_range_end
    }

    /// Check if a value is dead at the given position.
    /// opencoder.py: is_dead(opref, pos)
    pub fn is_dead_at(&self, opref: majit_ir::OpRef, pos: usize) -> bool {
        let idx = opref.0 as usize;
        if idx >= self.live_range_end.len() {
            return true; // never used
        }
        self.live_range_end[idx] < pos
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
/// Records operations and snapshots during tracing. The recorded data
/// can be iterated via `TraceIterator` or serialized via `encode_trace`.
pub type Trace = TraceRecordBuffer;
pub struct TraceRecordBuffer {
    /// Encoded operation stream (binary).
    data: Vec<u8>,
    /// Number of input arguments.
    num_inputargs: usize,
    /// Number of recorded operations.
    num_ops: usize,
    /// Snapshot storage.
    snapshots: SnapshotStorage,
    /// Cut points for bridge compilation.
    cut_points: Vec<usize>,
    /// Whether the trace has overflowed (too long).
    overflow: bool,
    /// Maximum allowed data size.
    max_size: usize,
    /// Memoized integer constants used by `_cached_const_int()`.
    const_int_cache: HashMap<i64, u32>,
    /// Memoized pointer constants used by `_cached_const_ptr()`.
    const_ptr_cache: HashMap<u64, u32>,
}

impl TraceRecordBuffer {
    /// Create a new trace recording buffer.
    pub fn new(num_inputargs: usize) -> Self {
        TraceRecordBuffer {
            data: Vec::with_capacity(4096),
            num_inputargs,
            num_ops: 0,
            snapshots: SnapshotStorage::new(),
            cut_points: Vec::new(),
            overflow: false,
            max_size: 1 << 20, // 1MB default
            const_int_cache: HashMap::new(),
            const_ptr_cache: HashMap::new(),
        }
    }

    /// opencoder.py: tag_overflow_imminent()
    /// Check if the recording buffer is nearly full.
    pub fn tag_overflow_imminent(&self) -> bool {
        self.data.len() > self.max_size * 9 / 10
    }

    /// opencoder.py: tracing_done()
    /// Finalize the trace and check for overflow.
    pub fn tracing_done(&mut self) -> bool {
        if self.data.len() > self.max_size {
            self.overflow = true;
        }
        !self.overflow
    }

    /// Record an operation.
    pub fn record_op(&mut self, opcode: u16, num_args: u8) {
        encode_varint(&mut self.data, opcode as u64);
        self.data.push(num_args);
        self.num_ops += 1;
    }

    /// opencoder.py: _op_start() — mark start offset for current opcode.
    pub fn _op_start(&self) -> usize {
        self.data.len()
    }

    /// opencoder.py: _op_end() — mark end offset for current opcode.
    pub fn _op_end(&self) -> usize {
        self.data.len()
    }

    /// RPython-compatible: `Trace._encode(op, arg)`.
    ///
    /// Returns a tagged representation for values stored in snapshots and
    /// arrays.
    #[inline]
    pub fn _encode(&self, kind: u8, value: u32) -> u32 {
        tag(kind, value)
    }

    /// RPython-compatible: `Trace._cached_const_int(value)`.
    /// Reuse pooled small constants to keep indexes stable.
    pub fn _cached_const_int(&mut self, value: i64) -> u32 {
        if let Some(idx) = self.const_int_cache.get(&value) {
            return *idx;
        }
        let idx = self.snapshots.add_const_bigint(value);
        self.const_int_cache.insert(value, idx);
        idx
    }

    /// RPython-compatible: `Trace._cached_const_ptr(ptr)`.
    /// Reuse pooled pointer constants to keep indexes stable.
    pub fn _cached_const_ptr(&mut self, ptr: u64) -> u32 {
        if let Some(idx) = self.const_ptr_cache.get(&ptr) {
            return *idx;
        }
        let idx = self.snapshots.add_const_ref(ptr);
        self.const_ptr_cache.insert(ptr, idx);
        idx
    }

    /// opencoder.py: _encode_descr(descr) — encode descriptor index.
    pub fn _encode_descr(&self, descr: u32) -> u32 {
        descr
    }

    /// opencoder.py: _add_box_to_storage(value).
    ///
    /// Store box value into a dedicated pool; return index for compatibility.
    pub fn _add_box_to_storage(&mut self, value: u32) -> usize {
        self.snapshots.const_refs.push(value as u64);
        self.snapshots.const_refs.len() - 1
    }

    /// opencoder.py: append_byte(byte) — append raw byte to trace data.
    pub fn append_byte(&mut self, byte: u8) {
        self.data.push(byte);
    }

    /// opencoder.py: append_int(value) — append LEB128 value to trace data.
    pub fn append_int(&mut self, value: u32) {
        self.record_arg(value);
    }

    /// opencoder.py: append_snapshot_array_data_int(data, value).
    pub fn append_snapshot_array_data_int(data: &mut Vec<u32>, value: i32) {
        data.push(value as u32);
    }

    /// opencoder.py: append_snapshot_data_int(data, value).
    pub fn append_snapshot_data_int(data: &mut Vec<u32>, value: i32) {
        data.push(value as u32);
    }

    /// opencoder.py: _encode_snapshot(snapshot) → encoded snapshot payload.
    pub fn _encode_snapshot(snapshot: &Snapshot) -> Vec<u32> {
        snapshot.values.clone()
    }

    /// opencoder.py: create_snapshot(values) — helper for building a snapshot.
    pub fn create_snapshot(&self, values: Vec<u32>) -> Snapshot {
        Snapshot {
            values,
            prev: None,
            jitcode_index: 0,
            pc: 0,
        }
    }

    /// opencoder.py: snapshot_add_prev(snapshot, prev).
    pub fn snapshot_add_prev(snapshot: &mut Snapshot, prev: Option<usize>) {
        snapshot.prev = prev;
    }

    /// opencoder.py compatibility: record fixed-arity opcodes with explicit
    /// argument slots.
    pub fn record_op0(&mut self, opcode: u16) {
        self.record_op(opcode, 0);
    }

    /// opencoder.py compatibility: record fixed-arity opcodes with explicit
    /// argument slots.
    pub fn record_op1(&mut self, opcode: u16, arg0: u32) {
        self.record_op(opcode, 1);
        self.record_arg(arg0);
    }

    /// opencoder.py compatibility: record fixed-arity opcodes with explicit
    /// argument slots.
    pub fn record_op2(&mut self, opcode: u16, arg0: u32, arg1: u32) {
        self.record_op(opcode, 2);
        self.record_arg(arg0);
        self.record_arg(arg1);
    }

    /// opencoder.py compatibility: record fixed-arity opcodes with explicit
    /// argument slots.
    pub fn record_op3(&mut self, opcode: u16, arg0: u32, arg1: u32, arg2: u32) {
        self.record_op(opcode, 3);
        self.record_arg(arg0);
        self.record_arg(arg1);
        self.record_arg(arg2);
    }

    /// RPython-compatible: encode a boxed value array.
    pub fn _list_of_boxes(&self, boxes: &[u32]) -> Vec<u32> {
        boxes.iter().map(|&b| self._encode(TAGBOX, b)).collect()
    }

    /// RPython-compatible: encode boxes for virtualizable state.
    pub fn _list_of_boxes_virtualizable(&self, boxes: &[u32]) -> Vec<u32> {
        boxes.iter().map(|&b| self._encode(TAGBOX, b)).collect()
    }

    /// RPython-compatible helper: return a copied encoded array payload.
    pub fn new_array(&self, items: &[u32]) -> Vec<u32> {
        items.to_vec()
    }

    /// Record an argument value.
    pub fn record_arg(&mut self, value: u32) {
        encode_varint(&mut self.data, value as u64);
    }

    /// opencoder.py: cut_point()
    /// Mark the current position as a potential bridge entry.
    pub fn cut_point(&mut self) {
        self.cut_points.push(self.data.len());
    }

    /// Number of recorded operations.
    pub fn num_ops(&self) -> usize {
        self.num_ops
    }

    /// Whether the trace overflowed.
    pub fn overflowed(&self) -> bool {
        self.overflow
    }

    /// Get the snapshot storage.
    pub fn snapshots(&self) -> &SnapshotStorage {
        &self.snapshots
    }

    /// Get mutable snapshot storage (for adding snapshots during tracing).
    pub fn snapshots_mut(&mut self) -> &mut SnapshotStorage {
        &mut self.snapshots
    }

    /// opencoder.py: capture_resumedata(snapshot)
    /// Record a snapshot at the current position (for guard resume data).
    pub fn capture_resumedata(&mut self, snapshot: Snapshot) -> usize {
        self.snapshots.add_snapshot(snapshot)
    }

    /// opencoder.py: create_top_snapshot(frame, vable_boxes, vref_boxes)
    /// Create a top-level snapshot with virtualizable and virtual ref arrays.
    pub fn create_top_snapshot(
        &mut self,
        snapshot: Snapshot,
        vable_array_index: Option<usize>,
        vref_array_index: Option<usize>,
    ) -> usize {
        let _snap_idx = self.snapshots.add_snapshot(snapshot.clone());
        self.snapshots.add_top_snapshot(TopSnapshot {
            snapshot,
            vable_array_index,
            vref_array_index,
        })
    }

    /// opencoder.py: create_empty_top_snapshot(vable_boxes, vref_boxes)
    /// Create a top snapshot with no frame data (for bridge entry).
    pub fn create_empty_top_snapshot(
        &mut self,
        vable_array_index: Option<usize>,
        vref_array_index: Option<usize>,
    ) -> usize {
        let empty_snap = Snapshot {
            values: Vec::new(),
            prev: None,
            jitcode_index: 0,
            pc: 0,
        };
        self.create_top_snapshot(empty_snap, vable_array_index, vref_array_index)
    }

    /// opencoder.py: get_live_ranges()
    /// Compute live ranges for all recorded values.
    /// Returns a vector where index i contains the last position
    /// where value i is used.
    pub fn get_live_ranges(&self, ops: &[majit_ir::Op]) -> Vec<usize> {
        let mut iter = TraceIterator::new(ops);
        iter.compute_dead_ranges().to_vec()
    }

    /// opencoder.py: unpack()
    /// Decode the recorded trace into (inputargs, ops).
    /// Convenience method for testing and debugging.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Maximum allowed data size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Set the maximum allowed data size.
    pub fn set_max_size(&mut self, size: usize) {
        self.max_size = size;
    }

    /// opencoder.py: capture_resumedata(framestack, vable_boxes, vref_boxes)
    ///
    /// Multi-frame version: creates a chain of snapshots for the full
    /// frame stack, with the topmost frame as a TopSnapshot.
    pub fn capture_resumedata_framestack(
        &mut self,
        frame_pcs: &[u64],
        frame_slots: &[Vec<u32>],
        _virtualizable_boxes: &[u32],
        _virtualref_boxes: &[u32],
    ) -> usize {
        if frame_pcs.is_empty() {
            let empty_snap = Snapshot {
                values: vec![],
                prev: None,
                jitcode_index: 0,
                pc: 0,
            };
            let top = TopSnapshot {
                snapshot: empty_snap,
                vable_array_index: None,
                vref_array_index: None,
            };
            return self.snapshots.add_top_snapshot(top);
        }

        let n = frame_pcs.len() - 1;

        // Create snapshots bottom-up (outermost first)
        let mut parent_idx: Option<usize> = None;
        for i in 0..n {
            let snapshot = Snapshot {
                values: if i < frame_slots.len() {
                    frame_slots[i].clone()
                } else {
                    vec![]
                },
                prev: parent_idx,
                jitcode_index: 0,
                pc: frame_pcs[i] as u32,
            };
            parent_idx = Some(self.snapshots.add_snapshot(snapshot));
        }

        // Top snapshot for innermost frame
        let top_snap = Snapshot {
            values: if n < frame_slots.len() {
                frame_slots[n].clone()
            } else {
                vec![]
            },
            prev: parent_idx,
            jitcode_index: 0,
            pc: frame_pcs[n] as u32,
        };
        let top = TopSnapshot {
            snapshot: top_snap,
            vable_array_index: None,
            vref_array_index: None,
        };
        self.snapshots.add_top_snapshot(top)
    }

    /// opencoder.py: get_dead_ranges()
    ///
    /// Compute dead ranges: for each op index x, the values that are
    /// known to be dead before x. Used by the register allocator to
    /// know when to free registers.
    pub fn get_dead_ranges(&self, ops: &[majit_ir::Op]) -> Vec<usize> {
        let live_ranges = self.get_live_ranges(ops);
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
        let values = [0i64, 1, -1, 127, -128, 1000, -1000, i64::MAX, i64::MIN];
        for &val in &values {
            let mut buf = Vec::new();
            encode_varint_signed(&mut buf, val);
            let (decoded, _consumed) = decode_varint_signed(&buf);
            assert_eq!(decoded, val, "signed varint roundtrip failed for {val}");
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

    #[test]
    fn test_trace_iterator_dead_ranges() {
        let ops = vec![
            majit_ir::Op::new(majit_ir::OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            majit_ir::Op::new(majit_ir::OpCode::IntAdd, &[OpRef(0), OpRef(101)]),
            majit_ir::Op::new(majit_ir::OpCode::Finish, &[OpRef(1)]),
        ];
        let mut iter = TraceIterator::new(&ops);
        assert_eq!(iter.num_ops(), 3);
        assert!(!iter.done());

        let ranges = iter.compute_dead_ranges();
        // OpRef(100) is used at position 0 only → dead after 0
        // OpRef(101) is used at positions 0 and 1 → dead after 1
        assert!(ranges.len() > 100);

        // Check iterator works
        assert!(iter.next_op().is_some());
        assert_eq!(iter.position(), 1);
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

    #[test]
    fn test_trace_record_buffer() {
        let mut buf = TraceRecordBuffer::new(2);
        assert_eq!(buf.num_ops(), 0);
        assert!(!buf.overflowed());
        assert!(!buf.tag_overflow_imminent());

        buf.record_op(OpCode::IntAdd as u16, 2);
        buf.record_arg(100);
        buf.record_arg(101);
        assert_eq!(buf.num_ops(), 1);

        buf.cut_point();
        assert!(buf.tracing_done());
    }

    #[test]
    fn test_trace_record_snapshot() {
        let mut buf = TraceRecordBuffer::new(1);
        let snap = Snapshot {
            values: vec![tag(TAGINT, 42)],
            prev: None,
            jitcode_index: 0,
            pc: 5,
        };
        let idx = buf.capture_resumedata(snap);
        assert_eq!(idx, 0);
        assert_eq!(buf.snapshots().num_snapshots(), 1);
    }
}
