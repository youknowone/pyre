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
        // if res.type != 'v':
        //     self._cache[self._index] = res
        //     self._index += 1
        if !src.pos.is_none() && src.opcode.result_type() != majit_ir::Type::Void {
            let orig = src.pos.0 as usize;
            if orig >= self._cache.len() {
                self._cache.resize(orig + 1, None);
            }
            let fresh = if let Some(existing) = self._cache[orig] {
                existing
            } else {
                // majit fresh OpRef allocation: allocate from `_fresh`
                // (independent of `_index` so phase 2 can start above
                // phase 1's high water).
                let f = OpRef(self._fresh);
                self._fresh += 1;
                self._cache[orig] = Some(f);
                f
            };
            res.pos = fresh;
            // RPython `_index` parity: advance past the cache slot we
            // just wrote. In RPython this happens via `_index += 1`
            // because `_index` == cache slot index; in majit the cache
            // key is the raw trace position, so we assign `orig + 1`
            // directly to keep `_cache[_index - 1]` (`replace_last_cached`)
            // pointing at the slot we just wrote.
            self._index = orig as u32 + 1;
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
        let total = self.num_ops.max(computed_max).max(self.num_inputargs);
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
        buf.num_ops = ops.len();
        // Pretend the recorder allocated positions 0..5 (3 inputargs + 2 ops).
        // The recorder's internal counter would normally bump op_count;
        // for the unit test we set it explicitly.
        let total = 5;
        buf.num_inputargs = 3;
        // Inject the op_count via a backdoor: the field is private, so use
        // get_live_ranges with the exact slice that has the right max pos.
        // Full-trace caller passes num_inputargs as t._count.
        let lr = buf.get_live_ranges(&ops, buf.num_inputargs);
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
        let lr = buf.get_live_ranges(&ops, buf.num_inputargs);
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
        let lr = buf.get_live_ranges(&ops, buf.num_inputargs);
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
        let lr = buf.get_live_ranges(&ops, buf.num_inputargs);

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
        let lr_full = buf.get_live_ranges(&ops, buf.num_inputargs);
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
