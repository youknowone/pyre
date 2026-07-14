/// IR → wasm bytecode compilation.
///
/// Generates a wasm module from majit IR ops using `wasm-encoder`.
/// Generated function signature: `(param $frame_ptr i32) (result i32)`
///
/// Frame layout in shared linear memory:
///   offset 0:       fail_index (i64)
///   offset 8:       slot[0] (i64)
///   offset 16:      slot[1] (i64)
///   ...
///   CALL_AREA_OFS:  func_ptr (i64)   — used by jit_call trampoline
///   CALL_AREA_OFS+8: num_args (i64)
///   CALL_AREA_OFS+16: arg[0] (i64)
///   CALL_AREA_OFS+24: arg[1] (i64)
///   ...
///   CALL_RESULT_OFS: result (i64)    — written by host after call
use std::collections::HashMap;

use majit_backend::BackendError;
use majit_gc::header::{GcHeader, TYPE_ID_MASK};
use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, InstructionSink, MemArg, MemoryType, Module, RefType, TableType, TypeSection,
    ValType,
};

/// Frame slot byte offset: slot[i] is at frame_ptr + 8 + i * 8.
pub const FRAME_SLOT_BASE: u64 = 8;
const SLOT_SIZE: u64 = 8;

/// Scratch i64 locals reserved past the value locals for `emit_umulhi`
/// (al, ah, bl, bh, mid1).
const UMULHI_SCRATCH: u32 = 5;

/// Call area layout in the historical fixed frame geometry.
const CALL_RESULT_OFS: u64 = 2000;
const CALL_FUNC_OFS: u64 = 2008;
const CALL_NARGS_OFS: u64 = 2016;
const CALL_ARGS_OFS: u64 = 2024;

/// Minimum frame allocation size in bytes to accommodate the call area.
pub const MIN_FRAME_BYTES: usize = 2024 + 16 * 8; // 16 max call args

/// Per-token layout of a wasm execution frame.  Every frozen geometry carries
/// the host-trampoline call area, so a later chained bridge can use it without
/// changing its source token's frame offsets.  CA callee frames alone allocate
/// the prefix ending after the Ref homes; the tail is protected by the
/// trampoline-decline floor in `compile_bridge`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameGeometry {
    /// Number of value slots before the dispatch key (including frame[0]).
    pub value_slots: usize,
    /// Byte offset of the call trampoline result word.
    pub call_result_ofs: u64,
    pub call_func_ofs: u64,
    pub call_nargs_ofs: u64,
    pub call_args_ofs: u64,
    /// Byte offset of the resume-at-LABEL key.
    pub dispatch_key_ofs: u64,
    /// Byte offset of Ref-home zero.
    pub home_slot_base: u64,
    /// Number of Ref-home slots the layout reserves.
    pub home_slots: usize,
    /// Bytes through the end of Ref homes. CA callee frames allocate exactly
    /// this many item bytes; the tail call area is intentionally omitted.
    pub ca_frame_bytes: u32,
    /// Full bytes in the frame layout, including the tail call area. Host entry
    /// frames and every chained bridge use this geometry and allocation size.
    pub frame_bytes: u32,
}

impl FrameGeometry {
    const CALL_AREA_SLOTS: usize = 3 + 16; // result, function, nargs, args

    /// Historical fixed geometry, used by direct codegen tests and by callers
    /// that deliberately need the arena-compatible layout.
    pub const fn fixed() -> Self {
        Self {
            value_slots: MIN_FRAME_BYTES / 8,
            call_result_ofs: CALL_RESULT_OFS,
            call_func_ofs: CALL_FUNC_OFS,
            call_nargs_ofs: CALL_NARGS_OFS,
            call_args_ofs: CALL_ARGS_OFS,
            dispatch_key_ofs: DISPATCH_KEY_OFS,
            home_slot_base: HOME_SLOT_BASE,
            home_slots: 0,
            ca_frame_bytes: HOME_SLOT_BASE as u32,
            frame_bytes: (MIN_FRAME_BYTES + SLOT_SIZE as usize) as u32,
        }
    }

    /// Compact frozen geometry for one token:
    /// `[value slots | dispatch key | Ref homes | call area]`.
    /// `value_slots` includes frame[0].  The trailing call area is always
    /// present, even for direct-only source traces, because later bridges are
    /// compiled against this immutable geometry.
    pub fn compact(value_slots: usize, home_slots: usize) -> Self {
        let value_slots = value_slots.max(1);
        let dispatch_key_ofs = (value_slots as u64) * SLOT_SIZE;
        let home_slot_base = dispatch_key_ofs + SLOT_SIZE;
        let ca_frame_bytes = home_slot_base + home_slots as u64 * SLOT_SIZE;
        let call_result_ofs = ca_frame_bytes;
        let call_func_ofs = call_result_ofs + SLOT_SIZE;
        let call_nargs_ofs = call_func_ofs + SLOT_SIZE;
        let call_args_ofs = call_nargs_ofs + SLOT_SIZE;
        let frame_bytes = call_result_ofs + Self::CALL_AREA_SLOTS as u64 * SLOT_SIZE;
        Self {
            value_slots,
            call_result_ofs,
            call_func_ofs,
            call_nargs_ofs,
            call_args_ofs,
            dispatch_key_ofs,
            home_slot_base,
            home_slots,
            ca_frame_bytes: ca_frame_bytes as u32,
            frame_bytes: frame_bytes as u32,
        }
    }
}

/// Byte offset of the Ref-home region within the frame. Each Ref value that is
/// live across a collecting call is given a dedicated home slot here: it is
/// null-initialized at trace entry and written on every definition
/// (store-on-def), so a home slot only ever holds null or a valid GcRef.
/// A collecting allocation registers these slots as GC roots and forwards them,
/// then the trace reloads the live Ref locals from their homes — making object
/// movement transparent without rooting Refs that never cross a collection.
///
/// In compact geometries this region follows the dispatch key and precedes the
/// trailing call area. Inert while `wasm_jit_alloc` is no-collect (epic B): the
/// extra stores write a region nothing reads until the allocator collects.
pub const HOME_SLOT_BASE: u64 = MIN_FRAME_BYTES as u64 + SLOT_SIZE;

/// Historical fixed-geometry resume-at-LABEL dispatch key (one reserved frame
/// slot, between the call area and the Ref-home region). 0 = preamble/host entry (the `vec![0i64]`
/// frame is always 0 here on a fresh `execute_token`); non-zero = a
/// loop-closing bridge re-entering a single-label peeled loop at its LABEL,
/// skipping the preamble. Compact geometries derive this offset from their
/// value-slot count and put the call area after the homes.
pub const DISPATCH_KEY_OFS: u64 = MIN_FRAME_BYTES as u64;
const _: () = assert!(HOME_SLOT_BASE == DISPATCH_KEY_OFS + SLOT_SIZE);

fn mem64(offset: u64) -> MemArg {
    MemArg {
        offset,
        align: 3,
        memory_index: 0,
    }
}

fn mem32(offset: u64) -> MemArg {
    memarg(offset, 2)
}

fn memarg(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

/// Invoke the residual-call trampoline. The historical import receives only a
/// frame pointer and therefore reads the fixed call area; compact frames use a
/// second import carrying their call-area base. The old trampoline remains
/// unchanged for fixed-layout frames.
fn emit_jit_call(sink: &mut InstructionSink<'_>, jit_call_idx: u32, frame: FrameGeometry) {
    if frame.call_result_ofs != CALL_RESULT_OFS {
        sink.i32_const(frame.call_result_ofs as i32);
    }
    sink.call(jit_call_idx);
}

/// Emit a width-correct integer load. The element address (i32) must be on
/// the stack; the result is an i64, sign- or zero-extended from `size`
/// bytes. Word-sized fields are 4 bytes on wasm32 (`isize`/`usize`/pointer),
/// 8 bytes on 64-bit; reading a fixed 8 bytes here would fold in the next
/// field's bytes on wasm32.
fn emit_sized_int_load(sink: &mut InstructionSink<'_>, offset: u64, size: usize, signed: bool) {
    match (size, signed) {
        (8, _) => {
            sink.i64_load(mem64(offset));
        }
        (4, true) => {
            sink.i32_load(memarg(offset, 2));
            sink.i64_extend_i32_s();
        }
        (4, false) => {
            sink.i32_load(memarg(offset, 2));
            sink.i64_extend_i32_u();
        }
        (2, true) => {
            sink.i32_load16_s(memarg(offset, 1));
            sink.i64_extend_i32_s();
        }
        (2, false) => {
            sink.i32_load16_u(memarg(offset, 1));
            sink.i64_extend_i32_u();
        }
        (1, true) => {
            sink.i32_load8_s(memarg(offset, 0));
            sink.i64_extend_i32_s();
        }
        (1, false) => {
            sink.i32_load8_u(memarg(offset, 0));
            sink.i64_extend_i32_u();
        }
        _ => {
            sink.i64_load(mem64(offset));
        }
    }
}

/// Emit a width-correct integer store. The stack must hold
/// `[addr_i32, value_i64]`; the low `size` bytes of the value are stored.
/// A fixed 8-byte store would clobber the adjacent field/item (or run past
/// the array end) for word-sized fields and pointer array items on wasm32.
fn emit_sized_int_store(sink: &mut InstructionSink<'_>, offset: u64, size: usize) {
    match size {
        8 => {
            sink.i64_store(mem64(offset));
        }
        4 => {
            sink.i32_wrap_i64();
            sink.i32_store(memarg(offset, 2));
        }
        2 => {
            sink.i32_wrap_i64();
            sink.i32_store16(memarg(offset, 1));
        }
        1 => {
            sink.i32_wrap_i64();
            sink.i32_store8(memarg(offset, 0));
        }
        _ => {
            sink.i64_store(mem64(offset));
        }
    }
}

/// `(field_size, is_signed)` from an op's FieldDescr; defaults to word-sized
/// signed when the descr is absent.
fn field_size_sign_from_descr(op: &Op) -> (usize, bool) {
    let descr = op.getdescr();
    if let Some(fd) = descr.as_ref().and_then(|d| d.as_field_descr()) {
        return (fd.field_size(), fd.is_field_signed());
    }
    (std::mem::size_of::<usize>(), true)
}

/// Store width for a `SetfieldGc`/`SetfieldRaw`. A pointer (`Type::Ref`) field
/// is stored at machine-word width regardless of the descr's recorded size: a
/// pointer is 4 bytes on wasm32, so a fixed 8-byte store would clobber the
/// adjacent field. There is no `SetfieldGcR` opcode, so the field type is the
/// only signal — mirroring the `GetfieldGcR` read, which always loads pointers
/// at i32 width. Non-pointer fields use the descr's true field width.
fn setfield_store_size_from_descr(op: &Op) -> usize {
    let descr = op.getdescr();
    if let Some(fd) = descr.as_ref().and_then(|d| d.as_field_descr()) {
        if fd.is_pointer_field() {
            return std::mem::size_of::<usize>();
        }
        return fd.field_size();
    }
    std::mem::size_of::<usize>()
}

/// `(item_size, is_signed)` from an op's ArrayDescr; defaults to 8-byte
/// signed when the descr is absent.
fn array_item_size_sign_from_descr(op: &Op) -> (usize, bool) {
    op.with_array_descr(|ad| (ad.item_size(), ad.is_item_signed()))
        .unwrap_or((8, true))
}

/// Dense census of every non-constant Ref-typed value (input arg / op result),
/// independent of whether it needs a home slot. Write-barrier selection still
/// needs the full Ref type set after homes are shrunk to only values live across
/// collecting calls.
struct RefValues {
    /// `Vec<bool>` is a wasteful container in general, but justified here: the
    /// set is built and dropped within one `build_wasm_module` call, sized to
    /// the trace's value count (tens to low hundreds), and only ever
    /// point-queried. At that size a direct byte index beats a bitset's
    /// shift/mask, and the workspace pulls in no bitset crate; it matches the
    /// backend's other id-indexed flag vectors (`label_resume_safety`,
    /// `failguard`).
    by_id: Vec<bool>,
}

impl RefValues {
    fn mark(by_id: &mut Vec<bool>, id: u32) {
        let i = id as usize;
        if i >= by_id.len() {
            by_id.resize(i + 1, false);
        }
        by_id[i] = true;
    }

    fn collect(inputargs: &[InputArg], ops: &[Op]) -> Self {
        let mut by_id = Vec::new();
        for ia in inputargs {
            if ia.tp == Type::Ref {
                Self::mark(&mut by_id, ia.index);
            }
        }
        for op in ops {
            let r = op.pos.get();
            if r != OpRef::NONE && !r.is_constant() && op.result_type() == Type::Ref {
                Self::mark(&mut by_id, r.raw());
            }
        }
        Self { by_id }
    }

    fn contains(&self, v: OpRef) -> bool {
        v != OpRef::NONE
            && !v.is_constant()
            && self.by_id.get(v.raw() as usize).copied().unwrap_or(false)
    }
}

/// Maps each homed Ref-typed value (input arg / op result) to a compact
/// home-slot index `0..len`, where its current `GcRef` is mirrored into the
/// frame's GC-root region (`HOME_SLOT_BASE + home * 8`) so a collecting
/// allocation inside the trace can forward it.
///
/// Keyed by value id (`OpRef::raw()` / input `index`), which is the dense
/// `[0, num_vars)` space the wasm value locals already use (`1 + raw`); a flat
/// vector indexed by that id is the natural fit — no hashing, and iteration is
/// in id order, so the emitted module stays deterministic without sorting. The
/// `is_constant` guard lives in one place (`home`): a constant `raw()` is a
/// distinct namespace that must never alias a value's home.
struct RefHomes {
    /// `by_id[raw] = home index`, or `NONE` where the value is not a Ref home.
    /// Sized to the last Ref id; queries for higher ids miss via `get`.
    by_id: Vec<u32>,
    len: usize,
}

impl RefHomes {
    const NONE: u32 = u32::MAX;

    fn assign(by_id: &mut Vec<u32>, next: &mut u32, id: u32) {
        let i = id as usize;
        if i >= by_id.len() {
            by_id.resize(i + 1, Self::NONE);
        }
        if by_id[i] == Self::NONE {
            by_id[i] = *next;
            *next += 1;
        }
    }

    fn collect(inputargs: &[InputArg], ops: &[Op], include_ca_collects: bool) -> Self {
        let liveness = HomeLiveness::collect(inputargs, ops);
        let collect_positions = collecting_call_positions(ops, include_ca_collects);
        let ref_values = RefValues::collect(inputargs, ops);
        let mut by_id = Vec::new();
        let mut next = 0u32;
        for ia in inputargs {
            if ia.tp == Type::Ref && liveness.live_across_any(ia.index, &collect_positions) {
                Self::assign(&mut by_id, &mut next, ia.index);
            }
        }
        for op in ops {
            let r = op.pos.get();
            if r != OpRef::NONE
                && !r.is_constant()
                && op.result_type() == Type::Ref
                && liveness.live_across_any(r.raw(), &collect_positions)
            {
                Self::assign(&mut by_id, &mut next, r.raw());
            }
        }
        if include_ca_collects {
            // The CA arm allocates its callee frame before it resolves this
            // CallAssemblerR's arguments. Those Ref operands are used at (not
            // after) this op, so ordinary `live_across` deliberately excludes
            // them; they nevertheless need homes through the prior allocation.
            for op in ops.iter().filter(|op| op.opcode == OpCode::CallAssemblerR) {
                for arg in op.getarglist() {
                    let arg = arg.to_opref();
                    if ref_values.contains(arg) {
                        Self::assign(&mut by_id, &mut next, arg.raw());
                    }
                }
            }
        }
        RefHomes {
            by_id,
            len: next as usize,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Home index of value id `id` (caller guarantees it is a value, not a
    /// constant — e.g. an input-arg index).
    fn home_id(&self, id: u32) -> Option<u32> {
        match self.by_id.get(id as usize) {
            Some(&h) if h != Self::NONE => Some(h),
            _ => None,
        }
    }

    /// Home index of `v`, or `None` if it is a constant or not a Ref home.
    fn home(&self, v: OpRef) -> Option<u32> {
        if v.is_constant() {
            return None;
        }
        self.home_id(v.raw())
    }

    /// `(value id, home index)` pairs in id order (deterministic).
    fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.by_id
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, h)| h != Self::NONE)
            .map(|(id, h)| (id as u32, h))
    }
}

/// Number of Ref-home slots a trace with these `inputargs`/`ops` reserves,
/// matching the `num_ref_homes` [`build_wasm_module`] returns. Lets a CA-arena
/// caller size the callee frame and the GC walker for a (wider) bridge's home
/// region before codegen runs.
pub fn count_ref_homes(inputargs: &[InputArg], ops: &[Op]) -> usize {
    // This pre-sizing query is used for CA bridges before `CaParams` exists, so
    // count `CallAssemblerR` as a collecting position to match CA codegen.
    RefHomes::collect(inputargs, ops, true).len()
}

/// Positional frame slots required for a token's inputs and guard spills.
/// Slot zero is the fail index; the returned count therefore also gives the
/// first free slot for the call trampoline.
pub fn frame_value_slots(inputargs: &[InputArg], ops: &[Op]) -> usize {
    let (guards, _) = collect_guards_and_vars(inputargs, ops);
    let max_fail_args = guards
        .iter()
        .map(|g| g.fail_arg_refs.len())
        .max()
        .unwrap_or(0);
    1 + max_fail_args.max(inputargs.len())
}

/// Argument index of the stored value for a GC ref-storing op. `SetfieldRaw` /
/// `SetarrayitemRaw` store into non-GC memory and never need a write barrier,
/// so only the `*Gc` variants are listed (rewrite.py only routes `SETFIELD_GC`
/// / `SETARRAYITEM_GC` / `SETINTERIORFIELD_GC` through the barrier).
fn ref_store_value_arg(op: &Op) -> Option<usize> {
    match op.opcode {
        OpCode::SetfieldGc => Some(1),
        OpCode::SetarrayitemGc | OpCode::SetinteriorfieldGc => Some(2),
        _ => None,
    }
}

/// If `op` stores a (non-constant) reference into a GC object, return the base
/// object operand that must be passed through the write barrier; otherwise
/// `None`. A value is a reference exactly when it has a Ref home slot
/// (`ref_homes` keys every Ref-typed input/result). This mirrors the native
/// `handle_write_barrier_setfield` gate `v.type == 'r' and not ConstPtr`: a
/// constant reference is an immortal/old object whose store never makes the base
/// point to young, so it needs no barrier (rewrite.py:930-931).
fn write_barrier_base(op: &Op, ref_values: &RefValues) -> Option<OpRef> {
    let val = op.arg(ref_store_value_arg(op)?).to_opref();
    // `contains` returns false for constants, matching the gate's `not ConstPtr`.
    ref_values.contains(val).then(|| op.arg(0).to_opref())
}

/// Whether any op in the trace needs a write-barrier trampoline call, which
/// requires the `jit_call` import to be present.
fn has_ref_store_op(ops: &[Op], ref_values: &RefValues) -> bool {
    ops.iter()
        .any(|op| write_barrier_base(op, ref_values).is_some())
}

/// Emit a write-barrier check on `base_ref` before a ref-storing field/array
/// store, standing in for the `COND_CALL_GC_WB` the native GC rewrite pass
/// inserts. When the residual type family is declared (`residual_type_base`),
/// the TRACK_YOUNG_PTRS flag is tested INLINE (assembler.py:2382
/// `genop_discard_cond_call_gc_wb` — test the header flag byte, jump over the
/// slow call when clear) and only a flagged old object takes the
/// `wasm_jit_write_barrier` `call_indirect`; a young or already-remembered
/// base skips the helper entirely. Otherwise the unconditional helper routes
/// through the `jit_call` host trampoline (the helper re-checks the flag).
/// Operand-stack-neutral: every push is consumed by a store, the call, or
/// the result drop.
fn emit_write_barrier(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    jit_call_idx: Option<u32>,
    residual_type_base: Option<u32>,
    wb_fn_ptr: i64,
    base_ref: OpRef,
    frame: FrameGeometry,
) {
    if let Some(base) = residual_type_base {
        // Header word is a u64 at `obj - GcHeader::SIZE` with the flags in
        // its upper half (`FLAG_SHIFT == 32`), so on little-endian wasm32 the
        // flags live in the i32 at `obj - 4`; TRACK_YOUNG_PTRS is flag bit 0.
        const FLAGS_HALF_BACKOFS: i32 = (majit_gc::header::GcHeader::SIZE / 2) as i32;
        const WB_FLAG: i32 = majit_gc::flags::TRACK_YOUNG_PTRS as i32;
        const _: () = assert!(majit_gc::header::FLAG_SHIFT == 32);
        const _: () = assert!(majit_gc::flags::TRACK_YOUNG_PTRS <= u32::MAX as u64);
        emit_resolve(sink, constants, base_ref);
        sink.i32_wrap_i64();
        sink.i32_const(FLAGS_HALF_BACKOFS);
        sink.i32_sub();
        sink.i32_load(MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        });
        sink.i32_const(WB_FLAG);
        sink.i32_and();
        sink.if_(BlockType::Empty);
        emit_resolve(sink, constants, base_ref);
        sink.i32_const(wb_fn_ptr as i32);
        sink.call_indirect(0, base + 1);
        sink.drop(); // returns 0; ignored
        sink.end();
        return;
    }
    let Some(jit_call) = jit_call_idx else {
        return;
    };
    // func_ptr = wasm_jit_write_barrier
    sink.local_get(0);
    sink.i64_const(wb_fn_ptr);
    sink.i64_store(mem64(frame.call_func_ofs));
    // num_args = 1 (the trampoline reflects arity from the wasm signature;
    // written for protocol symmetry with the alloc/call paths)
    sink.local_get(0);
    sink.i64_const(1);
    sink.i64_store(mem64(frame.call_nargs_ofs));
    // arg0 = base object pointer
    sink.local_get(0);
    emit_resolve(sink, constants, base_ref);
    sink.i64_store(mem64(frame.call_args_ofs));
    // call trampoline; void result ignored
    sink.local_get(0);
    emit_jit_call(sink, jit_call, frame);
}

/// Per-value def / last-use op positions over the trace, used to filter the
/// post-collection Ref reloads ([`emit_reload_refs_from_homes`]) down to
/// values that are both already defined and still read — the wasm-shaped
/// analog of the native regalloc reloading a spilled box on its next use
/// (llsupport/regalloc.py `longevity`) instead of eagerly rebinding every
/// home.
///
/// Positions: inputs are defined at `-1`; an op result at its op index; a
/// LABEL's args additionally at the label's index (a loop-carried value
/// re-enters the body there — a def index past a reload site must not hide
/// the stale local from the reload on the next iteration). Uses are op args
/// plus guard fail args; the loop-closing JUMP's args are op args, so
/// loop-carried values stay live through the backedge.
struct HomeLiveness {
    def_pos: Vec<i32>,
    last_use: Vec<i32>,
}

impl HomeLiveness {
    fn collect(inputargs: &[InputArg], ops: &[Op]) -> Self {
        let mut n = inputargs
            .iter()
            .map(|ia| ia.index as usize + 1)
            .max()
            .unwrap_or(0);
        for op in ops {
            let r = op.pos.get();
            if r != OpRef::NONE && !r.is_constant() {
                n = n.max(r.raw() as usize + 1);
            }
        }
        let mut def_pos = vec![i32::MAX; n];
        let mut last_use = vec![-1i32; n];
        for ia in inputargs {
            def_pos[ia.index as usize] = -1;
        }
        for (i, op) in ops.iter().enumerate() {
            let r = op.pos.get();
            if r != OpRef::NONE && !r.is_constant() && (r.raw() as usize) < n {
                let d = &mut def_pos[r.raw() as usize];
                *d = (*d).min(i as i32);
            }
            for a in op.getarglist().iter() {
                let a = a.to_opref();
                if a == OpRef::NONE || a.is_constant() || (a.raw() as usize) >= n {
                    continue;
                }
                last_use[a.raw() as usize] = i as i32;
                if op.opcode == OpCode::Label {
                    let d = &mut def_pos[a.raw() as usize];
                    *d = (*d).min(i as i32);
                }
            }
            if let Some(fa) = op.getfailargs() {
                for a in fa.iter() {
                    let a = a.to_opref();
                    if a != OpRef::NONE && !a.is_constant() && (a.raw() as usize) < n {
                        last_use[a.raw() as usize] = i as i32;
                    }
                }
            }
        }
        Self { def_pos, last_use }
    }

    /// Value `raw` is defined before op `at` and read after it — i.e. its
    /// local holds a value a collection at op `at` could invalidate.
    fn live_across(&self, raw: u32, at: usize) -> bool {
        let raw = raw as usize;
        raw < self.def_pos.len() && self.def_pos[raw] < at as i32 && self.last_use[raw] > at as i32
    }

    fn live_across_any(&self, raw: u32, positions: &[usize]) -> bool {
        positions.iter().any(|&at| self.live_across(raw, at))
    }
}

/// Static collecting-call positions whose gcmap-visible homes may be forwarded.
/// Alongside `New*`, conservatively include residual calls: an eligible direct
/// residual target may allocate or force, so it needs the same post-call home
/// reload as the allocation helpers.
fn collecting_call_positions(ops: &[Op], include_ca_collects: bool) -> Vec<usize> {
    ops.iter()
        .enumerate()
        .filter_map(|(i, op)| {
            (op.opcode.is_call()
                || matches!(
                    op.opcode,
                    OpCode::New | OpCode::NewWithVtable | OpCode::NewArray | OpCode::NewArrayClear
                ))
            .then_some(i)
            .or_else(|| (include_ca_collects && op.opcode == OpCode::CallAssemblerR).then_some(i))
        })
        .collect()
}

/// Reload the live Ref locals from their home slots after a collecting call
/// at op index `at_op`, optionally skipping one value id (`skip_raw` — the
/// freshly-allocated result, whose home is not yet written). The collection
/// forwarded the home slots (registered as GC roots), so reloading the
/// locals makes object movement transparent to the trace. Only values live
/// across `at_op` ([`HomeLiveness::live_across`]) are reloaded: a value not
/// yet defined has a null home and its local is written at its def, and a
/// value never read after `at_op` has no consumer for the reload — the
/// native regalloc likewise reloads a spilled box only on its next use.
fn emit_reload_refs_from_homes(
    sink: &mut InstructionSink<'_>,
    ref_homes: &RefHomes,
    liveness: &HomeLiveness,
    at_op: usize,
    skip_raw: Option<u32>,
    frame: FrameGeometry,
) {
    // `iter` yields id order, so the emitted module is reproducible without a
    // sort; each reload is independent (home and local storage are disjoint).
    for (raw, h) in ref_homes.iter() {
        if Some(raw) == skip_raw || !liveness.live_across(raw, at_op) {
            continue;
        }
        sink.local_get(0);
        sink.i64_load(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
        sink.local_set(1 + raw);
    }
}

/// RPython `_reload_frame_if_necessary` (x86 `assembler.py:1369`) for wasm
/// trace bodies: a collecting direct call may have forwarded the running
/// JitFrame, while wasm local 0 still holds its old ITEMS base.
fn emit_reload_frame_if_necessary(
    sink: &mut InstructionSink<'_>,
    residual_type_base: Option<u32>,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
) {
    if let Some(top_addr) = jf_top_addr {
        // assembler.py:1369-1377: reload the possibly-forwarded top JitFrame
        // directly from the shadow-stack cell. Unlike the helper-table call,
        // this does not need the residual direct-call type to be declared.
        emit_ca_reload_top(sink, top_addr);
        sink.local_set(0);
    } else if let Some(base) = residual_type_base {
        sink.i32_const(ca_reload_fn_ptr as i32);
        sink.call_indirect(0, base);
        sink.i32_wrap_i64();
        sink.local_set(0);
    } else {
        // The trampoline path still assumes a non-moving frame: its scratch writes use local 0.
    }
}

/// CA-arm-only variant of [`emit_reload_frame_if_necessary`]. The direct CA
/// configuration owns an inline shadow-stack top cell; all other call sites
/// retain their pre-existing helper reload.
fn emit_reload_ca_frame_if_necessary(
    sink: &mut InstructionSink<'_>,
    residual_type_base: Option<u32>,
    ca_reload_fn_ptr: i64,
    ca_inline: Option<CaInlineParams>,
) {
    if let Some(inline) = ca_inline {
        debug_assert!(residual_type_base.is_some());
        emit_ca_reload_top(sink, inline.jf_top_addr);
        sink.local_set(0);
    } else {
        emit_reload_frame_if_necessary(sink, residual_type_base, ca_reload_fn_ptr, None);
    }
}

/// assembler.py `_reload_frame_if_necessary`: `top[-WORD]` is the top
/// jitframe pointer. The wasm CA ABI carries its ITEMS base in local 0.
fn emit_ca_reload_top(sink: &mut InstructionSink<'_>, top_addr: u32) {
    sink.i32_const(top_addr as i32);
    sink.i32_load(mem32(0));
    sink.i32_const(4);
    sink.i32_sub();
    sink.i32_load(mem32(0));
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_add();
}

/// While a CA callee is pushed, its caller's `jf_ptr` is `top[-3 * WORD]`.
fn emit_ca_reload_caller(sink: &mut InstructionSink<'_>, top_addr: u32) {
    sink.i32_const(top_addr as i32);
    sink.i32_load(mem32(0));
    sink.i32_const(12);
    sink.i32_sub();
    sink.i32_load(mem32(0));
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_add();
}

/// Reload the Ref operands which the CA arm resolves only after its collecting
/// callee-frame allocation. Unlike ordinary post-call reloads, these are live
/// *at* the CALL_ASSEMBLER op, not after it.
fn emit_reload_ca_input_refs_from_homes(
    sink: &mut InstructionSink<'_>,
    ref_homes: &RefHomes,
    ref_values: &RefValues,
    op: &Op,
    frame: FrameGeometry,
) {
    for arg in op.getarglist() {
        let arg = arg.to_opref();
        if !ref_values.contains(arg) {
            continue;
        }
        let Some(home) = ref_homes.home(arg) else {
            continue;
        };
        sink.local_get(0);
        sink.i64_load(mem64(frame.home_slot_base + home as u64 * SLOT_SIZE));
        sink.local_set(1 + arg.raw());
    }
}

/// llsupport/gc.py:563 GcLLDescr_framework
///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
/// Looks up the materialized table populated by the runner from the
/// active gc_ll_descr. RPython resolves the same value via
/// `cpu.gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr`.
fn lookup_typeid_from_classptr(table: &HashMap<i64, u32>, classptr: usize) -> Option<u32> {
    table.get(&(classptr as i64)).copied()
}

/// Information about a guard exit collected during pre-scan.
pub struct GuardExit {
    pub fail_index: u32,
    pub fail_arg_refs: Vec<OpRef>,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// `op.descr` snapshot — passed through to `WasmFailDescr.meta_descr`
    /// so `get_latest_descr_arc` can return the canonical metainterp Arc
    /// (parity with dynasm/cranelift's `meta_descr` forwarding).
    pub meta_descr: Option<majit_ir::DescrRef>,
}

/// Pre-fetched GC-type-guard metadata for the wasm codegen.
///
/// RPython's `genop_guard_guard_*` methods call into
/// `self.cpu.gc_ll_descr` at codegen time to obtain the TYPE_INFO
/// table base, the `infobits` offset / byte mask, the subclassrange
/// field offset, and the `(subclassrange_min, subclassrange_max)`
/// bounds for the constant expected-class pointer. The wasm backend
/// has no direct handle on a `GcAllocator` at this layer, so the
/// caller (`WasmBackend::compile_loop`) pre-fetches each of those
/// values and bundles them here.
///
/// Parity references:
///  * `llsupport/gc.py:162` / `gc.py:318` — `supports_guard_gc_type`
///  * `llsupport/gc.py:592` — `get_translated_info_for_typeinfo`
///  * `llsupport/gc.py:619` — `get_translated_info_for_guard_is_object`
///  * `x86/assembler.py:1951` — `cpu.subclassrange_min_offset`
///  * `x86/assembler.py:1971-1974` — constant-time
///    `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
///
/// The default sets `supports_guard_gc_type = false`, matching
/// `AbstractCPU.supports_guard_gc_type` in `backend/model.py:21`; the
/// codegen arms assert this flag before reading any other field.
#[derive(Default)]
pub struct GuardGcTypeInfo {
    pub supports_guard_gc_type: bool,
    /// `get_translated_info_for_typeinfo()` = (base, shift, sizeof_ti).
    pub base_type_info: usize,
    pub shift_by: u8,
    pub sizeof_ti: usize,
    /// `get_translated_info_for_guard_is_object()`
    ///     = (infobits_offset, T_IS_RPYTHON_INSTANCE_BYTE).
    pub infobits_offset: usize,
    pub is_object_flag: u8,
    /// `cpu.subclassrange_min_offset` (x86/assembler.py:1951).
    pub subclassrange_min_offset: usize,
    /// `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
    /// looked up by constant classptr. Empty when
    /// `supports_guard_gc_type == false`.
    pub subclass_ranges: HashMap<i64, (i64, i64)>,
}

/// Check if any op in the trace is a CALL variant.
/// Lower an eligible residual CALL to a direct in-module `call_indirect` into
/// the callee's `__indirect_function_table` slot, instead of routing through the
/// `jit_call` host trampoline (guest→host→guest reflection + arg marshalling).
/// The residual-call ABI is uniformly `(i64×n) -> i64` for Int/Ref args+result
/// (verified: every fib residual call is `(i64,…)->i64`), so the static type is
/// fixed by the arity alone. `false` = byte-identical jit_call baseline.
const WASM_DIRECT_RESIDUAL_CALL: bool = true;

/// If `op` is a residual CALL whose ABI is uniformly i64 (all Int/Ref args and
/// an Int/Ref result), return its argument count — eligible for a direct
/// `call_indirect` of type `(i64×n) -> i64`. `None` keeps the `jit_call`
/// trampoline: void / float / release-GIL / cond / assembler calls, a missing
/// call descr, or an arg-count/descr-shape mismatch (defensive).
///
/// This includes `CallMayForce{I,R}` when their ABI is uniformly i64: the wasm
/// virtualizable is always materialized, so `GuardNotForced` is a no-op and a
/// direct call is sound. Float / release-GIL / cond / assembler calls and
/// non-reflectable descrs remain on the trampoline.
fn residual_call_i64_arity(op: &Op) -> Option<usize> {
    use OpCode::*;
    if !matches!(
        op.opcode,
        CallI
            | CallR
            | CallPureI
            | CallPureR
            | CallLoopinvariantI
            | CallLoopinvariantR
            | CallMayForceI
            | CallMayForceR
    ) {
        return None;
    }
    if !matches!(op.result_type(), Type::Int | Type::Ref) {
        return None;
    }
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    let arg_types = cd.arg_types();
    if arg_types
        .iter()
        .any(|t| !matches!(t, Type::Int | Type::Ref))
    {
        return None;
    }
    // `getarglist()[0]` is the func pointer; the call args are `[1..]`. The
    // descr's `arg_types` describes those call args, so the counts must match.
    let nargs = op.getarglist().len().saturating_sub(1);
    if arg_types.len() != nargs {
        return None;
    }
    Some(nargs)
}

/// Void-recorded counterpart of [`residual_call_i64_arity`]: an eligible
/// void residual CALL whose descr records the dummy-word C ABI
/// (`result_size == 8`, minted by `make_call_descr_void_word_abi`) — the
/// callee is really `(i64×n) -> i64` with the result ignored, so it lowers
/// through the same i64 type family with a trailing `drop`. A plain void
/// descr (`result_size == 0`) may target a genuinely `()`-returning callee
/// OR a word-returning one (the reflective host trampoline absorbs the
/// difference), so it stays on `jit_call`. This includes `CallMayForceN`
/// with the word ABI: the wasm virtualizable is always materialized, so
/// `GuardNotForced` is a no-op and a direct call is sound. Float / release-GIL
/// / cond / assembler calls and non-reflectable descrs remain on the trampoline.
fn residual_call_void_word_arity(op: &Op) -> Option<usize> {
    use OpCode::*;
    if !matches!(
        op.opcode,
        CallN | CallPureN | CallLoopinvariantN | CallMayForceN
    ) {
        return None;
    }
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    if cd.result_type() != Type::Void || cd.result_size() != 8 {
        return None;
    }
    let arg_types = cd.arg_types();
    if arg_types
        .iter()
        .any(|t| !matches!(t, Type::Int | Type::Ref))
    {
        return None;
    }
    let nargs = op.getarglist().len().saturating_sub(1);
    if arg_types.len() != nargs {
        return None;
    }
    Some(nargs)
}

/// Arity of `op`'s in-module `(i64×n) -> i64` lowering, if it has one: an
/// eligible residual CALL (word-result or word-ABI void), a `New*`
/// allocation (the `wasm_jit_alloc*` helper targets are plain
/// `extern "C" fn(i64×n) -> i64` table entries), or a ref-storing store
/// (its `wasm_jit_write_barrier` helper takes 1 arg). All of these share
/// the residual-call type family, so one max covers them.
fn direct_helper_i64_arity(op: &Op, ref_values: &RefValues) -> Option<usize> {
    if let Some(n) = residual_call_i64_arity(op) {
        return Some(n);
    }
    if let Some(n) = residual_call_void_word_arity(op) {
        return Some(n);
    }
    match op.opcode {
        // wasm_jit_alloc(type_id, size)
        OpCode::New | OpCode::NewWithVtable => Some(2),
        // wasm_jit_alloc_array(type_id, base_size, item_size, length, len_offset)
        OpCode::NewArray | OpCode::NewArrayClear => Some(5),
        // wasm_jit_write_barrier(base)
        _ => write_barrier_base(op, ref_values).map(|_| 1),
    }
}

fn has_call_ops(ops: &[Op]) -> bool {
    // Allocation ops (`New*`, `Newstr`/`Newunicode`) also reach the host via
    // the `jit_call` trampoline, so the import must be present for them too.
    ops.iter().any(|op| {
        op.opcode.is_call()
            || matches!(
                op.opcode,
                OpCode::New
                    | OpCode::NewWithVtable
                    | OpCode::NewArray
                    | OpCode::NewArrayClear
                    | OpCode::Newstr
                    | OpCode::Newunicode
            )
    })
}

/// Whether this trace emits a host `jit_call` / `jit_call_compact` trampoline
/// invocation.  CA frames are movable nursery objects, while the host
/// trampoline writes its result back through the pre-call frame pointer, so
/// `compile_bridge` uses this exact lowering census to keep such traces off a
/// live CA frame.
///
/// Keep this in lockstep with the individual emission arms below: the uniform
/// i64 residual family, `New*`, and write barriers are direct under
/// `WASM_DIRECT_RESIDUAL_CALL`; non-uniform CALLs and string allocation retain
/// the trampoline.  When the direct family is disabled, all of the existing
/// call-area users return to the trampoline baseline.
pub fn has_trampoline_calls(inputargs: &[InputArg], ops: &[Op], emit_ca: bool) -> bool {
    let ref_values = RefValues::collect(inputargs, ops);
    if !WASM_DIRECT_RESIDUAL_CALL {
        return has_call_ops(ops) || has_ref_store_op(ops, &ref_values);
    }

    ops.iter().any(|op| match op.opcode {
        // `build_function` handles an enabled CA `CallAssemblerR` before the
        // generic CALL arm, lowering it directly to the source-loop table
        // slot. It therefore never uses the host call area.
        OpCode::CallAssemblerR if emit_ca => false,
        // These arms have no direct helper lowering.
        OpCode::Newstr | OpCode::Newunicode => true,
        // Every residual CALL uses the trampoline unless its exact lowering
        // predicate supplies an i64 helper ABI.
        _ if op.opcode.is_call() => direct_helper_i64_arity(op, &ref_values).is_none(),
        // `New*` and ref-store write barriers are covered by
        // `direct_helper_i64_arity`, so their direct-family arms do not touch
        // the frame call area.
        _ => false,
    })
}

fn collect_guards_and_vars(inputargs: &[InputArg], ops: &[Op]) -> (Vec<GuardExit>, u32) {
    let mut guards = Vec::new();
    let mut max_var: u32 = 0;

    for ia in inputargs {
        if ia.index + 1 > max_var {
            max_var = ia.index + 1;
        }
    }

    let mut fail_index = 0u32;
    for op in ops {
        if op.pos.get() != OpRef::NONE
            && !op.pos.get().is_constant()
            && op.pos.get().raw() + 1 > max_var
        {
            max_var = op.pos.get().raw() + 1;
        }
        if op.opcode == OpCode::Label {
            for a in op.getarglist().iter() {
                let a = a.to_opref();
                if a != OpRef::NONE && !a.is_constant() && a.raw() + 1 > max_var {
                    max_var = a.raw() + 1;
                }
            }
        }

        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            let fail_args: Vec<OpRef> = op
                .getfailargs()
                .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
                .unwrap_or_else(|| op.getarglist().iter().map(|a| a.to_opref()).collect());
            let fail_arg_types = op
                .get_fail_arg_types()
                .unwrap_or_else(|| fail_args.iter().map(|_| Type::Int).collect());

            guards.push(GuardExit {
                fail_index,
                fail_arg_refs: fail_args,
                fail_arg_types,
                is_finish: op.opcode == OpCode::Finish,
                meta_descr: op.getdescr(),
            });
            fail_index += 1;
        }
    }

    (guards, max_var)
}

/// Assign each Ref-typed value (input arg or op result) a dense home-slot
/// index, keyed by its value id (`raw()`), the same id its wasm local uses
/// (`1 + raw`). Input args and op results share one value-id space (see
/// `collect_guards_and_vars`), so a single map covers both. Int / Float /
/// Void values are skipped — only GC references need a forwarding home.
/// Allocate the per-guard bridge-slot cell array for inter-trace chaining and
/// return `(base address in the shared linear memory, owner)`.
///
/// One zero-initialised i32 cell per guard, indexed by `fail_index`;
/// `compile_bridge` writes the bridge's table slot into the matching cell. The
/// returned `Box<[u32]>` is the array's owner — the caller stores it on the
/// compiled loop (or, for a bridge, on its source loop's owned-cells list) so
/// it is freed on `Drop`. The base address aliases the box's heap buffer, which
/// is stable across moves of the owning box, so baking it into the module here
/// stays valid for the loop's lifetime.
///
/// On native the trace is never executed, so the dispatch is omitted and no
/// cells are needed — returning `(0, None)` keeps the emitted module
/// byte-identical to the pre-chaining output and allocates nothing.
fn alloc_bridge_cells(num_guards: usize) -> (u32, Option<Box<[u32]>>) {
    #[cfg(target_arch = "wasm32")]
    {
        let mut cells = vec![0u32; num_guards].into_boxed_slice();
        let base = cells.as_mut_ptr() as usize as u32;
        (base, Some(cells))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = num_guards;
        (0, None)
    }
}

/// Parameters for the self-recursive CALL_ASSEMBLER guest→guest `call_indirect`
/// arm (`PYRE_WASM_CA`). `emit_ca == false` (the default) keeps every emitted
/// module byte-identical to the pre-feature backend.
#[derive(Clone, Copy, Default)]
pub struct CaParams {
    /// Emit the dedicated `CallAssemblerR` arm (an in-module `call_indirect`
    /// into the source loop). Only set by `compile_bridge` for a self-recursive
    /// single-int bridge; `compile_loop` never sets it.
    pub emit_ca: bool,
    /// Bytes to reserve per CA callee frame (the GC `JitFrame`'s data region,
    /// i.e. its Signed item area). This is the source geometry's prefix through
    /// the Ref homes, excluding its tail call area. The trampoline-decline
    /// floor in `WasmBackend::compile_bridge` (`source_ca_active &&
    /// bridge_has_trampoline_calls`) guarantees that no trampoline-lowered op
    /// runs on this movable frame, so the omitted tail is unreachable. The alloc
    /// trampoline derives the JitFrame item count from this exact byte count.
    pub callee_frame_bytes: u32,
    /// `fail_index` the SOURCE loop's DoneWithThisFrame Finish writes to frame[0]
    /// on the base-case return. The CA arm treats this — or this bridge's own
    /// finish index — as a clean callee finish; anything else is a deopt.
    pub loop_finish_fi: u32,
    /// `__indirect_function_table` slot of `wasm_ca_resume_deopt`
    /// (`lib.rs::ca_deopt_helper_slot`). When a callee `call_indirect` returns a
    /// non-finish `fail_index` (a guard deopt), the CA arm `call_indirect`s this
    /// slot to blackhole-resume the callee on the host and read back its result,
    /// instead of trapping. `0` (unset) ⇒ no helper, so `compile_bridge` declines
    /// the CA lift before reaching codegen.
    pub deopt_helper_slot: u32,
    /// Address of the source loop's `CompiledWasmLoop`, baked as the first
    /// argument to the deopt helper so it can resolve the deopted callee frame's
    /// `fail_descrs`.
    pub source_compiled_ptr: u64,
    /// `__indirect_function_table` slot (`fn as usize`) of
    /// `lib.rs::wasm_jit_ca_alloc_frame`, which allocates each callee frame as
    /// a young nursery GC-managed `JitFrame` (push_jf-rooted, traced by its own
    /// per-frame gcmap). `call_indirect`ed in-module through the residual
    /// `(i64,i64)->i64` type when declared, else via the `jit_call` trampoline.
    pub ca_alloc_fn_ptr: i64,
    /// `__indirect_function_table` slot of `lib.rs::wasm_jit_ca_pop_frame`,
    /// called on CA-arm exit to pop the callee frame off the jitframe shadow
    /// stack (strict LIFO).
    pub ca_pop_fn_ptr: i64,
    /// `__indirect_function_table` slot of `lib.rs::wasm_jit_ca_reload_frame`,
    /// called after the recursive call to recover this level's possibly-moved
    /// nursery frame from the jitframe shadow stack.
    pub ca_reload_fn_ptr: i64,
    /// Address of the active jitframe shadow-stack top cell, baked for every
    /// trace body so post-collecting-call local-0 reloads can match
    /// assembler.py without a helper round trip. `None` keeps the existing
    /// helper/trampoline behavior when compilation has no active GC.
    pub jf_top_addr: Option<u32>,
    /// `__indirect_function_table` slot of
    /// `lib.rs::wasm_jit_ca_reload_caller_frame`, called while the callee is
    /// still pushed to recover this invocation's possibly-moved local-0 frame.
    pub ca_reload_caller_fn_ptr: i64,
    /// Leaked per-bridge `jf_gcmap` (`lib.rs::build_callee_gcmap`) marking the
    /// callee frame's CA input + home Ref slots; baked into each frame's
    /// `jf_gcmap` field at alloc time.
    pub callee_gcmap_ptr: i64,
    /// Active-GC state for the direct CA-only inline allocation/frame path.
    /// `None` retains the helpers (including under gc_stress).
    pub inline: Option<CaInlineParams>,
}

/// Direct CA fast-path values baked at bridge compilation time.
#[derive(Clone, Copy)]
pub struct CaInlineParams {
    pub nursery_free_addr: u32,
    pub nursery_top_addr: u32,
    pub jf_top_addr: u32,
    pub jf_limit_addr: u32,
    pub jitframe_tid: u32,
}

/// Inline nursery-bump fast-path parameters for `New`/`NewWithVtable`
/// (rewrite.py's malloc fast path over the gc.py:525-531
/// `get_nursery_free_addr`/`get_nursery_top_addr` surface, which the x86
/// backend lowers as `malloc_cond`: load free, bump, compare top, call the
/// slow path only on overflow). `None` keeps every allocation on the
/// `wasm_jit_alloc` helper call.
pub struct NurseryAllocParams {
    /// Linear-memory address of the GC's `nursery_free` bump pointer.
    pub free_addr: u32,
    /// Linear-memory address of the GC's `nursery_top` limit pointer.
    pub top_addr: u32,
    /// `max_nursery_object_size` — a total size above this allocates in
    /// old-gen, so the inline path only applies below it.
    pub large_threshold: usize,
    /// Type ids whose allocation is a plain bump + header write (no
    /// destructor / weakref side-list registration).
    pub plain_tids: std::collections::HashSet<u32>,
}

/// Build a wasm module from majit IR.
pub fn build_wasm_module(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    vtable_offset: Option<usize>,
    classptr_to_typeid: &HashMap<i64, u32>,
    guard_gc_type_info: &GuardGcTypeInfo,
    alloc_fn_ptr: i64,
    alloc_array_fn_ptr: i64,
    wb_fn_ptr: i64,
    // Inline nursery-bump fast path for eligible `New`/`NewWithVtable`
    // (see `NurseryAllocParams`); `None` keeps allocations on the helper.
    nursery: Option<&NurseryAllocParams>,
    fail_index_base: u32,
    // Table slot of the loop a JUMP-with-no-local-LABEL re-enters (a loop-closing
    // bridge). `0` for a loop trace (its JUMP is a local back-edge `br`) and for a
    // straight-line bridge (no JUMP). When set, the terminal external JUMP writes
    // the loop's next inputargs into the frame and `return_call_indirect`s the
    // loop's table slot — a wasm tail call, so the loop⇄bridge cycle runs at
    // constant stack depth instead of growing one frame per iteration.
    external_jump_slot: u32,
    // Resume-at-LABEL dispatch key for the terminal external JUMP (`target
    // label ordinal + 1`, or 0 for a non-peeled target); see `build_function`.
    external_jump_key: u32,
    // Frozen layout of the frame this module executes on. A chained bridge
    // receives its source token's layout; a loop receives its own compact
    // layout at first compilation.
    frame: FrameGeometry,
    // Self-recursive CALL_ASSEMBLER arm parameters (`PYRE_WASM_CA`); `emit_ca`
    // off keeps the module byte-identical.
    ca: CaParams,
) -> Result<(Vec<u8>, Vec<GuardExit>, usize, u32, Option<Box<[u32]>>), BackendError> {
    let (mut guards, num_vars) = collect_guards_and_vars(inputargs, ops);

    // Every trace's guard/finish exits draw their indices from ONE global
    // fail-index space (`failguard::FAIL_DESCR_REGISTRY`): a cross-trace chain
    // can exit through a sibling loop's guard, so `frame[0]` must be
    // resolvable without knowing which chained module wrote it.
    // `build_function` seeds its `guard_idx` counter with this base so each
    // exit writes `base + local`; mirror that here on the returned
    // `GuardExit.fail_index`.
    for g in &mut guards {
        g.fail_index += fail_index_base;
    }

    // Inter-trace chaining: a loop trace's guard exits dispatch to a compiled
    // bridge in-module via `call_indirect` through the shared
    // `__indirect_function_table` (see the epilogue in `build_function`)
    // instead of returning the guard index to the host and round-tripping
    // through the interpreter. Each guard owns one i32 cell in a contiguous
    // `[u32]` array (indexed by `fail_index`) holding its bridge's table slot,
    // `0` = no bridge yet. The array lives in the shared linear memory so the
    // trace reads it and `compile_bridge` (guest-side) writes it. On native
    // builds the trace is never executed, so `alloc_bridge_cells` returns 0 and
    // the dispatch is omitted entirely — the module stays byte-identical.
    // Label-less traces still want guard cells: the self-recursive
    // CALL_ASSEMBLER case chains a guard exit of a Label-less recursion LOOP
    // into its CA bridge, and a BRIDGE's own guards chain nested sub-bridges the
    // same way (a hot guard inside a chained bridge would otherwise round-trip
    // to the host forever). So any guarded trace wants dispatch cells.
    let want_dispatch = !guards.is_empty();
    let (cells_base, cells_owner) = if want_dispatch {
        alloc_bridge_cells(guards.len())
    } else {
        (0, None)
    };
    let bridge_dispatch = cells_base != 0;

    // Frame value slots (inputs at entry, fail-arg spills at guard exit) occupy
    // `[1, 1 + max(num inputs, max fail args))`. They precede the dispatch key,
    // Ref homes, and the always-present tail call area; a chained bridge must
    // fit the source token's frozen value-slot count before it can share that
    // frame.
    let max_fail_args = guards
        .iter()
        .map(|g| g.fail_arg_refs.len())
        .max()
        .unwrap_or(0);
    let max_value_slots = 1 + max_fail_args.max(inputargs.len());
    if max_value_slots > frame.value_slots {
        return Err(BackendError::Unsupported(format!(
            "wasm backend: {max_value_slots} frame value slots exceed frozen frame layout \
             ({})",
            frame.value_slots,
        )));
    }

    let ref_values = RefValues::collect(inputargs, ops);
    let ref_homes = RefHomes::collect(inputargs, ops, ca.emit_ca);
    let num_ref_homes = ref_homes.len();
    if num_ref_homes > frame.home_slots {
        return Err(BackendError::Unsupported(format!(
            "wasm backend: {num_ref_homes} ref homes exceed frozen frame layout ({})",
            frame.home_slots,
        )));
    }

    // Self-recursive CALL_ASSEMBLER arm (`PYRE_WASM_CA`): `bridge_finish_fi` is
    // THIS bridge's own DoneWithThisFrame index (the recursive return), which the
    // CA arm accepts as a clean callee finish alongside the source loop's
    // base-case finish. Widen the callee-frame reservation to also fit this
    // bridge's frame: the source loop's guard-exit chains into this bridge
    // reusing the same arena frame, so an undersized frame would let the bridge's
    // home-slot writes overflow into the next arena slot.
    let bridge_finish_fi = guards
        .iter()
        .find(|g| g.is_finish)
        .map(|g| g.fail_index)
        .unwrap_or(0);
    // CA frames execute the source loop and this bridge on the same frozen
    // geometry.  `compile_bridge` rejects a bridge that needs more slots, so
    // no global floor or speculative slack is needed here.

    // This exact lowering census controls the host-trampoline import. Direct
    // residual helpers, including the CA arm's inline fast path, use
    // `call_indirect` and need no import, although their frozen frame still
    // keeps the tail call area for future bridges.
    let needs_call = has_trampoline_calls(inputargs, ops, ca.emit_ca);
    // In-module residual calls (`WASM_DIRECT_RESIDUAL_CALL`): the largest
    // eligible `(i64×n)->i64` arity in this trace — residual CALLs (word
    // result or word-ABI void) plus the `New*` / write-barrier helper
    // targets, which share the same uniform-i64 ABI — or `None` if there
    // are none. Each distinct arity `0..=max` gets its own function type
    // (declared below) so those arms can `call_indirect` with a static type.
    let residual_max_arity = if WASM_DIRECT_RESIDUAL_CALL {
        let scanned = ops
            .iter()
            .filter_map(|op| direct_helper_i64_arity(op, &ref_values))
            .max();
        if ca.emit_ca {
            // The CA arm's frame helpers (`wasm_jit_ca_reload_frame()`,
            // `wasm_jit_ca_pop_frame(frame_base)`, and
            // `wasm_jit_ca_alloc_frame(frame_bytes, gcmap_ptr)`) lower through
            // this same `(i64×n)->i64` family; make sure arity 2 is declared,
            // which declares the full 0..=2 range including reload's arity 0.
            Some(scanned.map_or(2, |m| m.max(2)))
        } else if ca.ca_reload_fn_ptr != 0 {
            // Every trace body can reload its own frame after a collecting
            // direct call, even though only bridges emit the CA arm.
            Some(scanned.map_or(0, |m| m.max(0)))
        } else {
            scanned
        }
    } else {
        None
    };
    // The shared indirect-function table backs direct residual helpers as well
    // as host-trampoline dispatch, chained bridges, and CA recursion.
    let needs_table = needs_call || bridge_dispatch || residual_max_arity.is_some() || ca.emit_ca;
    // `ca.emit_ca` forces the direct helper family to include arities 0..=2,
    // so all CA frame-helper trampoline `else` arms below are baseline-only.
    debug_assert!(!ca.emit_ca || residual_max_arity.is_some());

    let mut module = Module::new();

    // Type section
    let mut types = TypeSection::new();
    // Type 0: trace function (param i32) -> (result i32)
    types.ty().function(vec![ValType::I32], vec![ValType::I32]);
    if needs_call {
        // Type 1: fixed `jit_call(frame)` or compact
        // `jit_call_compact(frame, call_area_ofs)` trampoline.
        let params = if frame.call_result_ofs == CALL_RESULT_OFS {
            vec![ValType::I32]
        } else {
            vec![ValType::I32, ValType::I32]
        };
        types.ty().function(params, vec![]);
    }
    // Residual-call types follow: `(i64×n) -> i64` for arity `n`, indexed by
    // `residual_type_base + n`. `residual_type_base` = the count of types above.
    let residual_type_base = 1 + needs_call as u32;
    if let Some(max) = residual_max_arity {
        for n in 0..=max {
            types
                .ty()
                .function(vec![ValType::I64; n], vec![ValType::I64]);
        }
    }
    // CA deopt-helper type `(i64 frame_ptr, i64 compiled_ptr) -> i64`. The CA arm
    // `call_indirect`s `wasm_ca_resume_deopt` through it when a self-recursive
    // callee leaves its trace through a guard (a deopt). Declared after the
    // residual-call type family so its index is independent of which residual
    // arities the bridge happens to use.
    let ca_helper_type_idx = residual_type_base + residual_max_arity.map_or(0, |m| m as u32 + 1);
    if ca.emit_ca {
        types
            .ty()
            .function(vec![ValType::I64, ValType::I64], vec![ValType::I64]);
    }
    module.section(&types);

    // Import section
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        },
    );
    if needs_call {
        // Import jit_call trampoline as function index 0
        imports.import(
            "env",
            if frame.call_result_ofs == CALL_RESULT_OFS {
                "jit_call"
            } else {
                "jit_call_compact"
            },
            EntityType::Function(1),
        );
    }
    if needs_table {
        // Import the host's shared indirect function table as table index 0.
        // `jit_call`'s residual dispatch and the epilogue bridge
        // `call_indirect` both index it; the host registers every compiled
        // trace (and bridge) into this table by slot. A table import does not
        // shift the function index space, so `trace_func_idx` still depends
        // only on whether `jit_call` (a function import) is present.
        imports.import(
            "env",
            "__indirect_function_table",
            EntityType::Table(TableType {
                element_type: RefType::FUNCREF,
                table64: false,
                minimum: 0,
                maximum: None,
                shared: false,
            }),
        );
    }
    module.section(&imports);

    // Function section
    let mut functions = FunctionSection::new();
    functions.function(0); // type 0
    module.section(&functions);

    // Export section: trace function index depends on whether we imported jit_call
    let trace_func_idx = if needs_call { 1 } else { 0 };
    let mut exports = ExportSection::new();
    exports.export("trace", ExportKind::Func, trace_func_idx);
    module.section(&exports);

    // Code section
    let mut codes = CodeSection::new();
    let jit_call_idx = if needs_call { Some(0u32) } else { None };
    let func = build_function(
        inputargs,
        ops,
        constants,
        num_vars,
        jit_call_idx,
        vtable_offset,
        classptr_to_typeid,
        guard_gc_type_info,
        alloc_fn_ptr,
        alloc_array_fn_ptr,
        wb_fn_ptr,
        nursery,
        &ref_values,
        &ref_homes,
        cells_base,
        bridge_dispatch,
        fail_index_base,
        external_jump_slot,
        external_jump_key,
        frame,
        residual_max_arity.map(|_| residual_type_base),
        ca,
        bridge_finish_fi,
        ca_helper_type_idx,
    )?;
    codes.function(&func);
    module.section(&codes);

    Ok((
        module.finish(),
        guards,
        num_ref_homes,
        cells_base,
        cells_owner,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_function(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    num_vars: u32,
    jit_call_idx: Option<u32>,
    vtable_offset: Option<usize>,
    classptr_to_typeid: &HashMap<i64, u32>,
    guard_gc_type_info: &GuardGcTypeInfo,
    alloc_fn_ptr: i64,
    alloc_array_fn_ptr: i64,
    wb_fn_ptr: i64,
    nursery: Option<&NurseryAllocParams>,
    ref_values: &RefValues,
    ref_homes: &RefHomes,
    cells_base: u32,
    bridge_dispatch: bool,
    fail_index_base: u32,
    external_jump_slot: u32,
    // Resume-at-LABEL dispatch key the terminal external JUMP writes before
    // tail-calling `external_jump_slot`: `target label ordinal + 1`, so the
    // target's entry `br_table` lands on that label's resume loader. `0` when
    // the target is not peeled (no dispatch reads the slot).
    external_jump_key: u32,
    frame: FrameGeometry,
    // Base wasm type index of the `(i64×n)->i64` residual-call types (type
    // `residual_type_base + n` for arity `n`), or `None` when the trace has no
    // eligible residual call / `New*` / write barrier, so those arms always
    // use the `jit_call` path.
    residual_type_base: Option<u32>,
    // Self-recursive CALL_ASSEMBLER arm (`PYRE_WASM_CA`). `ca.emit_ca` off keeps
    // the body byte-identical.
    ca: CaParams,
    // This bridge's own DoneWithThisFrame Finish index (the recursive return);
    // the CA arm accepts it or `ca.loop_finish_fi` as a clean callee finish.
    bridge_finish_fi: u32,
    // wasm type index of the CA deopt helper `(i64, i64) -> i64`, declared in the
    // module type section when `ca.emit_ca`. The CA arm uses it to `call_indirect`
    // `ca.deopt_helper_slot` for a deopted callee.
    ca_helper_type_idx: u32,
) -> Result<Function, BackendError> {
    // The CA arm requires residual types (the setup above forces arity >= 2
    // while `WASM_DIRECT_RESIDUAL_CALL` is enabled). Its `jit_call` fallback
    // branches are retained solely for the direct-family-disabled baseline.
    debug_assert!(!ca.emit_ca || residual_type_base.is_some());
    // Value locals occupy `1 ..= num_vars`; reserve `UMULHI_SCRATCH` extra i64
    // locals past them (`num_vars+1 ..= num_vars+UMULHI_SCRATCH`) as scratch for
    // the `UintMulHigh` 32-bit-split expansion (`emit_umulhi`). One i32 local
    // past those (`num_vars+UMULHI_SCRATCH+1`) holds the bridge table slot for
    // the epilogue `call_indirect` dispatch (unused when `!bridge_dispatch`).
    let bridge_slot_local = num_vars + UMULHI_SCRATCH + 1;
    // The self-recursive CALL_ASSEMBLER arm needs two more i32 scratch locals:
    // `ca_cfp_local` (the current callee frame pointer) and `ca_fi_local` (the
    // returned frame[0] fail index). Reserve them only under `emit_ca` so a
    // flag-off module keeps exactly one i32 local (byte-identical).
    let ca_cfp_local = num_vars + UMULHI_SCRATCH + 2;
    let ca_fi_local = num_vars + UMULHI_SCRATCH + 3;
    // Extra i32 scratches when the inline nursery-bump fast path is armed:
    // one holds the loaded `nursery_free` across the bump/commit sequence;
    // runtime varsize array allocation also needs one for the computed
    // total/new-free word.
    let base_i32_locals: u32 = if ca.emit_ca { 3 } else { 1 };
    let alloc_scratch_local = num_vars + UMULHI_SCRATCH + 1 + base_i32_locals;
    let alloc_size_local = alloc_scratch_local + 1;
    let mut func = Function::new(vec![
        (num_vars + UMULHI_SCRATCH, ValType::I64),
        (
            base_i32_locals
                + if nursery.is_some() || ca.inline.is_some() {
                    2
                } else {
                    0
                },
            ValType::I32,
        ),
    ]);
    let mut sink = func.instructions();

    // Null-init every Ref-home slot so a slot read before its value is defined
    // is null (forwarding-safe), not a stale word from the reused host frame.
    for h in 0..ref_homes.len() as u64 {
        sink.local_get(0);
        sink.i64_const(0);
        sink.i64_store(mem64(frame.home_slot_base + h * SLOT_SIZE));
    }

    // A peeled loop arrives as `[preamble..][LABEL][body..][JUMP]`: the
    // preamble runs once on entry, the LABEL is the loop-back target, and
    // JUMP branches back to it. Emit the `loop` at the LABEL (not the top)
    // so the preamble is not re-executed every iteration; wrap everything in
    // a `block` so guard exits `br` out to the function epilogue. Use the
    // LAST label (a peeled trace may carry an outer entry label plus the
    // inner loop header).
    let loop_label_idx = ops.iter().rposition(|op| op.opcode == OpCode::Label);
    let has_loop = loop_label_idx.is_some();

    // Def / last-use positions for the post-collection Ref reload filter.
    let liveness = HomeLiveness::collect(inputargs, ops);

    // A Label-less trace with bridge dispatch — a `PYRE_WASM_CA` recursion
    // loop, or (chaining on) a bridge whose own guards chain nested
    // sub-bridges: there is no `loop`, but its guard/Finish exits still need
    // to `br` to the function epilogue so the epilogue's cell dispatch can
    // chain a failing guard in-module (instead of each guard early-returning
    // to the host). Wrap the body in one exit `block` and route exits through
    // it, exactly as a loop does. A loop-closing bridge's terminal external
    // JUMP is unaffected — `return_call_indirect` leaves the function from
    // inside the block.
    let straightline_dispatch = !has_loop && bridge_dispatch;

    // Resume-at-LABEL: a peeled loop wraps its preamble in a dispatch so a
    // loop-closing bridge can re-enter AT any LABEL — key = label ordinal + 1
    // — skipping the code before it, in-module instead of round-tripping
    // through the host. Keyed on the peeled shape (single- OR multi-label);
    // every other trace (non-peeled loop, straight-line, bridge) keeps its
    // byte-identical layout. Each label gets a (past_loader, loader) block
    // pair; the entry `br_table` jumps to the keyed label's resume loader,
    // and the fall-through path `br`s over each loader. Key 0 (and any
    // out-of-range key) runs the function from its entry (the preamble).
    let key_dispatch = is_resumable_peeled(ops);
    let num_labels = ops.iter().filter(|op| op.opcode == OpCode::Label).count();
    let all_label_args: Vec<Vec<OpRef>> = if key_dispatch {
        ops.iter()
            .filter(|op| op.opcode == OpCode::Label)
            .map(|op| op.getarglist().iter().map(|a| a.to_opref()).collect())
            .collect()
    } else {
        Vec::new()
    };
    if key_dispatch {
        // block $exit (A) — guard/Finish exits br here -> epilogue.
        // Per label j (opened outermost = last label):
        //   block $past_loader_j (B_j) — the fall-through path br's over the
        //     label-j resume loader.
        //   block $loader_j (C_j) — the `br_table` lands here (its end) for
        //     key j+1: the label-j resume loader.
        // block $dispatch (D) — key 0 br's here: run from the entry.
        sink.block(BlockType::Empty); // A $exit
        for _ in 0..num_labels {
            sink.block(BlockType::Empty); // B_j (j descending)
            sink.block(BlockType::Empty); // C_j
        }
        sink.block(BlockType::Empty); // D $dispatch
        sink.local_get(0);
        sink.i64_load(mem64(frame.dispatch_key_ofs));
        sink.i32_wrap_i64();
        // Depths at this point, innermost first: D=0, then (C_j, B_j) pairs
        // with C_j at 2j+1. Entry j+1 of the table targets C_j; entry 0 and
        // the default target D (the entry path).
        let br_targets: Vec<u32> = std::iter::once(0)
            .chain((0..num_labels as u32).map(|j| 2 * j + 1))
            .collect();
        sink.br_table(br_targets, 0);
        sink.end(); // end D $dispatch — key-0 entry path continues here
    }

    // Load inputs from frame into locals, and store Ref inputs to their homes.
    // The input value lives at the frame slot its producer wrote it to: the
    // caller fills slot `k` for the k-th input — `execute_token` for a loop
    // entry, `emit_guard_exit`'s positional fail-arg spill for a bridge entry —
    // so read from the POSITIONAL slot `k`, not `ia.index` (a value number that
    // equals `k` for a loop but not for a bridge, whose live-in args carry their
    // trace value numbers). The local index stays `1 + ia.index` because the
    // body addresses each value by its number. For `key_dispatch` this runs on
    // the key-0 (preamble) path only — past the `br_if` above — so a resuming
    // bridge never scatters its frame-passed label values into the function
    // inputargs' home slots; those stay null-initialized (GC-safe) and the
    // resume loader sets the live label-arg homes.
    for (k, ia) in inputargs.iter().enumerate() {
        let local_idx = 1 + ia.index;
        let offset = FRAME_SLOT_BASE + k as u64 * SLOT_SIZE;
        sink.local_get(0)
            .i64_load(mem64(offset))
            .local_set(local_idx);
        if let Some(h) = ref_homes.home_id(ia.index) {
            sink.local_get(0);
            sink.local_get(local_idx);
            sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
        }
    }

    // Non-key_dispatch loop: the single exit block A (preamble + body share it).
    // key_dispatch already opened A/B/C above. A Label-less dispatch trace (CA
    // loop, or a bridge chaining nested sub-bridges) also opens A so its
    // guard/Finish exits `br` out to the epilogue.
    if (has_loop || straightline_dispatch) && !key_dispatch {
        sink.block(BlockType::Empty);
    }

    // Seed with the fail-index base so each guard/finish exit writes
    // `base + local` into `frame[0]` (every trace passes the next free index
    // of the global fail-index space, `failguard::fail_descr_base`). The local
    // `guard_idx` counter and `collect_guards_and_vars`'s `fail_index` counter
    // increment in lockstep over the same ops, so the value written matches the
    // returned `GuardExit.fail_index` (also offset by the base).
    let mut guard_idx = fail_index_base;
    let mut in_loop_body = false;
    let mut labels_passed = 0usize;

    for (op_idx, op) in ops.iter().enumerate() {
        if op.opcode == OpCode::Label && key_dispatch {
            // End of the segment before label j (key-0 / earlier-label path).
            // Branch over the resume loader, then close C_j, emit the loader
            // (resume path only), and close B_j. From inside C_j, `br 1`
            // targets B_j's end, skipping the loader.
            sink.br(1); // segment done -> past_loader_j, over the resume loader
            sink.end(); // end C_j (the br_table lands here for key j+1)
            // Resume loader: a loop-closing bridge wrote each label arg into
            // frame slot i (positionally, matching the in-loop JUMP move);
            // load them into the label-arg locals and refresh their Ref
            // homes, mirroring the JUMP's ref-home refresh below. The
            // fall-through path skipped this via the `br 1` above.
            for (i, la) in all_label_args[labels_passed].iter().enumerate() {
                sink.local_get(0);
                sink.i64_load(mem64(FRAME_SLOT_BASE + i as u64 * SLOT_SIZE));
                sink.local_set(1 + la.raw());
                if let Some(h) = ref_homes.home(*la) {
                    sink.local_get(0);
                    sink.local_get(1 + la.raw());
                    sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
                }
            }
            sink.end(); // end B_j $past_loader
            labels_passed += 1;
        }
        if Some(op_idx) == loop_label_idx {
            sink.loop_(BlockType::Empty);
            in_loop_body = true;
        }
        // Depth (from statement level) of the enclosing `block` that guard
        // exits `br` to. Without `key_dispatch`: preamble = 0, loop body = 1
        // (the `loop` sits between the body and block A). With `key_dispatch`
        // a segment that still has `num_labels - labels_passed` labels ahead
        // sits inside that many (B_j, C_j) pairs, so it br's to depth
        // `2 * remaining`; the body is unchanged at 1 (every pair closes
        // before the loop). `None` for straight-line traces (no block
        // emitted).
        let block_exit_depth = match (has_loop, in_loop_body) {
            // Label-less dispatch trace: one exit block A (depth 0), no `loop`.
            (false, _) if straightline_dispatch => Some(0u32),
            (false, _) => None,
            (true, false) => Some(if key_dispatch {
                2 * (num_labels - labels_passed) as u32
            } else {
                0u32
            }),
            (true, true) => Some(1u32),
        };
        match op.opcode {
            OpCode::Label => {}

            OpCode::Jump if !has_loop => {
                // A JUMP in a trace with no local LABEL closes back into a
                // *separate* loop module (a loop-closing bridge). There is no
                // enclosing `loop` to `br` to, so re-enter the loop the way
                // `execute_token` does: write the jump args — the loop's next
                // inputargs, in inputarg order — into the loop's frame input
                // slots, then `return_call_indirect` the loop's table slot. The
                // tail call reuses this frame instead of nesting, so the
                // loop⇄bridge cycle holds at constant stack depth.
                //
                // The jump args are this bridge's SSA locals (or constants); the
                // input slots are a disjoint frame region from any Ref home slot
                // a resolve might load, so storing each pair in turn cannot feed
                // a clobbered read (unlike the local back-edge's parallel move
                // into shared loop locals).
                let jump_args = op.getarglist();
                for (i, jump_arg) in jump_args.iter().enumerate() {
                    sink.local_get(0); // frame_ptr
                    emit_resolve(&mut sink, constants, jump_arg.to_opref());
                    sink.i64_store(mem64(FRAME_SLOT_BASE + i as u64 * SLOT_SIZE));
                }
                // Set the resume-at-LABEL dispatch key so a peeled target
                // re-enters at the JUMP's target LABEL — skipping the code
                // before it — instead of re-running the function from its
                // entry. `compile_bridge` resolves the target label ordinal
                // from the JUMP descr and passes `ordinal + 1` here; the
                // target's entry `br_table` lands on that label's resume
                // loader. Harmless for a non-peeled target, which has no
                // dispatch and ignores the slot (`external_jump_key` 0).
                sink.local_get(0); // frame_ptr
                sink.i64_const(external_jump_key as i64); // dispatch key
                sink.i64_store(mem64(frame.dispatch_key_ofs));
                sink.local_get(0); // frame_ptr argument to the loop
                sink.i32_const(external_jump_slot as i32); // table slot
                sink.return_call_indirect(0, 0); // table 0, type 0: (i32) -> i32
            }

            OpCode::Jump => {
                // The jump rebinds the loop's label args to the jump args — a
                // parallel move. A jump arg may read a target local that another
                // pair overwrites (e.g. the swap `x, y = y, x` → x<-y, y<-x), so
                // resolving-then-storing each pair in turn would feed a clobbered
                // value to a later read. Do all reads first (push every resolved
                // jump arg onto the operand stack), then all writes (pop into the
                // targets in reverse, the stack being LIFO).
                let label_args = find_label_args(ops);
                let jump_args = op.getarglist();
                let n = jump_args.len().min(label_args.len());
                for jump_arg in jump_args.iter().take(n) {
                    emit_resolve(&mut sink, constants, jump_arg.to_opref());
                }
                for i in (0..n).rev() {
                    sink.local_set(1 + label_args[i].raw());
                }
                // The parallel move rebinds loop-carried locals without going
                // through store-on-def, so a Ref label arg that is REBOUND to a
                // new value has a stale home slot; refresh it before branching
                // back so the next iteration's reload-after-allocation sees the
                // current value. Skip identity self-moves (jump arg == label
                // arg): the value is loop-invariant, so the home written by the
                // entry/resume loader already holds it and re-storing it every
                // iteration is redundant.
                for i in 0..n {
                    let la = label_args[i];
                    if let Some(h) = ref_homes.home(la) {
                        // Skip the refresh for a loop-invariant self-move (the jump arg
                        // is the label arg itself, so the value flows back unchanged and
                        // the home written by the entry/resume loader is still current).
                        // A constant jump arg is never a self-move, and OpRef::raw() must
                        // not be called on an inline constant, so guard the comparison.
                        let jarg = jump_args[i].to_opref();
                        if !jarg.is_constant() && jarg.raw() == la.raw() {
                            continue;
                        }
                        sink.local_get(0);
                        sink.local_get(1 + la.raw());
                        sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
                    }
                }
                sink.br(0);
            }

            OpCode::Finish => {
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                if let Some(d) = block_exit_depth {
                    sink.br(d);
                }
                guard_idx += 1;
            }

            // ── Guards ──
            OpCode::GuardTrue => {
                emit_guard_true(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardFalse => {
                emit_guard_false(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardValue => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardNonnull => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_eqz();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardIsnull => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                // x86/assembler.py:1880-1891 _cmp_guard_class:
                //   offset = self.cpu.vtable_offset
                //   if offset is not None: CMP(mem(loc_ptr, offset), classptr)
                //   else:
                //       assert isinstance(loc_classptr, ImmedLoc)
                //       expected_typeid = gc_ll_descr.
                //           get_typeid_from_classptr_if_gcremovetypeptr(...)
                //       _cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
                if let Some(off_usize) = vtable_offset {
                    let off = off_usize as u64;
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64(); // struct ptr (i64) → i32 address
                    // The typeptr (`ob_type`) is a pointer-width field: 4
                    // bytes on wasm32. Reading it as i64 would fold in the
                    // following field's bytes and never match the class
                    // immediate. Load 4 bytes and zero-extend.
                    sink.i32_load(MemArg {
                        offset: off,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.i64_ne();
                } else {
                    // gcremovetypeptr fallback (assembler.py:1893-1901):
                    //   on x86_64 the typeid is a 32-bit value at offset 0.
                    let class_arg = op.arg(1).to_opref();
                    // history.py:227 — inline-Const carries its class pointer directly.
                    let classptr = class_arg.const_int_value().expect(
                        "_cmp_guard_class: gcremovetypeptr requires \
                             loc_classptr to be a ConstInt immediate \
                             (aarch64/regalloc.py:829 op.getarg(1).getint())",
                    );
                    let expected_typeid =
                        lookup_typeid_from_classptr(classptr_to_typeid, classptr as usize).expect(
                            "GuardClass: vtable_offset is None but the wasm \
                                 backend has no gc_ll_descr.\
                                 get_typeid_from_classptr_if_gcremovetypeptr",
                        );
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_const(expected_typeid as i32);
                    sink.i32_ne();
                }
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardNoOverflow => {
                // RPython: 0 args — overflow flag implicit from preceding ovf op.
                // Wasm MVP doesn't detect overflow, so always passes.
                guard_idx += 1;
            }
            OpCode::GuardOverflow => {
                // Always fails (no overflow detected in wasm MVP).
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                match block_exit_depth {
                    Some(d) => {
                        sink.br(d);
                    }
                    // Straight-line: return directly so the following ops and
                    // the terminal Finish do not overwrite this exit.
                    None => {
                        sink.local_get(0);
                        sink.return_();
                    }
                }
                guard_idx += 1;
            }
            // Guards that always pass in wasm MVP (no force-token /
            // invalidation tracking yet). GuardEvalBreaker is inert here too:
            // wasm32-unknown-unknown has no OS signals, so the ticker never
            // goes negative.
            OpCode::GuardNotInvalidated
            | OpCode::GuardNotForced
            | OpCode::GuardNotForced2
            | OpCode::GuardEvalBreaker => {
                guard_idx += 1;
            }
            OpCode::GuardNoException => {
                // x86/assembler.py:1799-1801 generate_guard_no_exception:
                // `CMP(pos_exception, imm0)` — fail the guard when a pending
                // exception is present, keyed on the exception TYPE slot
                // (pos_exception), the same slot GuardException reads and the
                // one llgraph's `last_exception is not None` tests. The slot
                // lives in the host's shared linear memory; load it by absolute
                // address (the trace imports env.memory).
                sink.i32_const(crate::jit_exc_type_addr() as i32);
                sink.i64_load(mem64(0));
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardException => {
                // x86/assembler.py:1808-1815 genop_guard_guard_exception:
                //   load pos_exception; CMP expected; guard on equal; then
                //   _store_and_reset_exception: resloc = pos_exc_value;
                //   pos_exception = 0; pos_exc_value = 0.
                let exc_type_addr = crate::jit_exc_type_addr() as i32;
                let exc_value_addr = crate::jit_exc_value_addr() as i32;
                sink.i32_const(exc_type_addr);
                sink.i64_load(mem64(0));
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
                // Success path: capture the caught exception into the result
                // var, then clear both slots.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i32_const(exc_value_addr);
                    sink.i64_load(mem64(0));
                    sink.local_set(1 + vi);
                }
                sink.i32_const(exc_type_addr);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
                sink.i32_const(exc_value_addr);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
            }

            // ── Integer arithmetic ──
            OpCode::IntAdd => emit_binop(&mut sink, constants, op, BinOp::I64Add),
            OpCode::IntSub => emit_binop(&mut sink, constants, op, BinOp::I64Sub),
            OpCode::IntMul => emit_binop(&mut sink, constants, op, BinOp::I64Mul),
            OpCode::IntFloorDiv => emit_binop(&mut sink, constants, op, BinOp::I64DivS),
            OpCode::IntMod => emit_binop(&mut sink, constants, op, BinOp::I64RemS),
            OpCode::IntAnd => emit_binop(&mut sink, constants, op, BinOp::I64And),
            OpCode::IntOr => emit_binop(&mut sink, constants, op, BinOp::I64Or),
            OpCode::IntXor => emit_binop(&mut sink, constants, op, BinOp::I64Xor),
            OpCode::IntLshift => emit_binop(&mut sink, constants, op, BinOp::I64Shl),
            OpCode::IntRshift => emit_binop(&mut sink, constants, op, BinOp::I64ShrS),
            OpCode::UintRshift => emit_binop(&mut sink, constants, op, BinOp::I64ShrU),
            // High 64 bits of the unsigned 64×64→128 product. The optimizer
            // emits this for division/modulo-by-constant strength reduction;
            // wasm has no mul-high instruction, so expand via 32-bit split.
            OpCode::UintMulHigh => emit_umulhi(&mut sink, constants, op, num_vars),

            // Overflow variants: compute result + overflow flag
            OpCode::IntAddOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Add),
            OpCode::IntSubOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Sub),
            OpCode::IntMulOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Mul),

            // ── Integer comparisons (signed) ──
            OpCode::IntLt => emit_cmp(&mut sink, constants, op, CmpOp::I64LtS),
            OpCode::IntLe => emit_cmp(&mut sink, constants, op, CmpOp::I64LeS),
            OpCode::IntEq => emit_cmp(&mut sink, constants, op, CmpOp::I64Eq),
            OpCode::IntNe => emit_cmp(&mut sink, constants, op, CmpOp::I64Ne),
            OpCode::IntGt => emit_cmp(&mut sink, constants, op, CmpOp::I64GtS),
            OpCode::IntGe => emit_cmp(&mut sink, constants, op, CmpOp::I64GeS),

            // ── Integer comparisons (unsigned) ──
            OpCode::UintLt => emit_cmp(&mut sink, constants, op, CmpOp::I64LtU),
            OpCode::UintLe => emit_cmp(&mut sink, constants, op, CmpOp::I64LeU),
            OpCode::UintGt => emit_cmp(&mut sink, constants, op, CmpOp::I64GtU),
            OpCode::UintGe => emit_cmp(&mut sink, constants, op, CmpOp::I64GeU),

            // ── Pointer comparisons ──
            OpCode::PtrEq | OpCode::InstancePtrEq => {
                emit_cmp(&mut sink, constants, op, CmpOp::I64Eq);
            }
            OpCode::PtrNe | OpCode::InstancePtrNe => {
                emit_cmp(&mut sink, constants, op, CmpOp::I64Ne);
            }

            // ── Unary ops ──
            OpCode::IntNeg => emit_unary_vi(
                &mut sink,
                constants,
                op,
                |s| {
                    s.i64_const(0);
                },
                |s| {
                    s.i64_sub();
                },
            ),
            OpCode::IntInvert => emit_unary_vi(
                &mut sink,
                constants,
                op,
                |s| {
                    s.i64_const(-1);
                },
                |s| {
                    s.i64_xor();
                },
            ),
            OpCode::IntIsTrue => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(0);
                    sink.i64_ne();
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::IntIsZero => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_eqz();
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }

            // ── Extended integer ops ──
            OpCode::IntSignext => {
                // int_signext(val, num_bytes): sign-extend from num_bytes width
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    // num_bytes is arg(1), typically a constant
                    let arg1 = op.arg(1).to_opref();
                    let num_bytes = if arg1.is_constant() {
                        arg1.inline_const_bits()
                            .or_else(|| constants.get(&arg1.raw()).copied())
                            .unwrap_or(8)
                    } else {
                        8 // default to no-op
                    };
                    let shift = 64 - num_bytes * 8;
                    if shift > 0 && shift < 64 {
                        sink.i64_const(shift);
                        sink.i64_shl();
                        sink.i64_const(shift);
                        sink.i64_shr_s();
                    }
                    sink.local_set(1 + vi);
                }
            }
            OpCode::IntForceGeZero => {
                // max(val, 0)
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    // if val < 0, use 0; else use val
                    // Wasm: local.tee + i64.const 0 + local.get + i64.lt_s + select
                    let tmp_local = 1 + vi; // reuse result local as temp
                    sink.local_tee(tmp_local);
                    sink.i64_const(0);
                    sink.local_get(tmp_local);
                    sink.i64_const(0);
                    sink.i64_lt_s();
                    sink.select();
                    sink.local_set(1 + vi);
                }
            }

            // ── Float comparisons ──
            OpCode::FloatLt => emit_float_cmp(&mut sink, constants, op, FloatCmp::Lt),
            OpCode::FloatLe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Le),
            OpCode::FloatEq => emit_float_cmp(&mut sink, constants, op, FloatCmp::Eq),
            OpCode::FloatNe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Ne),
            OpCode::FloatGt => emit_float_cmp(&mut sink, constants, op, FloatCmp::Gt),
            OpCode::FloatGe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Ge),

            // ── Float floor/mod ──
            OpCode::FloatFloorDiv => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.f64_div();
                    sink.f64_floor();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // ── Float/Int conversions ──
            OpCode::CastFloatToInt => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.i64_trunc_sat_f64_s();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::CastIntToFloat => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_convert_i64_s();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::ConvertFloatBytesToLonglong | OpCode::ConvertLonglongBytesToFloat => {
                // These are bitcast (no-op on the i64 representation)
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.local_set(1 + vi);
                }
            }

            // ── Pointer/Int conversions ──
            OpCode::CastPtrToInt | OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.local_set(1 + vi);
                }
            }

            // ── SameAs (forwarding) ──
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.local_set(1 + vi);
                }
            }

            // ── Field access (direct memory operations) ──
            OpCode::GetfieldGcI | OpCode::GetfieldGcPureI | OpCode::GetfieldRawI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // struct ptr (i64)
                    sink.i32_wrap_i64(); // convert to i32 address
                    let field_offset = field_offset_from_descr(op);
                    let (size, signed) = field_size_sign_from_descr(op);
                    emit_sized_int_load(&mut sink, field_offset, size, signed);
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetfieldGcR | OpCode::GetfieldGcPureR | OpCode::GetfieldRawR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    // Load as i32 (pointer on wasm32) and extend to i64
                    sink.i32_load(MemArg {
                        offset: field_offset,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                if let Some(base) = write_barrier_base(op, ref_values) {
                    emit_write_barrier(
                        &mut sink,
                        constants,
                        jit_call_idx,
                        residual_type_base,
                        wb_fn_ptr,
                        base,
                        frame,
                    );
                }
                emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // struct ptr
                sink.i32_wrap_i64();
                let field_offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(1).to_opref()); // value
                let size = setfield_store_size_from_descr(op);
                emit_sized_int_store(&mut sink, field_offset, size);
            }

            // ── Float field access ──
            OpCode::GetfieldGcF | OpCode::GetfieldGcPureF | OpCode::GetfieldRawF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    sink.f64_load(MemArg {
                        offset: field_offset,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // ── Array access ──
            OpCode::ArraylenGc => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // array ptr
                    sink.i32_wrap_i64();
                    let (len_offset, len_size) = array_len_layout_from_descr(op);
                    // The length is a word-sized field (`Signed`/`WORD`): read it
                    // at its real width, like `bh_arraylen_gc`. A fixed i64_load
                    // would fold the next field into the high half on wasm32.
                    emit_sized_int_load(&mut sink, len_offset, len_size, false);
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcPureI | OpCode::GetarrayitemRawI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // addr = base + base_size + index * item_size
                    emit_array_addr(&mut sink, constants, op);
                    let (item_size, signed) = array_item_size_sign_from_descr(op);
                    emit_sized_int_load(&mut sink, 0, item_size, signed);
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcPureR | OpCode::GetarrayitemRawR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_array_addr(&mut sink, constants, op);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetarrayitemGcF | OpCode::GetarrayitemGcPureF | OpCode::GetarrayitemRawF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // A Float item is 8 bytes; load it as f64 and carry its bit
                    // pattern in the i64 value slot (the IntArray/value-slot ABI
                    // is i64).
                    emit_array_addr(&mut sink, constants, op);
                    sink.f64_load(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                if let Some(base) = write_barrier_base(op, ref_values) {
                    emit_write_barrier(
                        &mut sink,
                        constants,
                        jit_call_idx,
                        residual_type_base,
                        wb_fn_ptr,
                        base,
                        frame,
                    );
                }
                emit_array_addr(&mut sink, constants, op);
                emit_resolve(&mut sink, constants, op.arg(2).to_opref()); // value
                // A Ref item is pointer-width (4 bytes on wasm32). Storing a
                // fixed 8 bytes would clobber the next item, or run past the
                // array end on the last item and corrupt the heap. A Float
                // item is 8 bytes, so its bit pattern stores via the i64 path.
                let (item_size, _signed) = array_item_size_sign_from_descr(op);
                emit_sized_int_store(&mut sink, 0, item_size);
            }

            // ── Interior field access ──
            OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // getinteriorfield(array, index, offset)
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // array ptr
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    // Simplified: use field_offset directly (RPython computes base+index*itemsize+offset)
                    let (size, signed) = field_size_sign_from_descr(op);
                    emit_sized_int_load(&mut sink, field_offset, size, signed);
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetinteriorfieldGcF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // array ptr
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    sink.f64_load(MemArg {
                        offset: field_offset,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetinteriorfieldGc => {
                if let Some(base) = write_barrier_base(op, ref_values) {
                    emit_write_barrier(
                        &mut sink,
                        constants,
                        jit_call_idx,
                        residual_type_base,
                        wb_fn_ptr,
                        base,
                        frame,
                    );
                }
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i32_wrap_i64();
                let field_offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(2).to_opref()); // value
                let (size, _signed) = field_size_sign_from_descr(op);
                emit_sized_int_store(&mut sink, field_offset, size);
            }

            // ── String/Unicode ops (direct memory access) ──
            OpCode::Strlen | OpCode::Unicodelen => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    // Length at offset 8 (after ob_type pointer on wasm32)
                    sink.i64_load(mem64(8));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::Strgetitem | OpCode::Unicodegetitem => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // str[index]: base + header_size + index
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref()); // index
                    sink.i32_wrap_i64();
                    sink.i32_add();
                    // String data starts after header (assume 16 bytes: ob_type + length)
                    sink.i32_load8_u(MemArg {
                        offset: 16,
                        align: 0,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }

            // ── GC memory ops ──
            OpCode::GcLoadI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let offset = field_offset_from_descr(op);
                    sink.i64_load(mem64(offset));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GcLoadR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let offset = field_offset_from_descr(op);
                    sink.i32_load(MemArg {
                        offset,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GcStore => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i32_wrap_i64();
                let offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                sink.i64_store(mem64(offset));
            }

            // ── Raw memory access ──
            OpCode::RawLoadI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // ptr
                    sink.i32_wrap_i64();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref()); // offset
                    sink.i32_wrap_i64();
                    sink.i32_add();
                    sink.i64_load(mem64(0));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::RawStore => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i32_wrap_i64();
                emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                sink.i32_wrap_i64();
                sink.i32_add();
                emit_resolve(&mut sink, constants, op.arg(2).to_opref());
                sink.i64_store(mem64(0));
            }

            // ── Exception handling ──
            OpCode::SaveException | OpCode::SaveExcClass | OpCode::RestoreException => {
                // No-op in wasm MVP — exception state is managed by the host.
            }

            // ── Conditional calls ──
            OpCode::CondCallN | OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                // No-op: the wasm backend does not consume the explicit
                // COND_CALL_GC_WB / COND_CALL_GC_WB_ARRAY barrier ops. It emits
                // the write barrier inline at each ref-store instead
                // (`write_barrier_base` + `emit_write_barrier`, calling the
                // `wasm_jit_write_barrier` host helper), so the standalone
                // barrier op is redundant here. CondCallN is a conditional void
                // call the wasm MVP does not need.
            }

            // x86/assembler.py:1919-1922 genop_guard_guard_gc_type:
            // GUARD_GC_TYPE: args[0] = object ref, args[1] = expected
            // type_id. The majit runtime stores the typeid in the GC
            // header word placed immediately before the object payload
            // (`majit_gc::header::GcHeader::tid_and_flags`, lower 32
            // bits). The cranelift backend lowers the same op this way
            // (compiler.rs GuardGcType branch). This is NOT the RPython
            // gcremovetypeptr layout — pyre's GC keeps the typeid in the
            // header, not at `obj[0]`.
            OpCode::GuardGcType => {
                let _ = classptr_to_typeid; // typeid is already an immediate
                if op.num_args() >= 2 {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    // header address = obj - GcHeader::SIZE
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_wrap_i64();
                    // Load 8-byte header word (tid_and_flags)
                    sink.i64_load(mem64(0));
                    // Mask lower TYPE_ID_BITS to extract the type id
                    sink.i64_const(TYPE_ID_MASK as i64);
                    sink.i64_and();
                    // Compare against expected_typeid (arg1 — already an
                    // i64 in the constant pool or a frame slot).
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.i64_ne();
                    emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                }
                guard_idx += 1;
            }
            // x86/assembler.py:1924-1943 genop_guard_guard_is_object.
            //     assert self.cpu.supports_guard_gc_type
            //     [loc_object, loc_typeid] = locs
            //     if IS_X86_32:
            //         self.mc.MOVZX16(loc_typeid, mem(loc_object, 0))
            //     else:
            //         self.mc.MOV32(loc_typeid, mem(loc_object, 0))
            //     base_type_info, shift_by, sizeof_ti = (
            //         self.cpu.gc_ll_descr
            //             .get_translated_info_for_typeinfo())
            //     infobits_offset, IS_OBJECT_FLAG = (
            //         self.cpu.gc_ll_descr
            //             .get_translated_info_for_guard_is_object())
            //     loc_infobits = addr_add(imm(base_type_info),
            //                             loc_typeid,
            //                             scale=shift_by,
            //                             offset=infobits_offset)
            //     self.mc.TEST8(loc_infobits, imm(IS_OBJECT_FLAG))
            //     self.guard_success_cc = rx86.Conditions['NZ']
            //     self.implement_guard(guard_token)
            OpCode::GuardIsObject => {
                // assembler.py:1925 assert self.cpu.supports_guard_gc_type
                assert!(
                    guard_gc_type_info.supports_guard_gc_type,
                    "x86/assembler.py:1925: assert self.cpu.\
                     supports_guard_gc_type (GcAllocator has not \
                     installed a TYPE_INFO layout)"
                );
                // assembler.py:1931-1932 MOV32 loc_typeid, mem(loc_object, 0).
                // majit's GC header sits at obj - GcHeader::SIZE; the
                // typeid occupies the lower TYPE_ID_BITS of that word.
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_const(GcHeader::SIZE as i64);
                sink.i64_sub();
                sink.i32_wrap_i64();
                sink.i64_load(mem64(0));
                sink.i64_const(TYPE_ID_MASK as i64);
                sink.i64_and();
                // Stack: [..., loc_typeid]

                // assembler.py:1938-1939 addr_add(imm(base_type_info),
                //     loc_typeid, scale=shift_by, offset=infobits_offset)
                if guard_gc_type_info.shift_by > 0 {
                    sink.i64_const(guard_gc_type_info.shift_by as i64);
                    sink.i64_shl();
                }
                sink.i64_const(guard_gc_type_info.base_type_info as i64);
                sink.i64_add();
                sink.i64_const(guard_gc_type_info.infobits_offset as i64);
                sink.i64_add();
                sink.i32_wrap_i64();
                // Stack: [..., loc_infobits(i32 addr)]

                // assembler.py:1940 TEST8 [loc_infobits], IS_OBJECT_FLAG
                sink.i32_load8_u(MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                });
                sink.i32_const(guard_gc_type_info.is_object_flag as i32);
                sink.i32_and();
                // assembler.py:1942 guard_success_cc = Conditions['NZ']:
                // guard passes when byte & flag != 0; fail when == 0.
                sink.i32_eqz();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            // x86/assembler.py:1945-1980 genop_guard_guard_subclass.
            //     assert self.cpu.supports_guard_gc_type
            //     [loc_object, loc_check_against_class, loc_tmp] = locs
            //     offset = self.cpu.vtable_offset
            //     offset2 = self.cpu.subclassrange_min_offset
            //     if offset is not None:
            //         self.mc.MOV_rm(loc_tmp, (loc_object, offset))
            //         self.mc.MOV_rm(loc_tmp, (loc_tmp, offset2))
            //     else:
            //         self.mc.MOV32(loc_tmp, mem(loc_object, 0))
            //         base_type_info, shift_by, sizeof_ti = (
            //             gc_ll_descr.get_translated_info_for_typeinfo())
            //         self.mc.MOV(loc_tmp, addr_add(
            //             imm(base_type_info), loc_tmp,
            //             scale=shift_by,
            //             offset=sizeof_ti + offset2))
            //     vtable_ptr = loc_check_against_class.getint()
            //     vtable_ptr = rffi.cast(rclass.CLASSTYPE, vtable_ptr)
            //     check_min = vtable_ptr.subclassrange_min
            //     check_max = vtable_ptr.subclassrange_max
            //     self.mc.SUB_ri(loc_tmp, check_min)
            //     self.mc.CMP_ri(loc_tmp, check_max - check_min)
            //     self.guard_success_cc = Conditions['B']
            //     self.implement_guard(guard_token)
            OpCode::GuardSubclass => {
                // assembler.py:1946 assert self.cpu.supports_guard_gc_type
                assert!(
                    guard_gc_type_info.supports_guard_gc_type,
                    "x86/assembler.py:1946: assert self.cpu.\
                     supports_guard_gc_type (GcAllocator has not \
                     installed a TYPE_INFO / rclass.CLASSTYPE layout)"
                );

                // assembler.py:1971 vtable_ptr = loc_check_against_class
                //   .getint(): the bounds are resolved at codegen time,
                //   so arg1 must be an immediate class pointer.
                let class_arg = op.arg(1).to_opref();
                // history.py:227 — inline-Const carries its class pointer directly.
                let loc_check_against_class = class_arg.const_int_value().unwrap_or_else(|| {
                    panic!(
                        "x86/assembler.py:1971 vtable_ptr = \
                             loc_check_against_class.getint(): \
                             GUARD_SUBCLASS requires arg1 to be a \
                             ConstInt immediate class pointer"
                    )
                });
                // assembler.py:1973-1974: vtable_ptr.subclassrange_{min,max}
                let (check_min, check_max) = guard_gc_type_info
                    .subclass_ranges
                    .get(&loc_check_against_class)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "x86/assembler.py:1973-1974 vtable_ptr.\
                             subclassrange_min/max: GcAllocator has no \
                             rclass.CLASSTYPE entry for classptr {:#x}",
                            loc_check_against_class
                        )
                    });

                // assembler.py:1950-1951 offset / offset2.
                let offset2 = guard_gc_type_info.subclassrange_min_offset;
                if let Some(vtable_off) = vtable_offset {
                    // assembler.py:1953-1956
                    //     MOV_rm(loc_tmp, (loc_object, offset))
                    //     MOV_rm(loc_tmp, (loc_tmp, offset2))
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(vtable_off as u64));
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(offset2 as u64));
                } else {
                    // assembler.py:1957-1969 gcremovetypeptr path.
                    //     MOV32 loc_tmp, mem(loc_object, 0)
                    //     base_type_info, shift_by, sizeof_ti = ...
                    //     MOV loc_tmp, [base_type_info
                    //         + (loc_tmp << shift_by)
                    //         + sizeof_ti + offset2]
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(0));
                    sink.i64_const(TYPE_ID_MASK as i64);
                    sink.i64_and();
                    if guard_gc_type_info.shift_by > 0 {
                        sink.i64_const(guard_gc_type_info.shift_by as i64);
                        sink.i64_shl();
                    }
                    sink.i64_const(guard_gc_type_info.base_type_info as i64);
                    sink.i64_add();
                    sink.i64_const((guard_gc_type_info.sizeof_ti + offset2) as i64);
                    sink.i64_add();
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(0));
                }
                // Stack: [..., loc_tmp (i64)]

                // assembler.py:1976-1978 unsigned comparison:
                //     (loc_tmp - check_min) <u (check_max - check_min)
                sink.i64_const(check_min);
                sink.i64_sub();
                sink.i64_const(check_max - check_min);
                // assembler.py:1979 guard_success_cc = Conditions['B']:
                // guard passes when sub <u limit; fail when sub >=u limit.
                sink.i64_ge_u();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardFutureCondition | OpCode::GuardAlwaysFails => {
                // GuardAlwaysFails always exits.
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                if let Some(d) = block_exit_depth {
                    sink.br(d);
                }
                guard_idx += 1;
            }

            // ── Quasi-immutable / record / assert ──
            OpCode::QuasiimmutField
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueI
            | OpCode::RecordExactValueR
            | OpCode::AssertNotNone => {
                // Metadata-only ops, no codegen needed.
            }

            // ── Allocation via trampoline ──
            OpCode::Newstr | OpCode::Newunicode => {
                // These may appear in traces that materialize strings.
                // Use CALL trampoline if available, otherwise skip.
                if let Some(jit_call) = jit_call_idx {
                    let vi = op.pos.get().raw();
                    sink.local_get(0);
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // length
                    sink.i64_store(mem64(frame.call_args_ofs));
                    sink.local_get(0);
                    sink.i64_const(0); // func_ptr = 0 signals "newstr" to host
                    sink.i64_store(mem64(frame.call_func_ofs));
                    sink.local_get(0);
                    sink.i64_const(1);
                    sink.i64_store(mem64(frame.call_nargs_ofs));
                    sink.local_get(0);
                    emit_jit_call(&mut sink, jit_call, frame);
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_get(0);
                        sink.i64_load(mem64(frame.call_result_ofs));
                        sink.local_set(1 + vi);
                    }
                }
            }

            // ── String content copy ──
            OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                // Bulk memory copy — use CALL trampoline or skip
            }

            // ── Misc ops ──
            OpCode::NurseryPtrIncrement => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.i64_add();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::CheckMemoryError => {
                // After allocation: check if result is null
                // No-op in wasm (allocations don't fail the same way)
            }
            OpCode::ZeroArray => {
                // Zero-initialize array region — skip for MVP
            }
            OpCode::LoadFromGcTable => {
                // `assembler.py:1545` `genop_load_from_gc_table`: this op is
                // produced only by the GC rewrite's `remove_constptr`
                // (`rewrite.py:1100`), whose arg is a `ConstInt(index)` into
                // a per-loop `GcTable` whose base is baked absolute. The
                // wasm backend does not run the GC rewrite and has no
                // host-address gc_table model (linear memory), so this op
                // never reaches here. Panic loudly rather than emit the old
                // SAME_AS pass-through, which after the rewrite flip would
                // load the raw index in place of the reference constant.
                panic!(
                    "wasm backend: LoadFromGcTable is unsupported (no gc_table model); \
                     the GC rewrite must not run for wasm"
                );
            }

            // ── Self-recursive CALL_ASSEMBLER (PYRE_WASM_CA) ──
            // Lower `vi = CallAssemblerR(frame, ec)` into an in-module
            // `call_indirect` into the SOURCE loop (self-recursion) instead of a
            // host round-trip. A fresh callee frame is allocated as a real
            // GC-managed nursery `JitFrame` (push_jf-rooted on the jitframe
            // shadow stack; traced by its OWN per-frame gcmap covering
            // its input + home Ref slots), the two red inputs are written to its
            // input slots, the loop runs on it (recursing through this same arm
            // for deeper levels), then the result Ref is read back from output
            // slot 0. Only emitted for the fib-shaped bridge `compile_bridge`
            // validated (`bridge_is_self_recursive_int_ca`); a loop never sets
            // `emit_ca`. The callee `call_indirect` runs the source loop's full
            // recursion, which allocates and collects; each live callee frame is
            // self-described by its gcmap so a collection forwards its Refs (no
            // shared-arena single-stride walker). This bridge's own wasm-local
            // Refs still hold pre-call (from-space) addresses on return, so
            // reload them from the (forwarded) homes after the call.
            OpCode::CallAssemblerR if ca.emit_ca => {
                let vi = op.pos.get().raw();
                // `external_jump_slot` is the source loop's table slot (the CA
                // self-target), plumbed by `compile_bridge`.
                let self_slot = external_jump_slot as i32;

                // Allocate the callee frame as a GC JitFrame
                // (`wasm_jit_ca_alloc_frame(frame_bytes, gcmap_ptr)` — a
                // collecting `(i64,i64)->i64` table entry whose caller's Refs
                // are rooted in frame homes, so it lowers like an eligible
                // residual call when the type family is declared; otherwise via
                // the jit_call trampoline).
                // `ca_cfp_local = frame_base + FIRST_ITEM_OFFSET` is the
                // bespoke-layout frame pointer — every `mem64(OFS)` below is
                // relative to it, exactly as the source loop reads its local 0.
                let ca_depth = ca.callee_frame_bytes as usize / std::mem::size_of::<isize>();
                let ca_payload_size = majit_backend::jitframe::JitFrame::alloc_size(ca_depth);
                let ca_total_size =
                    ((GcHeader::SIZE + ca_payload_size).max(GcHeader::MIN_NURSERY_OBJ_SIZE) + 7)
                        & !7;
                if let (Some(base), Some(inline)) = (residual_type_base, ca.inline) {
                    // rewrite.py's nursery fast path plus assembler.py's inline
                    // shadow-stack header. The `memory.fill` is deliberate:
                    // home slots are read as roots before every definition, and
                    // guard/deopt fail slots may be read by the host. Do not rely
                    // on nursery reset's pre-existing memset for either class.
                    sink.i32_const(inline.nursery_free_addr as i32);
                    sink.i32_load(mem32(0));
                    sink.local_tee(alloc_scratch_local);
                    sink.i32_const(ca_total_size as i32);
                    sink.i32_add();
                    sink.i32_const(inline.nursery_top_addr as i32);
                    sink.i32_load(mem32(0));
                    sink.i32_gt_u();
                    sink.i32_const(inline.jf_top_addr as i32);
                    sink.i32_load(mem32(0));
                    sink.local_tee(alloc_size_local);
                    sink.i32_const(8);
                    sink.i32_add();
                    sink.i32_const(inline.jf_limit_addr as i32);
                    sink.i32_load(mem32(0));
                    sink.i32_gt_u();
                    sink.i32_or();
                    sink.if_(BlockType::Result(ValType::I64));
                    // Slow path collects/allocates and performs init + push.
                    sink.i64_const(ca.callee_frame_bytes as i64);
                    sink.i64_const(ca.callee_gcmap_ptr);
                    sink.i32_const(ca.ca_alloc_fn_ptr as i32);
                    sink.call_indirect(0, base + 2);
                    sink.else_();
                    // Commit the nursery bump, then write the exact young
                    // `GcHeader::new(jitframe_tid)` word (flags are clear).
                    sink.i32_const(inline.nursery_free_addr as i32);
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(ca_total_size as i32);
                    sink.i32_add();
                    sink.i32_store(mem32(0));
                    sink.local_get(alloc_scratch_local);
                    sink.i64_const(inline.jitframe_tid as i64);
                    sink.i64_store(mem64(0));
                    // Explicitly initialise the complete JitFrame payload and
                    // item area, then replicate JitFrame::init + jf_gcmap.
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i32_const(0);
                    sink.i32_const(ca_payload_size as i32);
                    sink.memory_fill(0);
                    for offset in [
                        majit_backend::jitframe::JF_FRAME_INFO_OFS,
                        majit_backend::jitframe::JF_DESCR_OFS,
                        majit_backend::jitframe::JF_FORCE_DESCR_OFS,
                        majit_backend::jitframe::JF_SAVEDATA_OFS,
                        majit_backend::jitframe::JF_GUARD_EXC_OFS,
                        majit_backend::jitframe::JF_FORWARD_OFS,
                    ] {
                        sink.local_get(alloc_scratch_local);
                        sink.i32_const(0);
                        sink.i32_store(mem32(GcHeader::SIZE as u64 + offset as u64));
                    }
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(ca.callee_gcmap_ptr as i32);
                    sink.i32_store(mem32(
                        GcHeader::SIZE as u64 + majit_backend::jitframe::JF_GCMAP_OFS as u64,
                    ));
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(ca_depth as i32);
                    sink.i32_store(mem32(
                        GcHeader::SIZE as u64 + majit_backend::jitframe::JF_FRAME_OFS as u64,
                    ));
                    // Push `[is_minor=1, jf_ptr]`; the limit check above made
                    // these stores safe, so the helper's overflow assertion is
                    // retained only on the slow path.
                    sink.local_get(alloc_size_local);
                    sink.i32_const(1);
                    sink.i32_store(mem32(0));
                    sink.local_get(alloc_size_local);
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i32_store(mem32(4));
                    sink.i32_const(inline.jf_top_addr as i32);
                    sink.local_get(alloc_size_local);
                    sink.i32_const(8);
                    sink.i32_add();
                    sink.i32_store(mem32(0));
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                } else if let Some(base) = residual_type_base {
                    sink.i64_const(ca.callee_frame_bytes as i64);
                    sink.i64_const(ca.callee_gcmap_ptr);
                    sink.i32_const(ca.ca_alloc_fn_ptr as i32);
                    sink.call_indirect(0, base + 2);
                } else {
                    let jit_call =
                        jit_call_idx.expect("CA arm needs jit_call for the frame trampolines");
                    sink.local_get(0);
                    sink.i64_const(ca.ca_alloc_fn_ptr);
                    sink.i64_store(mem64(frame.call_func_ofs));
                    sink.local_get(0);
                    sink.i64_const(2);
                    sink.i64_store(mem64(frame.call_nargs_ofs));
                    sink.local_get(0);
                    sink.i64_const(ca.callee_frame_bytes as i64);
                    sink.i64_store(mem64(frame.call_args_ofs));
                    sink.local_get(0);
                    sink.i64_const(ca.callee_gcmap_ptr);
                    sink.i64_store(mem64(frame.call_args_ofs + SLOT_SIZE));
                    sink.local_get(0);
                    emit_jit_call(&mut sink, jit_call, frame);
                    sink.local_get(0);
                    sink.i64_load(mem64(frame.call_result_ofs));
                }
                sink.i32_wrap_i64();
                sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
                sink.i32_add();
                sink.local_set(ca_cfp_local);
                // The collecting callee allocation ran while this invocation's
                // own frame was the shadow-stack top. Now that the callee is
                // pushed, reload local 0 from the entry beneath it before
                // resolving inputs through local-0-relative homes. The
                // trampoline path intentionally keeps the A0-era assumption:
                // its scratch writes themselves dereference stale local 0.
                if let (Some(_base), Some(inline)) = (residual_type_base, ca.inline) {
                    emit_ca_reload_caller(&mut sink, inline.jf_top_addr);
                    sink.local_set(0);
                } else if let Some(base) = residual_type_base {
                    sink.i32_const(ca.ca_reload_caller_fn_ptr as i32);
                    sink.call_indirect(0, base);
                    sink.i32_wrap_i64();
                    sink.local_set(0);
                }
                emit_reload_ca_input_refs_from_homes(&mut sink, ref_homes, ref_values, op, frame);
                // dispatch key = 0: run the loop from its entry (preamble), not a
                // LABEL resume — this is a fresh call.
                sink.local_get(ca_cfp_local);
                sink.i64_const(0);
                sink.i64_store(mem64(frame.dispatch_key_ofs));
                // inputs: F'[1] = arg0 (callee frame), F'[2] = arg1 (ec).
                sink.local_get(ca_cfp_local);
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_store(mem64(FRAME_SLOT_BASE));
                sink.local_get(ca_cfp_local);
                emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                sink.i64_store(mem64(FRAME_SLOT_BASE + SLOT_SIZE));
                // run the source loop on F' (non-tail: one wasm frame per
                // recursion level); discard the returned frame_ptr.
                sink.local_get(ca_cfp_local);
                sink.i32_const(self_slot);
                sink.call_indirect(0, 0);
                sink.drop();
                // The recursive call may minor-collect and move this nursery
                // callee frame. Deeper levels have already popped, so the
                // jitframe shadow-stack top is this level's frame; reload its
                // ITEMS base before reading F'[0] or F'[1].
                if let (Some(_base), Some(inline)) = (residual_type_base, ca.inline) {
                    emit_ca_reload_top(&mut sink, inline.jf_top_addr);
                    sink.i64_extend_i32_u();
                } else if let Some(base) = residual_type_base {
                    sink.i32_const(ca.ca_reload_fn_ptr as i32);
                    sink.call_indirect(0, base + 0);
                } else {
                    let jit_call =
                        jit_call_idx.expect("CA arm needs jit_call for the frame trampolines");
                    sink.local_get(0);
                    sink.i64_const(ca.ca_reload_fn_ptr);
                    sink.i64_store(mem64(frame.call_func_ofs));
                    sink.local_get(0);
                    sink.i64_const(0);
                    sink.i64_store(mem64(frame.call_nargs_ofs));
                    sink.local_get(0);
                    emit_jit_call(&mut sink, jit_call, frame);
                    sink.local_get(0);
                    sink.i64_load(mem64(frame.call_result_ofs));
                }
                sink.i32_wrap_i64();
                sink.local_set(ca_cfp_local);
                // F'[0] is the callee's exit `fail_index`. The base-case loop
                // finish or this bridge's own recursive finish is a clean
                // DoneWithThisFrame — the result is already in the callee output
                // slot F'[1]. Any other value is a guard deopt the in-guest run
                // cannot finish; hand the callee frame to `wasm_ca_resume_deopt`,
                // which blackhole-resumes it on the host — resuming AT the guard,
                // so pre-guard work is not re-executed — and returns the result.
                sink.local_get(ca_cfp_local);
                sink.i64_load(mem64(0));
                sink.i32_wrap_i64();
                sink.local_set(ca_fi_local);
                // is_finish = (fi == loop_finish_fi) | (fi == bridge_finish_fi)
                sink.local_get(ca_fi_local);
                sink.i32_const(ca.loop_finish_fi as i32);
                sink.i32_eq();
                sink.local_get(ca_fi_local);
                sink.i32_const(bridge_finish_fi as i32);
                sink.i32_eq();
                sink.i32_or();
                sink.if_(BlockType::Result(ValType::I64));
                // clean finish: result Ref = F'[1] (output slot 0).
                sink.local_get(ca_cfp_local);
                sink.i64_load(mem64(FRAME_SLOT_BASE));
                sink.else_();
                // deopt: wasm_ca_resume_deopt(frame_ptr: i64, compiled_ptr: i64).
                sink.local_get(ca_cfp_local);
                sink.i64_extend_i32_u();
                sink.i64_const(ca.source_compiled_ptr as i64);
                sink.i32_const(ca.deopt_helper_slot as i32);
                // call_indirect(table_index, type_index): the shared table is 0.
                sink.call_indirect(0, ca_helper_type_idx);
                sink.end();
                // The recursive call or deopt helper may have collected and
                // moved this invocation's own frame. Reload it before the pop
                // trampoline and post-call home loads address local 0. As above,
                // the trampoline-only configuration retains its A0-era stale-
                // local-0 limitation because its scratch writes cannot reload it
                // safely.
                if let (Some(_base), Some(inline)) = (residual_type_base, ca.inline) {
                    emit_ca_reload_caller(&mut sink, inline.jf_top_addr);
                    sink.local_set(0);
                } else if let Some(base) = residual_type_base {
                    sink.i32_const(ca.ca_reload_caller_fn_ptr as i32);
                    sink.call_indirect(0, base);
                    sink.i32_wrap_i64();
                    sink.local_set(0);
                }
                // store-on-def homes the result Ref (from whichever branch).
                if !OpRef::raw_is_constant(vi) {
                    sink.local_set(1 + vi);
                } else {
                    sink.drop();
                }
                // Pop the callee frame off the jitframe shadow stack (strict
                // LIFO) via `wasm_jit_ca_pop_frame` — same direct-vs-trampoline
                // split as the alloc above (the pop only shrinks the shadow
                // stack; it never allocates or collects).
                if let (Some(_base), Some(inline)) = (residual_type_base, ca.inline) {
                    sink.i32_const(inline.jf_top_addr as i32);
                    sink.i32_const(inline.jf_top_addr as i32);
                    sink.i32_load(mem32(0));
                    sink.i32_const(8);
                    sink.i32_sub();
                    sink.i32_store(mem32(0));
                } else if let Some(base) = residual_type_base {
                    sink.local_get(ca_cfp_local);
                    sink.i64_extend_i32_u();
                    sink.i32_const(ca.ca_pop_fn_ptr as i32);
                    sink.call_indirect(0, base + 1);
                    sink.drop(); // returns 0; ignored
                } else {
                    let jit_call =
                        jit_call_idx.expect("CA arm needs jit_call for the frame trampolines");
                    sink.local_get(0);
                    sink.i64_const(ca.ca_pop_fn_ptr);
                    sink.i64_store(mem64(frame.call_func_ofs));
                    sink.local_get(0);
                    sink.i64_const(1);
                    sink.i64_store(mem64(frame.call_nargs_ofs));
                    sink.local_get(0);
                    sink.local_get(ca_cfp_local);
                    sink.i64_extend_i32_u();
                    sink.i64_store(mem64(frame.call_args_ofs));
                    sink.local_get(0);
                    emit_jit_call(&mut sink, jit_call, frame);
                }
                // The callee recursion minor-collected; this bridge's other live
                // Ref locals are now stale. Reload them from the forwarded homes.
                // Skip the result `vi`: its local holds the just-read callee output
                // and its home is not written until the store-on-def below, so a
                // reload would clobber it with the home's pre-call (stale) value.
                let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                emit_reload_ca_frame_if_necessary(
                    &mut sink,
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.inline,
                );
                emit_reload_refs_from_homes(&mut sink, ref_homes, &liveness, op_idx, skip, frame);
            }

            // ── CALL operations (via trampoline) ──
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallN
            | OpCode::CallF
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceN
            | OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilN
            | OpCode::CondCallValueI
            | OpCode::CondCallValueR
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantN
            | OpCode::CallLoopinvariantF
            | OpCode::CallPureF
            | OpCode::CallMayForceF
            | OpCode::CallAssemblerF
            | OpCode::CallReleaseGilF => {
                let vi = op.pos.get().raw();

                // Direct in-module residual call: skip the `jit_call` host hop and
                // `call_indirect` the callee's table slot with a static
                // `(i64×n)->i64` type. The residual ABI is uniformly i64 for
                // Int/Ref args+result, so args/result move on the wasm stack with
                // no marshalling and no call-area traffic. A direct target may
                // collect or force, so reload local 0 and its live Ref homes on
                // return. Falls back below when ineligible.
                if let (Some(base), Some(nargs)) = (residual_type_base, residual_call_i64_arity(op))
                {
                    let call_args = &op.getarglist()[1..];
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, arg.to_opref());
                    }
                    // func_ptr (arg 0) is the table slot — wrap to i32 index.
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    // call_indirect(table_index, type_index): table 0, type for arity n.
                    sink.call_indirect(0, base + nargs as u32);
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(1 + vi);
                    } else {
                        sink.drop(); // value-producing call whose result is unused
                    }
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                    // store-on-def (end of loop) homes a Ref result, so the
                    // direct path must NOT `continue` past it.
                } else if let (Some(base), Some(nargs)) =
                    (residual_type_base, residual_call_void_word_arity(op))
                {
                    // Direct in-module word-ABI void residual call: the callee
                    // really is `(i64×n)->i64` (descr result_size == 8), so use
                    // the i64 family and drop the dummy result.
                    let call_args = &op.getarglist()[1..];
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, arg.to_opref());
                    }
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.call_indirect(0, base + nargs as u32);
                    sink.drop();
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink, ref_homes, &liveness, op_idx, None, frame,
                    );
                } else {
                    let jit_call = jit_call_idx.expect("CALL op present but jit_call not imported");

                    // args[0] = func_ptr, args[1..] = call arguments
                    let func_ptr_ref = op.arg(0).to_opref();
                    let call_args = &op.getarglist()[1..];

                    // Store func_ptr to call area
                    sink.local_get(0);
                    emit_resolve(&mut sink, constants, func_ptr_ref);
                    sink.i64_store(mem64(frame.call_func_ofs));

                    // Store num_args
                    sink.local_get(0);
                    sink.i64_const(call_args.len() as i64);
                    sink.i64_store(mem64(frame.call_nargs_ofs));

                    // Store each arg
                    for (i, arg) in call_args.iter().enumerate() {
                        sink.local_get(0);
                        emit_resolve(&mut sink, constants, arg.to_opref());
                        sink.i64_store(mem64(frame.call_args_ofs + i as u64 * SLOT_SIZE));
                    }

                    // Call trampoline
                    sink.local_get(0);
                    emit_jit_call(&mut sink, jit_call, frame);

                    // Read result (for non-void calls)
                    let is_void = matches!(
                        op.opcode,
                        OpCode::CallN
                            | OpCode::CallPureN
                            | OpCode::CallMayForceN
                            | OpCode::CallAssemblerN
                            | OpCode::CallReleaseGilN
                            | OpCode::CallLoopinvariantN
                    );
                    if !OpRef::raw_is_constant(vi) && !is_void {
                        sink.local_get(0);
                        sink.i64_load(mem64(frame.call_result_ofs));
                        sink.local_set(1 + vi);
                    }
                    // Mirror the direct path: a trampoline residual call may force and collect.
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!is_void && !OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                }
            }

            // ── Allocation (via trampoline — treated as CALL) ──
            // llmodel.py:775-790 bh_new* parity: a `New*` survives
            // optimization whenever the allocated object escapes the trace
            // (e.g. reboxed result stored into a namespace). The trace cannot
            // allocate inline (the GC is host-side), so route through the
            // `jit_call` trampoline to the `wasm_jit_alloc` helper, then write
            // the vtable / length fields with pointer-width (i32) stores.
            OpCode::New | OpCode::NewWithVtable => {
                let vi = op.pos.get().raw();
                // llmodel.py:778-782: size, type_id, vtable from the size descr.
                let descr = op.getdescr();
                let sd = descr.as_ref().and_then(|d| d.as_size_descr());
                let (size, type_id, vtable) = sd.map_or((16i64, 0i64, 0usize), |sd| {
                    (sd.size() as i64, sd.type_id() as i64, sd.vtable())
                });

                // Inline nursery bump (rewrite.py malloc fast path, x86
                // `malloc_cond`): total = align8(max(header+size, MIN)); if
                // `free + total` fits below `nursery_top`, commit the bump and
                // write the header word (tid, no flags — young objects carry
                // none) inline; otherwise fall to the collecting helper.
                // Restricted to plain types (no destructor/weakref side-list)
                // under the large-object threshold, exactly the helper's own
                // fast path.
                let total_size = {
                    use majit_gc::header::GcHeader;
                    ((GcHeader::SIZE + size as usize).max(GcHeader::MIN_NURSERY_OBJ_SIZE) + 7) & !7
                };
                let inline_nursery = nursery.filter(|na| {
                    total_size <= na.large_threshold
                        && u32::try_from(type_id).is_ok_and(|t| na.plain_tids.contains(&t))
                });
                if let (Some(base), Some(na)) = (residual_type_base, inline_nursery) {
                    // free = *nursery_free; new_free = free + total
                    sink.i32_const(na.free_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_tee(alloc_scratch_local);
                    sink.i32_const(total_size as i32);
                    sink.i32_add();
                    // new_free > *nursery_top → slow path
                    sink.i32_const(na.top_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    // Slow: collecting helper. The collection may have moved
                    // every other live Ref; reload them from their (forwarded)
                    // homes — only here, the fast path moves nothing. Skip the
                    // fresh result (still on the operand stack; its home is
                    // written by store-on-def below).
                    sink.i64_const(type_id);
                    sink.i64_const(size);
                    sink.i32_const(alloc_fn_ptr as i32);
                    sink.call_indirect(0, base + 2);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                    sink.else_();
                    // Commit: *nursery_free = free + total.
                    sink.i32_const(na.free_addr as i32);
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(total_size as i32);
                    sink.i32_add();
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    // Header word: `GcHeader::new(tid)` — flags 0.
                    sink.local_get(alloc_scratch_local);
                    sink.i64_const(type_id);
                    sink.i64_store(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    // Result payload pointer = free + header size.
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(majit_gc::header::GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(1 + vi);
                    } else {
                        sink.drop();
                    }
                } else if let Some(base) = residual_type_base {
                    // Direct in-module allocation: `wasm_jit_alloc(type_id, size)`
                    // is a plain `(i64,i64)->i64` table entry, so call it like an
                    // eligible residual call — no host hop. Its fn ptr is a table
                    // index on wasm32.
                    sink.i64_const(type_id);
                    sink.i64_const(size);
                    sink.i32_const(alloc_fn_ptr as i32);
                    sink.call_indirect(0, base + 2);
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(1 + vi);
                    } else {
                        sink.drop();
                    }
                } else {
                    let jit_call = jit_call_idx.expect("New op present but jit_call not imported");
                    // func_ptr = wasm_jit_alloc
                    sink.local_get(0);
                    sink.i64_const(alloc_fn_ptr);
                    sink.i64_store(mem64(frame.call_func_ofs));
                    // num_args = 2
                    sink.local_get(0);
                    sink.i64_const(2);
                    sink.i64_store(mem64(frame.call_nargs_ofs));
                    // arg0 = type_id
                    sink.local_get(0);
                    sink.i64_const(type_id);
                    sink.i64_store(mem64(frame.call_args_ofs));
                    // arg1 = size
                    sink.local_get(0);
                    sink.i64_const(size);
                    sink.i64_store(mem64(frame.call_args_ofs + SLOT_SIZE));
                    // call trampoline
                    sink.local_get(0);
                    emit_jit_call(&mut sink, jit_call, frame);

                    if !OpRef::raw_is_constant(vi) {
                        // result pointer
                        sink.local_get(0);
                        sink.i64_load(mem64(frame.call_result_ofs));
                        sink.local_set(1 + vi);
                    }
                }

                if !OpRef::raw_is_constant(vi) {
                    // llmodel.py:779-781 write_int_at_mem(res, vtable_offset,
                    // WORD, vtable). The `ob_type` field is pointer-width: 4
                    // bytes on wasm32 (GuardClass reads it as i32), so store
                    // the low 32 bits to avoid clobbering the next field.
                    let write_vtable = op.opcode == OpCode::NewWithVtable
                        && vtable != 0
                        && vtable_offset.is_some();
                    if write_vtable {
                        let vt_off = vtable_offset.unwrap() as u64;
                        sink.local_get(1 + vi);
                        sink.i32_wrap_i64();
                        sink.i32_const(vtable as i32);
                        sink.i32_store(MemArg {
                            offset: vt_off,
                            align: 2,
                            memory_index: 0,
                        });
                    }
                }
                // The collecting allocation may have moved every other live
                // Ref; reload them from their (forwarded) homes. Skip the fresh
                // result — it was allocated after the collection and its home is
                // written by store-on-def below. The inline-bump path already
                // emitted this reload inside its slow arm (the fast bump moves
                // nothing).
                if residual_type_base.is_none() || inline_nursery.is_none() {
                    let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink, ref_homes, &liveness, op_idx, skip, frame,
                    );
                }
            }
            OpCode::NewArray | OpCode::NewArrayClear => {
                let vi = op.pos.get().raw();
                let descr = op.getdescr();
                let ad = descr.as_ref().and_then(|d| d.as_array_descr());
                let (base_size, item_size) = ad.map_or((16i64, 8i64), |ad| {
                    (ad.base_size() as i64, ad.item_size() as i64)
                });
                let len_offset = ad
                    .and_then(|ad| ad.len_descr())
                    .map_or(0i64, |ld| ld.offset() as i64);
                let type_id = ad.map_or(0i64, |ad| ad.type_id() as i64);

                // Inline nursery bump for arrays of a plain type under the
                // large-object threshold (same fast path as the `New` arm).
                // Constant lengths keep the existing compile-time total; a
                // runtime length uses malloc_cond_varsize's precheck against a
                // compile-time maxlength before computing the bump size.
                // The nursery is bulk-zeroed on reset so `NewArrayClear`'s
                // cleared items come for free, exactly like the helper.
                let length_const = const_operand_value(constants, op.arg(0).to_opref());
                let inline_nursery_total = length_const.and_then(|len| {
                    use majit_gc::header::GcHeader;
                    let len = usize::try_from(len).ok()?;
                    let payload =
                        (base_size as usize).checked_add((item_size as usize).checked_mul(len)?)?;
                    let total =
                        ((GcHeader::SIZE + payload).max(GcHeader::MIN_NURSERY_OBJ_SIZE) + 7) & !7;
                    let na = nursery.filter(|na| {
                        total <= na.large_threshold
                            && u32::try_from(type_id).is_ok_and(|t| na.plain_tids.contains(&t))
                    })?;
                    Some((total, len, na))
                });
                let inline_nursery_varsize = if length_const.is_none() {
                    (|| {
                        use majit_gc::header::GcHeader;
                        let base_size_usize = usize::try_from(base_size).ok()?;
                        let item_size_usize = usize::try_from(item_size).ok()?;
                        let base_total = GcHeader::SIZE.checked_add(base_size_usize)?;
                        let na = nursery.filter(|na| {
                            u32::try_from(type_id).is_ok_and(|t| na.plain_tids.contains(&t))
                        })?;
                        // malloc_cond_varsize checks the length before doing
                        // the scaled size calculation.  Use the largest length
                        // whose rounded total still fits under the nursery
                        // large-object threshold, capped to wasm32's usize
                        // length field.
                        let threshold = na.large_threshold.min(u32::MAX as usize) & !7;
                        if threshold < GcHeader::MIN_NURSERY_OBJ_SIZE || base_total > threshold {
                            return None;
                        }
                        let max_len = if item_size_usize == 0 {
                            u32::MAX as usize
                        } else {
                            ((threshold - base_total) / item_size_usize).min(u32::MAX as usize)
                        };
                        let max_len = i64::try_from(max_len).ok()?;
                        Some((max_len, base_total as i64, item_size_usize as i64, na))
                    })()
                } else {
                    None
                };
                if let (Some(base), Some((total_size, length, na))) =
                    (residual_type_base, inline_nursery_total)
                {
                    // free = *nursery_free; new_free = free + total
                    sink.i32_const(na.free_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_tee(alloc_scratch_local);
                    sink.i32_const(total_size as i32);
                    sink.i32_add();
                    sink.i32_const(na.top_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    // Slow: collecting helper; reload the other live Refs from
                    // their (forwarded) homes — only here, the fast bump moves
                    // nothing.
                    sink.i64_const(type_id);
                    sink.i64_const(base_size);
                    sink.i64_const(item_size);
                    sink.i64_const(length as i64);
                    sink.i64_const(len_offset);
                    sink.i32_const(alloc_array_fn_ptr as i32);
                    sink.call_indirect(0, base + 5);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                    sink.else_();
                    // Commit: *nursery_free = free + total.
                    sink.i32_const(na.free_addr as i32);
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(total_size as i32);
                    sink.i32_add();
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    // Header word: `GcHeader::new(tid)` — flags 0.
                    sink.local_get(alloc_scratch_local);
                    sink.i64_const(type_id);
                    sink.i64_store(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    // Length field (usize, 4 bytes on wasm32) at
                    // `payload + len_offset`.
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(length as i32);
                    sink.i32_store(MemArg {
                        offset: majit_gc::header::GcHeader::SIZE as u64 + len_offset as u64,
                        align: 2,
                        memory_index: 0,
                    });
                    // Result payload pointer = free + header size.
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(majit_gc::header::GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(1 + vi);
                    } else {
                        sink.drop();
                    }
                } else if let (Some(base), Some((max_len, base_total, item_size, na))) =
                    (residual_type_base, inline_nursery_varsize)
                {
                    // malloc_cond_varsize: negative lengths compare greater in
                    // the unsigned precheck and go to the collecting slow path.
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(max_len);
                    sink.i64_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    sink.i64_const(type_id);
                    sink.i64_const(base_size);
                    sink.i64_const(item_size);
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(len_offset);
                    sink.i32_const(alloc_array_fn_ptr as i32);
                    sink.call_indirect(0, base + 5);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                    sink.else_();
                    // total = round_up_8(max(header + base + item * length,
                    // MIN_NURSERY_OBJ_SIZE)).
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(item_size);
                    sink.i64_mul();
                    sink.i64_const(base_total);
                    sink.i64_add();
                    sink.i32_wrap_i64();
                    sink.local_set(alloc_size_local);
                    sink.local_get(alloc_size_local);
                    sink.i32_const(majit_gc::header::GcHeader::MIN_NURSERY_OBJ_SIZE as i32);
                    sink.i32_lt_u();
                    sink.if_(BlockType::Result(ValType::I32));
                    sink.i32_const(majit_gc::header::GcHeader::MIN_NURSERY_OBJ_SIZE as i32);
                    sink.else_();
                    sink.local_get(alloc_size_local);
                    sink.end();
                    sink.i32_const(7);
                    sink.i32_add();
                    sink.i32_const(-8);
                    sink.i32_and();
                    sink.local_set(alloc_size_local);

                    // free = *nursery_free; new_free = free + total
                    sink.i32_const(na.free_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_tee(alloc_scratch_local);
                    sink.local_get(alloc_size_local);
                    sink.i32_add();
                    sink.local_tee(alloc_size_local);
                    sink.i32_const(na.top_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    sink.i64_const(type_id);
                    sink.i64_const(base_size);
                    sink.i64_const(item_size);
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(len_offset);
                    sink.i32_const(alloc_array_fn_ptr as i32);
                    sink.call_indirect(0, base + 5);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                    sink.else_();
                    // Commit: *nursery_free = new_free.
                    sink.i32_const(na.free_addr as i32);
                    sink.local_get(alloc_size_local);
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    // Header word: `GcHeader::new(tid)` — flags 0.
                    sink.local_get(alloc_scratch_local);
                    sink.i64_const(type_id);
                    sink.i64_store(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    // Length field (usize, 4 bytes on wasm32) at
                    // `payload + len_offset`.
                    sink.local_get(alloc_scratch_local);
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i32_store(MemArg {
                        offset: majit_gc::header::GcHeader::SIZE as u64 + len_offset as u64,
                        align: 2,
                        memory_index: 0,
                    });
                    // Result payload pointer = free + header size.
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(majit_gc::header::GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                    sink.end();
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(1 + vi);
                    } else {
                        sink.drop();
                    }
                } else if let Some(base) = residual_type_base {
                    // Direct in-module allocation, like the `New` arm:
                    // `wasm_jit_alloc_array(type_id, base_size, item_size,
                    // length, len_offset)` is a `(i64×5)->i64` table entry.
                    sink.i64_const(type_id);
                    sink.i64_const(base_size);
                    sink.i64_const(item_size);
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(len_offset);
                    sink.i32_const(alloc_array_fn_ptr as i32);
                    sink.call_indirect(0, base + 5);
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(1 + vi);
                    } else {
                        sink.drop();
                    }
                } else {
                    let jit_call =
                        jit_call_idx.expect("NewArray op present but jit_call not imported");
                    // func_ptr = wasm_jit_alloc_array
                    sink.local_get(0);
                    sink.i64_const(alloc_array_fn_ptr);
                    sink.i64_store(mem64(frame.call_func_ofs));
                    // num_args = 5
                    sink.local_get(0);
                    sink.i64_const(5);
                    sink.i64_store(mem64(frame.call_nargs_ofs));
                    // arg0 = type_id
                    sink.local_get(0);
                    sink.i64_const(type_id);
                    sink.i64_store(mem64(frame.call_args_ofs));
                    // arg1 = base_size
                    sink.local_get(0);
                    sink.i64_const(base_size);
                    sink.i64_store(mem64(frame.call_args_ofs + SLOT_SIZE));
                    // arg2 = item_size
                    sink.local_get(0);
                    sink.i64_const(item_size);
                    sink.i64_store(mem64(frame.call_args_ofs + 2 * SLOT_SIZE));
                    // arg3 = length (op.arg(0))
                    sink.local_get(0);
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_store(mem64(frame.call_args_ofs + 3 * SLOT_SIZE));
                    // arg4 = len_offset
                    sink.local_get(0);
                    sink.i64_const(len_offset);
                    sink.i64_store(mem64(frame.call_args_ofs + 4 * SLOT_SIZE));
                    // call trampoline
                    sink.local_get(0);
                    emit_jit_call(&mut sink, jit_call, frame);

                    if !OpRef::raw_is_constant(vi) {
                        sink.local_get(0);
                        sink.i64_load(mem64(frame.call_result_ofs));
                        sink.local_set(1 + vi);
                    }
                }
                // `wasm_jit_alloc_array` collects; reload other live Refs. The
                // inline-bump paths already emitted this inside their slow arms.
                if residual_type_base.is_none()
                    || (inline_nursery_total.is_none() && inline_nursery_varsize.is_none())
                {
                    let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink, ref_homes, &liveness, op_idx, skip, frame,
                    );
                }
            }

            // ── Misc ──
            OpCode::ForceToken => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i64_const(0); // sentinel force token
                    sink.local_set(1 + vi);
                }
            }

            // Float operations
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // Values stored as i64 (bitcast from f64)
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.f64_reinterpret_i64();
                    match op.opcode {
                        OpCode::FloatAdd => {
                            sink.f64_add();
                        }
                        OpCode::FloatSub => {
                            sink.f64_sub();
                        }
                        OpCode::FloatMul => {
                            sink.f64_mul();
                        }
                        OpCode::FloatTrueDiv => {
                            sink.f64_div();
                        }
                        _ => unreachable!(),
                    }
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::FloatNeg => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.f64_neg();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::FloatAbs => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.f64_abs();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // Debug / metadata / no-op
            OpCode::DebugMergePoint
            | OpCode::IncrementDebugCounter
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::VirtualRefFinish
            | OpCode::ForceSpill
            | OpCode::Keepalive => {}

            _ => {
                // An opcode with no codegen arm. If it produces a value (a
                // result local that later ops read), silently skipping it
                // leaves a stale slot and yields wrong results, so decline the
                // whole trace and let the metainterp fall back to the
                // interpreter (correct, unaccelerated). Side-effect-free
                // metadata opcodes that produce no value are enumerated above.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    return Err(BackendError::Unsupported(format!(
                        "wasm codegen: unhandled value-producing opcode {:?}",
                        op.opcode
                    )));
                }
            }
        }

        // store-on-def: mirror a freshly-defined Ref result into its home slot
        // so a (future) collecting allocation can forward it. The local
        // `1 + raw` holds the value the matched arm just set; `ref_homes` only
        // keys Ref-typed value ids, so non-Ref / void / constant ops are
        // skipped. Each value-producing arm is operand-stack-neutral, so this
        // appended store is balanced.
        let result = op.pos.get();
        if let Some(h) = ref_homes.home(result) {
            sink.local_get(0);
            sink.local_get(1 + result.raw());
            sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
        }
    }

    if has_loop {
        sink.end(); // end loop
        sink.end(); // end block
    } else if straightline_dispatch {
        sink.end(); // end exit block A (Label-less dispatch trace, no `loop`)
    }

    // Epilogue bridge dispatch. Control reaches here only
    // after a guard `br`'d out of the exit block, having written its
    // `fail_index` into `frame[0]`. Look up that guard's bridge slot in the
    // shared cell array; if a bridge has been compiled (slot != 0), tail into
    // it via `call_indirect` through the shared table — staying inside wasm —
    // and return its result. Otherwise fall through to the host round-trip
    // (return `frame_ptr`, the metainterp reads `frame[0]`). With every cell 0
    // (no bridge yet) this is inert and behavior is unchanged.
    if bridge_dispatch {
        // slot = *(cells_base + (fail_index - fail_index_base) * 4)
        // The cell array is local to this trace (one i32 per local guard);
        // `frame[0]` carries the GLOBAL fail index, so subtract this trace's
        // base back to a local cell index.
        sink.i32_const(cells_base as i32);
        sink.local_get(0);
        sink.i64_load(mem64(0)); // frame[0] = fail_index
        sink.i32_wrap_i64();
        if fail_index_base != 0 {
            sink.i32_const(fail_index_base as i32);
            sink.i32_sub();
        }
        sink.i32_const(4);
        sink.i32_mul();
        sink.i32_add();
        sink.i32_load(memarg(0, 2));
        sink.local_tee(bridge_slot_local);
        sink.if_(BlockType::Empty);
        sink.local_get(0); // frame_ptr argument to the bridge
        sink.local_get(bridge_slot_local); // table slot
        sink.return_call_indirect(0, 0); // tail call, table 0, type 0: (i32) -> i32
        sink.end();
    }

    sink.local_get(0);
    sink.end(); // end function

    Ok(func)
}

// ── Helpers ──

/// A peeled loop — real work (the unrolled first iteration = preamble) precedes
/// the loop-header LABEL — whether it carries one LABEL or several. `loop` is
/// emitted at the LAST label, so `build_function` wraps the trace in the
/// resume-at-LABEL entry `br_table` (keyed on the frame dispatch-key slot,
/// key = label ordinal + 1) and a loop-closing bridge re-enters at ANY of the
/// loop's labels, in-module. `build_function` keys its wrapper on this
/// predicate; `compile_loop` records it on `CompiledWasmLoop` as
/// `has_preamble`. `compile_bridge` accepts a loop-closing bridge only when
/// its JUMP's descr identifies one of the source loop's OWN labels
/// (`label_descrs`) with matching arity and a resume-safe live set.
pub fn is_resumable_peeled(ops: &[Op]) -> bool {
    let Some(last_label) = ops.iter().rposition(|op| op.opcode == OpCode::Label) else {
        return false;
    };
    ops[..last_label]
        .iter()
        .any(|op| op.opcode != OpCode::Label)
}

/// The single-label subset of `is_resumable_peeled`: exactly one LABEL.
/// No longer consulted by the bridge accept-condition (which resolves the
/// JUMP's target label by descr identity uniformly); kept as a shape
/// predicate for tests.
pub fn is_single_label_peeled(ops: &[Op]) -> bool {
    let label_count = ops.iter().filter(|op| op.opcode == OpCode::Label).count();
    is_resumable_peeled(ops) && label_count == 1
}

/// Argument count of each `LABEL`, in ordinal order (the same ordinals
/// `compile_loop` stamps via `set_label_block_id`). `compile_bridge` declines
/// a loop-closing bridge whose JUMP arity differs from its target label's
/// count, since the resume loader reads exactly that many positional frame
/// slots.
pub fn label_arg_counts(ops: &[Op]) -> Vec<usize> {
    ops.iter()
        .filter(|op| op.opcode == OpCode::Label)
        .map(|op| op.getarglist().len())
        .collect()
}

/// Per-label resume safety, in ordinal order: label `j` is safe to resume at
/// when every op after it references only values that are constants, defined
/// after the label, or listed in the label's own args — i.e. the label's args
/// are the complete live set, so the resume loader reconstructs every value
/// the remainder of the trace reads. A value defined before the label and
/// read after it without being a label arg would resume as a null local (the
/// resume path skips the entry loader and every earlier segment). Guard fail
/// args count as reads — they spill into the deopt frame.
pub fn label_resume_safety(ops: &[Op]) -> Vec<bool> {
    ops.iter()
        .enumerate()
        .filter(|(_, op)| op.opcode == OpCode::Label)
        .map(|(p, label)| {
            let mut live: std::collections::HashSet<u32> = label
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .filter(|r| *r != OpRef::NONE && !r.is_constant())
                .map(|r| r.raw())
                .collect();
            for op in &ops[p + 1..] {
                let args = op.getarglist();
                let arg_reads = args.iter().map(|a| a.to_opref());
                let fail_reads = op
                    .getfailargs()
                    .map(|fa| fa.iter().map(|a| a.to_opref()).collect::<Vec<_>>())
                    .unwrap_or_default();
                for r in arg_reads.chain(fail_reads) {
                    if r != OpRef::NONE && !r.is_constant() && !live.contains(&r.raw()) {
                        return false;
                    }
                }
                let res = op.pos.get();
                if res != OpRef::NONE && !res.is_constant() {
                    live.insert(res.raw());
                }
            }
            true
        })
        .collect()
}

fn find_label_args(ops: &[Op]) -> Vec<OpRef> {
    // The JUMP branches back to the loop-header label, which is the LAST
    // label in a peeled trace (an outer entry label may precede it). The
    // `loop` is emitted at that same label in `build_function`.
    for op in ops.iter().rev() {
        if op.opcode == OpCode::Label {
            return op.getarglist().iter().map(|a| a.to_opref()).collect();
        }
    }
    Vec::new()
}

fn emit_resolve(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    opref: OpRef,
) {
    if opref.is_constant() {
        // history.py:227/268/314 — inline-Const variants carry value inline.
        let val = opref
            .inline_const_bits()
            .unwrap_or_else(|| constants.get(&opref.raw()).copied().unwrap_or(0));
        sink.i64_const(val);
    } else {
        sink.local_get(1 + opref.raw());
    }
}

/// Compile-time value of a constant operand (what `emit_resolve` would push
/// as `i64.const`), or `None` for a runtime value.
fn const_operand_value(constants: &indexmap::IndexMap<u32, i64>, opref: OpRef) -> Option<i64> {
    opref.is_constant().then(|| {
        opref
            .inline_const_bits()
            .unwrap_or_else(|| constants.get(&opref.raw()).copied().unwrap_or(0))
    })
}

/// Extract field offset from op's descr (FieldDescr).
fn field_offset_from_descr(op: &Op) -> u64 {
    let __descr_arc_descr = op.getdescr();
    if let Some(ref descr) = __descr_arc_descr.as_ref() {
        if let Some(fd) = descr.as_field_descr() {
            return fd.offset() as u64;
        }
    }
    0
}

/// `(length-field offset, length-field size)` from an op's ArrayDescr length
/// descriptor, mirroring `bh_arraylen_gc`, which reads the length at
/// `len_descr().offset()` at machine-word width. The offset is taken from the
/// registered descr (not hardcoded) so it tracks the real per-target layout,
/// and the size lets the caller load at the field's true width — a word-sized
/// length is 4 bytes on wasm32, so a fixed 8-byte read would pull the adjacent
/// field into the high half. Falls back to the conventional offset / word
/// width when no length descr is registered.
fn array_len_layout_from_descr(op: &Op) -> (u64, usize) {
    op.with_array_descr(|ad| {
        ad.len_descr()
            .map(|ld| (ld.offset() as u64, ld.field_size()))
    })
    .flatten()
    .unwrap_or((8, std::mem::size_of::<usize>()))
}

/// Compute array element address: base + base_size + index * item_size.
/// Leaves i32 address on the wasm stack.
fn emit_array_addr(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
) {
    let (base_size, item_size) = op
        .with_array_descr(|ad| (ad.base_size() as u64, ad.item_size() as u64))
        .unwrap_or((16, 8));
    emit_resolve(sink, constants, op.arg(0).to_opref()); // array ptr
    sink.i32_wrap_i64();
    // base + base_size + index * item_size
    emit_resolve(sink, constants, op.arg(1).to_opref()); // index
    sink.i32_wrap_i64();
    sink.i32_const(item_size as i32);
    sink.i32_mul();
    sink.i32_add();
    sink.i32_const(base_size as i32);
    sink.i32_add();
}

// ── Guard emission helpers ──

fn emit_guard_true(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: Option<u32>,
) {
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.i64_eqz();
    emit_guard_if_exit(sink, constants, guard_idx, op, block_exit_depth);
}

fn emit_guard_false(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: Option<u32>,
) {
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.i64_const(0);
    sink.i64_ne();
    emit_guard_if_exit(sink, constants, guard_idx, op, block_exit_depth);
}

/// Common guard exit: condition is on stack (i32), emit if + exit.
///
/// `block_exit_depth` is the statement-level depth of the enclosing exit
/// `block` (preamble = 0, loop body = 1); the `+ 1` accounts for the `if`
/// this opens. `None` for straight-line traces with no exit block.
fn emit_guard_if_exit(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: Option<u32>,
) {
    sink.if_(BlockType::Empty);
    emit_guard_exit(sink, constants, guard_idx, op);
    match block_exit_depth {
        // Loop traces: `br` out of this `if` and the enclosing exit `block`
        // (the `+ 1` accounts for the `if`) to the function epilogue.
        Some(d) => {
            sink.br(d + 1);
        }
        // Straight-line traces have no enclosing block, so fall-through would
        // reach the terminal Finish and overwrite frame[0] with its
        // fail_index, discarding this guard's exit. Return the frame pointer
        // directly (the epilogue's value) to hand control to the metainterp.
        None => {
            sink.local_get(0);
            sink.return_();
        }
    }
    sink.end();
}

fn emit_guard_exit(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    guard_idx: u32,
    op: &Op,
) {
    let fail_args: Vec<OpRef> = op
        .getfailargs()
        .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
        .unwrap_or_else(|| op.getarglist().iter().map(|a| a.to_opref()).collect());

    for (i, &arg_ref) in fail_args.iter().enumerate() {
        let offset = FRAME_SLOT_BASE + i as u64 * SLOT_SIZE;
        sink.local_get(0);
        emit_resolve(sink, constants, arg_ref);
        sink.i64_store(mem64(offset));
    }

    sink.local_get(0);
    sink.i64_const(guard_idx as i64);
    sink.i64_store(mem64(0));
}

// ── Binary ops ──

enum BinOp {
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64RemS,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
}

fn apply_binop(sink: &mut InstructionSink<'_>, op: BinOp) {
    match op {
        BinOp::I64Add => {
            sink.i64_add();
        }
        BinOp::I64Sub => {
            sink.i64_sub();
        }
        BinOp::I64Mul => {
            sink.i64_mul();
        }
        BinOp::I64DivS => {
            sink.i64_div_s();
        }
        BinOp::I64RemS => {
            sink.i64_rem_s();
        }
        BinOp::I64And => {
            sink.i64_and();
        }
        BinOp::I64Or => {
            sink.i64_or();
        }
        BinOp::I64Xor => {
            sink.i64_xor();
        }
        BinOp::I64Shl => {
            sink.i64_shl();
        }
        BinOp::I64ShrS => {
            sink.i64_shr_s();
        }
        BinOp::I64ShrU => {
            sink.i64_shr_u();
        }
    }
}

fn emit_binop(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
    binop: BinOp,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_resolve(sink, constants, op.arg(0).to_opref());
    emit_resolve(sink, constants, op.arg(1).to_opref());
    apply_binop(sink, binop);
    sink.local_set(1 + vi);
}

/// `UintMulHigh`: high 64 bits of the unsigned 64×64→128 product. Wasm has
/// only `i64.mul` (low 64 bits), so compute via the classic 32-bit split:
/// a = ah·2³²+al, b = bh·2³²+bl, with carry-safe intermediates
///   mid1 = ah·bl + (al·bl >> 32)
///   high = ah·bh + (mid1 >> 32) + ((al·bh + (mid1 & 0xFFFFFFFF)) >> 32)
/// Uses the five scratch locals reserved at `num_vars+1 ..= num_vars+5`.
fn emit_umulhi(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
    num_vars: u32,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    const MASK32: i64 = 0xFFFF_FFFF;
    let al = num_vars + 1;
    let ah = num_vars + 2;
    let bl = num_vars + 3;
    let bh = num_vars + 4;
    let mid1 = num_vars + 5;

    // al = a & 0xFFFFFFFF
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.local_set(al);
    // ah = a >>u 32
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_set(ah);
    // bl = b & 0xFFFFFFFF
    emit_resolve(sink, constants, op.arg(1).to_opref());
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.local_set(bl);
    // bh = b >>u 32
    emit_resolve(sink, constants, op.arg(1).to_opref());
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_set(bh);

    // mid1 = ah*bl + ((al*bl) >>u 32)
    sink.local_get(al);
    sink.local_get(bl);
    sink.i64_mul();
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_get(ah);
    sink.local_get(bl);
    sink.i64_mul();
    sink.i64_add();
    sink.local_set(mid1);

    // high = ah*bh + (mid1 >>u 32) + ((al*bh + (mid1 & MASK32)) >>u 32)
    sink.local_get(ah);
    sink.local_get(bh);
    sink.i64_mul();
    sink.local_get(mid1);
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.i64_add();
    sink.local_get(al);
    sink.local_get(bh);
    sink.i64_mul();
    sink.local_get(mid1);
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.i64_add();
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.i64_add();

    sink.local_set(1 + vi);
}

/// Overflow binary op: stores result in pos, overflow flag convention.
/// The overflow flag is not stored separately — GuardNoOverflow/GuardOverflow
/// is handled by checking after the fact (simplified for wasm MVP).
fn emit_ovf_binop(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
    binop: BinOp,
) {
    // For wasm MVP, just compute the result without overflow detection.
    // GuardNoOverflow/GuardOverflow are treated as always-pass.
    emit_binop(sink, constants, op, binop);
}

// ── Comparison ops ──

enum CmpOp {
    I64LtS,
    I64LeS,
    I64Eq,
    I64Ne,
    I64GtS,
    I64GeS,
    I64LtU,
    I64LeU,
    I64GtU,
    I64GeU,
}

fn apply_cmp(sink: &mut InstructionSink<'_>, op: CmpOp) {
    match op {
        CmpOp::I64LtS => {
            sink.i64_lt_s();
        }
        CmpOp::I64LeS => {
            sink.i64_le_s();
        }
        CmpOp::I64Eq => {
            sink.i64_eq();
        }
        CmpOp::I64Ne => {
            sink.i64_ne();
        }
        CmpOp::I64GtS => {
            sink.i64_gt_s();
        }
        CmpOp::I64GeS => {
            sink.i64_ge_s();
        }
        CmpOp::I64LtU => {
            sink.i64_lt_u();
        }
        CmpOp::I64LeU => {
            sink.i64_le_u();
        }
        CmpOp::I64GtU => {
            sink.i64_gt_u();
        }
        CmpOp::I64GeU => {
            sink.i64_ge_u();
        }
    }
}

// ── Float comparison helper ──

enum FloatCmp {
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
}

fn emit_float_cmp(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
    cmp: FloatCmp,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.f64_reinterpret_i64();
    emit_resolve(sink, constants, op.arg(1).to_opref());
    sink.f64_reinterpret_i64();
    match cmp {
        FloatCmp::Lt => {
            sink.f64_lt();
        }
        FloatCmp::Le => {
            sink.f64_le();
        }
        FloatCmp::Eq => {
            sink.f64_eq();
        }
        FloatCmp::Ne => {
            sink.f64_ne();
        }
        FloatCmp::Gt => {
            sink.f64_gt();
        }
        FloatCmp::Ge => {
            sink.f64_ge();
        }
    }
    sink.i64_extend_i32_u();
    sink.local_set(1 + vi);
}

fn emit_cmp(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
    cmpop: CmpOp,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_resolve(sink, constants, op.arg(0).to_opref());
    emit_resolve(sink, constants, op.arg(1).to_opref());
    apply_cmp(sink, cmpop);
    sink.i64_extend_i32_u();
    sink.local_set(1 + vi);
}

// ── Unary op helper ──

fn emit_unary_vi(
    sink: &mut InstructionSink<'_>,
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
    prefix: impl FnOnce(&mut InstructionSink<'_>),
    suffix: impl FnOnce(&mut InstructionSink<'_>),
) {
    let vi = op.pos.get().raw();
    if !OpRef::raw_is_constant(vi) {
        prefix(sink);
        emit_resolve(sink, constants, op.arg(0).to_opref());
        suffix(sink);
        sink.local_set(1 + vi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_geometry_keeps_tail_call_area_out_of_ca_prefix() {
        let frame = FrameGeometry::compact(32, 16);
        assert_eq!(frame.dispatch_key_ofs, 32 * SLOT_SIZE);
        assert_eq!(frame.home_slot_base, 33 * SLOT_SIZE);
        assert_eq!(frame.ca_frame_bytes, 392);
        assert_eq!(frame.call_result_ofs, frame.ca_frame_bytes as u64);
        assert_eq!(frame.call_args_ofs, 416);
        assert_eq!(frame.frame_bytes, 544);
    }
}
