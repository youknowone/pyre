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
///
/// The residual-call trampoline scratch is stored separately at the static
/// base returned by `jit_call_area_addr`.
use std::collections::HashMap;

use majit_backend::BackendError;
use majit_gc::header::{GcHeader, TYPE_ID_MASK};
use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, EntityType, ExportKind, ExportSection, Function,
    FunctionSection, GlobalSection, GlobalType, ImportSection, InstructionSink, MemArg, MemoryType,
    Module, RefType, TableType, TypeSection, ValType,
};

/// Frame slot byte offset: slot[i] is at frame_ptr + 8 + i * 8.
pub const FRAME_SLOT_BASE: u64 = 8;
const SLOT_SIZE: u64 = 8;

/// `frame[0]` is this backend's `jf_descr`: the u32 exit index a guard failure
/// stamps on its way out. Two bits above that index carry the force protocol
/// jitframe.py splits across two fields.
///
/// [`FORCE_ARMED_BIT`] stands in for `jf_force_descr`:
/// `_store_force_index_if_next_guard` publishes the bracketing
/// GUARD_NOT_FORCED's coordinate before a call that may force, and
/// `Backend::force` refuses a frame carrying none.
///
/// [`FORCE_TAKEN_BIT`] stands in for what `force` then writes into `jf_descr`,
/// the mark `genop_guard_guard_not_forced` reads with
/// `CMP [rbp + jf_descr], 0` to turn a force that landed inside the call into a
/// deopt.
///
/// Keeping both in `frame[0]` rather than in the real `JitFrame` header keeps
/// the protocol inside the local-0-relative data region, which is the one part
/// of the layout both the orthodox JitFrame path and the legacy host-Vec path
/// share. The host reads `frame[0] as u32`, so neither bit disturbs the index,
/// and a guard exit's own store clears both.
pub(crate) const FORCE_TAKEN_BIT: i64 = 1 << 32;
pub(crate) const FORCE_ARMED_BIT: i64 = 1 << 33;

/// Scratch i64 locals reserved past the value locals for `emit_umulhi`
/// (al, ah, bl, bh, mid1).
const UMULHI_SCRATCH: u32 = 5;

/// Dense wasm-local assignment for the sparse value-id namespace.
struct ValueLocals {
    by_id: Vec<Option<u32>>,
    types: Vec<ValType>,
    /// First non-parameter local.  Ordinary traces have one frame-pointer
    /// parameter; parameter-entry bridges have that plus their fail values.
    first_local: u32,
}

impl ValueLocals {
    fn mark(
        by_id: &mut [Option<u32>],
        id_types: &mut [ValType],
        has_authoritative_type: &mut [bool],
        id: u32,
        ty: ValType,
        authoritative: bool,
    ) {
        let i = id as usize;
        assert!(i < by_id.len(), "value id {id} exceeds pre-pass bounds");
        by_id[i] = Some(0);
        // InputArg::tp and Op::result_type describe the defining value.  An
        // operand may be visited before its producer, so its embedded tag
        // only supplies a type while no definition has claimed this id.
        if authoritative || !has_authoritative_type[i] {
            id_types[i] = ty;
        }
        has_authoritative_type[i] |= authoritative;
    }

    fn collect(inputargs: &[InputArg], ops: &[Op], num_vars: u32, first_local: u32) -> Self {
        let mut by_id = vec![None; num_vars as usize];
        let mut id_types = vec![ValType::I64; num_vars as usize];
        let mut has_authoritative_type = vec![false; num_vars as usize];

        for ia in inputargs {
            Self::mark(
                &mut by_id,
                &mut id_types,
                &mut has_authoritative_type,
                ia.index,
                if ia.tp == Type::Float {
                    ValType::F64
                } else {
                    ValType::I64
                },
                true,
            );
        }
        for op in ops {
            let result = op.pos.get();
            if result != OpRef::NONE && !result.is_constant() {
                Self::mark(
                    &mut by_id,
                    &mut id_types,
                    &mut has_authoritative_type,
                    result.raw(),
                    if op.result_type() == Type::Float {
                        ValType::F64
                    } else {
                        ValType::I64
                    },
                    true,
                );
            }
            for arg in op.getarglist() {
                let arg = arg.to_opref();
                if arg != OpRef::NONE && !arg.is_constant() {
                    Self::mark(
                        &mut by_id,
                        &mut id_types,
                        &mut has_authoritative_type,
                        arg.raw(),
                        if arg.ty() == Some(Type::Float) {
                            ValType::F64
                        } else {
                            ValType::I64
                        },
                        false,
                    );
                }
            }
            if let Some(failargs) = op.getfailargs() {
                for arg in failargs {
                    let arg = arg.to_opref();
                    if arg != OpRef::NONE && !arg.is_constant() {
                        Self::mark(
                            &mut by_id,
                            &mut id_types,
                            &mut has_authoritative_type,
                            arg.raw(),
                            if arg.ty() == Some(Type::Float) {
                                ValType::F64
                            } else {
                                ValType::I64
                            },
                            false,
                        );
                    }
                }
            }
        }

        let mut types = Vec::new();
        for (id, slot) in by_id.iter_mut().enumerate() {
            if slot.is_some() {
                *slot = Some(types.len() as u32 + first_local);
                types.push(id_types[id]);
            }
        }
        Self {
            by_id,
            types,
            first_local,
        }
    }

    fn local(&self, id: u32) -> u32 {
        self.by_id
            .get(id as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("wasm value local is unmapped for id {id}"))
    }

    fn ty(&self, id: u32) -> ValType {
        self.types[(self.local(id) - self.first_local) as usize]
    }

    fn count(&self) -> u32 {
        self.types.len() as u32
    }

    fn types(&self) -> &[ValType] {
        &self.types
    }

    /// Local index immediately after the dense value-local range.
    fn end_local(&self) -> u32 {
        self.first_local + self.count()
    }

    /// Last dense value-local index, used as the base before scratch locals.
    fn last_local(&self) -> u32 {
        self.end_local() - 1
    }
}

/// Call area layout in the historical fixed frame geometry.
///
/// These offsets are the host trampoline's ABI, not a private detail: a caller
/// writes the callee's function-table index, the argument count and the
/// arguments into this block, invokes the import, and reads the callee's
/// result back from it. Whoever satisfies that import reads the same block
/// from the other side, so both ends name these constants instead of each
/// restating the numbers.
pub const CALL_RESULT_OFS: u64 = 2000;
pub const CALL_FUNC_OFS: u64 = 2008;
pub const CALL_NARGS_OFS: u64 = 2016;
pub const CALL_ARGS_OFS: u64 = 2024;

/// Arguments the call area has room for. A residual call with more arguments
/// than this has nowhere to put them, so a caller checks its arity against
/// this bound rather than writing past the end of the frame.
pub const MAX_CALL_ARGS: usize = 16;

const STATIC_CALL_RESULT_OFS: u64 = 0;
const STATIC_CALL_FUNC_OFS: u64 = SLOT_SIZE;
const STATIC_CALL_NARGS_OFS: u64 = 2 * SLOT_SIZE;
const STATIC_CALL_ARGS_OFS: u64 = 3 * SLOT_SIZE;

/// Minimum frame allocation size in bytes to accommodate the call area.
///
/// Derived from where the arguments start and how many fit, so raising
/// [`MAX_CALL_ARGS`] cannot leave the frame one argument short of the area it
/// is sized to hold.
pub const MIN_FRAME_BYTES: usize = CALL_ARGS_OFS as usize + MAX_CALL_ARGS * 8;

/// Per-token layout of a wasm execution frame. Every frozen geometry retains
/// the historical host-trampoline call area even though emitted code uses the
/// module-static scratch area. CA callee frames allocate only the prefix ending
/// after the Ref homes.
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
    /// Number of slots at the END of the Ref-home region reserved for
    /// resume-at-LABEL live-ins.  Ordinary per-trace Ref homes grow upward
    /// from `home_slot_base`; these captures grow from the frozen boundary and
    /// therefore survive execution of a chained bridge, whose own home map may
    /// use the low slots.  The whole home region remains covered by jf_gcmap.
    pub label_ref_slots: usize,
    /// Bytes through the end of Ref homes. CA callee frames allocate exactly
    /// this many item bytes; the unused tail call area is intentionally omitted.
    pub ca_frame_bytes: u32,
    /// Full bytes in the frame layout, including the tail call area. Host entry
    /// frames and every chained bridge use this geometry and allocation size.
    pub frame_bytes: u32,
}

impl FrameGeometry {
    /// result, function, nargs, then one slot per argument.
    pub const CALL_AREA_SLOTS: usize = 3 + MAX_CALL_ARGS;

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
            label_ref_slots: 0,
            ca_frame_bytes: HOME_SLOT_BASE as u32,
            frame_bytes: (MIN_FRAME_BYTES + SLOT_SIZE as usize) as u32,
        }
    }

    /// Compact frozen geometry for one token:
    /// `[value slots | dispatch key | Ref homes | call area]`.
    /// `value_slots` includes frame[0].  The trailing call area is always
    /// present, even for direct-only source traces, because later bridges are
    /// compiled against this immutable geometry.
    pub fn compact(value_slots: usize, home_slots: usize, label_ref_slots: usize) -> Self {
        debug_assert!(label_ref_slots <= home_slots);
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
            label_ref_slots,
            ca_frame_bytes: ca_frame_bytes as u32,
            frame_bytes: frame_bytes as u32,
        }
    }

    /// Low Ref homes available to the trace currently executing on this
    /// geometry.  The high `label_ref_slots` belong to the source loop's LABEL
    /// capture plan and must not be cleared or reused by a chained bridge.
    pub const fn ordinary_home_slots(self) -> usize {
        self.home_slots - self.label_ref_slots
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

/// A small lookbehind buffer for local wasm instruction folds.
///
/// Every instruction method used by this emitter is spelled out below.  In
/// particular, this type deliberately does not implement `Deref`: reaching
/// the underlying sink without flushing would reorder pending instructions.
struct PeepSink<'sink, 'buf> {
    sink: &'sink mut InstructionSink<'buf>,
    pending: Vec<PendingInstruction>,
}

#[derive(Clone, Copy)]
enum PendingInstruction {
    LocalSet(u32),
    I64Const(i64),
    I32Const(i32),
}

macro_rules! forward_zero {
    ($($method:ident),* $(,)?) => {
        $(
            fn $method(&mut self) -> &mut Self {
                self.flush();
                self.sink.$method();
                self
            }
        )*
    };
}

macro_rules! forward_one {
    ($($method:ident($arg:ident: $ty:ty)),* $(,)?) => {
        $(
            fn $method(&mut self, $arg: $ty) -> &mut Self {
                self.flush();
                self.sink.$method($arg);
                self
            }
        )*
    };
}

macro_rules! forward_two {
    ($($method:ident($first:ident: $first_ty:ty, $second:ident: $second_ty:ty)),* $(,)?) => {
        $(
            fn $method(&mut self, $first: $first_ty, $second: $second_ty) -> &mut Self {
                self.flush();
                self.sink.$method($first, $second);
                self
            }
        )*
    };
}

#[allow(dead_code)]
impl<'sink, 'buf> PeepSink<'sink, 'buf> {
    fn new(sink: &'sink mut InstructionSink<'buf>) -> Self {
        Self {
            sink,
            pending: Vec::with_capacity(2),
        }
    }

    /// Commit every buffered instruction in program order.
    fn flush(&mut self) {
        for instruction in self.pending.drain(..) {
            match instruction {
                PendingInstruction::LocalSet(local) => {
                    self.sink.local_set(local);
                }
                PendingInstruction::I64Const(value) => {
                    self.sink.i64_const(value);
                }
                PendingInstruction::I32Const(value) => {
                    self.sink.i32_const(value);
                }
            }
        }
    }

    fn local_set(&mut self, local: u32) -> &mut Self {
        self.flush();
        self.pending.push(PendingInstruction::LocalSet(local));
        self
    }

    fn local_get(&mut self, local: u32) -> &mut Self {
        if matches!(self.pending.last(), Some(PendingInstruction::LocalSet(previous)) if *previous == local)
        {
            self.pending.pop();
            self.flush();
            self.sink.local_tee(local);
        } else {
            self.flush();
            self.sink.local_get(local);
        }
        self
    }

    fn i64_const(&mut self, value: i64) -> &mut Self {
        self.flush();
        self.pending.push(PendingInstruction::I64Const(value));
        self
    }

    fn i32_wrap_i64(&mut self) -> &mut Self {
        // Inspect the tail before removing it, the way every other fold here
        // does: `pending` is a lookbehind buffer, not an operand stack, so a
        // tail this fold does not consume still owes its instruction to
        // `flush`.
        if let Some(PendingInstruction::I64Const(value)) = self.pending.last().copied() {
            self.pending.pop();
            self.pending
                .push(PendingInstruction::I32Const(value as u64 as u32 as i32));
        } else {
            self.flush();
            self.sink.i32_wrap_i64();
        }
        self
    }

    fn i32_const(&mut self, value: i32) -> &mut Self {
        if !matches!(self.pending.as_slice(), [PendingInstruction::I32Const(_)]) {
            self.flush();
        }
        self.pending.push(PendingInstruction::I32Const(value));
        self
    }

    fn i32_mul(&mut self) -> &mut Self {
        if let [
            PendingInstruction::I32Const(lhs),
            PendingInstruction::I32Const(rhs),
        ] = self.pending.as_slice()
        {
            let value = lhs.wrapping_mul(*rhs);
            self.pending.clear();
            self.pending.push(PendingInstruction::I32Const(value));
        } else {
            self.flush();
            self.sink.i32_mul();
        }
        self
    }

    fn i32_add(&mut self) -> &mut Self {
        if matches!(self.pending.as_slice(), [PendingInstruction::I32Const(0)]) {
            self.pending.clear();
        } else {
            self.flush();
            self.sink.i32_add();
        }
        self
    }

    fn br_table<V: IntoIterator<Item = u32>>(&mut self, labels: V, default: u32) -> &mut Self
    where
        V::IntoIter: ExactSizeIterator,
    {
        self.flush();
        self.sink.br_table(labels, default);
        self
    }

    forward_zero!(
        drop,
        else_,
        end,
        f64_abs,
        f64_add,
        f64_convert_i64_s,
        f64_div,
        f64_eq,
        f64_floor,
        f64_ge,
        f64_gt,
        f64_le,
        f64_lt,
        f64_mul,
        f64_ne,
        f64_neg,
        f64_reinterpret_i64,
        f64_sub,
        i32_and,
        i32_eq,
        i32_eqz,
        i32_gt_u,
        i32_lt_u,
        i32_ne,
        i32_or,
        i32_shl,
        i32_shr_u,
        i32_sub,
        i32_xor,
        i64_add,
        i64_and,
        i64_div_s,
        i64_eq,
        i64_eqz,
        i64_extend32_s,
        i64_extend_i32_s,
        i64_extend_i32_u,
        i64_ge_s,
        i64_ge_u,
        i64_gt_s,
        i64_gt_u,
        i64_le_s,
        i64_le_u,
        i64_lt_s,
        i64_lt_u,
        i64_mul,
        i64_ne,
        i64_or,
        i64_reinterpret_f64,
        i64_rem_s,
        i64_shl,
        i64_shr_s,
        i64_shr_u,
        i64_sub,
        i64_trunc_sat_f64_s,
        i64_xor,
        return_,
        select,
        unreachable,
    );

    forward_one!(
        block(block_type: BlockType),
        br(label: u32),
        br_if(label: u32),
        call(function: u32),
        f64_load(memarg: MemArg),
        f64_store(memarg: MemArg),
        i32_load(memarg: MemArg),
        i64_load16_s(memarg: MemArg),
        i64_load16_u(memarg: MemArg),
        i64_load32_s(memarg: MemArg),
        i64_load32_u(memarg: MemArg),
        i64_load8_s(memarg: MemArg),
        i64_load8_u(memarg: MemArg),
        i64_store16(memarg: MemArg),
        i64_store32(memarg: MemArg),
        i64_store8(memarg: MemArg),
        i32_load16_s(memarg: MemArg),
        i32_load16_u(memarg: MemArg),
        i32_load8_s(memarg: MemArg),
        i32_load8_u(memarg: MemArg),
        i32_store(memarg: MemArg),
        i32_store16(memarg: MemArg),
        i32_store8(memarg: MemArg),
        i64_load(memarg: MemArg),
        i64_store(memarg: MemArg),
        if_(block_type: BlockType),
        local_tee(local: u32),
        loop_(block_type: BlockType),
        return_call(function: u32),
    );

    forward_two!(
        call_indirect(table_index: u32, type_index: u32),
        return_call_indirect(table_index: u32, type_index: u32),
    );
}

impl Drop for PeepSink<'_, '_> {
    fn drop(&mut self) {
        self.flush();
    }
}

fn emit_call_area_addr(sink: &mut PeepSink<'_, '_>) {
    sink.i32_const(crate::jit_call_area_addr() as i32);
}

/// Invoke the residual-call trampoline, which reads its scratch at
/// `base + offset`. The scratch no longer lives in the frame, so the pair is
/// always the static call area at offset zero, and the base-only import — whose
/// host side adds a baked `CALL_RESULT_OFS` — can no longer be used.
fn emit_jit_call(sink: &mut PeepSink<'_, '_>, jit_call_idx: u32) {
    emit_call_area_addr(sink);
    sink.i32_const(0);
    sink.call(jit_call_idx);
}

/// Emit a width-correct integer load. The element address (i32) must be on
/// the stack; the result is an i64, sign- or zero-extended from `size`
/// bytes. Word-sized fields are 4 bytes on wasm32 (`isize`/`usize`/pointer),
/// 8 bytes on 64-bit; reading a fixed 8 bytes here would fold in the next
/// field's bytes on wasm32.
fn emit_sized_int_load(sink: &mut PeepSink<'_, '_>, offset: u64, size: usize, signed: bool) {
    // The i64 family loads and extends in one instruction, so the widening
    // never has to be spelled separately.
    match (size, signed) {
        (4, true) => sink.i64_load32_s(memarg(offset, 2)),
        (4, false) => sink.i64_load32_u(memarg(offset, 2)),
        (2, true) => sink.i64_load16_s(memarg(offset, 1)),
        (2, false) => sink.i64_load16_u(memarg(offset, 1)),
        (1, true) => sink.i64_load8_s(memarg(offset, 0)),
        (1, false) => sink.i64_load8_u(memarg(offset, 0)),
        _ => sink.i64_load(mem64(offset)),
    };
}

/// Emit a width-correct integer store. The stack must hold
/// `[addr_i32, value_i64]`; the low `size` bytes of the value are stored.
/// A fixed 8-byte store would clobber the adjacent field/item (or run past
/// the array end) for word-sized fields and pointer array items on wasm32.
fn emit_sized_int_store(sink: &mut PeepSink<'_, '_>, offset: u64, size: usize) {
    // The i64 family truncates as it stores, so the narrowing never has to be
    // spelled separately.
    match size {
        4 => sink.i64_store32(memarg(offset, 2)),
        2 => sink.i64_store16(memarg(offset, 1)),
        1 => sink.i64_store8(memarg(offset, 0)),
        _ => sink.i64_store(mem64(offset)),
    };
}

/// `(field_size, is_signed)` from an op's FieldDescr. A field op always carries
/// a FieldDescr; a missing one is an invariant violation, so panic rather than
/// emit a silently-wrong width.
fn field_size_sign_from_descr(op: &Op) -> (usize, bool) {
    let descr = op.getdescr();
    if let Some(fd) = descr.as_ref().and_then(|d| d.as_field_descr()) {
        return (fd.field_size(), fd.is_field_signed());
    }
    missing_layout_descr("field descr (size/sign)", op)
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
    missing_layout_descr("field descr (store size)", op)
}

fn field_is_float_from_descr(op: &Op) -> bool {
    let descr = op.getdescr();
    match descr.as_ref().and_then(|d| d.as_field_descr()) {
        Some(fd) => fd.is_float_field(),
        None => missing_layout_descr("field descr (is_float)", op),
    }
}

/// `(item_size, is_signed)` from an op's ArrayDescr. An array op always carries
/// an ArrayDescr; a missing one is an invariant violation, so panic.
fn array_item_size_sign_from_descr(op: &Op) -> (usize, bool) {
    op.with_array_descr(|ad| (ad.item_size(), ad.is_item_signed()))
        .unwrap_or_else(|| missing_layout_descr("array descr (item size/sign)", op))
}

/// `raw_load` / `raw_store` address `arg(0) + arg(1)` and nothing else: a raw
/// buffer has no GC array header, so the descr's base size is not part of the
/// address the way it is for the `GETARRAYITEM` family.
fn emit_raw_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
) {
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i32_wrap_i64();
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    sink.i32_wrap_i64();
    sink.i32_add();
}

fn array_item_is_float_from_descr(op: &Op) -> bool {
    op.with_array_descr(|ad| ad.item_type() == Type::Float)
        .unwrap_or_else(|| missing_layout_descr("array descr (item is_float)", op))
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
/// `[0, num_vars)` value-id space; a flat vector indexed by that id is the
/// natural fit — no hashing, and iteration is
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

    fn collect(
        inputargs: &[InputArg],
        ops: &[Op],
        include_ca_collects: bool,
        forced_refs: &[OpRef],
        regions: &[InlinedRegionSpan],
    ) -> Self {
        let liveness = HomeLiveness::collect_with_regions(inputargs, ops, regions);
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
            // CALL_ASSEMBLER's arguments. Those Ref operands are used at (not
            // after) this op, so ordinary `live_across` deliberately excludes
            // them; they nevertheless need homes through the prior allocation.
            for op in ops.iter().filter(|op| op.opcode.is_call_assembler()) {
                for arg in op.getarglist() {
                    let arg = arg.to_opref();
                    if ref_values.contains(arg) {
                        Self::assign(&mut by_id, &mut next, arg.raw());
                    }
                }
            }
        }
        // `store_force_descr` publishes the bracketing guard's fail arguments
        // into the frame and leaves the bracket armed past the op, so a force
        // arriving later is what reads them; x86 keeps that guard's gcmap as
        // `finish_gcmap` for the same reason.  Ordinary liveness stops at the
        // guard — nothing consumes them after it — so a Ref that crosses no
        // collecting call would take no home and `emit_force_arm` would publish
        // its raw pointer into the untraced exit slots.  Give every one of them
        // a traced home to name instead.
        for op in ops
            .iter()
            .filter(|op| matches!(op.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2))
        {
            for arg in exit_fail_args(op) {
                if ref_values.contains(arg) {
                    Self::assign(&mut by_id, &mut next, arg.raw());
                }
            }
        }
        // Resume-at-LABEL Ref captures must also have an ordinary home.  The
        // high capture slot preserves the value while another bridge executes
        // on this frame; the ordinary home participates in the existing
        // post-collection local reload machinery once the target resumes.
        for &r in forced_refs {
            if ref_values.contains(r) {
                Self::assign(&mut by_id, &mut next, r.raw());
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

#[derive(Clone, Copy)]
enum LabelCaptureStorage {
    /// Absolute frame value-slot index (slot zero is the fail index).
    ValueSlot(usize),
    /// Ordinal within the high, GC-rooted LABEL-capture home region.
    RefSlot(usize),
}

/// Backend-only preservation plan for values that remain live across a peeled
/// LABEL without appearing in that LABEL's semantic argument list.  RPython's
/// assembler keeps such values in the frozen frame; wasm locals disappear on
/// a tail-call re-entry, so we explicitly mirror that storage shape here.
struct LabelResumeData {
    per_label: Vec<Vec<OpRef>>,
    uncapturable: Vec<bool>,
    capture_by_id: Vec<Option<LabelCaptureStorage>>,
    captured_refs: Vec<OpRef>,
    scalar_slots: usize,
    ref_slots: usize,
}

/// Where one inlined bridge region starts in a merged analysis stream, and
/// which value ids carry that region's own live-ins.
struct InlinedRegionSpan {
    ops_start: usize,
    inputarg_ids: Vec<u32>,
}

impl InlinedRegionSpan {
    /// The regions occupy the tail of the merged stream in `inlined_bridges`
    /// order, so their starts run back from the end of `ops`. `bridges` must be
    /// the rebased copies the merged stream was built from, so the recorded ids
    /// are the ids that stream reads.
    fn collect(ops_len: usize, bridges: &[InlinedBridge]) -> Vec<Self> {
        let mut start =
            ops_len.saturating_sub(bridges.iter().map(|bridge| bridge.ops.len()).sum::<usize>());
        bridges
            .iter()
            .map(|bridge| {
                let span = Self {
                    ops_start: start,
                    inputarg_ids: bridge.inputargs.iter().map(|ia| ia.index).collect(),
                };
                start += bridge.ops.len();
                span
            })
            .collect()
    }
}

impl LabelResumeData {
    fn collect(inputargs: &[InputArg], ops: &[Op]) -> Self {
        Self::collect_with_regions(inputargs, ops, &[])
    }

    fn collect_with_regions(
        inputargs: &[InputArg],
        ops: &[Op],
        regions: &[InlinedRegionSpan],
    ) -> Self {
        let (_, num_vars) = collect_guards_and_vars(inputargs, ops);
        let ref_values = RefValues::collect(inputargs, ops);
        let normal_value_slots = normal_frame_value_slots(inputargs, ops);
        let mut has_producer = vec![false; num_vars as usize];
        let mut is_input = vec![false; num_vars as usize];
        for ia in inputargs {
            if let Some(v) = is_input.get_mut(ia.index as usize) {
                *v = true;
            }
        }
        for op in ops {
            let r = op.pos.get();
            if r != OpRef::NONE
                && !r.is_constant()
                && let Some(v) = has_producer.get_mut(r.raw() as usize)
            {
                *v = true;
            }
        }
        let mut per_label = Vec::new();
        let mut uncapturable = Vec::new();

        // Only the labels the entry dispatch can land on need a capture plan;
        // an in-body label is never resumed, so reserving frame slots for its
        // live-ins would only inflate the frozen geometry.
        let resumable = resumable_label_count(ops);
        for (label_pos, label) in ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.opcode == OpCode::Label)
            .take(resumable)
        {
            let mut available = vec![false; num_vars as usize];
            let mut defined_before = vec![false; num_vars as usize];
            // Producer-less value ids are folded constant-pool seeds. Codegen
            // binds them before the entry dispatch, so they dominate both the
            // key-0 path and every LABEL resume and need no frame capture.
            for (id, produced) in has_producer.iter().copied().enumerate() {
                if !produced && !is_input[id] {
                    available[id] = true;
                    defined_before[id] = true;
                }
            }
            for ia in inputargs {
                if let Some(v) = defined_before.get_mut(ia.index as usize) {
                    *v = true;
                }
            }
            for op in &ops[..label_pos] {
                let r = op.pos.get();
                if r != OpRef::NONE
                    && !r.is_constant()
                    && let Some(v) = defined_before.get_mut(r.raw() as usize)
                {
                    *v = true;
                }
            }
            for arg in label.getarglist() {
                let r = arg.to_opref();
                if r != OpRef::NONE
                    && !r.is_constant()
                    && let Some(v) = available.get_mut(r.raw() as usize)
                {
                    *v = true;
                }
            }
            // An appended region's live-ins reach it only through the
            // guard-fail branch that is the region's sole predecessor, and that
            // branch assigns them. Nothing the entry dispatch can land on
            // reaches a region's first read without passing it, so those ids
            // are dead until written here. Treating them as live would reserve
            // one frozen-frame slot per region live-in at every resumable
            // label, and the resume loader would reload a value the guard
            // overwrites before anything reads it.
            for region in regions {
                if region.ops_start <= label_pos {
                    continue;
                }
                for &id in &region.inputarg_ids {
                    if let Some(v) = available.get_mut(id as usize) {
                        *v = true;
                    }
                }
            }

            let mut missing = Vec::new();
            let mut bad = false;
            for op in &ops[label_pos + 1..] {
                let mut reads: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
                if let Some(failargs) = op.getfailargs() {
                    reads.extend(failargs.iter().map(|a| a.to_opref()));
                }
                for r in reads {
                    if r == OpRef::NONE || r.is_constant() {
                        continue;
                    }
                    let id = r.raw() as usize;
                    if !available.get(id).copied().unwrap_or(false) {
                        if !defined_before.get(id).copied().unwrap_or(false) {
                            bad = true;
                            continue;
                        }
                        missing.push(r);
                        if let Some(v) = available.get_mut(id) {
                            *v = true;
                        }
                    }
                }
                let r = op.pos.get();
                if r != OpRef::NONE
                    && !r.is_constant()
                    && let Some(v) = available.get_mut(r.raw() as usize)
                {
                    *v = true;
                }
            }
            per_label.push(missing);
            uncapturable.push(bad);
        }

        let mut capture_by_id = vec![None; num_vars as usize];
        let mut captured_refs = Vec::new();
        let mut scalar_slots = 0usize;
        let mut ref_slots = 0usize;
        for &r in per_label.iter().flatten() {
            let id = r.raw() as usize;
            if capture_by_id[id].is_some() {
                continue;
            }
            let storage = if ref_values.contains(r) {
                captured_refs.push(r);
                let slot = LabelCaptureStorage::RefSlot(ref_slots);
                ref_slots += 1;
                slot
            } else {
                let slot = LabelCaptureStorage::ValueSlot(normal_value_slots + scalar_slots);
                scalar_slots += 1;
                slot
            };
            capture_by_id[id] = Some(storage);
        }

        Self {
            per_label,
            uncapturable,
            capture_by_id,
            captured_refs,
            scalar_slots,
            ref_slots,
        }
    }

    fn storage(&self, r: OpRef) -> Option<LabelCaptureStorage> {
        self.capture_by_id.get(r.raw() as usize).copied().flatten()
    }

    fn shortage(&self, frame: FrameGeometry) -> Option<super::FrameShortage> {
        if self.ref_slots > frame.label_ref_slots {
            return Some(super::FrameShortage::new(
                super::FrameShortageKind::LabelResumeRefSlots,
                self.ref_slots,
                frame.label_ref_slots,
            ));
        }
        for storage in self.capture_by_id.iter().flatten() {
            match storage {
                LabelCaptureStorage::ValueSlot(slot) if *slot >= frame.value_slots => {
                    return Some(super::FrameShortage::new(
                        super::FrameShortageKind::LabelResumeCaptureSlots,
                        slot + 1,
                        frame.value_slots,
                    ));
                }
                LabelCaptureStorage::RefSlot(slot) if *slot >= frame.label_ref_slots => {
                    return Some(super::FrameShortage::new(
                        super::FrameShortageKind::LabelResumeCaptureSlots,
                        slot + 1,
                        frame.label_ref_slots,
                    ));
                }
                LabelCaptureStorage::ValueSlot(_) | LabelCaptureStorage::RefSlot(_) => {}
            }
        }
        None
    }

    fn supported_by(&self, frame: FrameGeometry) -> bool {
        self.shortage(frame).is_none()
    }

    fn frame_offset(&self, storage: LabelCaptureStorage, frame: FrameGeometry) -> u64 {
        match storage {
            LabelCaptureStorage::ValueSlot(slot) => slot as u64 * SLOT_SIZE,
            LabelCaptureStorage::RefSlot(slot) => {
                frame.home_slot_base + (frame.ordinary_home_slots() + slot) as u64 * SLOT_SIZE
            }
        }
    }
}

/// Number of Ref-home slots a trace with these `inputargs`/`ops` reserves,
/// matching the `num_ref_homes` [`build_wasm_module`] returns. Lets a CA-arena
/// caller size the callee frame and the GC walker for a (wider) bridge's home
/// region before codegen runs.
pub fn count_ref_homes(inputargs: &[InputArg], ops: &[Op]) -> usize {
    // This pre-sizing query is used for CA bridges before `CaParams` exists, so
    // count CALL_ASSEMBLER as a collecting position to match CA codegen.
    let resume = LabelResumeData::collect(inputargs, ops);
    RefHomes::collect(inputargs, ops, true, &resume.captured_refs, &[]).len()
}

/// Number of high GC-rooted homes reserved exclusively for LABEL live-ins.
pub fn label_ref_capture_slots(inputargs: &[InputArg], ops: &[Op]) -> usize {
    LabelResumeData::collect(inputargs, ops).ref_slots
}

/// First free value position — one past the highest id any value reference in
/// the trace occupies (input args, op results, and every op argument, including
/// a folded value the constants pool alone binds).
/// `majit_gc::rewrite::remove_ref_constants` numbers the
/// `LoadFromGcTable` results it emits from here upward, so the operand
/// numbering the optimizer produced stays untouched. Same id set
/// `collect_guards_and_vars` sizes `num_vars` from, so the loads land inside
/// the locals the function declares.
pub fn next_value_pos(inputargs: &[InputArg], ops: &[Op]) -> u32 {
    collect_guards_and_vars(inputargs, ops).1
}

/// Positional frame slots required for a token's inputs and guard spills.
/// Slot zero is the fail index; the returned count therefore also gives the
/// first free slot for the call trampoline.
///
/// The GUARD_VALUE counter slot is reserved unconditionally, including for a
/// trace whose own guards spill nothing. A bridge runs in its source token's
/// frame, whose offsets froze when that token was compiled, and `compile_bridge`
/// refuses a bridge whose `frame_value_slots` exceeds `source_frame.value_slots`
/// — a refusal the guard descr makes permanent, so the guard blackholes
/// for the rest of the run. Reserving only when THIS trace spills would let a
/// loop with no GUARD_VALUE freeze a frame one slot too narrow for the first
/// bridge that promotes a value, which is the ordinary way a bridge acquires
/// one. Upstream never faces the question: `regalloc.py prepare_op_guard_value`
/// names a slot in the register save area `_push_all_regs_to_frame` writes at
/// every exit, so a slot always exists and no frame is ever sized for it.
fn normal_frame_value_slots(inputargs: &[InputArg], ops: &[Op]) -> usize {
    let (guards, _) = collect_guards_and_vars(inputargs, ops);
    let max_fail_args = guards
        .iter()
        .map(|g| live_fail_arg_extent(g.meta_descr.as_ref(), g.fail_arg_refs.len()))
        .max()
        .unwrap_or(0);
    let value_area = max_fail_args.max(inputargs.len());
    1 + value_area + 1
}

/// The trace-wide GUARD_VALUE counter slot, or `None` when no guard needs one.
///
/// The first slot past the value area every exit writes into, so it is free in
/// every exit's layout, and `normal_frame_value_slots` reserves it.
fn counter_slot(inputargs: &[InputArg], ops: &[Op]) -> Option<usize> {
    let (guards, _) = collect_guards_and_vars(inputargs, ops);
    if guards.iter().all(|g| g.counter_value_spill.is_none()) {
        return None;
    }
    let max_fail_args = guards
        .iter()
        .map(|g| live_fail_arg_extent(g.meta_descr.as_ref(), g.fail_arg_refs.len()))
        .max()
        .unwrap_or(0);
    Some(max_fail_args.max(inputargs.len()))
}

pub fn frame_value_slots(inputargs: &[InputArg], ops: &[Op]) -> usize {
    normal_frame_value_slots(inputargs, ops) + LabelResumeData::collect(inputargs, ops).scalar_slots
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

/// wasm emission's complete SETFIELD_GC write-barrier gate.  The import census
/// deliberately keeps using [`write_barrier_base`]: it must remain an
/// un-elided over-approximation so an arm that emits a barrier always has the
/// `jit_call` import available.  Unlike SETARRAYITEM_GC, a SETFIELD_GC can
/// carry a non-pointer field descriptor even when its value has a Ref home
/// (the ForceToken layout is one such case).  rewrite.rs's
/// `handle_write_barrier_setfield` rejects that store because the collector
/// does not trace the field.
fn emitted_write_barrier_base(op: &Op, ref_values: &RefValues) -> Option<OpRef> {
    let base = write_barrier_base(op, ref_values)?;
    if op.opcode == OpCode::SetfieldGc
        && !op
            .getdescr()
            .and_then(|d| d.as_field_descr().map(|fd| fd.is_pointer_field()))
            .unwrap_or(false)
    {
        return None;
    }
    Some(base)
}

/// Pre-pass `SameAsI`/`SameAsR` forwarding edges by result value id.  The
/// wasm backend materializes these ops, but rewrite.py keys its applied-barrier
/// set through their forwarded box identity.
fn same_as_forwardings(ops: &[Op], num_vars: u32) -> Vec<Option<OpRef>> {
    let mut forwardings = vec![None; num_vars as usize];
    for op in ops {
        if !matches!(op.opcode, OpCode::SameAsI | OpCode::SameAsR) {
            continue;
        }
        let result = op.pos.get();
        if result == OpRef::NONE || result.is_constant() {
            continue;
        }
        if let Some(slot) = forwardings.get_mut(result.raw() as usize) {
            *slot = Some(op.arg(0).to_opref());
        }
    }
    forwardings
}

/// Follow a `SameAsI`/`SameAsR` forwarding chain to its fixed point.  The
/// bounded walk also makes malformed cyclic forwarding terminate.
fn resolve_same_as_forwarding(base: OpRef, forwardings: &[Option<OpRef>]) -> OpRef {
    let mut current = base;
    for _ in 0..forwardings.len() {
        if current == OpRef::NONE || current.is_constant() {
            break;
        }
        let Some(next) = forwardings.get(current.raw() as usize).copied().flatten() else {
            break;
        };
        if next == current {
            break;
        }
        current = next;
    }
    current
}

/// `llsupport/gc.py WriteBarrierDescr` as the emitted barrier reads it, paired
/// with the addresses of the two helpers its arms call.
///
/// A zero `cards_set` is the collector saying it has no cards, which is the
/// gate `x86/assembler.py _write_barrier_fastpath` spells as
/// `if array and descr.jit_wb_cards_set`; the dynasm backends read the same
/// field off the same descriptor.
#[derive(Clone, Copy)]
pub struct WriteBarrierHelpers {
    /// `wasm_jit_write_barrier`, the `remember_young_pointer` entry point.
    pub fn_ptr: i64,
    /// `wasm_jit_write_barrier_from_array`, the
    /// `jit_remember_young_pointer_from_array` entry point.
    pub array_fn_ptr: i64,
    /// `jit_wb_if_flag_byteofs`: where the flag byte sits relative to the
    /// object pointer. Negative, because the header precedes the object.
    pub flag_byteofs: i32,
    /// `jit_wb_if_flag_singlebyte`.
    pub if_flag: u8,
    /// `jit_wb_cards_set_singlebyte`.
    pub cards_set: u8,
    /// `jit_wb_card_page_shift`.
    pub card_page_shift: u32,
}

impl WriteBarrierHelpers {
    /// Take the geometry from a collector's descriptor. The addresses are the
    /// backend's own exported helpers, so they are supplied separately.
    pub fn new(fn_ptr: i64, array_fn_ptr: i64, descr: &majit_gc::WriteBarrierDescr) -> Self {
        Self {
            fn_ptr,
            array_fn_ptr,
            flag_byteofs: descr.jit_wb_if_flag_byteofs,
            if_flag: descr.jit_wb_if_flag_singlebyte,
            cards_set: descr.jit_wb_cards_set_singlebyte as u8,
            card_page_shift: descr.jit_wb_card_page_shift,
        }
    }

    /// The geometry the running collector advertises, for a caller that has no
    /// descriptor in hand.
    pub fn for_current_gc(fn_ptr: i64, array_fn_ptr: i64) -> Self {
        Self::new(
            fn_ptr,
            array_fn_ptr,
            &majit_gc::WriteBarrierDescr::for_current_gc(),
        )
    }

    /// The `TEST8` mask: the TRACK_YOUNG_PTRS byte, widened to also catch
    /// CARDS_SET when this store can mark a card, so one test covers both
    /// arms (`_write_barrier_fastpath`: `mask = jit_wb_if_flag_singlebyte |
    /// -0x80`).
    fn flag_mask(&self, card_marking: bool) -> i32 {
        let mask = if card_marking {
            self.if_flag | self.cards_set
        } else {
            self.if_flag
        };
        i32::from(mask)
    }
}

/// `rewrite.py RewriteState._known_lengths`: the constant length a `NEW_ARRAY`
/// in this trace gave its result.
///
/// `emit_label` clears the map, because past a merge point the length is only
/// known on the path that allocated. A collecting operation does not clear it —
/// an array's length outlives a collection.
#[derive(Default)]
struct KnownArrayLengths(indexmap::IndexMap<OpRef, usize>);

impl KnownArrayLengths {
    /// `handle_new_array`: `if isinstance(v_length, ConstInt)`, keyed on the
    /// allocation's own result.
    fn observe(&mut self, op: &Op, constants: &indexmap::IndexMap<u32, i64>) {
        match op.opcode {
            OpCode::Label => self.0.clear(),
            OpCode::NewArray | OpCode::NewArrayClear => {
                if let Some(length) = const_operand_value(constants, op.arg(0).to_opref())
                    && let Ok(length) = usize::try_from(length)
                {
                    self.0.insert(op.pos.get(), length);
                }
            }
            _ => {}
        }
    }

    /// `known_length(op, default)`.
    fn get(&self, base: OpRef, default: usize) -> usize {
        self.0.get(&base).copied().unwrap_or(default)
    }
}

/// The element index a card-marking barrier needs, or `None` for a store that
/// takes the plain barrier.
///
/// `rewrite.py gen_write_barrier_array` reaches for `COND_CALL_GC_WB_ARRAY`
/// only when the collector has cards and the array is not statically known to
/// be short: a short array is cheaper to remember whole than to card-mark index
/// by index. SETFIELD_GC carries no index and never card-marks.
fn write_barrier_card_index(
    op: &Op,
    wb: &WriteBarrierHelpers,
    base_key: OpRef,
    known_lengths: &KnownArrayLengths,
) -> Option<OpRef> {
    /// `gen_write_barrier_array`'s own `LARGE`.
    const LARGE: usize = 130;
    if wb.cards_set == 0 || wb.array_fn_ptr == 0 {
        return None;
    }
    if known_lengths.get(base_key, LARGE) < LARGE {
        return None;
    }
    matches!(
        op.opcode,
        OpCode::SetarrayitemGc | OpCode::SetinteriorfieldGc
    )
    .then(|| op.arg(1).to_opref())
}

/// Emit a store write barrier unless the base's forwarded value already has
/// one on this path. The emitted barrier still receives the store's own base.
#[allow(clippy::too_many_arguments)]
fn emit_write_barrier_if_needed(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    jit_call_idx: Option<u32>,
    residual_type_base: Option<u32>,
    wb: &WriteBarrierHelpers,
    op: &Op,
    base: Option<OpRef>,
    same_as_forwardings: &[Option<OpRef>],
    wb_applied: &mut indexmap::IndexSet<OpRef>,
    known_lengths: &KnownArrayLengths,
) {
    let Some(base) = base else {
        return;
    };
    // rewrite.py `handle_write_barrier_setfield` and
    // `handle_write_barrier_setarrayitem` both open with the same
    // `write_barrier_applied(val)` test, before either picks an arm.
    let wb_key = resolve_same_as_forwarding(base, same_as_forwardings);
    if wb_applied.contains(&wb_key) {
        return;
    }
    let card_index = write_barrier_card_index(op, wb, wb_key, known_lengths);
    emit_write_barrier(
        sink,
        constants,
        value_types,
        jit_call_idx,
        residual_type_base,
        wb,
        base,
        card_index,
    );
    // rewrite.rs's `gen_write_barrier`: remember only after the barrier has
    // been emitted. `gen_write_barrier_array` remembers nothing at all, since
    // a card records one index and the next store may name another.
    if card_index.is_none() {
        wb_applied.insert(wb_key);
    }
}

/// Push the barrier flag byte, the operand of `TEST8 [obj + jit_wb_if_flag_byteofs]`.
fn emit_load_wb_flag_byte(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    wb: &WriteBarrierHelpers,
    base_ref: OpRef,
) {
    emit_resolve(sink, constants, value_types, base_ref);
    sink.i32_wrap_i64();
    sink.i32_const(wb.flag_byteofs);
    sink.i32_add();
    sink.i32_load8_u(memarg(0, 0));
}

/// Push `index >> card_page_shift`, the card bit's index.
fn emit_push_card_bitindex(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    wb: &WriteBarrierHelpers,
    index: OpRef,
) {
    emit_resolve(sink, constants, value_types, index);
    sink.i32_wrap_i64();
    if wb.card_page_shift != 0 {
        sink.i32_const(wb.card_page_shift as i32);
        sink.i32_shr_u();
    }
}

/// Push the address `incminimark.py get_card` computes:
/// `obj - HEADER + ~(bitindex >> 3)`.
///
/// `WriteBarrierSlowPath` builds it in the same order — shift, `NOT`, subtract
/// the header, add the base — so that only the last term needs the object.
fn emit_push_card_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    wb: &WriteBarrierHelpers,
    base_ref: OpRef,
    index: OpRef,
) {
    emit_push_card_bitindex(sink, constants, value_types, wb, index);
    sink.i32_const(3);
    sink.i32_shr_u();
    sink.i32_const(-1);
    sink.i32_xor();
    sink.i32_const(majit_gc::header::GcHeader::SIZE as i32);
    sink.i32_sub();
    emit_resolve(sink, constants, value_types, base_ref);
    sink.i32_wrap_i64();
    sink.i32_add();
}

/// `*get_card(obj, bitindex >> 3) |= 1 << (bitindex & 7)`.
///
/// The address is built twice rather than parked in a local: every term is
/// pure arithmetic, which the guest optimizer both folds and commons, while a
/// local would have to be carved out of the `UintMulHigh` scratch pool.
///
/// `remember_young_pointer_from_array2` returns early when the bit is already
/// set; the inlined form ORs unconditionally, exactly as
/// `WriteBarrierSlowPath` does.
fn emit_inline_card_mark(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    wb: &WriteBarrierHelpers,
    base_ref: OpRef,
    index: OpRef,
) {
    emit_push_card_addr(sink, constants, value_types, wb, base_ref, index);
    emit_push_card_addr(sink, constants, value_types, wb, base_ref, index);
    sink.i32_load8_u(memarg(0, 0));
    sink.i32_const(1);
    emit_push_card_bitindex(sink, constants, value_types, wb, index);
    sink.i32_const(7);
    sink.i32_and();
    sink.i32_shl();
    sink.i32_or();
    sink.i32_store8(memarg(0, 0));
}

/// Call a one-arg `(i64)->i64` barrier helper and drop the dummy 0 result.
fn emit_wb_helper_call(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    residual_type_base: u32,
    fn_ptr: i64,
    base_ref: OpRef,
) {
    emit_resolve(sink, constants, value_types, base_ref);
    sink.i32_const(fn_ptr as i32);
    sink.call_indirect(0, residual_type_base + 1);
    sink.drop();
}

/// Emit a write-barrier check on `base_ref` before a ref-storing field/array
/// store, standing in for the `COND_CALL_GC_WB` the native GC rewrite pass
/// inserts.
///
/// With the residual type family declared (`residual_type_base`), this is
/// `_write_barrier_fastpath`: one test of the flag byte, and only a flagged
/// object enters the body. A `card_index` store then follows
/// `WriteBarrierSlowPath` — CARDS_SET already armed marks the card inline,
/// otherwise `jit_remember_young_pointer_from_array` runs and the same test
/// decides again on its return. Everything else calls `wasm_jit_write_barrier`.
///
/// Operand-stack-neutral: every push is consumed by a store, the call, or the
/// result drop.
#[allow(clippy::too_many_arguments)]
fn emit_write_barrier(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    jit_call_idx: Option<u32>,
    residual_type_base: Option<u32>,
    wb: &WriteBarrierHelpers,
    base_ref: OpRef,
    card_index: Option<OpRef>,
) {
    if let Some(base) = residual_type_base {
        emit_load_wb_flag_byte(sink, constants, value_types, wb, base_ref);
        sink.i32_const(wb.flag_mask(card_index.is_some()));
        sink.i32_and();
        sink.if_(BlockType::Empty);
        match card_index {
            Some(index) => {
                emit_load_wb_flag_byte(sink, constants, value_types, wb, base_ref);
                sink.i32_const(i32::from(wb.cards_set));
                sink.i32_and();
                sink.if_(BlockType::Empty);
                emit_inline_card_mark(sink, constants, value_types, wb, base_ref, index);
                sink.else_();
                emit_wb_helper_call(
                    sink,
                    constants,
                    value_types,
                    base,
                    wb.array_fn_ptr,
                    base_ref,
                );
                emit_load_wb_flag_byte(sink, constants, value_types, wb, base_ref);
                sink.i32_const(i32::from(wb.cards_set));
                sink.i32_and();
                sink.if_(BlockType::Empty);
                emit_inline_card_mark(sink, constants, value_types, wb, base_ref, index);
                sink.end();
                sink.end();
            }
            None => {
                emit_wb_helper_call(sink, constants, value_types, base, wb.fn_ptr, base_ref);
            }
        }
        sink.end();
        return;
    }
    // Host-trampoline fallback: the call area carries the base and nothing
    // else, so this arm always takes the plain barrier, which is
    // `gen_write_barrier_array`'s own fall-back case. The helper re-checks the
    // flag itself.
    let Some(jit_call) = jit_call_idx else {
        return;
    };
    // func_ptr = wasm_jit_write_barrier
    emit_call_area_addr(sink);
    sink.i64_const(wb.fn_ptr);
    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
    // num_args = 1 (the trampoline reflects arity from the wasm signature;
    // written for protocol symmetry with the alloc/call paths)
    emit_call_area_addr(sink);
    sink.i64_const(1);
    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
    // arg0 = base object pointer
    emit_call_area_addr(sink);
    emit_resolve(sink, constants, value_types, base_ref);
    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS));
    // call trampoline; void result ignored
    emit_jit_call(sink, jit_call);
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
    fn collect_with_regions(
        inputargs: &[InputArg],
        ops: &[Op],
        regions: &[InlinedRegionSpan],
    ) -> Self {
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
        // An appended region's live-ins are written by the guard-fail branch
        // that is the region's sole predecessor, and that branch jumps straight
        // into the region. Their entry in the merged input-arg list would
        // otherwise date them to trace entry, making them live across every
        // collecting call in the owner's body: each would take a Ref home and
        // be reloaded there on every iteration, for a value nothing in the
        // owner reads. Date them to the region instead, so they stay live
        // across the region's own collect points and nowhere else.
        for region in regions {
            let defined_at = region.ops_start as i32 - 1;
            for &id in &region.inputarg_ids {
                if let Some(d) = def_pos.get_mut(id as usize) {
                    *d = defined_at;
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

    /// Index of the last op that reads `raw` (arg or fail arg), or `-1` when
    /// nothing reads it. `regalloc.py` spells this `Lifetime.last_usage`.
    fn last_use(&self, raw: u32) -> i32 {
        self.last_use.get(raw as usize).copied().unwrap_or(-1)
    }
}

/// Whether a call has to be followed by the frame and home reloads.
///
/// callbuilder.py splits this in two: `emit_no_collect` prepares the
/// arguments and calls, while `emit` additionally pushes a gcmap and reloads a
/// possibly-forwarded frame afterwards. Without a gcmap the collector cannot
/// reach the home slots, so a callee that cannot collect leaves the JitFrame
/// and every home exactly where they were and the reload only re-reads what
/// the locals already hold. `x86/assembler.py:2205-2209` takes the same
/// decision from the same bit, as does the cranelift residual call emission.
///
/// Two families keep their reloads whatever the effect info says, because
/// upstream pushes their gcmap without ever consulting it:
///
/// - `CALL_RELEASE_GIL` — `x86/assembler.py:2200-2203` dispatches to
///   `emit_call_release_gil` before the `check_can_collect()` test, and
///   `push_gcmap_for_call_release_gil` is unconditional; `emit`'s own
///   docstring reads "not for CALL_RELEASE_GIL". The bit describes the callee,
///   while another thread may collect for the span the GIL is released.
/// - `COND_CALL_VALUE` — `x86/regalloc.py:952-1011` builds the gcmap with
///   `get_gcmap()` and reads no effect info at all.
///
/// A call carrying no call descr also keeps its reloads: this narrows a
/// conservative answer where the effect info is there to narrow it, and never
/// widens one.
fn call_can_collect(op: &Op) -> bool {
    if matches!(
        op.opcode,
        OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilN
            | OpCode::CallReleaseGilF
            | OpCode::CondCallValueI
            | OpCode::CondCallValueR
    ) {
        return true;
    }
    op.with_call_descr(|descr| descr.get_extra_info().check_can_collect())
        .unwrap_or(true)
}

/// Static collecting-call positions whose gcmap-visible homes may be forwarded.
/// Every `is_malloc` op (the whole `New..=Newunicode` range, string allocations
/// included) routes through a collecting allocator. A residual call earns a
/// position only where [`call_can_collect`] admits it, which is the predicate
/// the reload side already applies: a home exists so a collection can forward
/// the value, so a call that cannot collect would buy a home that nothing ever
/// reloads, and its store-on-def and back-edge refresh are stores cranelift
/// does not remove. A value live across some other collecting position still
/// takes a home from that position.
fn collecting_call_positions(ops: &[Op], include_ca_collects: bool) -> Vec<usize> {
    ops.iter()
        .enumerate()
        .filter_map(|(i, op)| {
            ((op.opcode.is_call() && call_can_collect(op)) || op.opcode.is_malloc())
                .then_some(i)
                .or_else(|| {
                    (include_ca_collects && op.opcode == OpCode::CallAssemblerR).then_some(i)
                })
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
    sink: &mut PeepSink<'_, '_>,
    value_types: &ValueLocals,
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
        sink.local_set(value_types.local(raw));
    }
}

/// RPython `_reload_frame_if_necessary` (x86 `assembler.py:1369`) for wasm
/// trace bodies: a collecting direct call may have forwarded the running
/// JitFrame, while wasm local 0 still holds its old ITEMS base.
fn emit_reload_frame_if_necessary(
    sink: &mut PeepSink<'_, '_>,
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
    } else if let Some(base) = residual_type_base.filter(|_| ca_reload_fn_ptr != 0) {
        sink.i32_const(ca_reload_fn_ptr as i32);
        sink.call_indirect(0, base);
        sink.i32_wrap_i64();
        sink.local_set(0);
    } else {
        // No reload: either the trampoline path, which assumes a non-moving
        // frame because its scratch writes use local 0, or an embedder whose
        // host entry runs traces on a frame the shadow stack does not describe
        // — it published no reload helper, and reloading from a shadow stack
        // that never held this frame would install an unrelated one.
    }
}

/// CA-arm-only variant of [`emit_reload_frame_if_necessary`]. The direct CA
/// configuration owns an inline shadow-stack top cell; all other call sites
/// retain their pre-existing helper reload.
fn emit_reload_ca_frame_if_necessary(
    sink: &mut PeepSink<'_, '_>,
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
fn emit_ca_reload_top(sink: &mut PeepSink<'_, '_>, top_addr: u32) {
    sink.i32_const(top_addr as i32);
    sink.i32_load(mem32(0));
    sink.i32_const(4);
    sink.i32_sub();
    sink.i32_load(mem32(0));
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_add();
}

/// While a CA callee is pushed, its caller's `jf_ptr` is `top[-3 * WORD]`.
fn emit_ca_reload_caller(sink: &mut PeepSink<'_, '_>, top_addr: u32) {
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
    sink: &mut PeepSink<'_, '_>,
    value_types: &ValueLocals,
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
        sink.local_set(value_types.local(arg.raw()));
    }
}

/// Information about a guard exit collected during pre-scan.
/// The live-position mask a guard's bridge inputargs were filtered by.
///
/// `pyjitpl.rs initialize_state_from_guard_failure` builds the bridge history
/// from `rd_locs`: a position is live when its entry is not `0xFFFF`, and a
/// descr whose `rd_locs` has not been sized to the fail-arg list (a synthetic
/// one that never reached codegen) declares every position live. Any arity a
/// backend compares against `bridge.inputargs.len()` has to come from that same
/// table — `OpRef::is_none()` is this backend's own IR-level hole set and is a
/// different mask, so counting with it refuses bridges whose arity was fine.
pub fn live_fail_arg_mask(meta_descr: Option<&majit_ir::DescrRef>, n: usize) -> Vec<bool> {
    match meta_descr.and_then(|d| d.as_fail_descr()) {
        Some(fd) if fd.rd_locs().len() == n => {
            fd.rd_locs().iter().map(|&pos| pos != 0xFFFF).collect()
        }
        _ => vec![true; n],
    }
}

/// How many of a guard's fail-arg positions reach its bridge as inputargs.
pub fn live_fail_arg_count(meta_descr: Option<&majit_ir::DescrRef>, n: usize) -> usize {
    live_fail_arg_mask(meta_descr, n)
        .iter()
        .filter(|l| **l)
        .count()
}

/// One past a guard's highest LIVE fail-arg position: the frame slots its exit
/// has to write, and therefore the width the frozen layout has to hold.
///
/// A guard keeps its fail arguments in their *logical* resume positions here —
/// `optimizeopt/mod.rs` hands this backend an identity-with-holes `rd_locs` and
/// `emit_guard_fail_args_spill` writes position `i` into slot `i` — so the
/// width is a property of the numbering, not of how many values are live.
/// `optimizer.py:732` keeps one `ResumeDataLoopMemo` per `Optimizer`, so a
/// guard numbered late in a trace carries positions every earlier guard filled;
/// `resume.py:511 _invalidation_needed` only clears that cache once a guard
/// passes `failargs_limit // 2` live boxes, which a trace can stay under while
/// its logical width keeps growing.
///
/// Nothing reads a hole: its `rd_locs` entry is `0xFFFF`, so
/// `initialize_state_from_guard_failure` never asks for the slot, and
/// `emit_resolve` only spills a zero placeholder into it. Holes past the last
/// live position therefore cost slots that no reader can observe, which on this
/// backend is not free — a frame's offsets freeze when its token is compiled
/// and `compile_bridge` declines a later bridge that does not fit them.
pub fn live_fail_arg_extent(meta_descr: Option<&majit_ir::DescrRef>, n: usize) -> usize {
    live_fail_arg_mask(meta_descr, n)
        .iter()
        .rposition(|&live| live)
        .map_or(0, |i| i + 1)
}

/// Whether a guard's live fail-arg positions ARE the positional exit slots a
/// frame bridge entry reads.
///
/// `emit_guard_fail_args_spill` writes fail argument `i` into slot `i`, holes
/// included, because the deadframe readers index that same logical layout.
/// `initialize_state_from_guard_failure` instead drops the holes, so bridge
/// input `k` names the k-th LIVE position — the two orders coincide only while
/// every hole trails the last live position. `rebuild_faillocs_from_descr` is
/// where a location per live position is recovered; the frame entry in
/// `build_function` has no such table and can only read slot `k`, so a caller
/// that cannot honour this must decline the bridge.
///
/// A descr whose `rd_locs` was never sized to its fail-arg list declares every
/// position live — the reading `live_fail_arg_mask` takes — and its positions
/// are then the slots by construction.
pub fn frame_entry_reads_live_positions(
    fail_descr: &dyn majit_ir::FailDescr,
    bridge_inputs: usize,
) -> bool {
    let n = fail_descr.fail_arg_types().len();
    let locs = fail_descr.rd_locs();
    if locs.len() != n {
        return bridge_inputs == n;
    }
    let live = locs.iter().filter(|&&pos| pos != 0xFFFF).count();
    bridge_inputs == live && locs[..live].iter().all(|&pos| pos != 0xFFFF)
}

pub struct GuardExit {
    pub fail_index: u32,
    pub fail_arg_refs: Vec<OpRef>,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// The GUARD_VALUE operand this exit parks in the trace's counter slot,
    /// for `make_a_counter_per_value`. `None` when the guard is not a
    /// GUARD_VALUE, or when its operand is already one of the fail arguments
    /// and so already has a slot. See `counter_value_spill`.
    pub counter_value_spill: Option<OpRef>,
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
///  * `llsupport/gc.py` — `get_translated_info_for_typeinfo`
///  * `llsupport/gc.py` — `get_translated_info_for_guard_is_object`
///  * `x86/assembler.py` — `cpu.subclassrange_min_offset`
///  * `x86/assembler.py:1971-1974` — constant-time
///    `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
///
/// The default sets `supports_guard_gc_type = false`, matching
/// `AbstractCPU.supports_guard_gc_type` in `backend/model.py`; the
/// codegen arms assert this flag before reading any other field.
#[derive(Clone, Default)]
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
    /// `cpu.subclassrange_min_offset` (x86/assembler.py).
    pub subclassrange_min_offset: usize,
    /// `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
    /// looked up by constant classptr. Empty when
    /// `supports_guard_gc_type == false`.
    pub subclass_ranges: HashMap<i64, (i64, i64)>,
}

/// Check if any op in the trace is a CALL variant.
/// Whether an eligible residual CALL may be lowered to a direct in-module
/// `call_indirect` into the callee's `__indirect_function_table` slot, instead
/// of routing through the `jit_call` host trampoline (guest→host→guest
/// reflection + arg marshalling).
///
/// The lowering takes the callee's wasm type from the IR alone: word-typed
/// arguments and result become `(i64×n) -> i64`, so the static type is fixed by
/// the arity. That is a claim about the embedding language's residual helpers,
/// and one the IR cannot check — a helper declared to take a pointer has an
/// `i32` parameter on wasm32, and `call_indirect` type-checks its callee on
/// every call, so a call lowered this way traps instead of reaching a helper
/// whose real signature is narrower.
///
/// [`ResidualCallAbi`] is how an embedder says which of the two it is.
///
/// Read once per emitted call, so it must be set before the first compile.
static RESIDUAL_CALL_ABI: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// How faithfully a residual callee's call descr describes its real wasm
/// signature.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResidualCallAbi {
    /// Every callee reachable as a residual is spelled with word-sized `i64`
    /// parameters and result and casts at its own boundary, so a descr's word
    /// type *is* the callee's wasm type. Any word-typed residual then lowers
    /// to a direct in-module `call_indirect`. The default.
    Word,
    /// Only the callees named by [`crate::set_faithful_residual_call_addrs`]
    /// are known to be spelled that way. Every other residual keeps the host
    /// trampoline, which reads the callee's declared type before calling it:
    /// slower, but correct for a callee spelled with a pointer parameter,
    /// which is narrower than a word on wasm32. Vouching for too few callees
    /// costs speed; vouching for one whose parameters are not all words costs
    /// a trap, so the safe direction is to add them one at a time.
    Vouched,
}

/// Declare which [`ResidualCallAbi`] this embedder's residual helpers satisfy.
pub fn set_residual_call_abi(abi: ResidualCallAbi) {
    let encoded = match abi {
        ResidualCallAbi::Word => 0,
        ResidualCallAbi::Vouched => 1,
    };
    RESIDUAL_CALL_ABI.store(encoded, std::sync::atomic::Ordering::Relaxed);
}

fn residual_call_abi() -> ResidualCallAbi {
    match RESIDUAL_CALL_ABI.load(std::sync::atomic::Ordering::Relaxed) {
        0 => ResidualCallAbi::Word,
        _ => ResidualCallAbi::Vouched,
    }
}

/// Whether `op`'s callee may be called with the wasm type its descr's word
/// types imply, rather than through the reflecting trampoline.
fn residual_callee_abi_is_word(op: &Op, constants: &indexmap::IndexMap<u32, i64>) -> bool {
    match residual_call_abi() {
        ResidualCallAbi::Word => true,
        ResidualCallAbi::Vouched => {
            // Only a compile-time callee can be checked against the list; a
            // register-form func pointer is a different target on every
            // execution.
            let Some(func_ptr) = op.getarglist().first().map(|arg| arg.to_opref()) else {
                return false;
            };
            func_ptr.is_constant()
                && crate::residual_call_descr_is_faithful(resolve_const_bits(constants, func_ptr))
        }
    }
}

/// If `op` is a residual CALL whose ABI is uniformly i64 (all Int/Ref args and
/// an Int/Ref result), return its argument count — eligible for a direct
/// `call_indirect` of type `(i64×n) -> i64`. `None` keeps the `jit_call`
/// trampoline: void / float / release-GIL / cond / assembler calls, a missing
/// call descr, or an arg-count/descr-shape mismatch (defensive).
///
/// This includes `CallMayForce{I,R}` when their ABI is uniformly i64: the force
/// protocol rides the frame's own data region
/// (`emit_force_bracket_before_call` before the call, `GuardNotForced` after),
/// which neither lowering touches, so a direct call is sound. Float /
/// release-GIL / cond / assembler calls and non-reflectable descrs remain on
/// the trampoline.
fn residual_call_i64_arity(op: &Op, constants: &indexmap::IndexMap<u32, i64>) -> Option<usize> {
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
    if !residual_callee_abi_is_word(op, constants) {
        return None;
    }
    Some(nargs)
}

/// Wasm parameter types and result type of a residual call lowered directly.
/// A residual callee's wasm signature taken from its call descr: the
/// parameter sequence, and the result -- `None` for a callee that returns
/// nothing, which wasm spells as an empty result list rather than a type.
type TypedResidualSig = (Vec<ValType>, Option<ValType>);

/// If `op` is a residual float CALL with only float arguments, return its wasm
/// parameter types — eligible for a direct `call_indirect` returning `f64`.
/// Float-result targets are not audited for a uniform word ABI: a `Ref` or
/// `Int` argument may actually be an `i32` pointer, such as
/// `jit_bigint_to_f64_or_inf`. `None` keeps the `jit_call` trampoline:
/// non-float / release-GIL / assembler calls, a missing call descr, a
/// non-float argument or result type, or an arg-count/descr-shape mismatch
/// (defensive).
///
/// This includes `CallMayForceF`: the force protocol rides the frame's own data
/// region (`emit_force_bracket_before_call` before the call, `GuardNotForced`
/// after), which neither lowering touches, so a direct call is sound.
fn residual_call_typed_sig(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<TypedResidualSig> {
    use OpCode::*;
    if !matches!(
        op.opcode,
        CallF | CallPureF | CallLoopinvariantF | CallMayForceF
    ) && !matches!(
        op.opcode,
        CallI | CallPureI | CallLoopinvariantI | CallMayForceI
    ) && !matches!(
        op.opcode,
        CallR | CallPureR | CallLoopinvariantR | CallMayForceR
    ) && !matches!(
        op.opcode,
        CallN | CallPureN | CallLoopinvariantN | CallMayForceN
    ) {
        return None;
    }
    // The uniform families are preferred wherever they can express the callee,
    // and the emit arm tries them first. Declining here rather than at the call
    // site keeps the four predicates disjoint, so the signature collected for an
    // op is always the signature its emit reaches -- a type collected for an op
    // the i64 family claims would declare an index nothing branches to.
    if residual_call_i64_arity(op, constants).is_some()
        || residual_call_void_word_arity(op, constants).is_some()
        || residual_call_void_true_arity(op, constants).is_some()
    {
        return None;
    }
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    if op.result_type() != cd.result_type() {
        return None;
    }
    let result = match cd.result_type() {
        Type::Float => Some(ValType::F64),
        Type::Int | Type::Ref => Some(ValType::I64),
        // A callee that returns nothing still needs its own type when a float
        // parameter puts it outside `residual_call_void_true_arity`'s uniform
        // word family.  `all_float` below is false for it, so it reaches the
        // allow-list check like every other mixed shape.
        //
        // Only a descr that records `()` names such a callee.  A void-recorded
        // descr carrying `result_size == 8` (the `make_call_descr_void_word_abi`
        // shape) names one that really returns a machine word, and an empty
        // result list is a different type from the one the callee has.  The i64
        // family `residual_call_void_word_arity` selects is where that ABI is
        // spelled; where that family declines -- a float parameter it cannot
        // carry -- the reflecting trampoline is the arm that stays correct.
        Type::Void if cd.result_size() != 0 => return None,
        Type::Void => None,
    };
    let arg_types = cd.arg_types();
    // Every argument `f64` and an `f64` result is the shipped shape and needs
    // no vouching: a float-only descr has no word parameter to be an `i32`
    // pointer in disguise. Anything else -- a word beside a float, or a word
    // result over float arguments -- is only as good as the descr, so the
    // callee has to be named by `set_faithful_residual_call_addrs`.
    let all_float = result == Some(ValType::F64) && arg_types.iter().all(|t| *t == Type::Float);
    if !all_float {
        // Only a compile-time callee can be checked against the allow-list; a
        // register-form func pointer is a different target on every execution.
        let func_ptr = op.arg(0).to_opref();
        if !func_ptr.is_constant() {
            return None;
        }
        if !crate::residual_call_descr_is_faithful(resolve_const_bits(constants, func_ptr)) {
            return None;
        }
    }
    let mut params = Vec::with_capacity(arg_types.len());
    for ty in arg_types {
        params.push(match ty {
            Type::Float => ValType::F64,
            Type::Int | Type::Ref => ValType::I64,
            Type::Void => return None,
        });
    }
    // `getarglist()[0]` is the func pointer; the call args are `[1..]`. The
    // descr's `arg_types` describes those call args, so the counts must match.
    let nargs = op.getarglist().len().saturating_sub(1);
    if params.len() != nargs {
        return None;
    }
    Some((params, result))
}

/// Void-recorded counterpart of [`residual_call_i64_arity`]: an eligible
/// void residual CALL whose descr records the dummy-word C ABI
/// (`result_size == 8`, minted by `make_call_descr_void_word_abi`) — the
/// callee is really `(i64×n) -> i64` with the result ignored, so it lowers
/// through the same i64 type family with a trailing `drop`. This includes
/// `CallMayForceN` with the word ABI: the force protocol rides the frame's own
/// data region (`emit_force_bracket_before_call` before the call,
/// `GuardNotForced` after), which neither lowering touches, so a direct call is
/// sound.
/// Float / release-GIL / cond / assembler calls and non-reflectable descrs
/// remain on the trampoline.
fn residual_call_void_word_arity(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<usize> {
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
    if !residual_callee_abi_is_word(op, constants) {
        return None;
    }
    Some(nargs)
}

/// True-void counterpart of [`residual_call_void_word_arity`]: an eligible
/// void residual CALL whose descr records a `()` result (`result_size == 0`).
/// Int/Ref-only arguments lower through the `(i64×n) -> ()` type family with
/// no result to drop. Float / release-GIL / cond / assembler calls,
/// non-reflectable descrs, and descr/operand arity mismatches remain on the
/// trampoline.
fn residual_call_void_true_arity(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<usize> {
    use OpCode::*;
    if !matches!(
        op.opcode,
        CallN | CallPureN | CallLoopinvariantN | CallMayForceN
    ) {
        return None;
    }
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    if cd.result_type() != Type::Void || cd.result_size() != 0 {
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
    if !residual_callee_abi_is_word(op, constants) {
        return None;
    }
    Some(nargs)
}

/// Arity of `op`'s in-module `(i64×n) -> i64` lowering, if it has one: an
/// eligible residual CALL (word-result or word-ABI void), a `New*`
/// allocation (the `wasm_jit_alloc*` helper targets are plain
/// `extern "C" fn(i64×n) -> i64` table entries), or a ref-storing store
/// (its `wasm_jit_write_barrier` helper takes 1 arg). All of these share
/// the i64-result residual-call type family, so one max covers them. True-void
/// residuals use a separate result family and arity census.
fn direct_helper_i64_arity(
    op: &Op,
    ref_values: &RefValues,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<usize> {
    if let Some(n) = residual_call_i64_arity(op, constants) {
        return Some(n);
    }
    if let Some(n) = residual_call_void_word_arity(op, constants) {
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

/// Whether this trace emits a host `jit_call` / `jit_call_compact` trampoline
/// invocation and therefore needs the corresponding function import.
///
/// Keep this in lockstep with the individual emission arms below: the uniform
/// i64, typed float, and true-void residual families, `New*`, and write
/// barriers are direct as far as [`RESIDUAL_CALL_ABI`] lets each one be;
/// non-uniform CALLs, an unvouched callee, and string allocation retain the
/// trampoline.
fn has_trampoline_calls(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    emit_ca: bool,
) -> bool {
    let ref_values = RefValues::collect(inputargs, ops);
    ops.iter().any(|op| match op.opcode {
        // `build_function` handles an enabled CALL_ASSEMBLER before the generic
        // CALL arm, lowering it directly to the callee-loop table slot. It
        // therefore never uses the host call area.
        opcode if opcode.is_call_assembler() && emit_ca => false,
        // No direct helper lowering — and in fact no lowering at all: a trace
        // carrying either is declined. Kept as a conservative superset.
        OpCode::Newstr | OpCode::Newunicode => true,
        // Every residual CALL uses the trampoline unless its exact lowering
        // predicate supplies an i64, typed float, or true-void helper ABI.
        _ if op.opcode.is_call() => {
            direct_helper_i64_arity(op, &ref_values, constants).is_none()
                && residual_call_typed_sig(op, constants).is_none()
                && residual_call_void_true_arity(op, constants).is_none()
        }
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
        // Every value an op reads occupies a local, whether or not the trace
        // also contains an op that produces it: constant folding and the short
        // preamble leave a folded value bound only by the constants pool, and
        // `unbound_pool_const_seeds` materializes it in the prologue. Counting
        // only op results would under-size `num_vars` for such a value and,
        // through `next_value_pos`, let `remove_ref_constants` reuse its id for
        // a `LoadFromGcTable` — whose store then lands after the read, so the
        // read returns the zero wasm initializes the local to.
        let widen = |a: OpRef, max_var: &mut u32| {
            if a != OpRef::NONE && !a.is_constant() && a.raw() + 1 > *max_var {
                *max_var = a.raw() + 1;
            }
        };
        for a in op.getarglist().iter() {
            widen(a.to_opref(), &mut max_var);
        }
        if let Some(fa) = op.getfailargs() {
            for a in fa.iter() {
                widen(a.to_opref(), &mut max_var);
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

            let meta_descr = op.getdescr();
            // `regalloc.py consider_guard_value` — stamp the per-value
            // counter here, where the native backends stamp it during guard
            // layout, so `store_guard_hashes`' `status == 0` gate
            // (`compile.py`) leaves it alone and `must_compile` hashes
            // the (guard, failing value) pair. Without it a guard whose failing
            // value never repeats accumulates in one bucket and compiles
            // another bridge every `trace_eagerness` failures, without bound.
            // The compared operand of a GUARD_VALUE is a promoted value the
            // resume re-derives, so it is almost never one of the guard's own
            // fail arguments: 0 of 16 on a synthetic polymorphic call site, 0
            // of 21 on pyre/bench/fannkuch.py. Reading its slot out of the fail
            // arguments alone therefore left the stamp unwritten on nearly
            // every GUARD_VALUE, and an unstamped guard keeps the per-guard
            // hash: every `trace_eagerness` failures compile another bridge for
            // a value that never repeats, without bound
            // (`foriter_make_function_body`: 47 bridges, 0 of them entered).
            //
            // `regalloc.py consider_guard_value` hands
            // `all_reg_indexes[x.value]` — a deadframe slot, not a fail-argument
            // position — so an operand the guard does not carry is still
            // readable. This backend's slot space is the exit's own frame
            // slots, so give such an operand one past the last fail argument
            // and spill it there (`emit_guard_fail_args_spill`);
            // `normal_frame_value_slots` reserves it and
            // `resolve_guard_value_operand` reads it back through
            // `get_value_direct`.
            // Decided off the op alone, never off the descr: the emission
            // (`emit_guard_fail_args_spill`) and the frame sizing
            // (`normal_frame_value_slots`) both read the same predicate, and a
            // guard whose descr is absent must still agree with them or the
            // parked word lands in a slot the frame never reserved.
            let counter_value_spill = counter_value_spill(op, &fail_args);
            if op.opcode == OpCode::GuardValue
                && let Some(fd) = meta_descr.as_ref().and_then(|d| d.as_fail_descr())
            {
                let arg0 = op.arg(0).to_opref();
                // The parked case is stamped after this loop, where the
                // trace-wide slot is known.
                if counter_value_spill.is_none()
                    && let Some(idx) = fail_args.iter().position(|r| *r == arg0)
                {
                    let type_tag = match fail_arg_types.get(idx) {
                        Some(Type::Ref) => majit_backend::STATUS_TY_REF,
                        Some(Type::Float) => majit_backend::STATUS_TY_FLOAT,
                        _ => majit_backend::STATUS_TY_INT,
                    };
                    fd.make_a_counter_per_value(idx as u32, type_tag);
                }
            }
            guards.push(GuardExit {
                fail_index,
                fail_arg_refs: fail_args,
                fail_arg_types,
                is_finish: op.opcode == OpCode::Finish,
                counter_value_spill,
                meta_descr,
            });
            fail_index += 1;
        }
    }

    // The parked operands share ONE slot, past every exit's fail args and past
    // the inputargs (`counter_slot`), so it can only be named once every
    // exit's width is known. `must_compile` reads the stamp back through
    // `get_value_direct`, so the slot it names has to be the slot
    // `emit_guard_fail_args_spill` writes.
    if guards.iter().any(|g| g.counter_value_spill.is_some()) {
        let value_area = guards
            .iter()
            .map(|g| live_fail_arg_extent(g.meta_descr.as_ref(), g.fail_arg_refs.len()))
            .max()
            .unwrap_or(0)
            .max(inputargs.len());
        for g in &guards {
            if let Some(operand) = g.counter_value_spill
                && let Some(fd) = g.meta_descr.as_ref().and_then(|d| d.as_fail_descr())
            {
                let type_tag = match operand.ty() {
                    Some(Type::Ref) => majit_backend::STATUS_TY_REF,
                    Some(Type::Float) => majit_backend::STATUS_TY_FLOAT,
                    _ => majit_backend::STATUS_TY_INT,
                };
                fd.make_a_counter_per_value(value_area as u32, type_tag);
            }
        }
    }

    (guards, max_var)
}

/// Number of guard/finish exits a module will need bridge-dispatch cells for.
/// Cell ownership belongs to the compiled trace, outside module generation.
pub fn guard_exit_count(inputargs: &[InputArg], ops: &[Op]) -> usize {
    collect_guards_and_vars(inputargs, ops).0.len()
}

/// Dense wasm-local assignment and type lookup for each addressed SSA value.
fn collect_value_types(
    inputargs: &[InputArg],
    ops: &[Op],
    num_vars: u32,
    first_local: u32,
) -> ValueLocals {
    ValueLocals::collect(inputargs, ops, num_vars, first_local)
}

/// Assign each Ref-typed value (input arg or op result) a dense home-slot
/// index, keyed by its value id (`raw()`), the same id its wasm local uses
/// (the dense wasm local is assigned separately). Input args and op results
/// share one value-id space (see
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
pub fn alloc_bridge_cells(num_guards: usize) -> (u32, Option<Box<[u32]>>) {
    // `Box<[u32; 0]>` has a non-null dangling `as_mut_ptr()`.  The pointer is
    // not a dispatch table, so preserve the no-dispatch representation even
    // on wasm where allocating that empty box would otherwise make the
    // epilogue load an uninitialised bridge-slot local.
    if num_guards == 0 {
        return (0, None);
    }
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

/// Parameters for the guest→guest `CALL_ASSEMBLER` `call_indirect` arm.
/// `emit_ca == false` (the default) keeps every emitted module byte-identical
/// to the pre-feature backend.
#[derive(Clone, Default)]
pub struct CaParams {
    /// Emit the dedicated `CALL_ASSEMBLER` arm.
    pub emit_ca: bool,
    /// Geometry and entry metadata, keyed by the CALL_ASSEMBLER callee token.
    /// Every entry describes exactly the JitFrame allocated for that target.
    pub targets: HashMap<u64, CaTarget>,
    /// `__indirect_function_table` slot of `wasm_ca_resume_deopt`
    /// (`lib.rs::ca_deopt_helper_slot`). When a callee `call_indirect` returns a
    /// non-finish `fail_index` (a guard deopt), the CA arm `call_indirect`s this
    /// slot to blackhole-resume the callee on the host and read back its result,
    /// instead of trapping. `0` (unset) ⇒ no helper, so `compile_bridge` declines
    /// the CA lift before reaching codegen.
    pub deopt_helper_slot: u32,
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
    /// Active-GC state for the direct CA-only inline allocation/frame path.
    /// `None` retains the helpers (including under gc_stress).
    pub inline: Option<CaInlineParams>,
}

/// Per-CALL_ASSEMBLER target dispatch baked into the corresponding wasm arm.
/// Frame geometry is deliberately not stored here: PyPy permits
/// `redirect_call_assembler` to replace a temporary callback with a deeper
/// real loop, so every mutable target field is loaded through the stable entry.
#[derive(Clone)]
pub struct CaTarget {
    /// Stable guest-memory [`WasmCaDispatchEntry`](crate::failguard::WasmCaDispatchEntry)
    /// address.  The call slot, finish index, and deopt metadata are loaded
    /// through it at runtime so pending->real install and redirects do not
    /// require patching an already-compiled wasm module.
    pub dispatch_entry: u32,
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
#[derive(Clone)]
pub struct NurseryAllocParams {
    /// Linear-memory address of the GC's `nursery_free` bump pointer.
    pub free_addr: u32,
    /// Linear-memory address of the GC's `nursery_top` limit pointer.
    pub top_addr: u32,
    /// `max_nursery_object_size` / `JIT_max_size_of_young_obj` — the exclusive
    /// `large_object` boundary, so the inline path applies strictly below it.
    pub large_threshold: usize,
    /// Type ids whose allocation is a plain bump + header write (no
    /// destructor / weakref side-list registration).
    pub plain_tids: std::collections::HashSet<u32>,
}

/// `__indirect_function_table` indices of the allocation helpers a compiled
/// trace calls for `New*` / `NewArray*`.
///
/// Two generations, picked per operation from the descr's `non_moving` flag:
/// the nursery pair is the default, the old-gen pair is for a descr whose
/// object must not move (see `majit_backend_wasm::wasm_jit_alloc_oldgen`).
/// The native backends make the same choice inside the GC rewrite pass, which
/// the wasm backend bypasses in favour of this lowering.
#[derive(Clone, Copy, Default)]
pub struct AllocHelpers {
    pub new_fn_ptr: i64,
    pub new_array_fn_ptr: i64,
    pub new_oldgen_fn_ptr: i64,
    pub new_array_oldgen_fn_ptr: i64,
}

type BuildWasmModuleOutput = (Vec<u8>, Vec<GuardExit>, usize);

/// Counts entries into an out-of-line bridge module and calls out once there
/// have been enough of them to pay for merging that bridge into its owner.
///
/// The callee only records the request — the merge itself runs after the trace
/// returns, because the host holds the driver mutably across the whole
/// compiled run.
#[derive(Clone, Copy)]
pub struct InlineTripProbe {
    /// Address of this bridge's own `u64` entry counter.
    pub counter_addr: u32,
    /// Entry count at which the callback fires — once, on equality.
    pub threshold: u64,
    /// `__indirect_function_table` index of the `(i64) -> i64` callback.
    pub trip_fn_ptr: i64,
    /// The callback's only argument: which deferred merge to install.
    pub pending_id: i64,
    /// Address of the owner's live `bridge_cells_base`, read at trip time
    /// rather than baked, because re-emitting the owner moves the array.
    /// Zero here, or a zero read out of it, leaves the cell alone.
    pub cells_base_ptr: u32,
    /// This bridge's cell in that array.
    pub dispatch_cell_index: u32,
}

/// Owned inputs for one wasm module build.  A loop retains this after its
/// first build so it can emit the same trace again without revisiting mutable
/// backend state such as the constants pool or GC-reference interning pass.
pub struct ModuleBuildInputs {
    pub inputargs: Vec<InputArg>,
    /// These are the post-intern operations.  Re-interning them would lose the
    /// already allocated GC-table base encoded by `gc_table_base`.
    pub ops: Vec<Op>,
    /// Loop-closing bridge regions emitted inside this loop's wasm function.
    /// Each is retained in its own trace's numbering; `build_wasm_module`
    /// rebases them onto a private id range before merging, because the owner
    /// and every region number their values independently from zero.
    pub inlined_bridges: Vec<InlinedBridge>,
    pub constants: indexmap::IndexMap<u32, i64>,
    pub vtable_offset: Option<usize>,
    pub classptr_to_typeid: HashMap<i64, u32>,
    pub guard_gc_type_info: GuardGcTypeInfo,
    pub alloc: AllocHelpers,
    pub wb: WriteBarrierHelpers,
    pub nursery: Option<NurseryAllocParams>,
    pub invalidated_flag_addr: u32,
    pub gc_table_base: u32,
    pub fail_index_base: u32,
    pub bridge_cells_base: u32,
    /// A bridge reached from an armed guard takes its fail values as `i64`
    /// parameters after the frame pointer. Float bits use the same i64 carrier,
    /// so a single function type per arity covers every failure signature.
    pub bridge_entry_arity: Option<usize>,
    /// Emit fixed-arity guard-to-bridge parameter tail-call arms for this module.
    pub bridge_param_dispatch: bool,
    /// Guest-memory counters baked into an armed trace-entry census module.
    /// `None` keeps the generated module byte-identical to the normal path.
    pub trace_entry_census: Option<crate::TraceEntryCensusStorage>,
    /// Entry counter and callback for a bridge whose merge into its owner is
    /// deferred until it has been crossed often enough to pay for the owner's
    /// re-emission. `None` keeps the generated module byte-identical.
    pub inline_trip: Option<InlineTripProbe>,
    pub external_jump_slot: u32,
    pub external_jump_key: u32,
    /// The target loop's `trace_wide` table slot, when it published one. The
    /// slot the host appends beside `external_jump_slot` holds a fixed-arity
    /// parameter entry, so a loop-closing JUMP can hand its args over as wasm
    /// parameters instead of writing them to frame slots the target's narrow
    /// shim would immediately read back. `0` = the target has no wide entry.
    pub external_jump_wide_slot: u32,
    pub frame: FrameGeometry,
    pub ca: CaParams,
}

/// The cross-module target of a region's closing JUMP, as `compile_bridge`
/// resolved it for the bridge the region stands in for.
///
/// The target's fixed-arity parameter entry is deliberately absent. Calling it
/// needs its wasm type declared in the calling module, and a module declares
/// that type from its OWN `external_jump_wide_slot` — which the owner of a
/// merged region does not have. The narrow entry a region tail-calls instead
/// reads the same values back out of the frame slots stored here, so the merge
/// costs that round trip and nothing else.
#[derive(Clone, Debug)]
pub struct ExternalJump {
    /// `__indirect_function_table` slot of the target loop's entry.
    pub slot: u32,
    /// Resume-at-LABEL dispatch key: `target label ordinal + 1`, or `0` when
    /// the target is not peeled.
    pub key: u32,
}

pub struct InlinedBridge {
    /// Per-trace fail index of the guard that enters this region.
    pub source_fail_index: u32,
    /// Where this region's closing JUMP goes when it names a LABEL published
    /// by ANOTHER module. `None` is the in-module case: the JUMP rebinds the
    /// owner's own loop args and lowers to a `br`.
    pub external_jump: Option<ExternalJump>,
    /// Emit this region's block outside the header `loop` and its body past
    /// that loop's `end`, rather than inside it.
    ///
    /// Forced when the source guard is in the peeled preamble, which has not
    /// entered the loop the inside blocks are opened in. It is also the only
    /// placement left to a region attaching AFTER one of those, because merging
    /// is append-only: an outside region's ops are the tail of the merged
    /// stream, and splicing anything ahead of them would renumber the exits its
    /// own sub-bridges' dispatch cells are keyed by.
    pub outside_loop: bool,
    pub trace_id: u64,
    pub inputargs: Vec<InputArg>,
    pub ops: Vec<Op>,
    /// Base of this already-interned region's GC table. Each region retains
    /// its own roots; codegen selects it by the LoadFromGcTable producer.
    pub gc_table_base: u32,
    /// The constant pool registered for this region's own trace. A pool is
    /// per-trace (`Backend::set_constants_pool` names the next compile), and
    /// its value-id keys — the folded values that have no producing op — are
    /// in that trace's numbering, so the merge rebases them with the region.
    pub constants: indexmap::IndexMap<u32, i64>,
}

/// Whether the exact operation stream emitted for `inputs` has a local loop
/// back-edge target.  An inline bridge transfers with `br`, which can only
/// target the wasm `loop` opened for that LABEL.
pub fn merged_stream_has_loop_label(inputs: &ModuleBuildInputs) -> bool {
    let mut ops = inputs.ops.clone();
    for bridge in &inputs.inlined_bridges {
        ops.extend(bridge.ops.iter().cloned());
    }
    find_loop_label_index(&ops).is_some_and(|label_idx| label_idx < inputs.ops.len())
}

/// Whether the guard at exit ordinal `fail_index` sits in the peeled preamble,
/// ahead of the loop header LABEL.
///
/// A region attaching to such a guard cannot take the loop-body placement: its
/// block would be opened inside the `loop` the preamble has not entered.
/// `build_function` gives this class its own blocks outside that loop and
/// emits their bodies after it closes, which is why the ordinal decides the
/// emission order of the merged stream.
///
/// `fail_index` is the exit ordinal within the owner's own stream — the
/// numbering `collect_guards_and_vars` assigns and `InlinedBridge` records as
/// `source_fail_index`. The merged stream appends every region after the
/// owner, so the same ordinal reads the same guard there.
pub fn source_guard_precedes_loop_label(ops: &[Op], fail_index: u32) -> bool {
    match (exit_op_index(ops, fail_index), find_loop_label_index(ops)) {
        (Some(pos), Some(label_idx)) => pos < label_idx,
        _ => false,
    }
}

/// Whether these ops carry a `GUARD_NOT_INVALIDATED`, so their validity is
/// watched through the invalidation flag of whichever module emits them.
pub fn has_invalidation_guard(ops: &[Op]) -> bool {
    ops.iter()
        .any(|op| op.opcode == OpCode::GuardNotInvalidated)
}

/// Position in `ops` of the exit with ordinal `fail_index`.
fn exit_op_index(ops: &[Op], fail_index: u32) -> Option<usize> {
    ops.iter()
        .enumerate()
        .filter(|(_, op)| op.opcode.is_guard() || op.opcode == OpCode::Finish)
        .nth(fail_index as usize)
        .map(|(pos, _)| pos)
}

impl Clone for InlinedBridge {
    fn clone(&self) -> Self {
        Self {
            source_fail_index: self.source_fail_index,
            external_jump: self.external_jump.clone(),
            outside_loop: self.outside_loop,
            trace_id: self.trace_id,
            inputargs: self
                .inputargs
                .iter()
                .map(InputArg::fresh_value_copy)
                .collect(),
            ops: self.ops.clone(),
            gc_table_base: self.gc_table_base,
            constants: self.constants.clone(),
        }
    }
}

impl Clone for ModuleBuildInputs {
    fn clone(&self) -> Self {
        Self {
            inputargs: self
                .inputargs
                .iter()
                .map(InputArg::fresh_value_copy)
                .collect(),
            ops: self.ops.clone(),
            inlined_bridges: self.inlined_bridges.clone(),
            constants: self.constants.clone(),
            vtable_offset: self.vtable_offset,
            classptr_to_typeid: self.classptr_to_typeid.clone(),
            guard_gc_type_info: self.guard_gc_type_info.clone(),
            alloc: self.alloc,
            wb: self.wb,
            nursery: self.nursery.clone(),
            invalidated_flag_addr: self.invalidated_flag_addr,
            gc_table_base: self.gc_table_base,
            fail_index_base: self.fail_index_base,
            bridge_cells_base: self.bridge_cells_base,
            bridge_entry_arity: self.bridge_entry_arity,
            bridge_param_dispatch: self.bridge_param_dispatch,
            trace_entry_census: self.trace_entry_census,
            inline_trip: self.inline_trip,
            external_jump_slot: self.external_jump_slot,
            external_jump_key: self.external_jump_key,
            external_jump_wide_slot: self.external_jump_wide_slot,
            frame: self.frame,
            ca: self.ca.clone(),
        }
    }
}

/// One past the highest value id `inputargs`/`ops` define or read. Mirrors the
/// `max_var` half of `collect_guards_and_vars` without its guard collection,
/// which stamps per-value counters onto guard descrs and must run once only.
fn value_id_end(inputargs: &[InputArg], ops: &[Op]) -> u32 {
    let mut end: u32 = 0;
    let widen = |r: OpRef, end: &mut u32| {
        if r != OpRef::NONE && !r.is_constant() && r.raw() + 1 > *end {
            *end = r.raw() + 1;
        }
    };
    for ia in inputargs {
        if ia.index + 1 > end {
            end = ia.index + 1;
        }
    }
    for op in ops {
        widen(op.pos.get(), &mut end);
        for a in op.getarglist().iter() {
            widen(a.to_opref(), &mut end);
        }
        if let Some(fa) = op.getfailargs() {
            for a in fa.iter() {
                widen(a.to_opref(), &mut end);
            }
        }
    }
    end
}

/// Move every value id a region defines or reads up by `offset`, returning the
/// rebased region and the width of the id range it now occupies.
///
/// The owner trace and each region are separately recorded traces, so both
/// number their values from zero and their ids overlap. A region is entered by
/// `local.set`ting the id each of its input args carries
/// (`emit_guard_inline_bridge_move`) and leaves through the loop header, so an
/// id it shares with an owner value that is live across the back edge
/// overwrites that value for every following iteration. Rebasing onto a
/// disjoint range is what makes the merged stream's single local namespace
/// sound.
///
/// `TempVar` ids live in a reserved high strip and constants in their own
/// namespace; neither indexes a value local, so both pass through unchanged.
fn rebase_region_value_ids(
    bridge: &InlinedBridge,
    offset: u32,
) -> Result<(InlinedBridge, u32), BackendError> {
    use majit_ir::operand::Operand;

    let shift = |r: OpRef| -> OpRef {
        if r.is_none() || r.is_constant() || r.is_temp_var() {
            r
        } else {
            r.with_raw(r.raw() + offset)
        }
    };

    let width = value_id_end(&bridge.inputargs, &bridge.ops);
    // `with_raw` keeps the variant, but the emitters classify by raw payload
    // (`OpRef::raw_is_constant`), so an id shifted to or past the limit reads
    // as a constant and its result is skipped. Decline instead: the merged
    // stream is an optimization, and no renumbering is correct once the
    // region's range no longer fits below the limit.
    if offset
        .checked_add(width)
        .is_none_or(|end| end > OpRef::VALUE_ID_LIMIT)
    {
        return Err(BackendError::Unsupported(format!(
            "wasm backend: inlined bridge value ids exceed the value-id space \
             (offset {offset}, width {width})"
        )));
    }
    let inputargs: Vec<InputArg> = bridge
        .inputargs
        .iter()
        .map(|ia| InputArg::from_type(ia.tp, ia.index + offset))
        .collect();
    // `Op::clone` gives the copy its own arg/failarg slots, but the operands in
    // them keep pointing at the region's original producers, whose `pos` this
    // must not touch — the region is retained for the next re-emission. So each
    // moved reference is rebound to a synthetic producer carrying the new id.
    let ops: Vec<Op> = bridge.ops.to_vec();
    for op in &ops {
        op.pos.set(shift(op.pos.get()));
        for (i, arg) in op.getarglist().iter().enumerate() {
            let before = arg.to_opref();
            let after = shift(before);
            if after != before {
                op.setarg(i, Operand::bound_from_opref(after));
            }
        }
        if let Some(mut fail_args) = op.getfailargs() {
            let mut moved = false;
            for slot in fail_args.iter_mut() {
                let before = slot.to_opref();
                let after = shift(before);
                if after != before {
                    *slot = Operand::bound_from_opref(after);
                    moved = true;
                }
            }
            if moved {
                op.setfailargs(fail_args);
            }
        }
    }

    Ok((
        InlinedBridge {
            source_fail_index: bridge.source_fail_index,
            external_jump: bridge.external_jump.clone(),
            outside_loop: bridge.outside_loop,
            trace_id: bridge.trace_id,
            inputargs,
            ops,
            gc_table_base: bridge.gc_table_base,
            constants: bridge.constants.clone(),
        },
        width,
    ))
}

/// Build a wasm module from majit IR.
pub fn build_wasm_module(
    inputs: &ModuleBuildInputs,
) -> Result<BuildWasmModuleOutput, BackendError> {
    let ModuleBuildInputs {
        inputargs,
        ops,
        inlined_bridges,
        constants,
        vtable_offset,
        classptr_to_typeid,
        guard_gc_type_info,
        alloc,
        wb,
        nursery,
        invalidated_flag_addr,
        gc_table_base,
        fail_index_base,
        bridge_cells_base,
        bridge_entry_arity,
        bridge_param_dispatch,
        trace_entry_census,
        inline_trip,
        external_jump_slot,
        external_jump_key,
        external_jump_wide_slot,
        frame,
        ca,
    } = inputs;
    // A bridge region has no function-entry loads, but its InputArgs and ops
    // still need locals, liveness, homes, guard exits, and call signatures.
    // Analyse the complete function as one stream while keeping `inputargs`
    // below as the actual function-entry list.
    // A normal module is emitted directly from its retained vectors.  Keep
    // that path allocation-free: code generation runs in the guest process,
    // so transient merged-stream allocations can otherwise perturb the next
    // collection boundary before any bridge is attached.
    let mut merged_inputargs = Vec::new();
    let mut merged_ops = Vec::new();
    let mut gc_table_bases = HashMap::new();
    let mut rebased_bridges: Vec<InlinedBridge> = Vec::new();
    let mut rebased_constants = indexmap::IndexMap::new();
    let (analysis_inputargs, analysis_ops): (&[InputArg], &[Op]) = if inlined_bridges.is_empty() {
        (inputargs, ops)
    } else {
        merged_inputargs.extend(inputargs.iter().map(InputArg::fresh_value_copy));
        merged_ops.extend(ops.iter().cloned());
        // The merged stream has one local namespace, so every region has to be
        // moved off the ids the owner and the earlier regions already use.
        rebased_constants = constants.clone();
        let mut next_value_id = value_id_end(inputargs, ops);
        for bridge in inlined_bridges {
            let (bridge, width) = rebase_region_value_ids(bridge, next_value_id)?;
            // The pool is keyed by value position for a folded value with no
            // producing op, so rebasing the region's ids moved its reads off
            // its own entries. Replay that window at the offset, and drop a
            // key another trace left inside it, or `unbound_pool_const_seeds`
            // either declines a resolvable value or seeds an unrelated one's
            // bits. Keys outside the window are left alone: rewriting them
            // would overwrite the entries the owner's own operations read.
            for id in 0..width {
                match bridge.constants.get(&id) {
                    Some(&bits) => {
                        rebased_constants.insert(id + next_value_id, bits);
                    }
                    None => {
                        rebased_constants.shift_remove(&(id + next_value_id));
                    }
                }
            }
            next_value_id += width;
            merged_inputargs.extend(bridge.inputargs.iter().map(InputArg::fresh_value_copy));
            for op in &bridge.ops {
                if op.opcode == OpCode::LoadFromGcTable {
                    gc_table_bases.insert(op.pos.get().raw(), bridge.gc_table_base);
                }
            }
            merged_ops.extend(bridge.ops.iter().cloned());
            rebased_bridges.push(bridge);
        }
        (&merged_inputargs, &merged_ops)
    };
    // Guard-entry moves and region emission must name the rebased ids, not the
    // ids the retained regions still carry.
    let emitted_bridges: &[InlinedBridge] = if inlined_bridges.is_empty() {
        inlined_bridges
    } else {
        &rebased_bridges
    };
    let region_spans = InlinedRegionSpan::collect(analysis_ops.len(), emitted_bridges);
    let constants = if inlined_bridges.is_empty() {
        constants
    } else {
        &rebased_constants
    };
    let (mut guards, num_vars) = collect_guards_and_vars(analysis_inputargs, analysis_ops);

    // An inlined bridge branches back into the owner with wasm `br`.  The
    // merged stream must therefore contain the local LABEL that opens the
    // wasm loop; a label-less cross-loop bridge has no in-function target.
    if !inlined_bridges.is_empty() && !merged_stream_has_loop_label(inputs) {
        return Err(BackendError::Unsupported(
            "wasm backend: inlined bridge stream has no local loop LABEL".into(),
        ));
    }
    for bridge in inlined_bridges {
        if bridge.ops.is_empty() {
            return Err(BackendError::Unsupported(
                "wasm backend: inlined bridge stream has an empty region".into(),
            ));
        }
        // The back edge rebinds the target LABEL's args from the region's
        // closing JUMP as a parallel move bounded by `min(jump, label)`, so a
        // JUMP naming fewer args leaves the remaining loop-carried locals
        // holding whatever the failing iteration left in them. Nothing
        // downstream reports that: wasm offset 0 is valid linear memory, so a
        // stale or zero Ref is read as an object instead of trapping.
        // `resolve_cross_loop_jump_target` refuses an arity mismatch before a
        // region is ever retained; this asserts the same invariant where the
        // move is emitted, rather than trusting a check in another file.
        if let Some(jump) = bridge
            .ops
            .last()
            .filter(|op| op.opcode == OpCode::Jump && bridge.external_jump.is_none())
        {
            let label_args = find_label_args(analysis_ops, jump);
            let jump_arity = jump.getarglist().len();
            if jump_arity < label_args.len() {
                return Err(BackendError::Unsupported(format!(
                    "wasm backend: inlined bridge JUMP rebinds {jump_arity} of the \
                     target LABEL's {} args",
                    label_args.len()
                )));
            }
        }
        // A region can carry a CALL_ASSEMBLER this build has no arm for. The
        // dedicated arm is selected by `ca.emit_ca`, which is decided when the
        // OWNER is compiled, and it reads the callee's geometry out of
        // `ca.targets`; a region merged in later brings its own callee. An op
        // that misses that arm does not fail — it falls through to the ordinary
        // residual-call arm, which lowers arg 0 as an
        // `__indirect_function_table` slot, and a CALL_ASSEMBLER's arg 0 is the
        // callee's first frame slot. That calls whatever the slot happens to
        // index and returns its result as the callee's, which is a silent wrong
        // answer rather than a trap. `wasm_unsupported_trace_reason` asks this
        // question of every trace's own ops; the merged stream is the one place
        // it is never re-asked, so ask it here.
        for op in &bridge.ops {
            if !op.opcode.is_call_assembler() {
                continue;
            }
            let target = op
                .getdescr()
                .and_then(|descr| descr.as_call_descr().and_then(|d| d.call_target_token()));
            if !ca.emit_ca || target.is_none_or(|token| !ca.targets.contains_key(&token)) {
                return Err(BackendError::Unsupported(format!(
                    "wasm backend: inlined bridge carries {:?}, which the owner \
                     build has no CALL_ASSEMBLER arm for",
                    op.opcode
                )));
            }
        }
        let source_guard = guards
            .get(bridge.source_fail_index as usize)
            .ok_or_else(|| {
                BackendError::Unsupported(
                    "wasm backend: inlined bridge source guard is outside the owner stream".into(),
                )
            })?;
        let source_args = live_fail_arg_count(
            source_guard.meta_descr.as_ref(),
            source_guard.fail_arg_refs.len(),
        );
        if source_args != bridge.inputargs.len() {
            return Err(BackendError::Unsupported(format!(
                "wasm backend: inlined bridge input arity {} differs from source guard arity {source_args}",
                bridge.inputargs.len(),
            )));
        }
    }

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
    let bridge_dispatch = *bridge_cells_base != 0;
    // All boundary values use an i64 carrier, including raw Float bits. That
    // makes the call type depend only on arity while preserving f64 payloads.
    let bridge_param_arities: Vec<usize> = if *bridge_param_dispatch && bridge_dispatch {
        let mut arities: Vec<usize> = guards
            .iter()
            .map(|guard| live_fail_arg_count(guard.meta_descr.as_ref(), guard.fail_arg_refs.len()))
            .collect();
        arities.sort_unstable();
        arities.dedup();
        arities
    } else {
        Vec::new()
    };

    // Frame value slots (inputs at entry, fail-arg spills at guard exit) occupy
    // `[1, 1 + max(num inputs, max fail args))`. They precede the dispatch key,
    // Ref homes, and the always-present tail call area; a chained bridge must
    // fit the source token's frozen value-slot count before it can share that
    // frame.
    let label_resume =
        LabelResumeData::collect_with_regions(&analysis_inputargs, &analysis_ops, &region_spans);
    let max_value_slots =
        normal_frame_value_slots(&analysis_inputargs, &analysis_ops) + label_resume.scalar_slots;
    if max_value_slots > frame.value_slots {
        let shortage = super::FrameShortage::new(
            super::FrameShortageKind::FrameValueSlots,
            max_value_slots,
            frame.value_slots,
        );
        if !inlined_bridges.is_empty() {
            super::record_inline_geometry(shortage.kind, shortage.needed, shortage.available);
        }
        return Err(BackendError::Unsupported(format!(
            "wasm backend: {} frame value slots exceed frozen frame layout ({})",
            shortage.needed, shortage.available,
        )));
    }

    let label_param_entry = has_label_param_entry(inputargs, ops, *frame, *bridge_entry_arity);
    let entry_param_count = 1 + if label_param_entry {
        crate::FROZEN_LABEL_PARAM_ARITY
    } else {
        bridge_entry_arity.unwrap_or(0)
    } as u32;
    if let Some(arity) = bridge_entry_arity
        && *arity != inputargs.len()
    {
        return Err(BackendError::Unsupported(format!(
            "wasm backend: bridge parameter arity {arity} differs from input arity {}",
            inputargs.len(),
        )));
    }
    let value_types = collect_value_types(
        &analysis_inputargs,
        &analysis_ops,
        num_vars,
        entry_param_count,
    );
    let ref_values = RefValues::collect(&analysis_inputargs, &analysis_ops);
    let ref_homes = RefHomes::collect(
        &analysis_inputargs,
        &analysis_ops,
        ca.emit_ca,
        &label_resume.captured_refs,
        &region_spans,
    );
    let num_ref_homes = ref_homes.len();
    let shortage = if num_ref_homes > frame.ordinary_home_slots() {
        Some(super::FrameShortage::new(
            super::FrameShortageKind::OrdinaryRefHomes,
            num_ref_homes,
            frame.ordinary_home_slots(),
        ))
    } else {
        label_resume.shortage(*frame)
    };
    if let Some(shortage) = shortage {
        if !inlined_bridges.is_empty() {
            super::record_inline_geometry(shortage.kind, shortage.needed, shortage.available);
        }
        let reason = match shortage.kind {
            super::FrameShortageKind::OrdinaryRefHomes => format!(
                "wasm backend: {} ordinary ref homes exceed frozen frame layout ({})",
                shortage.needed, shortage.available,
            ),
            super::FrameShortageKind::LabelResumeRefSlots => format!(
                "wasm backend: {} LABEL ref captures exceed label resume layout ({} label ref slots)",
                shortage.needed, shortage.available,
            ),
            super::FrameShortageKind::LabelResumeCaptureSlots => format!(
                "wasm backend: {} LABEL capture slots exceed label resume layout ({})",
                shortage.needed, shortage.available,
            ),
            super::FrameShortageKind::FrameValueSlots => {
                unreachable!("value-slot shortage was checked above")
            }
        };
        return Err(BackendError::Unsupported(reason));
    }

    // CA frames execute the source loop and this bridge on the same frozen
    // geometry.  `compile_bridge` rejects a bridge that needs more slots, so
    // no global floor or speculative slack is needed here.

    // This exact lowering census controls the host-trampoline import. Direct
    // residual helpers, including the CA arm's inline fast path, use
    // `call_indirect` and need no import, although their frozen frame still
    // keeps the tail call area for future bridges.
    let needs_call =
        has_trampoline_calls(&analysis_inputargs, &analysis_ops, constants, ca.emit_ca);
    // In-module residual calls ([`RESIDUAL_CALL_ABI`]): the largest
    // eligible `(i64×n)->i64` arity in this trace — residual CALLs (word
    // result or word-ABI void) plus the `New*` / write-barrier helper
    // targets, which share the same uniform-i64 ABI — or `None` if there
    // are none. Each distinct arity `0..=max` gets its own function type
    // (declared below) so those arms can `call_indirect` with a static type.
    let residual_max_arity = {
        let scanned = analysis_ops
            .iter()
            .filter_map(|op| direct_helper_i64_arity(op, &ref_values, constants))
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
            Some(scanned.map_or(0, |m| m))
        } else {
            scanned
        }
    };
    // Typed float residual calls use their descr's faithful wasm ABI instead
    // of the uniform i64 helper family. Preserve first-use order so a given
    // trace gets stable type indices while declaring each signature once.
    let mut typed_residual_sigs = Vec::new();
    for op in analysis_ops {
        if let Some(sig) = residual_call_typed_sig(op, constants)
            && !typed_residual_sigs.contains(&sig)
        {
            typed_residual_sigs.push(sig);
        }
    }
    // True-void residual calls use `(i64×n) -> ()`, a separate family from the
    // i64- and f64-result types. As with the uniform i64 family, declaring
    // `0..=max` makes each type index a base plus the call arity.
    let true_void_residual_max_arity = analysis_ops
        .iter()
        .filter_map(|op| residual_call_void_true_arity(op, constants))
        .max();
    // The shared indirect-function table backs direct residual helpers as well
    // as host-trampoline dispatch, chained bridges, and CA recursion.
    let needs_table = needs_call
        || bridge_dispatch
        || residual_max_arity.is_some()
        || !typed_residual_sigs.is_empty()
        || true_void_residual_max_arity.is_some()
        || ca.emit_ca
        || inline_trip.is_some();
    // `ca.emit_ca` forces the direct helper family to include arities 0..=2,
    // so all CA frame-helper trampoline `else` arms below are baseline-only.
    debug_assert!(!ca.emit_ca || residual_max_arity.is_some());

    let mut module = Module::new();

    // Type section
    let mut types = TypeSection::new();
    // Type 0 remains the loop/host entry signature. A bridge parameter entry
    // receives a separate type so its terminal JUMP can still call a loop.
    types.ty().function(vec![ValType::I32], vec![ValType::I32]);
    let mut next_type_idx = 1u32;
    let bridge_entry_type_idx = bridge_entry_arity.map(|arity| {
        let idx = next_type_idx;
        next_type_idx += 1;
        types.ty().function(
            std::iter::once(ValType::I32)
                .chain(std::iter::repeat_n(ValType::I64, arity))
                .collect::<Vec<_>>(),
            vec![ValType::I32],
        );
        idx
    });
    // Declared by the module that *defines* a parameter entry and by one that
    // only *calls* another module's: a loop-closing JUMP naming a published
    // wide slot needs the callee's type to `return_call_indirect` it.
    let label_param_type_idx = (label_param_entry || *external_jump_wide_slot != 0).then(|| {
        let idx = next_type_idx;
        next_type_idx += 1;
        types.ty().function(
            std::iter::once(ValType::I32)
                .chain(std::iter::repeat_n(
                    ValType::I64,
                    crate::FROZEN_LABEL_PARAM_ARITY,
                ))
                .collect::<Vec<_>>(),
            vec![ValType::I32],
        );
        idx
    });
    let mut bridge_param_type_indices = indexmap::IndexMap::new();
    if let (Some(arity), Some(idx)) = (*bridge_entry_arity, bridge_entry_type_idx) {
        bridge_param_type_indices.insert(arity, idx);
    }
    let jit_call_type_idx = if needs_call {
        let idx = next_type_idx;
        next_type_idx += 1;
        types
            .ty()
            .function(vec![ValType::I32, ValType::I32], vec![]);
        Some(idx)
    } else {
        None
    };
    // Residual-call types follow: `(i64×n) -> i64` for arity `n`, indexed by
    // `residual_type_base + n`. `residual_type_base` = the count of types above.
    let residual_type_base = next_type_idx;
    if let Some(max) = residual_max_arity {
        for n in 0..=max {
            types
                .ty()
                .function(vec![ValType::I64; n], vec![ValType::I64]);
        }
        next_type_idx += max as u32 + 1;
    }
    // CA deopt-helper type `(i64 frame_ptr, i64 compiled_ptr) -> i64`. The CA arm
    // `call_indirect`s `wasm_ca_resume_deopt` through it when a self-recursive
    // callee leaves its trace through a guard (a deopt). Declared after the
    // residual-call type family so its index is independent of which residual
    // arities the bridge happens to use.
    let ca_helper_type_idx = next_type_idx;
    if ca.emit_ca {
        types
            .ty()
            .function(vec![ValType::I64, ValType::I64], vec![ValType::I64]);
        next_type_idx += 1;
    }
    // Typed residual types follow all pre-existing direct helper types. Both
    // the parameter sequence and the result come from the call descr (`i64`
    // for Int/Ref, `f64` for Float); the emitter uses this map to select the
    // exact `call_indirect` type for each callee.
    let typed_residual_type_base = next_type_idx;
    let typed_residual_type_indices = typed_residual_sigs
        .iter()
        .cloned()
        .enumerate()
        .map(|(offset, sig)| (sig, typed_residual_type_base + offset as u32))
        .collect::<indexmap::IndexMap<TypedResidualSig, u32>>();
    for (params, result) in typed_residual_type_indices.keys() {
        types
            .ty()
            .function(params.clone(), result.iter().copied().collect::<Vec<_>>());
    }
    next_type_idx += typed_residual_type_indices.len() as u32;
    let true_void_residual_type_base = next_type_idx;
    if let Some(max) = true_void_residual_max_arity {
        for n in 0..=max {
            types.ty().function(vec![ValType::I64; n], vec![]);
        }
        next_type_idx += max as u32 + 1;
    }
    // Deferred-merge trip callback `(i64 pending_id) -> i64`, declared before
    // the bridge-parameter arities so an armed probe cannot shift their
    // indices.
    let inline_trip_type_idx = next_type_idx;
    if inline_trip.is_some() {
        types.ty().function(vec![ValType::I64], vec![ValType::I64]);
        next_type_idx += 1;
    }
    for arity in bridge_param_arities {
        if bridge_param_type_indices.contains_key(&arity) {
            continue;
        }
        bridge_param_type_indices.insert(arity, next_type_idx);
        next_type_idx += 1;
        types.ty().function(
            std::iter::once(ValType::I32)
                .chain(std::iter::repeat_n(ValType::I64, arity))
                .collect::<Vec<_>>(),
            vec![ValType::I32],
        );
    }
    // Shared guard-exit spill functions, declared after every other family so
    // an added arity cannot shift an index a call site already baked.
    let spill_arities = spill_helper_arities(&guards);
    let mut spill_helper_type_indices: Vec<u32> = Vec::with_capacity(spill_arities.len());
    for &arity in &spill_arities {
        spill_helper_type_indices.push(next_type_idx);
        next_type_idx += 1;
        types.ty().function(
            std::iter::once(ValType::I32)
                .chain(std::iter::repeat_n(ValType::I64, arity))
                .collect::<Vec<_>>(),
            Vec::new(),
        );
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
            "jit_call_compact",
            EntityType::Function(jit_call_type_idx.expect("jit_call type")),
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
    if label_param_entry {
        // The narrow shim keeps type 0 so the host and every type-0 indirect
        // call still enter here; the wide entry follows it.
        functions.function(0);
        functions.function(label_param_type_idx.expect("a parameter entry declares its own type"));
    } else {
        functions.function(bridge_entry_type_idx.unwrap_or(0));
    }
    for &type_idx in &spill_helper_type_indices {
        functions.function(type_idx);
    }
    module.section(&functions);

    // Only armed modules carry this global. The runner reads it after
    // instantiation to give `PYRE_WASM_DUMP_ALL_TRACES` the same trace id the
    // census reports; it is omitted entirely from ordinary trace modules.
    if let Some(census) = trace_entry_census {
        let mut globals = GlobalSection::new();
        globals.global(
            GlobalType {
                val_type: ValType::I64,
                mutable: false,
                shared: false,
            },
            &ConstExpr::i64_const(census.trace_id as i64),
        );
        module.section(&globals);
    }

    // Export section: trace function index depends on whether we imported jit_call
    let trace_func_idx = if needs_call { 1 } else { 0 };
    let mut exports = ExportSection::new();
    exports.export("trace", ExportKind::Func, trace_func_idx);
    if label_param_entry {
        exports.export("trace_wide", ExportKind::Func, trace_func_idx + 1);
    }
    if trace_entry_census.is_some() {
        exports.export("trace_entry_census_id", ExportKind::Global, 0);
    }
    module.section(&exports);

    // Code section
    let mut codes = CodeSection::new();
    let jit_call_idx = if needs_call { Some(0u32) } else { None };
    // The spill functions follow this module's entry function(s) in the
    // function and code sections alike, so their indices start past them.
    let first_spill_func_idx = trace_func_idx + if label_param_entry { 2 } else { 1 };
    let spill_helper_indices: indexmap::IndexMap<usize, u32> = spill_arities
        .iter()
        .enumerate()
        .map(|(i, &arity)| (arity, first_spill_func_idx + i as u32))
        .collect();
    let func = build_function(
        inputargs,
        &analysis_inputargs,
        &analysis_ops,
        emitted_bridges,
        constants,
        num_vars,
        &value_types,
        jit_call_idx,
        *vtable_offset,
        classptr_to_typeid,
        guard_gc_type_info,
        *alloc,
        wb,
        nursery.as_ref(),
        &ref_values,
        &ref_homes,
        &label_resume,
        *bridge_cells_base,
        bridge_dispatch,
        *bridge_entry_arity,
        &bridge_param_type_indices,
        *invalidated_flag_addr,
        *gc_table_base,
        &gc_table_bases,
        *fail_index_base,
        *external_jump_slot,
        *external_jump_key,
        label_param_type_idx
            .filter(|_| *external_jump_wide_slot != 0)
            .map(|type_idx| (*external_jump_wide_slot, type_idx)),
        *frame,
        residual_max_arity.map(|_| residual_type_base),
        &typed_residual_type_indices,
        true_void_residual_max_arity.map(|_| true_void_residual_type_base),
        ca.clone(),
        ca_helper_type_idx,
        *trace_entry_census,
        label_param_entry,
        inline_trip.map(|probe| (probe, inline_trip_type_idx)),
        &spill_helper_indices,
    )?;
    if label_param_entry {
        codes.function(&build_label_param_shim(trace_func_idx + 1));
    }
    codes.function(&func);
    for &arity in &spill_arities {
        codes.function(&build_spill_helper(arity));
    }
    module.section(&codes);

    Ok((module.finish(), guards, num_ref_homes))
}

fn build_label_param_shim(wide_func_idx: u32) -> Function {
    let mut func = Function::new(Vec::new());
    let mut raw_sink = func.instructions();
    let mut sink = PeepSink::new(&mut raw_sink);

    sink.local_get(0);
    for k in 0..crate::FROZEN_LABEL_PARAM_ARITY {
        sink.local_get(0);
        sink.i64_load(mem64(FRAME_SLOT_BASE + k as u64 * SLOT_SIZE));
    }
    sink.return_call(wide_func_idx);
    sink.end();
    sink.flush();
    drop(sink);

    func
}

/// What a wasm function costs the host compiler before its body counts, in
/// units of the body instructions the same cost would buy.
///
/// Collapsing 30.9 KB of `for_iter_list_fold` spill runs into 97 extra
/// functions cut cranelift's compile time for its 44 modules from 130.8 ms to
/// 117.7 ms, where the bytes alone were worth 21.1 ms; the difference puts a
/// function's own fixed cost near 0.06 ms, about forty instructions of body.
/// Charging it here keeps the near-break-even counts out.
const SPILL_HELPER_FIXED_INSTRS: usize = 40;

/// Fail-argument counts worth a shared spill function, from the guard exits
/// this module is about to emit.
///
/// A guard exit writes its fail arguments to the positional exit slots, three
/// wasm instructions each (`local.get 0`, the value, `i64.store`). Those runs
/// are the largest single thing a trace module contains — 42% of the
/// instructions across `synth/for_iter_list_fold`'s 44 modules. The host
/// compiles every one of those modules with cranelift before the trace can
/// run, at roughly 0.6 ms per kilobyte handed to it against a per-module fixed
/// cost of about 0.2 ms, so what the module costs to admit is very nearly what
/// it weighs. The spill is purely positional, so one function per argument
/// count serves every exit of that count and the call site costs one
/// instruction per argument instead of three.
///
/// A count is admitted only when the exits that share it pay for the function:
/// `uses * (3n - (n + 2))` saved against `3n + 1` emitted, plus
/// [`SPILL_HELPER_FIXED_INSTRS`] for the function itself. Counts of one
/// argument never do (the call site is the same size as the stores), and a
/// count used once never does. An arity admitted here that no exit reaches —
/// a guard whose region was merged branches instead of spilling — costs its
/// unused body and nothing else, so the estimate may over-admit safely.
fn spill_helper_arities(guards: &[GuardExit]) -> Vec<usize> {
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for guard in guards {
        *uses.entry(guard.fail_arg_refs.len()).or_default() += 1;
    }
    let mut arities: Vec<usize> = uses
        .into_iter()
        .filter(|&(arity, uses)| {
            arity >= 2
                && uses >= 2
                && uses * (2 * arity - 2) > 3 * arity + 1 + SPILL_HELPER_FIXED_INSTRS
        })
        .map(|(arity, _)| arity)
        .collect();
    // `HashMap` iteration order is not stable across runs, and two compilations
    // of the same trace must emit byte-identical modules (`compile_module_cached`
    // keys its host handle on the bytes).
    arities.sort_unstable();
    arities
}

/// One shared spill function: `(i32 frame_ptr, i64 x arity) -> ()`, writing its
/// arguments to the positional exit slots `frame[1..=arity]`.
fn build_spill_helper(arity: usize) -> Function {
    let mut func = Function::new(Vec::new());
    let mut raw_sink = func.instructions();
    let mut sink = PeepSink::new(&mut raw_sink);
    for i in 0..arity {
        sink.local_get(0);
        sink.local_get(1 + i as u32);
        sink.i64_store(mem64(FRAME_SLOT_BASE + i as u64 * SLOT_SIZE));
    }
    sink.end();
    sink.flush();
    drop(sink);
    func
}

#[allow(clippy::too_many_arguments)]
fn build_function(
    entry_inputargs: &[InputArg],
    inputargs: &[InputArg],
    ops: &[Op],
    inlined_bridges: &[InlinedBridge],
    constants: &indexmap::IndexMap<u32, i64>,
    num_vars: u32,
    value_types: &ValueLocals,
    jit_call_idx: Option<u32>,
    vtable_offset: Option<usize>,
    classptr_to_typeid: &HashMap<i64, u32>,
    guard_gc_type_info: &GuardGcTypeInfo,
    alloc: AllocHelpers,
    wb: &WriteBarrierHelpers,
    nursery: Option<&NurseryAllocParams>,
    ref_values: &RefValues,
    ref_homes: &RefHomes,
    label_resume: &LabelResumeData,
    cells_base: u32,
    bridge_dispatch: bool,
    bridge_entry_arity: Option<usize>,
    bridge_param_type_indices: &indexmap::IndexMap<usize, u32>,
    invalidated_flag_addr: u32,
    gc_table_base: u32,
    gc_table_bases: &HashMap<u32, u32>,
    fail_index_base: u32,
    external_jump_slot: u32,
    // Resume-at-LABEL dispatch key the terminal external JUMP writes before
    // tail-calling `external_jump_slot`: `target label ordinal + 1`, so the
    // target's entry `br_table` lands on that label's resume loader. `0` when
    // the target is not peeled (no dispatch reads the slot).
    external_jump_key: u32,
    // `(wide table slot, wasm type index)` of the target's fixed-arity
    // parameter entry, when it published one. The terminal external JUMP then
    // passes its args as parameters instead of through the frame slots the
    // target's narrow shim reads back.
    external_jump_wide: Option<(u32, u32)>,
    frame: FrameGeometry,
    // Base wasm type index of the `(i64×n)->i64` residual-call types (type
    // `residual_type_base + n` for arity `n`), or `None` when the trace has no
    // eligible residual call / `New*` / write barrier, so those arms always
    // use the `jit_call` path.
    residual_type_base: Option<u32>,
    // Exact wasm type indices for direct typed residual calls, keyed by their
    // descr-derived parameter sequence and result. Float SSA values are
    // converted to/from their i64 bit carrier around the call.
    typed_residual_type_indices: &indexmap::IndexMap<TypedResidualSig, u32>,
    // Base wasm type index of the `(i64×n) -> ()` true-void residual-call
    // types (type `true_void_residual_type_base + n` for arity `n`), or
    // `None` when the trace has no eligible true-void residual call.
    true_void_residual_type_base: Option<u32>,
    // Self-recursive CALL_ASSEMBLER arm (`PYRE_WASM_CA`). `ca.emit_ca` off keeps
    // the body byte-identical.
    ca: CaParams,
    // wasm type index of the CA deopt helper `(i64, i64) -> i64`, declared in the
    // module type section when `ca.emit_ca`. The CA arm uses it to `call_indirect`
    // `ca.deopt_helper_slot` for a deopted callee.
    ca_helper_type_idx: u32,
    trace_entry_census: Option<crate::TraceEntryCensusStorage>,
    label_param_entry: bool,
    inline_trip: Option<(InlineTripProbe, u32)>,
    spill_helper_indices: &indexmap::IndexMap<usize, u32>,
) -> Result<Function, BackendError> {
    // The CA arm requires residual types (the setup above forces arity >= 2
    // whenever it is emitted). Its `jit_call` fallback branches are retained
    // solely for a trace that declared no residual type family at all.
    debug_assert!(!ca.emit_ca || residual_type_base.is_some());
    let value_locals_end = value_types.end_local();
    // Resume-at-LABEL shape, needed here because `resume_dispatch` costs a
    // local. A peeled loop wraps its preamble in a dispatch so a loop-closing
    // bridge can re-enter AT any LABEL up to the header; labels after the
    // header sit inside the `loop` and get no resume arm
    // (`resumable_label_count`).
    let key_dispatch = is_resumable_peeled(ops);
    let num_labels = if key_dispatch {
        resumable_label_count(ops)
    } else {
        0
    };
    let bridge_op_count = inlined_bridges
        .iter()
        .map(|bridge| bridge.ops.len())
        .sum::<usize>();
    let bridge_start = ops.len().checked_sub(bridge_op_count).ok_or_else(|| {
        BackendError::Unsupported(
            "wasm backend: inlined bridge operations are not contained in the merged stream".into(),
        )
    })?;
    // `InlinedBridge::outside_loop` names the placement. The outside ones are
    // emitted past the header `loop`'s `end`, so their ops are the tail of the
    // merged stream and every inside one has to precede them.
    let body_region_count = inlined_bridges
        .iter()
        .position(|bridge| bridge.outside_loop)
        .unwrap_or(inlined_bridges.len());
    let outside_region_count = inlined_bridges.len() - body_region_count;
    if inlined_bridges[body_region_count..]
        .iter()
        .any(|bridge| !bridge.outside_loop)
    {
        return Err(BackendError::Unsupported(
            "wasm backend: an inside-loop inline region follows an outside-loop one".into(),
        ));
    }
    if inlined_bridges.iter().any(|bridge| {
        !bridge.outside_loop && source_guard_precedes_loop_label(ops, bridge.source_fail_index)
    }) {
        return Err(BackendError::Unsupported(
            "wasm backend: a preamble-sourced inline region is placed inside the loop".into(),
        ));
    }
    // A region's block closes where its own ops begin, so a guard branching
    // into it must sit before them. That one inequality is what makes a
    // region's ordinal usable as a branch depth (`emit_guard_exit` counts it
    // from the family's still-open blocks), and it is also what refuses the two
    // structurally impossible attachments: a region reached from a guard in a
    // LATER region of the same family, and a body region reached from a guard
    // in an outside one, whose header `loop` has already closed.
    {
        let mut start = bridge_start;
        for bridge in inlined_bridges {
            let guard_pos = exit_op_index(ops, bridge.source_fail_index).ok_or_else(|| {
                BackendError::Unsupported(
                    "wasm backend: inlined bridge source guard is outside the merged stream".into(),
                )
            })?;
            if guard_pos >= start {
                return Err(BackendError::Unsupported(
                    "wasm backend: an inline region is reached from a guard at or past its \
                     own ops, so the block it branches to has already closed"
                        .into(),
                ));
            }
            start += bridge.ops.len();
        }
    }
    let outside_start = bridge_start
        + inlined_bridges[..body_region_count]
            .iter()
            .map(|bridge| bridge.ops.len())
            .sum::<usize>();
    if outside_region_count > 0 && !key_dispatch {
        return Err(BackendError::Unsupported(
            "wasm backend: an outside-loop inline region needs an entry dispatch".into(),
        ));
    }
    // An outside-loop region re-enters PAST its target LABEL's resume loader,
    // so it takes that label's captures from the frame slots the fall-through
    // path writes as it crosses the label — and a fresh entry clears them. A
    // region reached from the loop body has crossed every resumable label by
    // then; one reached from the preamble has only crossed the labels ahead of
    // its own guard.
    {
        let mut start = outside_start;
        for bridge in &inlined_bridges[body_region_count..] {
            let guard_pos = exit_op_index(ops, bridge.source_fail_index).ok_or_else(|| {
                BackendError::Unsupported(
                    "wasm backend: inlined bridge source guard is outside the owner stream".into(),
                )
            })?;
            if bridge.external_jump.is_some() {
                // A region that leaves by a cross-module tail call re-enters
                // no LABEL of this function, so it crosses none of them.
                start += bridge.ops.len();
                continue;
            }
            for op in &ops[start..start + bridge.ops.len()] {
                if op.opcode != OpCode::Jump {
                    continue;
                }
                if find_jump_target_label_index(ops, op).is_none_or(|idx| idx >= guard_pos) {
                    return Err(BackendError::Unsupported(
                        "wasm backend: an outside-loop inline region closes at a LABEL its \
                         own entry path had not crossed"
                            .into(),
                    ));
                }
            }
            start += bridge.ops.len();
        }
    }
    // A region whose closing JUMP names a resumable LABEL other than the header
    // cannot `br` to the `loop`. Wrap the dispatch in a `loop` such a region
    // re-enters through instead, and give the entry `br_table` a second bucket
    // per label: key `num_labels + 1 + j` lands PAST label j's resume loader,
    // so the region hands its values over in locals rather than through the
    // frame slots the loader reads. A preamble-sourced region always leaves by
    // that route: it is emitted past the `end` of the header `loop`, so no `br`
    // reaches it.
    let resume_dispatch = key_dispatch
        && (outside_region_count > 0
            || ops[bridge_start..].iter().any(|op| {
                op.opcode == OpCode::Jump && jump_resume_ordinal(ops, op, num_labels).is_some()
            }));

    // Value locals occupy the dense local range beginning at 1; reserve
    // `UMULHI_SCRATCH` i64 locals past them for the `UintMulHigh`
    // 32-bit-split expansion, plus one i64 local for the pending overflow flag.
    // One i32 local past those holds a bridge table slot while a guard arm
    // performs its direct indirect tail call (or while the frame-entry
    // dispatcher is enabled without parameter entries).
    let ovf_flag_local = value_locals_end + UMULHI_SCRATCH;
    let bridge_slot_local = ovf_flag_local + 1;
    // The CALL_ASSEMBLER arm needs three more i32 locals: the current callee
    // frame, its returned fail index, and the immutable runtime-target snapshot
    // loaded from the stable dispatch cell.  Keeping the snapshot address in a
    // local makes function/geometry/GC-map selection coherent across redirect.
    let ca_cfp_local = bridge_slot_local + 1;
    let ca_fi_local = ca_cfp_local + 1;
    let ca_target_local = ca_fi_local + 1;
    // Extra i32 scratches when the inline nursery-bump fast path is armed:
    // one holds the loaded `nursery_free` across the bump/commit sequence;
    // runtime varsize array allocation also needs one for the computed
    // total/new-free word.
    let base_i32_locals: u32 = 1 + if ca.emit_ca { 3 } else { 0 };
    let alloc_scratch_local = bridge_slot_local + base_i32_locals;
    let alloc_size_local = alloc_scratch_local + 1;
    // A keyed census must preserve the raw dispatch value until `br_table`.
    // Its counter-address scratch cannot share `bridge_slot_local`, because
    // the latter would replace the selector with a guest-memory address.
    let trace_entry_key_local = bridge_slot_local
        + base_i32_locals
        + if nursery.is_some() || ca.inline.is_some() {
            2
        } else {
            0
        };
    let trace_entry_needs_key_local = trace_entry_census.is_some() && is_resumable_peeled(ops);
    // `resume_dispatch` keeps the entry key in a local so a region can rewrite
    // it and branch back into the dispatch; without it the key is consumed
    // straight off the frame load.
    let resume_key_local = trace_entry_key_local + u32::from(trace_entry_needs_key_local);
    debug_assert_eq!(bridge_slot_local, ovf_flag_local + 1);
    debug_assert_eq!(ca_cfp_local, bridge_slot_local + 1);
    debug_assert_eq!(ca_fi_local, ca_cfp_local + 1);
    debug_assert_eq!(ca_target_local, ca_fi_local + 1);
    debug_assert_eq!(alloc_scratch_local, bridge_slot_local + base_i32_locals);
    debug_assert_eq!(alloc_size_local, alloc_scratch_local + 1);
    let inline_guards: Vec<InlineGuard<'_>> = inlined_bridges
        .iter()
        .enumerate()
        .map(|(region, bridge)| InlineGuard {
            guard_idx: fail_index_base + bridge.source_fail_index,
            inputargs: &bridge.inputargs,
            // Blocks open in reverse attach order, making region 0 of each
            // family innermost. The guard `if` contributes the final +1 in
            // `emit_guard_if_exit`.
            region_ordinal: if region < body_region_count {
                region as u32
            } else {
                (region - body_region_count) as u32
            },
            outside_loop: region >= body_region_count,
        })
        .collect();
    let guard_dispatch = BridgeDispatch {
        cells_base,
        fail_index_base,
        bridge_slot_local,
        enabled: bridge_dispatch,
        param_type_indices: bridge_param_type_indices,
        inline_guards: &inline_guards,
        outside_region_base: 0,
        closed_body_regions: 0,
        closed_outside_regions: 0,
        ref_homes,
        frame,
        counter_slot: counter_slot(inputargs, ops).map(|slot| slot as u64),
        spill_helpers: spill_helper_indices,
    };
    let mut locals = Vec::new();
    let mut start = 0;
    while start < value_types.types().len() {
        let ty = value_types.types()[start];
        let mut end = start + 1;
        while end < value_types.types().len() && value_types.types()[end] == ty {
            end += 1;
        }
        locals.push(((end - start) as u32, ty));
        start = end;
    }
    if let Some((count, ValType::I64)) = locals.last_mut() {
        *count += UMULHI_SCRATCH + 1;
    } else {
        locals.push((UMULHI_SCRATCH + 1, ValType::I64));
    }
    locals.push((
        base_i32_locals
            + if nursery.is_some() || ca.inline.is_some() {
                2
            } else {
                0
            }
            + u32::from(trace_entry_needs_key_local)
            + u32::from(resume_dispatch),
        ValType::I32,
    ));
    let mut func = Function::new(locals);
    let mut raw_sink = func.instructions();
    let mut sink = PeepSink::new(&mut raw_sink);

    // Bind the folded constants the optimizer left under a plain op position
    // (see `unbound_pool_const_seeds`). Emitted before every block so the
    // binding dominates the whole body, including a resume-at-LABEL entry.
    for (raw, bits) in unbound_pool_const_seeds(inputargs, ops, constants, num_vars)? {
        sink.i64_const(bits);
        if value_types.ty(raw) == ValType::F64 {
            sink.f64_reinterpret_i64();
        }
        sink.local_set(value_types.local(raw));
    }

    // A peeled loop arrives as `[preamble..][LABEL][body..][JUMP]`: the
    // preamble runs once on entry, the LABEL is the loop-back target, and
    // JUMP branches back to it. Emit the `loop` at the LABEL selected by the
    // terminal JUMP's descr (not merely the last LABEL) so multi-label traces
    // re-execute the complete loop body.
    let loop_label_idx = find_loop_label_index(ops);
    let has_loop = loop_label_idx.is_some();

    // Def / last-use positions for the post-collection Ref reload filter. The
    // spans must match the ones `RefHomes` was built from, or a home would be
    // reserved and never reloaded (or the reverse).
    let region_spans = InlinedRegionSpan::collect(ops.len(), inlined_bridges);
    let liveness = HomeLiveness::collect_with_regions(inputargs, ops, &region_spans);

    // Resume-at-LABEL: a peeled loop wraps its preamble in a dispatch so a
    // loop-closing bridge can re-enter AT any LABEL — key = label ordinal + 1
    // — skipping the code before it, in-module instead of round-tripping
    // through the host. Keyed on the peeled shape (single- OR multi-label);
    // every other trace (non-peeled loop, straight-line, bridge) keeps its
    // byte-identical layout. Each label gets a (past_loader, loader) block
    // pair; the entry `br_table` jumps to the keyed label's resume loader,
    // and the fall-through path `br`s over each loader. Key 0 (and any
    // out-of-range key) runs the function from its entry (the preamble).
    // Every count below is over the resumable prefix — labels 0..=header.
    let all_label_args: Vec<Vec<OpRef>> = ops
        .iter()
        .filter(|op| op.opcode == OpCode::Label)
        .take(num_labels)
        .map(|op| op.getarglist().iter().map(|a| a.to_opref()).collect())
        .collect();

    // The enclosing exit block gives each guard and Finish a direct path to
    // the bridge-dispatch epilogue after it has spilled its fail arguments.
    sink.block(BlockType::Empty); // A $hot_exit
    if resume_dispatch {
        // The key must survive into the `loop` a region branches back to, so
        // read it once here, outside that loop. The census stays outside too:
        // it counts entries into the module, and an in-module re-dispatch is
        // not one.
        sink.local_get(0);
        sink.i64_load(mem64(frame.dispatch_key_ofs));
        sink.i32_wrap_i64();
        sink.local_set(resume_key_local);
        if let Some(census) = trace_entry_census {
            emit_trace_entry_census(&mut sink, census, bridge_slot_local, Some(resume_key_local));
        }
        sink.loop_(BlockType::Empty); // R $resume
        // One block per preamble-sourced region, outside the (B_j, C_j) label
        // pairs and outside the header `loop`, so a guard anywhere in the
        // function can `br` to it. Region 0 is innermost, matching the order
        // the `end`s below close them in.
        for _ in 0..outside_region_count {
            sink.block(BlockType::Empty); // P_k
        }
    }
    if key_dispatch {
        // Per resumable label j (opened outermost = the loop header):
        //   block $past_loader_j (B_j) — the fall-through path br's over the
        //     label-j resume loader.
        //   block $loader_j (C_j) — the `br_table` lands here (its end) for
        //     key j+1: the label-j resume loader.
        // block $dispatch (D) — key 0 br's here: run from the entry.
        for _ in 0..num_labels {
            sink.block(BlockType::Empty); // B_j (j descending)
            sink.block(BlockType::Empty); // C_j
        }
        sink.block(BlockType::Empty); // D $dispatch
        if resume_dispatch {
            sink.local_get(resume_key_local);
        } else {
            sink.local_get(0);
            sink.i64_load(mem64(frame.dispatch_key_ofs));
            sink.i32_wrap_i64();
            // Without a census the key is already where `br_table` wants it.
            // Only the census needs it a second time, so only the census pays
            // to keep a copy: a `tee`/`get` pair here costs every entry into a
            // peeled module.
            if let Some(census) = trace_entry_census {
                let dispatch_key_local = if trace_entry_needs_key_local {
                    trace_entry_key_local
                } else {
                    bridge_slot_local
                };
                sink.local_tee(dispatch_key_local);
                emit_trace_entry_census(
                    &mut sink,
                    census,
                    bridge_slot_local,
                    Some(dispatch_key_local),
                );
                sink.local_get(dispatch_key_local);
            }
        }
        // Depths at this point, innermost first: D=0, then (C_j, B_j) pairs
        // with C_j at 2j+1 and B_j at 2j+2. Entry j+1 of the table targets
        // C_j — label j's resume loader; entry 0 and the default target D (the
        // entry path). Under `resume_dispatch` a second bucket per label,
        // `num_labels + 1 + j`, targets B_j: past that loader, for a region
        // that has already put the label args in their locals.
        let br_targets: Vec<u32> = std::iter::once(0)
            .chain((0..num_labels as u32).map(|j| 2 * j + 1))
            .chain(
                resume_dispatch
                    .then(|| (0..num_labels as u32).map(|j| 2 * j + 2))
                    .into_iter()
                    .flatten(),
            )
            .collect();
        sink.br_table(br_targets, 0);
        sink.end(); // end D $dispatch — key-0 entry path continues here
    } else if let Some(census) = trace_entry_census {
        emit_trace_entry_census(&mut sink, census, bridge_slot_local, None);
    }

    // Fresh entry owns key 0 and must clear both the trace's ordinary homes
    // and its high LABEL-capture homes.  A resume dispatch branches past this
    // code, preserving captures written when the source loop first crossed
    // the LABEL.  Chained bridges have no capture plan and clear only their
    // own low ordinary-home prefix.
    // A home the input loop fills below needs no null first: its store follows
    // immediately and nothing between the two allocates, so no collection can
    // read the slot while it is stale. Homes no input fills keep their clear
    // because store-on-def writes them only later.
    // The loop below fills `entry_inputargs`, not every arg of the merged
    // stream: an appended region's live-ins are stored by the guard-fail branch
    // that reaches the region, which is nowhere near this entry. Marking those
    // homes filled here would skip their clear and leave the collector reading
    // an uninitialised slot.
    let mut input_filled_home = vec![false; ref_homes.len()];
    for ia in entry_inputargs {
        if let Some(h) = ref_homes.home_id(ia.index) {
            input_filled_home[h as usize] = true;
        }
    }
    for h in 0..ref_homes.len() as u64 {
        if input_filled_home[h as usize] {
            continue;
        }
        sink.local_get(0);
        sink.i64_const(0);
        sink.i64_store(mem64(frame.home_slot_base + h * SLOT_SIZE));
    }
    for h in 0..label_resume.ref_slots as u64 {
        sink.local_get(0);
        sink.i64_const(0);
        sink.i64_store(mem64(
            frame.home_slot_base + (frame.ordinary_home_slots() as u64 + h) * SLOT_SIZE,
        ));
    }

    // Load inputs from frame into locals, and store Ref inputs to their homes.
    // The input value lives at the frame slot its producer wrote it to: the
    // caller fills slot `k` for the k-th input — `execute_token` for a loop
    // entry, `emit_guard_spill`'s positional fail-arg spill for a bridge entry —
    // so read from the POSITIONAL slot `k`, not `ia.index` (a value number that
    // equals `k` for a loop but not for a bridge, whose live-in args carry their
    // trace value numbers). `ValueLocals` maps each body value id to its dense
    // local index. For `key_dispatch` this runs on
    // the key-0 (preamble) path only — past the `br_if` above — so a resuming
    // bridge never scatters its frame-passed label values into the function
    // inputargs' home slots; those stay null-initialized (GC-safe) and the
    // resume loader sets the live label-arg homes.
    for (k, ia) in entry_inputargs.iter().enumerate() {
        let local_idx = value_types.local(ia.index);
        if bridge_entry_arity.is_some() || label_param_entry {
            // Parameter entries carry raw i64 words after frame_ptr. Float
            // values use their IEEE bit pattern, matching the guard boundary.
            sink.local_get(k as u32 + 1);
            if value_types.ty(ia.index) == ValType::F64 {
                sink.f64_reinterpret_i64();
            }
        } else {
            let offset = FRAME_SLOT_BASE + k as u64 * SLOT_SIZE;
            sink.local_get(0).i64_load(mem64(offset));
            if value_types.ty(ia.index) == ValType::F64 {
                sink.f64_reinterpret_i64();
            }
        }
        sink.local_set(local_idx);
        if let Some(h) = ref_homes.home_id(ia.index) {
            sink.local_get(0);
            sink.local_get(local_idx);
            sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
        }
    }
    // Past the entry loader, so the count is one per entry on the same path
    // the inputs are loaded on.
    if let Some((probe, type_idx)) = inline_trip {
        emit_inline_trip_probe(&mut sink, probe, type_idx);
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
    let mut ovf_flag_live = false;
    let mut fused_guard_at: Option<usize> = None;
    // rewrite.py:41-45 `_write_barrier_applied`, represented by
    // RewriteState::wb_applied in rewrite.rs.  A base enters this set after
    // its barrier is emitted or when a nursery allocation produces it; clear
    // at every potentially-collecting op and LABEL so this describes every
    // path reaching the next op, including entry through a LABEL loader.
    let mut wb_applied = indexmap::IndexSet::<OpRef>::new();
    let mut known_lengths = KnownArrayLengths::default();
    let same_as_forwardings = same_as_forwardings(ops, num_vars);

    // A merged region whose closing JUMP names a LABEL published by another
    // module leaves this function the way its out-of-line bridge did — by
    // tail-calling that module — because `br` cannot cross one. Its ops are in
    // this stream, so the transfer has to be selected per operation rather
    // than per function.
    let external_jump_by_op: Vec<Option<&ExternalJump>> =
        if inlined_bridges.iter().any(|b| b.external_jump.is_some()) {
            let mut by_op = vec![None; ops.len()];
            let mut start = bridge_start;
            for bridge in inlined_bridges {
                for slot in &mut by_op[start..start + bridge.ops.len()] {
                    *slot = bridge.external_jump.as_ref();
                }
                start += bridge.ops.len();
            }
            by_op
        } else {
            Vec::new()
        };

    for (op_idx, op) in ops.iter().enumerate() {
        if op.opcode == OpCode::Label || op.opcode.can_malloc() {
            wb_applied.clear();
        }
        known_lengths.observe(op, constants);
        if op.opcode == OpCode::Label && key_dispatch && labels_passed < num_labels {
            // End of the segment before label j (key-0 / earlier-label path).
            // Branch over the resume loader, then close C_j, emit the loader
            // (resume path only), and close B_j. From inside C_j, `br 1`
            // targets B_j's end, skipping the loader.
            // Preserve every non-argument live-in while its pre-LABEL local is
            // still available. Scalar bits use frozen value slots; Refs use
            // the high, GC-rooted capture region so a chained bridge cannot
            // overwrite them with its own low home mapping.
            for &r in &label_resume.per_label[labels_passed] {
                let storage = label_resume
                    .storage(r)
                    .expect("LABEL live-in has assigned capture storage");
                sink.local_get(0);
                emit_resolve(&mut sink, constants, value_types, r);
                sink.i64_store(mem64(label_resume.frame_offset(storage, frame)));
            }
            sink.br(1); // segment done -> past_loader_j, over the resume loader
            sink.end(); // end C_j (the br_table lands here for key j+1)
            // Resume loader: a loop-closing bridge wrote each label arg into
            // frame slot i (positionally, matching the in-loop JUMP move);
            // load them into the label-arg locals and refresh their Ref
            // homes, mirroring the JUMP's ref-home refresh below. The
            // fall-through path skipped this via the `br 1` above.
            for (i, la) in all_label_args[labels_passed].iter().enumerate() {
                if label_param_entry {
                    sink.local_get(i as u32 + 1);
                } else {
                    sink.local_get(0);
                    sink.i64_load(mem64(FRAME_SLOT_BASE + i as u64 * SLOT_SIZE));
                }
                if value_types.ty(la.raw()) == ValType::F64 {
                    sink.f64_reinterpret_i64();
                }
                sink.local_set(value_types.local(la.raw()));
                if let Some(h) = ref_homes.home(*la) {
                    sink.local_get(0);
                    sink.local_get(value_types.local(la.raw()));
                    sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
                }
            }
            // Restore backend-only live-ins after the semantic LABEL args.
            emit_label_capture_restore(
                &mut sink,
                label_resume,
                value_types,
                ref_homes,
                frame,
                labels_passed,
            );
            sink.end(); // end B_j $past_loader
            labels_passed += 1;
        }
        if Some(op_idx) == loop_label_idx {
            sink.loop_(BlockType::Empty);
            for _ in 0..body_region_count {
                sink.block(BlockType::Empty);
            }
            in_loop_body = true;
        }
        // The loop's normal body ends with its JUMP, which branches around all
        // regions. Closing one block before each attached region makes its
        // body reachable only from the guard that branched to that block.
        // Preamble-sourced regions follow the body-sourced ones, and the header
        // `loop` closes before the first of them: their blocks were opened
        // outside it, so their bodies cannot be emitted inside it.
        let in_outside_region = outside_region_count > 0 && op_idx >= outside_start;
        let mut started_body_regions = 0usize;
        let mut started_outside_regions = 0usize;
        if has_loop && op_idx >= bridge_start {
            let mut start = bridge_start;
            for (region, bridge) in inlined_bridges.iter().enumerate() {
                let outside_placed = region >= body_region_count;
                if op_idx == start {
                    if outside_placed && in_loop_body {
                        // A well-formed body ends in a branch, so this return
                        // is unreachable; emit it anyway so a malformed one
                        // cannot walk out of the loop into a region body.
                        sink.local_get(0);
                        sink.return_();
                        sink.end(); // end loop
                        in_loop_body = false;
                    }
                    sink.end();
                }
                if op_idx >= start {
                    if outside_placed {
                        started_outside_regions += 1;
                    } else {
                        started_body_regions += 1;
                    }
                }
                start += bridge.ops.len();
            }
        }
        // Compute the remaining nesting directly from this operation's
        // position, so a label-less stream cannot close a block that was never
        // opened. The body blocks exist only inside the `loop`; the preamble
        // ones are open from the resume `loop` to the end of the function.
        let open_region_blocks = |opened: usize, total: usize| -> Result<u32, BackendError> {
            let remaining = total.checked_sub(opened).ok_or_else(|| {
                BackendError::Unsupported(
                    "wasm backend: inlined bridge region bookkeeping exceeded its open blocks"
                        .into(),
                )
            })?;
            u32::try_from(remaining).map_err(|_| {
                BackendError::Unsupported(
                    "wasm backend: too many inlined bridge regions for wasm branch depth".into(),
                )
            })
        };
        let open_bridge_blocks = if in_loop_body {
            open_region_blocks(started_body_regions, body_region_count)?
        } else {
            0
        };
        let open_outside_blocks =
            open_region_blocks(started_outside_regions, outside_region_count)?;
        // Depth (from statement level) of the enclosing `block` that guard
        // exits `br` to, counted outward from this operation: the
        // (B_j, C_j) pairs a preamble segment still sits inside, the header
        // `loop`, the region blocks open here, the resume `loop`, and block A.
        // Without `key_dispatch` the preamble is at 0, and a straight-line
        // trace uses the universal hot exit block at 0.
        let block_exit_depth = if !has_loop {
            if open_bridge_blocks + open_outside_blocks != 0 {
                return Err(BackendError::Unsupported(
                    "wasm backend: inlined bridge regions require a local loop LABEL".into(),
                ));
            }
            0u32
        } else {
            let label_blocks = if key_dispatch && !in_loop_body {
                2 * (num_labels - labels_passed) as u32
            } else {
                0
            };
            label_blocks
                + u32::from(in_loop_body)
                + open_bridge_blocks
                + open_outside_blocks
                + u32::from(resume_dispatch)
        };
        // A region's block, from this operation's statement level. Region 0 is
        // the innermost of each family, so the ordinal adds to the base.
        let outside_region_base =
            block_exit_depth - open_outside_blocks - u32::from(resume_dispatch);
        let guard_dispatch = BridgeDispatch {
            outside_region_base,
            closed_body_regions: started_body_regions as u32,
            closed_outside_regions: started_outside_regions as u32,
            ..guard_dispatch
        };
        // The guard whose condition the previous op already pushed and tested.
        // `block_exit_depth` is unchanged across the pair: only a LABEL moves
        // `labels_passed` or opens the `loop`, and a guard is neither.
        if fused_guard_at == Some(op_idx) {
            fused_guard_at = None;
            continue;
        }
        if let Some(kind) = cond_kind_of(op.opcode) {
            match next_op_can_accept_cc(
                ops,
                op_idx,
                op.pos.get(),
                &liveness,
                label_resume,
                ref_homes,
            ) {
                Some(guard) => {
                    push_guard_failure_cond(
                        &mut sink,
                        constants,
                        value_types,
                        op,
                        kind,
                        guard.opcode,
                    );
                    emit_guard_if_exit(
                        &mut sink,
                        constants,
                        value_types,
                        guard_idx,
                        guard,
                        block_exit_depth,
                        guard_dispatch,
                    );
                    guard_idx += 1;
                    fused_guard_at = Some(op_idx + 1);
                }
                None => emit_cond(&mut sink, constants, value_types, op, kind),
            }
            // A comparison result is never a Ref, so the store-on-def tail has
            // nothing to do for it.
            continue;
        }
        // The whole-function target is the one a label-less bridge module
        // carries; a region brings its own.
        let jump_external: Option<(u32, u32, Option<(u32, u32)>)> = if op.opcode == OpCode::Jump {
            match external_jump_by_op.get(op_idx).copied().flatten() {
                Some(ext) => Some((ext.slot, ext.key, None)),
                None if !has_loop => {
                    Some((external_jump_slot, external_jump_key, external_jump_wide))
                }
                None => None,
            }
        } else {
            None
        };
        match op.opcode {
            OpCode::Label => {}

            OpCode::Jump if jump_external.is_some() => {
                let (external_jump_slot, external_jump_key, external_jump_wide) = jump_external
                    .expect("the arm guard just established this JUMP has a cross-module target");
                // A JUMP in a trace with no local LABEL closes back into a
                // *separate* loop module (a loop-closing bridge). There is no
                // enclosing `loop` to `br` to, so hand the jump args — the
                // loop's next inputargs, in inputarg order — to the target and
                // `return_call_indirect` its table slot. The tail call reuses
                // this frame instead of nesting, so the loop⇄bridge cycle holds
                // at constant stack depth.
                //
                // A target that published a parameter entry takes them as wasm
                // parameters; otherwise they go through the frame input slots
                // the way `execute_token` fills them. The jump args are this
                // bridge's SSA locals (or constants), and the input slots are a
                // disjoint frame region from any Ref home slot a resolve might
                // load, so storing each pair in turn cannot feed a clobbered
                // read (unlike the local back-edge's parallel move into shared
                // loop locals).
                let jump_args = op.getarglist();
                // Set the resume-at-LABEL dispatch key so a peeled target
                // re-enters at the JUMP's target LABEL — skipping the code
                // before it — instead of re-running the function from its
                // entry. `compile_bridge` resolves the target label ordinal
                // from the JUMP descr and passes `ordinal + 1` here; the
                // target's entry `br_table` lands on that label's resume
                // loader. Harmless for a non-peeled target, which has no
                // dispatch and ignores the slot (`external_jump_key` 0).
                //
                // The key travels through the frame either way: it selects the
                // entry `br_table` arm, which runs before any parameter is
                // read, so it is not one of the values the wide entry takes.
                let store_dispatch_key = |sink: &mut PeepSink<'_, '_>| {
                    sink.local_get(0); // frame_ptr
                    sink.i64_const(external_jump_key as i64); // dispatch key
                    sink.i64_store(mem64(frame.dispatch_key_ofs));
                };
                if let Some((wide_slot, wide_type_idx)) = external_jump_wide
                    .filter(|_| jump_args.len() <= crate::FROZEN_LABEL_PARAM_ARITY)
                {
                    // The target's narrow entry is a shim that loads
                    // `FROZEN_LABEL_PARAM_ARITY` frame slots and tail-calls the
                    // wide one, so storing the args here only to have them read
                    // straight back is a round trip through memory. Both its
                    // entry input loader and every LABEL resume loader read the
                    // parameters, so hand the values over directly.
                    store_dispatch_key(&mut sink);
                    sink.local_get(0); // frame_ptr argument to the loop
                    for k in 0..crate::FROZEN_LABEL_PARAM_ARITY {
                        match jump_args.get(k) {
                            Some(jump_arg) => {
                                emit_resolve(&mut sink, constants, value_types, jump_arg.to_opref())
                            }
                            // `compile_bridge` accepts this JUMP only when its
                            // arity equals the target label's argument count,
                            // and the loader reads exactly that many, so the
                            // parameters past it are never read. They exist to
                            // make one function type serve every arity.
                            None => {
                                sink.i64_const(0);
                            }
                        }
                    }
                    sink.i32_const(wide_slot as i32); // wide table slot
                    sink.return_call_indirect(0, wide_type_idx);
                } else {
                    for (i, jump_arg) in jump_args.iter().enumerate() {
                        sink.local_get(0); // frame_ptr
                        emit_resolve(&mut sink, constants, value_types, jump_arg.to_opref());
                        sink.i64_store(mem64(FRAME_SLOT_BASE + i as u64 * SLOT_SIZE));
                    }
                    store_dispatch_key(&mut sink);
                    sink.local_get(0); // frame_ptr argument to the loop
                    sink.i32_const(external_jump_slot as i32); // table slot
                    sink.return_call_indirect(0, 0); // table 0, type 0: (i32) -> i32
                }
            }

            OpCode::Jump => {
                // The jump rebinds the loop's label args to the jump args — a
                // parallel move. A jump arg may read a target local that another
                // pair overwrites (e.g. the swap `x, y = y, x` → x<-y, y<-x), so
                // resolving-then-storing each pair in turn would feed a clobbered
                // value to a later read. Do all reads first (push every resolved
                // jump arg onto the operand stack), then all writes (pop into the
                // targets in reverse, the stack being LIFO).
                let label_args = find_label_args(ops, op);
                let jump_args = op.getarglist();
                let n = jump_args.len().min(label_args.len());
                // A pair whose jump arg IS its label arg rebinds the local to
                // the value it already holds, so its read/write contributes
                // nothing: every read precedes every write, and no other pair
                // writes the same target (a LABEL's args are distinct boxes,
                // asserted below), so dropping the pair leaves every remaining
                // read and write unchanged. The home-refresh loop below already
                // skips this case for the same reason.
                let moved: Vec<usize> = (0..n)
                    .filter(|&i| {
                        let jarg = jump_args[i].to_opref();
                        jarg.is_constant() || jarg.raw() != label_args[i].raw()
                    })
                    .collect();
                debug_assert!(
                    {
                        let mut seen: Vec<u32> = label_args[..n].iter().map(|a| a.raw()).collect();
                        seen.sort_unstable();
                        seen.windows(2).all(|w| w[0] != w[1])
                    },
                    "LABEL args must be distinct for the identity-pair skip to be a no-op"
                );
                for &i in &moved {
                    let label_arg = label_args[i];
                    if value_types.ty(label_arg.raw()) == ValType::F64 {
                        emit_resolve_f64(
                            &mut sink,
                            constants,
                            value_types,
                            jump_args[i].to_opref(),
                        );
                    } else {
                        emit_resolve(&mut sink, constants, value_types, jump_args[i].to_opref());
                    }
                }
                for &i in moved.iter().rev() {
                    sink.local_set(value_types.local(label_args[i].raw()));
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
                        sink.local_get(value_types.local(la.raw()));
                        sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
                    }
                }
                // A region closing at a LABEL it has no `br` to re-enters the
                // dispatch at the key that lands past that label's resume
                // loader: the parallel move above already left the label args
                // in their locals, so none of them goes through a frame slot.
                // The captures still take the loader's restore. A preamble
                // region sits past the `end` of the header `loop`, so it leaves
                // that way even when it names the header itself.
                let resume_at = if !resume_dispatch {
                    None
                } else if in_outside_region {
                    Some(
                        jump_label_ordinal(ops, op)
                            .filter(|&j| j < num_labels)
                            .ok_or_else(|| {
                                BackendError::Unsupported(
                                    "wasm backend: an outside-loop inline region closes at \
                                     no resumable LABEL"
                                        .into(),
                                )
                            })?,
                    )
                } else {
                    jump_resume_ordinal(ops, op, num_labels)
                };
                match resume_at {
                    Some(j) => {
                        emit_label_capture_restore(
                            &mut sink,
                            label_resume,
                            value_types,
                            ref_homes,
                            frame,
                            j,
                        );
                        sink.i32_const((num_labels + 1 + j) as i32);
                        sink.local_set(resume_key_local);
                        sink.br(block_exit_depth - 1);
                    }
                    None => {
                        sink.br(open_bridge_blocks);
                    }
                }
            }

            OpCode::Finish => {
                emit_guard_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                    // Emitted at statement level: this exit is unconditional,
                    // so no `if` stands between it and the region blocks.
                    0,
                );
                guard_idx += 1;
            }

            // ── Guards ──
            OpCode::GuardTrue => {
                emit_guard_true(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardFalse => {
                emit_guard_false(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardValue => {
                // GUARD_VALUE checks bit-equality against the promoted constant:
                // Value::eq (value.rs) compares floats by to_bits() (0.0 != -0.0,
                // NaN == same-bit NaN, per history.py same_constant), which the
                // dynasm/cranelift siblings implement as an integer bit-compare.
                // emit_resolve pushes an F64 operand's i64 bits, so i64_ne is the
                // correct compare for both int and float — an IEEE f64.ne would
                // wrongly pass -0.0 == +0.0 (and fail NaN == same-bit NaN).
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardNonnull => {
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_eqz();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardIsnull => {
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                // x86/assembler.py _cmp_guard_class:
                //   offset = self.cpu.vtable_offset
                //   if offset is not None: CMP(mem(loc_ptr, offset), classptr)
                //   else:
                //       assert isinstance(loc_classptr, ImmedLoc)
                //       expected_typeid = gc_ll_descr.
                //           get_typeid_from_classptr_if_gcremovetypeptr(...)
                //       _cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
                if let Some(off_usize) = vtable_offset {
                    let off = off_usize as u64;
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64(); // struct ptr (i64) → i32 address
                    // The typeptr (`ob_type`) is a pointer-width field: 4
                    // bytes on wasm32. Reading it as i64 would fold in the
                    // following field's bytes and never match the class
                    // immediate. Load 4 bytes and zero-extend.
                    sink.i64_load32_u(memarg(off, 2));
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i64_ne();
                } else {
                    // x86/assembler.py `_cmp_guard_class` hands the
                    // gcremovetypeptr case to `_cmp_guard_gc_type`, whose
                    // layout keeps the type id in the object's first word.
                    // majit keeps it in the GC header word placed immediately
                    // before the payload — the lower `TYPE_ID_BITS` of
                    // `majit_gc::header::GcHeader`'s `tid_and_flags`, the
                    // address the `GuardGcType`, `GuardIsObject` and
                    // `GuardSubclass` arms read. Under that layout `obj[0]` is
                    // a payload field, so comparing it against a type id
                    // answers a different question.
                    //
                    // Reading the header instead is not enough on its own
                    // here: this arm evaluates the class compare
                    // unconditionally and ORs the null test in afterwards, so
                    // for a NULL receiver `obj - GcHeader::SIZE` addresses
                    // below linear memory and traps instead of failing the
                    // guard — `genop_guard_guard_nonnull_class` avoids that
                    // with a forward jump this arm does not have. Decline: a
                    // frontend that emits GUARD_CLASS configures the vtable
                    // offset (pyre passes `OB_TYPE_OFFSET`), so no trace pays
                    // for the decline.
                    return Err(BackendError::Unsupported(format!(
                        "wasm backend: {:?} with cpu.vtable_offset = None \
                         (gcremovetypeptr) is unsupported; the type id lives \
                         in the GC header and this arm has no null-safe \
                         header compare",
                        op.opcode
                    )));
                }
                if op.opcode == OpCode::GuardNonnullClass {
                    // x86/assembler.py genop_guard_guard_nonnull_class wraps
                    // `_cmp_guard_class` in `CMP(ptr, 1)` plus a forward `B`
                    // jump, so a NULL receiver reaches the guard already
                    // failing and never has its class read. Here the class
                    // compare above has already run — harmlessly, since the
                    // only shape this arm lowers reads the vtable offset, and
                    // a NULL receiver puts that read inside the first page —
                    // so the guard's answer is the disjunction.
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_eqz();
                    sink.i32_or();
                }
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardNoOverflow => {
                // RPython: 0 args — overflow flag implicit from preceding ovf op.
                // If the optimizer proved the operation cannot overflow, the
                // overflow op is absent and this guard is redundant.
                if !ovf_flag_live {
                    guard_idx += 1;
                    continue;
                }
                ovf_flag_live = false;
                sink.local_get(ovf_flag_local);
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardOverflow => {
                assert!(ovf_flag_live, "GuardOverflow without preceding overflow op");
                ovf_flag_live = false;
                sink.local_get(ovf_flag_local);
                sink.i64_eqz();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardNotInvalidated => {
                // x86/assembler.py:4618-4637 parity: the guard site observes
                // the owning loop token's invalidation flag on every entry.
                // On wasm32 the Arc allocation lives in shared linear memory,
                // so its stable pointer is directly addressable by the trace.
                if invalidated_flag_addr != 0 {
                    sink.i32_const(invalidated_flag_addr as i32);
                    // The flag byte is the test: `emit_guard_if_exit` opens
                    // with an `if`, which is already `!= 0`.
                    sink.i32_load8_u(memarg(0, 0));
                    emit_guard_if_exit(
                        &mut sink,
                        constants,
                        value_types,
                        guard_idx,
                        op,
                        block_exit_depth,
                        guard_dispatch,
                    );
                }
                guard_idx += 1;
            }
            OpCode::GuardNotForced => {
                // x86/assembler.py genop_guard_guard_not_forced:
                // `CMP [rbp + jf_descr], 0`, fail when nonzero. `Backend::force`
                // stamps that mark on its way out, so this guard is what turns a
                // force that landed inside the preceding call into a deopt: the
                // trace must not run on holding virtualized fields the force has
                // already written back, and the virtuals `handle_async_forcing`
                // materialized are attached for THIS exit's resume to consume.
                // The bit sits in the upper half of `frame[0]`, so on
                // little-endian wasm32 it is bit 0 of the i32 at frame offset
                // 4; masking it leaves the `!= 0` the `if` already applies.
                const FORCE_TAKEN_HALF_OFS: u64 = 4;
                const _: () = assert!(FORCE_TAKEN_BIT == 1 << 32);
                sink.local_get(0);
                sink.i32_load(memarg(FORCE_TAKEN_HALF_OFS, 2));
                sink.i32_const(1);
                sink.i32_and();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardNotForced2 => {
                // x86/regalloc.py consider_guard_not_forced_2 answers with
                // `assembler.store_force_descr`, not with a branch: unlike
                // GUARD_NOT_FORCED this one is not paired with a preceding call
                // to test, it is what `store_token_in_vable` emits before a
                // FINISH so a force arriving while the virtualizable is still
                // armed can still rebuild a deadframe. Arm, do not test.
                emit_force_arm(
                    &mut sink,
                    constants,
                    value_types,
                    ref_homes,
                    frame,
                    op,
                    exit_index(op, guard_idx),
                    None,
                );
                guard_idx += 1;
            }
            OpCode::GuardNoException => {
                // x86/assembler.py generate_guard_no_exception:
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
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardException => {
                // x86/assembler.py genop_guard_guard_exception:
                //   load pos_exception; CMP expected; guard on equal; then
                //   _store_and_reset_exception: resloc = pos_exc_value;
                //   pos_exception = 0; pos_exc_value = 0.
                let exc_type_addr = crate::jit_exc_type_addr() as i32;
                let exc_value_addr = crate::jit_exc_value_addr() as i32;
                sink.i32_const(exc_type_addr);
                sink.i64_load(mem64(0));
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
                // Success path: capture the caught exception into the result
                // var, then clear both slots.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i32_const(exc_value_addr);
                    sink.i64_load(mem64(0));
                    sink.local_set(value_types.local(vi));
                }
                sink.i32_const(exc_type_addr);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
                sink.i32_const(exc_value_addr);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
            }

            // ── Integer arithmetic ──
            OpCode::IntAdd => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Add),
            OpCode::IntSub => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Sub),
            OpCode::IntMul => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Mul),
            OpCode::IntFloorDiv => {
                emit_binop(&mut sink, constants, value_types, op, BinOp::I64DivS)
            }
            OpCode::IntMod => emit_binop(&mut sink, constants, value_types, op, BinOp::I64RemS),
            OpCode::IntAnd => emit_binop(&mut sink, constants, value_types, op, BinOp::I64And),
            OpCode::IntOr => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Or),
            OpCode::IntXor => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Xor),
            OpCode::IntLshift => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Shl),
            OpCode::IntRshift => emit_binop(&mut sink, constants, value_types, op, BinOp::I64ShrS),
            OpCode::UintRshift => emit_binop(&mut sink, constants, value_types, op, BinOp::I64ShrU),
            // High 64 bits of the unsigned 64×64→128 product. The optimizer
            // emits this for division/modulo-by-constant strength reduction;
            // wasm has no mul-high instruction, so expand via 32-bit split.
            OpCode::UintMulHigh => emit_umulhi(
                &mut sink,
                constants,
                value_types,
                op,
                value_types.last_local(),
            ),

            // Overflow variants: compute result + overflow flag
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf => {
                let binop = match op.opcode {
                    OpCode::IntAddOvf => BinOp::I64Add,
                    OpCode::IntSubOvf => BinOp::I64Sub,
                    OpCode::IntMulOvf => BinOp::I64Mul,
                    _ => unreachable!(),
                };
                // Every overflow form leaves its predicate on the stack, so
                // any of them can hand it straight to an adjacent guard.
                let fused_guard = next_ovf_guard(ops, op_idx);
                ovf_flag_live = match emit_ovf_binop(
                    &mut sink,
                    constants,
                    value_types,
                    op,
                    binop,
                    value_types.last_local(),
                    ovf_flag_local,
                    fused_guard.map(|guard| guard.opcode),
                ) {
                    OvfFlag::Absent => false,
                    OvfFlag::InLocal => true,
                    OvfFlag::FusedCond => {
                        let guard = fused_guard.expect("fused overflow condition requires guard");
                        emit_guard_if_exit(
                            &mut sink,
                            constants,
                            value_types,
                            guard_idx,
                            guard,
                            block_exit_depth,
                            guard_dispatch,
                        );
                        guard_idx += 1;
                        fused_guard_at = Some(op_idx + 1);
                        false
                    }
                };
            }

            // ── Unary ops ──
            OpCode::IntNeg => emit_unary_vi(
                &mut sink,
                constants,
                value_types,
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
                value_types,
                op,
                |s| {
                    s.i64_const(-1);
                },
                |s| {
                    s.i64_xor();
                },
            ),
            // resoperation.py `int_between(a, b, c)` is the three-operand
            // range test `a <= b < c`, signed on all three. jtransform lowers
            // the name directly (`jtransform_opname.rs`), and the bigint
            // compare path mints it as `int_between(-1, i2 >> 48, 1)`, so a
            // trace can carry it.
            OpCode::IntBetween => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i64_le_s();
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    emit_resolve(&mut sink, constants, value_types, op.arg(2).to_opref());
                    sink.i64_lt_s();
                    sink.i32_and();
                    sink.i64_extend_i32_u();
                    sink.local_set(value_types.local(vi));
                }
            }

            // `float_mod` is C `fmod`: the interpreter evaluates it as Rust's
            // `a % b`, which truncates toward zero. Wasm has no float
            // remainder instruction, and `a - (a/b).floor() * b` answers
            // Python's floored `%` instead — a different result for mixed
            // signs, and inexact for a large quotient either way. Decline
            // until there is a host helper to call.
            OpCode::FloatMod => {
                return Err(BackendError::Unsupported(
                    "wasm backend: FloatMod is unsupported (fmod has no wasm \
                     instruction); declining the trace"
                        .to_string(),
                ));
            }

            // ── Extended integer ops ──
            OpCode::IntSignext => {
                // int_signext(val, num_bytes): sign-extend from num_bytes width
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // The static shift below needs num_bytes (arg(1)) as an
                    // emit-time constant. A non-constant width is still a valid
                    // IR shape — int_signext/ii>i is a two-operand blackhole op
                    // and the cranelift backend resolves arg(1) as a runtime
                    // operand — just one this backend does not lower, so decline
                    // for interpreter fallback rather than aborting the compile.
                    let arg1 = op.arg(1).to_opref();
                    let Some(num_bytes) = const_operand_value(constants, arg1) else {
                        return Err(BackendError::Unsupported(format!(
                            "wasm int_signext: non-constant num_bytes operand (raw={})",
                            arg1.raw()
                        )));
                    };
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    let shift = 64 - num_bytes * 8;
                    if shift > 0 && shift < 64 {
                        sink.i64_const(shift);
                        sink.i64_shl();
                        sink.i64_const(shift);
                        sink.i64_shr_s();
                    }
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::IntForceGeZero => {
                // max(val, 0)
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    // `select` answers with the FIRST value when its condition
                    // holds, so the condition is the one that keeps `val`.
                    let tmp_local = value_types.local(vi); // reuse result local as temp
                    sink.local_tee(tmp_local);
                    sink.i64_const(0);
                    sink.local_get(tmp_local);
                    sink.i64_const(0);
                    sink.i64_ge_s();
                    sink.select();
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Float floor/mod ──
            OpCode::FloatFloorDiv => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.f64_div();
                    sink.f64_floor();
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Float/Int conversions ──
            OpCode::CastFloatToInt => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_trunc_sat_f64_s();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::CastIntToFloat => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f64_convert_i64_s();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::ConvertFloatBytesToLonglong => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::ConvertLonglongBytesToFloat => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Pointer/Int conversions ──
            OpCode::CastPtrToInt => {
                // `cast_ptr_to_int` produces `Signed` (a machine word). On
                // wasm32 a pointer is 4 bytes, so the value carried in the i64
                // value ABI must be the 32-bit pointer reinterpreted as a
                // signed word — a sign-extending widen, not the zero-extension
                // a Ref receives on entry (`i64_extend_i32_u` loads, or a Rust
                // residual shim's `ptr as i64`). Without this, a tagged small
                // int with the top payload bit set (`(v<<1)|1` for v<0 or large
                // v, rtagged.py `ll_unboxed_to_int`) reads back with a zero
                // high half, and the trailing arithmetic `IntRshift(,1)` untag
                // (a 64-bit `i64.shr_s`) recovers the wrong value. `i32.wrap` +
                // `i64.extend_i32_s` is a no-op for a real heap pointer (top bit
                // clear on a <2GB linear memory), so this is the width-correct
                // lowering for both tagged and boxed operands.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i64_extend_i32_s();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── SameAs (forwarding) ──
            OpCode::SameAsI | OpCode::SameAsR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::SameAsF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Field access (direct memory operations) ──
            OpCode::GetfieldGcI | OpCode::GetfieldRawI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref()); // struct ptr (i64)
                    sink.i32_wrap_i64(); // convert to i32 address
                    let field_offset = field_offset_from_descr(op);
                    let (size, signed) = field_size_sign_from_descr(op);
                    emit_sized_int_load(&mut sink, field_offset, size, signed);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GetfieldGcR | OpCode::GetfieldRawR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    // Load as i32 (pointer on wasm32) and extend to i64
                    sink.i64_load32_u(memarg(field_offset, 2));
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                emit_write_barrier_if_needed(
                    &mut sink,
                    constants,
                    value_types,
                    jit_call_idx,
                    residual_type_base,
                    wb,
                    op,
                    emitted_write_barrier_base(op, ref_values),
                    &same_as_forwardings,
                    &mut wb_applied,
                    &known_lengths,
                );
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref()); // struct ptr
                sink.i32_wrap_i64();
                let field_offset = field_offset_from_descr(op);
                if field_is_float_from_descr(op) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.f64_store(MemArg {
                        offset: field_offset,
                        align: 3,
                        memory_index: 0,
                    });
                } else {
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref()); // value
                    let size = setfield_store_size_from_descr(op);
                    emit_sized_int_store(&mut sink, field_offset, size);
                }
            }

            // ── Float field access ──
            OpCode::GetfieldGcF | OpCode::GetfieldRawF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    sink.f64_load(MemArg {
                        offset: field_offset,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Array access ──
            OpCode::ArraylenGc => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref()); // array ptr
                    sink.i32_wrap_i64();
                    let (len_offset, len_size) = array_len_layout_from_descr(op);
                    // The length is a word-sized field (`Signed`/`WORD`): read it
                    // at its real width, like `bh_arraylen_gc`. A fixed i64_load
                    // would fold the next field into the high half on wasm32.
                    emit_sized_int_load(&mut sink, len_offset, len_size, false);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcPureI | OpCode::GetarrayitemRawI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let base_size = emit_array_addr(&mut sink, constants, value_types, op);
                    let (item_size, signed) = array_item_access_size_sign(op);
                    emit_sized_int_load(&mut sink, base_size, item_size, signed);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcPureR | OpCode::GetarrayitemRawR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let base_size = emit_array_addr(&mut sink, constants, value_types, op);
                    let (item_size, signed) = array_item_access_size_sign(op);
                    emit_sized_int_load(&mut sink, base_size, item_size, signed);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GetarrayitemGcF | OpCode::GetarrayitemGcPureF | OpCode::GetarrayitemRawF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let base_size = emit_array_addr(&mut sink, constants, value_types, op);
                    sink.f64_load(mem64(base_size));
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                emit_write_barrier_if_needed(
                    &mut sink,
                    constants,
                    value_types,
                    jit_call_idx,
                    residual_type_base,
                    wb,
                    op,
                    emitted_write_barrier_base(op, ref_values),
                    &same_as_forwardings,
                    &mut wb_applied,
                    &known_lengths,
                );
                let base_size = emit_array_addr(&mut sink, constants, value_types, op);
                if array_item_is_float_from_descr(op) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(2).to_opref());
                    sink.f64_store(mem64(base_size));
                } else {
                    emit_resolve(&mut sink, constants, value_types, op.arg(2).to_opref()); // value
                    // A Ref item is pointer-width (4 bytes on wasm32). Storing a
                    // fixed 8 bytes would clobber the next item, or run past the
                    // array end on the last item and corrupt the heap.
                    let (item_size, _signed) = array_item_access_size_sign(op);
                    emit_sized_int_store(&mut sink, base_size, item_size);
                }
            }

            // ── Interior field access ──
            // rewrite.py transform_to_gc_load / unpack_interiorfielddescr:
            // addr = base + index * itemsize + (basesize + field.offset).
            // Wasm skips the GC rewrite, so the GET/SETINTERIORFIELD ops
            // themselves carry that address, matching cranelift's
            // emit_scaled_index_addr rather than being rewritten to
            // GC_LOAD_INDEXED first.
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let field = unpack_interior_field(op);
                    let base = emit_scaled_index_addr(
                        &mut sink,
                        constants,
                        value_types,
                        op.arg(0).to_opref(),
                        op.arg(1).to_opref(),
                        field.item_size,
                        field.offset,
                    );
                    // The opcode, not the descriptor, names the result's
                    // register class — it is what the value local was declared
                    // from — so it picks the load the same way the three
                    // `Getarrayitem` arms do.
                    if op.opcode == OpCode::GetinteriorfieldGcF {
                        sink.f64_load(mem64(base));
                    } else {
                        let (size, signed) = field.access_size_sign();
                        emit_sized_int_load(&mut sink, base, size, signed);
                    }
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => {
                if op.opcode == OpCode::SetinteriorfieldGc {
                    emit_write_barrier_if_needed(
                        &mut sink,
                        constants,
                        value_types,
                        jit_call_idx,
                        residual_type_base,
                        wb,
                        op,
                        emitted_write_barrier_base(op, ref_values),
                        &same_as_forwardings,
                        &mut wb_applied,
                        &known_lengths,
                    );
                }
                let field = unpack_interior_field(op);
                let base = emit_scaled_index_addr(
                    &mut sink,
                    constants,
                    value_types,
                    op.arg(0).to_opref(),
                    op.arg(1).to_opref(),
                    field.item_size,
                    field.offset,
                );
                if field.is_float {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(2).to_opref());
                    sink.f64_store(mem64(base));
                } else {
                    emit_resolve(&mut sink, constants, value_types, op.arg(2).to_opref());
                    emit_sized_int_store(&mut sink, base, field.access_size_sign().0);
                }
            }

            // ── String/Unicode ops (direct memory access) ──
            // strlen/strgetitem/unicodelen/unicodegetitem were lowered with a
            // hardcoded layout (length as an 8-byte load of a 4-byte word field;
            // item as a 1-byte, stride-1 read at a fixed offset) that is wrong for
            // UNICODE (4-byte code units, stride 4) and folds garbage into a str
            // length's high bits — a silent wrong value on wasm, where offset is
            // valid linear memory and does not trap. pyre models strings/unicode
            // as Array(Char) and routes these through the descr-driven
            // GETARRAYITEM/ARRAYLEN paths, so no producer emits these ops (verified
            // with PYRE_DUMP_PERFN_JITCODE: a str-subscript / len / compare / find
            // hot loop traces to GETARRAYITEM, never STRGETITEM). Decline them
            // (interpreter fallback) rather than ship a descr-driven lowering that
            // no trace exercises — a valid but untestable path here.
            OpCode::Strlen | OpCode::Unicodelen | OpCode::Strgetitem | OpCode::Unicodegetitem => {
                return Err(BackendError::Unsupported(format!(
                    "wasm codegen: string/unicode direct-memory op {:?} (no descr-driven layout)",
                    op.opcode
                )));
            }

            // ── GC memory ops ──
            // The indexed forms have the same sole producer as the bare ones
            // below — `GcRewriterImpl`, which this backend does not run — and
            // no opimpl records them, so nothing carries them here. The
            // blackhole wiring for `gc_load_indexed_{i,f}` /
            // `gc_store_indexed_{i,f}` executes them, it does not trace them.
            // Decline (interpreter fallback) rather than aborting the whole
            // compile.
            OpCode::GcLoadIndexedI
            | OpCode::GcLoadIndexedR
            | OpCode::GcLoadIndexedF
            | OpCode::GcStoreIndexed => {
                return Err(BackendError::Unsupported(format!(
                    "wasm codegen: indexed GC op {:?} (no descr-driven layout)",
                    op.opcode
                )));
            }
            // The bare GC_LOAD/GC_STORE forms are produced only by the GC rewrite's
            // load/store lowering (majit-gc/src/rewrite.rs): the true semantics are
            // offset=arg1, size=arg2 (load) / value=arg2, size=arg3 (store), with no
            // FieldDescr attached. The wasm backend runs only the rewrite's
            // reference-constant half (`remove_ref_constants`) and lowers loads,
            // stores, allocations and barriers itself, so these never reach here.
            // The prior lowering read a nonexistent field_offset_from_descr (→ 0)
            // and, for GcStore, stored arg(1) (the offset operand) as the value — a
            // silent miscompile. Panic loudly rather than emit a wrong memory access.
            OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF | OpCode::GcStore => {
                panic!(
                    "wasm backend: {:?} is unsupported (GC_LOAD/GC_STORE); \
                     the load/store GC rewrite must not run for wasm",
                    op.opcode
                );
            }

            // ── Raw memory access ──
            OpCode::RawLoadI | OpCode::RawLoadF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_raw_addr(&mut sink, constants, value_types, op);
                    if op.opcode == OpCode::RawLoadF {
                        sink.f64_load(mem64(0));
                    } else {
                        let (item_size, signed) = array_item_size_sign_from_descr(op);
                        emit_sized_int_load(&mut sink, 0, item_size, signed);
                    }
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::RawStore => {
                emit_raw_addr(&mut sink, constants, value_types, op);
                // `emit_resolve` hands back a Float operand as the `i64` its
                // bits spell, so the width-sized integer store writes the same
                // eight bytes an `f64.store` would.
                emit_resolve(&mut sink, constants, value_types, op.arg(2).to_opref());
                let (item_size, _signed) = array_item_size_sign_from_descr(op);
                emit_sized_int_store(&mut sink, 0, item_size);
            }

            // ── Exception handling ──
            OpCode::SaveException => {
                // x86/assembler.py genop_save_exception:
                //   _store_and_reset_exception → resloc = [pos_exc_value];
                //   [pos_exception] = 0; [pos_exc_value] = 0.
                // The result is the caught exception the resumed handler reads,
                // so it must be written even though the slots themselves are
                // shared with the host: skipping the op leaves the local null.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i32_const(crate::jit_exc_value_addr() as i32);
                    sink.i64_load(mem64(0));
                    sink.local_set(value_types.local(vi));
                }
                sink.i32_const(crate::jit_exc_type_addr() as i32);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
                sink.i32_const(crate::jit_exc_value_addr() as i32);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
            }
            OpCode::SaveExcClass => {
                // x86/assembler.py genop_save_exc_class:
                //   MOV resloc, [pos_exception]
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i32_const(crate::jit_exc_type_addr() as i32);
                    sink.i64_load(mem64(0));
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::RestoreException => {
                // x86/assembler.py _restore_exception:
                //   MOV [pos_exc_value], excvalloc
                //   MOV [pos_exception], exctploc
                sink.i32_const(crate::jit_exc_value_addr() as i32);
                emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                sink.i64_store(mem64(0));
                sink.i32_const(crate::jit_exc_type_addr() as i32);
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_store(mem64(0));
            }

            // ── Conditional calls ──
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                // No-op: the wasm backend does not consume the explicit
                // COND_CALL_GC_WB / COND_CALL_GC_WB_ARRAY barrier ops. It emits
                // the write barrier inline at each ref-store instead
                // (`write_barrier_base` + `emit_write_barrier`).
            }
            OpCode::CondCallN => {
                // x86/assembler.py `genop_discard_cond_call`: TEST cond; JZ
                // skip; CALL. The predicate is arg 0, the callee is arg 1, and
                // the rest are the call's own arguments. The callee reaches the
                // host through the same trampoline `CallN` uses; none of the
                // residual `call_indirect` families admits this opcode, so
                // `has_trampoline_calls` always imports `jit_call` for it.
                //
                // `do_conditional_call` asserts the callee forces no virtual or
                // virtualizable, so unlike the CALL arm this needs no force
                // bracket.
                let jit_call =
                    jit_call_idx.expect("COND_CALL op present but jit_call not imported");
                let func = op.arg(1).to_opref();
                let call_args = &op.getarglist()[2..];
                // The predicate is a full word: `i32.wrap_i64` would read a
                // value whose only set bits are above 32 as false, so the test
                // has to be `i64.eqz` and the call has to sit in the else arm.
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_eqz();
                sink.if_(BlockType::Empty);
                sink.else_();
                emit_call_area_addr(&mut sink);
                emit_resolve(&mut sink, constants, value_types, func);
                sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                emit_call_area_addr(&mut sink);
                sink.i64_const(call_args.len() as i64);
                sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                for (i, arg) in call_args.iter().enumerate() {
                    emit_call_area_addr(&mut sink);
                    emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + i as u64 * SLOT_SIZE));
                }
                emit_jit_call(&mut sink, jit_call);
                // COND_CALL sits inside the CALL opcode range, so a Ref living
                // across it already owns a home. Only the arm that called can
                // have collected, so the reload belongs on it: after the `if`
                // it would run on the untaken arm too, re-reading slots that
                // nothing moved, once per iteration of a loop whose whole
                // reason for a COND_CALL is that the call is rare.
                if call_can_collect(op) {
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        ref_homes,
                        &liveness,
                        op_idx,
                        None,
                        frame,
                    );
                }
                sink.end();
            }
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                // x86/regalloc.py `consider_cond_call`, COND_CALL_VALUE arm:
                // "Calls the function when args[0] is equal to 0 or NULL.
                // Returns the result from the function call if done, or args[0]
                // if it was not 0/NULL." Upstream forces the result into
                // args[0]'s register and lets the call overwrite it; the result
                // local plays that register's part here.
                //
                // The operand roles are COND_CALL's: predicate, callee, then the
                // call's own arguments. Sharing the generic CALL arm read args[0]
                // as the callee and args[1] as the first argument, and called
                // unconditionally.
                //
                // `do_conditional_call` asserts the callee forces no virtual or
                // virtualizable, so unlike the CALL arm this needs no force
                // bracket.
                let jit_call =
                    jit_call_idx.expect("COND_CALL_VALUE op present but jit_call not imported");
                let vi = op.pos.get().raw();
                let has_result = !OpRef::raw_is_constant(vi);
                let cond = op.arg(0).to_opref();
                let func = op.arg(1).to_opref();
                let call_args = &op.getarglist()[2..];

                if has_result {
                    emit_resolve(&mut sink, constants, value_types, cond);
                    sink.local_set(value_types.local(vi));
                }
                // The predicate is a full word: `i32.wrap_i64` would read a
                // value whose only set bits are above 32 as NULL and call on a
                // live one, so the test has to be `i64.eqz`.
                emit_resolve(&mut sink, constants, value_types, cond);
                sink.i64_eqz();
                sink.if_(BlockType::Empty);
                emit_call_area_addr(&mut sink);
                emit_resolve(&mut sink, constants, value_types, func);
                sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                emit_call_area_addr(&mut sink);
                sink.i64_const(call_args.len() as i64);
                sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                for (i, arg) in call_args.iter().enumerate() {
                    emit_call_area_addr(&mut sink);
                    emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + i as u64 * SLOT_SIZE));
                }
                emit_jit_call(&mut sink, jit_call);
                if has_result {
                    // Int and Ref are the only result types
                    // `do_conditional_call` mints this opcode for, so the
                    // trampoline's word needs no float reinterpretation.
                    emit_call_area_addr(&mut sink);
                    sink.i64_load(mem64(STATIC_CALL_RESULT_OFS));
                    sink.local_set(value_types.local(vi));
                }
                // Only the arm that called can have collected, so the reload
                // sits on it rather than after the `if`.
                if call_can_collect(op) {
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        ref_homes,
                        &liveness,
                        op_idx,
                        has_result.then_some(vi),
                        frame,
                    );
                }
                sink.end();
            }

            // x86/assembler.py genop_guard_guard_gc_type:
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
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
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
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i64_ne();
                    emit_guard_if_exit(
                        &mut sink,
                        constants,
                        value_types,
                        guard_idx,
                        op,
                        block_exit_depth,
                        guard_dispatch,
                    );
                }
                guard_idx += 1;
            }
            // x86/assembler.py genop_guard_guard_is_object.
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
                // assembler.py MOV32 loc_typeid, mem(loc_object, 0).
                // majit's GC header sits at obj - GcHeader::SIZE; the
                // typeid occupies the lower TYPE_ID_BITS of that word.
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
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
                sink.i32_wrap_i64();
                // Stack: [..., loc_type_info(i32 addr)]

                // assembler.py:1940 TEST8 [loc_infobits], IS_OBJECT_FLAG. The
                // `offset=infobits_offset` of the address computation above is
                // a constant, so it rides in the load's own MemArg.
                sink.i32_load8_u(memarg(guard_gc_type_info.infobits_offset as u64, 0));
                sink.i32_const(guard_gc_type_info.is_object_flag as i32);
                sink.i32_and();
                // assembler.py:1942 guard_success_cc = Conditions['NZ']:
                // guard passes when byte & flag != 0; fail when == 0.
                sink.i32_eqz();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            // x86/assembler.py genop_guard_guard_subclass.
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
                // history.py — inline-Const carries its class pointer directly.
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
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(vtable_off as u64));
                    sink.i32_wrap_i64();
                    // subclassrange_min is an 8-byte i64 on every target
                    // (pyobject.rs `PyType::subclassrange_min: AtomicI64`); read
                    // the full field width, not the wasm32 4-byte `usize`, or the
                    // guard truncates/sign-extends the object's min.
                    emit_sized_int_load(
                        &mut sink,
                        offset2 as u64,
                        std::mem::size_of::<i64>(),
                        true,
                    );
                } else {
                    // assembler.py:1957-1969 gcremovetypeptr path.
                    //     MOV32 loc_tmp, mem(loc_object, 0)
                    //     base_type_info, shift_by, sizeof_ti = ...
                    //     MOV loc_tmp, [base_type_info
                    //         + (loc_tmp << shift_by)
                    //         + sizeof_ti + offset2]
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
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
                    // 8-byte i64 subclassrange_min (see the vtable path above).
                    emit_sized_int_load(&mut sink, 0, std::mem::size_of::<i64>(), true);
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
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardAlwaysFails => {
                // This guard always exits, and what it exits INTO is the
                // interpreter: it is the cut a segmented trace ends with
                // (`rewrite.py:419-426` lowers it to `GUARD_VALUE(SAME_AS_I(0),
                // 1)` for the backends that run the GC rewrite, so those reach
                // the ordinary GUARD_VALUE path). This backend does not run
                // that rewrite, so the raw opcode arrives here and has to
                // publish its own fail args — the resume rebuilds the frame
                // from them, and an exit that writes none leaves the
                // interpreter reading whatever the slots last held.
                emit_guard_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                    // Emitted at statement level: this guard has no passing
                    // outcome, so no `if` stands between it and the region
                    // blocks. A loop-closing bridge merges into exactly this
                    // exit when the owner is a segmented trace's cut.
                    0,
                );
                guard_idx += 1;
            }
            // `reached_loop_header` mints this op only to donate its
            // `rd_resume_position` to the guards `jump_to_existing_trace` and
            // `inline_short_preamble` stamp; both `optimize_GUARD_FUTURE_CONDITION`
            // arms then consume it into `patchguardop` and emit nothing, which is
            // why nothing under `rpython/jit/backend` lowers it. Reaching a
            // backend means the optimizer did not consume it, and there is
            // nothing correct to lower it to: it is nullary, so there is no
            // condition to test, and an exit publishing neither `frame[0]` nor
            // the fail args leaves the resume reading whatever the frame last
            // held.
            OpCode::GuardFutureCondition => {
                return Err(BackendError::Unsupported(
                    "wasm backend: GuardFutureCondition is unsupported (the optimizer \
                     consumes it into patchguardop and no backend lowers it); \
                     declining the trace"
                        .to_string(),
                ));
            }

            // ── Quasi-immutable / record / assert ──
            OpCode::QuasiimmutField
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueI
            | OpCode::RecordExactValueR
            | OpCode::RecordKnownResult
            | OpCode::AssertNotNone => {
                // Metadata-only ops, no codegen needed. `RecordKnownResult`
                // is an optimizer hint with no backend arm upstream at all —
                // `optimize_RECORD_KNOWN_RESULT` consumes it and simplify
                // drops it — so one that still reaches here owes no code.
            }

            // ── The rstr IR family: declined, not lowered ──
            //
            // Nothing on the pyre side lowers Python strings to these opcodes
            // — there is no `Newstr` / `Strsetitem` / `Copystrcontent`
            // producer outside majit's ported RPython infrastructure, so
            // string work stays in residual interpreter calls and none of
            // these reach a wasm trace today. The setitem pair is named here
            // rather than left to the catch-all so the whole family declines
            // from one place.
            //
            // They are declined rather than left as silent no-ops. The old
            // arms emitted nothing for the copies and, for the allocations, a
            // trampoline call whose host side has no string allocator (the
            // runner's `func_ptr == 0` sentinel returns 0). Either shape
            // produces wrong code the moment the opcode becomes reachable,
            // with no signal — the same failure mode as the dropped
            // `non_moving` flag. A decline keeps the trace interpreted and
            // says so.
            OpCode::Newstr
            | OpCode::Newunicode
            | OpCode::Copystrcontent
            | OpCode::Copyunicodecontent
            | OpCode::Strsetitem
            | OpCode::Unicodesetitem => {
                return Err(BackendError::Unsupported(format!(
                    "wasm backend: {:?} is unsupported (no rstr lowering); \
                     declining the trace",
                    op.opcode
                )));
            }

            // ── Misc ops ──
            OpCode::NurseryPtrIncrement => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i64_add();
                    sink.local_set(value_types.local(vi));
                }
            }
            // Emitted by `RawBufferPtrInfo::_force_elements` after a
            // `RAW_MALLOC_VARSIZE_CHAR`. The sole carrier of that oopspec is
            // `_cffi_backend::cdataobj::raw_malloc_varsize_char`, and
            // `_cffi_backend` is not compiled for wasm32, so this does not
            // reach a wasm trace. Declined rather than skipped:
            // the allocation helpers do return 0 on OOM (that is how
            // `alloc_with_type` drives this op on the native backends), and an
            // unchecked null is worse on wasm than on native — address 0 is
            // ordinary linear memory, so the following stores would corrupt it
            // silently instead of trapping.
            OpCode::CheckMemoryError => {
                return Err(BackendError::Unsupported(
                    "wasm backend: CheckMemoryError is unsupported (a null allocation \
                     result would be stored into valid linear memory at address 0); \
                     declining the trace"
                        .to_string(),
                ));
            }
            // Emitted only by the GC rewrite pass (`_gen_zero_array`), which
            // wasm does not run, so this cannot reach here. Skipping it used to
            // be harmless only by accident: wasm allocation hands back zeroed
            // memory on both routes (`nursery.rs` memsets the whole nursery on
            // reset under `target_arch = "wasm32"`, and `finish_alloc_in_oldgen`
            // zero-fills the payload). That is a property of the allocator, not
            // of this op, so do not silently rely on it.
            OpCode::ZeroArray => {
                return Err(BackendError::Unsupported(
                    "wasm backend: ZeroArray is unsupported (it is a GC-rewrite op and \
                     wasm does not run the rewrite); declining the trace"
                        .to_string(),
                ));
            }
            OpCode::LoadFromGcTable => {
                // `assembler.py` `genop_load_from_gc_table`: the arg is a
                // `ConstInt(index)` into the per-loop `GcTable`
                // (`remove_ref_constants`, rewrite.py `remove_constptr`)
                // whose base is baked absolute. The table is a plain guest heap
                // allocation, so `base + index*WORD` is an ordinary linear-memory
                // address; the collector forwards the slot in place, so the load
                // reads the reference at its current address.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let index = resolve_const_bits(constants, op.arg(0).to_opref());
                    let base = gc_table_bases.get(&vi).copied().unwrap_or(gc_table_base);
                    let slot =
                        base as u64 + index as u64 * std::mem::size_of::<majit_ir::GcRef>() as u64;
                    sink.i32_const(slot as i32);
                    sink.i64_load32_u(memarg(0, 2));
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── CALL_ASSEMBLER ──
            // Lower the call into an in-module `call_indirect` into its compiled
            // callee loop instead of a
            // host round-trip. A fresh callee frame is allocated as a real
            // GC-managed nursery `JitFrame` (push_jf-rooted on the jitframe
            // shadow stack; traced by its OWN per-frame gcmap covering
            // its input + home Ref slots), the descriptor inputs are written to its
            // input slots, the loop runs on it (recursing through this same arm
            // for deeper levels), then the result Ref is read back from output
            // slot 0. `compile_loop` and `compile_bridge` validate the descriptor
            // and target metadata before enabling this arm. The callee
            // `call_indirect` runs a full compiled loop, which allocates and
            // collects; each live callee frame is
            // self-described by its gcmap so a collection forwards its Refs (no
            // shared-arena single-stride walker). This bridge's own wasm-local
            // Refs still hold pre-call (from-space) addresses on return, so
            // reload them from the (forwarded) homes after the call.
            opcode if opcode.is_call_assembler() && ca.emit_ca => {
                emit_force_bracket_before_call(
                    &mut sink,
                    constants,
                    value_types,
                    ref_homes,
                    frame,
                    ops,
                    op_idx,
                    guard_idx,
                );
                let vi = op.pos.get().raw();
                let descr = op
                    .getdescr()
                    .expect("CALL_ASSEMBLER op must carry a descriptor");
                let op_token = descr
                    .as_call_descr()
                    .and_then(|descr| descr.call_target_token())
                    .expect("CALL_ASSEMBLER op must carry a callee token");
                let tgt = ca
                    .targets
                    .get(&op_token)
                    .expect("CA op target must be registered");
                let dispatch_entry = tgt.dispatch_entry as i32;
                sink.i32_const(dispatch_entry);
                sink.i32_load(mem32(crate::failguard::WASM_CA_DISPATCH_TARGET_PTR_OFS));
                sink.local_tee(ca_target_local);
                sink.i32_eqz();
                sink.if_(BlockType::Empty);
                sink.unreachable();
                sink.end();

                // A terminally-declined target cannot be restarted from the
                // CALL_ASSEMBLER reds: these are loop-header live-ins, not a
                // function-entry PyFrame or necessarily the function's call
                // arguments.  Continue through the orthodox CA frame path
                // below instead.  It marshals every live-in into a callee
                // JitFrame and its non-finish path blackhole-resumes the
                // callee correctly.  This is temporarily more expensive in
                // the bounded caller-invalidation window; deopting the outer
                // trace at the Python CALL needs resume metadata that a
                // CALL_ASSEMBLER op does not currently carry.

                // Allocate the callee frame as a GC JitFrame
                // (`wasm_jit_ca_alloc_frame(frame_bytes, gcmap_ptr)` — a
                // collecting `(i64,i64)->i64` table entry whose caller's Refs
                // are rooted in frame homes, so it lowers like an eligible
                // residual call when the type family is declared; otherwise via
                // the jit_call trampoline).
                // `ca_cfp_local = frame_base + FIRST_ITEM_OFFSET` is the
                // bespoke-layout frame pointer — every `mem64(OFS)` below is
                // relative to it, exactly as the source loop reads its local 0.
                // The tmp callback and the real loop may have different frame
                // depths.  PyPy updates the old token's frame info during
                // redirect; wasm mirrors that by loading both allocation
                // fields from the old token's stable dispatch entry.  Keep the
                // helper allocation here: the former inline path required a
                // compile-time size and therefore could not honor a later
                // depth increase.
                if let Some(base) = residual_type_base {
                    sink.local_get(ca_target_local);
                    sink.i64_load32_u(memarg(crate::failguard::WASM_CA_TARGET_FRAME_BYTES_OFS, 2));
                    sink.local_get(ca_target_local);
                    sink.i64_load(mem64(crate::failguard::WASM_CA_TARGET_GCMAP_PTR_OFS));
                    sink.i32_const(ca.ca_alloc_fn_ptr as i32);
                    sink.call_indirect(0, base + 2);
                } else {
                    let jit_call =
                        jit_call_idx.expect("CA arm needs jit_call for the frame trampolines");
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(ca.ca_alloc_fn_ptr);
                    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(2);
                    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.local_get(ca_target_local);
                    sink.i64_load32_u(memarg(crate::failguard::WASM_CA_TARGET_FRAME_BYTES_OFS, 2));
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.local_get(ca_target_local);
                    sink.i64_load(mem64(crate::failguard::WASM_CA_TARGET_GCMAP_PTR_OFS));
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + SLOT_SIZE));
                    emit_jit_call(&mut sink, jit_call);
                    emit_call_area_addr(&mut sink);
                    sink.i64_load(mem64(STATIC_CALL_RESULT_OFS));
                }
                sink.i32_wrap_i64();
                sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
                sink.i32_add();
                sink.local_set(ca_cfp_local);
                // The collecting callee allocation ran while this invocation's
                // own frame was the shadow-stack top. Now that the callee is
                // pushed, reload local 0 from the entry beneath it before
                // resolving inputs through local-0-relative homes. The
                if let (Some(_base), Some(inline)) = (residual_type_base, ca.inline) {
                    emit_ca_reload_caller(&mut sink, inline.jf_top_addr);
                    sink.local_set(0);
                } else if let Some(base) = residual_type_base {
                    sink.i32_const(ca.ca_reload_caller_fn_ptr as i32);
                    sink.call_indirect(0, base);
                    sink.i32_wrap_i64();
                    sink.local_set(0);
                }
                emit_reload_ca_input_refs_from_homes(
                    &mut sink,
                    value_types,
                    ref_homes,
                    ref_values,
                    op,
                    frame,
                );
                // dispatch key = 0: run the loop from its entry (preamble), not a
                // LABEL resume — this is a fresh call.
                sink.local_get(ca_cfp_local);
                sink.local_get(ca_target_local);
                sink.i32_load(mem32(crate::failguard::WASM_CA_TARGET_DISPATCH_KEY_OFS_OFS));
                sink.i32_add();
                sink.i64_const(0);
                sink.i64_store(mem64(0));
                // Marshal the descriptor's uniform i64 Int/Ref ABI inputs into
                // the callee's positional frame slots.
                for (arg_index, arg) in op.getarglist().iter().enumerate() {
                    sink.local_get(ca_cfp_local);
                    emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    sink.i64_store(mem64(FRAME_SLOT_BASE + arg_index as u64 * SLOT_SIZE));
                }
                // Run the callee loop on F'; discard the returned frame_ptr.
                sink.local_get(ca_cfp_local);
                // The immutable target snapshot was loaded before allocating
                // F', so this function is exactly the one whose geometry and
                // gcmap initialized that frame.
                sink.local_get(ca_target_local);
                sink.i32_load(mem32(crate::failguard::WASM_CA_TARGET_FUNC_HANDLE_OFS));
                sink.local_tee(ca_fi_local);
                sink.i32_eqz();
                sink.if_(BlockType::Empty);
                sink.unreachable();
                sink.end();
                sink.local_get(ca_fi_local);
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
                    sink.call_indirect(0, base);
                } else {
                    let jit_call =
                        jit_call_idx.expect("CA arm needs jit_call for the frame trampolines");
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(ca.ca_reload_fn_ptr);
                    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(0);
                    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                    emit_jit_call(&mut sink, jit_call);
                    emit_call_area_addr(&mut sink);
                    sink.i64_load(mem64(STATIC_CALL_RESULT_OFS));
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
                // `_call_assembler_check_descr` — every clean finish of this
                // result kind writes the one `done_with_this_frame_descr_<kind>`
                // the cpu was handed, so the check is a compare against that
                // single value. A raising callee writes
                // `exit_frame_with_exception_descr_ref` and a guard deopt writes
                // its own exit, so both fail this compare and take the helper
                // path, which is what propagates the exception.
                sink.local_get(ca_fi_local);
                sink.i32_const(crate::failguard::done_with_this_frame_exit_index(
                    op.opcode.result_type(),
                ) as i32);
                sink.i32_eq();
                sink.if_(BlockType::Result(ValType::I64));
                // clean finish: result Ref = F'[1] (output slot 0).
                sink.local_get(ca_cfp_local);
                sink.i64_load(mem64(FRAME_SLOT_BASE));
                sink.else_();
                // deopt: wasm_ca_resume_deopt(frame_ptr: i64, compiled_ptr: i64).
                sink.local_get(ca_cfp_local);
                sink.i64_extend_i32_u();
                sink.local_get(ca_target_local);
                sink.i64_load32_u(memarg(crate::failguard::WASM_CA_TARGET_COMPILED_PTR_OFS, 2));
                sink.i32_const(ca.deopt_helper_slot as i32);
                // call_indirect(table_index, type_index): the shared table is 0.
                sink.call_indirect(0, ca_helper_type_idx);
                sink.end();
                // The recursive call or deopt helper may have collected and
                // moved this invocation's own frame. Reload it before the pop
                // trampoline and post-call home loads address local 0. As above,
                // the trampoline-only configuration retains its earlier stale-
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
                    sink.local_set(value_types.local(vi));
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
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(ca.ca_pop_fn_ptr);
                    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(1);
                    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.local_get(ca_cfp_local);
                    sink.i64_extend_i32_u();
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS));
                    emit_jit_call(&mut sink, jit_call);
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
                emit_reload_refs_from_homes(
                    &mut sink,
                    value_types,
                    ref_homes,
                    &liveness,
                    op_idx,
                    skip,
                    frame,
                );
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
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantN
            | OpCode::CallLoopinvariantF
            | OpCode::CallPureF
            | OpCode::CallMayForceF
            | OpCode::CallAssemblerF
            | OpCode::CallReleaseGilF => {
                emit_force_bracket_before_call(
                    &mut sink,
                    constants,
                    value_types,
                    ref_homes,
                    frame,
                    ops,
                    op_idx,
                    guard_idx,
                );
                let vi = op.pos.get().raw();
                let can_collect = call_can_collect(op);

                // pyjitpl.py `direct_call_release_gil` records CALL_RELEASE_GIL_*
                // as `[savebox, funcbox] + argboxes[1:]`, so its callee is arg 1
                // and its own arguments start at 2. Every other CALL keeps the
                // callee at arg 0.
                let func_ofs = usize::from(matches!(
                    op.opcode,
                    OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN
                ));
                let func_ptr_ref = op.arg(func_ofs).to_opref();

                // Direct in-module residual call: skip the `jit_call` host hop and
                // `call_indirect` the callee's table slot with a static
                // `(i64×n)->i64` type. The residual ABI is uniformly i64 for
                // Int/Ref args+result, so args/result move on the wasm stack with
                // no marshalling and no call-area traffic. A direct target may
                // collect or force, so reload local 0 and its live Ref homes on
                // return. Falls back below when ineligible.
                if let (Some(base), Some(nargs)) =
                    (residual_type_base, residual_call_i64_arity(op, constants))
                {
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    // func_ptr (arg 0) is the table slot — wrap to i32 index.
                    emit_resolve(&mut sink, constants, value_types, func_ptr_ref);
                    sink.i32_wrap_i64();
                    // call_indirect(table_index, type_index): table 0, type for arity n.
                    sink.call_indirect(0, base + nargs as u32);
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(value_types.local(vi));
                    } else {
                        sink.drop(); // value-producing call whose result is unused
                    }
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            ref_homes,
                            &liveness,
                            op_idx,
                            (!OpRef::raw_is_constant(vi)).then_some(vi),
                            frame,
                        );
                    }
                    // store-on-def (end of loop) homes a Ref result, so the
                    // direct path must NOT `continue` past it.
                } else if let (Some(base), Some(nargs)) = (
                    residual_type_base,
                    residual_call_void_word_arity(op, constants),
                ) {
                    // Direct in-module word-ABI void residual call: the callee
                    // really is `(i64×n)->i64` (descr result_size == 8), so use
                    // the i64 family and drop the dummy result.
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    emit_resolve(&mut sink, constants, value_types, func_ptr_ref);
                    sink.i32_wrap_i64();
                    sink.call_indirect(0, base + nargs as u32);
                    sink.drop();
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            ref_homes,
                            &liveness,
                            op_idx,
                            None,
                            frame,
                        );
                    }
                } else if let Some((sig, &type_idx)) = residual_call_typed_sig(op, constants)
                    .and_then(|sig| {
                        typed_residual_type_indices
                            .get(&sig)
                            .map(|type_idx| (sig, type_idx))
                    })
                {
                    // Direct in-module typed residual call with the
                    // descr-derived mixed `(i64/f64...) -> i64/f64` signature.
                    let (params, _) = &sig;
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    debug_assert_eq!(call_args.len(), params.len());
                    for (arg, ty) in call_args.iter().zip(params) {
                        if *ty == ValType::F64 {
                            emit_resolve_f64(&mut sink, constants, value_types, arg.to_opref());
                        } else {
                            emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                        }
                    }
                    // func_ptr (arg 0) is the table slot — wrap to i32 index.
                    emit_resolve(&mut sink, constants, value_types, func_ptr_ref);
                    sink.i32_wrap_i64();
                    sink.call_indirect(0, type_idx);
                    // A void callee leaves nothing on the stack, so there is
                    // neither a local to home it in nor a value to drop.
                    let homed = if sig.1.is_none() {
                        None
                    } else if !OpRef::raw_is_constant(vi) {
                        sink.local_set(value_types.local(vi));
                        Some(vi)
                    } else {
                        sink.drop(); // value-producing call whose result is unused
                        None
                    };
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            ref_homes,
                            &liveness,
                            op_idx,
                            homed,
                            frame,
                        );
                    }
                } else if let (Some(base), Some(nargs)) = (
                    true_void_residual_type_base,
                    residual_call_void_true_arity(op, constants),
                ) {
                    // Direct in-module true-void residual call: the callee is
                    // `(i64×n)->()` (descr result_size == 0), so the call has no
                    // result to drop.
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    emit_resolve(&mut sink, constants, value_types, func_ptr_ref);
                    sink.i32_wrap_i64();
                    sink.call_indirect(0, base + nargs as u32);
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            ref_homes,
                            &liveness,
                            op_idx,
                            None,
                            frame,
                        );
                    }
                } else {
                    let jit_call = jit_call_idx.expect("CALL op present but jit_call not imported");

                    let call_args = &op.getarglist()[func_ofs + 1..];

                    // Store func_ptr to call area
                    emit_call_area_addr(&mut sink);
                    emit_resolve(&mut sink, constants, value_types, func_ptr_ref);
                    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));

                    // Store num_args
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(call_args.len() as i64);
                    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));

                    // Store each arg
                    for (i, arg) in call_args.iter().enumerate() {
                        emit_call_area_addr(&mut sink);
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                        sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + i as u64 * SLOT_SIZE));
                    }

                    // Call trampoline
                    emit_jit_call(&mut sink, jit_call);

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
                        emit_call_area_addr(&mut sink);
                        sink.i64_load(mem64(STATIC_CALL_RESULT_OFS));
                        if value_types.ty(vi) == ValType::F64 {
                            sink.f64_reinterpret_i64();
                        }
                        sink.local_set(value_types.local(vi));
                    }
                    // Mirror the direct path: a trampoline residual call may force and collect.
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            ref_homes,
                            &liveness,
                            op_idx,
                            (!is_void && !OpRef::raw_is_constant(vi)).then_some(vi),
                            frame,
                        );
                    }
                }
            }

            // ── Allocation (via trampoline — treated as CALL) ──
            // llmodel.py bh_new* parity: a `New*` survives
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
                let (size, type_id, vtable) = sd.map_or_else(
                    || missing_layout_descr("size descr", op),
                    |sd| (sd.size() as i64, sd.type_id() as i64, sd.vtable()),
                );
                // `(w_class_offset, w_class)` — the `PyObject.w_class` field
                // and the class pointer instances of this type carry
                // (`get_instantiate(vtable_type)`). `fuse_boxing_alloc` drops
                // the boxing ctor's `ob_header` stores expecting the runtime to
                // stamp both `ob_type` and `w_class` from the size descr; the
                // vtable write above covers `ob_type`, this covers `w_class`.
                let w_class_init = sd.and_then(|sd| {
                    sd.w_class_obj().and_then(|w_class| {
                        sd.class_word_field()
                            .map(|fd| (fd.offset() as u64, w_class))
                    })
                });

                // `rewrite.rs handle_new`: a `non_moving` descr declines the
                // nursery outright — both the inline bump and the collecting
                // helper — and allocates through the old-generation twin, whose
                // address is the only thing that differs from the nursery call.
                let non_moving = sd.is_some_and(|sd| sd.non_moving());
                let alloc_fn_ptr = if non_moving {
                    alloc.new_oldgen_fn_ptr
                } else {
                    alloc.new_fn_ptr
                };
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
                let inline_nursery = nursery.filter(|_| !non_moving).filter(|na| {
                    total_size < na.large_threshold
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
                    // The sum is the committed `nursery_free`, so keep it
                    // rather than adding it again on the arm that takes it.
                    sink.local_tee(alloc_size_local);
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
                        value_types,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                    sink.else_();
                    // Commit: *nursery_free = free + total.
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
                    // Result payload pointer = free + header size.
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(majit_gc::header::GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(value_types.local(vi));
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
                        sink.local_set(value_types.local(vi));
                    } else {
                        sink.drop();
                    }
                } else {
                    let jit_call = jit_call_idx.expect("New op present but jit_call not imported");
                    // func_ptr = wasm_jit_alloc
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(alloc_fn_ptr);
                    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                    // num_args = 2
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(2);
                    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                    // arg0 = type_id
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(type_id);
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS));
                    // arg1 = size
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(size);
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + SLOT_SIZE));
                    // call trampoline
                    emit_jit_call(&mut sink, jit_call);

                    if !OpRef::raw_is_constant(vi) {
                        // result pointer
                        emit_call_area_addr(&mut sink);
                        sink.i64_load(mem64(STATIC_CALL_RESULT_OFS));
                        sink.local_set(value_types.local(vi));
                    }
                }

                // The check `rewrite.py` `_gen_call_malloc_gc` puts after a
                // collecting malloc: the vtable and class-word stores below
                // address the result directly and must not run on a NULL.
                emit_memory_error_check(
                    &mut sink,
                    value_types,
                    vi,
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                );
                if !OpRef::raw_is_constant(vi) {
                    // llmodel.py write_int_at_mem(res, vtable_offset,
                    // WORD, vtable). The `ob_type` field is pointer-width: 4
                    // bytes on wasm32 (GuardClass reads it as i32), so store
                    // the low 32 bits to avoid clobbering the next field.
                    let write_vtable = op.opcode == OpCode::NewWithVtable
                        && vtable != 0
                        && vtable_offset.is_some();
                    if write_vtable {
                        let vt_off = vtable_offset.unwrap() as u64;
                        sink.local_get(value_types.local(vi));
                        sink.i32_wrap_i64();
                        sink.i32_const(vtable as i32);
                        sink.i32_store(MemArg {
                            offset: vt_off,
                            align: 2,
                            memory_index: 0,
                        });
                    }
                    // Stamp `w_class = get_instantiate(vtable_type)` so the
                    // materialized builtin box carries the class pointer
                    // OptVirtualize folded its `w_class` header reads to. Mirrors
                    // dynasm `genop_new_with_vtable` (aarch64/assembler.rs).
                    // Pointer-width (4 bytes on wasm32): store the low 32 bits.
                    // Without it the nursery-zeroed `w_class` stays 0 and the
                    // promoted-`w_class` GuardValue fails every iteration on any
                    // escaping-builtin loop (e.g. `while: lst.append(i)`).
                    if op.opcode == OpCode::NewWithVtable
                        && let Some((w_class_offset, w_class)) = w_class_init
                        && w_class != 0
                    {
                        sink.local_get(value_types.local(vi));
                        sink.i32_wrap_i64();
                        sink.i32_const(w_class as i32);
                        sink.i32_store(MemArg {
                            offset: w_class_offset,
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
                        &mut sink,
                        value_types,
                        ref_homes,
                        &liveness,
                        op_idx,
                        skip,
                        frame,
                    );
                }
                // No `remember_wb` for the result. rewrite.rs's `handle_new`
                // and `gen_malloc_nursery` seed it only on the branches that
                // reached `can_use_nursery`; the `gen_malloc_fixedsize`
                // decline does not, and `gen_malloc_nursery` spells out why
                // — "the first result can come
                // from an old-gen slow path whose TRACK_YOUNG_PTRS flag
                // gen_initialize_tid intentionally preserves". Here the
                // generation is not a codegen-time fact at all: a
                // `non_moving` descr routes to `new_oldgen_fn_ptr`, whose
                // `alloc_in_oldgen` stamps TRACK_YOUNG_PTRS, and the plain
                // helper picks nursery or old-gen at runtime (the GC spells
                // that out through `alloc_nursery_collecting_typed_rooted`'s
                // `needs_write_barrier` out-parameter). Eliding here would
                // drop the barrier on a fresh old object, losing an
                // old-to-young edge. The inline flag test already makes the
                // young case a load and a not-taken branch, so the elision
                // this forgoes is the cheap one.
            }
            OpCode::NewArray | OpCode::NewArrayClear => {
                let vi = op.pos.get().raw();
                let descr = op.getdescr();
                let ad = descr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .unwrap_or_else(|| missing_layout_descr("array descr", op));
                let (base_size, item_size) = (ad.base_size() as i64, ad.item_size() as i64);
                let len_offset = ad.len_descr().map_or(0i64, |ld| ld.offset() as i64);
                let type_id = ad.type_id() as i64;

                // `rewrite.rs handle_new_array`: a `non_moving` descr declines
                // both nursery routes and allocates through the old-generation
                // twin. See the `New` arm.
                let non_moving = ad.non_moving();
                let alloc_array_fn_ptr = if non_moving {
                    alloc.new_array_oldgen_fn_ptr
                } else {
                    alloc.new_array_fn_ptr
                };
                let nursery = nursery.filter(|_| !non_moving);

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
                        total < na.large_threshold
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
                        // whose rounded total is strictly below the nursery
                        // large-object boundary, capped to wasm32's usize
                        // length field.  Totals are eight-byte aligned, so
                        // round the largest admitted word down after removing
                        // the exclusive endpoint.
                        let threshold =
                            na.large_threshold.saturating_sub(1).min(u32::MAX as usize) & !7;
                        if threshold < GcHeader::MIN_NURSERY_OBJ_SIZE || base_total > threshold {
                            return None;
                        }
                        let max_len = (threshold - base_total)
                            .checked_div(item_size_usize)
                            .unwrap_or(u32::MAX as usize)
                            .min(u32::MAX as usize);
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
                    // The sum is the committed `nursery_free`, so keep it
                    // rather than adding it again on the arm that takes it.
                    sink.local_tee(alloc_size_local);
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
                        value_types,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                    sink.else_();
                    // Commit: *nursery_free = free + total.
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
                        sink.local_set(value_types.local(vi));
                    } else {
                        sink.drop();
                    }
                } else if let (Some(base), Some((max_len, base_total, item_size, na))) =
                    (residual_type_base, inline_nursery_varsize)
                {
                    // malloc_cond_varsize: negative lengths compare greater in
                    // the unsigned precheck and go to the collecting slow path.
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_const(max_len);
                    sink.i64_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    sink.i64_const(type_id);
                    sink.i64_const(base_size);
                    sink.i64_const(item_size);
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
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
                        value_types,
                        ref_homes,
                        &liveness,
                        op_idx,
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                    );
                    sink.else_();
                    // total = round_up_8(max(header + base + item * length,
                    // MIN_NURSERY_OBJ_SIZE)).
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
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
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
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
                        value_types,
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
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
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
                        sink.local_set(value_types.local(vi));
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
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_const(len_offset);
                    sink.i32_const(alloc_array_fn_ptr as i32);
                    sink.call_indirect(0, base + 5);
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(value_types.local(vi));
                    } else {
                        sink.drop();
                    }
                } else {
                    let jit_call =
                        jit_call_idx.expect("NewArray op present but jit_call not imported");
                    // func_ptr = wasm_jit_alloc_array
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(alloc_array_fn_ptr);
                    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                    // num_args = 5
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(5);
                    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                    // arg0 = type_id
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(type_id);
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS));
                    // arg1 = base_size
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(base_size);
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + SLOT_SIZE));
                    // arg2 = item_size
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(item_size);
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + 2 * SLOT_SIZE));
                    // arg3 = length (op.arg(0))
                    emit_call_area_addr(&mut sink);
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + 3 * SLOT_SIZE));
                    // arg4 = len_offset
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(len_offset);
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + 4 * SLOT_SIZE));
                    // call trampoline
                    emit_jit_call(&mut sink, jit_call);

                    if !OpRef::raw_is_constant(vi) {
                        emit_call_area_addr(&mut sink);
                        sink.i64_load(mem64(STATIC_CALL_RESULT_OFS));
                        sink.local_set(value_types.local(vi));
                    }
                }
                // The check `rewrite.py` `_gen_call_malloc_gc` puts after a
                // collecting malloc. Here the helper writes the length field
                // itself, so the NULL escapes into the following item stores
                // rather than into a store this arm emits.
                emit_memory_error_check(
                    &mut sink,
                    value_types,
                    vi,
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                );
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
                        &mut sink,
                        value_types,
                        ref_homes,
                        &liveness,
                        op_idx,
                        skip,
                        frame,
                    );
                }
                // No `remember_wb` for the result, for the reason the
                // New/NewWithVtable arm above gives: a `non_moving` array
                // descr routes to `new_array_oldgen_fn_ptr`.
            }

            // ── Misc ──
            OpCode::ForceToken => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // `FORCE_TOKEN/0/r` — "nowadays, returns the jitframe".
                    // The token is what the SETFIELD_GC that follows parks in
                    // the virtualizable's `vable_token`, and what
                    // `Backend::force` is handed to rebuild a deadframe from,
                    // so it has to NAME this frame. A zero here reads as "no
                    // JIT frame is holding this virtualizable", which makes
                    // `force_virtualizable_if_necessary` skip the force and
                    // leaves an `f_locals` read to answer out of whatever the
                    // frame's own array last received.
                    //
                    // Answer the `JitFrame` BASE, not the items base local 0
                    // holds: the result of this op is Ref-typed, so it takes a
                    // Ref home slot, and both `build_home_gcmap` and
                    // `build_callee_gcmap` mark those slots for the collector.
                    // An items base is an interior pointer; traced as an object
                    // it reads its type id out of the frame's `jf_forward` word.
                    // The object base is a real GCREF, so a CA callee frame that
                    // moves out of the nursery under the very call this token
                    // brackets is forwarded here like any other reference.
                    sink.local_get(0);
                    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
                    sink.i32_sub();
                    sink.i64_extend_i32_u();
                    sink.local_set(value_types.local(vi));
                }
            }

            // Float operations
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(1).to_opref());
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
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::FloatNeg => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f64_neg();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::FloatAbs => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f64_abs();
                    sink.local_set(value_types.local(vi));
                }
            }

            // Debug / metadata / no-op
            OpCode::DebugMergePoint
            | OpCode::JitDebug
            | OpCode::IncrementDebugCounter
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::VirtualRefFinish
            | OpCode::ForceSpill
            | OpCode::Keepalive => {
                // `JitDebug` realizes its effect in the recorded trace, not in
                // emitted code: `consider_jit_debug` is `pass`.
            }

            _ => {
                // An opcode with no codegen arm declines the whole trace and
                // lets the metainterp fall back to the interpreter (correct,
                // unaccelerated). That covers a void opcode too: its `pos` is
                // `OpRef::NONE`, whose raw is `u32::MAX`, and
                // `raw_is_constant` rejects the sentinel range. So the only op
                // that reaches here and emits nothing is one the optimizer
                // folded into the constant namespace, which by then is pure and
                // has no side effect left to drop.
                //
                // An opcode that must emit nothing belongs in one of the no-op
                // arms above, where the reason it owes no code is written down;
                // a side-effecting op put there is dropped in silence, which is
                // what `CondCallN` was.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    return Err(BackendError::Unsupported(format!(
                        "wasm codegen: unhandled opcode {:?}",
                        op.opcode
                    )));
                }
            }
        }

        // store-on-def: mirror a freshly-defined Ref result into its home slot
        // so a (future) collecting allocation can forward it. The local
        // The mapped value local holds the value the matched arm just set; `ref_homes` only
        // keys Ref-typed value ids, so non-Ref / void / constant ops are
        // skipped. Each value-producing arm is operand-stack-neutral, so this
        // appended store is balanced.
        let result = op.pos.get();
        if let Some(h) = ref_homes.home(result) {
            sink.local_get(0);
            sink.local_get(value_types.local(result.raw()));
            sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
        }
    }

    if in_loop_body {
        sink.end(); // end loop
    }
    if resume_dispatch {
        sink.end(); // end R $resume
    }
    // A well-formed trace exits through a guard or Finish. Preserve the old
    // malformed/natural-fallthrough behavior without reaching bridge dispatch
    // with a stale frame fail index.
    sink.local_get(0);
    sink.return_();
    sink.end(); // end A $hot_exit

    // Frame-entry bridge dispatch for exits that branch out of the hot exit
    // block. Parameter-entry bridges tail-call from their own guard arm: that
    // arm knows the fixed failure arity and therefore the fixed wasm type.
    // The shared epilogue remains only for the established frame-entry form.
    if bridge_dispatch && bridge_param_type_indices.is_empty() {
        // slot = *(bridge_slot_local), where the local holds a cell address.
        sink.local_get(bridge_slot_local);
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
    sink.flush();
    drop(sink);

    Ok(func)
}

// ── Helpers ──

/// A peeled loop — real work (the unrolled first iteration = preamble) precedes
/// the loop-header LABEL — whether it carries one LABEL or several. `loop` is
/// emitted at the JUMP's target label, so `build_function` wraps the trace in
/// the resume-at-LABEL entry `br_table` (keyed on the frame dispatch-key slot,
/// key = label ordinal + 1) and a loop-closing bridge re-enters at any of the
/// loop's labels up to and including the header, in-module
/// (`resumable_label_count`). `build_function` keys its wrapper on this
/// predicate; `compile_loop` records it on `CompiledWasmLoop` as
/// `has_preamble`. `compile_bridge` accepts a loop-closing bridge only when
/// its JUMP's descr identifies one of the source loop's OWN labels
/// (`label_descrs`) with matching arity and a resume-safe live set.
pub fn is_resumable_peeled(ops: &[Op]) -> bool {
    let Some(loop_label) = find_loop_label_index(ops) else {
        return false;
    };
    ops[..loop_label]
        .iter()
        .any(|op| op.opcode != OpCode::Label)
}

/// How many of a peeled loop's LABELs the entry `br_table` can re-enter at:
/// those at or before the loop header. Each resume point costs a
/// (past_loader, loader) block pair opened before the wasm `loop`, so a pair
/// belonging to a label INSIDE the loop body would have to close inside the
/// loop — which structured control flow forbids. Such a label stays an in-body
/// marker: it emits nothing and is not published as a target.
pub fn resumable_label_count(ops: &[Op]) -> usize {
    let Some(loop_label) = find_loop_label_index(ops) else {
        return 0;
    };
    ops[..=loop_label]
        .iter()
        .filter(|op| op.opcode == OpCode::Label)
        .count()
}

/// Number of entry-dispatch keys an armed trace module can observe. Ordinary
/// traces have only the fresh-entry bucket; a resumable peeled loop has key 0
/// plus one bucket for each `br_table` resume arm.
pub fn entry_dispatch_key_count(ops: &[Op]) -> usize {
    if is_resumable_peeled(ops) {
        resumable_label_count(ops) + 1
    } else {
        1
    }
}

/// `counter += 1; if counter == threshold { trip(pending_id) }`, at the entry
/// of an out-of-line bridge whose merge into its owner is waiting on this
/// count. Equality rather than `>=` so the callback fires exactly once; the
/// merge takes the bridge out of the dispatch, so nothing reaches this code
/// again anyway.
///
/// Operand-stack-neutral, and it reads no frame slot.
fn emit_inline_trip_probe(sink: &mut PeepSink<'_, '_>, probe: InlineTripProbe, type_idx: u32) {
    sink.i32_const(probe.counter_addr as i32);
    sink.i32_const(probe.counter_addr as i32);
    sink.i64_load(mem64(0));
    sink.i64_const(1);
    sink.i64_add();
    sink.i64_store(mem64(0));
    sink.i32_const(probe.counter_addr as i32);
    sink.i64_load(mem64(0));
    sink.i64_const(probe.threshold as i64);
    sink.i64_eq();
    sink.if_(BlockType::Empty);
    // Zero the source guard's dispatch cell, so its next failure leaves the
    // guest instead of calling in here. Without it the host is not reached
    // again until the owner's loop finishes, and the merge lands after every
    // crossing it was meant to remove.
    if probe.cells_base_ptr != 0 {
        sink.i32_const(probe.cells_base_ptr as i32);
        sink.i32_load(memarg(0, 2));
        sink.if_(BlockType::Empty);
        sink.i32_const(probe.cells_base_ptr as i32);
        sink.i32_load(memarg(0, 2));
        sink.i32_const((probe.dispatch_cell_index * 4) as i32);
        sink.i32_add();
        sink.i32_const(0);
        sink.i32_store(memarg(0, 2));
        sink.end();
    }
    sink.i64_const(probe.pending_id);
    sink.i32_const(probe.trip_fn_ptr as i32);
    sink.call_indirect(0, type_idx);
    sink.drop(); // returns 0; ignored
    sink.end();
}

/// Increment one module's guest-memory entry counter. `key_local` holds the
/// i32 value consumed by the entry `br_table`; out-of-range values retain that
/// table's normal default-to-fresh-entry behaviour but do not index beyond the
/// fixed counter array.
fn emit_trace_entry_census(
    sink: &mut PeepSink<'_, '_>,
    census: crate::TraceEntryCensusStorage,
    scratch_local: u32,
    key_local: Option<u32>,
) {
    if let Some(key_local) = key_local {
        sink.local_get(key_local);
        sink.i32_const(census.key_count as i32);
        sink.i32_lt_u();
        sink.if_(BlockType::Empty);
        sink.i32_const(census.base as i32);
        sink.local_get(key_local);
        sink.i32_const(std::mem::size_of::<u64>() as i32);
        sink.i32_mul();
        sink.i32_add();
        sink.local_set(scratch_local);
        sink.local_get(scratch_local);
        sink.local_get(scratch_local);
        sink.i64_load(mem64(0));
        sink.i64_const(1);
        sink.i64_add();
        sink.i64_store(mem64(0));
        sink.end();
    } else {
        sink.i32_const(census.base as i32);
        sink.local_set(scratch_local);
        sink.local_get(scratch_local);
        sink.local_get(scratch_local);
        sink.i64_load(mem64(0));
        sink.i64_const(1);
        sink.i64_add();
        sink.i64_store(mem64(0));
    }
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

pub fn has_label_param_entry(
    inputargs: &[InputArg],
    ops: &[Op],
    frame: FrameGeometry,
    bridge_entry_arity: Option<usize>,
) -> bool {
    if bridge_entry_arity.is_some() || !is_resumable_peeled(ops) {
        return false;
    }
    let resumable = resumable_label_count(ops);
    let labels_fit = label_arg_counts(ops)
        .into_iter()
        .take(resumable)
        .all(|arity| arity <= crate::FROZEN_LABEL_PARAM_ARITY);
    labels_fit
        && inputargs.len() <= crate::FROZEN_LABEL_PARAM_ARITY
        // The shim loads from `FRAME_SLOT_BASE`, so its `FROZEN_LABEL_PARAM_ARITY`
        // reads occupy slots 1..=FROZEN_LABEL_PARAM_ARITY — slot 0 is the
        // dispatch key. A frame with exactly that many slots would let the last
        // load run off the end.
        && frame.value_slots >= crate::FROZEN_LABEL_PARAM_ARITY + 1
}

/// Per-label `(resume_safe, requires_own_frame)` metadata in ordinal order.
/// Missing pre-LABEL live-ins are safe when the frozen geometry contains the
/// capture plan. Such a plan is tied to the physical frame on which the owning
/// loop populated it; a sibling specialization may share the same geometry
/// but not those values, so bridge chaining must then stay on the owner.
pub fn label_resume_info(
    inputargs: &[InputArg],
    ops: &[Op],
    frame: FrameGeometry,
) -> Vec<(bool, bool)> {
    let resume = LabelResumeData::collect(inputargs, ops);
    let storage_supported = resume.supported_by(frame);
    resume
        .per_label
        .iter()
        .enumerate()
        .map(|(j, missing)| {
            (
                !resume.uncapturable[j] && (missing.is_empty() || storage_supported),
                !missing.is_empty(),
            )
        })
        .collect()
}

fn find_jump_target_label_index(ops: &[Op], jump: &Op) -> Option<usize> {
    let target = jump.getdescr()?;
    ops.iter().position(|op| {
        op.opcode == OpCode::Label
            && op
                .getdescr()
                .is_some_and(|descr| std::sync::Arc::ptr_eq(&descr, &target))
    })
}

pub(crate) fn find_loop_label_index(ops: &[Op]) -> Option<usize> {
    // The FIRST JUMP, not the last: a merged stream appends each inlined region
    // after the owner's ops, so the owner's terminal JUMP — the one that
    // defines the loop — precedes every region's. Reading the last would let a
    // region closing at an earlier LABEL move the `loop` onto that label.
    match ops.iter().find(|op| op.opcode == OpCode::Jump) {
        // x86/assembler.py:2463 `if target_token in
        // self.target_tokens_currently_compiling` — the TOKEN decides. A JUMP
        // that names a token this compilation does not define is upstream's
        // `else` arm at :2467 (`JMP(imm(target))`, an absolute jump into
        // another trace), even when this trace defines labels of its own.
        // That is exactly a `jump_to_preamble` retrace: compile.py:381 keeps
        // the retrace's own label_op in the middle while unroll.py:238-242
        // retargets the JUMP at the ORIGINAL loop's start descr. Answering
        // with the trailing label here would turn that JUMP into a back-edge
        // to a label it does not name.
        Some(jump) if jump.has_descr() => find_jump_target_label_index(ops, jump),
        // No descr to decide with (legacy IR whose JUMP carries none), or no
        // JUMP at all: keep the historical last-LABEL answer.
        _ => ops.iter().rposition(|op| op.opcode == OpCode::Label),
    }
}

/// Ordinal of the resumable LABEL a JUMP names, when that label is not the loop
/// header. `None` for the header, for a label past the resumable prefix, and
/// for a JUMP naming no local label. An inlined region with `Some(j)` has no
/// `br` target: the `loop` opens at the header, so branching there would skip
/// the segment between label `j` and the header.
fn jump_resume_ordinal(ops: &[Op], jump: &Op, num_labels: usize) -> Option<usize> {
    let ordinal = jump_label_ordinal(ops, jump)?;
    (ordinal + 1 < num_labels).then_some(ordinal)
}

/// Ordinal of the LABEL a JUMP names among this stream's LABELs. `None` when
/// the JUMP names no local label at all.
fn jump_label_ordinal(ops: &[Op], jump: &Op) -> Option<usize> {
    let label_idx = find_jump_target_label_index(ops, jump)?;
    Some(
        ops[..label_idx]
            .iter()
            .filter(|op| op.opcode == OpCode::Label)
            .count(),
    )
}

fn find_label_args(ops: &[Op], jump: &Op) -> Vec<OpRef> {
    // A multi-label trace's JUMP does not necessarily target its last label.
    // LABEL and JUMP share the loop-target descr, so resolve the target by Arc
    // identity just like compile_bridge's external-JUMP path. Falling back to
    // the last label preserves the historical behavior for legacy IR whose
    // JUMP carries no descr.
    if let Some(label_idx) = find_jump_target_label_index(ops, jump) {
        return ops[label_idx]
            .getarglist()
            .iter()
            .map(|arg| arg.to_opref())
            .collect();
    }
    for op in ops.iter().rev() {
        if op.opcode == OpCode::Label {
            return op.getarglist().iter().map(|a| a.to_opref()).collect();
        }
    }
    Vec::new()
}

/// A legacy pool-indexed const that is absent from the constants pool at emit
/// time is an optimizer-seeding invariant violation — panic loudly, matching
/// `collect_constants_from_ops`' `missing_legacy_const`, instead of emitting a
/// silent `0`. On native a null Ref traps on the first dereference; wasm's
/// offset 0 is valid linear memory, so a silent `0` is read as garbage and
/// miscompiles quietly rather than crashing.
#[cold]
#[inline(never)]
fn missing_emit_const(opref: OpRef) -> ! {
    panic!(
        "wasm emit_resolve: legacy pool-indexed const OpRef (raw={}) is absent \
         from the constants pool — the optimizer producer must seed it (or mint \
         an inline Const) instead of emitting a silent 0.",
        opref.raw()
    );
}

/// A memory-access or allocation op reached codegen without the layout descr it
/// must carry (Field/Array/Size). Emitting a default offset/size/type_id would
/// silently miscompile — on wasm, offset 0 is valid linear memory, so a bogus
/// address reads/writes garbage instead of trapping. Fail loud instead. Dead on
/// valid traces: every such op carries its descr (RPython invariant), and the
/// native x86 backend defaults identically without ever hitting the default.
#[cold]
#[inline(never)]
fn missing_layout_descr(what: &str, op: &Op) -> ! {
    panic!(
        "wasm codegen: {what} is absent for {:?} — a memory-access/allocation op \
         must carry its layout descr; a default offset/size/type_id would \
         silently miscompile.",
        op.opcode
    );
}

/// Resolve a constant operand's i64 bits: the inline `Const` value if the
/// variant carries one (`history.py:227/268/314`), else the legacy pool entry.
/// A pool miss panics via [`missing_emit_const`] rather than falling back to a
/// silent `0`.
fn resolve_const_bits(constants: &indexmap::IndexMap<u32, i64>, opref: OpRef) -> i64 {
    opref.inline_const_bits().unwrap_or_else(|| {
        constants
            .get(&opref.raw())
            .copied()
            .unwrap_or_else(|| missing_emit_const(opref))
    })
}

/// Reload the backend-only live-ins captured at `label` into their locals and
/// refresh their Ref homes, the ordinary homes the resumed body's
/// collecting-call reload path reads. Both resume paths need it: the entry
/// `br_table` arrives with every local zero-initialised, and an inlined region
/// arrives after the loop's own back edge may have rebound one of these locals
/// since the peeled pass wrote it.
fn emit_label_capture_restore(
    sink: &mut PeepSink<'_, '_>,
    label_resume: &LabelResumeData,
    value_types: &ValueLocals,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    label: usize,
) {
    for &r in &label_resume.per_label[label] {
        let storage = label_resume
            .storage(r)
            .expect("LABEL live-in has assigned capture storage");
        sink.local_get(0);
        sink.i64_load(mem64(label_resume.frame_offset(storage, frame)));
        if value_types.ty(r.raw()) == ValType::F64 {
            sink.f64_reinterpret_i64();
        }
        sink.local_set(value_types.local(r.raw()));
        if let Some(h) = ref_homes.home(r) {
            sink.local_get(0);
            sink.local_get(value_types.local(r.raw()));
            sink.i64_store(mem64(frame.home_slot_base + h as u64 * SLOT_SIZE));
        }
    }
}

fn emit_resolve(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    opref: OpRef,
) {
    if opref.is_constant() {
        let val = resolve_const_bits(constants, opref);
        sink.i64_const(val);
    } else if opref.is_none() {
        // A `NONE` fail-arg is a dead deopt slot: the optimizer numbered no
        // value for it, so the blackhole never reads it back (its resume data
        // carries the live values). Spill a zero placeholder — matching the
        // native backends, whose deadframe slot for an unmapped fail-arg is
        // never consumed. Resolving it as a local would index `value_types`
        // out of bounds (`raw() == u32::MAX`).
        sink.i64_const(0);
    } else {
        sink.local_get(value_types.local(opref.raw()));
        if value_types.ty(opref.raw()) == ValType::F64 {
            sink.i64_reinterpret_f64();
        }
    }
}

/// Resolve a Float operand as f64. Constants retain their i64 bit encoding in
/// the constant pool and are converted at the local boundary.
fn emit_resolve_f64(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    opref: OpRef,
) {
    if opref.is_constant() {
        let val = resolve_const_bits(constants, opref);
        sink.i64_const(val);
        sink.f64_reinterpret_i64();
    } else {
        debug_assert_eq!(value_types.ty(opref.raw()), ValType::F64);
        sink.local_get(value_types.local(opref.raw()));
    }
}

/// Values the optimizer left as plain (non-`Const`) OpRefs whose only
/// definition is a constant-pool entry, paired with that entry's raw bits.
///
/// Constant folding and the short preamble both hand the backend a folded
/// value under its original op position, with no producing op left in the
/// trace. `RegisterManager::loc` (dynasm `regalloc.rs`) covers that case with
/// a constants-map fallback taken once no register and no frame binding is
/// found. wasm materializes every value in a local instead of a location, so
/// the equivalent binding is a prologue store: without it the never-written
/// local reads as the zero wasm initializes it to, silently substituting 0
/// for the folded constant at every use.
///
/// Only positions actually read as a plain OpRef are returned, so a trace
/// whose pool holds no such value emits no extra prologue instruction.
fn unbound_pool_const_seeds(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    num_vars: u32,
) -> Result<Vec<(u32, i64)>, BackendError> {
    use std::collections::HashSet;
    let mut defined: HashSet<u32> = inputargs.iter().map(|ia| ia.index).collect();
    for op in ops {
        let r = op.pos.get();
        if r != OpRef::NONE && !r.is_constant() {
            defined.insert(r.raw());
        }
    }
    let mut seeds: Vec<(u32, i64)> = Vec::new();
    let mut unresolved: Vec<u32> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut consider = |a: OpRef, seeds: &mut Vec<(u32, i64)>, seen: &mut HashSet<u32>| {
        if a == OpRef::NONE || a.is_constant() {
            return;
        }
        let raw = a.raw();
        if raw >= num_vars || defined.contains(&raw) || !seen.insert(raw) {
            return;
        }
        match constants.get(&raw) {
            Some(&bits) => seeds.push((raw, bits)),
            // No producer and no pool entry: the local would read as the zero
            // wasm initializes it to, which is a wrong value, not a missing
            // one. Decline the trace (the interpreter runs it correctly,
            // unaccelerated) exactly as the unhandled-opcode arm does.
            None => unresolved.push(raw),
        }
    };
    for op in ops {
        for a in op.getarglist().iter() {
            consider(a.to_opref(), &mut seeds, &mut seen);
        }
        if let Some(fa) = op.getfailargs() {
            for a in fa.iter() {
                consider(a.to_opref(), &mut seeds, &mut seen);
            }
        }
    }
    if !unresolved.is_empty() {
        return Err(BackendError::Unsupported(format!(
            "wasm codegen: value{unresolved:?} read with no producing op and no \
             constant-pool entry"
        )));
    }
    Ok(seeds)
}

/// Compile-time value of a constant operand (what `emit_resolve` would push
/// as `i64.const`), or `None` for a runtime value.
fn const_operand_value(constants: &indexmap::IndexMap<u32, i64>, opref: OpRef) -> Option<i64> {
    opref
        .is_constant()
        .then(|| resolve_const_bits(constants, opref))
}

/// Extract field offset from op's descr (FieldDescr).
fn field_offset_from_descr(op: &Op) -> u64 {
    let __descr_arc_descr = op.getdescr();
    if let Some(descr) = __descr_arc_descr.as_ref()
        && let Some(fd) = descr.as_field_descr()
    {
        return fd.offset() as u64;
    }
    missing_layout_descr("field descr (offset)", op)
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
    .unwrap_or_else(|| missing_layout_descr("array descr (len layout)", op))
}

/// `llsupport/regalloc.py valid_addressing_size`: the scales x86 SIB (and a
/// wasm `i32.shl`) can form without a multiply.
fn valid_addressing_size(size: u64) -> bool {
    matches!(size, 1 | 2 | 4 | 8)
}

/// `llsupport/regalloc.py get_scale`: 1,2,4,8 → shift 0,1,2,3.
fn get_scale(size: u64) -> u32 {
    debug_assert!(valid_addressing_size(size));
    if size < 4 {
        (size as u32) - 1
    } else {
        (size as u32 >> 2) + 1
    }
}

/// Scale the i32 index already on the stack by `item_size`.
///
/// `x86/assembler.py` getarrayitem skips the scale when `itemsize == 1`.
/// `valid_addressing_size` / `get_scale` turn 2/4/8 into `i32.shl`; every
/// other stride keeps `i32.mul`, matching the IMUL fallback of
/// `_imul_const_scaled`.
fn emit_scale_index(sink: &mut PeepSink<'_, '_>, item_size: u64) {
    if item_size == 1 {
        return;
    }
    if valid_addressing_size(item_size) {
        sink.i32_const(get_scale(item_size) as i32);
        sink.i32_shl();
    } else {
        sink.i32_const(item_size as i32);
        sink.i32_mul();
    }
}

/// Leave `base + index * item_size` on the wasm stack as an i32 address, and
/// return the remaining displacement (`extra_offset`).
///
/// A ConstInt index is folded into that displacement — `rewrite.py
/// emit_gc_load_or_indexed` picks the non-indexed `GC_LOAD` arm for the
/// same case. The header / interior-field offset stays in the access's own
/// MemArg, the same place the `getfield` arms put `field_offset_from_descr`.
fn emit_scaled_index_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    base: OpRef,
    index: OpRef,
    item_size: u64,
    extra_offset: u64,
) -> u64 {
    emit_resolve(sink, constants, value_types, base);
    sink.i32_wrap_i64();
    // Only a constant that really lands inside the addressable range folds: a
    // MemArg displacement is unsigned and traps past the end of memory, so a
    // negative or overflowing index has to keep the run-time `i32` arithmetic,
    // which wraps instead.
    if let Some(idx) = const_operand_value(constants, index)
        && let Some(offset) = u64::try_from(idx)
            .ok()
            .and_then(|idx| idx.checked_mul(item_size))
            .and_then(|scaled| scaled.checked_add(extra_offset))
            .filter(|offset| u32::try_from(*offset).is_ok())
    {
        return offset;
    }
    emit_resolve(sink, constants, value_types, index);
    sink.i32_wrap_i64();
    emit_scale_index(sink, item_size);
    sink.i32_add();
    extra_offset
}

/// Leave `base + index * item_size` on the wasm stack as an i32 address, and
/// return the `base_size` displacement the access still owes.
fn emit_array_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
) -> u64 {
    let (base_size, item_size) = op
        .with_array_descr(|ad| (ad.base_size() as u64, ad.item_size() as u64))
        .unwrap_or_else(|| missing_layout_descr("array descr (base/item size)", op));
    emit_scaled_index_addr(
        sink,
        constants,
        value_types,
        op.arg(0).to_opref(),
        op.arg(1).to_opref(),
        item_size,
        base_size,
    )
}

/// The width of a GC pointer in the wasm32 guest heap.
///
/// A descriptor mints its width from the compiling target, so for a pyre descr
/// this is already what `field_size` / `item_size` says. It stays as its own
/// constant for the descr that does not: a pointer is four bytes here whatever
/// the descr claims, and a reading arm and its writing arm must take the width
/// from ONE place — the two spellings agreeing today is not the same as them
/// being one rule.
const GUEST_PTR_SIZE: usize = 4;

/// The width an access to one array item moves, and how a read of it extends.
/// The array twin of [`InteriorFieldLayout::access_size`]; every
/// `GETARRAYITEM` / `SETARRAYITEM` arm reads it from here.
///
/// The address stride still comes from the descriptor's own `item_size`
/// (`emit_array_addr`), which is what the allocation laid the array out with.
fn array_item_access_size_sign(op: &Op) -> (usize, bool) {
    op.with_array_descr(|ad| {
        if ad.is_array_of_pointers() {
            (GUEST_PTR_SIZE, false)
        } else {
            (ad.item_size(), ad.is_item_signed())
        }
    })
    .unwrap_or_else(|| missing_layout_descr("array descr (item size/sign)", op))
}

/// `descr.py unpack_interiorfielddescr`: `ofs = basesize + field.offset`,
/// plus the element stride and the field's own width / signedness / kind.
struct InteriorFieldLayout {
    /// `basesize + field.offset`, the displacement past the scaled index.
    offset: u64,
    /// The array's element stride.
    item_size: u64,
    /// The field's own width, and how a read of it extends.
    field_size: usize,
    signed: bool,
    is_float: bool,
    is_ptr: bool,
}

impl InteriorFieldLayout {
    /// The width an access to this field moves, and how a read of it extends.
    /// A pointer is guest-pointer-wide and never extends as signed, whatever
    /// the descriptor says; every reading and writing arm takes both from here
    /// so the two cannot drift apart.
    fn access_size_sign(&self) -> (usize, bool) {
        if self.is_ptr {
            (GUEST_PTR_SIZE, false)
        } else {
            (self.field_size, self.signed)
        }
    }
}

fn unpack_interior_field(op: &Op) -> InteriorFieldLayout {
    let descr = op
        .getdescr()
        .unwrap_or_else(|| missing_layout_descr("interior-field descr", op));
    let ifd = descr
        .as_interior_field_descr()
        .unwrap_or_else(|| missing_layout_descr("interior-field descr", op));
    let ad = ifd.array_descr();
    let fd = ifd.field_descr();
    InteriorFieldLayout {
        offset: (ad.base_size() + fd.offset()) as u64,
        item_size: ad.item_size() as u64,
        field_size: fd.field_size(),
        signed: fd.is_field_signed(),
        is_float: fd.is_float_field(),
        is_ptr: fd.is_pointer_field(),
    }
}

// ── Guard emission helpers ──

#[derive(Clone, Copy)]
struct InlineGuard<'a> {
    guard_idx: u32,
    inputargs: &'a [InputArg],
    /// Ordinal within this region's family, region 0 attached first. NOT a
    /// branch depth on its own: a family's blocks close one per region as the
    /// walk reaches each region's ops, so the depth of region N's block is this
    /// ordinal less however many of the family have already closed where the
    /// branching guard sits. `BridgeDispatch` carries that running count, and
    /// `emit_guard_exit` does the subtraction.
    region_ordinal: u32,
    outside_loop: bool,
}

#[derive(Clone, Copy)]
struct BridgeDispatch<'a> {
    cells_base: u32,
    fail_index_base: u32,
    bridge_slot_local: u32,
    enabled: bool,
    /// `arity -> indirect-call type` for armed parameter dispatch. Every
    /// signature carries values as i64, including Float bit patterns.
    param_type_indices: &'a indexmap::IndexMap<usize, u32>,
    inline_guards: &'a [InlineGuard<'a>],
    /// Depth of the innermost still-open preamble-region block, at the
    /// statement level of the operation being emitted.
    outside_region_base: u32,
    /// Regions of each family whose block has already been closed at the
    /// operation being emitted — one closes at each region's first op. A guard
    /// in the owner's own stream sees zero of both; a guard nested inside
    /// region P sees P+1 of P's family.
    closed_body_regions: u32,
    closed_outside_regions: u32,
    ref_homes: &'a RefHomes,
    frame: FrameGeometry,
    /// The trace's one GUARD_VALUE counter slot (`counter_slot`), or `None`
    /// when no guard needs one.
    counter_slot: Option<u64>,
    /// Fail-argument count -> the module function that spills that many
    /// arguments, for the counts `spill_helper_arities` admitted. An exit whose
    /// count is absent writes its own stores.
    spill_helpers: &'a indexmap::IndexMap<usize, u32>,
}

fn emit_guard_true(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: u32,
    dispatch: BridgeDispatch<'_>,
) {
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i64_eqz();
    emit_guard_if_exit(
        sink,
        constants,
        value_types,
        guard_idx,
        op,
        block_exit_depth,
        dispatch,
    );
}

fn emit_guard_false(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: u32,
    dispatch: BridgeDispatch<'_>,
) {
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i64_const(0);
    sink.i64_ne();
    emit_guard_if_exit(
        sink,
        constants,
        value_types,
        guard_idx,
        op,
        block_exit_depth,
        dispatch,
    );
}

/// `llsupport/regalloc.py next_op_can_accept_cc` — the comparison at `i`
/// may hand its condition straight to the op at `i + 1` instead of
/// materialising a boolean, when that op is the condition's only reader. x86
/// leaves the condition in the flags (`x86/regalloc.py:265
/// force_allocate_reg_or_cc`, ported to the dynasm sibling at
/// `next_op_can_accept_cc` in `majit-backend-dynasm/src/regalloc.rs`); wasm's
/// operand stack plays that role — [`push_cond`]'s i32 stays on the stack and
/// the guard's `if` tests it, so the `i64.extend_i32_u`/`local.set` and the
/// guard's own `local.get`/re-test disappear.
///
/// Narrower than the dynasm port on purpose: only `GuardTrue`/`GuardFalse`,
/// whose wasm arms do nothing but re-test the boolean.
fn next_op_can_accept_cc<'a>(
    ops: &'a [Op],
    i: usize,
    result: OpRef,
    liveness: &HomeLiveness,
    label_resume: &LabelResumeData,
    ref_homes: &RefHomes,
) -> Option<&'a Op> {
    if result == OpRef::NONE || result.is_constant() {
        return None;
    }
    let next_op = ops.get(i + 1)?;
    if !matches!(next_op.opcode, OpCode::GuardTrue | OpCode::GuardFalse) {
        return None;
    }
    // history.py `Const.is_constant()` — a Const operand is not an
    // op-result identity, so comparing raw positions against it is invalid.
    if next_op.num_args() == 0 || next_op.arg(0).is_constant() {
        return None;
    }
    if next_op.arg(0).to_opref().raw() != result.raw() {
        return None;
    }
    // Any later reader (including this guard's own fail args, which
    // `HomeLiveness` records as uses at `i + 1`) needs the materialised local.
    if liveness.last_use(result.raw()) > i as i32 + 1 {
        return None;
    }
    if next_op
        .getfailargs()
        .is_some_and(|fa| fa.iter().any(|a| a.to_opref() == result))
    {
        return None;
    }
    // A LABEL resume loader restores its capture set from the frame, so a
    // captured value must have been bound; skipping the `local.set` would leave
    // wasm's zero-init in its place.
    if label_resume.storage(result).is_some() {
        return None;
    }
    // The store-on-def tail reads the result local for a Ref-homed value. A
    // comparison result is never a Ref, so this only pins the invariant.
    if ref_homes.home(result).is_some() {
        return None;
    }
    Some(next_op)
}

/// The overflow guard immediately following an overflow op. The flag is not an
/// SSA value — it has no local, no home slot, no LABEL capture, and can never be
/// a fail argument — so unlike `next_op_can_accept_cc` this needs no liveness
/// test: adjacency is the whole condition.
fn next_ovf_guard(ops: &[Op], i: usize) -> Option<&Op> {
    let next = ops.get(i + 1)?;
    matches!(next.opcode, OpCode::GuardNoOverflow | OpCode::GuardOverflow).then_some(next)
}

/// Common guard exit: condition is on stack (i32), spill and leave on failure.
///
/// The spill belongs in this arm rather than in one shared exit handler after
/// the trace. x86/assembler.py `write_pending_failure_recoveries` can
/// place its recovery stubs after the hot code because `GuardToken.fail_locs`
/// (llsupport/assembler.py:24) freezes the register or stack location the
/// allocator gave each fail argument *at the guard*, so a stub reads a fixed
/// home and nothing keeps the value live past its own guard. Wasm has no way
/// to record such a location: the allocator is the engine's, and it derives
/// liveness from where the emitted code reads a local. Routing every guard to
/// one handler block therefore makes it a join point whose live-in set is the
/// union of every guard's fail arguments, so each becomes live at every guard
/// in the trace, and the resulting long ranges spill in the hot body. Reading
/// them here ends each range at the guard that needs it.
///
/// `block_exit_depth` is the statement-level depth of the enclosing exit
/// `block` (preamble = 0, loop body = 1); the `+ 1` accounts for the `if`
/// this opens. The stores run only on the failing edge, so the fallthrough
/// carries no frame traffic. With bridge dispatch enabled, the failing arm
/// writes its fail index to `frame[0]`, records its constant bridge-cell
/// address in a local, and branches to the shared dispatch epilogue.
fn emit_guard_if_exit(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: u32,
    dispatch: BridgeDispatch<'_>,
) {
    sink.if_(BlockType::Empty);
    emit_guard_exit(
        sink,
        constants,
        value_types,
        guard_idx,
        op,
        block_exit_depth + 1,
        dispatch,
        1,
    );
    sink.end();
}

/// Branch depth from the emitting instruction out to the `block` opened for
/// `inline`'s region.
///
/// A family's blocks close one per region as the walk reaches each region's
/// ops, so the ordinal counts from whichever of them are still open here.
/// `build_function` refuses any region whose source guard does not precede its
/// own ops, which is what keeps this subtraction from going negative.
///
/// `enclosing_frames` is what the caller opened between those blocks and this
/// instruction — one for the failing `if` of a conditional guard, none for an
/// exit emitted at statement level. Assuming the `if` unconditionally sends a
/// statement-level exit one frame too far out: for a region inside the header
/// `loop`, to the `loop` itself, which turns the exit into a back edge over the
/// owner's ops alone and leaves the region unreachable.
fn inline_region_br_depth(
    inline: &InlineGuard<'_>,
    dispatch: &BridgeDispatch<'_>,
    enclosing_frames: u32,
) -> u32 {
    let ordinal = if inline.outside_loop {
        inline.region_ordinal - dispatch.closed_outside_regions
    } else {
        inline.region_ordinal - dispatch.closed_body_regions
    };
    let depth = if inline.outside_loop {
        dispatch.outside_region_base + ordinal
    } else {
        ordinal
    };
    depth + enclosing_frames
}

fn emit_guard_exit(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: u32,
    dispatch: BridgeDispatch<'_>,
    enclosing_frames: u32,
) {
    if let Some(inline) = dispatch
        .inline_guards
        .iter()
        .find(|g| g.guard_idx == guard_idx)
    {
        emit_guard_inline_bridge_move(
            sink,
            constants,
            value_types,
            dispatch.ref_homes,
            dispatch.frame,
            op,
            inline.inputargs,
        );
        sink.br(inline_region_br_depth(inline, &dispatch, enclosing_frames));
        return;
    }
    if dispatch.param_type_indices.is_empty() {
        emit_guard_spill(
            sink,
            constants,
            value_types,
            guard_idx,
            op,
            dispatch.counter_slot,
            dispatch.spill_helpers,
        );
        if dispatch.enabled {
            emit_guard_bridge_dispatch(sink, guard_idx, dispatch);
        }
    } else {
        emit_guard_param_tail_call(sink, constants, value_types, guard_idx, op, dispatch);
        // A missing cell keeps the historical recovery path. It is deliberately
        // after the cell test so a bridge crossing performs no frame spill.
        emit_guard_spill(
            sink,
            constants,
            value_types,
            guard_idx,
            op,
            dispatch.counter_slot,
            dispatch.spill_helpers,
        );
    }
    sink.br(block_exit_depth);
}

/// Tail-call this guard's bridge directly when its cell is armed. The guard's
/// failure list fixes both the values and the wasm function type, so this path
/// needs neither an arity tag nor staging locals.
/// A guard op's fail args restricted to the positions its bridge received.
///
/// Same rule as `live_fail_arg_mask`, read off the op's own descr.
fn live_fail_args_of(op: &Op) -> Vec<OpRef> {
    let all: Vec<OpRef> = op
        .getfailargs()
        .map(|args| args.iter().map(|arg| arg.to_opref()).collect::<Vec<_>>())
        .unwrap_or_else(|| op.getarglist().iter().map(|arg| arg.to_opref()).collect());
    let descr = op.getdescr();
    let mask = live_fail_arg_mask(descr.as_ref(), all.len());
    all.into_iter()
        .zip(mask)
        .filter_map(|(arg, live)| live.then_some(arg))
        .collect()
}

fn emit_guard_param_tail_call(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    dispatch: BridgeDispatch<'_>,
) {
    let fail_args: Vec<OpRef> = live_fail_args_of(op);
    let arity = fail_args.len();
    let type_idx = *dispatch
        .param_type_indices
        .get(&arity)
        .expect("parameter dispatch type missing for guard fail arity");
    debug_assert!(dispatch.enabled);
    debug_assert!(guard_idx >= dispatch.fail_index_base);
    let cell_addr = dispatch.cells_base
        + (guard_idx - dispatch.fail_index_base) * std::mem::size_of::<u32>() as u32;
    sink.i32_const(cell_addr as i32);
    sink.i32_load(memarg(0, 2));
    sink.local_tee(dispatch.bridge_slot_local);
    sink.if_(BlockType::Empty);
    sink.local_get(0);
    for arg in fail_args {
        if arg.ty() == Some(Type::Float) {
            emit_resolve_f64(sink, constants, value_types, arg);
            sink.i64_reinterpret_f64();
        } else {
            emit_resolve(sink, constants, value_types, arg);
        }
    }
    sink.local_get(dispatch.bridge_slot_local);
    sink.return_call_indirect(0, type_idx);
    sink.end();
}

/// Transfer a failing guard directly into an inlined bridge.  All sources are
/// pushed before any destination local is written, preserving parallel-move
/// semantics when fail arguments overlap bridge input locals.
fn emit_guard_inline_bridge_move(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    op: &Op,
    inputargs: &[InputArg],
) {
    let fail_args: Vec<OpRef> = live_fail_args_of(op);
    assert_eq!(
        fail_args.len(),
        inputargs.len(),
        "guard and bridge input arity diverged"
    );
    for (arg, input) in fail_args.iter().zip(inputargs) {
        if value_types.ty(input.index) == ValType::F64 {
            emit_resolve_f64(sink, constants, value_types, *arg);
        } else {
            emit_resolve(sink, constants, value_types, *arg);
        }
    }
    for input in inputargs.iter().rev() {
        sink.local_set(value_types.local(input.index));
    }
    for input in inputargs {
        if let Some(home) = ref_homes.home_id(input.index) {
            sink.local_get(0);
            sink.local_get(value_types.local(input.index));
            sink.i64_store(mem64(frame.home_slot_base + home as u64 * SLOT_SIZE));
        }
    }
}

fn emit_guard_bridge_dispatch(
    sink: &mut PeepSink<'_, '_>,
    guard_idx: u32,
    dispatch: BridgeDispatch<'_>,
) {
    debug_assert!(guard_idx >= dispatch.fail_index_base);
    let cell_addr = dispatch.cells_base
        + (guard_idx - dispatch.fail_index_base) * std::mem::size_of::<u32>() as u32;
    sink.i32_const(cell_addr as i32);
    sink.local_set(dispatch.bridge_slot_local);
}

/// x86 `_store_force_index_if_next_guard`: a call that may force is bracketed
/// by the `GUARD_NOT_FORCED` immediately after it, so publish that guard's
/// coordinate before the call runs.
fn emit_force_bracket_before_call(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    ops: &[Op],
    op_idx: usize,
    guard_idx: u32,
) {
    let Some(next_op) = ops.get(op_idx + 1) else {
        return;
    };
    if !matches!(
        next_op.opcode,
        OpCode::GuardNotForced | OpCode::GuardNotForced2
    ) {
        return;
    }
    // Everything the guard names is defined by an op at or before the call --
    // except the call's own result, whose local still holds the PREVIOUS
    // iteration's value here.
    emit_force_arm(
        sink,
        constants,
        value_types,
        ref_homes,
        frame,
        next_op,
        exit_index(next_op, guard_idx),
        Some(ops[op_idx].pos.get().raw()),
    );
}

/// x86 `store_force_descr` / `_store_force_index`: publish where a force that
/// lands while this frame is still reachable reads its state from — upstream
/// writes the guard's descr into `jf_force_descr` and its fail arguments into
/// the frame. Publish the same coordinate here: the guard's exit index plus
/// [`FORCE_ARMED_BIT`] in `frame[0]`, and its fail arguments in the exit slots.
/// This is written unconditionally, not on a failure branch, because the reader
/// runs while the bracketed call is still on the stack.
///
/// A Ref argument is published as its **home slot offset**, tagged
/// `offset * 2 + 1`, rather than as its value. The exit slots are not in
/// `build_home_gcmap`'s traced set — that set is type-precise, and blanket
/// marking a slot that holds a scalar would offer the collector an integer to
/// mistake for a nursery address — so a Ref value copied here would not be
/// forwarded by a collection the bracketed call performs, and
/// `dead_frame_from_forced_frame` would read a from-space address. The home
/// slot IS traced and holds the same value, so naming it survives the
/// collection. Ref pointers are 8-aligned, which is what makes the low tag bit
/// free to tell an offset from a value; `undefined` and any Ref without a home
/// (a constant) still publish a literal, which is even.
#[allow(clippy::too_many_arguments)]
fn emit_force_arm(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    guard_op: &Op,
    exit_idx: u32,
    undefined: Option<u32>,
) {
    // `counter_value_spill` answers `None` for anything but a GUARD_VALUE, so
    // the counter slot has nothing to contribute to a force bracket.
    //
    // Same range `emit_guard_fail_args_spill` writes and
    // `normal_frame_value_slots` reserves: one past the last live position.
    let mut force_args = exit_fail_args(guard_op);
    force_args.truncate(live_fail_arg_extent(
        guard_op.getdescr().as_ref(),
        force_args.len(),
    ));
    for (i, &arg_ref) in force_args.iter().enumerate() {
        sink.local_get(0);
        if undefined == Some(arg_ref.raw()) {
            sink.i64_const(0);
        } else if let Some(home) = ref_homes.home(arg_ref) {
            let ofs = frame.home_slot_base + home as u64 * SLOT_SIZE;
            sink.i64_const((ofs as i64) * 2 + 1);
        } else {
            emit_resolve(sink, constants, value_types, arg_ref);
        }
        sink.i64_store(mem64(FRAME_SLOT_BASE + i as u64 * SLOT_SIZE));
    }
    sink.local_get(0);
    sink.i64_const(exit_idx as i64 | FORCE_ARMED_BIT);
    sink.i64_store(mem64(0));
}

fn emit_guard_spill(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    counter_slot: Option<u64>,
    spill_helpers: &indexmap::IndexMap<usize, u32>,
) {
    emit_guard_fail_args_spill(
        sink,
        constants,
        value_types,
        op,
        counter_slot,
        spill_helpers,
    );
    emit_guard_fail_index_store(sink, exit_index(op, guard_idx));
}

/// The exit a failing `op` writes into `frame[0]`.
///
/// `compile_done_with_this_frame` / `compile_exit_frame_with_exception` stamp
/// the FINISH with the singleton the cpu was handed, so every trace that
/// finishes the same way names the same exit and `_call_assembler_check_descr`
/// can recognise it with one compare. Guards, and the N-ary finishes pyre adds
/// on top of the `_DoneWithThisFrameDescr` family's 0/1-result classes, have no
/// shared identity and keep their own exit.
fn exit_index(op: &Op, guard_idx: u32) -> u32 {
    if op.opcode != OpCode::Finish {
        return guard_idx;
    }
    crate::failguard::attached_finish_exit_index(&op.getdescr()).unwrap_or(guard_idx)
}

fn emit_guard_fail_args_spill(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    counter_slot: Option<u64>,
    spill_helpers: &indexmap::IndexMap<usize, u32>,
) {
    // Only through the last live position: `normal_frame_value_slots` sizes the
    // value area the same way, so writing past it would write past the frame.
    let mut fail_args = exit_fail_args(op);
    fail_args.truncate(live_fail_arg_extent(
        op.getdescr().as_ref(),
        fail_args.len(),
    ));

    // The shared function writes the same slots in the same order; the call
    // site pushes the frame pointer once and each value once. The counter slot
    // below is per-trace rather than positional, so it stays inline.
    if let Some(&helper) = spill_helpers.get(&fail_args.len()) {
        sink.local_get(0);
        for &arg_ref in &fail_args {
            emit_resolve(sink, constants, value_types, arg_ref);
        }
        sink.call(helper);
    } else {
        for (i, &arg_ref) in fail_args.iter().enumerate() {
            let offset = FRAME_SLOT_BASE + i as u64 * SLOT_SIZE;
            sink.local_get(0);
            emit_resolve(sink, constants, value_types, arg_ref);
            sink.i64_store(mem64(offset));
        }
    }
    if let Some((operand, slot)) = counter_value_spill(op, &fail_args).zip(counter_slot) {
        let offset = FRAME_SLOT_BASE + slot * SLOT_SIZE;
        sink.local_get(0);
        emit_resolve(sink, constants, value_types, operand);
        sink.i64_store(mem64(offset));
    }
}

/// The GUARD_VALUE operand `op` must park in the trace's counter slot so that
/// `make_a_counter_per_value`'s index is readable, or `None` when the operand
/// already occupies a fail-argument slot, or is a constant the optimizer has
/// already decided the guard on.
///
/// `regalloc.py prepare_op_guard_value` hands `cpu.all_reg_indexes[arg.value]`
/// — the operand's index in the register save area every guard exit writes, so
/// upstream always has a slot for a box the guard does not carry among its
/// fail args. An exit here writes its fail args and nothing else, so the
/// operand needs a slot of its own.
///
/// That slot is ONE per trace (`counter_slot`), past every exit's fail args
/// and past the inputargs. A per-guard "one past MY fail args" index would
/// land inside a wider guard's fail-arg range, where the parked word would be
/// read back as that guard's fail argument.
fn counter_value_spill(op: &Op, fail_args: &[OpRef]) -> Option<OpRef> {
    if op.opcode != OpCode::GuardValue {
        return None;
    }
    let arg0 = op.arg(0).to_opref();
    if arg0 == OpRef::NONE || arg0.is_constant() || fail_args.contains(&arg0) {
        return None;
    }
    Some(arg0)
}

/// This op's fail arguments as the exit writes them, in slot order.
fn exit_fail_args(op: &Op) -> Vec<OpRef> {
    op.getfailargs()
        .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
        .unwrap_or_else(|| op.getarglist().iter().map(|a| a.to_opref()).collect())
}

/// x86/assembler.py `genop_discard_check_memory_error`: the NULL test
/// `rewrite.py` `_gen_call_malloc_gc` attaches to every collecting malloc.
/// wasm lowers `New` / `NewArray` itself in place of the GC rewrite, so it owes
/// itself the same check — address 0 is ordinary linear memory here, so the
/// vtable, class-word and item stores that follow an allocation would corrupt
/// it silently rather than fault.
///
/// The failing arm is `_build_propagate_exception_path` in this backend's exit
/// spelling. `_store_and_reset_exception` moves the `MemoryError` the
/// allocation helper published (`lib.rs` `oom_signal_if_zero`) out of the
/// shared cells and into the frame's first exit slot, `frame[0]` takes the
/// reserved `exit_frame_with_exception_descr_ref` exit, and the function
/// returns its frame pointer. It returns rather than branching to the hot-exit
/// block because the epilogue there dispatches on a per-guard bridge cell, and
/// this exit belongs to no guard and owns no cell.
///
/// A collecting helper can move the frame before it fails, so local 0 is
/// reloaded on this arm; the reload reads the shadow-stack top, so a caller
/// that already reloaded pays nothing for the second one.
///
/// Two configurations cannot deliver the exception and trap instead of
/// reporting a wrong one: a cpu that was never handed
/// `exit_frame_with_exception_descr_ref`, whose exit would resolve to a bare
/// finish and hand the exception back as the loop's result (the same choice
/// the cranelift sibling's `emit_memory_error_check` makes for an unattached
/// `propagate_exception_descr`), and a `MemoryError` provider that was never
/// registered, whose exit slot would carry a null reference into the
/// frontend's re-raise.
fn emit_memory_error_check(
    sink: &mut PeepSink<'_, '_>,
    value_types: &ValueLocals,
    vi: u32,
    residual_type_base: Option<u32>,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
) {
    if OpRef::raw_is_constant(vi) {
        return;
    }
    sink.local_get(value_types.local(vi));
    sink.i64_eqz();
    sink.if_(BlockType::Empty);
    if crate::failguard::exit_frame_with_exception_attached() {
        emit_reload_frame_if_necessary(sink, residual_type_base, ca_reload_fn_ptr, jf_top_addr);
        sink.local_get(0);
        sink.i32_const(crate::jit_exc_value_addr() as i32);
        sink.i64_load(mem64(0));
        sink.i64_store(mem64(FRAME_SLOT_BASE));
        sink.i32_const(crate::jit_exc_value_addr() as i32);
        sink.i64_const(0);
        sink.i64_store(mem64(0));
        sink.i32_const(crate::jit_exc_type_addr() as i32);
        sink.i64_const(0);
        sink.i64_store(mem64(0));
        sink.local_get(0);
        sink.i64_load(mem64(FRAME_SLOT_BASE));
        sink.i64_eqz();
        sink.if_(BlockType::Empty);
        sink.unreachable();
        sink.end();
        emit_guard_fail_index_store(sink, crate::failguard::FINISH_EXIT_INDEX_EXC);
        sink.local_get(0);
        sink.return_();
    } else {
        sink.unreachable();
    }
    sink.end();
}

fn emit_guard_fail_index_store(sink: &mut PeepSink<'_, '_>, guard_idx: u32) {
    sink.local_get(0);
    sink.i64_const(guard_idx as i64);
    sink.i64_store(mem64(0));
}

// ── Binary ops ──

#[derive(Clone, Copy, Debug)]
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

fn apply_binop(sink: &mut PeepSink<'_, '_>, op: BinOp) {
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
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    binop: BinOp,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    apply_binop(sink, binop);
    sink.local_set(value_types.local(vi));
}

/// `UintMulHigh`: high 64 bits of the unsigned 64×64→128 product. Wasm has
/// only `i64.mul` (low 64 bits), so compute via the classic 32-bit split:
/// a = ah·2³²+al, b = bh·2³²+bl, with carry-safe intermediates
///   mid1 = ah·bl + (al·bl >> 32)
///   high = ah·bh + (mid1 >> 32) + ((al·bh + (mid1 & 0xFFFFFFFF)) >> 32)
/// Uses the five scratch locals reserved after the dense value-local range.
fn emit_umulhi(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    value_local_count: u32,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_umulhi_to_local(
        sink,
        constants,
        value_types,
        op,
        value_local_count,
        value_types.local(vi),
    );
}

fn emit_umulhi_to_local(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    value_local_count: u32,
    output_local: u32,
) {
    const MASK32: i64 = 0xFFFF_FFFF;
    let al = value_local_count + 1;
    let ah = value_local_count + 2;
    let bl = value_local_count + 3;
    let bh = value_local_count + 4;
    let mid1 = value_local_count + 5;

    // al = a & 0xFFFFFFFF
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.local_set(al);
    // ah = a >>u 32
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_set(ah);
    // bl = b & 0xFFFFFFFF
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.local_set(bl);
    // bh = b >>u 32
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
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

    sink.local_set(output_local);
}

/// The overflow condition for `a + c` / `a - c` against a constant `c`, as
/// `(limit, greater_than)`: the operation overflows exactly when `a > limit`
/// (`greater_than`) or when `a < limit`. `None` means it cannot overflow.
///
/// The general form needs both operands and the result to compare sign bits.
/// Against a constant the same predicate is one comparison against a bound
/// folded here, which also drops the dependency on the result. Each bound is
/// taken from the opposite extreme, so none of them can itself overflow:
/// `MAX - c` only for `c > 0`, `MIN - c` only for `c < 0`, and so on.
fn ovf_const_bound(binop: BinOp, c: i64) -> Option<(i64, bool)> {
    use std::cmp::Ordering;
    match binop {
        BinOp::I64Add => match c.cmp(&0) {
            Ordering::Greater => Some((i64::MAX - c, true)),
            Ordering::Less => Some((i64::MIN - c, false)),
            Ordering::Equal => None,
        },
        BinOp::I64Sub => match c.cmp(&0) {
            Ordering::Greater => Some((i64::MIN + c, false)),
            Ordering::Less => Some((i64::MAX + c, true)),
            Ordering::Equal => None,
        },
        _ => None,
    }
}

/// The variable operand and constant operand of an add/sub whose overflow can
/// take the [`ovf_const_bound`] test. Addition is commutative, so either side
/// may supply the constant; for subtraction only the subtrahend does, since
/// `c - a` has a different bound shape and keeps the general form.
fn ovf_const_operand(
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
    binop: BinOp,
) -> Option<(OpRef, i64)> {
    let (a, b) = (op.arg(0).to_opref(), op.arg(1).to_opref());
    match binop {
        BinOp::I64Add if a.is_constant() && !b.is_constant() => {
            Some((b, resolve_const_bits(constants, a)))
        }
        BinOp::I64Add | BinOp::I64Sub if !a.is_constant() && b.is_constant() => {
            Some((a, resolve_const_bits(constants, b)))
        }
        _ => None,
    }
}

/// What an overflow op left for the guard that follows it.
enum OvfFlag {
    /// The op was constant-folded and emitted nothing; there is no flag.
    Absent,
    /// `ovf_flag_local` holds the flag — nonzero means the op overflowed.
    InLocal,
    /// The following guard's failure condition is on the stack as an i32,
    /// ready for `emit_guard_if_exit`.
    FusedCond,
}

/// The comparison a fused guard makes on the overflow predicate.
/// `GuardNoOverflow` exits when the op overflowed and so takes the predicate as
/// it stands; `GuardOverflow` exits when it did not, and flipping the
/// comparison costs nothing where negating its result would cost an
/// instruction.
fn overflow_failure_cmp(cmp: CmpOp, fused_guard: OpCode) -> CmpOp {
    match fused_guard {
        OpCode::GuardNoOverflow => cmp,
        OpCode::GuardOverflow => match cmp {
            CmpOp::I64GtS => CmpOp::I64LeS,
            CmpOp::I64LtS => CmpOp::I64GeS,
            CmpOp::I64Ne => CmpOp::I64Eq,
            _ => unreachable!(
                "overflow comparison must be a constant bound or a sign-word inequality"
            ),
        },
        _ => unreachable!("overflow fusion requires an overflow guard"),
    }
}

/// Overflow binary op: stores the wrapping result in pos and either leaves the
/// overflow flag for a following guard or writes it to the scratch local.
fn emit_ovf_binop(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    binop: BinOp,
    value_local_count: u32,
    ovf_flag_local: u32,
    fused_guard: Option<OpCode>,
) -> OvfFlag {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return OvfFlag::Absent;
    }
    let result_local = value_types.local(vi);

    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    apply_binop(sink, binop);
    sink.local_set(result_local);

    if let Some((var, c)) = ovf_const_operand(constants, op, binop) {
        match ovf_const_bound(binop, c) {
            Some((limit, greater_than)) => {
                let cmp = if greater_than {
                    CmpOp::I64GtS
                } else {
                    CmpOp::I64LtS
                };
                emit_resolve(sink, constants, value_types, var);
                sink.i64_const(limit);
                if let Some(guard) = fused_guard {
                    apply_cmp(sink, overflow_failure_cmp(cmp, guard));
                    return OvfFlag::FusedCond;
                }
                apply_cmp(sink, cmp);
                sink.i64_extend_i32_u();
            }
            // Adding or subtracting zero: the flag stays live so the paired
            // guard still finds it, and folds against a constant zero. There
            // is no predicate to hand a fused guard, so this answers in the
            // local whether or not one was offered.
            None => {
                sink.i64_const(0);
            }
        }
        sink.local_set(ovf_flag_local);
        return OvfFlag::InLocal;
    }

    match binop {
        BinOp::I64Add => {
            // (a ^ result) & (b ^ result) — negative exactly when it overflowed
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.local_get(result_local);
            sink.i64_xor();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.local_get(result_local);
            sink.i64_xor();
            sink.i64_and();
        }
        BinOp::I64Sub => {
            // (a ^ b) & (a ^ result) — negative exactly when it overflowed
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_xor();
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.local_get(result_local);
            sink.i64_xor();
            sink.i64_and();
        }
        BinOp::I64Mul => {
            // Multiplying two signed-32-bit integers cannot overflow i64: the
            // largest magnitude is 2^62. This is the common Python-loop shape
            // (e.g. nested_loop's 0..19999 counters), and avoids expanding
            // every multiplication into a software 64x64->128 product. The
            // exact sign-extension checks preserve the full-width slow path
            // for every value outside that proven-safe domain.
            // Resolving an operand is the same single `local.get` that reading
            // a scratch copy of it would be, so the check reads the operands
            // and the umulhi bank stays the slow arm's alone.
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_extend32_s();
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_eq();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_extend32_s();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_eq();
            sink.i32_and();
            // Both arms answer the same one-bit predicate, so the block
            // yields it rather than each arm storing it: an adjacent guard
            // reads it off the stack the way the add and sub forms do, and
            // only an unpaired op pays for the flag local. With no paired
            // guard the predicate to yield is `GuardNoOverflow`'s -- "it
            // overflowed" -- which is what that local is defined to hold.
            let failure_of = fused_guard.unwrap_or(OpCode::GuardNoOverflow);
            sink.if_(BlockType::Result(ValType::I32));
            sink.i32_const(i32::from(matches!(failure_of, OpCode::GuardOverflow)));
            sink.else_();

            // Convert the unsigned high word to the signed high word:
            // smulhi = umulhi - ((a >>s 63) & b) - ((b >>s 63) & a).
            let high_local = value_local_count + 1;
            emit_umulhi_to_local(
                sink,
                constants,
                value_types,
                op,
                value_local_count,
                high_local,
            );
            sink.local_get(high_local);
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_const(63);
            sink.i64_shr_s();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_and();
            sink.i64_sub();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_const(63);
            sink.i64_shr_s();
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_and();
            sink.i64_sub();
            sink.local_get(result_local);
            sink.i64_const(63);
            sink.i64_shr_s();
            apply_cmp(sink, overflow_failure_cmp(CmpOp::I64Ne, failure_of));
            sink.end();
            if fused_guard.is_some() {
                return OvfFlag::FusedCond;
            }
            sink.i64_extend_i32_u();
            sink.local_set(ovf_flag_local);
            return OvfFlag::InLocal;
        }
        _ => unreachable!("overflow emitter requires add, sub, or mul"),
    }

    // The sign bit of the word both arms left on the stack is the answer, so a
    // fused guard reads it with the comparison it was going to make anyway.
    if let Some(guard) = fused_guard {
        sink.i64_const(0);
        apply_cmp(sink, overflow_failure_cmp(CmpOp::I64LtS, guard));
        OvfFlag::FusedCond
    } else {
        sink.i64_const(63);
        sink.i64_shr_s();
        sink.local_set(ovf_flag_local);
        OvfFlag::InLocal
    }
}

// ── Comparison ops ──

#[derive(Clone, Copy)]
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

fn apply_cmp(sink: &mut PeepSink<'_, '_>, op: CmpOp) {
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

#[derive(Clone, Copy)]
enum FloatCmp {
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
}

/// An op whose result is a 0/1 boolean produced by a single wasm comparison.
/// [`push_cond`] leaves that comparison's i32 on the operand stack; [`emit_cond`]
/// is the ordinary spelling that widens and binds it to the result local.
#[derive(Clone, Copy)]
enum CondKind {
    Int(CmpOp),
    Float(FloatCmp),
    IsTrue,
    IsZero,
}

fn cond_kind_of(opcode: OpCode) -> Option<CondKind> {
    Some(match opcode {
        // ── Integer comparisons (signed) ──
        OpCode::IntLt => CondKind::Int(CmpOp::I64LtS),
        OpCode::IntLe => CondKind::Int(CmpOp::I64LeS),
        OpCode::IntEq => CondKind::Int(CmpOp::I64Eq),
        OpCode::IntNe => CondKind::Int(CmpOp::I64Ne),
        OpCode::IntGt => CondKind::Int(CmpOp::I64GtS),
        OpCode::IntGe => CondKind::Int(CmpOp::I64GeS),
        // ── Integer comparisons (unsigned) ──
        OpCode::UintLt => CondKind::Int(CmpOp::I64LtU),
        OpCode::UintLe => CondKind::Int(CmpOp::I64LeU),
        OpCode::UintGt => CondKind::Int(CmpOp::I64GtU),
        OpCode::UintGe => CondKind::Int(CmpOp::I64GeU),
        // ── Pointer comparisons ──
        OpCode::PtrEq | OpCode::InstancePtrEq => CondKind::Int(CmpOp::I64Eq),
        OpCode::PtrNe | OpCode::InstancePtrNe => CondKind::Int(CmpOp::I64Ne),
        // ── Float comparisons ──
        OpCode::FloatLt => CondKind::Float(FloatCmp::Lt),
        OpCode::FloatLe => CondKind::Float(FloatCmp::Le),
        OpCode::FloatEq => CondKind::Float(FloatCmp::Eq),
        OpCode::FloatNe => CondKind::Float(FloatCmp::Ne),
        OpCode::FloatGt => CondKind::Float(FloatCmp::Gt),
        OpCode::FloatGe => CondKind::Float(FloatCmp::Ge),
        // ── Truth tests ──
        OpCode::IntIsTrue => CondKind::IsTrue,
        OpCode::IntIsZero => CondKind::IsZero,
        _ => return None,
    })
}

/// Push the comparison's i32 result (0 or 1) onto the operand stack.
fn push_cond(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    kind: CondKind,
) {
    match kind {
        CondKind::Int(cmpop) => {
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            apply_cmp(sink, cmpop);
        }
        CondKind::Float(cmp) => {
            emit_resolve_f64(sink, constants, value_types, op.arg(0).to_opref());
            emit_resolve_f64(sink, constants, value_types, op.arg(1).to_opref());
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
        }
        CondKind::IsTrue => {
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_const(0);
            sink.i64_ne();
        }
        CondKind::IsZero => {
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_eqz();
        }
    }
}

/// Push whether a fused GuardTrue/GuardFalse fails. Native backends invert the
/// integer condition code in place (`x86/assembler.py:1778-1784`); spelling the
/// inverse Wasm comparison directly avoids materialising `cmp; i32.eqz` at the
/// hot guard site. Float ordered comparisons deliberately keep `i32.eqz`:
/// their apparent inverse is not equivalent for NaN/unordered operands.
fn push_guard_failure_cond(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    kind: CondKind,
    guard_opcode: OpCode,
) {
    if guard_opcode == OpCode::GuardFalse {
        push_cond(sink, constants, value_types, op, kind);
        return;
    }
    debug_assert_eq!(guard_opcode, OpCode::GuardTrue);
    let inverse = match kind {
        CondKind::Int(cmp) => Some(CondKind::Int(match cmp {
            CmpOp::I64LtS => CmpOp::I64GeS,
            CmpOp::I64LeS => CmpOp::I64GtS,
            CmpOp::I64Eq => CmpOp::I64Ne,
            CmpOp::I64Ne => CmpOp::I64Eq,
            CmpOp::I64GtS => CmpOp::I64LeS,
            CmpOp::I64GeS => CmpOp::I64LtS,
            CmpOp::I64LtU => CmpOp::I64GeU,
            CmpOp::I64LeU => CmpOp::I64GtU,
            CmpOp::I64GtU => CmpOp::I64LeU,
            CmpOp::I64GeU => CmpOp::I64LtU,
        })),
        CondKind::IsTrue => Some(CondKind::IsZero),
        CondKind::IsZero => Some(CondKind::IsTrue),
        CondKind::Float(_) => None,
    };
    if let Some(inverse) = inverse {
        push_cond(sink, constants, value_types, op, inverse);
    } else {
        push_cond(sink, constants, value_types, op, kind);
        sink.i32_eqz();
    }
}

fn emit_cond(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    kind: CondKind,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    push_cond(sink, constants, value_types, op, kind);
    sink.i64_extend_i32_u();
    sink.local_set(value_types.local(vi));
}

// ── Unary op helper ──

fn emit_unary_vi(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    prefix: impl FnOnce(&mut PeepSink<'_, '_>),
    suffix: impl FnOnce(&mut PeepSink<'_, '_>),
) {
    let vi = op.pos.get().raw();
    if !OpRef::raw_is_constant(vi) {
        prefix(sink);
        emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
        suffix(sink);
        sink.local_set(value_types.local(vi));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peep_sink_applies_all_local_folds() {
        let mut bytes = Vec::new();
        {
            let mut raw_sink = InstructionSink::new(&mut bytes);
            let mut sink = PeepSink::new(&mut raw_sink);

            sink.local_set(1)
                .local_get(1)
                .i64_const(0)
                .i32_wrap_i64()
                .i32_const(4)
                .i32_mul()
                .i32_add();
            sink.flush();
        }

        assert_eq!(bytes, [0x22, 0x01]);
    }

    /// A guard exit that jumps into a merged region counts only the frames its
    /// own emission opened. A conditional guard branches from inside its
    /// failing `if`; `GUARD_ALWAYS_FAILS` and `FINISH` branch at statement
    /// level, and charging them for an `if` they never open would send the
    /// branch to the enclosing `loop` instead of the region's block.
    #[test]
    fn inline_region_br_depth_counts_only_the_frames_the_caller_opened() {
        let ref_homes = RefHomes {
            by_id: Vec::new(),
            len: 0,
        };
        let param_type_indices = indexmap::IndexMap::new();
        let spill_helpers = indexmap::IndexMap::new();
        let inline = InlineGuard {
            guard_idx: 0,
            inputargs: &[],
            region_ordinal: 0,
            outside_loop: false,
        };
        let dispatch = BridgeDispatch {
            cells_base: 0,
            fail_index_base: 0,
            bridge_slot_local: 0,
            enabled: false,
            param_type_indices: &param_type_indices,
            inline_guards: std::slice::from_ref(&inline),
            outside_region_base: 4,
            closed_body_regions: 0,
            closed_outside_regions: 0,
            ref_homes: &ref_homes,
            frame: FrameGeometry::compact(1, 0, 0),
            counter_slot: None,
            spill_helpers: &spill_helpers,
        };

        assert_eq!(inline_region_br_depth(&inline, &dispatch, 0), 0);
        assert_eq!(inline_region_br_depth(&inline, &dispatch, 1), 1);

        // A preamble region is reached from the outside-loop base instead.
        let outside = InlineGuard {
            outside_loop: true,
            ..inline
        };
        assert_eq!(inline_region_br_depth(&outside, &dispatch, 0), 4);
        assert_eq!(inline_region_br_depth(&outside, &dispatch, 1), 5);
    }

    #[test]
    fn compact_geometry_keeps_tail_call_area_out_of_ca_prefix() {
        let frame = FrameGeometry::compact(32, 16, 0);
        assert_eq!(frame.dispatch_key_ofs, 32 * SLOT_SIZE);
        assert_eq!(frame.home_slot_base, 33 * SLOT_SIZE);
        assert_eq!(frame.ca_frame_bytes, 392);
        assert_eq!(frame.call_result_ofs, frame.ca_frame_bytes as u64);
        assert_eq!(frame.call_args_ofs, 416);
        assert_eq!(frame.frame_bytes, 544);
    }

    /// The constant-operand bound must answer exactly what the wrapping
    /// arithmetic does, including at the extremes where the bound itself is
    /// closest to overflowing (`c` = `MIN` makes `MAX + c` and `MIN - c` the
    /// interesting cases).
    #[test]
    fn ovf_const_bound_agrees_with_checked_arithmetic() {
        let edges = [
            i64::MIN,
            i64::MIN + 1,
            -3,
            -1,
            0,
            1,
            3,
            i64::MAX - 1,
            i64::MAX,
        ];
        for &c in &edges {
            for &a in &edges {
                for (binop, expected) in [
                    (BinOp::I64Add, a.checked_add(c).is_none()),
                    (BinOp::I64Sub, a.checked_sub(c).is_none()),
                ] {
                    let got = match ovf_const_bound(binop, c) {
                        None => false,
                        Some((limit, true)) => a > limit,
                        Some((limit, false)) => a < limit,
                    };
                    assert_eq!(got, expected, "{binop:?}: a={a} c={c}");
                }
            }
        }
    }
}
