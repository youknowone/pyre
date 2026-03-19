use std::cmp::max;
use std::collections::HashMap;
use std::marker::PhantomData;

use majit_codegen::JitCellToken;
use majit_ir::{OpCode, OpRef};

use crate::{SymbolicStack, TraceAction, TraceCtx};

const BC_LOAD_CONST_I: u8 = 1;
const BC_POP_I: u8 = 2;
const BC_PEEK_I: u8 = 3;
const BC_PUSH_I: u8 = 4;
const BC_POP_DISCARD: u8 = 5;
const BC_DUP_STACK: u8 = 6;
const BC_SWAP_STACK: u8 = 7;
const BC_RECORD_BINOP_I: u8 = 8;
const BC_RECORD_UNARY_I: u8 = 9;
const BC_REQUIRE_STACK: u8 = 10;
const BC_BRANCH_ZERO: u8 = 11;
const BC_JUMP_TARGET: u8 = 12;
const BC_ABORT: u8 = 13;
const BC_ABORT_PERMANENT: u8 = 14;
const BC_BRANCH_REG_ZERO: u8 = 15;
const BC_JUMP: u8 = 16;
const BC_INLINE_CALL: u8 = 17;
const BC_RESIDUAL_CALL_VOID: u8 = 18;
const BC_SET_SELECTED: u8 = 19;
const BC_PUSH_TO: u8 = 20;
const BC_MOVE_I: u8 = 21;
const BC_CALL_INT: u8 = 22;
const BC_CALL_PURE_INT: u8 = 23;
// Ref-typed bytecodes
const BC_LOAD_CONST_R: u8 = 24;
const BC_POP_R: u8 = 25;
const BC_PUSH_R: u8 = 26;
const BC_MOVE_R: u8 = 27;
const BC_CALL_REF: u8 = 28;
const BC_CALL_PURE_REF: u8 = 29;
// Float-typed bytecodes
const BC_LOAD_CONST_F: u8 = 30;
const BC_POP_F: u8 = 31;
const BC_PUSH_F: u8 = 32;
const BC_MOVE_F: u8 = 33;
const BC_CALL_FLOAT: u8 = 34;
const BC_CALL_PURE_FLOAT: u8 = 35;
const BC_RECORD_BINOP_F: u8 = 36;
const BC_RECORD_UNARY_F: u8 = 37;
const BC_CALL_MAY_FORCE_INT: u8 = 38;
const BC_CALL_MAY_FORCE_REF: u8 = 39;
const BC_CALL_MAY_FORCE_FLOAT: u8 = 40;
const BC_CALL_MAY_FORCE_VOID: u8 = 41;
const BC_CALL_RELEASE_GIL_INT: u8 = 42;
const BC_CALL_RELEASE_GIL_REF: u8 = 43;
const BC_CALL_RELEASE_GIL_FLOAT: u8 = 44;
const BC_CALL_RELEASE_GIL_VOID: u8 = 45;
const BC_CALL_LOOPINVARIANT_INT: u8 = 46;
const BC_CALL_LOOPINVARIANT_REF: u8 = 47;
const BC_CALL_LOOPINVARIANT_FLOAT: u8 = 48;
const BC_CALL_LOOPINVARIANT_VOID: u8 = 49;
const BC_CALL_ASSEMBLER_INT: u8 = 50;
const BC_CALL_ASSEMBLER_REF: u8 = 51;
const BC_CALL_ASSEMBLER_FLOAT: u8 = 52;
const BC_CALL_ASSEMBLER_VOID: u8 = 53;
const BC_COPY_FROM_BOTTOM: u8 = 54;
const BC_STORE_DOWN: u8 = 55;
const BC_LOAD_STATE_FIELD: u8 = 56;
const BC_STORE_STATE_FIELD: u8 = 57;
const BC_LOAD_STATE_ARRAY: u8 = 58;
const BC_STORE_STATE_ARRAY: u8 = 59;
const BC_LOAD_STATE_VARRAY: u8 = 60;
const BC_STORE_STATE_VARRAY: u8 = 61;
const BC_GETFIELD_VABLE_I: u8 = 62;
const BC_GETFIELD_VABLE_R: u8 = 63;
const BC_GETFIELD_VABLE_F: u8 = 64;
const BC_SETFIELD_VABLE_I: u8 = 65;
const BC_SETFIELD_VABLE_R: u8 = 66;
const BC_SETFIELD_VABLE_F: u8 = 67;
const BC_GETARRAYITEM_VABLE_I: u8 = 68;
const BC_GETARRAYITEM_VABLE_R: u8 = 69;
const BC_GETARRAYITEM_VABLE_F: u8 = 70;
const BC_SETARRAYITEM_VABLE_I: u8 = 71;
const BC_SETARRAYITEM_VABLE_R: u8 = 72;
const BC_SETARRAYITEM_VABLE_F: u8 = 73;
const BC_ARRAYLEN_VABLE: u8 = 74;
const BC_HINT_FORCE_VIRTUALIZABLE: u8 = 75;
const MAX_HOST_CALL_ARITY: usize = 16;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LivenessInfo {
    pub pc: u16,
    pub live_i_regs: Vec<u16>,
}

/// Serialized interpreter step description, mirroring RPython's `JitCode`
/// at a much smaller subset for the current proc-macro tracing surface.
#[derive(Clone, Debug, Default)]
pub struct JitCode {
    /// Encoded bytecode stream.
    pub code: Vec<u8>,
    /// Number of registers by kind: int/ref/float.
    pub num_regs: [u16; 3],
    /// Integer constant pool.
    pub constants_i: Vec<i64>,
    /// Liveness metadata for GC / deopt expansion.
    pub liveness: Vec<LivenessInfo>,
    /// Pool of majit IR opcodes referenced from the bytecode stream.
    pub opcodes: Vec<OpCode>,
    /// Sub-JitCodes for `inline_call` targets (compound methods).
    pub sub_jitcodes: Vec<JitCode>,
    /// Function pointers for `residual_call` targets (I/O shims, external calls).
    pub fn_ptrs: Vec<JitCallTarget>,
    /// CALL_ASSEMBLER targets keyed by loop token number plus a concrete hook.
    assembler_targets: Vec<JitCallAssemblerTarget>,
}

// ── RPython jitcode.py parity methods ──

impl JitCode {
    /// RPython: `JitCode.num_regs_i()`
    pub fn num_regs_i(&self) -> u16 {
        self.num_regs[0]
    }

    /// RPython: `JitCode.num_regs_r()`
    pub fn num_regs_r(&self) -> u16 {
        self.num_regs[1]
    }

    /// RPython: `JitCode.num_regs_f()`
    pub fn num_regs_f(&self) -> u16 {
        self.num_regs[2]
    }

    /// RPython: `JitCode.num_regs_and_consts_i()`
    pub fn num_regs_and_consts_i(&self) -> usize {
        self.num_regs[0] as usize + self.constants_i.len()
    }

    /// RPython: `JitCode.follow_jump(position)` — follow a label at position.
    pub fn follow_jump(&self, position: usize) -> usize {
        if position < 2 || position - 2 + 1 >= self.code.len() {
            return 0;
        }
        let pos = position - 2;
        (self.code[pos] as usize) | ((self.code[pos + 1] as usize) << 8)
    }

    /// RPython: `JitCode.dump()`
    pub fn dump(&self) -> String {
        format!(
            "<JitCode: {} bytes, {} int regs, {} consts>",
            self.code.len(),
            self.num_regs[0],
            self.constants_i.len()
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallTarget {
    pub trace_ptr: *const (),
    pub concrete_ptr: *const (),
}

impl JitCallTarget {
    pub fn new(trace_ptr: *const (), concrete_ptr: *const ()) -> Self {
        Self {
            trace_ptr,
            concrete_ptr,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct JitCallAssemblerTarget {
    token_number: u64,
    concrete_ptr: *const (),
}

impl JitCallAssemblerTarget {
    fn new(token_number: u64, concrete_ptr: *const ()) -> Self {
        Self {
            token_number,
            concrete_ptr,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitArgKind {
    Int = 0,
    Ref = 1,
    Float = 2,
}

impl JitArgKind {
    fn encode(self) -> u8 {
        self as u8
    }

    fn decode(byte: u8) -> Self {
        match byte {
            0 => Self::Int,
            1 => Self::Ref,
            2 => Self::Float,
            other => panic!("unknown jitcode arg kind {other}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallArg {
    pub kind: JitArgKind,
    pub reg: u16,
}

impl JitCallArg {
    pub fn int(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Int,
            reg,
        }
    }

    pub fn reference(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Ref,
            reg,
        }
    }

    pub fn float(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Float,
            reg,
        }
    }
}

// SAFETY: call targets are function pointers to immutable code.
unsafe impl Send for JitCode {}
unsafe impl Sync for JitCode {}

pub trait JitCodeSym {
    fn current_selected(&self) -> usize;
    fn current_selected_value(&self) -> Option<OpRef>;
    fn set_current_selected(&mut self, selected: usize);
    fn set_current_selected_value(&mut self, selected: usize, value: OpRef);
    fn guard_selected(&self) -> usize {
        self.current_selected()
    }
    fn guard_selected_value(&self) -> Option<OpRef> {
        self.current_selected_value()
    }
    fn selected_in_fail_args_prefix(&self) -> bool {
        false
    }
    fn close_requires_header_selected(&self) -> bool {
        true
    }
    fn begin_portal_op(&mut self, _pc: usize) {}
    fn commit_portal_op(&mut self) {}
    fn abort_portal_op(&mut self) {}
    fn stack(&self, selected: usize) -> Option<&SymbolicStack>;
    fn stack_mut(&mut self, selected: usize) -> Option<&mut SymbolicStack>;
    fn total_slots(&self) -> usize;
    fn loop_header_pc(&self) -> usize;
    /// RPython parity: initial_selected at the trace header.
    /// Used by BC_JUMP_TARGET to check green key match before CloseLoop.
    fn header_selected(&self) -> usize {
        self.current_selected()
    }
    /// Create a symbolic stack for a storage not in the initial layout.
    fn ensure_stack(&mut self, selected: usize, offset: usize, len: usize);
    /// Full interpreter-visible state to materialize on guard failure.
    ///
    /// When `None`, guards fall back to the legacy auto-generated fail args.
    fn fail_args(&self) -> Option<Vec<OpRef>>;

    /// Guard-failure state materialization that may record extra IR.
    ///
    /// Compact storage backends use this to attach logical stack state
    /// (lengths and pending writes) without exposing it as loop inputargs.
    fn fail_args_with_ctx(&mut self, _ctx: &mut TraceCtx) -> Option<Vec<OpRef>> {
        self.fail_args()
    }

    /// Current stack lengths in the same storage order as `fail_args()`.
    fn fail_storage_lengths(&self) -> Option<Vec<usize>> {
        None
    }

    /// Compact storage-pool support: raw data pointer for a traced storage.
    fn compact_storage_ptr(&self, _selected: usize) -> Option<OpRef> {
        None
    }

    /// Compact storage-pool support: current logical length for a traced storage.
    fn compact_storage_len(&self, _selected: usize) -> Option<OpRef> {
        None
    }

    /// Compact storage-pool support: current capacity for a traced storage.
    fn compact_storage_cap(&self, _selected: usize) -> Option<OpRef> {
        None
    }

    /// Update the symbolic length for a compact traced storage.
    fn set_compact_storage_len(&mut self, _selected: usize, _value: OpRef) {}

    /// Get a cached raw word for a concrete compact storage slot.
    fn compact_storage_slot_raw(&self, _selected: usize, _index: usize) -> Option<OpRef> {
        None
    }

    /// Record a cached raw word for a concrete compact storage slot.
    fn set_compact_storage_slot_raw(&mut self, _selected: usize, _index: usize, _raw: OpRef) {}

    /// Drop cached compact storage slots at and above `len`.
    fn truncate_compact_storage_slots(&mut self, _selected: usize, _len: usize) {}

    /// Ensure compact storage refs are available, recording any required loads.
    fn ensure_compact_storage_loaded(
        &mut self,
        _ctx: &mut TraceCtx,
        selected: usize,
    ) -> Option<(OpRef, OpRef, OpRef)> {
        Some((
            self.compact_storage_ptr(selected)?,
            self.compact_storage_len(selected)?,
            self.compact_storage_cap(selected)?,
        ))
    }

    /// Write the latest compact length back to heap metadata if needed.
    fn compact_storage_writeback_len(
        &mut self,
        _ctx: &mut TraceCtx,
        _selected: usize,
        _new_len: OpRef,
    ) {
    }

    /// Optional semantic bounds for compact storage values.
    fn compact_storage_bounds(&self) -> Option<(i64, i64)> {
        None
    }

    /// Decode a raw compact storage word into a semantic integer value.
    fn compact_storage_decode(&self, _ctx: &mut TraceCtx, raw: OpRef) -> OpRef {
        raw
    }

    /// Encode a semantic integer value into the raw compact storage representation.
    fn compact_storage_encode(&self, _ctx: &mut TraceCtx, value: OpRef) -> OpRef {
        value
    }

    // ── State field support (register/tape machines) ─────────────
    //
    // When state_fields is configured, scalar and array fields on the
    // interpreter state are tracked as OpRefs in the Sym.

    /// Read a scalar state field's current OpRef.
    fn state_field_ref(&self, _field_idx: usize) -> Option<OpRef> {
        None
    }

    /// Update a scalar state field's OpRef.
    fn set_state_field_ref(&mut self, _field_idx: usize, _value: OpRef) {}

    /// Read an array state field element's current OpRef.
    fn state_array_ref(&self, _array_idx: usize, _elem_idx: usize) -> Option<OpRef> {
        None
    }

    /// Update an array state field element's OpRef.
    fn set_state_array_ref(&mut self, _array_idx: usize, _elem_idx: usize, _value: OpRef) {}

    // ── State virtualizable array support ─────────────────────────
    //
    // For state_fields with `[type; virt]`: array stays on heap,
    // accessed via GetarrayitemRawI/SetarrayitemRaw. Only the data
    // pointer and length are tracked as inputargs.

    /// Get the data pointer OpRef for a virtualizable state array.
    fn state_varray_ptr(&self, _array_idx: usize) -> Option<OpRef> {
        None
    }

    /// Get the length OpRef for a virtualizable state array.
    fn state_varray_len(&self, _array_idx: usize) -> Option<OpRef> {
        None
    }
}

pub trait JitCodeRuntime {
    fn stack_len(&self, selected: usize) -> usize;
    fn stack_peek(&self, selected: usize, pos: usize) -> i64;
    fn label_at(&self, pc: usize) -> usize;
}

pub struct ClosureRuntime<FLen, FPeek, FLabel> {
    stack_len: FLen,
    stack_peek: FPeek,
    label_at: FLabel,
}

impl<FLen, FPeek, FLabel> ClosureRuntime<FLen, FPeek, FLabel> {
    pub fn new(stack_len: FLen, stack_peek: FPeek, label_at: FLabel) -> Self {
        Self {
            stack_len,
            stack_peek,
            label_at,
        }
    }
}

impl<FLen, FPeek, FLabel> JitCodeRuntime for ClosureRuntime<FLen, FPeek, FLabel>
where
    FLen: Fn(usize) -> usize,
    FPeek: Fn(usize, usize) -> i64,
    FLabel: Fn(usize) -> usize,
{
    fn stack_len(&self, selected: usize) -> usize {
        (self.stack_len)(selected)
    }

    fn stack_peek(&self, selected: usize, pos: usize) -> i64 {
        (self.stack_peek)(selected, pos)
    }

    fn label_at(&self, pc: usize) -> usize {
        (self.label_at)(pc)
    }
}

pub struct MIFrame<'a> {
    pub jitcode: &'a JitCode,
    pub pc: usize,
    pub code_cursor: usize,
    pub int_regs: Vec<Option<OpRef>>,
    pub int_values: Vec<Option<i64>>,
    pub ref_regs: Vec<Option<OpRef>>,
    pub ref_values: Vec<Option<i64>>,
    pub float_regs: Vec<Option<OpRef>>,
    pub float_values: Vec<Option<i64>>,
    pub inline_frame: bool,
    pub return_i: Option<(usize, usize)>,
    pub return_r: Option<(usize, usize)>,
    pub return_f: Option<(usize, usize)>,
}

impl<'a> MIFrame<'a> {
    pub fn new(jitcode: &'a JitCode, pc: usize) -> Self {
        Self {
            jitcode,
            pc,
            code_cursor: 0,
            int_regs: vec![None; jitcode.num_regs[0] as usize],
            int_values: vec![None; jitcode.num_regs[0] as usize],
            ref_regs: vec![None; jitcode.num_regs[1] as usize],
            ref_values: vec![None; jitcode.num_regs[1] as usize],
            float_regs: vec![None; jitcode.num_regs[2] as usize],
            float_values: vec![None; jitcode.num_regs[2] as usize],
            inline_frame: false,
            return_i: None,
            return_r: None,
            return_f: None,
        }
    }

    pub fn next_u8(&mut self) -> u8 {
        read_u8(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn next_u16(&mut self) -> u16 {
        read_u16(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn finished(&self) -> bool {
        self.code_cursor >= self.jitcode.code.len()
    }
}

pub struct MIFrameStack<'a> {
    frames: Vec<MIFrame<'a>>,
}

impl<'a> MIFrameStack<'a> {
    pub fn new(root: MIFrame<'a>) -> Self {
        Self { frames: vec![root] }
    }

    pub fn current_mut(&mut self) -> &mut MIFrame<'a> {
        self.frames.last_mut().expect("empty JitCode frame stack")
    }

    pub fn push(&mut self, frame: MIFrame<'a>) {
        self.frames.push(frame);
    }

    pub fn pop(&mut self) -> Option<MIFrame<'a>> {
        self.frames.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

pub struct JitCodeMachine<'a, S, R> {
    frames: MIFrameStack<'a>,
    runtime_stacks: HashMap<usize, Vec<i64>>,
    marker: PhantomData<(S, R)>,
}

#[derive(Clone)]
struct ActiveStandardVirtualizable {
    vable_opref: OpRef,
    info: crate::virtualizable::VirtualizableInfo,
    obj_ptr: *mut u8,
}

impl<'a, S, R> JitCodeMachine<'a, S, R>
where
    S: JitCodeSym,
    R: JitCodeRuntime,
{
    fn active_standard_virtualizable(
        &self,
        ctx: &TraceCtx,
    ) -> Option<ActiveStandardVirtualizable> {
        let vable_opref = ctx.standard_virtualizable_box()?;
        let info = ctx.virtualizable_info()?.clone();
        let obj_ptr = self.frames.frames.iter().rev().find_map(|frame| {
            frame
                .ref_regs
                .iter()
                .zip(frame.ref_values.iter())
                .find_map(|(opref, concrete)| {
                    (*opref == Some(vable_opref))
                        .then_some(*concrete)
                        .flatten()
                        .map(|value| value as usize as *mut u8)
                })
        })?;
        Some(ActiveStandardVirtualizable {
            vable_opref,
            info,
            obj_ptr,
        })
    }

    fn prepare_standard_virtualizable_before_residual_call(
        &mut self,
        ctx: &mut TraceCtx,
    ) -> Option<ActiveStandardVirtualizable> {
        let active = self.active_standard_virtualizable(ctx)?;
        unsafe {
            active.info.tracing_before_residual_call(active.obj_ptr);
        }
        let force_token = ctx.force_token();
        ctx.vable_setfield_descr(
            active.vable_opref,
            force_token,
            active.info.token_field_descr(),
        );
        Some(active)
    }

    fn finish_standard_virtualizable_after_residual_call(
        active: Option<ActiveStandardVirtualizable>,
    ) -> bool {
        let Some(active) = active else {
            return false;
        };
        unsafe { active.info.tracing_after_residual_call(active.obj_ptr) }
    }

    fn record_state_guard(
        ctx: &mut TraceCtx,
        sym: &mut S,
        opcode: OpCode,
        args: &[OpRef],
        extra_fail_args: &[OpRef],
    ) {
        if let Some(mut fail_args) = sym.fail_args_with_ctx(ctx) {
            if let Some(lengths) = sym.fail_storage_lengths() {
                fail_args.extend(lengths.into_iter().map(|len| ctx.const_int(len as i64)));
            }
            if !sym.selected_in_fail_args_prefix() {
                let selected = sym
                    .guard_selected_value()
                    .unwrap_or_else(|| ctx.const_int(sym.guard_selected() as i64));
                fail_args.push(selected);
            }
            fail_args.extend_from_slice(extra_fail_args);
            ctx.record_guard_with_fail_args(opcode, args, fail_args.len(), &fail_args);
        } else {
            ctx.record_guard(opcode, args, sym.total_slots());
        }
    }

    fn compact_storage_refs(
        ctx: &mut TraceCtx,
        sym: &mut S,
        selected: usize,
    ) -> Option<(OpRef, OpRef, OpRef)> {
        sym.ensure_compact_storage_loaded(ctx, selected)
    }

    fn raw_word_array_descr() -> majit_ir::DescrRef {
        majit_ir::descr::make_array_descr(0, 8, majit_ir::Type::Int)
    }

    fn standard_vable_field_offset(ctx: &TraceCtx, field_idx: usize) -> Option<(OpRef, usize)> {
        let vable_opref = ctx.standard_virtualizable_box()?;
        let info = ctx.virtualizable_info()?;
        let field = info.static_fields.get(field_idx)?;
        Some((vable_opref, field.offset))
    }

    fn standard_vable_array_offset(ctx: &TraceCtx, array_idx: usize) -> Option<(OpRef, usize)> {
        let vable_opref = ctx.standard_virtualizable_box()?;
        let info = ctx.virtualizable_info()?;
        let array = info.array_fields.get(array_idx)?;
        Some((vable_opref, array.field_offset))
    }

    fn compact_load_raw(ctx: &mut TraceCtx, ptr: OpRef, index: OpRef) -> OpRef {
        ctx.record_op_with_descr(
            OpCode::GetarrayitemRawI,
            &[ptr, index],
            Self::raw_word_array_descr(),
        )
    }

    fn compact_store_raw(ctx: &mut TraceCtx, ptr: OpRef, index: OpRef, raw: OpRef) {
        ctx.record_op_with_descr(
            OpCode::SetarrayitemRaw,
            &[ptr, index, raw],
            Self::raw_word_array_descr(),
        );
    }

    fn compact_resume_pc(&mut self, ctx: &mut TraceCtx) -> OpRef {
        ctx.const_int(self.frames.current_mut().pc as i64)
    }

    fn compact_guard_push_capacity(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        len: OpRef,
        cap: OpRef,
    ) {
        let has_room = ctx.record_op(OpCode::IntLt, &[len, cap]);
        let resume_pc = self.compact_resume_pc(ctx);
        Self::record_state_guard(ctx, sym, OpCode::GuardTrue, &[has_room], &[resume_pc]);
    }

    fn compact_guard_required_stack(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        len: OpRef,
        required: usize,
        concrete_len: usize,
    ) {
        let required_ref = ctx.const_int(required as i64);
        let has_enough = ctx.record_op(OpCode::IntGe, &[len, required_ref]);
        let opcode = if concrete_len < required {
            OpCode::GuardFalse
        } else {
            OpCode::GuardTrue
        };
        let resume_pc = if concrete_len < required {
            // BRPOP-style traces that took the branch must resume on the
            // fallthrough instruction when the guard fails.
            ctx.const_int((self.frames.current_mut().pc + 1) as i64)
        } else {
            // Non-branching traces must re-run the current control opcode so
            // the interpreter can take the branch concretely on guard failure.
            self.compact_resume_pc(ctx)
        };
        Self::record_state_guard(ctx, sym, opcode, &[has_enough], &[resume_pc]);
    }

    fn compact_guard_value_bounds(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        value: OpRef,
        concrete: i64,
    ) -> Result<(), TraceAction> {
        let Some((min, max)) = sym.compact_storage_bounds() else {
            return Ok(());
        };
        if concrete < min || concrete > max {
            return Err(TraceAction::Abort);
        }
        let resume_pc = self.compact_resume_pc(ctx);
        let min_ref = ctx.const_int(min);
        let lower_ok = ctx.record_op(OpCode::IntGe, &[value, min_ref]);
        Self::record_state_guard(ctx, sym, OpCode::GuardTrue, &[lower_ok], &[resume_pc]);
        let max_ref = ctx.const_int(max);
        let upper_ok = ctx.record_op(OpCode::IntLe, &[value, max_ref]);
        Self::record_state_guard(ctx, sym, OpCode::GuardTrue, &[upper_ok], &[resume_pc]);
        Ok(())
    }

    fn compact_pop_int(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        runtime: &R,
        selected: usize,
    ) -> Result<(OpRef, i64), TraceAction> {
        let concrete = self
            .runtime_stack_mut(selected, runtime)
            .pop()
            .ok_or(TraceAction::Abort)?;
        let concrete_new_len = self.runtime_stack_mut(selected, runtime).len();
        let Some((ptr, len, _)) = Self::compact_storage_refs(ctx, sym, selected) else {
            return Err(TraceAction::Abort);
        };
        let one = ctx.const_int(1);
        let new_len = ctx.record_op(OpCode::IntSub, &[len, one]);
        let raw = sym
            .compact_storage_slot_raw(selected, concrete_new_len)
            .unwrap_or_else(|| Self::compact_load_raw(ctx, ptr, new_len));
        sym.truncate_compact_storage_slots(selected, concrete_new_len);
        let decoded = sym.compact_storage_decode(ctx, raw);
        sym.set_compact_storage_len(selected, new_len);
        sym.compact_storage_writeback_len(ctx, selected, new_len);
        Ok((decoded, concrete))
    }

    fn compact_peek_int(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        runtime: &R,
        selected: usize,
    ) -> Result<(OpRef, i64), TraceAction> {
        let concrete_stack = self.runtime_stack_mut(selected, runtime);
        let concrete = concrete_stack.last().copied().ok_or(TraceAction::Abort)?;
        let concrete_index = concrete_stack.len() - 1;
        let Some((ptr, len, _)) = Self::compact_storage_refs(ctx, sym, selected) else {
            return Err(TraceAction::Abort);
        };
        let one = ctx.const_int(1);
        let index = ctx.record_op(OpCode::IntSub, &[len, one]);
        let raw = sym
            .compact_storage_slot_raw(selected, concrete_index)
            .unwrap_or_else(|| Self::compact_load_raw(ctx, ptr, index));
        let decoded = sym.compact_storage_decode(ctx, raw);
        Ok((decoded, concrete))
    }

    fn compact_push_int(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        runtime: &R,
        selected: usize,
        value: OpRef,
        concrete: i64,
    ) -> Result<(), TraceAction> {
        let Some((ptr, len, cap)) = Self::compact_storage_refs(ctx, sym, selected) else {
            return Err(TraceAction::Abort);
        };
        let concrete_index = self.runtime_stack_mut(selected, runtime).len();
        self.compact_guard_push_capacity(ctx, sym, len, cap);
        self.compact_guard_value_bounds(ctx, sym, value, concrete)?;
        let raw = sym.compact_storage_encode(ctx, value);
        Self::compact_store_raw(ctx, ptr, len, raw);
        sym.set_compact_storage_slot_raw(selected, concrete_index, raw);
        let one = ctx.const_int(1);
        let new_len = ctx.record_op(OpCode::IntAdd, &[len, one]);
        sym.set_compact_storage_len(selected, new_len);
        sym.compact_storage_writeback_len(ctx, selected, new_len);
        self.runtime_stack_mut(selected, runtime).push(concrete);
        Ok(())
    }

    fn compact_push_raw(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        runtime: &R,
        selected: usize,
        raw: OpRef,
        concrete: i64,
    ) -> Result<(), TraceAction> {
        let Some((ptr, len, cap)) = Self::compact_storage_refs(ctx, sym, selected) else {
            return Err(TraceAction::Abort);
        };
        let concrete_index = self.runtime_stack_mut(selected, runtime).len();
        self.compact_guard_push_capacity(ctx, sym, len, cap);
        Self::compact_store_raw(ctx, ptr, len, raw);
        sym.set_compact_storage_slot_raw(selected, concrete_index, raw);
        let one = ctx.const_int(1);
        let new_len = ctx.record_op(OpCode::IntAdd, &[len, one]);
        sym.set_compact_storage_len(selected, new_len);
        sym.compact_storage_writeback_len(ctx, selected, new_len);
        self.runtime_stack_mut(selected, runtime).push(concrete);
        Ok(())
    }

    pub fn new(
        root: MIFrame<'a>,
        _sub_jitcodes: &'a [JitCode],
        _fn_ptrs: &'a [JitCallTarget],
    ) -> Self {
        Self {
            frames: MIFrameStack::new(root),
            runtime_stacks: HashMap::new(),
            marker: PhantomData,
        }
    }

    pub fn run_to_end(&mut self, ctx: &mut TraceCtx, sym: &mut S, runtime: &R) -> TraceAction {
        let portal_pc = self.frames.current_mut().pc;
        sym.begin_portal_op(portal_pc);
        while !self.frames.is_empty() {
            // Catch panics from BigInt overflow in runtime stack operations.
            // RPython doesn't have this issue (no BigInt); we abort the trace.
            let action = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.run_one_step(ctx, sym, runtime)
            })) {
                Ok(a) => a,
                Err(payload) => {
                    if crate::majit_log_enabled() {
                        let message = if let Some(msg) = payload.downcast_ref::<&str>() {
                            *msg
                        } else if let Some(msg) = payload.downcast_ref::<String>() {
                            msg.as_str()
                        } else {
                            "<non-string panic payload>"
                        };
                        eprintln!(
                            "[jit] trace_jitcode panic while tracing pc={}: {}",
                            self.frames.current_mut().pc,
                            message
                        );
                    }
                    sym.abort_portal_op();
                    return TraceAction::AbortPermanent;
                }
            };
            if !matches!(action, TraceAction::Continue) {
                match action {
                    TraceAction::CloseLoop => sym.commit_portal_op(),
                    _ => sym.abort_portal_op(),
                }
                return action;
            }
        }

        if ctx.is_too_long() {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] trace_jitcode aborting: trace too long at portal pc={}",
                    portal_pc
                );
            }
            sym.abort_portal_op();
            TraceAction::AbortPermanent
        } else {
            sym.commit_portal_op();
            TraceAction::Continue
        }
    }

    pub fn run_one_step(&mut self, ctx: &mut TraceCtx, sym: &mut S, runtime: &R) -> TraceAction {
        if self.frames.is_empty() {
            return if ctx.is_too_long() {
                if crate::majit_log_enabled() {
                    eprintln!("[jit] trace_jitcode aborting: trace too long with empty frame");
                }
                TraceAction::AbortPermanent
            } else {
                TraceAction::Continue
            };
        }

        let finished = {
            let frame = self.frames.current_mut();
            frame.finished()
        };
        if finished {
            let finished_frame = self.frames.pop().expect("finished frame stack was empty");
            if finished_frame.inline_frame {
                ctx.pop_inline_frame();
            }
            if let Some(parent) = self.frames.frames.last_mut() {
                if let Some((callee_src, caller_dst)) = finished_frame.return_i {
                    parent.int_regs[caller_dst] = finished_frame.int_regs[callee_src];
                    parent.int_values[caller_dst] = finished_frame.int_values[callee_src];
                }
                if let Some((callee_src, caller_dst)) = finished_frame.return_r {
                    parent.ref_regs[caller_dst] = finished_frame.ref_regs[callee_src];
                    parent.ref_values[caller_dst] = finished_frame.ref_values[callee_src];
                }
                if let Some((callee_src, caller_dst)) = finished_frame.return_f {
                    parent.float_regs[caller_dst] = finished_frame.float_regs[callee_src];
                    parent.float_values[caller_dst] = finished_frame.float_values[callee_src];
                }
            }
            return TraceAction::Continue;
        }

        let bytecode = self.frames.current_mut().next_u8();
        match bytecode {
            BC_LOAD_CONST_I => {
                let (dst, value) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let const_idx = frame.next_u16() as usize;
                    let value = *frame
                        .jitcode
                        .constants_i
                        .get(const_idx)
                        .expect("jitcode const index out of bounds");
                    (dst, value)
                };
                self.set_int_reg(dst, Some(ctx.const_int(value)), Some(value));
            }
            BC_POP_I => {
                let dst = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                    let Ok((symbolic, concrete)) =
                        self.compact_pop_int(ctx, sym, runtime, selected)
                    else {
                        return TraceAction::Abort;
                    };
                    self.set_int_reg(dst, Some(symbolic), Some(concrete));
                } else {
                    let symbolic = {
                        let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                        stack.pop()
                    };
                    let concrete = self.runtime_stack_mut(selected, runtime).pop();
                    self.set_int_reg(dst, symbolic, concrete);
                }
            }
            BC_PEEK_I => {
                let dst = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                    let Ok((symbolic, concrete)) =
                        self.compact_peek_int(ctx, sym, runtime, selected)
                    else {
                        return TraceAction::Abort;
                    };
                    self.set_int_reg(dst, Some(symbolic), Some(concrete));
                } else {
                    // RPython parity: peek from boxes (SymbolicStack), no IR.
                    let symbolic = {
                        let stack = sym.stack(selected).expect("missing symbolic stack");
                        stack.peek()
                    };
                    let concrete = self.runtime_stack_mut(selected, runtime).last().copied();
                    self.set_int_reg(dst, symbolic, concrete);
                }
            }
            BC_PUSH_I => {
                let src = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                let (value, concrete) = self.read_int_reg(src);
                if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                    if self
                        .compact_push_int(ctx, sym, runtime, selected, value, concrete)
                        .is_err()
                    {
                        return TraceAction::Abort;
                    }
                } else {
                    let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                    stack.push(value);
                    self.runtime_stack_mut(selected, runtime).push(concrete);
                }
            }
            BC_POP_DISCARD => {
                let selected = sym.current_selected();
                if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                    if self.runtime_stack_mut(selected, runtime).pop().is_none() {
                        return TraceAction::Abort;
                    }
                    let concrete_new_len = self.runtime_stack_mut(selected, runtime).len();
                    let Some((_, len, _)) = Self::compact_storage_refs(ctx, sym, selected) else {
                        return TraceAction::Abort;
                    };
                    let one = ctx.const_int(1);
                    let new_len = ctx.record_op(OpCode::IntSub, &[len, one]);
                    sym.truncate_compact_storage_slots(selected, concrete_new_len);
                    sym.set_compact_storage_len(selected, new_len);
                    sym.compact_storage_writeback_len(ctx, selected, new_len);
                } else {
                    let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                    let _ = stack.pop();
                    let _ = self.runtime_stack_mut(selected, runtime).pop();
                }
            }
            BC_DUP_STACK => {
                let selected = sym.current_selected();
                if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                    let concrete_stack = self.runtime_stack_mut(selected, runtime);
                    let Some(concrete) = concrete_stack.last().copied() else {
                        return TraceAction::Abort;
                    };
                    let concrete_top_index = concrete_stack.len() - 1;
                    let Some((ptr, len, _)) = Self::compact_storage_refs(ctx, sym, selected) else {
                        return TraceAction::Abort;
                    };
                    let one = ctx.const_int(1);
                    let top_index = ctx.record_op(OpCode::IntSub, &[len, one]);
                    let raw = sym
                        .compact_storage_slot_raw(selected, concrete_top_index)
                        .unwrap_or_else(|| Self::compact_load_raw(ctx, ptr, top_index));
                    if self
                        .compact_push_raw(ctx, sym, runtime, selected, raw, concrete)
                        .is_err()
                    {
                        return TraceAction::Abort;
                    }
                } else {
                    let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                    stack.dup();
                    let value = self
                        .runtime_stack_mut(selected, runtime)
                        .last()
                        .copied()
                        .expect("cannot dup from empty runtime stack");
                    self.runtime_stack_mut(selected, runtime).push(value);
                }
            }
            BC_SWAP_STACK => {
                let selected = sym.current_selected();
                if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                    let runtime_stack = self.runtime_stack_mut(selected, runtime);
                    let len = runtime_stack.len();
                    assert!(
                        len >= 2,
                        "cannot swap runtime stack with fewer than two values"
                    );
                    runtime_stack.swap(len - 1, len - 2);

                    let Some((ptr, len_ref, _)) = Self::compact_storage_refs(ctx, sym, selected)
                    else {
                        return TraceAction::Abort;
                    };
                    let one = ctx.const_int(1);
                    let two = ctx.const_int(2);
                    let top_index = ctx.record_op(OpCode::IntSub, &[len_ref, one]);
                    let prev_index = ctx.record_op(OpCode::IntSub, &[len_ref, two]);
                    let raw_top = sym
                        .compact_storage_slot_raw(selected, len - 1)
                        .unwrap_or_else(|| Self::compact_load_raw(ctx, ptr, top_index));
                    let raw_prev = sym
                        .compact_storage_slot_raw(selected, len - 2)
                        .unwrap_or_else(|| Self::compact_load_raw(ctx, ptr, prev_index));
                    Self::compact_store_raw(ctx, ptr, prev_index, raw_top);
                    Self::compact_store_raw(ctx, ptr, top_index, raw_prev);
                    sym.set_compact_storage_slot_raw(selected, len - 2, raw_top);
                    sym.set_compact_storage_slot_raw(selected, len - 1, raw_prev);
                } else {
                    let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                    stack.swap();
                    let stack = self.runtime_stack_mut(selected, runtime);
                    let len = stack.len();
                    assert!(
                        len >= 2,
                        "cannot swap runtime stack with fewer than two values"
                    );
                    stack.swap(len - 1, len - 2);
                }
            }
            BC_COPY_FROM_BOTTOM => {
                // Copy element at index (from bottom) to top of stack.
                let idx_reg = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                    return TraceAction::Abort;
                }
                let (idx_sym, idx_concrete) = self.read_int_reg(idx_reg);
                let index = idx_concrete as usize;

                let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                let opref = stack.peek_at(index);
                stack.push(opref);

                let rt_stack = self.runtime_stack_mut(selected, runtime);
                let val = rt_stack[index];
                rt_stack.push(val);
                let _ = idx_sym; // index is a trace-time constant; symbolic opref comes from stack
            }
            BC_STORE_DOWN => {
                // Pop top of stack, store at index (from bottom).
                let idx_reg = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                    return TraceAction::Abort;
                }
                let (idx_sym, idx_concrete) = self.read_int_reg(idx_reg);
                let index = idx_concrete as usize;

                let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                let opref = stack.pop().expect("store_down on empty stack");
                stack.set_at(index, opref);

                let rt_stack = self.runtime_stack_mut(selected, runtime);
                let val = rt_stack.pop().expect("store_down on empty runtime stack");
                rt_stack[index] = val;
                let _ = idx_sym;
            }

            // ── State field access (register/tape machines) ──
            BC_LOAD_STATE_FIELD => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let opref = sym
                    .state_field_ref(field_idx)
                    .expect("state field not initialized");
                // Runtime value not needed — we track symbolically.
                self.set_int_reg(dest, Some(opref), Some(0));
            }
            BC_STORE_STATE_FIELD => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let (opref, _) = self.read_int_reg(src);
                sym.set_state_field_ref(field_idx, opref);
            }
            BC_LOAD_STATE_ARRAY => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let (_, index_concrete) = self.read_int_reg(index_reg);
                let elem_idx = index_concrete as usize;
                let opref = sym.state_array_ref(array_idx, elem_idx);
                if let Some(opref) = opref {
                    self.set_int_reg(dest, Some(opref), Some(0));
                } else {
                    // Array element beyond initialized range (e.g., push expanded).
                    // Abort trace — this path needs dynamic array support.
                    return TraceAction::Abort;
                }
            }
            BC_STORE_STATE_ARRAY => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let (_, index_concrete) = self.read_int_reg(index_reg);
                let elem_idx = index_concrete as usize;
                let (opref, _) = self.read_int_reg(src);
                sym.set_state_array_ref(array_idx, elem_idx, opref);
            }

            // ── First-class virtualizable access (RPython getfield_vable_*) ──
            BC_GETFIELD_VABLE_I => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, field_offset)) =
                    Self::standard_vable_field_offset(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let result = ctx.vable_getfield_int(vable_opref, field_offset);
                self.set_int_reg(dest, Some(result), Some(0));
            }
            BC_GETFIELD_VABLE_R => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, field_offset)) =
                    Self::standard_vable_field_offset(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let result = ctx.vable_getfield_ref(vable_opref, field_offset);
                self.set_ref_reg(dest, Some(result), Some(0));
            }
            BC_GETFIELD_VABLE_F => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, field_offset)) =
                    Self::standard_vable_field_offset(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let result = ctx.vable_getfield_float(vable_opref, field_offset);
                self.set_float_reg(dest, Some(result), Some(0));
            }
            BC_SETFIELD_VABLE_I => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, field_offset)) =
                    Self::standard_vable_field_offset(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let (value, _) = self.read_int_reg(src);
                ctx.vable_setfield(vable_opref, field_offset, value);
            }
            BC_SETFIELD_VABLE_R => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, field_offset)) =
                    Self::standard_vable_field_offset(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let (value, _) = self.read_ref_reg(src);
                ctx.vable_setfield(vable_opref, field_offset, value);
            }
            BC_SETFIELD_VABLE_F => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, field_offset)) =
                    Self::standard_vable_field_offset(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let (value, _) = self.read_float_reg(src);
                ctx.vable_setfield(vable_opref, field_offset, value);
            }
            BC_GETARRAYITEM_VABLE_I => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, array_field_offset)) =
                    Self::standard_vable_array_offset(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, _) = self.read_int_reg(index_reg);
                let result =
                    ctx.vable_getarrayitem_int_indexed(vable_opref, index, array_field_offset);
                self.set_int_reg(dest, Some(result), Some(0));
            }
            BC_GETARRAYITEM_VABLE_R => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, array_field_offset)) =
                    Self::standard_vable_array_offset(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, _) = self.read_int_reg(index_reg);
                let result =
                    ctx.vable_getarrayitem_ref_indexed(vable_opref, index, array_field_offset);
                self.set_ref_reg(dest, Some(result), Some(0));
            }
            BC_GETARRAYITEM_VABLE_F => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, array_field_offset)) =
                    Self::standard_vable_array_offset(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, _) = self.read_int_reg(index_reg);
                let result =
                    ctx.vable_getarrayitem_float_indexed(vable_opref, index, array_field_offset);
                self.set_float_reg(dest, Some(result), Some(0));
            }
            BC_SETARRAYITEM_VABLE_I => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, array_field_offset)) =
                    Self::standard_vable_array_offset(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, _) = self.read_int_reg(index_reg);
                let (value, _) = self.read_int_reg(src);
                ctx.vable_setarrayitem_indexed(vable_opref, index, array_field_offset, value);
            }
            BC_SETARRAYITEM_VABLE_R => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, array_field_offset)) =
                    Self::standard_vable_array_offset(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, _) = self.read_int_reg(index_reg);
                let (value, _) = self.read_ref_reg(src);
                ctx.vable_setarrayitem_indexed(vable_opref, index, array_field_offset, value);
            }
            BC_SETARRAYITEM_VABLE_F => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, array_field_offset)) =
                    Self::standard_vable_array_offset(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, _) = self.read_int_reg(index_reg);
                let (value, _) = self.read_float_reg(src);
                ctx.vable_setarrayitem_indexed(vable_opref, index, array_field_offset, value);
            }
            BC_ARRAYLEN_VABLE => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, array_field_offset)) =
                    Self::standard_vable_array_offset(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let result = ctx.vable_arraylen_vable(vable_opref, array_field_offset);
                self.set_int_reg(dest, Some(result), Some(0));
            }
            BC_HINT_FORCE_VIRTUALIZABLE => {
                let Some(vable_opref) = ctx.standard_virtualizable_box() else {
                    return TraceAction::Abort;
                };
                ctx.gen_store_back_in_vable(vable_opref);
            }

            // ── Virtualizable state array access ──
            // Array stays on heap; emit raw memory load/store IR ops.
            BC_LOAD_STATE_VARRAY => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let (index_opref, _) = self.read_int_reg(index_reg);
                let array_ptr = sym
                    .state_varray_ptr(array_idx)
                    .expect("virtualizable array not initialized");
                let result = ctx.record_op_with_descr(
                    OpCode::GetarrayitemRawI,
                    &[array_ptr, index_opref],
                    Self::raw_word_array_descr(),
                );
                self.set_int_reg(dest, Some(result), Some(0));
            }
            BC_STORE_STATE_VARRAY => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let (index_opref, _) = self.read_int_reg(index_reg);
                let (value_opref, _) = self.read_int_reg(src);
                let array_ptr = sym
                    .state_varray_ptr(array_idx)
                    .expect("virtualizable array not initialized");
                ctx.record_op_with_descr(
                    OpCode::SetarrayitemRaw,
                    &[array_ptr, index_opref, value_opref],
                    Self::raw_word_array_descr(),
                );
            }

            BC_RECORD_BINOP_I => {
                let (dst, lhs_idx, rhs_idx, opcode) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let opcode_idx = frame.next_u16() as usize;
                    let lhs_idx = frame.next_u16() as usize;
                    let rhs_idx = frame.next_u16() as usize;
                    let opcode = *frame
                        .jitcode
                        .opcodes
                        .get(opcode_idx)
                        .expect("jitcode opcode index out of bounds");
                    (dst, lhs_idx, rhs_idx, opcode)
                };
                let (lhs, lhs_value) = self.read_int_reg(lhs_idx);
                let (rhs, rhs_value) = self.read_int_reg(rhs_idx);
                if opcode.is_ovf() {
                    // Overflow-checked op: abort trace if overflow occurs.
                    match eval_binop_ovf(opcode, lhs_value, rhs_value) {
                        Some(value) => {
                            let result = ctx.record_op(opcode, &[lhs, rhs]);
                            let resume_pc = ctx.const_int(self.frames.current_mut().pc as i64);
                            Self::record_state_guard(
                                ctx,
                                sym,
                                OpCode::GuardNoOverflow,
                                &[],
                                &[resume_pc],
                            );
                            self.set_int_reg(dst, Some(result), Some(value));
                        }
                        None => return TraceAction::Abort,
                    }
                } else {
                    let value = eval_binop_i(opcode, lhs_value, rhs_value);
                    self.set_int_reg(dst, Some(ctx.record_op(opcode, &[lhs, rhs])), Some(value));
                }
            }
            BC_RECORD_UNARY_I => {
                let (dst, src_idx, opcode) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let opcode_idx = frame.next_u16() as usize;
                    let src_idx = frame.next_u16() as usize;
                    let opcode = *frame
                        .jitcode
                        .opcodes
                        .get(opcode_idx)
                        .expect("jitcode opcode index out of bounds");
                    (dst, src_idx, opcode)
                };
                let (src, src_value) = self.read_int_reg(src_idx);
                let value = eval_unary_i(opcode, src_value);
                self.set_int_reg(dst, Some(ctx.record_op(opcode, &[src])), Some(value));
            }
            BC_REQUIRE_STACK => {
                let required = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                let concrete_len = self.runtime_stack_mut(selected, runtime).len();
                if let Some((_, len, _)) = Self::compact_storage_refs(ctx, sym, selected) {
                    // RPython parity: the BRPOP path is part of the traced path,
                    // so compact-storage mode must guard on the current stack
                    // depth before the interpreter decides whether to jump.
                    self.compact_guard_required_stack(ctx, sym, len, required, concrete_len);
                }
                if concrete_len < required {
                    // Stack insufficient: the interpreter will take the branch.
                    // Don't abort the trace — just skip this jitcode's remaining
                    // bytecodes. The branch direction is handled at interpreter level.
                }
            }
            BC_BRANCH_ZERO => {
                let selected = sym.current_selected();
                let (cond, runtime_cond) =
                    if Self::compact_storage_refs(ctx, sym, selected).is_some() {
                        let Ok((cond, runtime_cond)) =
                            self.compact_pop_int(ctx, sym, runtime, selected)
                        else {
                            return TraceAction::Abort;
                        };
                        (cond, runtime_cond)
                    } else {
                        let runtime_cond = {
                            let runtime_stack = self.runtime_stack_mut(selected, runtime);
                            let Some(value) = runtime_stack.pop() else {
                                return TraceAction::Abort;
                            };
                            value
                        };
                        let cond = {
                            let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                            stack.pop().expect("branch_zero on empty symbolic stack")
                        };
                        (cond, runtime_cond)
                    };
                let pc = self.frames.current_mut().pc;
                // RPython parity: check ALL green keys (pc + selected).
                let close_loop = runtime.label_at(pc) == sym.loop_header_pc()
                    && (!sym.close_requires_header_selected()
                        || sym.current_selected() == sym.header_selected());
                let opcode = if runtime_cond == 0 {
                    OpCode::GuardFalse
                } else {
                    OpCode::GuardTrue
                };
                let resume_pc = if runtime_cond == 0 {
                    (pc + 1) as i64
                } else {
                    runtime.label_at(pc) as i64
                };
                let resume_pc = ctx.const_int(resume_pc);
                Self::record_state_guard(ctx, sym, opcode, &[cond], &[resume_pc]);
                if runtime_cond == 0 && close_loop {
                    return TraceAction::CloseLoop;
                }
            }
            BC_BRANCH_REG_ZERO => {
                let (cond_idx, target) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (cond, cond_value) = self.read_int_reg(cond_idx);
                let branch_taken = cond_value == 0;
                let opcode = if branch_taken {
                    OpCode::GuardFalse
                } else {
                    OpCode::GuardTrue
                };
                let resume_pc = ctx.const_int(self.frames.current_mut().pc as i64);
                Self::record_state_guard(ctx, sym, opcode, &[cond], &[resume_pc]);
                if branch_taken {
                    self.frames.current_mut().code_cursor = target;
                }
            }
            BC_JUMP_TARGET => {
                let pc = self.frames.current_mut().pc;
                // RPython parity: reached_loop_header checks ALL green keys.
                // Only close if both pc AND selected match the header.
                if runtime.label_at(pc) == sym.loop_header_pc()
                    && (!sym.close_requires_header_selected()
                        || sym.current_selected() == sym.header_selected())
                {
                    return TraceAction::CloseLoop;
                }
            }
            BC_JUMP => {
                let target = self.frames.current_mut().next_u16() as usize;
                self.frames.current_mut().code_cursor = target;
            }
            BC_INLINE_CALL => {
                let (sub_idx, arg_triples, return_i, return_r, return_f) = {
                    let frame = self.frames.current_mut();
                    let sub_idx = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_triples = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let caller_src = frame.next_u16() as usize;
                        let callee_dst = frame.next_u16() as usize;
                        arg_triples.push((kind, caller_src, callee_dst));
                    }
                    let decode_return_slot = |f: &mut MIFrame| {
                        let src = f.next_u16() as usize;
                        let dst = f.next_u16() as usize;
                        if src == u16::MAX as usize && dst == u16::MAX as usize {
                            None
                        } else {
                            Some((src, dst))
                        }
                    };
                    let return_i = decode_return_slot(frame);
                    let return_r = decode_return_slot(frame);
                    let return_f = decode_return_slot(frame);
                    (sub_idx, arg_triples, return_i, return_r, return_f)
                };
                let pc = self.frames.current_mut().pc;
                let sub_jitcode = &self.frames.current_mut().jitcode.sub_jitcodes[sub_idx];
                let mut sub_frame = MIFrame::new(sub_jitcode, pc);
                ctx.push_inline_frame(((pc as u64) << 32) | sub_idx as u64, u32::MAX);
                sub_frame.inline_frame = true;
                for (kind, caller_src, callee_dst) in arg_triples {
                    match kind {
                        JitArgKind::Int => {
                            let (value, concrete) = self.read_int_reg(caller_src);
                            sub_frame.int_regs[callee_dst] = Some(value);
                            sub_frame.int_values[callee_dst] = Some(concrete);
                        }
                        JitArgKind::Ref => {
                            let (value, concrete) = self.read_ref_reg(caller_src);
                            sub_frame.ref_regs[callee_dst] = Some(value);
                            sub_frame.ref_values[callee_dst] = Some(concrete);
                        }
                        JitArgKind::Float => {
                            let (value, concrete) = self.read_float_reg(caller_src);
                            sub_frame.float_regs[callee_dst] = Some(value);
                            sub_frame.float_values[callee_dst] = Some(concrete);
                        }
                    }
                }
                sub_frame.return_i = return_i;
                sub_frame.return_r = return_r;
                sub_frame.return_f = return_f;
                self.frames.push(sub_frame);
            }
            BC_RESIDUAL_CALL_VOID
            | BC_CALL_MAY_FORCE_VOID
            | BC_CALL_RELEASE_GIL_VOID
            | BC_CALL_LOOPINVARIANT_VOID
            | BC_CALL_ASSEMBLER_VOID => {
                let (fn_ptr_idx, arg_regs) = {
                    let frame = self.frames.current_mut();
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    (fn_ptr_idx, arg_regs)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                if bytecode == BC_CALL_ASSEMBLER_VOID {
                    let target = self.frames.current_mut().jitcode.assembler_targets[fn_ptr_idx];
                    let token = JitCellToken::new(target.token_number);
                    ctx.call_assembler_void_typed(&token, &args, &arg_types);
                    call_void_function(target.concrete_ptr, &concrete_args);
                } else {
                    let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                    let trace_ptr = if target.trace_ptr.is_null() {
                        target.concrete_ptr
                    } else {
                        target.trace_ptr
                    };
                    let concrete_ptr = if target.concrete_ptr.is_null() {
                        trace_ptr
                    } else {
                        target.concrete_ptr
                    };
                    let active_vable = if bytecode == BC_CALL_MAY_FORCE_VOID {
                        self.prepare_standard_virtualizable_before_residual_call(ctx)
                    } else {
                        None
                    };
                    match bytecode {
                        BC_RESIDUAL_CALL_VOID => ctx.call_void_typed(trace_ptr, &args, &arg_types),
                        BC_CALL_MAY_FORCE_VOID => {
                            ctx.call_may_force_void_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_RELEASE_GIL_VOID => {
                            ctx.call_release_gil_void_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_LOOPINVARIANT_VOID => {
                            ctx.call_loopinvariant_void_typed(trace_ptr, &args, &arg_types)
                        }
                        _ => unreachable!(),
                    }
                    call_void_function(concrete_ptr, &concrete_args);
                    if bytecode == BC_CALL_MAY_FORCE_VOID {
                        if Self::finish_standard_virtualizable_after_residual_call(active_vable) {
                            return TraceAction::Abort;
                        }
                        ctx.guard_not_forced(sym.total_slots());
                    }
                }
            }
            BC_SET_SELECTED => {
                let const_idx = {
                    let frame = self.frames.current_mut();
                    frame.next_u16() as usize
                };
                let new_selected = {
                    let frame = self.frames.current_mut();
                    *frame
                        .jitcode
                        .constants_i
                        .get(const_idx)
                        .expect("jitcode const index out of bounds") as usize
                };
                if Self::compact_storage_refs(ctx, sym, new_selected).is_none()
                    && sym.stack(new_selected).is_none()
                {
                    if Self::compact_storage_refs(ctx, sym, sym.current_selected()).is_some() {
                        return TraceAction::Abort;
                    }
                    let len = runtime.stack_len(new_selected);
                    let offset = sym.total_slots();
                    sym.ensure_stack(new_selected, offset, len);
                    let _ = self.runtime_stack_mut(new_selected, runtime);
                }
                let selected_value = ctx.const_int(new_selected as i64);
                sym.set_current_selected_value(new_selected, selected_value);
            }
            BC_PUSH_TO => {
                let (src_idx, target) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (value, concrete) = self.read_int_reg(src_idx);
                if Self::compact_storage_refs(ctx, sym, target).is_some() {
                    if self
                        .compact_push_int(ctx, sym, runtime, target, value, concrete)
                        .is_err()
                    {
                        return TraceAction::Abort;
                    }
                } else {
                    if Self::compact_storage_refs(ctx, sym, sym.current_selected()).is_some()
                        && sym.stack(target).is_none()
                    {
                        return TraceAction::Abort;
                    }
                    if sym.stack(target).is_none() {
                        let len = runtime.stack_len(target);
                        let offset = sym.total_slots();
                        sym.ensure_stack(target, offset, len);
                        let _ = self.runtime_stack_mut(target, runtime);
                    }
                    let stack = sym.stack_mut(target).expect("missing target stack");
                    stack.push(value);
                    self.runtime_stack_mut(target, runtime).push(concrete);
                }
            }
            BC_MOVE_I => {
                let (dst, src) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (value, concrete) = self.read_int_reg(src);
                self.set_int_reg(dst, Some(value), Some(concrete));
            }
            BC_CALL_INT
            | BC_CALL_PURE_INT
            | BC_CALL_MAY_FORCE_INT
            | BC_CALL_RELEASE_GIL_INT
            | BC_CALL_LOOPINVARIANT_INT
            | BC_CALL_ASSEMBLER_INT => {
                let (opcode, fn_ptr_idx, dst, arg_regs) = {
                    let frame = self.frames.current_mut();
                    let opcode = bytecode;
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let dst = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    (opcode, fn_ptr_idx, dst, arg_regs)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                if opcode == BC_CALL_ASSEMBLER_INT {
                    let target = self.frames.current_mut().jitcode.assembler_targets[fn_ptr_idx];
                    let token = JitCellToken::new(target.token_number);
                    let traced = ctx.call_assembler_int_typed(&token, &args, &arg_types);
                    let concrete = call_int_function(target.concrete_ptr, &concrete_args);
                    self.set_int_reg(dst, Some(traced), Some(concrete));
                } else {
                    let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                    let trace_ptr = if target.trace_ptr.is_null() {
                        target.concrete_ptr
                    } else {
                        target.trace_ptr
                    };
                    let concrete_ptr = if target.concrete_ptr.is_null() {
                        trace_ptr
                    } else {
                        target.concrete_ptr
                    };
                    let active_vable = if opcode == BC_CALL_MAY_FORCE_INT {
                        self.prepare_standard_virtualizable_before_residual_call(ctx)
                    } else {
                        None
                    };
                    let traced = match opcode {
                        BC_CALL_INT => ctx.call_int_typed(trace_ptr, &args, &arg_types),
                        BC_CALL_PURE_INT => {
                            ctx.call_elidable_int_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_MAY_FORCE_INT => {
                            ctx.call_may_force_int_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_RELEASE_GIL_INT => {
                            ctx.call_release_gil_int_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_LOOPINVARIANT_INT => {
                            ctx.call_loopinvariant_int_typed(trace_ptr, &args, &arg_types)
                        }
                        _ => unreachable!(),
                    };
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    if opcode == BC_CALL_MAY_FORCE_INT {
                        if Self::finish_standard_virtualizable_after_residual_call(active_vable) {
                            return TraceAction::Abort;
                        }
                        ctx.guard_not_forced(sym.total_slots());
                    }
                    self.set_int_reg(dst, Some(traced), Some(concrete));
                }
            }
            // ── Ref-typed bytecodes ────────────────────────────────
            BC_LOAD_CONST_R => {
                let (dst, value) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let const_idx = frame.next_u16() as usize;
                    let value = *frame
                        .jitcode
                        .constants_i
                        .get(const_idx)
                        .expect("jitcode const index out of bounds");
                    (dst, value)
                };
                self.set_ref_reg(dst, Some(ctx.const_int(value)), Some(value));
            }
            BC_POP_R => {
                let dst = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                let symbolic = {
                    let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                    stack.pop()
                };
                let concrete = self.runtime_stack_mut(selected, runtime).pop();
                self.set_ref_reg(dst, symbolic, concrete);
            }
            BC_PUSH_R => {
                let src = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                let (value, concrete) = self.read_ref_reg(src);
                let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                stack.push(value);
                self.runtime_stack_mut(selected, runtime).push(concrete);
            }
            BC_MOVE_R => {
                let (dst, src) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (value, concrete) = self.read_ref_reg(src);
                self.set_ref_reg(dst, Some(value), Some(concrete));
            }
            BC_CALL_REF
            | BC_CALL_PURE_REF
            | BC_CALL_MAY_FORCE_REF
            | BC_CALL_RELEASE_GIL_REF
            | BC_CALL_LOOPINVARIANT_REF
            | BC_CALL_ASSEMBLER_REF => {
                let (opcode, fn_ptr_idx, dst, arg_regs) = {
                    let frame = self.frames.current_mut();
                    let opcode = bytecode;
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let dst = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    (opcode, fn_ptr_idx, dst, arg_regs)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                if opcode == BC_CALL_ASSEMBLER_REF {
                    let target = self.frames.current_mut().jitcode.assembler_targets[fn_ptr_idx];
                    let token = JitCellToken::new(target.token_number);
                    let traced = ctx.call_assembler_ref_typed(&token, &args, &arg_types);
                    let concrete = call_int_function(target.concrete_ptr, &concrete_args);
                    self.set_ref_reg(dst, Some(traced), Some(concrete));
                } else {
                    let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                    let trace_ptr = if target.trace_ptr.is_null() {
                        target.concrete_ptr
                    } else {
                        target.trace_ptr
                    };
                    let concrete_ptr = if target.concrete_ptr.is_null() {
                        trace_ptr
                    } else {
                        target.concrete_ptr
                    };
                    let active_vable = if opcode == BC_CALL_MAY_FORCE_REF {
                        self.prepare_standard_virtualizable_before_residual_call(ctx)
                    } else {
                        None
                    };
                    let traced = match opcode {
                        BC_CALL_REF => ctx.call_ref_typed(trace_ptr, &args, &arg_types),
                        BC_CALL_PURE_REF => {
                            ctx.call_elidable_ref_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_MAY_FORCE_REF => {
                            ctx.call_may_force_ref_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_RELEASE_GIL_REF => {
                            ctx.call_release_gil_ref_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_LOOPINVARIANT_REF => {
                            ctx.call_loopinvariant_ref_typed(trace_ptr, &args, &arg_types)
                        }
                        _ => unreachable!(),
                    };
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    if opcode == BC_CALL_MAY_FORCE_REF {
                        if Self::finish_standard_virtualizable_after_residual_call(active_vable) {
                            return TraceAction::Abort;
                        }
                        ctx.guard_not_forced(sym.total_slots());
                    }
                    self.set_ref_reg(dst, Some(traced), Some(concrete));
                }
            }
            // ── Float-typed bytecodes ───────────────────────────────
            BC_LOAD_CONST_F => {
                let (dst, value) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let const_idx = frame.next_u16() as usize;
                    let value = *frame
                        .jitcode
                        .constants_i
                        .get(const_idx)
                        .expect("jitcode const index out of bounds");
                    (dst, value)
                };
                self.set_float_reg(dst, Some(ctx.const_int(value)), Some(value));
            }
            BC_POP_F => {
                let dst = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                let symbolic = {
                    let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                    stack.pop()
                };
                let concrete = self.runtime_stack_mut(selected, runtime).pop();
                self.set_float_reg(dst, symbolic, concrete);
            }
            BC_PUSH_F => {
                let src = self.frames.current_mut().next_u16() as usize;
                let selected = sym.current_selected();
                let (value, concrete) = self.read_float_reg(src);
                let stack = sym.stack_mut(selected).expect("missing symbolic stack");
                stack.push(value);
                self.runtime_stack_mut(selected, runtime).push(concrete);
            }
            BC_MOVE_F => {
                let (dst, src) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (value, concrete) = self.read_float_reg(src);
                self.set_float_reg(dst, Some(value), Some(concrete));
            }
            BC_CALL_FLOAT
            | BC_CALL_PURE_FLOAT
            | BC_CALL_MAY_FORCE_FLOAT
            | BC_CALL_RELEASE_GIL_FLOAT
            | BC_CALL_LOOPINVARIANT_FLOAT
            | BC_CALL_ASSEMBLER_FLOAT => {
                let (opcode, fn_ptr_idx, dst, arg_regs) = {
                    let frame = self.frames.current_mut();
                    let opcode = bytecode;
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let dst = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    (opcode, fn_ptr_idx, dst, arg_regs)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                if opcode == BC_CALL_ASSEMBLER_FLOAT {
                    let target = self.frames.current_mut().jitcode.assembler_targets[fn_ptr_idx];
                    let token = JitCellToken::new(target.token_number);
                    let traced = ctx.call_assembler_float_typed(&token, &args, &arg_types);
                    let concrete = call_int_function(target.concrete_ptr, &concrete_args);
                    self.set_float_reg(dst, Some(traced), Some(concrete));
                } else {
                    let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                    let trace_ptr = if target.trace_ptr.is_null() {
                        target.concrete_ptr
                    } else {
                        target.trace_ptr
                    };
                    let concrete_ptr = if target.concrete_ptr.is_null() {
                        trace_ptr
                    } else {
                        target.concrete_ptr
                    };
                    let active_vable = if opcode == BC_CALL_MAY_FORCE_FLOAT {
                        self.prepare_standard_virtualizable_before_residual_call(ctx)
                    } else {
                        None
                    };
                    let traced = match opcode {
                        BC_CALL_FLOAT => ctx.call_float_typed(trace_ptr, &args, &arg_types),
                        BC_CALL_PURE_FLOAT => {
                            ctx.call_elidable_float_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_MAY_FORCE_FLOAT => {
                            ctx.call_may_force_float_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_RELEASE_GIL_FLOAT => {
                            ctx.call_release_gil_float_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_LOOPINVARIANT_FLOAT => {
                            ctx.call_loopinvariant_float_typed(trace_ptr, &args, &arg_types)
                        }
                        _ => unreachable!(),
                    };
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    if opcode == BC_CALL_MAY_FORCE_FLOAT {
                        if Self::finish_standard_virtualizable_after_residual_call(active_vable) {
                            return TraceAction::Abort;
                        }
                        ctx.guard_not_forced(sym.total_slots());
                    }
                    self.set_float_reg(dst, Some(traced), Some(concrete));
                }
            }
            BC_RECORD_BINOP_F => {
                let (dst, lhs_idx, rhs_idx, opcode) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let opcode_idx = frame.next_u16() as usize;
                    let lhs_idx = frame.next_u16() as usize;
                    let rhs_idx = frame.next_u16() as usize;
                    let opcode = *frame
                        .jitcode
                        .opcodes
                        .get(opcode_idx)
                        .expect("jitcode opcode index out of bounds");
                    (dst, lhs_idx, rhs_idx, opcode)
                };
                let (lhs, lhs_value) = self.read_float_reg(lhs_idx);
                let (rhs, rhs_value) = self.read_float_reg(rhs_idx);
                let value = eval_binop_f(opcode, lhs_value, rhs_value);
                self.set_float_reg(dst, Some(ctx.record_op(opcode, &[lhs, rhs])), Some(value));
            }
            BC_RECORD_UNARY_F => {
                let (dst, src_idx, opcode) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let opcode_idx = frame.next_u16() as usize;
                    let src_idx = frame.next_u16() as usize;
                    let opcode = *frame
                        .jitcode
                        .opcodes
                        .get(opcode_idx)
                        .expect("jitcode opcode index out of bounds");
                    (dst, src_idx, opcode)
                };
                let (src, src_value) = self.read_float_reg(src_idx);
                let value = eval_unary_f(opcode, src_value);
                self.set_float_reg(dst, Some(ctx.record_op(opcode, &[src])), Some(value));
            }
            BC_ABORT => return TraceAction::Abort,
            BC_ABORT_PERMANENT => return TraceAction::AbortPermanent,
            other => panic!("unknown jitcode bytecode {other}"),
        }

        TraceAction::Continue
    }

    fn runtime_stack_mut(&mut self, selected: usize, runtime: &R) -> &mut Vec<i64> {
        self.runtime_stacks.entry(selected).or_insert_with(|| {
            let len = runtime.stack_len(selected);
            (0..len)
                .map(|index| runtime.stack_peek(selected, index))
                .collect()
        })
    }

    fn set_int_reg(&mut self, reg: usize, opref: Option<OpRef>, value: Option<i64>) {
        let frame = self.frames.current_mut();
        frame.int_regs[reg] = opref;
        frame.int_values[reg] = value;
    }

    fn read_int_reg(&mut self, reg: usize) -> (OpRef, i64) {
        let frame = self.frames.current_mut();
        (
            frame.int_regs[reg].expect("jitcode register was uninitialized"),
            frame.int_values[reg].expect("jitcode concrete register was uninitialized"),
        )
    }

    fn set_ref_reg(&mut self, reg: usize, opref: Option<OpRef>, value: Option<i64>) {
        let frame = self.frames.current_mut();
        frame.ref_regs[reg] = opref;
        frame.ref_values[reg] = value;
    }

    fn read_ref_reg(&mut self, reg: usize) -> (OpRef, i64) {
        let frame = self.frames.current_mut();
        (
            frame.ref_regs[reg].expect("jitcode ref register was uninitialized"),
            frame.ref_values[reg].expect("jitcode concrete ref register was uninitialized"),
        )
    }

    fn set_float_reg(&mut self, reg: usize, opref: Option<OpRef>, value: Option<i64>) {
        let frame = self.frames.current_mut();
        frame.float_regs[reg] = opref;
        frame.float_values[reg] = value;
    }

    fn read_float_reg(&mut self, reg: usize) -> (OpRef, i64) {
        let frame = self.frames.current_mut();
        (
            frame.float_regs[reg].expect("jitcode float register was uninitialized"),
            frame.float_values[reg].expect("jitcode concrete float register was uninitialized"),
        )
    }

    fn read_call_arg(&mut self, arg: JitCallArg) -> (OpRef, i64, majit_ir::Type) {
        match arg.kind {
            JitArgKind::Int => {
                let (opref, value) = self.read_int_reg(arg.reg as usize);
                (opref, value, majit_ir::Type::Int)
            }
            JitArgKind::Ref => {
                let (opref, value) = self.read_ref_reg(arg.reg as usize);
                (opref, value, majit_ir::Type::Ref)
            }
            JitArgKind::Float => {
                let (opref, value) = self.read_float_reg(arg.reg as usize);
                (opref, value, majit_ir::Type::Float)
            }
        }
    }
}

#[derive(Default)]
pub struct JitCodeBuilder {
    code: Vec<u8>,
    num_regs_i: u16,
    num_regs_r: u16,
    num_regs_f: u16,
    constants_i: Vec<i64>,
    opcodes: Vec<OpCode>,
    labels: Vec<Option<usize>>,
    patches: Vec<(usize, usize)>,
    sub_jitcodes: Vec<JitCode>,
    fn_ptrs: Vec<JitCallTarget>,
    assembler_targets: Vec<JitCallAssemblerTarget>,
}

impl JitCodeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_const_i(&mut self, value: i64) -> u16 {
        if let Some(index) = self
            .constants_i
            .iter()
            .position(|&existing| existing == value)
        {
            return index as u16;
        }
        let index = self.constants_i.len() as u16;
        self.constants_i.push(value);
        index
    }

    pub fn load_const_i_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_i(value);
        self.load_const_i(dst, const_idx);
    }

    pub fn load_const_i(&mut self, dst: u16, const_idx: u16) {
        self.touch_reg(dst);
        self.push_u8(BC_LOAD_CONST_I);
        self.push_u16(dst);
        self.push_u16(const_idx);
    }

    pub fn pop_i(&mut self, dst: u16) {
        self.touch_reg(dst);
        self.push_u8(BC_POP_I);
        self.push_u16(dst);
    }

    pub fn peek_i(&mut self, dst: u16) {
        self.touch_reg(dst);
        self.push_u8(BC_PEEK_I);
        self.push_u16(dst);
    }

    pub fn push_i(&mut self, src: u16) {
        self.touch_reg(src);
        self.push_u8(BC_PUSH_I);
        self.push_u16(src);
    }

    pub fn pop_discard(&mut self) {
        self.push_u8(BC_POP_DISCARD);
    }

    pub fn dup_stack(&mut self) {
        self.push_u8(BC_DUP_STACK);
    }

    pub fn swap_stack(&mut self) {
        self.push_u8(BC_SWAP_STACK);
    }

    /// Copy the element at `index` (from bottom, 0-based) to top of stack.
    // ── State field access (register/tape machines) ──

    /// Load a scalar state field value into an int register.
    pub fn load_state_field(&mut self, field_idx: u16, dest: u16) {
        self.touch_reg(dest);
        self.push_u8(BC_LOAD_STATE_FIELD);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    /// Store an int register value into a scalar state field.
    pub fn store_state_field(&mut self, field_idx: u16, src: u16) {
        self.touch_reg(src);
        self.push_u8(BC_STORE_STATE_FIELD);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    /// Load an array state field element into an int register.
    /// The element index comes from another int register.
    pub fn load_state_array(&mut self, array_idx: u16, index_reg: u16, dest: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        self.push_u8(BC_LOAD_STATE_ARRAY);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    /// Store an int register value into an array state field element.
    /// The element index comes from another int register.
    pub fn store_state_array(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.push_u8(BC_STORE_STATE_ARRAY);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    // ── First-class virtualizable access (RPython getfield_vable_*) ──

    pub fn vable_getfield_int(&mut self, dest: u16, field_idx: u16) {
        self.touch_reg(dest);
        self.push_u8(BC_GETFIELD_VABLE_I);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_getfield_ref(&mut self, dest: u16, field_idx: u16) {
        self.touch_ref_reg(dest);
        self.push_u8(BC_GETFIELD_VABLE_R);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_getfield_float(&mut self, dest: u16, field_idx: u16) {
        self.touch_float_reg(dest);
        self.push_u8(BC_GETFIELD_VABLE_F);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_setfield_int(&mut self, field_idx: u16, src: u16) {
        self.touch_reg(src);
        self.push_u8(BC_SETFIELD_VABLE_I);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_setfield_ref(&mut self, field_idx: u16, src: u16) {
        self.touch_ref_reg(src);
        self.push_u8(BC_SETFIELD_VABLE_R);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_setfield_float(&mut self, field_idx: u16, src: u16) {
        self.touch_float_reg(src);
        self.push_u8(BC_SETFIELD_VABLE_F);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_getarrayitem_int(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_reg(dest);
        self.touch_reg(index_reg);
        self.push_u8(BC_GETARRAYITEM_VABLE_I);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_getarrayitem_ref(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_ref_reg(dest);
        self.touch_reg(index_reg);
        self.push_u8(BC_GETARRAYITEM_VABLE_R);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_getarrayitem_float(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_float_reg(dest);
        self.touch_reg(index_reg);
        self.push_u8(BC_GETARRAYITEM_VABLE_F);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_setarrayitem_int(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.push_u8(BC_SETARRAYITEM_VABLE_I);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn vable_setarrayitem_ref(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_ref_reg(src);
        self.push_u8(BC_SETARRAYITEM_VABLE_R);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn vable_setarrayitem_float(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_float_reg(src);
        self.push_u8(BC_SETARRAYITEM_VABLE_F);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn vable_arraylen(&mut self, dest: u16, array_idx: u16) {
        self.touch_reg(dest);
        self.push_u8(BC_ARRAYLEN_VABLE);
        self.push_u16(array_idx);
        self.push_u16(dest);
    }

    pub fn vable_force(&mut self) {
        self.push_u8(BC_HINT_FORCE_VIRTUALIZABLE);
    }

    /// Load from a virtualizable state array: emit GETARRAYITEM_RAW_I.
    /// The array stays on heap; only ptr+len are tracked as inputargs.
    pub fn load_state_varray(&mut self, array_idx: u16, index_reg: u16, dest: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        self.push_u8(BC_LOAD_STATE_VARRAY);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    /// Store to a virtualizable state array: emit SETARRAYITEM_RAW.
    /// The array stays on heap; only ptr+len are tracked as inputargs.
    pub fn store_state_varray(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.push_u8(BC_STORE_STATE_VARRAY);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn copy_from_bottom(&mut self, idx_reg: u16) {
        self.touch_reg(idx_reg);
        self.push_u8(BC_COPY_FROM_BOTTOM);
        self.push_u16(idx_reg);
    }

    /// Pop top of stack and store at `index` (from bottom, 0-based).
    pub fn store_down(&mut self, idx_reg: u16) {
        self.touch_reg(idx_reg);
        self.push_u8(BC_STORE_DOWN);
        self.push_u16(idx_reg);
    }

    pub fn record_binop_i(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let opcode_idx = self.intern_opcode(opcode);
        self.touch_reg(dst);
        self.touch_reg(lhs);
        self.touch_reg(rhs);
        self.push_u8(BC_RECORD_BINOP_I);
        self.push_u16(dst);
        self.push_u16(opcode_idx);
        self.push_u16(lhs);
        self.push_u16(rhs);
    }

    pub fn record_unary_i(&mut self, dst: u16, opcode: OpCode, src: u16) {
        let opcode_idx = self.intern_opcode(opcode);
        self.touch_reg(dst);
        self.touch_reg(src);
        self.push_u8(BC_RECORD_UNARY_I);
        self.push_u16(dst);
        self.push_u16(opcode_idx);
        self.push_u16(src);
    }

    pub fn require_stack(&mut self, required: u16) {
        self.push_u8(BC_REQUIRE_STACK);
        self.push_u16(required);
    }

    pub fn branch_zero(&mut self) {
        self.push_u8(BC_BRANCH_ZERO);
    }

    pub fn new_label(&mut self) -> u16 {
        let label = self.labels.len() as u16;
        self.labels.push(None);
        label
    }

    pub fn mark_label(&mut self, label: u16) {
        let slot = self
            .labels
            .get_mut(label as usize)
            .expect("jitcode label out of bounds");
        *slot = Some(self.code.len());
    }

    pub fn branch_reg_zero(&mut self, reg: u16, label: u16) {
        self.touch_reg(reg);
        self.push_u8(BC_BRANCH_REG_ZERO);
        self.push_u16(reg);
        self.push_label_ref(label);
    }

    pub fn jump(&mut self, label: u16) {
        self.push_u8(BC_JUMP);
        self.push_label_ref(label);
    }

    pub fn jump_target(&mut self) {
        self.push_u8(BC_JUMP_TARGET);
    }

    pub fn abort(&mut self) {
        self.push_u8(BC_ABORT);
    }

    pub fn abort_permanent(&mut self) {
        self.push_u8(BC_ABORT_PERMANENT);
    }

    pub fn inline_call(&mut self, sub_jitcode_idx: u16) {
        self.inline_call_i(sub_jitcode_idx, &[], None);
    }

    pub fn inline_call_i(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(u16, u16)],
        return_i: Option<(u16, u16)>,
    ) {
        self.inline_call_full(sub_jitcode_idx, args, return_i, None, None);
    }

    pub fn inline_call_r(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(u16, u16)],
        return_r: Option<(u16, u16)>,
    ) {
        self.inline_call_full(sub_jitcode_idx, args, None, return_r, None);
    }

    pub fn inline_call_f(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(u16, u16)],
        return_f: Option<(u16, u16)>,
    ) {
        self.inline_call_full(sub_jitcode_idx, args, None, None, return_f);
    }

    /// Inline call with typed arguments and a typed return slot.
    ///
    /// `args` maps each typed caller register to a callee register,
    /// `return_kind` selects which register file the return is routed through:
    /// 0 = Int, 1 = Ref, 2 = Float.
    pub fn inline_call_with_typed_args(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(JitArgKind, u16, u16)],
        return_slot: Option<(u16, u16)>,
        return_kind: u8,
    ) {
        let (return_i, return_r, return_f) = match return_kind {
            1 => (None, return_slot, None),
            2 => (None, None, return_slot),
            _ => (return_slot, None, None),
        };
        self.inline_call_typed(sub_jitcode_idx, args, return_i, return_r, return_f);
    }

    fn inline_call_full(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(u16, u16)],
        return_i: Option<(u16, u16)>,
        return_r: Option<(u16, u16)>,
        return_f: Option<(u16, u16)>,
    ) {
        // Default: all args are int-typed when using (u16, u16) pairs
        let typed_args: Vec<_> = args
            .iter()
            .map(|&(src, dst)| (JitArgKind::Int, src, dst))
            .collect();
        self.inline_call_typed(sub_jitcode_idx, &typed_args, return_i, return_r, return_f);
    }

    fn inline_call_typed(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(JitArgKind, u16, u16)],
        return_i: Option<(u16, u16)>,
        return_r: Option<(u16, u16)>,
        return_f: Option<(u16, u16)>,
    ) {
        for &(kind, caller_src, _) in args {
            match kind {
                JitArgKind::Int => self.touch_reg(caller_src),
                JitArgKind::Ref => self.touch_ref_reg(caller_src),
                JitArgKind::Float => self.touch_float_reg(caller_src),
            }
        }
        if let Some((_, caller_dst)) = return_i {
            self.touch_reg(caller_dst);
        }
        if let Some((_, caller_dst)) = return_r {
            self.touch_ref_reg(caller_dst);
        }
        if let Some((_, caller_dst)) = return_f {
            self.touch_float_reg(caller_dst);
        }
        self.push_u8(BC_INLINE_CALL);
        self.push_u16(sub_jitcode_idx);
        self.push_u16(args.len() as u16);
        for &(kind, caller_src, callee_dst) in args {
            self.push_u8(kind.encode());
            self.push_u16(caller_src);
            self.push_u16(callee_dst);
        }
        self.push_return_slot(return_i);
        self.push_return_slot(return_r);
        self.push_return_slot(return_f);
    }

    fn push_return_slot(&mut self, ret: Option<(u16, u16)>) {
        match ret {
            Some((callee_src, caller_dst)) => {
                self.push_u16(callee_src);
                self.push_u16(caller_dst);
            }
            None => {
                self.push_u16(u16::MAX);
                self.push_u16(u16::MAX);
            }
        }
    }

    pub fn residual_call_void(&mut self, fn_ptr_idx: u16, src_reg: u16) {
        self.residual_call_void_args(fn_ptr_idx, &[src_reg]);
    }

    pub fn residual_call_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.residual_call_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn residual_call_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(BC_RESIDUAL_CALL_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_may_force_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_may_force_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(BC_CALL_MAY_FORCE_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_release_gil_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_release_gil_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(BC_CALL_RELEASE_GIL_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_loopinvariant_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_loopinvariant_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(BC_CALL_LOOPINVARIANT_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_assembler_void_args(&mut self, target_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_void_typed_args(target_idx, &args);
    }

    pub fn call_assembler_void_typed_args(&mut self, target_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_assembler_void_like(BC_CALL_ASSEMBLER_VOID, target_idx, arg_regs);
    }

    pub fn call_may_force_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(BC_CALL_MAY_FORCE_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_int_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_int_like(BC_CALL_RELEASE_GIL_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_loopinvariant_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_int_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_int_like(BC_CALL_LOOPINVARIANT_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_assembler_int(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_int_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_int_typed(&mut self, target_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_assembler_int_like(BC_CALL_ASSEMBLER_INT, target_idx, arg_regs, dst);
    }

    pub fn call_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(BC_CALL_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(BC_CALL_PURE_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn set_selected(&mut self, const_idx: u16) {
        self.push_u8(BC_SET_SELECTED);
        self.push_u16(const_idx);
    }

    pub fn push_to(&mut self, src_reg: u16, target_stack: u16) {
        self.touch_reg(src_reg);
        self.push_u8(BC_PUSH_TO);
        self.push_u16(src_reg);
        self.push_u16(target_stack);
    }

    pub fn move_i(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_reg(src);
        self.push_u8(BC_MOVE_I);
        self.push_u16(dst);
        self.push_u16(src);
    }

    pub fn ensure_i_regs(&mut self, count: u16) {
        self.num_regs_i = max(self.num_regs_i, count);
    }

    pub fn ensure_r_regs(&mut self, count: u16) {
        self.num_regs_r = max(self.num_regs_r, count);
    }

    pub fn ensure_f_regs(&mut self, count: u16) {
        self.num_regs_f = max(self.num_regs_f, count);
    }

    // ── Ref-typed builder methods ─────────────────────────────

    pub fn load_const_r_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_i(value);
        self.load_const_r(dst, const_idx);
    }

    pub fn load_const_r(&mut self, dst: u16, const_idx: u16) {
        self.touch_ref_reg(dst);
        self.push_u8(BC_LOAD_CONST_R);
        self.push_u16(dst);
        self.push_u16(const_idx);
    }

    pub fn pop_r(&mut self, dst: u16) {
        self.touch_ref_reg(dst);
        self.push_u8(BC_POP_R);
        self.push_u16(dst);
    }

    pub fn push_r(&mut self, src: u16) {
        self.touch_ref_reg(src);
        self.push_u8(BC_PUSH_R);
        self.push_u16(src);
    }

    pub fn move_r(&mut self, dst: u16, src: u16) {
        self.touch_ref_reg(dst);
        self.touch_ref_reg(src);
        self.push_u8(BC_MOVE_R);
        self.push_u16(dst);
        self.push_u16(src);
    }

    pub fn call_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(BC_CALL_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(BC_CALL_PURE_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_may_force_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(BC_CALL_MAY_FORCE_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_ref_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_ref_like(BC_CALL_RELEASE_GIL_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_loopinvariant_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_ref_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_ref_like(BC_CALL_LOOPINVARIANT_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_assembler_ref(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_ref_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_ref_typed(&mut self, target_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_assembler_ref_like(BC_CALL_ASSEMBLER_REF, target_idx, arg_regs, dst);
    }

    // ── Float-typed builder methods ───────────────────────────

    pub fn load_const_f_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_i(value);
        self.load_const_f(dst, const_idx);
    }

    pub fn load_const_f(&mut self, dst: u16, const_idx: u16) {
        self.touch_float_reg(dst);
        self.push_u8(BC_LOAD_CONST_F);
        self.push_u16(dst);
        self.push_u16(const_idx);
    }

    pub fn pop_f(&mut self, dst: u16) {
        self.touch_float_reg(dst);
        self.push_u8(BC_POP_F);
        self.push_u16(dst);
    }

    pub fn push_f(&mut self, src: u16) {
        self.touch_float_reg(src);
        self.push_u8(BC_PUSH_F);
        self.push_u16(src);
    }

    pub fn move_f(&mut self, dst: u16, src: u16) {
        self.touch_float_reg(dst);
        self.touch_float_reg(src);
        self.push_u8(BC_MOVE_F);
        self.push_u16(dst);
        self.push_u16(src);
    }

    pub fn call_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_float_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_float_like(BC_CALL_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_float_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_float_like(BC_CALL_PURE_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_may_force_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(BC_CALL_MAY_FORCE_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(BC_CALL_RELEASE_GIL_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_loopinvariant_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(BC_CALL_LOOPINVARIANT_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_assembler_float(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_float_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_float_typed(
        &mut self,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_assembler_float_like(BC_CALL_ASSEMBLER_FLOAT, target_idx, arg_regs, dst);
    }

    pub fn record_binop_f(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let opcode_idx = self.intern_opcode(opcode);
        self.touch_float_reg(dst);
        self.touch_float_reg(lhs);
        self.touch_float_reg(rhs);
        self.push_u8(BC_RECORD_BINOP_F);
        self.push_u16(dst);
        self.push_u16(opcode_idx);
        self.push_u16(lhs);
        self.push_u16(rhs);
    }

    pub fn record_unary_f(&mut self, dst: u16, opcode: OpCode, src: u16) {
        let opcode_idx = self.intern_opcode(opcode);
        self.touch_float_reg(dst);
        self.touch_float_reg(src);
        self.push_u8(BC_RECORD_UNARY_F);
        self.push_u16(dst);
        self.push_u16(opcode_idx);
        self.push_u16(src);
    }

    pub fn add_sub_jitcode(&mut self, jitcode: JitCode) -> u16 {
        let idx = self.sub_jitcodes.len() as u16;
        self.sub_jitcodes.push(jitcode);
        idx
    }

    pub fn add_fn_ptr(&mut self, ptr: *const ()) -> u16 {
        self.add_call_target(ptr, ptr)
    }

    pub fn add_call_target(&mut self, trace_ptr: *const (), concrete_ptr: *const ()) -> u16 {
        let target = JitCallTarget::new(trace_ptr, concrete_ptr);
        if let Some(index) = self.fn_ptrs.iter().position(|existing| *existing == target) {
            return index as u16;
        }
        let idx = self.fn_ptrs.len() as u16;
        self.fn_ptrs.push(target);
        idx
    }

    pub fn add_call_assembler_target_number(
        &mut self,
        token_number: u64,
        concrete_ptr: *const (),
    ) -> u16 {
        let target = JitCallAssemblerTarget::new(token_number, concrete_ptr);
        if let Some(index) = self
            .assembler_targets
            .iter()
            .position(|existing| *existing == target)
        {
            return index as u16;
        }
        let idx = self.assembler_targets.len() as u16;
        self.assembler_targets.push(target);
        idx
    }

    pub fn add_call_assembler_target(
        &mut self,
        target: &JitCellToken,
        concrete_ptr: *const (),
    ) -> u16 {
        self.add_call_assembler_target_number(target.number, concrete_ptr)
    }

    pub fn finish(mut self) -> JitCode {
        self.patch_labels();
        JitCode {
            code: self.code,
            num_regs: [self.num_regs_i, self.num_regs_r, self.num_regs_f],
            constants_i: self.constants_i,
            liveness: Vec::new(),
            opcodes: self.opcodes,
            sub_jitcodes: self.sub_jitcodes,
            fn_ptrs: self.fn_ptrs,
            assembler_targets: self.assembler_targets,
        }
    }

    fn push_u8(&mut self, value: u8) {
        self.code.push(value);
    }

    fn push_u16(&mut self, value: u16) {
        self.code.extend_from_slice(&value.to_le_bytes());
    }

    fn call_int_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_void_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_void_like(&mut self, opcode: u8, target_idx: u16, arg_regs: &[JitCallArg]) {
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(target_idx);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn push_label_ref(&mut self, label: u16) {
        let patch_offset = self.code.len();
        self.push_u16(0);
        self.patches.push((label as usize, patch_offset));
    }

    fn touch_reg(&mut self, reg: u16) {
        self.num_regs_i = max(self.num_regs_i, reg.saturating_add(1));
    }

    fn touch_ref_reg(&mut self, reg: u16) {
        self.num_regs_r = max(self.num_regs_r, reg.saturating_add(1));
    }

    fn touch_float_reg(&mut self, reg: u16) {
        self.num_regs_f = max(self.num_regs_f, reg.saturating_add(1));
    }

    fn call_ref_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_ref_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_int_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_ref_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_ref_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_float_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_float_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_float_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_float_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn touch_call_arg(&mut self, arg: JitCallArg) {
        match arg.kind {
            JitArgKind::Int => self.touch_reg(arg.reg),
            JitArgKind::Ref => self.touch_ref_reg(arg.reg),
            JitArgKind::Float => self.touch_float_reg(arg.reg),
        }
    }

    fn patch_labels(&mut self) {
        for &(label_idx, patch_offset) in &self.patches {
            let target = self.labels[label_idx].expect("jitcode label was never marked") as u16;
            let bytes = target.to_le_bytes();
            self.code[patch_offset] = bytes[0];
            self.code[patch_offset + 1] = bytes[1];
        }
    }

    fn intern_opcode(&mut self, opcode: OpCode) -> u16 {
        if let Some(index) = self.opcodes.iter().position(|&existing| existing == opcode) {
            return index as u16;
        }
        let index = self.opcodes.len() as u16;
        self.opcodes.push(opcode);
        index
    }
}

pub fn trace_jitcode<S, FLen, FPeek, FLabel>(
    ctx: &mut TraceCtx,
    sym: &mut S,
    jitcode: &JitCode,
    pc: usize,
    runtime_stack_len: FLen,
    runtime_stack_peek: FPeek,
    label_at: FLabel,
) -> TraceAction
where
    S: JitCodeSym,
    FLen: Fn(usize) -> usize,
    FPeek: Fn(usize, usize) -> i64,
    FLabel: Fn(usize) -> usize,
{
    let runtime = ClosureRuntime::new(runtime_stack_len, runtime_stack_peek, label_at);
    let root = MIFrame::new(jitcode, pc);
    let mut machine = JitCodeMachine::<S, _>::new(root, &jitcode.sub_jitcodes, &jitcode.fn_ptrs);
    machine.run_to_end(ctx, sym, &runtime)
}

fn read_u8(code: &[u8], cursor: &mut usize) -> u8 {
    let value = *code.get(*cursor).expect("truncated jitcode");
    *cursor += 1;
    value
}

fn read_u16(code: &[u8], cursor: &mut usize) -> u16 {
    let lo = *code.get(*cursor).expect("truncated jitcode");
    let hi = *code.get(*cursor + 1).expect("truncated jitcode");
    *cursor += 2;
    u16::from_le_bytes([lo, hi])
}

fn eval_binop_i(opcode: OpCode, lhs: i64, rhs: i64) -> i64 {
    match opcode {
        OpCode::IntAdd => lhs.wrapping_add(rhs),
        OpCode::IntSub => lhs.wrapping_sub(rhs),
        OpCode::IntMul => lhs.wrapping_mul(rhs),
        OpCode::IntFloorDiv => {
            if rhs == 0 {
                0
            } else {
                lhs.wrapping_div(rhs)
            }
        }
        OpCode::IntMod => {
            if rhs == 0 {
                0
            } else {
                lhs.wrapping_rem(rhs)
            }
        }
        OpCode::IntAnd => lhs & rhs,
        OpCode::IntOr => lhs | rhs,
        OpCode::IntXor => lhs ^ rhs,
        OpCode::IntLshift => lhs.wrapping_shl(rhs as u32),
        OpCode::IntRshift => lhs.wrapping_shr(rhs as u32),
        OpCode::IntEq => i64::from(lhs == rhs),
        OpCode::IntNe => i64::from(lhs != rhs),
        OpCode::IntLt => i64::from(lhs < rhs),
        OpCode::IntLe => i64::from(lhs <= rhs),
        OpCode::IntGt => i64::from(lhs > rhs),
        OpCode::IntGe => i64::from(lhs >= rhs),
        other => panic!("unsupported jitcode integer binop {other:?}"),
    }
}

/// Evaluate an overflow-checked binop. Returns `None` on overflow.
fn eval_binop_ovf(opcode: OpCode, lhs: i64, rhs: i64) -> Option<i64> {
    match opcode {
        OpCode::IntAddOvf => lhs.checked_add(rhs),
        OpCode::IntSubOvf => lhs.checked_sub(rhs),
        OpCode::IntMulOvf => lhs.checked_mul(rhs),
        other => panic!("unsupported jitcode overflow binop {other:?}"),
    }
}

fn eval_unary_i(opcode: OpCode, value: i64) -> i64 {
    match opcode {
        OpCode::IntNeg => value.wrapping_neg(),
        other => panic!("unsupported jitcode integer unary op {other:?}"),
    }
}

/// Evaluate a float binary operation. Values are stored as i64 (bit-cast).
fn eval_binop_f(opcode: OpCode, lhs: i64, rhs: i64) -> i64 {
    let a = f64::from_bits(lhs as u64);
    let b = f64::from_bits(rhs as u64);
    let result = match opcode {
        OpCode::FloatAdd => a + b,
        OpCode::FloatSub => a - b,
        OpCode::FloatMul => a * b,
        OpCode::FloatTrueDiv => a / b,
        OpCode::FloatFloorDiv => (a / b).floor(),
        OpCode::FloatMod => a % b,
        other => panic!("unsupported jitcode float binop {other:?}"),
    };
    f64::to_bits(result) as i64
}

/// Evaluate a float unary operation.
fn eval_unary_f(opcode: OpCode, value: i64) -> i64 {
    let a = f64::from_bits(value as u64);
    let result = match opcode {
        OpCode::FloatNeg => -a,
        OpCode::FloatAbs => a.abs(),
        other => panic!("unsupported jitcode float unary op {other:?}"),
    };
    f64::to_bits(result) as i64
}

fn call_int_function(func_ptr: *const (), args: &[i64]) -> i64 {
    unsafe {
        match args {
            [] => {
                let func: extern "C" fn() -> i64 = std::mem::transmute(func_ptr);
                func()
            }
            [a0] => {
                let func: extern "C" fn(i64) -> i64 = std::mem::transmute(func_ptr);
                func(*a0)
            }
            [a0, a1] => {
                let func: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1)
            }
            [a0, a1, a2] => {
                let func: extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2)
            }
            [a0, a1, a2, a3] => {
                let func: extern "C" fn(i64, i64, i64, i64) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3)
            }
            [a0, a1, a2, a3, a4] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4)
            }
            [a0, a1, a2, a3, a4, a5] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5)
            }
            [a0, a1, a2, a3, a4, a5, a6] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12,
                )
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13,
                )
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14,
                )
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14,
                    *a15,
                )
            }
            _ => panic!(
                "unsupported JitCode int call arity {} (max {})",
                args.len(),
                MAX_HOST_CALL_ARITY
            ),
        }
    }
}

fn call_void_function(func_ptr: *const (), args: &[i64]) {
    unsafe {
        match args {
            [] => {
                let func: extern "C" fn() = std::mem::transmute(func_ptr);
                func()
            }
            [a0] => {
                let func: extern "C" fn(i64) = std::mem::transmute(func_ptr);
                func(*a0)
            }
            [a0, a1] => {
                let func: extern "C" fn(i64, i64) = std::mem::transmute(func_ptr);
                func(*a0, *a1)
            }
            [a0, a1, a2] => {
                let func: extern "C" fn(i64, i64, i64) = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2)
            }
            [a0, a1, a2, a3] => {
                let func: extern "C" fn(i64, i64, i64, i64) = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3)
            }
            [a0, a1, a2, a3, a4] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4)
            }
            [a0, a1, a2, a3, a4, a5] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5)
            }
            [a0, a1, a2, a3, a4, a5, a6] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12,
                )
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13,
                )
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14,
                )
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14,
                    *a15,
                )
            }
            _ => panic!(
                "unsupported JitCode void call arity {} (max {})",
                args.len(),
                MAX_HOST_CALL_ARITY
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtualizable::VirtualizableInfo;
    use majit_ir::Type;

    #[derive(Default)]
    struct DummySym {
        selected: usize,
        selected_value: Option<OpRef>,
    }

    impl JitCodeSym for DummySym {
        fn current_selected(&self) -> usize {
            self.selected
        }

        fn current_selected_value(&self) -> Option<OpRef> {
            self.selected_value
        }

        fn set_current_selected(&mut self, selected: usize) {
            self.selected = selected;
        }

        fn set_current_selected_value(&mut self, selected: usize, value: OpRef) {
            self.selected = selected;
            self.selected_value = Some(value);
        }

        fn stack(&self, _selected: usize) -> Option<&SymbolicStack> {
            None
        }

        fn stack_mut(&mut self, _selected: usize) -> Option<&mut SymbolicStack> {
            None
        }

        fn total_slots(&self) -> usize {
            0
        }

        fn loop_header_pc(&self) -> usize {
            0
        }

        fn ensure_stack(&mut self, _selected: usize, _offset: usize, _len: usize) {}

        fn fail_args(&self) -> Option<Vec<OpRef>> {
            None
        }
    }

    fn make_test_vable_info() -> VirtualizableInfo {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field("stack", Type::Int, 24);
        info
    }

    #[repr(C)]
    struct ResidualVable {
        token: u64,
    }

    extern "C" fn residual_no_force(_vable: i64) {}

    extern "C" fn residual_force(vable: i64) {
        unsafe {
            (*(vable as usize as *mut ResidualVable)).token = 0;
        }
    }

    #[test]
    fn jitcode_vable_reads_use_standard_boxes_without_heap_ops() {
        let mut builder = JitCodeBuilder::new();
        builder.load_const_i_value(0, 0);
        builder.vable_getfield_int(1, 0);
        builder.vable_getarrayitem_int(2, 0, 0);
        builder.vable_arraylen(3, 0);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let info = make_test_vable_info();
        let field_box = ctx.const_int(111);
        let array_box = ctx.const_int(222);
        let vable_ref = ctx.const_int(999);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[field_box, array_box], &[1]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(
            &mut ctx,
            &mut sym,
            &jitcode,
            0,
            |_sel| 0,
            |_sel, _idx| 0,
            |_pc| 0,
        );
        assert!(matches!(action, TraceAction::Continue));

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 0);
    }

    #[test]
    fn jitcode_call_may_force_marks_standard_virtualizable_token_and_guards() {
        let mut obj = ResidualVable { token: 0 };
        let obj_ptr = (&mut obj as *mut ResidualVable) as usize as i64;

        let mut builder = JitCodeBuilder::new();
        builder.load_const_r_value(0, obj_ptr);
        let fn_idx = builder.add_fn_ptr(residual_no_force as *const ());
        builder.call_may_force_void_typed_args(fn_idx, &[JitCallArg::reference(0)]);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let info = VirtualizableInfo::new(0);
        let vable_ref = ctx.const_int(obj_ptr);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[], &[]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(
            &mut ctx,
            &mut sym,
            &jitcode,
            0,
            |_sel| 0,
            |_sel, _idx| 0,
            |_pc| 0,
        );
        assert!(matches!(action, TraceAction::Continue));
        assert_eq!(obj.token, 0, "tracing side must restore TOKEN_NONE");

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 4);
        assert_eq!(recorder.get_op_by_pos(OpRef(0)).unwrap().opcode, OpCode::ForceToken);
        let set_token = recorder.get_op_by_pos(OpRef(1)).unwrap();
        assert_eq!(set_token.opcode, OpCode::SetfieldGc);
        assert_eq!(
            set_token.descr.as_ref().map(|d| d.index()),
            Some(info.token_field_descr().index())
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(2)).unwrap().opcode,
            OpCode::CallMayForceN
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(3)).unwrap().opcode,
            OpCode::GuardNotForced
        );
    }

    #[test]
    fn jitcode_call_may_force_aborts_when_standard_virtualizable_escapes() {
        let mut obj = ResidualVable { token: 0 };
        let obj_ptr = (&mut obj as *mut ResidualVable) as usize as i64;

        let mut builder = JitCodeBuilder::new();
        builder.load_const_r_value(0, obj_ptr);
        let fn_idx = builder.add_fn_ptr(residual_force as *const ());
        builder.call_may_force_void_typed_args(fn_idx, &[JitCallArg::reference(0)]);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let info = VirtualizableInfo::new(0);
        let vable_ref = ctx.const_int(obj_ptr);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[], &[]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(
            &mut ctx,
            &mut sym,
            &jitcode,
            0,
            |_sel| 0,
            |_sel, _idx| 0,
            |_pc| 0,
        );
        assert!(matches!(action, TraceAction::Abort));
        assert_eq!(obj.token, 0, "forced residual call must clear the token");

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 3);
        assert_eq!(recorder.get_op_by_pos(OpRef(0)).unwrap().opcode, OpCode::ForceToken);
        assert_eq!(
            recorder.get_op_by_pos(OpRef(1)).unwrap().opcode,
            OpCode::SetfieldGc
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(2)).unwrap().opcode,
            OpCode::CallMayForceN
        );
    }
}
