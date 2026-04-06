/// assembler.py: Assembler386 — main JIT code generation backend.
///
/// Generates machine code from IR operations via dynasm-rs.
/// This combines assembler.py + rx86.py (instruction encoding) since
/// dynasm-rs handles the encoding layer.
///
/// Key methods:
///   assemble_loop — assembler.py:501
///   assemble_bridge — assembler.py:623
///   _assemble — assembler.py:779 (walk ops + emit code)
///   patch_jump_for_descr — assembler.py:965
///   redirect_call_assembler — assembler.py:1138
use std::collections::HashMap;
use std::sync::Arc;

// On x86_64: use x64 assembler. On aarch64: use aarch64 assembler.
// RPython has separate backend/x86/ and backend/aarch64/ directories.
// We use cfg to select the correct instruction set.
#[cfg(target_arch = "aarch64")]
use dynasmrt::aarch64::Assembler;
#[cfg(target_arch = "x86_64")]
use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, ExecutableBuffer, dynasm};

use majit_backend::BackendError;
use majit_ir::{FailDescr, InputArg, Op, OpCode, OpRef, Type};

use crate::arch::*;
use crate::guard::DynasmFailDescr;

/// Resolved argument: either a frame slot (frame-pointer-relative offset) or a constant.
enum ResolvedArg {
    /// Frame-pointer-relative byte offset: [rbp + offset] on x64, [x29, #offset] on aarch64.
    Slot(i32),
    /// Immediate constant value.
    Const(i64),
}

// ── Abstract condition codes ──
// Architecture-independent CC values used throughout the assembler.
// Converted to arch-specific encoding at emission time.
const CC_O: u8 = 0;
const CC_NO: u8 = 1;
const CC_B: u8 = 2; // unsigned <
const CC_AE: u8 = 3; // unsigned >=
const CC_E: u8 = 4; // ==
const CC_NE: u8 = 5; // !=
const CC_BE: u8 = 6; // unsigned <=
const CC_A: u8 = 7; // unsigned >
const CC_S: u8 = 8;
const CC_NS: u8 = 9;
const CC_L: u8 = 10; // signed <
const CC_GE: u8 = 11; // signed >=
const CC_LE: u8 = 12; // signed <=
const CC_G: u8 = 13; // signed >

/// Invert a condition code.
fn invert_cc(cc: u8) -> u8 {
    match cc {
        CC_O => CC_NO,
        CC_NO => CC_O,
        CC_B => CC_AE,
        CC_AE => CC_B,
        CC_E => CC_NE,
        CC_NE => CC_E,
        CC_BE => CC_A,
        CC_A => CC_BE,
        CC_S => CC_NS,
        CC_NS => CC_S,
        CC_L => CC_GE,
        CC_GE => CC_L,
        CC_LE => CC_G,
        CC_G => CC_LE,
        _ => CC_E, // fallback
    }
}

/// assembler.py:47 Assembler386.
/// In Rust, this is a transient builder — created per compilation,
/// not a long-lived object like RPython's.
pub struct Assembler386 {
    /// The dynasm assembler (rx86.py + codebuf.py combined).
    mc: Assembler,
    /// assembler.py:83 pending_guard_tokens — guards awaiting recovery stubs.
    pending_guard_tokens: Vec<GuardToken>,
    /// Frame depth (in WORD units) for the current trace.
    frame_depth: usize,
    /// Fail descriptors built during assembly.
    fail_descrs: Vec<Arc<DynasmFailDescr>>,
    /// trace_id for this compilation.
    trace_id: u64,
    /// header_pc (green_key) for this compilation.
    header_pc: u64,
    /// Input argument types.
    input_types: Vec<Type>,

    // ── State tracking for code generation ──
    /// Maps OpRef index → jitframe slot index.
    opref_to_slot: HashMap<u32, usize>,
    /// Constants: OpRef index (>= 10000) → i64 value.
    constants: HashMap<u32, i64>,
    /// Next available frame slot index.
    next_slot: usize,
    /// Condition code from the most recent CMP/TEST instruction,
    /// consumed by a following GUARD_TRUE/GUARD_FALSE.
    /// Stores an abstract condition code (CC_* constants).
    guard_success_cc: Option<u8>,
    /// Dynamic label for the LABEL op (back-edge target for JUMP).
    loop_label: Option<DynamicLabel>,
    /// llmodel.py:64-69 self.vtable_offset — typeptr field byte offset.
    /// `None` corresponds to RPython's gcremovetypeptr config.
    vtable_offset: Option<usize>,
}

/// assembler.py GuardToken — represents a pending guard needing
/// a recovery stub to be written after the main loop body.
struct GuardToken {
    /// Offset in machine code where the guard's conditional jump
    /// was emitted. We'll patch this to point to the recovery stub.
    jump_offset: AssemblyOffset,
    /// Dynamic label that the guard's Jcc jumps to — bound in
    /// write_pending_failure_recoveries to the recovery stub.
    fail_label: DynamicLabel,
    /// The fail descriptor for this guard.
    fail_descr: Arc<DynasmFailDescr>,
    /// Fail argument OpRefs for recovery (to save to sequential output slots).
    fail_args: Vec<OpRef>,
}

/// Compiled output from assemble_loop/assemble_bridge.
pub struct CompiledCode {
    /// Executable memory buffer (keeps code alive).
    pub buffer: ExecutableBuffer,
    /// Entry point offset within the buffer.
    pub entry_offset: AssemblyOffset,
    /// Fail descriptors for guards + FINISH ops.
    pub fail_descrs: Vec<Arc<DynasmFailDescr>>,
    /// Input argument types.
    pub input_types: Vec<Type>,
    /// trace_id.
    pub trace_id: u64,
    /// header_pc (green_key).
    pub header_pc: u64,
}

impl Assembler386 {
    /// assembler.py:54 __init__
    pub fn new(
        trace_id: u64,
        header_pc: u64,
        constants: HashMap<u32, i64>,
        vtable_offset: Option<usize>,
    ) -> Self {
        Assembler386 {
            mc: Assembler::new().unwrap(),
            pending_guard_tokens: Vec::new(),
            frame_depth: JITFRAME_FIXED_SIZE,
            fail_descrs: Vec::new(),
            trace_id,
            header_pc,
            input_types: Vec::new(),
            opref_to_slot: HashMap::new(),
            constants,
            next_slot: 0,
            guard_success_cc: None,
            loop_label: None,
            vtable_offset,
        }
    }

    // ----------------------------------------------------------------
    // Helper methods
    // ----------------------------------------------------------------

    /// Frame-pointer-relative byte offset for a given slot index.
    /// jf_ptr[0] = descr, jf_ptr[1..] = frame slots.
    /// slot 0 → [fp + 8], slot 1 → [fp + 16], etc.
    fn slot_offset(slot: usize) -> i32 {
        ((1 + slot) * 8) as i32
    }

    /// Resolve an OpRef to either a frame slot offset or an immediate constant.
    fn resolve_opref(&self, opref: OpRef) -> ResolvedArg {
        if opref.is_constant() {
            let val = self.constants.get(&opref.0).copied().unwrap_or(0);
            ResolvedArg::Const(val)
        } else if let Some(&slot) = self.opref_to_slot.get(&opref.0) {
            ResolvedArg::Slot(Self::slot_offset(slot))
        } else {
            // Unmapped OpRef — treat as 0 (shouldn't happen in well-formed IR).
            ResolvedArg::Const(0)
        }
    }

    /// Allocate a new frame slot for an OpRef and return the slot index.
    fn allocate_slot(&mut self, opref: OpRef) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        if self.next_slot + 1 > self.frame_depth {
            self.frame_depth = self.next_slot + 1;
        }
        self.opref_to_slot.insert(opref.0, slot);
        slot
    }

    // ── AArch64 helper: load a 64-bit immediate into register Xn ──
    #[cfg(target_arch = "aarch64")]
    fn emit_mov_imm64(&mut self, reg: u32, val: i64) {
        let v = val as u64;
        let r = reg as u8;
        dynasm!(self.mc ; .arch aarch64
            ; movz X(r), (v & 0xFFFF) as u32
            ; movk X(r), ((v >> 16) & 0xFFFF) as u32, lsl 16
            ; movk X(r), ((v >> 32) & 0xFFFF) as u32, lsl 32
            ; movk X(r), ((v >> 48) & 0xFFFF) as u32, lsl 48
        );
    }

    /// Emit: load the value of `opref` into RAX (x64) / X0 (aarch64).
    fn load_arg_to_rax(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, [rbp + offset]
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x0, [x29, offset as u32]
                );
            }
            ResolvedArg::Const(val) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, QWORD val as i64
                );
                #[cfg(target_arch = "aarch64")]
                self.emit_mov_imm64(0, val);
            }
        }
    }

    /// Emit: load the value of `opref` into RCX (x64) / X1 (aarch64).
    fn load_arg_to_rcx(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rcx, [rbp + offset]
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x1, [x29, offset as u32]
                );
            }
            ResolvedArg::Const(val) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rcx, QWORD val as i64
                );
                #[cfg(target_arch = "aarch64")]
                self.emit_mov_imm64(1, val);
            }
        }
    }

    /// Emit: store RAX/X0 to the frame slot for `result_opref`.
    /// Allocates a new slot if needed.
    fn store_rax_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; mov [rbp + offset], rax
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x29, offset as u32]
        );
    }

    // ----------------------------------------------------------------
    // assembler.py:543 _call_header — function prologue
    // ----------------------------------------------------------------

    /// Emit the function prologue.
    /// x64: System V AMD64 ABI — first arg (jf_ptr) in RDI.
    /// aarch64: AAPCS64 — first arg (jf_ptr) in X0.
    fn _call_header(&mut self, inputargs: &[InputArg]) {
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; push rbp
            ; mov rbp, rdi
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; stp x29, x30, [sp, #-32]!
            ; stp x19, x20, [sp, #16]   // save callee-saved regs
            ; mov x29, x0
        );

        // Input args are pre-loaded at jf_ptr[1..n+1].
        // Map them to slots 0..n.
        for (i, _ia) in inputargs.iter().enumerate() {
            // The caller already placed the value at jf_ptr[1+i].
            // We just record the slot mapping.
            self.opref_to_slot.insert(i as u32, i);
        }
        self.next_slot = inputargs.len();
    }

    // ----------------------------------------------------------------
    // assembler.py:2153 _call_footer — function epilogue
    // ----------------------------------------------------------------

    /// Emit the function epilogue: return jf_ptr in RAX/X0.
    fn _call_footer(&mut self) {
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, rbp
            ; pop rbp
            ; ret
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x29
            ; ldp x19, x20, [sp, #16]   // restore callee-saved regs
            ; ldp x29, x30, [sp], #32
            ; ret
        );
    }

    // ----------------------------------------------------------------
    // assembler.py:501 assemble_loop
    // ----------------------------------------------------------------

    /// assembler.py:501 assemble_loop: compile a loop trace.
    ///
    /// Returns compiled code with fail descriptors and entry point.
    pub fn assemble_loop(
        mut self,
        inputargs: &[InputArg],
        ops: &[Op],
    ) -> Result<CompiledCode, BackendError> {
        self.input_types = inputargs.iter().map(|ia| ia.tp).collect();

        // assembler.py:537 prepare_loop — set up regalloc
        // For now, simplified: all args in frame slots

        // assembler.py:547 _assemble — generate code for all ops
        let entry = self.mc.offset();
        self._assemble(inputargs, ops)?;

        // assembler.py:553 write_pending_failure_recoveries
        self.write_pending_failure_recoveries();

        // assembler.py:556 materialize_loop — finalize to executable memory
        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs,
            input_types: self.input_types,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
        })
    }

    /// assembler.py:623 assemble_bridge: compile a bridge trace.
    pub fn assemble_bridge(
        mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
    ) -> Result<CompiledCode, BackendError> {
        self.input_types = inputargs.iter().map(|ia| ia.tp).collect();

        // assembler.py:641 prepare_bridge
        let entry = self.mc.offset();
        self._assemble(inputargs, ops)?;
        self.write_pending_failure_recoveries();

        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs,
            input_types: self.input_types,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
        })
    }

    /// assembler.py:779 _assemble — walk operations and emit code.
    fn _assemble(&mut self, inputargs: &[InputArg], ops: &[Op]) -> Result<(), BackendError> {
        // Populate constants from all ops' args.
        for op in ops {
            for &arg in &op.args {
                if arg.is_constant() && !self.constants.contains_key(&arg.0) {
                    // The constant value should be provided externally.
                    // If not in our map yet, it will be looked up as 0.
                }
            }
            if let Some(ref fa) = op.fail_args {
                for &arg in fa.iter() {
                    if arg.is_constant() && !self.constants.contains_key(&arg.0) {
                        // Same: will resolve to 0 if not provided.
                    }
                }
            }
        }

        // Emit prologue.
        self._call_header(inputargs);

        let mut fail_index = 0u32;

        for (op_idx, op) in ops.iter().enumerate() {
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[dynasm] emit[{}]: {:?} pos={:?} args={:?}",
                    op_idx, op.opcode, op.pos, op.args
                );
            }
            if op.opcode.is_guard() {
                self.genop_guard(op, fail_index);
                fail_index += 1;
            } else {
                match op.opcode {
                    // ---- Integer arithmetic ----
                    OpCode::IntAdd => self.genop_int_add(op),
                    OpCode::IntSub => self.genop_int_sub(op),
                    OpCode::IntMul => self.genop_int_mul(op),
                    OpCode::IntAnd => self.genop_int_and(op),
                    OpCode::IntOr => self.genop_int_or(op),
                    OpCode::IntXor => self.genop_int_xor(op),
                    OpCode::IntNeg => self.genop_int_neg(op),
                    OpCode::IntInvert => self.genop_int_invert(op),
                    OpCode::IntLshift => self.genop_int_lshift(op),
                    OpCode::IntRshift => self.genop_int_rshift(op),
                    OpCode::UintRshift => self.genop_uint_rshift(op),

                    // ---- Overflow arithmetic ----
                    OpCode::IntAddOvf => self.genop_int_add_ovf(op),
                    OpCode::IntSubOvf => self.genop_int_sub_ovf(op),
                    OpCode::IntMulOvf => self.genop_int_mul_ovf(op),

                    // ---- Comparisons ----
                    OpCode::IntLt
                    | OpCode::IntLe
                    | OpCode::IntGt
                    | OpCode::IntGe
                    | OpCode::IntEq
                    | OpCode::IntNe
                    | OpCode::UintLt
                    | OpCode::UintLe
                    | OpCode::UintGt
                    | OpCode::UintGe => {
                        self.genop_int_cmp(op);
                    }

                    // ---- Float arithmetic ----
                    OpCode::FloatAdd => self.genop_float_add(op),
                    OpCode::FloatSub => self.genop_float_sub(op),
                    OpCode::FloatMul => self.genop_float_mul(op),
                    OpCode::FloatTrueDiv => self.genop_float_truediv(op),
                    OpCode::FloatNeg => self.genop_float_neg(op),
                    OpCode::CastIntToFloat => self.genop_cast_int_to_float(op),
                    OpCode::CastFloatToInt => self.genop_cast_float_to_int(op),

                    // ---- Memory operations ----
                    OpCode::GetfieldGcI
                    | OpCode::GetfieldGcR
                    | OpCode::GetfieldGcF
                    | OpCode::GetfieldGcPureI
                    | OpCode::GetfieldGcPureR
                    | OpCode::GetfieldGcPureF
                    | OpCode::GetfieldRawI
                    | OpCode::GetfieldRawR
                    | OpCode::GetfieldRawF => {
                        self.genop_getfield(op);
                    }
                    OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                        self.genop_setfield(op);
                    }
                    OpCode::GetarrayitemGcI
                    | OpCode::GetarrayitemGcR
                    | OpCode::GetarrayitemGcF
                    | OpCode::GetarrayitemGcPureI
                    | OpCode::GetarrayitemGcPureR
                    | OpCode::GetarrayitemGcPureF
                    | OpCode::GetarrayitemRawI
                    | OpCode::GetarrayitemRawR
                    | OpCode::GetarrayitemRawF => {
                        self.genop_getarrayitem(op);
                    }
                    OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                        self.genop_setarrayitem(op);
                    }
                    OpCode::ArraylenGc => self.genop_arraylen(op),

                    // ---- Control flow ----
                    OpCode::Jump => self.genop_jump(op),
                    OpCode::Finish => {
                        self.genop_finish(op, fail_index);
                        fail_index += 1;
                    }
                    OpCode::Label => self.genop_label(op),

                    // ---- Calls ----
                    OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN => {
                        self.genop_call(op);
                    }
                    OpCode::CallAssemblerI
                    | OpCode::CallAssemblerR
                    | OpCode::CallAssemblerF
                    | OpCode::CallAssemblerN => {
                        self.genop_call_assembler(op);
                    }

                    // ---- Type conversions ----
                    OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => self.genop_same_as(op),
                    OpCode::IntIsTrue => self.genop_int_is_true(op),
                    OpCode::IntIsZero => self.genop_int_is_zero(op),

                    // ---- Allocation ----
                    OpCode::New => self.genop_new(op),
                    OpCode::NewWithVtable => self.genop_new_with_vtable(op),
                    OpCode::NewArray | OpCode::NewArrayClear => self.genop_new_array(op),

                    // ---- Misc ----
                    OpCode::ForceToken => self.genop_force_token(op),
                    OpCode::Strlen | OpCode::Unicodelen => self.genop_strlen(op),
                    OpCode::Strgetitem | OpCode::Unicodegetitem => self.genop_strgetitem(op),

                    // ---- Overflow checks (guards handle by is_guard) ----

                    // ---- Misc ops with result ----
                    OpCode::IntForceGeZero => {
                        // result = max(arg0, 0)
                        self.load_arg_to_rax(op.arg(0));
                        #[cfg(target_arch = "x86_64")]
                        dynasm!(self.mc ; .arch x64
                            ; test rax, rax
                            ; jge >pos
                            ; xor rax, rax
                            ; pos:
                        );
                        #[cfg(target_arch = "aarch64")]
                        dynasm!(self.mc ; .arch aarch64
                            ; cmp x0, 0
                            ; csel x0, x0, xzr, ge
                        );
                        self.store_rax_to_result(op.pos);
                    }

                    // ---- GC write barrier (discard — no result) ----
                    OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                        // Write barrier — no-op for now (no GC integration)
                    }

                    // ---- Ops that produce no result (discard) ----
                    OpCode::JitDebug | OpCode::DebugMergePoint => {
                        // No-op ops — nothing to emit
                    }

                    _ => {
                        // Unhandled opcode — emit trap for debugging
                        #[cfg(debug_assertions)]
                        eprintln!("[dynasm] WARNING: unhandled opcode {:?}", op.opcode);
                    }
                }
            }
        }
        Ok(())
    }

    // ----------------------------------------------------------------
    // assembler.py:652 write_pending_failure_recoveries
    // ----------------------------------------------------------------

    /// assembler.py:652 — emit recovery stubs for all pending guards.
    /// Each stub saves live values to sequential output frame slots,
    /// assembler.py:835-847 write_pending_failure_recoveries +
    /// assembler.py:1996 generate_quick_failure +
    /// assembler.py:2080 _build_failure_recovery.
    ///
    /// RPython pattern: on guard failure:
    /// 1. _push_all_regs_to_frame — flush all registers to jitframe
    /// 2. Store jf_descr and jf_gcmap
    /// 3. _call_footer — return jf_ptr
    ///
    /// In our frame-slot approach (no regalloc), all values are already
    /// in the jitframe. Recovery only needs to store jf_descr and return.
    fn write_pending_failure_recoveries(&mut self) {
        for guard_token in std::mem::take(&mut self.pending_guard_tokens) {
            // Bind the fail_label — guard's Jcc lands here.
            let fail_label = guard_token.fail_label;
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64 ; =>fail_label);
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64 ; =>fail_label);

            // assembler.py:2087 _push_all_regs_to_frame:
            // All values are already in jitframe slots (no registers to flush).
            // When we add regalloc, this is where we'd flush registers.

            // assembler.py:2106-2107: store jf_descr
            let descr_ptr = Arc::as_ptr(&guard_token.fail_descr) as i64;
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc
                ; .arch x64
                ; mov rax, QWORD descr_ptr
                ; mov [rbp], rax
            );
            #[cfg(target_arch = "aarch64")]
            {
                self.emit_mov_imm64(0, descr_ptr);
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x29]
                );
            }

            // assembler.py:2112 _call_footer — return jf_ptr
            self._call_footer();

            // assembler.py:987 — store adr_jump_offset on the descriptor.
            let recovery_offset = self.mc.offset();
            guard_token
                .fail_descr
                .set_adr_jump_offset(recovery_offset.0);
        }
    }

    // ----------------------------------------------------------------
    // assembler.py:965-987 patch_jump_for_descr
    // ----------------------------------------------------------------

    /// assembler.py:965 patch_jump_for_descr: redirect a guard's
    /// conditional jump to the bridge at adr_new_target.
    pub fn patch_jump_for_descr(descr: &DynasmFailDescr, adr_new_target: usize) {
        let adr_jump_offset = descr.adr_jump_offset();
        assert!(adr_jump_offset != 0, "guard already patched");

        #[cfg(target_arch = "x86_64")]
        {
            // assembler.py:968 — x86_64: patch rel32 offset
            let offset = adr_new_target as isize - (adr_jump_offset as isize + 4);

            if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
                // assembler.py:976-977: fits in rel32 — patch directly
                let patch_ptr = adr_jump_offset as *mut i32;
                unsafe { patch_ptr.write(offset as i32) };
            } else {
                // assembler.py:982-986: long jump — patch recovery stub
                // with MOV r11, addr; JMP r11 (13 bytes)
                let stub_target = adr_jump_offset as isize
                    + 4
                    + unsafe { *(adr_jump_offset as *const i32) } as isize;
                let stub_ptr = stub_target as *mut u8;
                // MOV r11, imm64 (10 bytes) + JMP r11 (3 bytes)
                unsafe {
                    // REX.W + MOV r11, imm64: 49 BB <imm64>
                    *stub_ptr = 0x49;
                    *stub_ptr.add(1) = 0xBB;
                    (stub_ptr.add(2) as *mut u64).write(adr_new_target as u64);
                    // JMP r11: 41 FF E3
                    *stub_ptr.add(10) = 0x41;
                    *stub_ptr.add(11) = 0xFF;
                    *stub_ptr.add(12) = 0xE3;
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // aarch64: patch B (branch) instruction with new offset.
            // B instruction: bits [25:0] = imm26 (signed, * 4 byte units)
            let offset = adr_new_target as isize - adr_jump_offset as isize;
            let imm26 = (offset >> 2) & 0x03FF_FFFF;
            let patch_ptr = adr_jump_offset as *mut u32;
            unsafe {
                let old_insn = patch_ptr.read();
                // Preserve opcode bits, replace imm26
                let new_insn = (old_insn & 0xFC00_0000) | (imm26 as u32);
                patch_ptr.write(new_insn);
            }
            flush_icache(adr_jump_offset as *const u8, 4);
        }

        // assembler.py:987
        descr.set_adr_jump_offset(0); // "patched"

        #[cfg(target_arch = "aarch64")]
        flush_icache(adr_jump_offset as *const u8, 16);
    }

    /// assembler.py:1138 redirect_call_assembler: patch old loop entry
    /// to JMP to new loop after retrace.
    pub fn redirect_call_assembler(old_addr: *const u8, new_addr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        {
            // assembler.py:1153 JMP imm — overwrite old entry with JMP to new
            let old_ptr = old_addr as *mut u8;
            let offset = new_addr as isize - (old_addr as isize + 5); // JMP rel32 is 5 bytes

            if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
                unsafe {
                    // JMP rel32: E9 <rel32>
                    *old_ptr = 0xE9;
                    (old_ptr.add(1) as *mut i32).write(offset as i32);
                }
            } else {
                unsafe {
                    // MOV r11, imm64 (10 bytes) + JMP r11 (3 bytes) = 13 bytes
                    *old_ptr = 0x49;
                    *old_ptr.add(1) = 0xBB;
                    (old_ptr.add(2) as *mut u64).write(new_addr as u64);
                    *old_ptr.add(10) = 0x41;
                    *old_ptr.add(11) = 0xFF;
                    *old_ptr.add(12) = 0xE3;
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // aarch64: overwrite old entry with B (unconditional branch)
            let offset = new_addr as isize - old_addr as isize;
            let old_ptr = old_addr as *mut u32;
            let imm26 = ((offset >> 2) & 0x03FF_FFFF) as u32;
            // B imm26: 0001_01 + imm26
            let insn = 0x1400_0000 | imm26;
            unsafe { old_ptr.write(insn) };
            flush_icache(old_addr, 4);
        }
    }

    // ----------------------------------------------------------------
    // genop_* — integer arithmetic
    // ----------------------------------------------------------------

    /// INT_ADD: result = arg0 + arg1
    fn genop_int_add(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; add rax, rcx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_SUB: result = arg0 - arg1
    fn genop_int_sub(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; sub rax, rcx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; sub x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_MUL: result = arg0 * arg1
    fn genop_int_mul(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; imul rax, rcx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; mul x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_AND: result = arg0 & arg1
    fn genop_int_and(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; and rax, rcx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; and x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_OR: result = arg0 | arg1
    fn genop_int_or(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; or rax, rcx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; orr x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_XOR: result = arg0 ^ arg1
    fn genop_int_xor(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; xor rax, rcx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; eor x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_NEG: result = -arg0
    fn genop_int_neg(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; neg rax
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; neg x0, x0
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_INVERT: result = ~arg0
    fn genop_int_invert(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; not rax
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; mvn x0, x0
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_LSHIFT: result = arg0 << arg1
    fn genop_int_lshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; shl rax, cl
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; lsl x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_RSHIFT: result = arg0 >> arg1 (arithmetic/signed)
    fn genop_int_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; sar rax, cl
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; asr x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// UINT_RSHIFT: result = arg0 >> arg1 (logical/unsigned)
    fn genop_uint_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; shr rax, cl
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; lsr x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // genop_* — overflow arithmetic (assembler.py:1413-1425)
    // ----------------------------------------------------------------

    /// INT_ADD_OVF: result = arg0 + arg1, sets overflow flag.
    /// The following GUARD_NO_OVERFLOW will check the flag.
    fn genop_int_add_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; adds x0, x0, x1);
        // Overflow flag is set — GUARD_NO_OVERFLOW/GUARD_OVERFLOW will read it.
        self.guard_success_cc = Some(CC_NO); // "no overflow" = success
        self.store_rax_to_result(op.pos);
    }

    /// INT_SUB_OVF: result = arg0 - arg1, sets overflow flag.
    fn genop_int_sub_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; sub rax, rcx);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; subs x0, x0, x1);
        self.guard_success_cc = Some(CC_NO);
        self.store_rax_to_result(op.pos);
    }

    /// INT_MUL_OVF: result = arg0 * arg1, sets overflow flag.
    fn genop_int_mul_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; imul rax, rcx
        );
        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 doesn't have a direct overflow flag from MUL.
            // Use SMULH to get the high 64 bits, compare with sign extension.
            dynasm!(self.mc ; .arch aarch64
                ; mul x2, x0, x1        // x2 = low result
                ; smulh x3, x0, x1      // x3 = high bits
                ; asr x4, x2, 63        // x4 = sign extension of low
                ; cmp x3, x4            // overflow if high != sign_ext(low)
                ; mov x0, x2            // result = low
            );
        }
        self.guard_success_cc = Some(CC_NO);
        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // genop_* — comparisons
    // ----------------------------------------------------------------

    /// Map OpCode to abstract condition code for comparisons.
    fn opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::IntLt => CC_L,
            OpCode::IntLe => CC_LE,
            OpCode::IntGt => CC_G,
            OpCode::IntGe => CC_GE,
            OpCode::IntEq => CC_E,
            OpCode::IntNe => CC_NE,
            OpCode::UintLt => CC_B,
            OpCode::UintLe => CC_BE,
            OpCode::UintGt => CC_A,
            OpCode::UintGe => CC_AE,
            _ => CC_E, // fallback
        }
    }

    /// INT_LT/LE/GT/GE/EQ/NE/UINT_*: CMP arg0, arg1 then store CC.
    /// If the next op is a guard, guard_success_cc is set and consumed.
    /// Otherwise, materialize the boolean result via SETcc/CSET.
    fn genop_int_cmp(&mut self, op: &Op) {
        let cc = Self::opcode_to_cc(op.opcode);

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; cmp rax, rcx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, x1
        );

        // Store the CC for a following guard to consume.
        self.guard_success_cc = Some(cc);

        // Also materialize the boolean result for non-guard consumers.
        if !op.pos.is_none() {
            self.emit_setcc_to_result(cc, op.pos);
        }
    }

    /// Emit SETcc/CSET to materialize a boolean result.
    /// x64: SETcc AL; MOVZX EAX, AL
    /// aarch64: CSET X0, cc
    fn emit_setcc_to_result(&mut self, cc: u8, result_opref: OpRef) {
        #[cfg(target_arch = "x86_64")]
        {
            match cc {
                CC_L => dynasm!(self.mc ; .arch x64 ; setl al),
                CC_LE => dynasm!(self.mc ; .arch x64 ; setle al),
                CC_G => dynasm!(self.mc ; .arch x64 ; setg al),
                CC_GE => dynasm!(self.mc ; .arch x64 ; setge al),
                CC_E => dynasm!(self.mc ; .arch x64 ; sete al),
                CC_NE => dynasm!(self.mc ; .arch x64 ; setne al),
                CC_B => dynasm!(self.mc ; .arch x64 ; setb al),
                CC_BE => dynasm!(self.mc ; .arch x64 ; setbe al),
                CC_A => dynasm!(self.mc ; .arch x64 ; seta al),
                CC_AE => dynasm!(self.mc ; .arch x64 ; setae al),
                CC_O => dynasm!(self.mc ; .arch x64 ; seto al),
                CC_NO => dynasm!(self.mc ; .arch x64 ; setno al),
                _ => dynasm!(self.mc ; .arch x64 ; sete al),
            }
            dynasm!(self.mc
                ; .arch x64
                ; movzx eax, al
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            // CSET Xd, cc — sets Xd to 1 if condition is true, 0 otherwise.
            // Note: CSET Xd, cc is an alias for CSINC Xd, XZR, XZR, invert(cc).
            match cc {
                CC_L => dynasm!(self.mc ; .arch aarch64 ; cset x0, lt),
                CC_LE => dynasm!(self.mc ; .arch aarch64 ; cset x0, le),
                CC_G => dynasm!(self.mc ; .arch aarch64 ; cset x0, gt),
                CC_GE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ge),
                CC_E => dynasm!(self.mc ; .arch aarch64 ; cset x0, eq),
                CC_NE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ne),
                CC_B => dynasm!(self.mc ; .arch aarch64 ; cset x0, lo),
                CC_BE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ls),
                CC_A => dynasm!(self.mc ; .arch aarch64 ; cset x0, hi),
                CC_AE => dynasm!(self.mc ; .arch aarch64 ; cset x0, hs),
                CC_O => dynasm!(self.mc ; .arch aarch64 ; cset x0, vs),
                CC_NO => dynasm!(self.mc ; .arch aarch64 ; cset x0, vc),
                _ => dynasm!(self.mc ; .arch aarch64 ; cset x0, eq),
            }
        }
        self.store_rax_to_result(result_opref);
    }

    /// INT_IS_TRUE: result = (arg0 != 0)
    fn genop_int_is_true(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; test rax, rax
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.is_none() {
            self.emit_setcc_to_result(CC_NE, op.pos);
        }
    }

    /// INT_IS_ZERO: result = (arg0 == 0)
    fn genop_int_is_zero(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; test rax, rax
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_E);
        if !op.pos.is_none() {
            self.emit_setcc_to_result(CC_E, op.pos);
        }
    }

    // ----------------------------------------------------------------
    // genop_* — guards
    // ----------------------------------------------------------------

    /// Emit a conditional jump to a recovery stub for a guard op.
    /// The condition is inverted: we jump to the stub when the guard FAILS.
    /// x86/assembler.py:1880-1891 _cmp_guard_class:
    ///   loc_ptr = locs[0]; loc_classptr = locs[1]
    ///   offset = self.cpu.vtable_offset
    ///   if offset is not None:
    ///       self.mc.CMP(mem(loc_ptr, offset), loc_classptr)
    ///   else:
    ///       assert isinstance(loc_classptr, ImmedLoc)
    ///       classptr = loc_classptr.value
    ///       expected_typeid = gc_ll_descr.
    ///           get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    ///       self._cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
    ///
    /// Inputs are pre-loaded: rax = loc_ptr (object), rcx = loc_classptr.
    /// Sets ZF=1 on equal, ZF=0 on not-equal — caller branches via Jcc.
    /// `expected_classptr_imm` is the immediate value of loc_classptr, used
    /// only by the gcremovetypeptr branch (RPython requires this to be an
    /// `ImmedLoc`).
    fn emit_cmp_guard_class_rax_rcx(&mut self, expected_classptr_imm: Option<i64>) {
        if let Some(off_usize) = self.vtable_offset {
            // x86/assembler.py:1884-1885 vtable_offset path: full classptr CMP.
            #[cfg(target_arch = "x86_64")]
            {
                let off_i32 = off_usize as i32;
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, [rax + off_i32]
                    ; cmp rax, rcx
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                let off_u32 = off_usize as u32;
                dynasm!(self.mc
                    ; .arch aarch64
                    ; ldr x0, [x0, #off_u32]
                    ; cmp x0, x1
                );
            }
        } else {
            // x86/assembler.py:1886-1891 gcremovetypeptr fallback +
            // x86/assembler.py:1893-1901 _cmp_guard_gc_type:
            //   on x86_64 the typeid is a 32-bit half-word at offset 0.
            let classptr = expected_classptr_imm.expect(
                "_cmp_guard_class: gcremovetypeptr requires loc_classptr \
                 to be an immediate (assert isinstance(loc_classptr, \
                 ImmedLoc) in x86/assembler.py:1887)",
            );
            let expected_typeid = Self::lookup_typeid_from_classptr(classptr as usize).expect(
                "GuardClass: vtable_offset is None but the dynasm \
                     backend has no gc_ll_descr.get_typeid_from_classptr_if_\
                     gcremovetypeptr",
            );
            #[cfg(target_arch = "x86_64")]
            {
                let typeid_i32 = expected_typeid as i32;
                dynasm!(self.mc
                    ; .arch x64
                    ; cmp DWORD [rax], typeid_i32
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                let typeid_w = expected_typeid as i32;
                dynasm!(self.mc
                    ; .arch aarch64
                    ; ldr w0, [x0]
                    ; movz w1, #(typeid_w as u32 & 0xffff)
                    ; movk w1, #((typeid_w as u32 >> 16) & 0xffff), lsl #16
                    ; cmp w0, w1
                );
            }
        }
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Default `None` — pyre uses vtable_offset and never reaches the
    /// gcremovetypeptr branch. A future GC layer that disables typeptr
    /// can install a real resolver here.
    fn lookup_typeid_from_classptr(_classptr: usize) -> Option<u32> {
        None
    }

    fn emit_guard_jcc(&mut self, fail_cc: u8) -> DynamicLabel {
        let fail_label = self.mc.new_dynamic_label();
        #[cfg(target_arch = "x86_64")]
        match fail_cc {
            CC_L => dynasm!(self.mc ; .arch x64 ; jl =>fail_label),
            CC_LE => dynasm!(self.mc ; .arch x64 ; jle =>fail_label),
            CC_G => dynasm!(self.mc ; .arch x64 ; jg =>fail_label),
            CC_GE => dynasm!(self.mc ; .arch x64 ; jge =>fail_label),
            CC_E => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
            CC_NE => dynasm!(self.mc ; .arch x64 ; jne =>fail_label),
            CC_B => dynasm!(self.mc ; .arch x64 ; jb =>fail_label),
            CC_BE => dynasm!(self.mc ; .arch x64 ; jbe =>fail_label),
            CC_A => dynasm!(self.mc ; .arch x64 ; ja =>fail_label),
            CC_AE => dynasm!(self.mc ; .arch x64 ; jae =>fail_label),
            CC_O => dynasm!(self.mc ; .arch x64 ; jo =>fail_label),
            CC_NO => dynasm!(self.mc ; .arch x64 ; jno =>fail_label),
            CC_S => dynasm!(self.mc ; .arch x64 ; js =>fail_label),
            CC_NS => dynasm!(self.mc ; .arch x64 ; jns =>fail_label),
            _ => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
        }
        #[cfg(target_arch = "aarch64")]
        match fail_cc {
            CC_L => dynasm!(self.mc ; .arch aarch64 ; b.lt =>fail_label),
            CC_LE => dynasm!(self.mc ; .arch aarch64 ; b.le =>fail_label),
            CC_G => dynasm!(self.mc ; .arch aarch64 ; b.gt =>fail_label),
            CC_GE => dynasm!(self.mc ; .arch aarch64 ; b.ge =>fail_label),
            CC_E => dynasm!(self.mc ; .arch aarch64 ; b.eq =>fail_label),
            CC_NE => dynasm!(self.mc ; .arch aarch64 ; b.ne =>fail_label),
            CC_B => dynasm!(self.mc ; .arch aarch64 ; b.lo =>fail_label),
            CC_BE => dynasm!(self.mc ; .arch aarch64 ; b.ls =>fail_label),
            CC_A => dynasm!(self.mc ; .arch aarch64 ; b.hi =>fail_label),
            CC_AE => dynasm!(self.mc ; .arch aarch64 ; b.hs =>fail_label),
            CC_O => dynasm!(self.mc ; .arch aarch64 ; b.vs =>fail_label),
            CC_NO => dynasm!(self.mc ; .arch aarch64 ; b.vc =>fail_label),
            CC_S => dynasm!(self.mc ; .arch aarch64 ; b.mi =>fail_label),
            CC_NS => dynasm!(self.mc ; .arch aarch64 ; b.pl =>fail_label),
            _ => dynasm!(self.mc ; .arch aarch64 ; b.eq =>fail_label),
        }
        fail_label
    }

    /// Unified guard handler: dispatches to guard-specific code gen.
    fn genop_guard(&mut self, op: &Op, fail_index: u32) {
        let fail_arg_types = op
            .fail_arg_types
            .as_ref()
            .map(|ts| ts.clone())
            .unwrap_or_default();
        let descr = Arc::new(DynasmFailDescr::new(
            fail_index,
            self.trace_id,
            fail_arg_types,
            op.opcode == OpCode::Finish,
        ));

        // regalloc.py:496-501 make_a_counter_per_value for GUARD_VALUE
        if op.opcode == OpCode::GuardValue {
            if let Some(fa) = op.fail_args.as_ref() {
                let arg0 = op.arg(0);
                if let Some(idx) = fa.iter().position(|&r| r == arg0) {
                    let type_tag = match descr.fail_arg_types.get(idx) {
                        Some(Type::Ref) => DynasmFailDescr::TY_REF,
                        Some(Type::Float) => DynasmFailDescr::TY_FLOAT,
                        _ => DynasmFailDescr::TY_INT,
                    };
                    descr.make_a_counter_per_value(idx as u32, type_tag);
                }
            }
        }

        // Emit the guard check and conditional jump.
        let _fail_label = match op.opcode {
            OpCode::GuardTrue | OpCode::VecGuardTrue => {
                // Guard succeeds if the condition is true.
                // Jump to recovery if the condition is FALSE (inverted).
                let cc = self.guard_success_cc.take().unwrap_or(CC_NE);
                let fail_cc = invert_cc(cc);
                self.emit_guard_jcc(fail_cc)
            }
            OpCode::GuardFalse | OpCode::VecGuardFalse => {
                // Guard succeeds if the condition is false.
                // Jump to recovery if the condition is TRUE.
                let cc = self.guard_success_cc.take().unwrap_or(CC_NE);
                self.emit_guard_jcc(cc)
            }
            OpCode::GuardValue => {
                // CMP arg0, arg1; JNE recovery
                self.load_arg_to_rax(op.arg(0));
                self.load_arg_to_rcx(op.arg(1));
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; cmp rax, rcx
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; cmp x0, x1
                );
                self.emit_guard_jcc(CC_NE)
            }
            OpCode::GuardNonnull => {
                // TEST arg0, arg0; JZ recovery
                self.load_arg_to_rax(op.arg(0));
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; test rax, rax
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; cmp x0, 0
                );
                self.emit_guard_jcc(CC_E)
            }
            OpCode::GuardIsnull => {
                // TEST arg0, arg0; JNZ recovery
                self.load_arg_to_rax(op.arg(0));
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; test rax, rax
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; cmp x0, 0
                );
                self.emit_guard_jcc(CC_NE)
            }
            OpCode::GuardNonnullClass => {
                // TEST arg0, arg0; JZ recovery
                // Then _cmp_guard_class via vtable_offset; JNE recovery
                self.load_arg_to_rax(op.arg(0));
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; test rax, rax
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; cmp x0, 0
                );
                let fail_label = self.emit_guard_jcc(CC_E);
                // x86/assembler.py:1887 assert isinstance(loc_classptr, ImmedLoc)
                let expected_classptr_imm = match self.resolve_opref(op.arg(1)) {
                    ResolvedArg::Const(v) => Some(v),
                    _ => None,
                };
                self.load_arg_to_rax(op.arg(0));
                self.load_arg_to_rcx(op.arg(1));
                self.emit_cmp_guard_class_rax_rcx(expected_classptr_imm);
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64 ; jne =>fail_label);
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64 ; b.ne =>fail_label);
                fail_label
            }
            OpCode::GuardClass => {
                // x86/assembler.py:1880-1891 _cmp_guard_class:
                //   offset = self.cpu.vtable_offset
                //   if offset is not None:
                //       CMP(mem(loc_ptr, offset), loc_classptr)
                //   else: _cmp_guard_gc_type(...)
                let expected_classptr_imm = match self.resolve_opref(op.arg(1)) {
                    ResolvedArg::Const(v) => Some(v),
                    _ => None,
                };
                self.load_arg_to_rax(op.arg(0));
                self.load_arg_to_rcx(op.arg(1));
                self.emit_cmp_guard_class_rax_rcx(expected_classptr_imm);
                self.emit_guard_jcc(CC_NE)
            }
            OpCode::GuardNoException
            | OpCode::GuardNotInvalidated
            | OpCode::GuardNotForced
            | OpCode::GuardNotForced2 => {
                // These guards check runtime state that isn't represented in
                // the frame-only model yet. Emit a NOP-style guard that always
                // succeeds for now (no conditional jump). The recovery stub is
                // still registered so bridges can be attached.
                let fail_label = self.mc.new_dynamic_label();
                // No jump emitted — guard always passes.
                fail_label
            }
            OpCode::GuardOverflow => {
                // Jump to recovery on overflow.
                self.emit_guard_jcc(CC_O)
            }
            OpCode::GuardNoOverflow => {
                // Jump to recovery on overflow (inverted: guard succeeds if no overflow).
                self.emit_guard_jcc(CC_O)
            }
            _ => {
                // Unknown guard — no-op, register stub anyway.
                let fail_label = self.mc.new_dynamic_label();
                fail_label
            }
        };

        // Collect fail_args for recovery.
        let fail_args = op
            .fail_args
            .as_ref()
            .map(|fa| fa.to_vec())
            .unwrap_or_default();

        let jump_offset = self.mc.offset();
        self.pending_guard_tokens.push(GuardToken {
            jump_offset,
            fail_label: _fail_label,
            fail_descr: descr.clone(),
            fail_args,
        });
        self.fail_descrs.push(descr);
    }

    // ----------------------------------------------------------------
    // genop_* — control flow
    // ----------------------------------------------------------------

    /// LABEL: define the back-edge target for JUMP.
    /// Remap Label's args to canonical slots 0..n so JUMP can write
    /// back to slots 0..n and the body reads from the same slots.
    /// Also emit code to copy values from old slots to new canonical slots.
    fn genop_label(&mut self, op: &Op) {
        // First, emit moves from old slots to new canonical slots.
        // This is needed because the preamble may have allocated
        // values to non-canonical slots (e.g., OpRef(13) at slot 5).
        // The body expects them at canonical slots (slot 1).
        for (i, &arg_ref) in op.args.iter().enumerate() {
            if arg_ref.is_none() || arg_ref.is_constant() {
                continue;
            }
            let new_offset = Self::slot_offset(i);
            if let Some(&old_slot) = self.opref_to_slot.get(&arg_ref.0) {
                let old_offset = Self::slot_offset(old_slot);
                if old_offset != new_offset {
                    // Copy value from old slot to new canonical slot.
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64
                        ; mov rax, [rbp + old_offset]
                        ; mov [rbp + new_offset], rax
                    );
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x0, [x29, old_offset as u32]
                        ; str x0, [x29, new_offset as u32]
                    );
                }
            }
        }
        // Now remap: Label's arg[i] → slot i
        for (i, &arg_ref) in op.args.iter().enumerate() {
            if !arg_ref.is_none() {
                self.opref_to_slot.insert(arg_ref.0, i);
            }
        }
        self.next_slot = self.next_slot.max(op.args.len());

        let label = self.mc.new_dynamic_label();
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; =>label
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; =>label
        );
        self.loop_label = Some(label);
    }

    /// JUMP: unconditional branch to the loop label.
    fn genop_jump(&mut self, op: &Op) {
        // Before jumping, write JUMP args back to their canonical input slots.
        for (i, &arg_ref) in op.args.iter().enumerate() {
            let dst_offset = Self::slot_offset(i);
            match self.resolve_opref(arg_ref) {
                ResolvedArg::Slot(src_offset) => {
                    if src_offset != dst_offset {
                        #[cfg(target_arch = "x86_64")]
                        dynasm!(self.mc
                            ; .arch x64
                            ; mov rax, [rbp + src_offset]
                            ; mov [rbp + dst_offset], rax
                        );
                        #[cfg(target_arch = "aarch64")]
                        dynasm!(self.mc ; .arch aarch64
                            ; ldr x0, [x29, src_offset as u32]
                            ; str x0, [x29, dst_offset as u32]
                        );
                    }
                }
                ResolvedArg::Const(val) => {
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc
                        ; .arch x64
                        ; mov rax, QWORD val as i64
                        ; mov [rbp + dst_offset], rax
                    );
                    #[cfg(target_arch = "aarch64")]
                    {
                        self.emit_mov_imm64(0, val);
                        dynasm!(self.mc ; .arch aarch64
                            ; str x0, [x29, dst_offset as u32]
                        );
                    }
                }
            }
        }

        if let Some(label) = self.loop_label {
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc
                ; .arch x64
                ; jmp =>label
            );
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64
                ; b =>label
            );
        }
    }

    /// FINISH: store result (if any), store descr ptr, return jf_ptr.
    fn genop_finish(&mut self, op: &Op, fail_index: u32) {
        let fail_arg_types = op
            .fail_arg_types
            .as_ref()
            .map(|ts| ts.clone())
            .unwrap_or_default();
        let descr = Arc::new(DynasmFailDescr::new(
            fail_index,
            self.trace_id,
            fail_arg_types,
            true,
        ));

        // If there's a result argument, store it to jf_frame[0].
        if op.num_args() > 0 {
            let arg0 = op.arg(0);
            self.load_arg_to_rax(arg0);
            let slot0_offset = Self::slot_offset(0);
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc
                ; .arch x64
                ; mov [rbp + slot0_offset], rax
            );
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64
                ; str x0, [x29, slot0_offset as u32]
            );
        }

        // Store descr pointer at jf_ptr[0] (jf_descr slot).
        let descr_ptr = Arc::as_ptr(&descr) as i64;
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, QWORD descr_ptr
            ; mov [rbp], rax
        );
        #[cfg(target_arch = "aarch64")]
        {
            self.emit_mov_imm64(0, descr_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; str x0, [x29]
            );
        }

        // Emit epilogue (return jf_ptr).
        self._call_footer();

        self.fail_descrs.push(descr);
    }

    // ----------------------------------------------------------------
    // genop_* — type conversions
    // ----------------------------------------------------------------

    /// SAME_AS: result = arg0 (copy value)
    fn genop_same_as(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // Float helpers
    // ----------------------------------------------------------------

    /// Load a float value from `opref` into XMM0 (x64) / D0 (aarch64).
    /// Float values are stored as bit-cast i64 in frame slots.
    fn load_float_arg_to_d0(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; movsd xmm0, [rbp + offset]
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; ldr d0, [x29, offset as u32]
                );
            }
            ResolvedArg::Const(val) => {
                // Load constant via integer register, then move to float register.
                #[cfg(target_arch = "x86_64")]
                {
                    dynasm!(self.mc
                        ; .arch x64
                        ; mov rax, QWORD val as i64
                        ; movq xmm0, rax
                    );
                }
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(0, val);
                    dynasm!(self.mc ; .arch aarch64
                        ; fmov d0, x0
                    );
                }
            }
        }
    }

    /// Load a float value from `opref` into XMM1 (x64) / D1 (aarch64).
    fn load_float_arg_to_d1(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; movsd xmm1, [rbp + offset]
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; ldr d1, [x29, offset as u32]
                );
            }
            ResolvedArg::Const(val) => {
                #[cfg(target_arch = "x86_64")]
                {
                    dynasm!(self.mc
                        ; .arch x64
                        ; mov rax, QWORD val as i64
                        ; movq xmm1, rax
                    );
                }
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(1, val);
                    dynasm!(self.mc ; .arch aarch64
                        ; fmov d1, x1
                    );
                }
            }
        }
    }

    /// Store XMM0 (x64) / D0 (aarch64) to the frame slot for `result_opref`.
    fn store_d0_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; movsd [rbp + offset], xmm0
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; str d0, [x29, offset as u32]
        );
    }

    // ----------------------------------------------------------------
    // genop_* — float arithmetic
    // x86/assembler.py:1648 genop_float_add etc.
    // aarch64/assembler.py float equivalents
    // ----------------------------------------------------------------

    /// FLOAT_ADD: result = arg0 + arg1
    fn genop_float_add(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; addsd xmm0, xmm1
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fadd d0, d0, d1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_SUB: result = arg0 - arg1
    fn genop_float_sub(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; subsd xmm0, xmm1
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fsub d0, d0, d1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_MUL: result = arg0 * arg1
    fn genop_float_mul(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; mulsd xmm0, xmm1
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fmul d0, d0, d1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_TRUEDIV: result = arg0 / arg1
    fn genop_float_truediv(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; divsd xmm0, xmm1
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fdiv d0, d0, d1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_NEG: result = -arg0
    /// x64: XOR with sign-bit mask (0x8000000000000000).
    /// aarch64: FNEG d0, d0.
    fn genop_float_neg(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        {
            // Load the sign-bit mask (0x8000_0000_0000_0000) into XMM1
            // via integer register, then XOR.
            let sign_mask: i64 = i64::MIN; // 0x8000000000000000
            dynasm!(self.mc
                ; .arch x64
                ; mov rax, QWORD sign_mask
                ; movq xmm1, rax
                ; xorpd xmm0, xmm1
            );
        }
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fneg d0, d0
        );
        self.store_d0_to_result(op.pos);
    }

    /// CAST_INT_TO_FLOAT: result = (f64)arg0
    fn genop_cast_int_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; cvtsi2sd xmm0, rax
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; scvtf d0, x0
        );
        self.store_d0_to_result(op.pos);
    }

    /// CAST_FLOAT_TO_INT: result = (i64)arg0 (truncation)
    fn genop_cast_float_to_int(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; cvttsd2si rax, xmm0
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fcvtzs x0, d0
        );
        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // genop_* — memory operations
    // x86/assembler.py:1747 genop_getfield_gc etc.
    // ----------------------------------------------------------------

    /// Extract the byte offset from an op's FieldDescr.
    /// Returns 0 if no field descriptor is present.
    fn field_offset_from_descr(op: &Op) -> i32 {
        op.descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .map(|fd| fd.offset() as i32)
            .unwrap_or(0)
    }

    /// Extract the field size from an op's FieldDescr.
    /// Returns 8 (WORD) if no field descriptor is present.
    fn field_size_from_descr(op: &Op) -> usize {
        op.descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .map(|fd| fd.field_size())
            .unwrap_or(8)
    }

    /// GETFIELD_GC_*: result = [arg0 + offset]
    /// The offset comes from the op's FieldDescr.
    fn genop_getfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load the object pointer from arg0.
        self.load_arg_to_rax(op.arg(0));

        // Load the field value at [rax + offset] into rax/x0.
        #[cfg(target_arch = "x86_64")]
        match size {
            1 => dynasm!(self.mc
                ; .arch x64
                ; movzx eax, BYTE [rax + offset]
            ),
            2 => dynasm!(self.mc
                ; .arch x64
                ; movzx eax, WORD [rax + offset]
            ),
            4 => dynasm!(self.mc
                ; .arch x64
                ; mov eax, [rax + offset]
            ),
            _ => dynasm!(self.mc
                ; .arch x64
                ; mov rax, [rax + offset]
            ),
        }

        #[cfg(target_arch = "aarch64")]
        match size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; ldrb w0, [x0, offset as u32]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; ldrh w0, [x0, offset as u32]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; ldr w0, [x0, offset as u32]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x0, offset as u32]
            ),
        }

        self.store_rax_to_result(op.pos);
    }

    /// SETFIELD_GC: [arg0 + offset] = arg1
    fn genop_setfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load object pointer into rax/x0 and value into rcx/x1.
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        #[cfg(target_arch = "x86_64")]
        match size {
            1 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax + offset], cl
            ),
            2 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax + offset], cx
            ),
            4 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax + offset], ecx
            ),
            _ => dynasm!(self.mc
                ; .arch x64
                ; mov [rax + offset], rcx
            ),
        }

        #[cfg(target_arch = "aarch64")]
        match size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; strb w1, [x0, offset as u32]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; strh w1, [x0, offset as u32]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; str w1, [x0, offset as u32]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; str x1, [x0, offset as u32]
            ),
        }
    }

    /// GETARRAYITEM_GC_*: result = array[index]
    /// arg0 = array pointer, arg1 = index.
    /// The base_size and item_size come from the op's ArrayDescr.
    fn genop_getarrayitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((8, 8));

        // Load array pointer and index.
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        // Compute address: rax = rax + base_size + rcx * item_size
        #[cfg(target_arch = "x86_64")]
        {
            // rcx = rcx * item_size
            if item_size != 1 {
                dynasm!(self.mc
                    ; .arch x64
                    ; imul rcx, rcx, item_size
                );
            }
            // rax = rax + base_size + rcx
            dynasm!(self.mc
                ; .arch x64
                ; add rax, base_size
                ; add rax, rcx
            );
            match item_size {
                1 => dynasm!(self.mc
                    ; .arch x64
                    ; movzx eax, BYTE [rax]
                ),
                2 => dynasm!(self.mc
                    ; .arch x64
                    ; movzx eax, WORD [rax]
                ),
                4 => dynasm!(self.mc
                    ; .arch x64
                    ; mov eax, [rax]
                ),
                _ => dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, [rax]
                ),
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // x1 = x1 * item_size; x0 = x0 + base_size + x1
            if item_size != 1 {
                self.emit_mov_imm64(2, item_size as i64); // x2 = item_size
                dynasm!(self.mc ; .arch aarch64
                    ; mul x1, x1, x2
                );
            }
            if base_size != 0 {
                dynasm!(self.mc ; .arch aarch64
                    ; add x0, x0, base_size as u32
                );
            }
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, x1
            );
            match item_size {
                1 => dynasm!(self.mc ; .arch aarch64
                    ; ldrb w0, [x0]
                ),
                2 => dynasm!(self.mc ; .arch aarch64
                    ; ldrh w0, [x0]
                ),
                4 => dynasm!(self.mc ; .arch aarch64
                    ; ldr w0, [x0]
                ),
                _ => dynasm!(self.mc ; .arch aarch64
                    ; ldr x0, [x0]
                ),
            }
        }

        self.store_rax_to_result(op.pos);
    }

    /// SETARRAYITEM_GC: array[index] = value
    /// arg0 = array pointer, arg1 = index, arg2 = value.
    fn genop_setarrayitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((8, 8));

        // Load array pointer.
        self.load_arg_to_rax(op.arg(0));
        // Load index.
        self.load_arg_to_rcx(op.arg(1));

        // Compute element address: rax = rax + base_size + rcx * item_size
        #[cfg(target_arch = "x86_64")]
        {
            if item_size != 1 {
                dynasm!(self.mc
                    ; .arch x64
                    ; imul rcx, rcx, item_size
                );
            }
            dynasm!(self.mc
                ; .arch x64
                ; add rax, base_size
                ; add rax, rcx
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            if item_size != 1 {
                self.emit_mov_imm64(2, item_size as i64);
                dynasm!(self.mc ; .arch aarch64
                    ; mul x1, x1, x2
                );
            }
            if base_size != 0 {
                dynasm!(self.mc ; .arch aarch64
                    ; add x0, x0, base_size as u32
                );
            }
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, x1
            );
        }

        // Now load value from arg2 and store it.
        // We need a third register: use rcx/x1 again for the value
        // (the address is in rax/x0).
        // Save rax/x0 (element address) before loading value.
        #[cfg(target_arch = "x86_64")]
        {
            // Push address, load value into rcx, pop address into rax.
            dynasm!(self.mc
                ; .arch x64
                ; push rax
            );
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc
                ; .arch x64
                ; pop rax
            );

            match item_size {
                1 => dynasm!(self.mc
                    ; .arch x64
                    ; mov [rax], cl
                ),
                2 => dynasm!(self.mc
                    ; .arch x64
                    ; mov [rax], cx
                ),
                4 => dynasm!(self.mc
                    ; .arch x64
                    ; mov [rax], ecx
                ),
                _ => dynasm!(self.mc
                    ; .arch x64
                    ; mov [rax], rcx
                ),
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Save x0 (element address) in x2, load value into x1.
            dynasm!(self.mc ; .arch aarch64
                ; mov x2, x0
            );
            self.load_arg_to_rcx(op.arg(2)); // loads into x1
            match item_size {
                1 => dynasm!(self.mc ; .arch aarch64
                    ; strb w1, [x2]
                ),
                2 => dynasm!(self.mc ; .arch aarch64
                    ; strh w1, [x2]
                ),
                4 => dynasm!(self.mc ; .arch aarch64
                    ; str w1, [x2]
                ),
                _ => dynasm!(self.mc ; .arch aarch64
                    ; str x1, [x2]
                ),
            }
        }
    }

    /// ARRAYLEN_GC: result = array.length
    /// The length field location comes from the ArrayDescr's len_descr().
    fn genop_arraylen(&mut self, op: &Op) {
        let len_offset = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .and_then(|ad| ad.len_descr())
            .map(|ld| ld.offset() as i32)
            .unwrap_or(0); // Default: length at offset 0 in array header

        // Load array pointer.
        self.load_arg_to_rax(op.arg(0));

        // Load length from [array + len_offset].
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, [rax + len_offset]
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x0, len_offset as u32]
        );

        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // genop_* — calls
    // x86/assembler.py:2230 _genop_call
    // ----------------------------------------------------------------

    /// CALL_*: invoke a function via the calling convention.
    /// arg0 = function pointer, arg1.. = arguments.
    /// The CallDescr from op.descr provides argument/result types.
    fn genop_call(&mut self, op: &Op) {
        let arg_count = op.num_args();
        // arg0 is the function pointer.
        // arg1..N are the call arguments.

        // Save frame pointer before call (callee may clobber it).
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; push rbp
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; stp x29, x30, [sp, #-16]!
        );

        // Load arguments into ABI registers.
        // x64: rdi, rsi, rdx, rcx, r8, r9
        // aarch64: x0-x7
        for i in 1..arg_count.min(7) {
            let arg = op.arg(i);
            match self.resolve_opref(arg) {
                ResolvedArg::Slot(offset) => {
                    #[cfg(target_arch = "x86_64")]
                    match i - 1 {
                        0 => dynasm!(self.mc ; .arch x64 ; mov rdi, [rbp + offset]),
                        1 => dynasm!(self.mc ; .arch x64 ; mov rsi, [rbp + offset]),
                        2 => dynasm!(self.mc ; .arch x64 ; mov rdx, [rbp + offset]),
                        3 => dynasm!(self.mc ; .arch x64 ; mov rcx, [rbp + offset]),
                        4 => dynasm!(self.mc ; .arch x64 ; mov r8, [rbp + offset]),
                        5 => dynasm!(self.mc ; .arch x64 ; mov r9, [rbp + offset]),
                        _ => {}
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        let reg = (i - 1) as u8;
                        dynasm!(self.mc ; .arch aarch64
                            ; ldr X(reg), [x29, offset as u32]
                        );
                    }
                }
                ResolvedArg::Const(val) => {
                    #[cfg(target_arch = "x86_64")]
                    match i - 1 {
                        0 => dynasm!(self.mc ; .arch x64 ; mov rdi, QWORD val as i64),
                        1 => dynasm!(self.mc ; .arch x64 ; mov rsi, QWORD val as i64),
                        2 => dynasm!(self.mc ; .arch x64 ; mov rdx, QWORD val as i64),
                        3 => dynasm!(self.mc ; .arch x64 ; mov rcx, QWORD val as i64),
                        4 => dynasm!(self.mc ; .arch x64 ; mov r8, QWORD val as i64),
                        5 => dynasm!(self.mc ; .arch x64 ; mov r9, QWORD val as i64),
                        _ => {}
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        let reg = (i - 1) as u32;
                        self.emit_mov_imm64(reg, val);
                    }
                }
            }
        }

        // Load function pointer into rax/x8 and call.
        match self.resolve_opref(op.arg(0)) {
            ResolvedArg::Slot(offset) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, [rbp + offset]
                    ; call rax
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x8, [x29, offset as u32]
                    ; blr x8
                );
            }
            ResolvedArg::Const(val) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, QWORD val as i64
                    ; call rax
                );
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(8, val);
                    dynasm!(self.mc ; .arch aarch64
                        ; blr x8
                    );
                }
            }
        }

        // Restore frame pointer after call.
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; pop rbp
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; ldp x19, x20, [sp, #16]
            ; ldp x29, x30, [sp], #32
        );

        // Store return value (in rax/x0) to result slot.
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// CALL_ASSEMBLER_*: invoke a compiled JIT loop.
    /// Similar to CALL but targets a JIT-compiled trace.
    /// For now, implemented identically to genop_call.
    fn genop_call_assembler(&mut self, op: &Op) {
        // CALL_ASSEMBLER passes jf_ptr as the sole argument to the
        // target trace. The target trace returns jf_ptr.
        // For now, delegate to the general call mechanism.
        self.genop_call(op);
    }

    // ----------------------------------------------------------------
    // genop_* — allocation
    // x86/assembler.py:2338 genop_new etc.
    // These require GC runtime support. Emit trap for now.
    // ----------------------------------------------------------------

    /// NEW: allocate a fixed-size object. Requires GC runtime.
    /// Emits a trap (UD2/BRK) until GC nursery allocation is wired.
    fn genop_new(&mut self, op: &Op) {
        // Simple allocation: call libc malloc(obj_size).
        // RPython uses GC nursery bump allocation; we use malloc as stub.
        let obj_size = op
            .descr
            .as_ref()
            .and_then(|d| d.as_size_descr())
            .map(|sd| sd.size())
            .unwrap_or(16) as i64;
        let malloc_ptr = libc::malloc as *const () as i64;
        // Call malloc(obj_size)
        #[cfg(target_arch = "x86_64")]
        {
            dynasm!(self.mc ; .arch x64
                ; mov rdi, QWORD obj_size
                ; mov rax, QWORD malloc_ptr
                ; call rax
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.emit_mov_imm64(0, obj_size);
            self.emit_mov_imm64(2, malloc_ptr);
            dynasm!(self.mc ; .arch aarch64 ; blr x2);
        }
        // rax/x0 = pointer to allocated memory
        // Zero-initialize
        #[cfg(target_arch = "x86_64")]
        {
            dynasm!(self.mc ; .arch x64
                ; push rax           // save ptr
                ; mov rdi, rax       // dest = ptr
                ; xor esi, esi       // val = 0
                ; mov rdx, QWORD obj_size // size
                ; mov rax, QWORD (libc::memset as *const () as i64)
                ; call rax
                ; pop rax            // restore ptr
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(self.mc ; .arch aarch64
                ; mov x19, x0       // save ptr in callee-saved
            );
            // memset(ptr, 0, size)
            dynasm!(self.mc ; .arch aarch64
                ; mov x1, 0         // val = 0
            );
            self.emit_mov_imm64(2, obj_size);
            self.emit_mov_imm64(3, libc::memset as *const () as i64);
            dynasm!(self.mc ; .arch aarch64
                ; blr x3
                ; mov x0, x19       // restore ptr
            );
        }
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// NEW_WITH_VTABLE: allocate and set vtable pointer.
    fn genop_new_with_vtable(&mut self, op: &Op) {
        // Same as New, but also write vtable at offset 0.
        let obj_size = op
            .descr
            .as_ref()
            .and_then(|d| d.as_size_descr())
            .map(|sd| sd.size())
            .unwrap_or(16) as i64;
        let vtable = op
            .descr
            .as_ref()
            .and_then(|d| d.as_size_descr())
            .map(|sd| sd.vtable())
            .unwrap_or(0) as i64;
        let malloc_ptr = libc::malloc as *const () as i64;
        #[cfg(target_arch = "x86_64")]
        {
            dynasm!(self.mc ; .arch x64
                ; mov rdi, QWORD obj_size
                ; mov rax, QWORD malloc_ptr
                ; call rax
                ; push rax
                ; mov rdi, rax
                ; xor esi, esi
                ; mov rdx, QWORD obj_size
                ; mov rax, QWORD (libc::memset as *const () as i64)
                ; call rax
                ; pop rax
            );
            // Write vtable at offset 0
            if vtable != 0 {
                dynasm!(self.mc ; .arch x64
                    ; mov rcx, QWORD vtable
                    ; mov [rax], rcx
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.emit_mov_imm64(0, obj_size);
            self.emit_mov_imm64(2, malloc_ptr);
            dynasm!(self.mc ; .arch aarch64 ; blr x2);
            dynasm!(self.mc ; .arch aarch64 ; mov x19, x0);
            dynasm!(self.mc ; .arch aarch64 ; mov x1, 0);
            self.emit_mov_imm64(2, obj_size);
            self.emit_mov_imm64(3, libc::memset as *const () as i64);
            dynasm!(self.mc ; .arch aarch64 ; blr x3);
            dynasm!(self.mc ; .arch aarch64 ; mov x0, x19);
            if vtable != 0 {
                self.emit_mov_imm64(1, vtable);
                dynasm!(self.mc ; .arch aarch64 ; str x1, [x0]);
            }
        }
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// NEW_ARRAY / NEW_ARRAY_CLEAR: allocate an array.
    fn genop_new_array(&mut self, op: &Op) {
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; ud2
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; brk 0
        );
        if !op.pos.is_none() {
            let _ = self.allocate_slot(op.pos);
        }
    }

    // ----------------------------------------------------------------
    // genop_* — misc
    // ----------------------------------------------------------------

    /// FORCE_TOKEN: return the jitframe pointer itself.
    /// x86/assembler.py genop_force_token: mov resloc, ebp
    fn genop_force_token(&mut self, op: &Op) {
        // The frame pointer is the force token.
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, rbp
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x29
        );
        self.store_rax_to_result(op.pos);
    }

    /// STRLEN / UNICODELEN: result = string.length
    /// Load the length field from the string/unicode object header.
    /// arg0 = string pointer. The length is at a fixed offset in the
    /// RPython string representation. For RPython strings, the length
    /// is typically at offset 8 (after the GC header / hash field).
    fn genop_strlen(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        self.load_arg_to_rax(op.arg(0));

        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, [rax + offset]
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x0, offset as u32]
        );

        self.store_rax_to_result(op.pos);
    }

    /// STRGETITEM / UNICODEGETITEM: result = string[index]
    /// arg0 = string pointer, arg1 = index.
    /// Characters are stored after the header. For RPython strings
    /// (1 byte per char), address = base + base_size + index.
    fn genop_strgetitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((16, 1)); // Default for RPython rstr: 16-byte header, 1-byte items

        self.load_arg_to_rax(op.arg(0)); // string pointer
        self.load_arg_to_rcx(op.arg(1)); // index

        #[cfg(target_arch = "x86_64")]
        {
            // Address = rax + base_size + rcx * item_size
            if item_size != 1 {
                dynasm!(self.mc
                    ; .arch x64
                    ; imul rcx, rcx, item_size
                );
            }
            dynasm!(self.mc
                ; .arch x64
                ; add rax, base_size
                ; add rax, rcx
            );
            match item_size {
                1 => dynasm!(self.mc
                    ; .arch x64
                    ; movzx eax, BYTE [rax]
                ),
                2 => dynasm!(self.mc
                    ; .arch x64
                    ; movzx eax, WORD [rax]
                ),
                4 => dynasm!(self.mc
                    ; .arch x64
                    ; mov eax, [rax]
                ),
                _ => dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, [rax]
                ),
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if item_size != 1 {
                self.emit_mov_imm64(2, item_size as i64);
                dynasm!(self.mc ; .arch aarch64
                    ; mul x1, x1, x2
                );
            }
            if base_size != 0 {
                dynasm!(self.mc ; .arch aarch64
                    ; add x0, x0, base_size as u32
                );
            }
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, x1
            );
            match item_size {
                1 => dynasm!(self.mc ; .arch aarch64
                    ; ldrb w0, [x0]
                ),
                2 => dynasm!(self.mc ; .arch aarch64
                    ; ldrh w0, [x0]
                ),
                4 => dynasm!(self.mc ; .arch aarch64
                    ; ldr w0, [x0]
                ),
                _ => dynasm!(self.mc ; .arch aarch64
                    ; ldr x0, [x0]
                ),
            }
        }

        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // Public: set constants from external source
    // ----------------------------------------------------------------

    /// Populate the constants map. Called by the frontend before assembly
    /// if constant OpRefs are used (OpRef.0 >= 10000).
    pub fn set_constants(&mut self, constants: HashMap<u32, i64>) {
        self.constants = constants;
    }
}

/// Flush icache — aarch64 only.
#[cfg(target_arch = "aarch64")]
fn flush_icache(addr: *const u8, len: usize) {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sys_icache_invalidate(start: *mut u8, size: usize);
        }
        unsafe { sys_icache_invalidate(addr as *mut u8, len) };
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        unsafe { __clear_cache(addr as *mut u8, (addr as *mut u8).add(len)) };
    }
}
