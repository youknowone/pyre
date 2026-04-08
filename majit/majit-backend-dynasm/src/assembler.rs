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
use crate::codebuf;
use crate::guard::DynasmFailDescr;
use crate::regalloc::{RegAlloc, RegAllocOp};
use crate::regloc::Loc;

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
    /// Bridge source slots: maps bridge InputArg[i] to the parent
    /// trace's jitframe slot. None for loop traces.
    bridge_source_slots: Option<Vec<usize>>,

    // ── State tracking for code generation ──
    /// Maps OpRef index → jitframe slot index.
    opref_to_slot: HashMap<u32, usize>,
    /// resoperation.py Box.type parity: OpRef → Type for fail_arg_types
    /// inference. Populated during code generation from input types and
    /// op result types.
    value_types: HashMap<u32, Type>,
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
    /// Buffer-relative offset of the LABEL op.
    loop_label_offset: Option<AssemblyOffset>,
    /// Regalloc: LABEL's arg locations — JUMP must remap to these.
    loop_arglocs: Vec<Loc>,
    /// llmodel.py:64-69 self.vtable_offset — typeptr field byte offset.
    /// `None` corresponds to RPython's gcremovetypeptr config.
    vtable_offset: Option<usize>,
    /// llsupport/gc.py:563 vtable→typeid table, materialized by the runner
    /// via gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr. Used by
    /// the gcremovetypeptr branch of `_cmp_guard_class`.
    classptr_to_typeid: HashMap<i64, u32>,
    /// Dynamic label at the function entry for self-recursive CALL_ASSEMBLER.
    self_entry_label: Option<DynamicLabel>,
    /// assembler.py:320 descr._ll_function_addr parity:
    /// Maps call_target_token → compiled code address for CALL_ASSEMBLER.
    /// Populated by the runner before compilation, from registered loop targets.
    call_assembler_targets: HashMap<u64, usize>,
    /// regalloc.py jump_target_descr parity: absolute address of the
    /// LABEL in the original loop, for bridge JUMP to return to.
    /// Set by assemble_bridge from the parent loop's label_addr.
    jump_target_addr: Option<usize>,
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
    /// regalloc parity: snapshot of opref_to_slot at guard emission time.
    /// Needed by recovery stubs to read fail_args from correct slots.
    opref_to_slot_snapshot: HashMap<u32, usize>,
    /// Constants to store in frame during recovery.
    /// Each entry: (frame_slot_index, constant_value).
    const_stores: Vec<(usize, i64)>,
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
    /// Frame depth (number of jitframe slots used).
    /// AtomicUsize for redirect_call_assembler's update_frame_info
    /// parity: may be updated through &CompiledCode (shared ref).
    pub frame_depth: std::sync::atomic::AtomicUsize,
    /// Absolute address of the LABEL op (loop body entry).
    /// Used by bridge JUMP to return to the loop. RPython stores
    /// this as TargetToken._ll_loop_code.
    pub label_addr: usize,
}

impl Assembler386 {
    /// assembler.py:54 __init__
    pub fn new(
        trace_id: u64,
        header_pc: u64,
        constants: HashMap<u32, i64>,
        vtable_offset: Option<usize>,
        classptr_to_typeid: HashMap<i64, u32>,
    ) -> Self {
        Assembler386 {
            mc: Assembler::new().unwrap(),
            pending_guard_tokens: Vec::new(),
            frame_depth: JITFRAME_FIXED_SIZE,
            fail_descrs: Vec::new(),
            trace_id,
            header_pc,
            input_types: Vec::new(),
            bridge_source_slots: None,
            opref_to_slot: HashMap::new(),
            value_types: HashMap::new(),
            constants,
            next_slot: 0,
            guard_success_cc: None,
            loop_label: None,
            loop_label_offset: None,
            loop_arglocs: Vec::new(),
            vtable_offset,
            classptr_to_typeid,
            self_entry_label: None,
            call_assembler_targets: HashMap::new(),
            jump_target_addr: None,
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
    /// Cranelift resolve_opref parity: check constants map FIRST
    /// (regardless of CONST_BIT), then fall back to slot mapping.
    fn resolve_opref(&self, opref: OpRef) -> ResolvedArg {
        // Op results take precedence over constants (Cranelift parity).
        if let Some(&slot) = self.opref_to_slot.get(&opref.0) {
            return ResolvedArg::Slot(Self::slot_offset(slot));
        }
        if let Some(&val) = self.constants.get(&opref.0) {
            return ResolvedArg::Const(val);
        }
        // Unmapped OpRef — treat as 0.
        ResolvedArg::Const(0)
    }

    /// Allocate a frame slot for an OpRef and return the slot index.
    /// Reuses existing slot if the OpRef already has one.
    fn allocate_slot(&mut self, opref: OpRef) -> usize {
        if let Some(&existing) = self.opref_to_slot.get(&opref.0) {
            return existing;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        if self.next_slot + 1 > self.frame_depth {
            self.frame_depth = self.next_slot + 1;
        }
        self.opref_to_slot.insert(opref.0, slot);
        slot
    }

    // ── Location-aware code emission (RPython regalloc parity) ──
    // assembler.py regalloc_mov: move value between any two locations.

    /// assembler.py:1145 regalloc_mov(from_loc, to_loc).
    /// Emit a move between any two locations: reg↔reg, reg↔frame, imm→reg, imm→frame.
    fn regalloc_mov(&mut self, src: &Loc, dst: &Loc) {
        match (src, dst) {
            (Loc::Reg(s), Loc::Reg(d)) if s == d => {}
            (Loc::Reg(s), Loc::Reg(d)) => {
                #[cfg(target_arch = "x86_64")]
                {
                    if s.is_xmm && d.is_xmm {
                        dynasm!(self.mc ; .arch x64 ; movsd Rx(d.value), Rx(s.value));
                    } else if !s.is_xmm && !d.is_xmm {
                        dynasm!(self.mc ; .arch x64 ; mov Rq(d.value), Rq(s.value));
                    } else if s.is_xmm && !d.is_xmm {
                        dynasm!(self.mc ; .arch x64 ; movq Rq(d.value), Rx(s.value));
                    } else {
                        dynasm!(self.mc ; .arch x64 ; movq Rx(d.value), Rq(s.value));
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    if s.is_xmm && d.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), D(s.value));
                    } else if !s.is_xmm && !d.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; mov X(d.value), X(s.value));
                    } else if s.is_xmm && !d.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; fmov X(d.value), D(s.value));
                    } else {
                        dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), X(s.value));
                    }
                }
            }
            (Loc::Reg(s), Loc::Frame(f)) => {
                let ofs = f.ebp_loc.value;
                #[cfg(target_arch = "x86_64")]
                {
                    if s.is_xmm {
                        dynasm!(self.mc ; .arch x64 ; movsd [rbp + ofs], Rx(s.value));
                    } else {
                        dynasm!(self.mc ; .arch x64 ; mov [rbp + ofs], Rq(s.value));
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    if s.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; str D(s.value), [x29, ofs as u32]);
                    } else {
                        dynasm!(self.mc ; .arch aarch64 ; str X(s.value), [x29, ofs as u32]);
                    }
                }
            }
            (Loc::Frame(f), Loc::Reg(d)) => {
                let ofs = f.ebp_loc.value;
                #[cfg(target_arch = "x86_64")]
                {
                    if d.is_xmm {
                        dynasm!(self.mc ; .arch x64 ; movsd Rx(d.value), [rbp + ofs]);
                    } else {
                        dynasm!(self.mc ; .arch x64 ; mov Rq(d.value), [rbp + ofs]);
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    if d.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; ldr D(d.value), [x29, ofs as u32]);
                    } else {
                        dynasm!(self.mc ; .arch aarch64 ; ldr X(d.value), [x29, ofs as u32]);
                    }
                }
            }
            (Loc::Immed(i), Loc::Reg(d)) => {
                #[cfg(target_arch = "x86_64")]
                {
                    if d.is_xmm {
                        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                        dynasm!(self.mc ; .arch x64
                            ; mov Rq(scratch), QWORD i.value
                            ; movq Rx(d.value), Rq(scratch)
                        );
                    } else {
                        dynasm!(self.mc ; .arch x64 ; mov Rq(d.value), QWORD i.value);
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    if d.is_xmm {
                        self.emit_mov_imm64(16, i.value); // x16 = scratch
                        dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), X(16));
                    } else {
                        self.emit_mov_imm64(d.value as u32, i.value);
                    }
                }
            }
            (Loc::Immed(i), Loc::Frame(f)) => {
                let ofs = f.ebp_loc.value;
                #[cfg(target_arch = "x86_64")]
                {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD i.value
                        ; mov [rbp + ofs], Rq(scratch)
                    );
                }
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(16, i.value);
                    dynasm!(self.mc ; .arch aarch64 ; str x16, [x29, ofs as u32]);
                }
            }
            (Loc::Frame(f1), Loc::Frame(f2)) if f1.position == f2.position => {}
            (Loc::Frame(f1), Loc::Frame(f2)) => {
                let o1 = f1.ebp_loc.value;
                let o2 = f2.ebp_loc.value;
                #[cfg(target_arch = "x86_64")]
                {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), [rbp + o1]
                        ; mov [rbp + o2], Rq(scratch)
                    );
                }
                #[cfg(target_arch = "aarch64")]
                {
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x16, [x29, o1 as u32]
                        ; str x16, [x29, o2 as u32]
                    );
                }
            }
            _ => {}
        }
    }

    /// Emit: ADD/SUB/AND/OR/XOR reg, loc
    fn emit_binop_reg_loc(&mut self, opcode: OpCode, dst_reg: u8, src: &Loc) {
        // aarch64: load src to x16 scratch if not in register
        #[cfg(target_arch = "aarch64")]
        {
            let src_reg = match src {
                Loc::Reg(s) => s.value,
                Loc::Frame(f) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                    16
                }
                Loc::Immed(i) => {
                    self.emit_mov_imm64(16, i.value);
                    16
                }
                _ => return,
            };
            let d = dst_reg;
            let s = src_reg as u8;
            match opcode {
                OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                    dynasm!(self.mc ; .arch aarch64 ; add X(d), X(d), X(s));
                }
                OpCode::IntSub | OpCode::IntSubOvf => {
                    dynasm!(self.mc ; .arch aarch64 ; sub X(d), X(d), X(s));
                }
                OpCode::IntMul | OpCode::IntMulOvf => {
                    dynasm!(self.mc ; .arch aarch64 ; mul X(d), X(d), X(s));
                }
                OpCode::IntAnd => {
                    dynasm!(self.mc ; .arch aarch64 ; and X(d), X(d), X(s));
                }
                OpCode::IntOr => {
                    dynasm!(self.mc ; .arch aarch64 ; orr X(d), X(d), X(s));
                }
                OpCode::IntXor => {
                    dynasm!(self.mc ; .arch aarch64 ; eor X(d), X(d), X(s));
                }
                _ => {}
            }
            return;
        }
        #[cfg(target_arch = "x86_64")]
        match src {
            Loc::Reg(s) => match opcode {
                OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                    dynasm!(self.mc ; .arch x64 ; add Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntSub | OpCode::IntSubOvf => {
                    dynasm!(self.mc ; .arch x64 ; sub Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntMul | OpCode::IntMulOvf => {
                    dynasm!(self.mc ; .arch x64 ; imul Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntAnd => {
                    dynasm!(self.mc ; .arch x64 ; and Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntOr => {
                    dynasm!(self.mc ; .arch x64 ; or  Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntXor => {
                    dynasm!(self.mc ; .arch x64 ; xor Rq(dst_reg), Rq(s.value));
                }
                _ => {}
            },
            Loc::Frame(f) => {
                let ofs = f.ebp_loc.value;
                match opcode {
                    OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                        dynasm!(self.mc ; .arch x64 ; add Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntSub | OpCode::IntSubOvf => {
                        dynasm!(self.mc ; .arch x64 ; sub Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntMul | OpCode::IntMulOvf => {
                        dynasm!(self.mc ; .arch x64 ; imul Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntAnd => {
                        dynasm!(self.mc ; .arch x64 ; and Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntOr => {
                        dynasm!(self.mc ; .arch x64 ; or  Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntXor => {
                        dynasm!(self.mc ; .arch x64 ; xor Rq(dst_reg), [rbp + ofs]);
                    }
                    _ => {}
                }
            }
            Loc::Immed(i) => {
                let v = i.value as i32;
                match opcode {
                    OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                        dynasm!(self.mc ; .arch x64 ; add Rq(dst_reg), v);
                    }
                    OpCode::IntSub | OpCode::IntSubOvf => {
                        dynasm!(self.mc ; .arch x64 ; sub Rq(dst_reg), v);
                    }
                    OpCode::IntAnd => {
                        dynasm!(self.mc ; .arch x64 ; and Rq(dst_reg), v);
                    }
                    OpCode::IntOr => {
                        dynasm!(self.mc ; .arch x64 ; or  Rq(dst_reg), v);
                    }
                    OpCode::IntXor => {
                        dynasm!(self.mc ; .arch x64 ; xor Rq(dst_reg), v);
                    }
                    _ => {
                        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                        dynasm!(self.mc ; .arch x64 ; mov Rq(scratch), QWORD i.value ; imul Rq(dst_reg), Rq(scratch));
                    }
                }
            }
            _ => {}
        }
    }

    /// Emit: CMP loc0, loc1
    fn emit_cmp_loc_loc(&mut self, loc0: &Loc, loc1: &Loc) {
        #[cfg(target_arch = "aarch64")]
        {
            // Load loc0 into x16 if needed, loc1 into x17 if needed
            let r0 = match loc0 {
                Loc::Reg(r) => r.value,
                Loc::Frame(f) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                    16
                }
                Loc::Immed(i) => {
                    self.emit_mov_imm64(16, i.value);
                    16
                }
                _ => return,
            };
            let r1 = match loc1 {
                Loc::Reg(s) => s.value,
                Loc::Frame(f) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x29, f.ebp_loc.value as u32]);
                    17
                }
                Loc::Immed(i) => {
                    self.emit_mov_imm64(17, i.value);
                    17
                }
                _ => return,
            };
            dynasm!(self.mc ; .arch aarch64 ; cmp X(r0 as u8), X(r1 as u8));
            return;
        }
        #[cfg(target_arch = "x86_64")]
        match (loc0, loc1) {
            (Loc::Reg(r), Loc::Reg(s)) => {
                dynasm!(self.mc ; .arch x64 ; cmp Rq(r.value), Rq(s.value));
            }
            (Loc::Reg(r), Loc::Frame(f)) => {
                dynasm!(self.mc ; .arch x64 ; cmp Rq(r.value), [rbp + f.ebp_loc.value]);
            }
            (Loc::Reg(r), Loc::Immed(i)) => {
                dynasm!(self.mc ; .arch x64 ; cmp Rq(r.value), i.value as i32);
            }
            (Loc::Frame(f), Loc::Reg(s)) => {
                dynasm!(self.mc ; .arch x64 ; cmp [rbp + f.ebp_loc.value], Rq(s.value));
            }
            (Loc::Frame(f), Loc::Immed(i)) => {
                dynasm!(self.mc ; .arch x64 ; cmp QWORD [rbp + f.ebp_loc.value], i.value as i32);
            }
            _ => {
                self.regalloc_mov(loc0, &Loc::Reg(crate::regloc::X86_64_SCRATCH_REG));
                self.emit_cmp_loc_loc(&Loc::Reg(crate::regloc::X86_64_SCRATCH_REG), loc1);
            }
        }
    }

    /// Emit: TEST loc, loc (for guard_true/guard_false)
    fn emit_test_loc(&mut self, loc: &Loc) {
        #[cfg(target_arch = "aarch64")]
        {
            let r = match loc {
                Loc::Reg(r) => r.value,
                Loc::Frame(f) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                    16
                }
                Loc::Immed(i) => {
                    self.emit_mov_imm64(16, i.value);
                    16
                }
                _ => return,
            };
            dynasm!(self.mc ; .arch aarch64 ; tst X(r as u8), X(r as u8));
            return;
        }
        #[cfg(target_arch = "x86_64")]
        match loc {
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch x64 ; test Rq(r.value), Rq(r.value));
            }
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch x64 ; cmp QWORD [rbp + f.ebp_loc.value], 0);
            }
            Loc::Immed(i) => {
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                dynasm!(self.mc ; .arch x64 ; mov Rq(scratch), QWORD i.value ; test Rq(scratch), Rq(scratch));
            }
            _ => {}
        }
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
            ; push r12               // save r12 (used by genop_call_assembler)
            ; mov rbp, rdi
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; stp x29, x30, [sp, #-32]!
            ; stp x19, x20, [sp, #16]   // save callee-saved regs
            ; mov x29, x0
        );

        if let Some(ref source_slots) = self.bridge_source_slots {
            let mut max_slot = 0;
            for (i, ia) in inputargs.iter().enumerate() {
                let src_slot = source_slots.get(i).copied().unwrap_or(i);
                self.opref_to_slot.insert(i as u32, src_slot);
                self.value_types.insert(i as u32, ia.tp);
                if src_slot >= max_slot {
                    max_slot = src_slot + 1;
                }
            }
            self.next_slot = max_slot;
        } else {
            for (i, ia) in inputargs.iter().enumerate() {
                self.opref_to_slot.insert(i as u32, i);
                self.value_types.insert(i as u32, ia.tp);
            }
            self.next_slot = inputargs.len();
        }
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
            ; pop r12                // restore r12
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
        // Create a dynamic label at the entry point for self-recursive
        // CALL_ASSEMBLER (redirect_call_assembler parity).
        let entry_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; =>entry_label);
        self.self_entry_label = Some(entry_label);
        let entry = self.mc.offset();
        self._assemble(inputargs, ops)?;

        // regalloc sets fail_arg_locs in append_guard_token_with_faillocs.
        // No allocate_unmapped_fail_arg_slots needed.

        // assembler.py:553 write_pending_failure_recoveries
        let stub_offsets = self.write_pending_failure_recoveries();

        // assembler.py:556 materialize_loop — finalize to executable memory
        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        // assembler.py:849 patch_pending_failure_recoveries
        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);

        // TargetToken._ll_loop_code parity: absolute address of the LABEL.
        let label_addr = self
            .loop_label_offset
            .map(|off| rawstart + off.0)
            .unwrap_or(0);

        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs,
            input_types: self.input_types,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            label_addr,
        })
    }

    /// Set the jump target address for bridge JUMP ops.
    /// assembler.py closing_jump parity: bridge JUMP returns to
    /// the original loop's LABEL via absolute address.
    pub fn set_jump_target_addr(&mut self, addr: usize) {
        self.jump_target_addr = Some(addr);
    }

    /// assembler.py:320 descr._ll_function_addr parity: store
    /// call_target_token → code_addr mappings for CALL_ASSEMBLER.
    pub fn set_call_assembler_targets(&mut self, targets: HashMap<u64, usize>) {
        self.call_assembler_targets = targets;
    }

    /// assembler.py:623 assemble_bridge: compile a bridge trace.
    pub fn assemble_bridge(
        mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        source_slots: &[usize],
    ) -> Result<CompiledCode, BackendError> {
        self.input_types = inputargs.iter().map(|ia| ia.tp).collect();
        self.bridge_source_slots = if source_slots.is_empty() {
            None
        } else {
            Some(source_slots.to_vec())
        };

        // assembler.py:641 prepare_bridge
        let entry = self.mc.offset();
        self._assemble(inputargs, ops)?;
        let stub_offsets = self.write_pending_failure_recoveries();

        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);

        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs,
            input_types: self.input_types,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            label_addr: 0,
        })
    }

    /// assembler.py:779 _assemble — walk operations and emit code.
    ///
    /// Uses the register allocator (regalloc.rs) to assign registers/frame
    /// locations, then emits code using those locations. This replaces the
    /// old frame-slot model where every value went through [rbp+offset].
    fn _assemble(&mut self, inputargs: &[InputArg], ops: &[Op]) -> Result<(), BackendError> {
        // Emit prologue.
        self._call_header(inputargs);

        // ── Run register allocator ──
        // assembler.py:537 prepare_loop / assembler.py:638 prepare_bridge
        let mut ra = RegAlloc::new(self.constants.clone());
        if self.bridge_source_slots.is_some() {
            // Bridge: inputargs are already in known locations from parent guard.
            // For now, prepare_loop places all inputs in frame slots.
            ra.prepare_loop(inputargs, ops);
        } else {
            ra.prepare_loop(inputargs, ops);
        }
        // assembler.py:374 walk_operations — get allocation decisions.
        let ra_ops = ra.walk_operations(inputargs, ops);
        self.frame_depth = self
            .frame_depth
            .max(ra.get_final_frame_depth() + JITFRAME_FIXED_SIZE);

        // Sync regalloc frame positions to opref_to_slot for backward
        // compatibility with genop_call/genop_call_assembler which still
        // use resolve_opref. When regalloc spills a value to a frame slot,
        // that slot's position must be visible to resolve_opref.
        for iarg in inputargs {
            self.opref_to_slot.insert(iarg.index, iarg.index as usize);
            self.value_types.insert(iarg.index, iarg.tp);
        }
        // Also sync any frame allocations from regalloc's FrameManager.
        for (&opref, lifetime) in ra.longevity.lifetimes_iter() {
            if let Some(floc) = lifetime.current_frame_loc {
                self.opref_to_slot.insert(opref.0, floc.position);
            }
        }
        self.next_slot = ra.get_final_frame_depth();

        let mut fail_index = 0u32;

        // ── Emit code from regalloc decisions ──
        for ra_op in &ra_ops {
            match ra_op {
                RegAllocOp::Skip => {
                    // Dead operation — skip.
                    continue;
                }
                RegAllocOp::Move { src, dst } => {
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!("[dynasm] move: {:?} → {:?}", src, dst);
                    }
                    self.regalloc_mov(src, dst);
                    continue;
                }
                RegAllocOp::Perform {
                    op_index,
                    arglocs,
                    result_loc,
                } => {
                    let op = &ops[*op_index];
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        let al: Vec<String> = arglocs.iter().map(|l| format!("{:?}", l)).collect();
                        eprintln!(
                            "[dynasm] emit[{}]: {:?} args=[{}] result={:?}",
                            op_index,
                            op.opcode,
                            al.join(", "),
                            result_loc
                        );
                    }
                    self.regalloc_perform(op, arglocs, result_loc.as_ref(), fail_index, ops);
                    if op.opcode.is_guard() {
                        fail_index += 1;
                    }
                    if op.opcode == OpCode::Finish {
                        fail_index += 1;
                    }
                    if !op.pos.is_none() {
                        self.value_types.insert(op.pos.0, op.result_type());
                    }
                }
                RegAllocOp::PerformGuard {
                    op_index,
                    arglocs,
                    result_loc,
                    faillocs,
                } => {
                    let op = &ops[*op_index];
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[dynasm] guard[{}]: {:?} args=[{}] faillocs={}",
                            op_index,
                            op.opcode,
                            arglocs
                                .iter()
                                .map(|l| format!("{:?}", l))
                                .collect::<Vec<_>>()
                                .join(", "),
                            faillocs.len()
                        );
                    }
                    self.regalloc_perform_guard(
                        op,
                        arglocs,
                        result_loc.as_ref(),
                        faillocs,
                        fail_index,
                    );
                    fail_index += 1;
                    if !op.pos.is_none() {
                        self.value_types.insert(op.pos.0, op.result_type());
                    }
                }
                RegAllocOp::PerformDiscard { op_index, arglocs } => {
                    let op = &ops[*op_index];
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!("[dynasm] discard[{}]: {:?}", op_index, op.opcode);
                    }
                    self.regalloc_perform(op, arglocs, None, fail_index, ops);
                }
            }
        }

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] _assemble done: pending_guard_tokens={} fail_index={}",
                self.pending_guard_tokens.len(),
                fail_index
            );
        }

        Ok(())
    }

    /// assembler.py:326 regalloc_perform — emit code for a non-guard op.
    /// Called from the regalloc dispatch loop with pre-computed locations.
    fn regalloc_perform(
        &mut self,
        op: &Op,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        fail_index: u32,
        ops: &[Op],
    ) {
        match op.opcode {
            // ── Integer binary (result_loc == arglocs[0], guaranteed by regalloc) ──
            OpCode::IntAdd
            | OpCode::IntAddOvf
            | OpCode::IntSub
            | OpCode::IntSubOvf
            | OpCode::IntMul
            | OpCode::IntMulOvf
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::NurseryPtrIncrement => {
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                }
            }
            // ── Unary integer (result in arglocs[0] register) ──
            OpCode::IntNeg => {
                if let Some(Loc::Reg(r)) = result_loc {
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64 ; neg Rq(r.value));
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; neg X(r.value), X(r.value));
                }
            }
            OpCode::IntInvert => {
                if let Some(Loc::Reg(r)) = result_loc {
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64 ; not Rq(r.value));
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; mvn X(r.value), X(r.value));
                }
            }
            // ── Shifts ──
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                if let (Some(Loc::Reg(dst)), Some(shift_loc)) = (result_loc, arglocs.get(1)) {
                    #[cfg(target_arch = "aarch64")]
                    {
                        // aarch64: 3-operand shifts, no ECX constraint
                        let sr = match shift_loc {
                            Loc::Reg(s) => s.value,
                            Loc::Immed(i) => {
                                self.emit_mov_imm64(16, i.value);
                                16
                            }
                            Loc::Frame(f) => {
                                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                                16
                            }
                            _ => 16,
                        };
                        match op.opcode {
                            OpCode::IntLshift => {
                                dynasm!(self.mc ; .arch aarch64 ; lsl X(dst.value), X(dst.value), X(sr as u8));
                            }
                            OpCode::IntRshift => {
                                dynasm!(self.mc ; .arch aarch64 ; asr X(dst.value), X(dst.value), X(sr as u8));
                            }
                            OpCode::UintRshift => {
                                dynasm!(self.mc ; .arch aarch64 ; lsr X(dst.value), X(dst.value), X(sr as u8));
                            }
                            _ => {}
                        }
                    }
                    #[cfg(target_arch = "x86_64")]
                    match shift_loc {
                        Loc::Immed(i) => {
                            let sh = i.value as i8;
                            match op.opcode {
                                OpCode::IntLshift => {
                                    dynasm!(self.mc ; .arch x64 ; shl Rq(dst.value), sh);
                                }
                                OpCode::IntRshift => {
                                    dynasm!(self.mc ; .arch x64 ; sar Rq(dst.value), sh);
                                }
                                OpCode::UintRshift => {
                                    dynasm!(self.mc ; .arch x64 ; shr Rq(dst.value), sh);
                                }
                                _ => {}
                            }
                        }
                        Loc::Reg(s) if s.value == 1 => match op.opcode {
                            OpCode::IntLshift => {
                                dynasm!(self.mc ; .arch x64 ; shl Rq(dst.value), cl);
                            }
                            OpCode::IntRshift => {
                                dynasm!(self.mc ; .arch x64 ; sar Rq(dst.value), cl);
                            }
                            OpCode::UintRshift => {
                                dynasm!(self.mc ; .arch x64 ; shr Rq(dst.value), cl);
                            }
                            _ => {}
                        },
                        _ => {
                            self.regalloc_mov(shift_loc, &Loc::Reg(crate::regloc::ECX));
                            match op.opcode {
                                OpCode::IntLshift => {
                                    dynasm!(self.mc ; .arch x64 ; shl Rq(dst.value), cl);
                                }
                                OpCode::IntRshift => {
                                    dynasm!(self.mc ; .arch x64 ; sar Rq(dst.value), cl);
                                }
                                OpCode::UintRshift => {
                                    dynasm!(self.mc ; .arch x64 ; shr Rq(dst.value), cl);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // ── Integer comparisons ──
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe
            | OpCode::PtrEq
            | OpCode::PtrNe
            | OpCode::InstancePtrEq
            | OpCode::InstancePtrNe => {
                if arglocs.len() >= 2 {
                    self.emit_cmp_loc_loc(&arglocs[0], &arglocs[1]);
                }
                if let Some(Loc::Reg(r)) = result_loc {
                    let cc = Self::opcode_to_cc(op.opcode);
                    self.emit_setcc(cc, r.value);
                }
            }
            OpCode::IntIsTrue => {
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.emit_test_loc(src);
                    self.emit_setcc(CC_NE, r.value);
                }
            }
            OpCode::IntIsZero => {
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.emit_test_loc(src);
                    self.emit_setcc(CC_E, r.value);
                }
            }
            OpCode::IntForceGeZero => {
                if let Some(Loc::Reg(r)) = result_loc {
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64
                        ; test Rq(r.value), Rq(r.value)
                        ; jge >pos
                        ; xor Rq(r.value), Rq(r.value)
                        ; pos:
                    );
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64
                        ; cmp X(r.value), xzr
                        ; csel X(r.value), X(r.value), xzr, ge
                    );
                }
            }
            OpCode::IntSignext => {
                // arglocs = [argloc, numbytes_loc], result_loc = separate reg
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, &Loc::Reg(*r));
                    // signext handled by assembler based on numbytes
                }
            }
            // ── Float binary ──
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                if let Some(Loc::Reg(dst)) = result_loc {
                    // Ensure second arg is in an XMM register
                    let src_reg = if let Some(Loc::Reg(s)) = arglocs.get(1) {
                        *s
                    } else if let Some(src_loc) = arglocs.get(1) {
                        // Immed or Frame — load to scratch XMM (d14/xmm14)
                        let scratch = crate::regloc::RegLoc::new(14, true);
                        self.regalloc_mov(src_loc, &Loc::Reg(scratch));
                        scratch
                    } else {
                        return; // shouldn't happen
                    };
                    #[cfg(target_arch = "x86_64")]
                    match op.opcode {
                        OpCode::FloatAdd => {
                            dynasm!(self.mc ; .arch x64 ; addsd Rx(dst.value), Rx(src_reg.value));
                        }
                        OpCode::FloatSub => {
                            dynasm!(self.mc ; .arch x64 ; subsd Rx(dst.value), Rx(src_reg.value));
                        }
                        OpCode::FloatMul => {
                            dynasm!(self.mc ; .arch x64 ; mulsd Rx(dst.value), Rx(src_reg.value));
                        }
                        OpCode::FloatTrueDiv => {
                            dynasm!(self.mc ; .arch x64 ; divsd Rx(dst.value), Rx(src_reg.value));
                        }
                        _ => {}
                    }
                    #[cfg(target_arch = "aarch64")]
                    match op.opcode {
                        OpCode::FloatAdd => {
                            dynasm!(self.mc ; .arch aarch64 ; fadd D(dst.value), D(dst.value), D(src_reg.value));
                        }
                        OpCode::FloatSub => {
                            dynasm!(self.mc ; .arch aarch64 ; fsub D(dst.value), D(dst.value), D(src_reg.value));
                        }
                        OpCode::FloatMul => {
                            dynasm!(self.mc ; .arch aarch64 ; fmul D(dst.value), D(dst.value), D(src_reg.value));
                        }
                        OpCode::FloatTrueDiv => {
                            dynasm!(self.mc ; .arch aarch64 ; fdiv D(dst.value), D(dst.value), D(src_reg.value));
                        }
                        _ => {}
                    }
                }
            }
            OpCode::FloatNeg => {
                if let Some(Loc::Reg(r)) = result_loc {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                        dynasm!(self.mc ; .arch x64
                            ; xorpd Rx(scratch as u8), Rx(scratch as u8)
                            ; subsd Rx(scratch as u8), Rx(r.value)
                            ; movsd Rx(r.value), Rx(scratch as u8)
                        );
                    }
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; fneg D(r.value), D(r.value));
                }
            }
            OpCode::FloatAbs => {
                if let Some(Loc::Reg(r)) = result_loc {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                        dynasm!(self.mc ; .arch x64
                            ; mov Rq(scratch), QWORD 0x7FFFFFFFFFFFFFFF_u64 as i64
                            ; movq Rx(scratch as u8), Rq(scratch)
                            ; andpd Rx(r.value), Rx(scratch as u8)
                        );
                    }
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; fabs D(r.value), D(r.value));
                }
            }
            // ── Float comparisons ──
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe => {
                if let (Some(Loc::Reg(a)), Some(Loc::Reg(b))) = (arglocs.first(), arglocs.get(1)) {
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64 ; ucomisd Rx(a.value), Rx(b.value));
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; fcmp D(a.value), D(b.value));
                    if let Some(Loc::Reg(r)) = result_loc {
                        let cc = Self::float_opcode_to_cc(op.opcode);
                        self.emit_setcc(cc, r.value);
                    }
                }
            }
            // ── Casts ──
            OpCode::CastIntToFloat => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let sr = match src {
                        Loc::Reg(s) => s.value,
                        _ => {
                            self.regalloc_mov(
                                src,
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64 ; cvtsi2sd Rx(dst.value), Rq(sr));
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; scvtf D(dst.value), X(sr));
                }
            }
            OpCode::CastFloatToInt => {
                if let (Some(Loc::Reg(src)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64 ; cvttsd2si Rq(dst.value), Rx(src.value));
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; fcvtzs X(dst.value), D(src.value));
                }
            }
            // ── Same-as / identity ──
            OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::CastPtrToInt
            | OpCode::CastIntToPtr
            | OpCode::CastOpaquePtr
            | OpCode::LoadFromGcTable
            | OpCode::VirtualRefR
            | OpCode::ConvertFloatBytesToLonglong
            | OpCode::ConvertLonglongBytesToFloat => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // ── Memory loads: getfield pattern ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldGcPureI
            | OpCode::GetfieldGcPureR
            | OpCode::GetfieldGcPureF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF
            | OpCode::ArraylenGc
            | OpCode::Strlen
            | OpCode::Unicodelen => {
                if let (Some(Loc::Reg(base)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let ofs = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_field_descr())
                        .map(|fd| fd.offset() as i32)
                        .unwrap_or(0);
                    let field_size = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_field_descr())
                        .map(|fd| fd.field_size())
                        .unwrap_or(8);
                    #[cfg(target_arch = "x86_64")]
                    {
                        if dst.is_xmm {
                            dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [Rq(base.value) + ofs]);
                        } else {
                            match field_size {
                                1 => {
                                    dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), BYTE [Rq(base.value) + ofs]);
                                }
                                2 => {
                                    dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), WORD [Rq(base.value) + ofs]);
                                }
                                4 => {
                                    dynasm!(self.mc ; .arch x64 ; movsxd Rq(dst.value), DWORD [Rq(base.value) + ofs]);
                                }
                                _ => {
                                    dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(base.value) + ofs]);
                                }
                            }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if dst.is_xmm {
                            dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [X(base.value), ofs as u32]);
                        } else {
                            match field_size {
                                1 => {
                                    dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [X(base.value), ofs as u32]);
                                }
                                2 => {
                                    dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [X(base.value), ofs as u32]);
                                }
                                4 => {
                                    dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [X(base.value), ofs as u32]);
                                }
                                _ => {
                                    dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(base.value), ofs as u32]);
                                }
                            }
                        }
                    }
                }
            }
            // ── Memory stores: setfield pattern ──
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                if let (Some(Loc::Reg(base)), Some(val_loc)) = (arglocs.first(), arglocs.get(1)) {
                    let ofs = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_field_descr())
                        .map(|fd| fd.offset() as i32)
                        .unwrap_or(0);
                    let field_size = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_field_descr())
                        .map(|fd| fd.field_size())
                        .unwrap_or(8);
                    match val_loc {
                        Loc::Reg(v) if v.is_xmm => {
                            dynasm!(self.mc ; .arch x64 ; movsd [Rq(base.value) + ofs], Rx(v.value));
                        }
                        Loc::Reg(v) => match field_size {
                            1 => {
                                dynasm!(self.mc ; .arch x64 ; mov BYTE [Rq(base.value) + ofs], Rb(v.value));
                            }
                            2 => {
                                dynasm!(self.mc ; .arch x64 ; mov WORD [Rq(base.value) + ofs], Rw(v.value));
                            }
                            4 => {
                                dynasm!(self.mc ; .arch x64 ; mov DWORD [Rq(base.value) + ofs], Rd(v.value));
                            }
                            _ => {
                                dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rq(v.value));
                            }
                        },
                        _ => {
                            // val in frame/immed → move to scratch first
                            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                            self.regalloc_mov(
                                val_loc,
                                &Loc::Reg(crate::regloc::X86_64_SCRATCH_REG),
                            );
                            dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rq(scratch));
                        }
                    }
                }
            }
            // ── GC load / raw load ──
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::RawLoadI
            | OpCode::RawLoadF => {
                if let (Some(Loc::Reg(base)), Some(ofs_loc), Some(Loc::Reg(dst))) =
                    (arglocs.first(), arglocs.get(1), result_loc)
                {
                    match ofs_loc {
                        Loc::Immed(i) => {
                            let o = i.value as i32;
                            if dst.is_xmm {
                                dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [Rq(base.value) + o]);
                            } else {
                                dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(base.value) + o]);
                            }
                        }
                        Loc::Reg(ofs_r) => {
                            if dst.is_xmm {
                                dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [Rq(base.value) + Rq(ofs_r.value)]);
                            } else {
                                dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(base.value) + Rq(ofs_r.value)]);
                            }
                        }
                        _ => {}
                    }
                }
            }
            // ── GC store / raw store ──
            OpCode::GcStore | OpCode::RawStore => {
                if arglocs.len() >= 3 {
                    if let (Some(Loc::Reg(base)), Some(Loc::Reg(val))) =
                        (arglocs.first(), arglocs.get(2))
                    {
                        match &arglocs[1] {
                            Loc::Immed(i) => {
                                let o = i.value as i32;
                                dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + o], Rq(val.value));
                            }
                            Loc::Reg(ofs_r) => {
                                dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + Rq(ofs_r.value)], Rq(val.value));
                            }
                            _ => {}
                        }
                    }
                }
            }
            // ── Control flow ──
            OpCode::Jump => {
                // RPython: remap_frame_layout — move JUMP args from their
                // current locations to LABEL's expected locations.
                // arglocs[i] = current location of jump arg i
                // loop_arglocs[i] = LABEL's expected location for arg i
                for (i, loc) in arglocs.iter().enumerate() {
                    let dst = if i < self.loop_arglocs.len() {
                        self.loop_arglocs[i]
                    } else {
                        let dst_ofs = crate::regalloc::get_ebp_ofs(0, i);
                        Loc::Frame(crate::regloc::FrameLoc::new(i, dst_ofs, false))
                    };
                    self.regalloc_mov(loc, &dst);
                }
                // Emit jump back to LABEL
                if let Some(label) = self.loop_label {
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64 ; jmp =>label);
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; b =>label);
                } else if let Some(target) = self.jump_target_addr {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let addr = target as i64;
                        dynasm!(self.mc ; .arch x64
                            ; mov rax, QWORD addr
                            ; jmp rax
                        );
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        self.emit_mov_imm64(0, target as i64);
                        dynasm!(self.mc ; .arch aarch64 ; br x0);
                    }
                }
            }
            OpCode::Finish => {
                // RPython: genop_finish stores result at jf_frame[0] (base_ofs),
                // writes descr ptr to jf_descr, then calls _call_footer.
                // arglocs[0] = result location (if any)
                let fail_arg_types = self.infer_fail_arg_types(op);
                let result_type = if fail_arg_types.is_empty() {
                    Type::Void
                } else {
                    fail_arg_types[0]
                };
                let global_descr_ptr =
                    crate::guard::done_with_this_frame_descr_ptr_for_type(result_type) as i64;
                let descr = Arc::new(DynasmFailDescr::new(
                    fail_index,
                    self.trace_id,
                    fail_arg_types.clone(),
                    true,
                ));

                // Store result to jf_frame[0]
                if let Some(result) = arglocs.first() {
                    let slot0 = Loc::Frame(crate::regloc::FrameLoc::new(
                        0,
                        crate::regalloc::get_ebp_ofs(0, 0),
                        result_type == Type::Float,
                    ));
                    self.regalloc_mov(result, &slot0);
                }

                // Store descr ptr to jf_descr [rbp+0]
                #[cfg(target_arch = "x86_64")]
                {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD global_descr_ptr
                        ; mov [rbp], Rq(scratch)
                    );
                }
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(0, global_descr_ptr);
                    dynasm!(self.mc ; .arch aarch64 ; str x0, [x29]);
                }

                self._call_footer();
                self.fail_descrs.push(descr);
            }
            OpCode::Label => {
                let label = self.mc.new_dynamic_label();
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!("[dynasm] LABEL: new DynamicLabel({:?})", label);
                }
                dynasm!(self.mc ; =>label);
                self.loop_label = Some(label);
                self.loop_label_offset = Some(self.mc.offset());
                // Save LABEL's arglocs — JUMP must remap to these positions.
                self.loop_arglocs = arglocs.to_vec();
            }
            // ── Calls ──
            // RPython: regalloc consider_call does before_call (save caller-saved
            // regs), collects arglocs, calls after_call for result. The assembler
            // receives arglocs = [func_addr_or_descr_info..., arg_locs...] and
            // result_loc = register for return value.
            //
            // For now, flush all register-resident values to their frame slots
            // before the call, use the existing frame-slot genop_call, then
            // mark the result in the allocated register.
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilR
            | OpCode::CallReleaseGilF
            | OpCode::CallReleaseGilN => {
                // Save registers to frame before call (regalloc already handled this
                // via before_call / Move ops). Use existing genop_call.
                self.genop_call(op);
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                self.genop_call_assembler(op);
            }
            OpCode::CondCallN => self.genop_discard_cond_call(op),
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.genop_cond_call_value(op);
            }
            // ── Allocation ──
            OpCode::New => self.genop_new(op),
            OpCode::NewWithVtable => self.genop_new_with_vtable(op),
            OpCode::NewArray | OpCode::NewArrayClear => self.genop_new_array(op),
            OpCode::Newstr => self.genop_newstr(op),
            OpCode::Newunicode => self.genop_newunicode(op),
            // ── Misc ──
            OpCode::ForceToken => {
                if let Some(Loc::Reg(r)) = result_loc {
                    #[cfg(target_arch = "x86_64")]
                    dynasm!(self.mc ; .arch x64 ; mov Rq(r.value), rbp);
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(self.mc ; .arch aarch64 ; mov X(r.value), x29);
                }
            }
            OpCode::SaveException => self.genop_save_exception(op),
            OpCode::SaveExcClass => self.genop_save_exc_class(op),
            // ── Guards (via regalloc_perform_guard, shouldn't reach here) ──
            _ if op.opcode.is_guard() => {
                // Guards should go through regalloc_perform_guard, not here.
                // Fallback: use existing guard implementation.
                self.implement_guard_nojump(op, fail_index);
            }
            // ── No-ops ──
            _ => {}
        }
    }

    /// assembler.py:329 regalloc_perform_guard — emit guard with faillocs.
    fn regalloc_perform_guard(
        &mut self,
        op: &Op,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        faillocs: &[Option<Loc>],
        fail_index: u32,
    ) {
        match op.opcode {
            OpCode::GuardTrue | OpCode::VecGuardTrue | OpCode::GuardNonnull => {
                // arglocs[0] = condition location
                if let Some(loc) = arglocs.first() {
                    self.emit_test_loc(loc);
                    self.guard_success_cc = Some(CC_NE);
                }
                self.implement_guard_with_faillocs(op, fail_index, faillocs);
            }
            OpCode::GuardFalse | OpCode::VecGuardFalse | OpCode::GuardIsnull => {
                if let Some(loc) = arglocs.first() {
                    self.emit_test_loc(loc);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, fail_index, faillocs);
            }
            OpCode::GuardValue => {
                if arglocs.len() >= 2 {
                    self.emit_cmp_loc_loc(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, fail_index, faillocs);
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass | OpCode::GuardGcType => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, fail_index, faillocs);
            }
            OpCode::GuardNoException | OpCode::GuardNoOverflow | OpCode::GuardOverflow => {
                self.implement_guard_nojump_with_faillocs(op, fail_index, faillocs);
            }
            OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64 ; cmp QWORD [rbp], 0);
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64 ; ldr X(16), [x29] ; cmp X(16), xzr);
                self.guard_success_cc = Some(CC_E);
                self.implement_guard_with_faillocs(op, fail_index, faillocs);
            }
            OpCode::GuardNotInvalidated => {
                self.implement_guard_nojump_with_faillocs(op, fail_index, faillocs);
            }
            _ => {
                self.implement_guard_nojump_with_faillocs(op, fail_index, faillocs);
            }
        }
    }

    /// Helper: guard class comparison
    fn _cmp_guard_class(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        if let Loc::Reg(obj) = obj_loc {
            // Load typeptr/class into scratch register
            #[cfg(target_arch = "x86_64")]
            {
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                if let Some(vtable_offset) = self.vtable_offset {
                    let ofs = vtable_offset as i32;
                    dynasm!(self.mc ; .arch x64 ; mov Rq(scratch), [Rq(obj.value) + ofs]);
                } else {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(scratch), [Rq(obj.value)]);
                }
                match class_loc {
                    Loc::Reg(c) => {
                        dynasm!(self.mc ; .arch x64 ; cmp Rq(scratch), Rq(c.value));
                    }
                    Loc::Immed(i) => {
                        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD i.value ; cmp Rq(scratch), rax);
                    }
                    _ => {}
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(vtable_offset) = self.vtable_offset {
                    let ofs = vtable_offset as u32;
                    dynasm!(self.mc ; .arch aarch64 ; ldr x16, [X(obj.value), ofs]);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; ldr x16, [X(obj.value)]);
                }
                // Compare scratch (x16) with expected class
                self.regalloc_mov(class_loc, &Loc::Reg(crate::regloc::RegLoc::new(17, false)));
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            }
        }
    }

    /// Emit SETcc into a register (zero-extend to 64-bit).
    fn emit_setcc(&mut self, cc: u8, dst_reg: u8) {
        #[cfg(target_arch = "x86_64")]
        {
            dynasm!(self.mc ; .arch x64 ; xor Rd(dst_reg), Rd(dst_reg));
            match cc {
                CC_E => {
                    dynasm!(self.mc ; .arch x64 ; sete  Rb(dst_reg));
                }
                CC_NE => {
                    dynasm!(self.mc ; .arch x64 ; setne Rb(dst_reg));
                }
                CC_L => {
                    dynasm!(self.mc ; .arch x64 ; setl  Rb(dst_reg));
                }
                CC_GE => {
                    dynasm!(self.mc ; .arch x64 ; setge Rb(dst_reg));
                }
                CC_LE => {
                    dynasm!(self.mc ; .arch x64 ; setle Rb(dst_reg));
                }
                CC_G => {
                    dynasm!(self.mc ; .arch x64 ; setg  Rb(dst_reg));
                }
                CC_B => {
                    dynasm!(self.mc ; .arch x64 ; setb  Rb(dst_reg));
                }
                CC_AE => {
                    dynasm!(self.mc ; .arch x64 ; setae Rb(dst_reg));
                }
                CC_BE => {
                    dynasm!(self.mc ; .arch x64 ; setbe Rb(dst_reg));
                }
                CC_A => {
                    dynasm!(self.mc ; .arch x64 ; seta  Rb(dst_reg));
                }
                CC_S => {
                    dynasm!(self.mc ; .arch x64 ; sets  Rb(dst_reg));
                }
                CC_NS => {
                    dynasm!(self.mc ; .arch x64 ; setns Rb(dst_reg));
                }
                CC_O => {
                    dynasm!(self.mc ; .arch x64 ; seto  Rb(dst_reg));
                }
                CC_NO => {
                    dynasm!(self.mc ; .arch x64 ; setno Rb(dst_reg));
                }
                _ => {
                    dynasm!(self.mc ; .arch x64 ; sete  Rb(dst_reg));
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            match cc {
                CC_E => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), eq);
                }
                CC_NE => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ne);
                }
                CC_L => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), lt);
                }
                CC_GE => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ge);
                }
                CC_LE => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), le);
                }
                CC_G => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), gt);
                }
                CC_B => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), lo);
                }
                CC_AE => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), hs);
                }
                CC_BE => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ls);
                }
                CC_A => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), hi);
                }
                CC_S => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), mi);
                }
                CC_NS => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), pl);
                }
                CC_O => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), vs);
                }
                CC_NO => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), vc);
                }
                _ => {
                    dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), eq);
                }
            }
        }
    }

    /// Map an integer comparison OpCode to a condition code.
    fn opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::IntLt => CC_L,
            OpCode::IntLe => CC_LE,
            OpCode::IntGt => CC_G,
            OpCode::IntGe => CC_GE,
            OpCode::IntEq | OpCode::PtrEq | OpCode::InstancePtrEq => CC_E,
            OpCode::IntNe | OpCode::PtrNe | OpCode::InstancePtrNe => CC_NE,
            OpCode::UintLt => CC_B,
            OpCode::UintLe => CC_BE,
            OpCode::UintGt => CC_A,
            OpCode::UintGe => CC_AE,
            _ => CC_E,
        }
    }

    /// Map a float comparison OpCode to a condition code (after ucomisd).
    fn float_opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::FloatLt => CC_B,  // ucomisd: below = less than
            OpCode::FloatLe => CC_BE, // below or equal
            OpCode::FloatGt => CC_A,  // above
            OpCode::FloatGe => CC_AE, // above or equal
            OpCode::FloatEq => CC_E,  // equal
            OpCode::FloatNe => CC_NE, // not equal
            _ => CC_E,
        }
    }

    /// Guard with faillocs — emit conditional jump and store faillocs on descr.
    fn implement_guard_with_faillocs(
        &mut self,
        op: &Op,
        fail_index: u32,
        faillocs: &[Option<Loc>],
    ) {
        let cc = self
            .guard_success_cc
            .take()
            .expect("implement_guard_with_faillocs: guard_success_cc not set");
        let fail_cc = invert_cc(cc);
        let fail_label = self.emit_guard_jcc(fail_cc);
        self.append_guard_token_with_faillocs(op, fail_index, fail_label, faillocs);
    }

    /// Guard no-jump with faillocs.
    fn implement_guard_nojump_with_faillocs(
        &mut self,
        op: &Op,
        fail_index: u32,
        faillocs: &[Option<Loc>],
    ) {
        let fail_label = self.mc.new_dynamic_label();
        self.append_guard_token_with_faillocs(op, fail_index, fail_label, faillocs);
    }

    /// Append guard token with regalloc faillocs instead of opref_to_slot snapshot.
    fn append_guard_token_with_faillocs(
        &mut self,
        op: &Op,
        fail_index: u32,
        fail_label: DynamicLabel,
        faillocs: &[Option<Loc>],
    ) {
        let fail_arg_types = self.infer_fail_arg_types(op);
        let descr = Arc::new(DynasmFailDescr::new(
            fail_index,
            self.trace_id,
            fail_arg_types,
            false,
        ));

        // Convert regalloc faillocs to frame slot indices for the descr.
        // Reg locs: register save area at JITFRAME_FIXED_SIZE + reg_value.
        // Frame locs: use position directly.
        // Immed locs: allocate a frame slot and store the constant in the
        // recovery stub. RPython uses rd_locs with JITFRAME encoding.
        let mut const_stores: Vec<(usize, i64)> = Vec::new();
        let fail_arg_locs: Vec<Option<usize>> = faillocs
            .iter()
            .map(|fl| match fl {
                Some(Loc::Reg(r)) => Some(JITFRAME_FIXED_SIZE + r.value as usize),
                Some(Loc::Frame(f)) => Some(f.position),
                Some(Loc::Immed(i)) => {
                    // Allocate a slot for this constant in the save area.
                    let slot = self.frame_depth;
                    self.frame_depth += 1;
                    const_stores.push((slot, i.value));
                    Some(slot)
                }
                _ => None,
            })
            .collect();
        unsafe {
            let descr_mut = &mut *(Arc::as_ptr(&descr) as *mut DynasmFailDescr);
            descr_mut.fail_arg_locs = fail_arg_locs;
        }

        self.pending_guard_tokens.push(GuardToken {
            jump_offset: self.mc.offset(),
            fail_label,
            fail_descr: descr.clone(),
            fail_args: op
                .fail_args
                .as_ref()
                .map(|fa| fa.to_vec())
                .unwrap_or_default(),
            opref_to_slot_snapshot: self.opref_to_slot.clone(),
            const_stores,
        });
        self.fail_descrs.push(descr);
    }

    /// Update fail_arg_locs on all pending guard descriptors.
    /// Unmapped (virtual/dead) OpRefs get None — the resume system
    /// handles them via rd_numb TAGVIRTUAL/TAGCONST encoding.
    fn allocate_unmapped_fail_arg_slots(&mut self) {
        for gt in &self.pending_guard_tokens {
            let locs: Vec<Option<usize>> = gt
                .fail_args
                .iter()
                .map(|opref| {
                    if opref.is_none() || opref.is_constant() {
                        None
                    } else {
                        self.opref_to_slot.get(&opref.0).copied()
                    }
                })
                .collect();
            unsafe {
                let descr_mut = &mut *(Arc::as_ptr(&gt.fail_descr) as *mut DynasmFailDescr);
                descr_mut.fail_arg_locs = locs;
            }
        }
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
    /// assembler.py:835 write_pending_failure_recoveries +
    /// assembler.py:849 patch_pending_failure_recoveries.
    /// Returns recovery stub offsets for post-finalize address fixup.
    fn write_pending_failure_recoveries(&mut self) -> Vec<(Arc<DynasmFailDescr>, usize)> {
        // Emit a shared _push_all_regs_to_frame routine once, then each
        // recovery stub calls it. This keeps stubs small (call + descr + ret)
        // and avoids aarch64 branch range issues.
        let save_regs_label = self.mc.new_dynamic_label();
        #[cfg(target_arch = "x86_64")]
        {
            dynasm!(self.mc ; .arch x64 ; =>save_regs_label);
            use crate::regloc::*;
            let gprs = [
                ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
            ];
            for &reg in &gprs {
                let save_slot = JITFRAME_FIXED_SIZE + reg.value as usize;
                let ofs = ((1 + save_slot) * 8) as i32;
                dynasm!(self.mc ; .arch x64 ; mov [rbp + ofs], Rq(reg.value));
            }
            let xmms = [
                XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12,
                XMM13, XMM14,
            ];
            for &reg in &xmms {
                let save_slot = JITFRAME_FIXED_SIZE + reg.value as usize;
                let ofs = ((1 + save_slot) * 8) as i32;
                dynasm!(self.mc ; .arch x64 ; movsd [rbp + ofs], Rx(reg.value));
            }
            dynasm!(self.mc ; .arch x64 ; ret);
        }
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(self.mc ; .arch aarch64 ; =>save_regs_label);
            use crate::regloc::*;
            let gprs = [
                ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
            ];
            for &reg in &gprs {
                let save_slot = JITFRAME_FIXED_SIZE + reg.value as usize;
                let ofs = ((1 + save_slot) * 8) as u32;
                dynasm!(self.mc ; .arch aarch64 ; str X(reg.value), [x29, ofs]);
            }
            dynasm!(self.mc ; .arch aarch64 ; ret);
        }

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] write_pending_failure_recoveries: {} tokens",
                self.pending_guard_tokens.len()
            );
        }
        let mut stub_offsets = Vec::new();
        for guard_token in std::mem::take(&mut self.pending_guard_tokens) {
            let stub_start = self.mc.offset();

            // Bind the fail_label — guard's Jcc lands here.
            let fail_label = guard_token.fail_label;
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!("[dynasm] recovery stub: binding {:?}", fail_label);
            }
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64 ; =>fail_label);
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64 ; =>fail_label);

            // Call shared _push_all_regs_to_frame
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64 ; call =>save_regs_label);
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64 ; bl =>save_regs_label);

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

            // Store constant fail_args into their allocated frame slots.
            // These are values that were Loc::Immed in faillocs — they need
            // to be written to the frame since they don't live in registers.
            for &(slot, val) in &guard_token.const_stores {
                let ofs = ((1 + slot) * 8) as i32;
                #[cfg(target_arch = "x86_64")]
                {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD val
                        ; mov [rbp + ofs], Rq(scratch)
                    );
                }
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(16, val);
                    dynasm!(self.mc ; .arch aarch64 ; str x16, [x29, ofs as u32]);
                }
            }

            // assembler.py:2112 _call_footer — return jf_ptr
            self._call_footer();

            stub_offsets.push((guard_token.fail_descr, stub_start.0));
        }
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[dynasm] write_pending done: {} stubs", stub_offsets.len());
        }
        stub_offsets
    }

    /// assembler.py:849 patch_pending_failure_recoveries — convert
    /// buffer-relative offsets to absolute addresses after finalize.
    fn patch_pending_failure_recoveries(
        rawstart: usize,
        stub_offsets: &[(Arc<DynasmFailDescr>, usize)],
    ) {
        for (descr, stub_offset) in stub_offsets {
            let abs_addr = rawstart + stub_offset;
            descr.set_adr_jump_offset(abs_addr);
        }
    }

    // ----------------------------------------------------------------
    // assembler.py:965-987 patch_jump_for_descr
    // ----------------------------------------------------------------

    /// assembler.py:965 patch_jump_for_descr: redirect a guard to a
    /// bridge by overwriting the recovery stub with a JMP to bridge.
    ///
    /// `adr_jump_offset` is the absolute address of the recovery stub
    /// (set by patch_pending_failure_recoveries). We overwrite the
    /// stub with "MOV r11, bridge_addr; JMP r11" (x64) or "B imm26"
    /// (aarch64).
    pub fn patch_jump_for_descr(descr: &DynasmFailDescr, adr_new_target: usize) {
        let stub_addr = descr.adr_jump_offset();
        assert!(stub_addr != 0, "guard already patched");

        codebuf::with_writable(stub_addr as *mut u8, 16, || {
            #[cfg(target_arch = "x86_64")]
            {
                let stub_ptr = stub_addr as *mut u8;
                let offset = adr_new_target as isize - (stub_addr as isize + 5);
                if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
                    unsafe {
                        *stub_ptr = 0xE9;
                        (stub_ptr.add(1) as *mut i32).write(offset as i32);
                    }
                } else {
                    unsafe {
                        *stub_ptr = 0x49;
                        *stub_ptr.add(1) = 0xBB;
                        (stub_ptr.add(2) as *mut u64).write(adr_new_target as u64);
                        *stub_ptr.add(10) = 0x41;
                        *stub_ptr.add(11) = 0xFF;
                        *stub_ptr.add(12) = 0xE3;
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                let offset = adr_new_target as isize - stub_addr as isize;
                let imm26 = ((offset >> 2) & 0x03FF_FFFF) as u32;
                let insn = 0x1400_0000 | imm26;
                unsafe { (stub_addr as *mut u32).write(insn) };
            }
        });

        #[cfg(target_arch = "aarch64")]
        flush_icache(stub_addr as *const u8, 16);

        // assembler.py:987
        descr.set_adr_jump_offset(0); // "patched"
    }

    /// assembler.py:1138 redirect_call_assembler: patch old loop entry
    /// to JMP to new loop after retrace.
    pub fn redirect_call_assembler(old_addr: *const u8, new_addr: *const u8) {
        codebuf::with_writable(old_addr as *mut u8, 16, || {
            #[cfg(target_arch = "x86_64")]
            {
                let old_ptr = old_addr as *mut u8;
                let offset = new_addr as isize - (old_addr as isize + 5);
                if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
                    unsafe {
                        *old_ptr = 0xE9;
                        (old_ptr.add(1) as *mut i32).write(offset as i32);
                    }
                } else {
                    unsafe {
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
                let offset = new_addr as isize - old_addr as isize;
                let imm26 = ((offset >> 2) & 0x03FF_FFFF) as u32;
                let insn = 0x1400_0000 | imm26;
                unsafe { (old_addr as *mut u32).write(insn) };
            }
        });

        #[cfg(target_arch = "aarch64")]
        flush_icache(old_addr, 4);
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

    /// assembler.py:1856 genop_int_add_ovf — delegates to genop_int_add,
    /// then sets guard_success_cc = 'NO'. On x86, ADD always sets OF.
    fn genop_int_add_ovf(&mut self, op: &Op) {
        #[cfg(target_arch = "x86_64")]
        self.genop_int_add(op); // ADD sets OF on x86
        #[cfg(target_arch = "aarch64")]
        {
            self.load_arg_to_rax(op.arg(0));
            self.load_arg_to_rcx(op.arg(1));
            dynasm!(self.mc ; .arch aarch64 ; adds x0, x0, x1);
            self.store_rax_to_result(op.pos);
        }
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py:1860 genop_int_sub_ovf.
    fn genop_int_sub_ovf(&mut self, op: &Op) {
        #[cfg(target_arch = "x86_64")]
        self.genop_int_sub(op);
        #[cfg(target_arch = "aarch64")]
        {
            self.load_arg_to_rax(op.arg(0));
            self.load_arg_to_rcx(op.arg(1));
            dynasm!(self.mc ; .arch aarch64 ; subs x0, x0, x1);
            self.store_rax_to_result(op.pos);
        }
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py:1864 genop_int_mul_ovf.
    fn genop_int_mul_ovf(&mut self, op: &Op) {
        #[cfg(target_arch = "x86_64")]
        self.genop_int_mul(op); // IMUL sets OF on x86
        #[cfg(target_arch = "aarch64")]
        {
            // ARM64: no direct overflow flag from MUL.
            // Use SMULH to get high bits, compare with sign extension.
            self.load_arg_to_rax(op.arg(0));
            self.load_arg_to_rcx(op.arg(1));
            dynasm!(self.mc ; .arch aarch64
                ; mul x2, x0, x1
                ; smulh x3, x0, x1
                ; asr x4, x2, 63
                ; cmp x3, x4
                ; mov x0, x2
            );
            self.store_rax_to_result(op.pos);
        }
        self.guard_success_cc = Some(CC_NO);
    }

    // ----------------------------------------------------------------
    // genop_* — comparisons
    // ----------------------------------------------------------------

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
    fn _cmp_guard_class_old(&mut self, expected_classptr_imm: Option<i64>) {
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
            let expected_typeid = self.lookup_typeid_from_classptr(classptr as usize).expect(
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
    /// Looks up the materialized table populated by the runner from
    /// the active gc_ll_descr. RPython resolves the same value via
    /// `cpu.gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr`.
    fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        self.classptr_to_typeid.get(&(classptr as i64)).copied()
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
        // aarch64: b.cond has 19-bit range (±1MB), which is too short
        // for forward references to recovery stubs. Use inverted condition
        // + unconditional branch (26-bit / ±128MB) pattern instead:
        //   b.NOT_cond >skip ; b =>fail_label ; skip:
        #[cfg(target_arch = "aarch64")]
        {
            let skip = self.mc.new_dynamic_label();
            let inv = invert_cc(fail_cc);
            match inv {
                CC_L => dynasm!(self.mc ; .arch aarch64 ; b.lt =>skip),
                CC_LE => dynasm!(self.mc ; .arch aarch64 ; b.le =>skip),
                CC_G => dynasm!(self.mc ; .arch aarch64 ; b.gt =>skip),
                CC_GE => dynasm!(self.mc ; .arch aarch64 ; b.ge =>skip),
                CC_E => dynasm!(self.mc ; .arch aarch64 ; b.eq =>skip),
                CC_NE => dynasm!(self.mc ; .arch aarch64 ; b.ne =>skip),
                CC_B => dynasm!(self.mc ; .arch aarch64 ; b.lo =>skip),
                CC_BE => dynasm!(self.mc ; .arch aarch64 ; b.ls =>skip),
                CC_A => dynasm!(self.mc ; .arch aarch64 ; b.hi =>skip),
                CC_AE => dynasm!(self.mc ; .arch aarch64 ; b.hs =>skip),
                CC_O => dynasm!(self.mc ; .arch aarch64 ; b.vs =>skip),
                CC_NO => dynasm!(self.mc ; .arch aarch64 ; b.vc =>skip),
                CC_S => dynasm!(self.mc ; .arch aarch64 ; b.mi =>skip),
                CC_NS => dynasm!(self.mc ; .arch aarch64 ; b.pl =>skip),
                _ => dynasm!(self.mc ; .arch aarch64 ; b.eq =>skip),
            }
            dynasm!(self.mc ; .arch aarch64 ; b =>fail_label);
            dynasm!(self.mc ; .arch aarch64 ; =>skip);
        }
        fail_label
    }

    /// assembler.py:2157 implement_guard — emit conditional jump to
    /// failure stub and append guard token to pending list.
    fn implement_guard(&mut self, op: &Op, fail_index: u32) {
        let cc = self
            .guard_success_cc
            .take()
            .expect("implement_guard: guard_success_cc not set");
        let fail_cc = invert_cc(cc);
        let fail_label = self.emit_guard_jcc(fail_cc);
        // guard_success_cc is already None after .take()

        self.append_guard_token(op, fail_index, fail_label);
    }

    /// Register a guard token without emitting a conditional jump.
    /// assembler.py:1799-1806 genop_guard_guard_not_invalidated pattern.
    fn implement_guard_nojump(&mut self, op: &Op, fail_index: u32) {
        let fail_label = self.mc.new_dynamic_label();
        self.append_guard_token(op, fail_index, fail_label);
    }

    /// Infer fail_arg_types from value_types or op.fail_arg_types.
    fn infer_fail_arg_types(&self, op: &Op) -> Vec<Type> {
        if let Some(ref ts) = op.fail_arg_types {
            ts.clone()
        } else if let Some(ref fa) = op.fail_args {
            fa.iter()
                .map(|opref| {
                    if opref.is_none() {
                        Type::Ref
                    } else {
                        self.value_types.get(&opref.0).copied().unwrap_or(Type::Ref)
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Common tail: build DynasmFailDescr, create GuardToken, push to
    /// pending_guard_tokens and fail_descrs.
    fn append_guard_token(&mut self, op: &Op, fail_index: u32, fail_label: DynamicLabel) {
        let fail_arg_types = self.infer_fail_arg_types(op);
        let fail_args = op
            .fail_args
            .as_ref()
            .map(|fa| fa.to_vec())
            .unwrap_or_default();
        // assembler.py:238-270 store_info_on_descr parity: fail_arg_locs
        // maps each fail_arg to its jitframe slot. None = 0xFFFF (virtual).
        let fail_arg_locs: Vec<Option<usize>> = fail_args
            .iter()
            .map(|opref| {
                if opref.is_none() || opref.is_constant() {
                    None
                } else {
                    self.opref_to_slot.get(&opref.0).copied()
                }
            })
            .collect();
        let mut descr = DynasmFailDescr::new(fail_index, self.trace_id, fail_arg_types, false);
        descr.fail_arg_locs = fail_arg_locs;
        let descr = Arc::new(descr);
        let jump_offset = self.mc.offset();
        self.pending_guard_tokens.push(GuardToken {
            jump_offset,
            fail_label,
            fail_descr: descr.clone(),
            fail_args,
            opref_to_slot_snapshot: self.opref_to_slot.clone(),
            const_stores: Vec::new(),
        });
        self.fail_descrs.push(descr);
    }

    // ----------------------------------------------------------------
    // assembler.py:1773 genop_guard_guard_true
    // ----------------------------------------------------------------

    /// regalloc.py:429 load_condition_into_cc — if guard_success_cc
    /// is not yet set, emit TEST arg0 and set cc = NZ.
    fn load_condition_into_cc(&mut self, op: &Op) {
        if self.guard_success_cc.is_none() {
            self.load_arg_to_rax(op.arg(0));
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64 ; test rax, rax);
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64 ; cmp x0, 0);
            self.guard_success_cc = Some(CC_NE); // rx86.Conditions['NZ']
        }
    }

    /// assembler.py:1773 genop_guard_guard_true.
    /// genop_guard_guard_nonnull = genop_guard_guard_true (alias)
    /// genop_guard_guard_no_overflow = genop_guard_guard_true (alias)
    fn genop_guard_guard_true(&mut self, op: &Op, fail_index: u32) {
        self.load_condition_into_cc(op);
        self.implement_guard(op, fail_index);
    }

    /// assembler.py:1777 genop_guard_guard_false.
    /// genop_guard_guard_isnull = genop_guard_guard_false (alias)
    /// genop_guard_guard_overflow = genop_guard_guard_false (alias)
    fn genop_guard_guard_false(&mut self, op: &Op, fail_index: u32) {
        self.load_condition_into_cc(op);
        self.guard_success_cc = Some(invert_cc(self.guard_success_cc.take().unwrap()));
        self.implement_guard(op, fail_index);
    }

    /// assembler.py:1871 genop_guard_guard_value.
    fn genop_guard_guard_value(&mut self, op: &Op, fail_index: u32) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; cmp rax, rcx);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; cmp x0, x1);

        self.guard_success_cc = Some(CC_E);
        self.implement_guard(op, fail_index);

        // regalloc.py:496-501 make_a_counter_per_value
        if let Some(descr) = self.fail_descrs.last() {
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
    }

    /// assembler.py:1903 genop_guard_guard_class.
    fn genop_guard_guard_class(&mut self, op: &Op, fail_index: u32) {
        let expected_classptr_imm = match self.resolve_opref(op.arg(1)) {
            ResolvedArg::Const(v) => Some(v),
            _ => None,
        };
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        self._cmp_guard_class_old(expected_classptr_imm);
        self.guard_success_cc = Some(CC_E);
        self.implement_guard(op, fail_index);
    }

    /// assembler.py:1908 genop_guard_guard_nonnull_class.
    /// CMP(locs[0], imm1); JB → fail; _cmp_guard_class; JNE → fail
    fn genop_guard_guard_nonnull_class(&mut self, op: &Op, fail_index: u32) {
        self.load_arg_to_rax(op.arg(0));
        // assembler.py:1909 CMP(locs[0], imm1) — JB catches NULL (0)
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; cmp rax, 1);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; cmp x0, 1);

        // assembler.py:1911 emit_forward_jump('B') — jump if below (NULL)
        let fail_label = self.emit_guard_jcc(CC_B);

        let expected_classptr_imm = match self.resolve_opref(op.arg(1)) {
            ResolvedArg::Const(v) => Some(v),
            _ => None,
        };
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        self._cmp_guard_class_old(expected_classptr_imm);

        // assembler.py:1914 patch_forward_jump — both paths share the
        // same fail_label, so a second JNE to the same target suffices.
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; jne =>fail_label);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; b.ne =>fail_label);

        // assembler.py:1916-1917
        // guard_success_cc is not used here — we already emitted the jumps.
        // Manually append the guard token.
        self.append_guard_token(op, fail_index, fail_label);
    }

    /// assembler.py:1782 genop_guard_guard_no_exception.
    /// Stub: no exception tracking yet.
    fn genop_guard_guard_no_exception(&mut self, op: &Op, fail_index: u32) {
        self.implement_guard_nojump(op, fail_index);
    }

    /// assembler.py:1799 genop_guard_guard_not_invalidated.
    /// Does NOT emit a conditional jump — records position for external
    /// invalidation patching.
    fn genop_guard_guard_not_invalidated(&mut self, op: &Op, fail_index: u32) {
        self.implement_guard_nojump(op, fail_index);
    }

    /// assembler.py:2207-2222 _store_force_index: before a call that may force,
    /// zero jf_descr so GUARD_NOT_FORCED's CMP [jf_descr], 0 starts clean.
    /// RPython stores the guard descr pointer to jf_force_descr; in our
    /// frame layout jf_descr is used directly (zeroed here, set by force).
    fn _store_force_index_if_next_guard(&mut self, ops: &[Op], op_idx: usize, _fail_index: u32) {
        // assembler.py:2224-2226 _find_nearby_operation(+1)
        let next_idx = op_idx + 1;
        if next_idx >= ops.len() {
            return;
        }
        let next_op = &ops[next_idx];
        if next_op.opcode != OpCode::GuardNotForced && next_op.opcode != OpCode::GuardNotForced2 {
            return;
        }
        // Zero jf_descr before the call so GUARD_NOT_FORCED sees 0 (not forced).
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; mov QWORD [rbp], 0
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; str xzr, [x29]
        );
    }

    /// assembler.py:2228-2232 genop_guard_guard_not_forced.
    /// Check jf_descr == 0 (not forced); if non-zero, guard fails.
    fn genop_guard_guard_not_forced(&mut self, op: &Op, fail_index: u32) {
        // assembler.py:2229-2231: CMP [jf_descr], 0; guard_success_cc = 'E'
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; cmp QWORD [rbp], 0           // jf_descr == 0?
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x29]               // x0 = jf_descr
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_E); // guard succeeds if equal (not forced)
        self.implement_guard(op, fail_index);
    }

    // ----------------------------------------------------------------
    // genop_* — control flow
    // ----------------------------------------------------------------

    /// LABEL: define the back-edge target for JUMP.
    ///
    /// RPython: LABEL does NOT emit code. The regalloc establishes
    /// the slot mapping. JUMP handles slot remapping.
    ///
    /// In our frame-slot model: preamble values may be in non-canonical
    /// slots. We emit copies from old→canonical slots BEFORE the
    /// LABEL binding, so they execute only on first entry from the
    /// preamble. JUMP writes directly to canonical slots and jumps
    /// to the LABEL, skipping the copies.
    fn genop_label(&mut self, op: &Op) {
        // Emit preamble→canonical copies BEFORE the label.
        // Two-pass push/pop: safely handles slot overlaps.
        let n_label = op.args.len();
        // Pass 1: push source values
        for i in 0..n_label {
            let arg_ref = op.args[i];
            if arg_ref.is_none() {
                let dst = Self::slot_offset(i);
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64 ; push QWORD [rbp + dst]);
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x29, dst as u32] ; str x0, [sp, #-16]!);
            } else if arg_ref.is_constant() {
                let val = self.constants.get(&arg_ref.0).copied().unwrap_or(0);
                #[cfg(target_arch = "x86_64")]
                {
                    dynasm!(self.mc ; .arch x64 ; mov rax, QWORD val as i64 ; push rax);
                }
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(0, val);
                    dynasm!(self.mc ; .arch aarch64 ; str x0, [sp, #-16]!);
                }
            } else if let Some(&old_slot) = self.opref_to_slot.get(&arg_ref.0) {
                let src = Self::slot_offset(old_slot);
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64 ; push QWORD [rbp + src]);
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x29, src as u32] ; str x0, [sp, #-16]!);
            } else {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64 ; push 0);
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [sp, #-16]!);
            }
        }
        // Pass 2: pop in reverse into canonical slots
        for i in (0..n_label).rev() {
            let dst = Self::slot_offset(i);
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64 ; pop QWORD [rbp + dst]);
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64 ; ldr x0, [sp], #16 ; str x0, [x29, dst as u32]);
        }

        // Bind the LABEL — JUMP targets here (after the copies).
        let label = self.mc.new_dynamic_label();
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; =>label);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; =>label);
        self.loop_label = Some(label);
        self.loop_label_offset = Some(self.mc.offset());

        // Remap: Label's arg[i] → canonical slot i
        for (i, &arg_ref) in op.args.iter().enumerate() {
            if !arg_ref.is_none() {
                self.opref_to_slot.insert(arg_ref.0, i);
            }
        }
        self.next_slot = self.next_slot.max(op.args.len());
    }

    /// jump.py:66 _move: emit a single slot-to-slot or const-to-slot move.
    fn emit_slot_move(&mut self, src: i32, dst: i32, is_const: bool, val: i64) {
        if is_const {
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64
                ; mov rax, QWORD val
                ; mov [rbp + dst], rax
            );
            #[cfg(target_arch = "aarch64")]
            {
                self.emit_mov_imm64(0, val);
                dynasm!(self.mc ; .arch aarch64 ; str x0, [x29, dst as u32]);
            }
        } else if src != dst {
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64
                ; mov rax, [rbp + src]
                ; mov [rbp + dst], rax
            );
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x29, src as u32]
                ; str x0, [x29, dst as u32]
            );
        }
    }

    /// JUMP: unconditional branch to the loop label.
    /// jump.py:1 remap_frame_layout parity: parallel move algorithm
    /// to handle cyclic slot dependencies.
    fn genop_jump(&mut self, op: &Op) {
        // Build src→dst move list.
        // Each entry: (src_offset_or_const, dst_offset, is_const, const_val)
        let n = op.args.len();
        let mut moves: Vec<(i32, i32, bool, i64)> = Vec::with_capacity(n);
        for (i, &arg_ref) in op.args.iter().enumerate() {
            let dst = Self::slot_offset(i);
            match self.resolve_opref(arg_ref) {
                ResolvedArg::Slot(src) => moves.push((src, dst, false, 0)),
                ResolvedArg::Const(val) => moves.push((0, dst, true, val)),
            }
        }

        // jump.py:1-64 remap_frame_layout: topological order with
        // cycle breaking via push/pop.
        // srccount[dst] = number of times dst appears as a src
        let mut srccount: HashMap<i32, i32> = HashMap::new();
        for m in &moves {
            srccount.entry(m.1).or_insert(0); // ensure dst exists
        }
        let mut pending = n as i32;
        for (i, m) in moves.iter().enumerate() {
            if m.2 {
                continue;
            } // constant → no src dependency
            let src = m.0;
            if let Some(cnt) = srccount.get_mut(&src) {
                if src == moves[i].1 {
                    // self-move: skip
                    *cnt = -(n as i32) - 1;
                    pending -= 1;
                } else {
                    *cnt += 1;
                }
            }
        }

        while pending > 0 {
            let mut progress = false;
            for i in 0..n {
                let dst = moves[i].1;
                if srccount.get(&dst).copied().unwrap_or(-1) == 0 {
                    *srccount.get_mut(&dst).unwrap() = -1; // done
                    pending -= 1;
                    if !moves[i].2 {
                        let src = moves[i].0;
                        if let Some(cnt) = srccount.get_mut(&src) {
                            *cnt -= 1;
                        }
                    }
                    self.emit_slot_move(moves[i].0, dst, moves[i].2, moves[i].3);
                    progress = true;
                }
            }
            if !progress {
                // Cycle: use push/pop to break it.
                for i in 0..n {
                    let dst = moves[i].1;
                    if srccount.get(&dst).copied().unwrap_or(-1) >= 0 {
                        // Push first dst in the cycle
                        #[cfg(target_arch = "x86_64")]
                        dynasm!(self.mc ; .arch x64 ; push QWORD [rbp + dst]);
                        #[cfg(target_arch = "aarch64")]
                        {
                            dynasm!(self.mc ; .arch aarch64
                                ; ldr x0, [x29, dst as u32]
                                ; str x0, [sp, #-16]!
                            );
                        }
                        // Walk the cycle
                        let mut cur = i;
                        loop {
                            let cd = moves[cur].1;
                            *srccount.get_mut(&cd).unwrap() = -1;
                            pending -= 1;
                            // Find the move whose dst is this src
                            let src = moves[cur].0;
                            let next = moves.iter().position(|m| m.1 == src);
                            if let Some(ni) = next {
                                if srccount.get(&moves[ni].1).copied().unwrap_or(-1) < 0 {
                                    // End of cycle: pop into this slot
                                    #[cfg(target_arch = "x86_64")]
                                    dynasm!(self.mc ; .arch x64 ; pop QWORD [rbp + cd]);
                                    #[cfg(target_arch = "aarch64")]
                                    {
                                        dynasm!(self.mc ; .arch aarch64
                                            ; ldr x0, [sp], #16
                                            ; str x0, [x29, cd as u32]
                                        );
                                    }
                                    break;
                                }
                                self.emit_slot_move(src, cd, false, 0);
                                cur = ni;
                            } else {
                                // No cycle found — emit move and break
                                self.emit_slot_move(moves[cur].0, cd, moves[cur].2, moves[cur].3);
                                break;
                            }
                        }
                    }
                }
            }
        }

        if let Some(label) = self.loop_label {
            // Same-buffer jump (loop body)
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64 ; jmp =>label);
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64 ; b =>label);
        } else if let Some(target) = self.jump_target_addr {
            // assembler.py closing_jump parity: bridge jumps back to
            // the original loop's LABEL via absolute address.
            #[cfg(target_arch = "x86_64")]
            {
                let addr = target as i64;
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD addr
                    ; jmp rax
                );
            }
            #[cfg(target_arch = "aarch64")]
            {
                self.emit_mov_imm64(0, target as i64);
                dynasm!(self.mc ; .arch aarch64 ; br x0);
            }
        }
    }

    /// FINISH: store result (if any), store descr ptr, return jf_ptr.
    fn genop_finish(&mut self, op: &Op, fail_index: u32) {
        // Infer fail_arg_types for FINISH result (same as guard inference).
        let fail_arg_types = if let Some(ref ts) = op.fail_arg_types {
            ts.clone()
        } else if op.num_args() > 0 {
            vec![
                self.value_types
                    .get(&op.arg(0).0)
                    .copied()
                    .unwrap_or(Type::Int),
            ]
        } else {
            Vec::new()
        };
        // compile.py:618-669 parity: use type-specific global singleton.
        let descr = Arc::new(DynasmFailDescr::new(
            fail_index,
            self.trace_id,
            fail_arg_types.clone(),
            true,
        ));
        // Finish ops write the type-appropriate singleton pointer to jf_descr
        // so CALL_ASSEMBLER's fast path CMP matches the correct variant.
        let result_type = if fail_arg_types.is_empty() {
            Type::Void
        } else {
            fail_arg_types[0]
        };
        let global_descr_ptr =
            crate::guard::done_with_this_frame_descr_ptr_for_type(result_type) as i64;

        // If there's a result argument, store it to jf_frame[0].
        // assembler.py:2291-2303 parity: float results use xmm0/MOVSD.
        if op.num_args() > 0 {
            let arg0 = op.arg(0);
            let slot0_offset = Self::slot_offset(0);
            if result_type == Type::Float {
                // Float: load to xmm0, store via MOVSD
                self.load_arg_to_rax(arg0); // loads raw bits
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov [rbp + slot0_offset], rax  // store float bits via GPR
                );
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x29, slot0_offset as u32]
                );
            } else {
                self.load_arg_to_rax(arg0);
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
        }

        // Store descr pointer at jf_ptr[0] (jf_descr slot).
        // compile.py:665-674 parity: use global singleton pointer.
        let descr_ptr = global_descr_ptr;
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
    /// SAME_AS: result = arg0 (identity).
    /// regalloc.py parity: no code emitted — just alias the slot.
    fn genop_same_as(&mut self, op: &Op) {
        let arg = op.arg(0);
        if let Some(&slot) = self.opref_to_slot.get(&arg.0) {
            self.opref_to_slot.insert(op.pos.0, slot);
        } else {
            self.load_arg_to_rax(arg);
            self.store_rax_to_result(op.pos);
        }
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
    fn genop_discard_setfield(&mut self, op: &Op) {
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
    fn genop_discard_setarrayitem(&mut self, op: &Op) {
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

    /// Emit a function call. `func_arg` is the index of the function
    /// pointer arg; call arguments start at `func_arg + 1`.
    fn emit_call(&mut self, op: &Op, func_arg: usize) {
        let arg_count = op.num_args();

        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; push rbp);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);

        for i in (func_arg + 1)..arg_count.min(func_arg + 7) {
            let arg = op.arg(i);
            let abi_idx = i - func_arg - 1;
            match self.resolve_opref(arg) {
                ResolvedArg::Slot(offset) => {
                    #[cfg(target_arch = "x86_64")]
                    match abi_idx {
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
                        let reg = abi_idx as u8;
                        dynasm!(self.mc ; .arch aarch64
                            ; ldr X(reg), [x29, offset as u32]
                        );
                    }
                }
                ResolvedArg::Const(val) => {
                    #[cfg(target_arch = "x86_64")]
                    match abi_idx {
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
                        let reg = abi_idx as u32;
                        self.emit_mov_imm64(reg, val);
                    }
                }
            }
        }

        match self.resolve_opref(op.arg(func_arg)) {
            ResolvedArg::Slot(offset) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64
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
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD val as i64
                    ; call rax
                );
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(8, val);
                    dynasm!(self.mc ; .arch aarch64 ; blr x8);
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; pop rbp);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; ldp x29, x30, [sp], #16);
    }

    /// assembler.py:2176 _genop_call — internal call implementation.
    fn _genop_call(&mut self, op: &Op) {
        self.emit_call(op, 0);
    }

    /// assembler.py:2169-2174 _genop_real_call.
    /// genop_call_i = genop_call_r = genop_call_f = genop_call_n
    fn genop_call(&mut self, op: &Op) {
        self._genop_call(op);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// assembler.py:295-360 call_assembler: invoke a compiled JIT loop.
    ///
    /// RPython fast path (assembler.py:295-360):
    ///   1. _call_assembler_emit_call — call the target trace
    ///   2. _call_assembler_check_descr — CMP jf_descr == done_with_this_frame_descr
    ///   3. Path A (slow): call assembler_helper
    ///   4. Path B (fast): MOV result, [frame + ofs]
    ///   5. join paths
    ///
    /// RPython allocates the callee jitframe via malloc_jitframe (heap).
    /// We use malloc for parity: the callee's frame-slot model needs
    /// frame_depth slots (not just num_args), and heap allocation avoids
    /// stack overflow on deep recursion.
    fn genop_call_assembler(&mut self, op: &Op) {
        let _descr = op.descr.as_ref().and_then(|d| d.as_call_descr());

        let num_args = op.args.len();
        // llmodel.py:298 malloc_jitframe parity: callee needs frame_depth
        // slots (frame-slot model stores ALL intermediates in jitframe).
        let jf_slots = self.frame_depth.max(num_args);
        let jf_alloc_bytes = ((1 + jf_slots) * 8) as i64;
        let malloc_ptr = libc::malloc as *const () as i64;
        let free_ptr = libc::free as *const () as i64;

        // Save caller's jf_ptr before clobbering rbp/x29.
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; mov r12, rbp            // save caller's jf_ptr in r12
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; mov x19, x29            // save caller's jf_ptr
        );

        // Allocate callee jitframe on heap via malloc.
        // Stack alignment: after prologue (push rbp + push r12) + return
        // addr, rsp ≡ -8 (mod 16). sub 8 aligns to 16 for ABI call.
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; sub rsp, 8                        // align to 16
            ; mov rdi, QWORD jf_alloc_bytes     // malloc size
            ; mov rax, QWORD malloc_ptr
            ; call rax                          // rax = heap jf_ptr
            ; add rsp, 8                        // unalign
        );
        #[cfg(target_arch = "aarch64")]
        {
            self.emit_mov_imm64(0, jf_alloc_bytes);
            self.emit_mov_imm64(2, malloc_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; sub sp, sp, 16                // align
                ; blr x2                        // x0 = heap jf_ptr
                ; add sp, sp, 16                // unalign
            );
        }

        // rdx/x20 = heap jf_ptr (held across arg stores).
        // load_arg_to_rax reads from [rbp+offset], rbp still = caller's jf.
        // Wait: rbp was saved to r12 but NOT changed yet. Actually we did
        // `mov r12, rbp` above, so rbp still points to caller's jf. ✓
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; mov rdx, rax            // rdx = heap jf_ptr
            ; mov QWORD [rdx], 0      // zero jf_descr
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; mov x20, x0             // x20 = heap jf_ptr
            ; str xzr, [x20]          // zero jf_descr
        );

        // Store callee args at jf_frame[i] = heap_jf + (1+i)*8.
        for (i, &arg_ref) in op.args.iter().enumerate() {
            let dest_offset = ((1 + i) * 8) as i32;
            self.load_arg_to_rax(arg_ref); // reads from [rbp+...], doesn't clobber rdx
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64
                ; mov [rdx + dest_offset], rax
            );
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64
                ; str x0, [x20, dest_offset as u32]
            );
        }

        // _call_assembler_emit_call (assembler.py:2267-2269):
        // rdi/x0 = callee jf_ptr.
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; mov rdi, rdx            // rdi = heap jf ptr
            ; sub rsp, 8              // align stack for call
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x20             // x0 = heap jf ptr
        );

        // assembler.py:320 _call_assembler_emit_call(descr._ll_function_addr, ...)
        // Resolve target address from descr.call_target_token() or self_entry_label.
        let target_addr: Option<usize> = op
            .descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .and_then(|cd| cd.call_target_token())
            .and_then(|token| self.call_assembler_targets.get(&token).copied());

        let is_resolved = target_addr.is_some() || self.self_entry_label.is_some();

        // assembler.py:324-336 call_assembler: select done_descr by op.type.
        let result_type = op.opcode.result_type();
        let done_descr_ptr =
            crate::guard::done_with_this_frame_descr_ptr_for_type(result_type) as i64;
        let helper_addr = crate::call_assembler_helper_addr() as i64;
        let green_key = self.header_pc as i64;

        if !is_resolved {
            // Pending/unresolved target: code not yet compiled.
            // RPython parity: call_assembler_fast_path (compiler.rs:2430)
            // detects null code_ptr and calls force_fn(inputs[0]) where
            // inputs[0] = the callee's frame pointer (a PyFrame).
            //
            // In dynasm, args[0] was stored at jf_frame[0] = [rdx + 8].
            // Read it, free the heap jf, then call force_fn(frame_ptr).
            let force_addr = crate::call_assembler_force_fn_addr() as i64;
            #[cfg(target_arch = "x86_64")]
            {
                if force_addr != 0 {
                    dynasm!(self.mc ; .arch x64
                        ; mov r13, [rdx + 8]            // r13 = jf_frame[0] = args[0] (PyFrame ptr)
                        ; mov rdi, rdx                   // free(heap_jf)
                        ; sub rsp, 8
                        ; mov rax, QWORD free_ptr
                        ; call rax
                        ; add rsp, 8
                        ; mov rbp, r12                   // restore caller jf_ptr
                        ; mov rdi, r13                   // arg0 = PyFrame ptr
                        ; sub rsp, 8
                        ; mov rax, QWORD force_addr
                        ; call rax                       // rax = force_fn(frame_ptr) = result
                        ; add rsp, 8
                    );
                } else {
                    // No force_fn registered — free and return 0.
                    dynasm!(self.mc ; .arch x64
                        ; mov rdi, rdx                   // free(heap_jf)
                        ; sub rsp, 8
                        ; mov rax, QWORD free_ptr
                        ; call rax
                        ; add rsp, 8
                        ; mov rbp, r12                   // restore caller jf_ptr
                        ; xor eax, eax                   // result = 0
                    );
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if force_addr != 0 {
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x21, [x20, 8]             // x21 = jf_frame[0] (PyFrame ptr)
                    );
                    self.emit_mov_imm64(2, free_ptr);
                    dynasm!(self.mc ; .arch aarch64
                        ; mov x0, x20
                        ; blr x2                        // free(heap_jf)
                        ; mov x29, x19                  // restore caller jf_ptr
                        ; mov x0, x21                   // arg0 = PyFrame ptr
                    );
                    self.emit_mov_imm64(2, force_addr);
                    dynasm!(self.mc ; .arch aarch64
                        ; blr x2                        // x0 = force_fn(frame_ptr)
                    );
                } else {
                    self.emit_mov_imm64(2, free_ptr);
                    dynasm!(self.mc ; .arch aarch64
                        ; mov x0, x20
                        ; blr x2
                        ; mov x29, x19
                        ; mov x0, 0
                    );
                }
            }
        } else {
            // Resolved target: call callee and handle fast/slow path.
            if let Some(addr) = target_addr {
                let addr = addr as i64;
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD addr
                    ; call rax
                );
                #[cfg(target_arch = "aarch64")]
                {
                    self.emit_mov_imm64(2, addr);
                    dynasm!(self.mc ; .arch aarch64 ; blr x2);
                }
            } else if let Some(label) = self.self_entry_label {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64 ; call =>label);
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64 ; bl =>label);
            }

            // rax/x0 = callee's returned jf_ptr (= heap jf_ptr we passed).
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64
                ; add rsp, 8              // undo alignment
            );

            // Restore caller's jf_ptr.
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64
                ; mov rbp, r12            // restore caller's jf_ptr
            );
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64
                ; mov x29, x19            // restore
            );

            // rax = callee's returned jf_ptr (heap-allocated).
            // Save it in rdx for descr check and free.
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64
                ; mov rdx, rax            // rdx = callee jf_ptr
            );
            #[cfg(target_arch = "aarch64")]
            dynasm!(self.mc ; .arch aarch64
                ; mov x20, x0             // x20 = callee jf_ptr
            );

            // _call_assembler_check_descr (assembler.py:2274-2278):
            //   CMP [jf_ptr + jf_descr_ofs], done_with_this_frame_descr_{type}
            #[cfg(target_arch = "x86_64")]
            {
                let fast_path = self.mc.new_dynamic_label();
                let merge = self.mc.new_dynamic_label();
                dynasm!(self.mc ; .arch x64
                    ; mov rcx, [rdx]                    // rcx = jf_descr
                    ; mov rax, QWORD done_descr_ptr
                    ; cmp rcx, rax
                    ; je =>fast_path
                );

                // Path A (slow): guard failure.
                // assembler.py:345-350 _call_assembler_emit_helper_call
                dynasm!(self.mc ; .arch x64
                    ; mov rdi, rdx                      // arg0 = callee_jf_ptr
                    ; mov rsi, QWORD green_key as i64   // arg1 = green_key
                    ; sub rsp, 8
                    ; mov rax, QWORD helper_addr
                    ; call rax                          // rax = helper result
                    ; add rsp, 8
                    ; jmp =>merge
                );

                // Path B (fast): _call_assembler_load_result (assembler.py:2291-2303)
                dynasm!(self.mc ; .arch x64
                    ; =>fast_path
                );
                if result_type == Type::Float {
                    dynasm!(self.mc ; .arch x64
                        ; movsd xmm0, [rdx + 8]        // xmm0 = jf_frame[0] (float)
                        ; movq r12, xmm0               // preserve bits across free
                        ; mov rdi, rdx                  // free(callee_jf)
                        ; sub rsp, 8
                        ; mov rax, QWORD free_ptr
                        ; call rax
                        ; add rsp, 8
                        ; mov rax, r12                  // float bits in rax
                        ; =>merge
                    );
                } else {
                    dynasm!(self.mc ; .arch x64
                        ; mov r12, [rdx + 8]            // r12 = jf_frame[0] (result)
                        ; mov rdi, rdx                  // free(callee_jf)
                        ; sub rsp, 8
                        ; mov rax, QWORD free_ptr
                        ; call rax
                        ; add rsp, 8
                        ; mov rax, r12                  // rax = result
                        ; =>merge
                    );
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                let fast_path = self.mc.new_dynamic_label();
                let merge = self.mc.new_dynamic_label();
                self.emit_mov_imm64(2, done_descr_ptr);
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x1, [x20]                     // x1 = jf_descr
                    ; cmp x1, x2
                    ; b.eq =>fast_path
                );
                // Path A (slow)
                {
                    self.emit_mov_imm64(2, helper_addr);
                    self.emit_mov_imm64(1, green_key); // x1 = green_key
                    dynasm!(self.mc ; .arch aarch64
                        ; mov x0, x20                   // arg0 = callee_jf_ptr
                        ; blr x2                        // x0 = helper result
                        ; b =>merge
                    );
                }
                // Path B (fast)
                {
                    dynasm!(self.mc ; .arch aarch64
                        ; =>fast_path
                        ; ldr x19, [x20, 8]             // x19 = result
                    );
                    self.emit_mov_imm64(2, free_ptr);
                    dynasm!(self.mc ; .arch aarch64
                        ; mov x0, x20
                        ; blr x2
                        ; mov x0, x19                   // x0 = result
                        ; =>merge
                    );
                }
            }
        } // end if is_resolved

        // Store result to the output slot.
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
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
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((8, 8));
        self.genop_alloc_varsize(op, base_size, item_size);
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

    // ================================================================
    // assembler.py:1817 genop_save_exc_class / genop_save_exception
    // ================================================================

    /// assembler.py:1817 genop_save_exc_class — stub: returns 0.
    fn genop_save_exc_class(&mut self, op: &Op) {
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; xor eax, eax);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; mov x0, 0);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// assembler.py:1827 genop_save_exception — stub: returns 0.
    fn genop_save_exception(&mut self, op: &Op) {
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; xor eax, eax);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; mov x0, 0);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    // ================================================================
    // genop_* — extended integer arithmetic
    // ================================================================

    /// INT_FLOORDIV: result = arg0 / arg1 (signed)
    fn genop_int_floordiv(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; cqo
            ; idiv rcx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; sdiv x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_MOD: result = arg0 % arg1 (signed)
    fn genop_int_mod(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; cqo
            ; idiv rcx
            ; mov rax, rdx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; sdiv x2, x0, x1
            ; msub x0, x2, x1, x0
        );
        self.store_rax_to_result(op.pos);
    }

    /// UINT_MUL_HIGH: upper 64 bits of unsigned multiply
    fn genop_uint_mul_high(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; mul rcx
            ; mov rax, rdx
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; umulh x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_SIGNEXT: sign-extend from num_bytes width to 64 bits.
    fn genop_int_signext(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        let num_bytes = match self.resolve_opref(op.arg(1)) {
            ResolvedArg::Const(v) => v,
            _ => 8,
        };
        let shift = 64 - num_bytes * 8;
        if shift > 0 && shift < 64 {
            let sh = shift as u8;
            #[cfg(target_arch = "x86_64")]
            dynasm!(self.mc ; .arch x64
                ; shl rax, sh
                ; sar rax, sh
            );
            #[cfg(target_arch = "aarch64")]
            {
                let sh32 = shift as u32;
                dynasm!(self.mc ; .arch aarch64
                    ; lsl x0, x0, sh32
                    ; asr x0, x0, sh32
                );
            }
        }
        self.store_rax_to_result(op.pos);
    }

    // ================================================================
    // genop_* — extended float operations
    // ================================================================

    /// FLOAT_ABS: result = |arg0|
    fn genop_float_abs(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        {
            let mask: i64 = i64::MAX; // 0x7FFF_FFFF_FFFF_FFFF
            dynasm!(self.mc ; .arch x64
                ; mov rax, QWORD mask
                ; movq xmm1, rax
                ; andpd xmm0, xmm1
            );
        }
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fabs d0, d0
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_LT/LE/EQ/NE/GT/GE: float comparison.
    /// For lt/le, swap operands so JA/JAE handles NaN correctly.
    fn genop_float_cmp(&mut self, op: &Op) {
        let swap = matches!(op.opcode, OpCode::FloatLt | OpCode::FloatLe);
        if swap {
            self.load_float_arg_to_d0(op.arg(1));
            self.load_float_arg_to_d1(op.arg(0));
        } else {
            self.load_float_arg_to_d0(op.arg(0));
            self.load_float_arg_to_d1(op.arg(1));
        }

        #[cfg(target_arch = "x86_64")]
        {
            dynasm!(self.mc ; .arch x64 ; ucomisd xmm0, xmm1);
            match op.opcode {
                OpCode::FloatLt | OpCode::FloatGt => {
                    dynasm!(self.mc ; .arch x64 ; seta al ; movzx eax, al);
                }
                OpCode::FloatLe | OpCode::FloatGe => {
                    dynasm!(self.mc ; .arch x64 ; setae al ; movzx eax, al);
                }
                OpCode::FloatEq => {
                    dynasm!(self.mc ; .arch x64
                        ; sete al ; setnp cl ; and al, cl ; movzx eax, al
                    );
                }
                OpCode::FloatNe => {
                    dynasm!(self.mc ; .arch x64
                        ; setne al ; setp cl ; or al, cl ; movzx eax, al
                    );
                }
                _ => {
                    dynasm!(self.mc ; .arch x64 ; sete al ; movzx eax, al);
                }
            }
            dynasm!(self.mc ; .arch x64 ; test rax, rax);
        }
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(self.mc ; .arch aarch64 ; fcmp d0, d1);
            match op.opcode {
                OpCode::FloatLt | OpCode::FloatGt => {
                    dynasm!(self.mc ; .arch aarch64 ; cset x0, gt);
                }
                OpCode::FloatLe | OpCode::FloatGe => {
                    dynasm!(self.mc ; .arch aarch64 ; cset x0, ge);
                }
                OpCode::FloatEq => {
                    dynasm!(self.mc ; .arch aarch64 ; cset x0, eq);
                }
                OpCode::FloatNe => {
                    dynasm!(self.mc ; .arch aarch64 ; cset x0, ne);
                }
                _ => {
                    dynasm!(self.mc ; .arch aarch64 ; cset x0, eq);
                }
            }
            dynasm!(self.mc ; .arch aarch64 ; cmp x0, 0);
        }
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// CAST_FLOAT_TO_SINGLEFLOAT: f64 → f32 (bits in lower 32 of i64)
    fn genop_cast_float_to_singlefloat(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; cvtsd2ss xmm0, xmm0
            ; movd eax, xmm0
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fcvt s0, d0
            ; fmov w0, s0
        );
        self.store_rax_to_result(op.pos);
    }

    /// CAST_SINGLEFLOAT_TO_FLOAT: f32 (bits in lower 32) → f64
    fn genop_cast_singlefloat_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64
            ; movd xmm0, eax
            ; cvtss2sd xmm0, xmm0
        );
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64
            ; fmov s0, w0
            ; fcvt d0, s0
        );
        self.store_d0_to_result(op.pos);
    }

    // ================================================================
    // genop_* — GC memory operations
    // ================================================================

    /// Emit a sized load from [rax]/[x0]. Positive size = zero-extend,
    /// negative = sign-extend.
    fn emit_load_from_rax_sized(&mut self, itemsize: i32) {
        let abs_size = itemsize.unsigned_abs() as usize;
        let signed = itemsize < 0;
        #[cfg(target_arch = "x86_64")]
        match (abs_size, signed) {
            (1, true) => dynasm!(self.mc ; .arch x64 ; movsx rax, BYTE [rax]),
            (2, true) => dynasm!(self.mc ; .arch x64 ; movsx rax, WORD [rax]),
            (4, true) => dynasm!(self.mc ; .arch x64
                ; mov eax, [rax]
                ; cdqe
            ),
            (1, false) => dynasm!(self.mc ; .arch x64 ; movzx eax, BYTE [rax]),
            (2, false) => dynasm!(self.mc ; .arch x64 ; movzx eax, WORD [rax]),
            (4, false) => dynasm!(self.mc ; .arch x64 ; mov eax, [rax]),
            _ => dynasm!(self.mc ; .arch x64 ; mov rax, [rax]),
        }
        #[cfg(target_arch = "aarch64")]
        match (abs_size, signed) {
            (1, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsb x0, [x0]),
            (2, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsh x0, [x0]),
            (4, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsw x0, [x0]),
            (1, false) => dynasm!(self.mc ; .arch aarch64 ; ldrb w0, [x0]),
            (2, false) => dynasm!(self.mc ; .arch aarch64 ; ldrh w0, [x0]),
            (4, false) => dynasm!(self.mc ; .arch aarch64 ; ldr w0, [x0]),
            _ => dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x0]),
        }
    }

    /// Emit a sized store of rcx/x1 to [rax]/[x0].
    fn emit_store_to_rax_sized(&mut self, size: usize) {
        #[cfg(target_arch = "x86_64")]
        match size {
            1 => dynasm!(self.mc ; .arch x64 ; mov [rax], cl),
            2 => dynasm!(self.mc ; .arch x64 ; mov [rax], cx),
            4 => dynasm!(self.mc ; .arch x64 ; mov [rax], ecx),
            _ => dynasm!(self.mc ; .arch x64 ; mov [rax], rcx),
        }
        #[cfg(target_arch = "aarch64")]
        match size {
            1 => dynasm!(self.mc ; .arch aarch64 ; strb w1, [x0]),
            2 => dynasm!(self.mc ; .arch aarch64 ; strh w1, [x0]),
            4 => dynasm!(self.mc ; .arch aarch64 ; str w1, [x0]),
            _ => dynasm!(self.mc ; .arch aarch64 ; str x1, [x0]),
        }
    }

    /// Resolve an OpRef that is expected to be a compile-time constant.
    fn resolve_const_or(&self, opref: OpRef, default: i64) -> i64 {
        match self.resolve_opref(opref) {
            ResolvedArg::Const(v) => v,
            _ => default,
        }
    }

    /// GC_LOAD_I/R/F: load from base + offset with given itemsize.
    /// arg(0) = base, arg(1) = offset, arg(2) = itemsize.
    fn genop_gc_load(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);

        let itemsize = self.resolve_const_or(op.arg(2), 8) as i32;
        self.emit_load_from_rax_sized(itemsize);
        self.store_rax_to_result(op.pos);
    }

    /// GC_LOAD_INDEXED_I/R/F: load from base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=scale, arg(3)=base_offset, arg(4)=itemsize.
    fn genop_gc_load_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(2), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(3), 0) as i32;
        let itemsize = self.resolve_const_or(op.arg(4), 8) as i32;

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        #[cfg(target_arch = "x86_64")]
        {
            if scale != 1 {
                dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale);
            }
            dynasm!(self.mc ; .arch x64 ; add rax, rcx);
            if base_offset != 0 {
                dynasm!(self.mc ; .arch x64 ; add rax, base_offset);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if scale != 1 {
                self.emit_mov_imm64(2, scale as i64);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
            if base_offset != 0 {
                dynasm!(self.mc ; .arch aarch64 ; add x0, x0, base_offset as u32);
            }
        }

        self.emit_load_from_rax_sized(itemsize);
        self.store_rax_to_result(op.pos);
    }

    /// GC_STORE: store value to base + offset.
    /// 4-arg form: arg(0)=base, arg(1)=offset, arg(2)=value, arg(3)=itemsize.
    fn genop_discard_gc_store(&mut self, op: &Op) {
        if op.num_args() < 4 {
            return; // 3-arg GC rewrite form — skip for now
        }
        let itemsize = self.resolve_const_or(op.arg(3), 8).unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        {
            dynasm!(self.mc ; .arch x64 ; add rax, rcx);
            dynasm!(self.mc ; .arch x64 ; push rax);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch x64 ; pop rax);
        }
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
            dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        }
        self.emit_store_to_rax_sized(itemsize);
    }

    /// GC_STORE_INDEXED: store to base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=value, arg(3)=scale,
    /// arg(4)=base_offset, arg(5)=itemsize.
    fn genop_discard_gc_store_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(3), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(4), 0) as i32;
        let itemsize = self.resolve_const_or(op.arg(5), 8).unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        {
            if scale != 1 {
                dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale);
            }
            dynasm!(self.mc ; .arch x64 ; add rax, rcx);
            if base_offset != 0 {
                dynasm!(self.mc ; .arch x64 ; add rax, base_offset);
            }
            dynasm!(self.mc ; .arch x64 ; push rax);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch x64 ; pop rax);
        }
        #[cfg(target_arch = "aarch64")]
        {
            if scale != 1 {
                self.emit_mov_imm64(2, scale as i64);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
            if base_offset != 0 {
                dynasm!(self.mc ; .arch aarch64 ; add x0, x0, base_offset as u32);
            }
            dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        }
        self.emit_store_to_rax_sized(itemsize);
    }

    /// RAW_LOAD_I/F: load from base + offset using descriptor.
    fn genop_raw_load(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);

        self.emit_load_from_rax_sized(size as i32);
        let _ = offset; // offset is in the descriptor, not used for raw_load
        self.store_rax_to_result(op.pos);
    }

    /// RAW_STORE: store value to base + offset using descriptor.
    fn genop_discard_raw_store(&mut self, op: &Op) {
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        #[cfg(target_arch = "x86_64")]
        {
            dynasm!(self.mc ; .arch x64 ; add rax, rcx);
            dynasm!(self.mc ; .arch x64 ; push rax);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch x64 ; pop rax);
        }
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
            dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        }
        self.emit_store_to_rax_sized(size);
    }

    // ================================================================
    // genop_* — interior field operations
    // ================================================================

    /// GETINTERIORFIELD_GC_I/R/F: load field from array-of-structs element.
    fn genop_getinteriorfield(&mut self, op: &Op) {
        let (base_size, item_size, field_offset, field_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_interior_field_descr())
            .map(|id| {
                let ad = id.array_descr();
                let fd = id.field_descr();
                (
                    ad.base_size() as i32,
                    ad.item_size() as i32,
                    fd.offset() as i32,
                    fd.field_size(),
                )
            })
            .unwrap_or((8, 8, 0, 8));

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        let total_offset = base_size + field_offset;

        #[cfg(target_arch = "x86_64")]
        {
            if item_size != 1 {
                dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
            }
            dynasm!(self.mc ; .arch x64
                ; add rax, total_offset
                ; add rax, rcx
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            if item_size != 1 {
                self.emit_mov_imm64(2, item_size as i64);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            if total_offset != 0 {
                dynasm!(self.mc ; .arch aarch64 ; add x0, x0, total_offset as u32);
            }
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        }

        self.emit_load_from_rax_sized(field_size as i32);
        self.store_rax_to_result(op.pos);
    }

    /// SETINTERIORFIELD_GC/RAW: write field in array-of-structs element.
    fn genop_discard_setinteriorfield(&mut self, op: &Op) {
        let (base_size, item_size, field_offset, field_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_interior_field_descr())
            .map(|id| {
                let ad = id.array_descr();
                let fd = id.field_descr();
                (
                    ad.base_size() as i32,
                    ad.item_size() as i32,
                    fd.offset() as i32,
                    fd.field_size(),
                )
            })
            .unwrap_or((8, 8, 0, 8));

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        let total_offset = base_size + field_offset;

        #[cfg(target_arch = "x86_64")]
        {
            if item_size != 1 {
                dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
            }
            dynasm!(self.mc ; .arch x64
                ; add rax, total_offset
                ; add rax, rcx
            );
            dynasm!(self.mc ; .arch x64 ; push rax);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch x64 ; pop rax);
        }
        #[cfg(target_arch = "aarch64")]
        {
            if item_size != 1 {
                self.emit_mov_imm64(2, item_size as i64);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            if total_offset != 0 {
                dynasm!(self.mc ; .arch aarch64 ; add x0, x0, total_offset as u32);
            }
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
            dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        }

        self.emit_store_to_rax_sized(field_size);
    }

    // ================================================================
    // genop_* — call variants
    // ================================================================

    /// COND_CALL_N: if arg(0) != 0, call function at arg(1).
    fn genop_discard_cond_call(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        let skip_label = self.mc.new_dynamic_label();
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; test rax, rax ; jz =>skip_label);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; cbz x0, =>skip_label);

        self.emit_call(op, 1);

        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; =>skip_label);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; =>skip_label);
    }

    /// COND_CALL_VALUE_I/R: if arg(0) == 0, call function; else result = arg(0).
    fn genop_cond_call_value(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        let skip_label = self.mc.new_dynamic_label();
        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; test rax, rax ; jnz =>skip_label);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; cbnz x0, =>skip_label);

        self.emit_call(op, 1);

        #[cfg(target_arch = "x86_64")]
        dynasm!(self.mc ; .arch x64 ; =>skip_label);
        #[cfg(target_arch = "aarch64")]
        dynasm!(self.mc ; .arch aarch64 ; =>skip_label);

        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    // ================================================================
    // genop_* — string/array operations
    // ================================================================

    /// STRSETITEM / UNICODESETITEM: string[index] = value.
    fn genop_discard_strsetitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((16, 1));

        self.load_arg_to_rax(op.arg(0)); // string
        self.load_arg_to_rcx(op.arg(1)); // index
        #[cfg(target_arch = "x86_64")]
        {
            if item_size != 1 {
                dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
            }
            dynasm!(self.mc ; .arch x64 ; add rax, base_size ; add rax, rcx);
            dynasm!(self.mc ; .arch x64 ; push rax);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch x64 ; pop rax);
        }
        #[cfg(target_arch = "aarch64")]
        {
            if item_size != 1 {
                self.emit_mov_imm64(2, item_size as i64);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
                ; add x0, x0, x1
            );
            dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
            self.load_arg_to_rcx(op.arg(2));
            dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        }
        self.emit_store_to_rax_sized(item_size as usize);
    }

    /// COPYSTRCONTENT / COPYUNICODECONTENT: copy substring.
    /// arg(0)=src, arg(1)=dst, arg(2)=src_start, arg(3)=dst_start, arg(4)=length.
    fn genop_discard_copystrcontent(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((16, 1));

        // Compute byte_count = length * item_size
        self.load_arg_to_rax(op.arg(4));
        #[cfg(target_arch = "x86_64")]
        {
            if item_size != 1 {
                dynasm!(self.mc ; .arch x64
                    ; mov rcx, QWORD item_size
                    ; imul rax, rcx
                );
            }
            dynasm!(self.mc ; .arch x64 ; push rax); // [rsp] = byte_count

            // Compute src_addr = src + base_size + src_start * item_size
            self.load_arg_to_rax(op.arg(0));
            self.load_arg_to_rcx(op.arg(2));
            if item_size != 1 {
                dynasm!(self.mc ; .arch x64
                    ; mov rdx, QWORD item_size
                    ; imul rcx, rdx
                );
            }
            dynasm!(self.mc ; .arch x64
                ; add rax, DWORD base_size as i32
                ; add rax, rcx
                ; push rax  // [rsp] = src_addr
            );

            // Compute dst_addr = dst + base_size + dst_start * item_size
            self.load_arg_to_rax(op.arg(1));
            self.load_arg_to_rcx(op.arg(3));
            if item_size != 1 {
                dynasm!(self.mc ; .arch x64
                    ; mov rdx, QWORD item_size
                    ; imul rcx, rdx
                );
            }
            dynasm!(self.mc ; .arch x64
                ; add rax, DWORD base_size as i32
                ; add rax, rcx
            );

            // memmove(dst_addr, src_addr, byte_count)
            let memmove_ptr = libc::memmove as *const () as i64;
            dynasm!(self.mc ; .arch x64
                ; mov rdi, rax   // dst
                ; pop rsi        // src
                ; pop rdx        // count
                ; push rbp
                ; mov rax, QWORD memmove_ptr
                ; call rax
                ; pop rbp
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            if item_size != 1 {
                self.emit_mov_imm64(1, item_size);
                dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
            }
            dynasm!(self.mc ; .arch aarch64 ; str x0, [sp, #-16]!); // byte_count

            self.load_arg_to_rax(op.arg(0));
            self.load_arg_to_rcx(op.arg(2));
            if item_size != 1 {
                self.emit_mov_imm64(2, item_size);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
                ; add x0, x0, x1
                ; str x0, [sp, #-16]!  // src_addr
            );

            self.load_arg_to_rax(op.arg(1));
            self.load_arg_to_rcx(op.arg(3));
            if item_size != 1 {
                self.emit_mov_imm64(2, item_size);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
                ; add x0, x0, x1
            );

            let memmove_ptr = libc::memmove as *const () as i64;
            dynasm!(self.mc ; .arch aarch64
                ; ldr x1, [sp], #16  // src
                ; ldr x2, [sp], #16  // count
            );
            // x0 = dst already
            dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);
            self.emit_mov_imm64(3, memmove_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; blr x3
                ; ldp x29, x30, [sp], #16
            );
        }
    }

    /// NEWSTR: allocate a byte string of given length.
    fn genop_newstr(&mut self, op: &Op) {
        self.genop_alloc_varsize(op, 16, 1);
    }

    /// NEWUNICODE: allocate a unicode string (4-byte chars).
    fn genop_newunicode(&mut self, op: &Op) {
        self.genop_alloc_varsize(op, 16, 4);
    }

    /// Shared implementation for NEWSTR / NEWUNICODE / NEW_ARRAY.
    /// Allocates base_size + length * item_size bytes, zero-fills,
    /// and writes length to the header.
    fn genop_alloc_varsize(&mut self, op: &Op, base_size: i64, item_size: i64) {
        // arg(0) = length
        self.load_arg_to_rax(op.arg(0));
        let malloc_ptr = libc::malloc as *const () as i64;
        let memset_ptr = libc::memset as *const () as i64;

        #[cfg(target_arch = "x86_64")]
        {
            // Save length, compute total_size = base_size + length * item_size
            dynasm!(self.mc ; .arch x64
                ; push rax                           // save length
                ; imul rax, rax, item_size as i32
                ; add rax, base_size as i32
                ; mov rdi, rax                       // malloc arg
                ; push rax                           // save total_size
                ; mov rax, QWORD malloc_ptr
                ; call rax
                ; pop rcx                            // rcx = total_size
                ; push rax                           // save ptr
                // memset(ptr, 0, total_size)
                ; mov rdi, rax
                ; xor esi, esi
                ; mov rdx, rcx
                ; mov rax, QWORD memset_ptr
                ; call rax
                ; pop rax                            // rax = ptr
                ; pop rcx                            // rcx = length
                // Store length at offset 8 (RPython string header)
                ; mov [rax + 8], rcx
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            dynasm!(self.mc ; .arch aarch64
                ; str x0, [sp, #-16]!               // save length
            );
            if item_size != 1 {
                self.emit_mov_imm64(1, item_size);
                dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
            }
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
                ; str x0, [sp, #-16]!               // save total_size
            );
            dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);
            self.emit_mov_imm64(8, malloc_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; blr x8
                ; ldp x29, x30, [sp], #16
                ; ldr x2, [sp], #16                 // total_size
                ; mov x19, x0                        // save ptr
                ; mov x1, 0                          // val = 0
                ; stp x29, x30, [sp, #-16]!
            );
            self.emit_mov_imm64(8, memset_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; blr x8
                ; ldp x29, x30, [sp], #16
                ; mov x0, x19                        // restore ptr
                ; ldr x1, [sp], #16                  // length
                ; str x1, [x0, 8]                    // store length at offset 8
            );
        }

        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// ZERO_ARRAY: zero a range in an array.
    /// arg(0)=base, arg(1)=start, arg(2)=size, arg(3)=scale_start, arg(4)=scale_size.
    fn genop_discard_zero_array(&mut self, op: &Op) {
        let (base_size, _) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((8, 8));

        let scale_start = self.resolve_const_or(op.arg(3), 1);
        let scale_size = self.resolve_const_or(op.arg(4), 1);
        let memset_ptr = libc::memset as *const () as i64;

        // byte_offset = base_size + start * scale_start
        // byte_length = size * scale_size
        self.load_arg_to_rax(op.arg(0)); // base
        self.load_arg_to_rcx(op.arg(1)); // start

        #[cfg(target_arch = "x86_64")]
        {
            if scale_start != 1 {
                dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale_start as i32);
            }
            dynasm!(self.mc ; .arch x64
                ; add rax, DWORD base_size as i32
                ; add rax, rcx
                ; push rax                           // save dest
            );
            self.load_arg_to_rax(op.arg(2)); // size
            if scale_size != 1 {
                dynasm!(self.mc ; .arch x64 ; imul rax, rax, scale_size as i32);
            }
            // memset(dest, 0, byte_length)
            dynasm!(self.mc ; .arch x64
                ; mov rdx, rax                       // byte_length
                ; pop rdi                            // dest
                ; xor esi, esi                       // val = 0
                ; push rbp
                ; mov rax, QWORD memset_ptr
                ; call rax
                ; pop rbp
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            if scale_start != 1 {
                self.emit_mov_imm64(2, scale_start);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
                ; add x0, x0, x1
                ; str x0, [sp, #-16]!               // save dest
            );
            self.load_arg_to_rax(op.arg(2));
            if scale_size != 1 {
                self.emit_mov_imm64(1, scale_size);
                dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
            }
            dynasm!(self.mc ; .arch aarch64
                ; mov x2, x0                         // byte_length
                ; ldr x0, [sp], #16                  // dest
                ; mov x1, 0
                ; stp x29, x30, [sp, #-16]!
            );
            self.emit_mov_imm64(8, memset_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; blr x8
                ; ldp x29, x30, [sp], #16
            );
        }
    }

    // ================================================================
    // genop_* — address computation
    // ================================================================

    /// LOAD_EFFECTIVE_ADDRESS: result = base + index * scale + offset.
    /// arg(0)=base, arg(1)=index, arg(2)=scale, arg(3)=offset.
    fn genop_load_effective_address(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(2), 1) as i32;
        let offset = self.resolve_const_or(op.arg(3), 0) as i32;

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        #[cfg(target_arch = "x86_64")]
        {
            if scale != 1 {
                dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale);
            }
            dynasm!(self.mc ; .arch x64 ; add rax, rcx);
            if offset != 0 {
                dynasm!(self.mc ; .arch x64 ; add rax, offset);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if scale != 1 {
                self.emit_mov_imm64(2, scale as i64);
                dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
            }
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
            if offset != 0 {
                dynasm!(self.mc ; .arch aarch64 ; add x0, x0, offset as u32);
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
