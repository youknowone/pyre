//! Backward liveness analysis on assembled JitCode bytecodes.
//!
//! RPython equivalent: `rpython/jit/codewriter/liveness.py:compute_liveness`.
//!
//! RPython computes liveness on the SSARepr (pre-assembly IR). Pyre
//! operates on assembled JitCode bytecodes instead, parsing each
//! instruction to determine source/destination registers and running
//! the same backward dataflow analysis.

use super::{
    BC_ABORT, BC_ABORT_PERMANENT, BC_ARRAYLEN_VABLE, BC_BRANCH_REG_ZERO, BC_BRANCH_ZERO,
    BC_CALL_ASSEMBLER_FLOAT, BC_CALL_ASSEMBLER_INT, BC_CALL_ASSEMBLER_REF, BC_CALL_ASSEMBLER_VOID,
    BC_CALL_FLOAT, BC_CALL_INT, BC_CALL_LOOPINVARIANT_FLOAT, BC_CALL_LOOPINVARIANT_INT,
    BC_CALL_LOOPINVARIANT_REF, BC_CALL_LOOPINVARIANT_VOID, BC_CALL_MAY_FORCE_FLOAT,
    BC_CALL_MAY_FORCE_INT, BC_CALL_MAY_FORCE_REF, BC_CALL_MAY_FORCE_VOID, BC_CALL_PURE_FLOAT,
    BC_CALL_PURE_INT, BC_CALL_PURE_REF, BC_CALL_REF, BC_CALL_RELEASE_GIL_FLOAT,
    BC_CALL_RELEASE_GIL_INT, BC_CALL_RELEASE_GIL_REF, BC_CALL_RELEASE_GIL_VOID,
    BC_COND_CALL_VALUE_INT, BC_COND_CALL_VALUE_REF, BC_COND_CALL_VOID, BC_COPY_FROM_BOTTOM,
    BC_DUP_STACK, BC_FLOAT_GUARD_VALUE, BC_GETARRAYITEM_VABLE_F, BC_GETARRAYITEM_VABLE_I,
    BC_GETARRAYITEM_VABLE_R, BC_GETFIELD_VABLE_F, BC_GETFIELD_VABLE_I, BC_GETFIELD_VABLE_R,
    BC_HINT_FORCE_VIRTUALIZABLE, BC_INLINE_CALL, BC_INT_GUARD_VALUE, BC_JIT_MERGE_POINT, BC_JUMP,
    BC_JUMP_TARGET, BC_LOAD_CONST_F, BC_LOAD_CONST_I, BC_LOAD_CONST_R, BC_LOAD_STATE_ARRAY,
    BC_LOAD_STATE_FIELD, BC_LOAD_STATE_VARRAY, BC_MOVE_F, BC_MOVE_I, BC_MOVE_R, BC_PEEK_I,
    BC_POP_DISCARD, BC_POP_F, BC_POP_I, BC_POP_R, BC_PUSH_F, BC_PUSH_I, BC_PUSH_R, BC_PUSH_TO,
    BC_RAISE, BC_RECORD_BINOP_F, BC_RECORD_BINOP_I, BC_RECORD_KNOWN_RESULT_INT,
    BC_RECORD_KNOWN_RESULT_REF, BC_RECORD_UNARY_F, BC_RECORD_UNARY_I, BC_REF_GUARD_VALUE,
    BC_REF_RETURN, BC_REQUIRE_STACK, BC_RERAISE, BC_RESIDUAL_CALL_VOID, BC_SET_SELECTED,
    BC_SETARRAYITEM_VABLE_F, BC_SETARRAYITEM_VABLE_I, BC_SETARRAYITEM_VABLE_R, BC_SETFIELD_VABLE_F,
    BC_SETFIELD_VABLE_I, BC_SETFIELD_VABLE_R, BC_STORE_DOWN, BC_STORE_STATE_ARRAY,
    BC_STORE_STATE_FIELD, BC_STORE_STATE_VARRAY, BC_SWAP_STACK, JitArgKind, JitCode, LivenessInfo,
};

/// Register bank identifier for liveness tracking.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bank {
    Int,
    Ref,
    Float,
}

/// A parsed bytecode instruction's register effects.
struct InsnInfo {
    /// Next PC after this instruction.
    next_pc: usize,
    /// Registers read (source). (bank, reg_index)
    reads: Vec<(Bank, u16)>,
    /// Registers written (destination). (bank, reg_index)
    /// RPython liveness.py:61-64: `args[-2] == '->'` kills one register.
    /// BC_INLINE_CALL can kill up to 3 (i/r/f return slots).
    writes: Vec<(Bank, u16)>,
    /// Branch target PC (for conditional/unconditional jumps).
    branch_target: Option<usize>,
    /// Whether control flow falls through to next_pc.
    falls_through: bool,
}

fn read_u16(code: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes([code[pos], code[pos + 1]])
}

/// Parse one instruction at `pc` and return its register effects.
fn parse_insn(code: &[u8], pc: usize) -> InsnInfo {
    let opcode = code[pc];
    let mut pos = pc + 1;
    let mut reads = Vec::new();
    let mut writes: Vec<(Bank, u16)> = Vec::new();
    let mut branch_target = None;
    let mut falls_through = true;

    macro_rules! u16_at {
        () => {{
            let v = read_u16(code, pos);
            pos += 2;
            v
        }};
    }

    match opcode {
        // -- LOAD_CONST: write only --
        BC_LOAD_CONST_I => {
            let dst = u16_at!();
            let _const_idx = u16_at!();
            writes.push((Bank::Int, dst));
        }
        BC_LOAD_CONST_R => {
            let dst = u16_at!();
            let _const_idx = u16_at!();
            writes.push((Bank::Ref, dst));
        }
        BC_LOAD_CONST_F => {
            let dst = u16_at!();
            let _const_idx = u16_at!();
            writes.push((Bank::Float, dst));
        }

        // -- MOVE: read src, write dst --
        BC_MOVE_I => {
            let dst = u16_at!();
            let src = u16_at!();
            writes.push((Bank::Int, dst));
            reads.push((Bank::Int, src));
        }
        BC_MOVE_R => {
            let dst = u16_at!();
            let src = u16_at!();
            writes.push((Bank::Ref, dst));
            reads.push((Bank::Ref, src));
        }
        BC_MOVE_F => {
            let dst = u16_at!();
            let src = u16_at!();
            writes.push((Bank::Float, dst));
            reads.push((Bank::Float, src));
        }

        // -- STACK POP: write register --
        BC_POP_I => {
            let dst = u16_at!();
            writes.push((Bank::Int, dst));
        }
        BC_POP_R => {
            let dst = u16_at!();
            writes.push((Bank::Ref, dst));
        }
        BC_POP_F => {
            let dst = u16_at!();
            writes.push((Bank::Float, dst));
        }

        // -- STACK PUSH: read register --
        BC_PUSH_I => {
            let src = u16_at!();
            reads.push((Bank::Int, src));
        }
        BC_PUSH_R => {
            let src = u16_at!();
            reads.push((Bank::Ref, src));
        }
        BC_PUSH_F => {
            let src = u16_at!();
            reads.push((Bank::Float, src));
        }
        BC_PUSH_TO => {
            let _sel = u16_at!();
            let src = u16_at!();
            reads.push((Bank::Int, src));
        }

        // -- STACK misc: no register operands --
        BC_POP_DISCARD | BC_DUP_STACK | BC_SWAP_STACK => {}
        BC_PEEK_I => {
            let dst = u16_at!();
            let _stack_pos = u16_at!();
            writes.push((Bank::Int, dst));
        }
        BC_COPY_FROM_BOTTOM => {
            let _stack_pos = u16_at!();
        }
        BC_STORE_DOWN => {
            let _stack_pos = u16_at!();
        }
        BC_REQUIRE_STACK => {
            let _count = u16_at!();
        }

        // -- ARITHMETIC --
        BC_RECORD_BINOP_I => {
            let dst = u16_at!();
            let _opcode_idx = u16_at!();
            let lhs = u16_at!();
            let rhs = u16_at!();
            writes.push((Bank::Int, dst));
            reads.push((Bank::Int, lhs));
            reads.push((Bank::Int, rhs));
        }
        BC_RECORD_UNARY_I => {
            let dst = u16_at!();
            let _opcode_idx = u16_at!();
            let src = u16_at!();
            writes.push((Bank::Int, dst));
            reads.push((Bank::Int, src));
        }
        BC_RECORD_BINOP_F => {
            let dst = u16_at!();
            let _opcode_idx = u16_at!();
            let lhs = u16_at!();
            let rhs = u16_at!();
            writes.push((Bank::Float, dst));
            reads.push((Bank::Float, lhs));
            reads.push((Bank::Float, rhs));
        }
        BC_RECORD_UNARY_F => {
            let dst = u16_at!();
            let _opcode_idx = u16_at!();
            let src = u16_at!();
            writes.push((Bank::Float, dst));
            reads.push((Bank::Float, src));
        }

        // -- CONTROL FLOW --
        BC_BRANCH_ZERO => {
            // Stack-based branch, no register operands.
            // Target is implicit (PC after condition eval).
            // Actually this opcode is not used by pyre codewriter.
        }
        BC_BRANCH_REG_ZERO => {
            let cond = u16_at!();
            let target = u16_at!();
            reads.push((Bank::Int, cond));
            branch_target = Some(target as usize);
        }
        BC_JUMP => {
            let target = u16_at!();
            branch_target = Some(target as usize);
            falls_through = false;
        }
        BC_JUMP_TARGET => {
            // Label marker, no operands.
        }
        BC_SET_SELECTED => {
            let _sel = u16_at!();
        }

        // -- RETURN/RAISE: terminates --
        BC_REF_RETURN => {
            let src = u16_at!();
            reads.push((Bank::Ref, src));
            falls_through = false;
        }
        BC_ABORT | BC_ABORT_PERMANENT => {
            falls_through = false;
        }
        BC_RAISE => {
            let src = u16_at!();
            reads.push((Bank::Ref, src));
            falls_through = false;
        }
        BC_RERAISE => {
            falls_through = false;
        }
        BC_JIT_MERGE_POINT => {
            // JIT_MERGE_POINT is a marker — not a terminator. The
            // blackhole (machine.rs) either closes the loop OR falls
            // through to the next instruction. For liveness purposes
            // it must fall through so that registers used after the
            // merge point are visible to the backward analysis.
            // RPython: `-live-` markers always fall through.
            //
            // New format: [jdindex][len_gi][regs...][len_gr][regs...]
            //   [len_gf][regs...][len_ri][regs...][len_rr][regs...]
            //   [len_rf][regs...]
            // Skip jdindex + 6 list-of-kind sections, read red_r for GEN.
            pos += 1; // jdindex
            // green_i, green_r, green_f
            for _ in 0..3 {
                let list_len = code[pos] as usize;
                pos += 1 + list_len;
            }
            // red_i
            let len_ri = code[pos] as usize;
            pos += 1 + len_ri;
            // red_r: these are READ at the merge point — GEN them.
            let len_rr = code[pos] as usize;
            pos += 1;
            for _ in 0..len_rr {
                let reg = code[pos] as u16;
                pos += 1;
                reads.push((Bank::Ref, reg));
            }
            // red_f
            let len_rf = code[pos] as usize;
            pos += 1 + len_rf;
        }

        // -- VABLE FIELD ACCESS --
        BC_GETFIELD_VABLE_I => {
            let _field = u16_at!();
            let dst = u16_at!();
            writes.push((Bank::Int, dst));
        }
        BC_GETFIELD_VABLE_R => {
            let _field = u16_at!();
            let dst = u16_at!();
            writes.push((Bank::Ref, dst));
        }
        BC_GETFIELD_VABLE_F => {
            let _field = u16_at!();
            let dst = u16_at!();
            writes.push((Bank::Float, dst));
        }
        BC_SETFIELD_VABLE_I => {
            let _field = u16_at!();
            let src = u16_at!();
            reads.push((Bank::Int, src));
        }
        BC_SETFIELD_VABLE_R => {
            let _field = u16_at!();
            let src = u16_at!();
            reads.push((Bank::Ref, src));
        }
        BC_SETFIELD_VABLE_F => {
            let _field = u16_at!();
            let src = u16_at!();
            reads.push((Bank::Float, src));
        }

        // -- VABLE ARRAY ACCESS --
        BC_GETARRAYITEM_VABLE_I => {
            let _arr = u16_at!();
            let idx = u16_at!();
            let dst = u16_at!();
            reads.push((Bank::Int, idx));
            writes.push((Bank::Int, dst));
        }
        BC_GETARRAYITEM_VABLE_R => {
            let _arr = u16_at!();
            let idx = u16_at!();
            let dst = u16_at!();
            reads.push((Bank::Int, idx));
            writes.push((Bank::Ref, dst));
        }
        BC_GETARRAYITEM_VABLE_F => {
            let _arr = u16_at!();
            let idx = u16_at!();
            let dst = u16_at!();
            reads.push((Bank::Int, idx));
            writes.push((Bank::Float, dst));
        }
        BC_SETARRAYITEM_VABLE_I => {
            let _arr = u16_at!();
            let idx = u16_at!();
            let src = u16_at!();
            reads.push((Bank::Int, idx));
            reads.push((Bank::Int, src));
        }
        BC_SETARRAYITEM_VABLE_R => {
            let _arr = u16_at!();
            let idx = u16_at!();
            let src = u16_at!();
            reads.push((Bank::Int, idx));
            reads.push((Bank::Ref, src));
        }
        BC_SETARRAYITEM_VABLE_F => {
            let _arr = u16_at!();
            let idx = u16_at!();
            let src = u16_at!();
            reads.push((Bank::Int, idx));
            reads.push((Bank::Float, src));
        }
        BC_ARRAYLEN_VABLE => {
            let _arr = u16_at!();
            let dst = u16_at!();
            writes.push((Bank::Int, dst));
        }
        BC_HINT_FORCE_VIRTUALIZABLE => {}

        // -- GUARD VALUE --
        BC_INT_GUARD_VALUE => {
            let src = u16_at!();
            reads.push((Bank::Int, src));
        }
        BC_REF_GUARD_VALUE => {
            let src = u16_at!();
            reads.push((Bank::Ref, src));
        }
        BC_FLOAT_GUARD_VALUE => {
            let src = u16_at!();
            reads.push((Bank::Float, src));
        }

        // -- STATE ACCESS (no-op for registers) --
        BC_LOAD_STATE_FIELD => {
            let _field = u16_at!();
            let _dst = u16_at!();
        }
        BC_STORE_STATE_FIELD => {
            let _field = u16_at!();
            let _src = u16_at!();
        }
        BC_LOAD_STATE_VARRAY => {
            let _field = u16_at!();
            let _dst = u16_at!();
        }
        BC_STORE_STATE_VARRAY => {
            let _field = u16_at!();
            let _src = u16_at!();
        }
        BC_LOAD_STATE_ARRAY => {
            let _arr = u16_at!();
            let _elem = u16_at!();
            let _dst = u16_at!();
        }
        BC_STORE_STATE_ARRAY => {
            let _arr = u16_at!();
            let _elem = u16_at!();
            let _src = u16_at!();
        }

        // -- RECORD_KNOWN_RESULT --
        BC_RECORD_KNOWN_RESULT_INT => {
            let _result = u16_at!();
            let _arg1 = u16_at!();
            let _arg2 = u16_at!();
        }
        BC_RECORD_KNOWN_RESULT_REF => {
            let _result = u16_at!();
            let _arg1 = u16_at!();
            let _arg2 = u16_at!();
        }

        // -- COND_CALL --
        BC_COND_CALL_VOID | BC_COND_CALL_VALUE_INT | BC_COND_CALL_VALUE_REF => {
            // Variable-length: parse like call ops
            let _fn_ptr_idx = u16_at!();
            // For VALUE variants, dst comes before nargs
            let has_dst = opcode != BC_COND_CALL_VOID;
            let dst = if has_dst { Some(u16_at!()) } else { None };
            let num_args = u16_at!() as usize;
            for _ in 0..num_args {
                let kind = JitArgKind::decode(code[pos]);
                pos += 1;
                let reg = u16_at!();
                match kind {
                    JitArgKind::Int => reads.push((Bank::Int, reg)),
                    JitArgKind::Ref => reads.push((Bank::Ref, reg)),
                    JitArgKind::Float => reads.push((Bank::Float, reg)),
                }
            }
            if let Some(d) = dst {
                let bank = if opcode == BC_COND_CALL_VALUE_INT {
                    Bank::Int
                } else {
                    Bank::Ref
                };
                writes.push((bank, d));
            }
        }

        // -- FUNCTION CALLS (int/ref/float/void typed) --
        // All follow: fn_ptr_idx:u16, dst:u16 (except void), nargs:u16, args...
        BC_CALL_INT
        | BC_CALL_PURE_INT
        | BC_CALL_MAY_FORCE_INT
        | BC_CALL_RELEASE_GIL_INT
        | BC_CALL_LOOPINVARIANT_INT
        | BC_CALL_ASSEMBLER_INT => {
            let _fn = u16_at!();
            let dst = u16_at!();
            let nargs = u16_at!() as usize;
            parse_call_args(code, &mut pos, nargs, &mut reads);
            writes.push((Bank::Int, dst));
        }
        BC_CALL_REF
        | BC_CALL_PURE_REF
        | BC_CALL_MAY_FORCE_REF
        | BC_CALL_RELEASE_GIL_REF
        | BC_CALL_LOOPINVARIANT_REF
        | BC_CALL_ASSEMBLER_REF => {
            let _fn = u16_at!();
            let dst = u16_at!();
            let nargs = u16_at!() as usize;
            parse_call_args(code, &mut pos, nargs, &mut reads);
            writes.push((Bank::Ref, dst));
        }
        BC_CALL_FLOAT
        | BC_CALL_PURE_FLOAT
        | BC_CALL_MAY_FORCE_FLOAT
        | BC_CALL_RELEASE_GIL_FLOAT
        | BC_CALL_LOOPINVARIANT_FLOAT
        | BC_CALL_ASSEMBLER_FLOAT => {
            let _fn = u16_at!();
            let dst = u16_at!();
            let nargs = u16_at!() as usize;
            parse_call_args(code, &mut pos, nargs, &mut reads);
            writes.push((Bank::Float, dst));
        }
        BC_CALL_MAY_FORCE_VOID
        | BC_CALL_RELEASE_GIL_VOID
        | BC_CALL_LOOPINVARIANT_VOID
        | BC_CALL_ASSEMBLER_VOID
        | BC_RESIDUAL_CALL_VOID => {
            let _fn = u16_at!();
            let nargs = u16_at!() as usize;
            parse_call_args(code, &mut pos, nargs, &mut reads);
        }

        // -- INLINE_CALL: complex --
        BC_INLINE_CALL => {
            let _sub_idx = u16_at!();
            let num_args = u16_at!() as usize;
            for _ in 0..num_args {
                let kind = JitArgKind::decode(code[pos]);
                pos += 1;
                let caller_src = u16_at!();
                let _callee_dst = u16_at!();
                match kind {
                    JitArgKind::Int => reads.push((Bank::Int, caller_src)),
                    JitArgKind::Ref => reads.push((Bank::Ref, caller_src)),
                    JitArgKind::Float => reads.push((Bank::Float, caller_src)),
                }
            }
            // 3 return slot pairs: (callee_src:u16, caller_dst:u16) for i/r/f
            for bank in [Bank::Int, Bank::Ref, Bank::Float] {
                let _callee_src = u16_at!();
                let caller_dst = u16_at!();
                if caller_dst != u16::MAX {
                    writes.push((bank, caller_dst));
                }
            }
        }

        _ => {
            // Unknown opcode: conservatively skip 0 bytes.
            // This will cause the parser to fail at the next iteration.
            // In practice, all opcodes used by pyre's codewriter are covered above.
            if crate::majit_log_enabled() {
                eprintln!("[jit][liveness] unknown opcode {} at pc={}", opcode, pc);
            }
        }
    }

    InsnInfo {
        next_pc: pos,
        reads,
        writes,
        branch_target,
        falls_through,
    }
}

/// Parse call arguments: (u8 kind, u16 reg) × nargs
fn parse_call_args(code: &[u8], pos: &mut usize, nargs: usize, reads: &mut Vec<(Bank, u16)>) {
    for _ in 0..nargs {
        let kind = JitArgKind::decode(code[*pos]);
        *pos += 1;
        let reg = read_u16(code, *pos);
        *pos += 2;
        match kind {
            JitArgKind::Int => reads.push((Bank::Int, reg)),
            JitArgKind::Ref => reads.push((Bank::Ref, reg)),
            JitArgKind::Float => reads.push((Bank::Float, reg)),
        }
    }
}

/// Compute precise jitcode-level liveness at each Python PC.
///
/// RPython: `liveness.py:compute_liveness` runs backward dataflow on
/// SSA instructions. This function does the equivalent on assembled
/// JitCode bytecodes.
///
/// Returns `Vec<LivenessInfo>` with one entry per Python PC that has
/// a corresponding jitcode PC (via `py_to_jit_pc`).
pub fn compute_liveness_bytecode(jitcode: &JitCode) -> Vec<LivenessInfo> {
    let code = &jitcode.code;
    if code.is_empty() {
        return Vec::new();
    }

    // Phase 1: parse all instructions, build instruction list with PCs.
    let mut insns: Vec<(usize, InsnInfo)> = Vec::new();
    let mut pc = 0;
    while pc < code.len() {
        let info = parse_insn(code, pc);
        let this_pc = pc;
        pc = info.next_pc;
        insns.push((this_pc, info));
    }

    // Phase 2: backward dataflow analysis.
    // Track live registers as three bitsets (one per bank, 256 bits each).
    let n = insns.len();
    // live_i[insn_idx], live_r[insn_idx], live_f[insn_idx]: bitsets BEFORE each insn.
    let mut live_i: Vec<[u64; 4]> = vec![[0u64; 4]; n + 1];
    let mut live_r: Vec<[u64; 4]> = vec![[0u64; 4]; n + 1];
    let mut live_f: Vec<[u64; 4]> = vec![[0u64; 4]; n + 1];

    // Map from jitcode PC → instruction index for branch resolution.
    let mut pc_to_idx: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (idx, &(pc, _)) in insns.iter().enumerate() {
        pc_to_idx.insert(pc, idx);
    }

    fn set_bit(bv: &mut [u64; 4], reg: u16) {
        let r = reg as usize;
        if r < 256 {
            bv[r / 64] |= 1u64 << (r % 64);
        }
    }
    fn clear_bit(bv: &mut [u64; 4], reg: u16) {
        let r = reg as usize;
        if r < 256 {
            bv[r / 64] &= !(1u64 << (r % 64));
        }
    }
    fn union(dst: &mut [u64; 4], src: &[u64; 4]) -> bool {
        let mut changed = false;
        for i in 0..4 {
            let old = dst[i];
            dst[i] |= src[i];
            if dst[i] != old {
                changed = true;
            }
        }
        changed
    }
    fn bv_to_regs(bv: &[u64; 4]) -> Vec<u16> {
        let mut regs = Vec::new();
        for word in 0..4 {
            let mut bits = bv[word];
            while bits != 0 {
                let bit = bits.trailing_zeros() as u16;
                regs.push(word as u16 * 64 + bit);
                bits &= bits - 1;
            }
        }
        regs
    }

    // Fixed-point iteration.
    let mut changed = true;
    while changed {
        changed = false;
        for idx in (0..n).rev() {
            let (_, ref info) = insns[idx];
            // Start with successor's live set.
            let mut cur_i = if info.falls_through && idx + 1 <= n {
                live_i[idx + 1]
            } else {
                [0u64; 4]
            };
            let mut cur_r = if info.falls_through && idx + 1 <= n {
                live_r[idx + 1]
            } else {
                [0u64; 4]
            };
            let mut cur_f = if info.falls_through && idx + 1 <= n {
                live_f[idx + 1]
            } else {
                [0u64; 4]
            };
            // Union with branch target.
            if let Some(target) = info.branch_target {
                if let Some(&tidx) = pc_to_idx.get(&target) {
                    union(&mut cur_i, &live_i[tidx]);
                    union(&mut cur_r, &live_r[tidx]);
                    union(&mut cur_f, &live_f[tidx]);
                }
            }
            // KILL: remove written registers.
            // liveness.py:61-64: `args[-2] == '->'` discards the dest.
            for &(bank, reg) in &info.writes {
                match bank {
                    Bank::Int => clear_bit(&mut cur_i, reg),
                    Bank::Ref => clear_bit(&mut cur_r, reg),
                    Bank::Float => clear_bit(&mut cur_f, reg),
                }
            }
            // GEN: add read registers.
            for &(bank, reg) in &info.reads {
                match bank {
                    Bank::Int => set_bit(&mut cur_i, reg),
                    Bank::Ref => set_bit(&mut cur_r, reg),
                    Bank::Float => set_bit(&mut cur_f, reg),
                }
            }
            // Monotonic update: live sets only grow (fixed-point).
            // liveness.py:38-41: alive_at_point.update(alive).
            if union(&mut live_i[idx], &cur_i) {
                changed = true;
            }
            if union(&mut live_r[idx], &cur_r) {
                changed = true;
            }
            if union(&mut live_f[idx], &cur_f) {
                changed = true;
            }
        }
    }

    // Phase 3: extract LivenessInfo at each Python PC boundary.
    // py_to_jit_pc is Vec<usize> indexed by py_pc.
    let mut result: Vec<LivenessInfo> = Vec::new();
    for (py_pc, &jit_pc) in jitcode.py_to_jit_pc.iter().enumerate() {
        let _ = py_pc;
        if let Some(&idx) = pc_to_idx.get(&jit_pc) {
            result.push(LivenessInfo {
                pc: jit_pc as u16,
                live_i_regs: bv_to_regs(&live_i[idx]),
                live_r_regs: bv_to_regs(&live_r[idx]),
                live_f_regs: bv_to_regs(&live_f[idx]),
            });
        }
    }
    result.sort_by_key(|l| l.pc);
    // Deduplicate (same jit_pc from multiple py_pcs).
    result.dedup_by_key(|l| l.pc);
    result
}
