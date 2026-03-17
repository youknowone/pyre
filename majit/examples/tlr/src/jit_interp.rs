/// JIT-enabled TLR interpreter using JitDriver + JitState.
///
/// Greens: [pc, bytecode]   (bytecode is constant per trace — not tracked)
/// Reds:   [a, regs]
///
/// Mirrors rpython/jit/tl/tlr.py with jit_merge_point / can_enter_jit.
use majit_ir::{OpCode, OpRef};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

const MOV_A_R: u8 = 1;
const MOV_R_A: u8 = 2;
const JUMP_IF_A: u8 = 3;
const SET_A: u8 = 4;
const ADD_R_TO_A: u8 = 5;
const RETURN_A: u8 = 6;
const ALLOCATE: u8 = 7;
const NEG_A: u8 = 8;

const DEFAULT_THRESHOLD: u32 = 3;

/// Virtual register index for the accumulator.
const ACC: u8 = 255;

// ── JitState types ──

/// Red variables: accumulator + registers.
pub struct TlrState {
    a: i64,
    regs: Vec<i64>,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct TlrMeta {
    header_pc: usize,
    /// Which registers (including ACC=255) are live at the loop header.
    live_regs: Vec<u8>,
}

/// Symbolic state during tracing — OpRef for each live register.
pub struct TlrSym {
    /// Maps register index (or ACC) → current OpRef.
    regs: std::collections::HashMap<u8, OpRef>,
    /// Ordered list of live register indices (same order as inputargs).
    live_regs: Vec<u8>,
}

impl TlrSym {
    fn get_or_const(&self, ctx: &mut TraceCtx, reg: u8, runtime_val: i64) -> OpRef {
        self.regs
            .get(&reg)
            .copied()
            .unwrap_or_else(|| ctx.const_int(runtime_val))
    }
}

impl JitState for TlrState {
    type Meta = TlrMeta;
    type Sym = TlrSym;
    type Env = [u8];

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> TlrMeta {
        let live_regs = scan_live(env, header_pc, self.regs.len());
        TlrMeta {
            header_pc,
            live_regs,
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        meta.live_regs
            .iter()
            .map(|&r| {
                if r == ACC {
                    self.a
                } else {
                    self.regs[r as usize]
                }
            })
            .collect()
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> TlrSym {
        // Inputargs OpRef(0), OpRef(1), ... correspond to live_regs in order.
        let mut regs = std::collections::HashMap::new();
        for (i, &r) in meta.live_regs.iter().enumerate() {
            regs.insert(r, OpRef(i as u32));
        }
        TlrSym {
            regs,
            live_regs: meta.live_regs.clone(),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        meta.live_regs.iter().all(|&r| {
            r == ACC || (r as usize) < self.regs.len()
        })
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        for (i, &r) in meta.live_regs.iter().enumerate() {
            if r == ACC {
                self.a = values[i];
            } else {
                self.regs[r as usize] = values[i];
            }
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.live_regs
            .iter()
            .filter_map(|r| sym.regs.get(r).copied())
            .collect()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

/// Scan bytecode to determine which registers are live at the loop header.
/// All registers + accumulator are conservatively marked live.
fn scan_live(_bytecode: &[u8], _header_pc: usize, num_regs: usize) -> Vec<u8> {
    let mut live = vec![ACC];
    for i in 0..num_regs.min(256) {
        live.push(i as u8);
    }
    live
}

/// Trace one instruction, recording IR into ctx.
fn trace_instruction(
    ctx: &mut TraceCtx,
    sym: &mut TlrSym,
    bytecode: &[u8],
    pc: usize,
    state: &TlrState,
    header_pc: usize,
) -> TraceAction {
    let opcode = bytecode[pc];

    if opcode == SET_A {
        let val = bytecode[pc + 1] as i64;
        let opref = ctx.const_int(val);
        sym.regs.insert(ACC, opref);
    } else if opcode == MOV_A_R {
        let n = bytecode[pc + 1];
        let acc = sym.get_or_const(ctx, ACC, state.a);
        sym.regs.insert(n, acc);
    } else if opcode == MOV_R_A {
        let n = bytecode[pc + 1];
        let reg = sym.get_or_const(ctx, n, state.regs[n as usize]);
        sym.regs.insert(ACC, reg);
    } else if opcode == ADD_R_TO_A {
        let n = bytecode[pc + 1];
        let acc = sym.get_or_const(ctx, ACC, state.a);
        let reg = sym.get_or_const(ctx, n, state.regs[n as usize]);
        let result = ctx.record_op(OpCode::IntAdd, &[acc, reg]);
        sym.regs.insert(ACC, result);
    } else if opcode == NEG_A {
        let acc = sym.get_or_const(ctx, ACC, state.a);
        let result = ctx.record_op(OpCode::IntNeg, &[acc]);
        sym.regs.insert(ACC, result);
    } else if opcode == JUMP_IF_A {
        let target = bytecode[pc + 1] as usize;
        if target == header_pc {
            // Back-edge to loop header: guard a != 0, close loop.
            let acc = sym.get_or_const(ctx, ACC, state.a);
            let num_live = sym.live_regs.len();
            ctx.record_guard(OpCode::GuardTrue, &[acc], num_live);
            return TraceAction::CloseLoop;
        }
    } else if opcode == RETURN_A {
        return TraceAction::Abort;
    }
    // ALLOCATE: no-op during tracing (regs already exist)

    TraceAction::Continue
}

pub struct JitTlrInterp {
    driver: JitDriver<TlrState>,
}

impl JitTlrInterp {
    pub fn new() -> Self {
        JitTlrInterp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    pub fn run(&mut self, bytecode: &[u8], initial_a: i64) -> i64 {
        let mut state = TlrState {
            a: initial_a,
            regs: Vec::new(),
        };
        let mut pc: usize = 0;

        loop {
            // jit_merge_point(bytecode=bytecode, pc=pc, a=a, regs=regs)
            let header_pc = self.driver.current_trace_green_key()
                .map(|k| k as usize)
                .unwrap_or(0);
            self.driver.merge_point(|ctx, sym| {
                trace_instruction(ctx, sym, bytecode, pc, &state, header_pc)
            });

            let opcode = bytecode[pc];
            pc += 1;

            if opcode == MOV_A_R {
                let n = bytecode[pc] as usize;
                pc += 1;
                state.regs[n] = state.a;
            } else if opcode == MOV_R_A {
                let n = bytecode[pc] as usize;
                pc += 1;
                state.a = state.regs[n];
            } else if opcode == JUMP_IF_A {
                let target = bytecode[pc] as usize;
                pc += 1;
                if state.a != 0 {
                    // can_enter_jit(bytecode=bytecode, pc=target, a=a, regs=regs)
                    if target < pc {
                        if self.driver.back_edge(target, &mut state, bytecode, || {}) {
                            pc = target;
                            continue;
                        }
                    }
                    pc = target;
                }
            } else if opcode == SET_A {
                state.a = bytecode[pc] as i64;
                pc += 1;
            } else if opcode == ADD_R_TO_A {
                let n = bytecode[pc] as usize;
                pc += 1;
                state.a += state.regs[n];
            } else if opcode == RETURN_A {
                return state.a;
            } else if opcode == ALLOCATE {
                let n = bytecode[pc] as usize;
                pc += 1;
                state.regs = vec![0; n];
            } else if opcode == NEG_A {
                state.a = -state.a;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    fn square_bytecode() -> Vec<u8> {
        vec![
            ALLOCATE, 3, MOV_A_R, 0, MOV_A_R, 1, SET_A, 0, MOV_A_R, 2, SET_A, 1, NEG_A, ADD_R_TO_A,
            0, MOV_A_R, 0, MOV_R_A, 2, ADD_R_TO_A, 1, MOV_A_R, 2, MOV_R_A, 0, JUMP_IF_A, 10,
            MOV_R_A, 2, RETURN_A,
        ]
    }

    #[test]
    fn jit_square_5() {
        let bc = square_bytecode();
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&bc, 5), 25);
    }

    #[test]
    fn jit_square_100() {
        let bc = square_bytecode();
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&bc, 100), 10_000);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = square_bytecode();
        for a in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::interpret(&bc, a);
            let mut jit = JitTlrInterp::new();
            let got = jit.run(&bc, a);
            assert_eq!(got, expected, "mismatch for a={a}");
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![SET_A, 42, RETURN_A];
        let mut jit = JitTlrInterp::new();
        assert_eq!(jit.run(&prog, 0), 42);
    }
}
