/// JIT-enabled tinyframe interpreter using JitDriver + JitState.
///
/// Greens: [pc, code]       (bytecode and position are loop constants)
/// Reds:   [registers]      (register values are the live state)
///
/// RPython correspondence (tinyframe.py):
///   - JitDriver(greens=['i', 'code'], reds=['self'], virtualizables=['self'])
///     — tinyframe.py:215
///   - _virtualizable_ = ['registers[*]', 'code'] — tinyframe.py:219
///     Implicit via TinyFrameState: regs vec serves as virtualizable array.
///   - hint(self, access_directly=True, fresh_virtualizable=True) — tinyframe.py:222
///     Implicit: state_fields / extract_live / restore handle this.
///   - @dont_look_inside Frame.introspect() — tinyframe.py:272
///     Not traced here; INTROSPECT would abort tracing.
///   - CALL, LOAD_FUNCTION, PRINT — object/function ops, not traced (interpreter fallback).
///
/// JIT traces the integer-only path through the register machine.
/// JUMP_IF_ABOVE is the back-edge that triggers tracing.
///
/// This example hand-writes `trace_instruction` for educational purposes.
/// In production, the `#[jit_interp]` proc macro auto-generates tracing
/// code from the interpreter's match dispatch — see aheuijit for an example.
use std::collections::HashMap;

use majit_ir::{OpCode, OpRef};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

use crate::interp::{ADD, Code, JUMP_IF_ABOVE, LOAD, RETURN};

const DEFAULT_THRESHOLD: u32 = 3;

// ── JitState types ──

/// Red variables: all registers.
/// Corresponds to _virtualizable_ = ['registers[*]', 'code'] in tinyframe.py:219.
/// Registers are the virtualizable array — carried as JUMP args in compiled traces.
pub struct TinyFrameState {
    regs: Vec<i64>,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct TinyFrameMeta {
    #[allow(dead_code)]
    header_pc: usize,
    /// Number of registers (all are live).
    num_regs: usize,
}

/// Symbolic state during tracing — OpRef for each register.
pub struct TinyFrameSym {
    /// Register index → current OpRef.
    regs: HashMap<usize, OpRef>,
    /// Number of registers (determines inputarg/jump_arg count).
    num_regs: usize,
}

impl JitState for TinyFrameState {
    type Meta = TinyFrameMeta;
    type Sym = TinyFrameSym;
    type Env = [u8];

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> TinyFrameMeta {
        TinyFrameMeta {
            header_pc,
            num_regs: self.regs.len(),
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        self.regs[..meta.num_regs].to_vec()
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> TinyFrameSym {
        let mut regs = HashMap::new();
        for i in 0..meta.num_regs {
            regs.insert(i, OpRef(i as u32));
        }
        TinyFrameSym {
            regs,
            num_regs: meta.num_regs,
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        self.regs.len() == meta.num_regs
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        for i in 0..meta.num_regs {
            self.regs[i] = values[i];
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        (0..sym.num_regs)
            .map(|i| *sym.regs.get(&i).unwrap())
            .collect()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

/// Trace one instruction, recording IR into ctx.
fn trace_instruction(
    ctx: &mut TraceCtx,
    sym: &mut TinyFrameSym,
    bytecode: &[u8],
    pc: usize,
    header_pc: usize,
) -> TraceAction {
    let opcode = bytecode[pc];

    if opcode == LOAD {
        let val = bytecode[pc + 1] as i64;
        let reg = bytecode[pc + 2] as usize;
        let opref = ctx.const_int(val);
        sym.regs.insert(reg, opref);
    } else if opcode == ADD {
        let r1 = bytecode[pc + 1] as usize;
        let r2 = bytecode[pc + 2] as usize;
        let r3 = bytecode[pc + 3] as usize;
        let a = *sym.regs.get(&r1).unwrap();
        let b = *sym.regs.get(&r2).unwrap();
        let result = ctx.record_op(OpCode::IntAdd, &[a, b]);
        sym.regs.insert(r3, result);
    } else if opcode == JUMP_IF_ABOVE {
        let r1 = bytecode[pc + 1] as usize;
        let r2 = bytecode[pc + 2] as usize;
        let tgt = bytecode[pc + 3] as usize;

        if tgt == header_pc {
            // Back-edge to loop header: guard and close.
            let a = *sym.regs.get(&r1).unwrap();
            let b = *sym.regs.get(&r2).unwrap();
            let cond = ctx.record_op(OpCode::IntGt, &[a, b]);
            ctx.record_guard(OpCode::GuardTrue, &[cond], sym.num_regs);
            return TraceAction::CloseLoop;
        }
        // Non-loop branch: abort.
        return TraceAction::Abort;
    } else if opcode == RETURN {
        return TraceAction::Abort;
    } else {
        // INTROSPECT (@dont_look_inside in tinyframe.py:272),
        // CALL, LOAD_FUNCTION, PRINT — not traced, abort to interpreter.
        return TraceAction::Abort;
    }

    TraceAction::Continue
}

pub struct JitTinyFrameInterp {
    driver: JitDriver<TinyFrameState>,
}

impl JitTinyFrameInterp {
    pub fn new() -> Self {
        JitTinyFrameInterp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    /// Run a tinyframe Code with initial integer register values.
    /// `init_regs` maps register index → initial i64 value.
    pub fn run(&mut self, code: &Code, init_regs: &[(usize, i64)]) -> i64 {
        let mut state = TinyFrameState {
            regs: vec![0; code.regno],
        };
        for &(r, v) in init_regs {
            state.regs[r] = v;
        }
        let bytecode = &code.code;
        let mut pc: usize = 0;

        loop {
            if pc >= bytecode.len() {
                break;
            }

            // jit_merge_point
            let header_pc = self
                .driver
                .current_trace_green_key()
                .map(|k| k as usize)
                .unwrap_or(0);
            self.driver
                .merge_point(|ctx, sym| trace_instruction(ctx, sym, bytecode, pc, header_pc));

            let opcode = bytecode[pc];

            if opcode == LOAD {
                let val = bytecode[pc + 1] as i64;
                let reg = bytecode[pc + 2] as usize;
                state.regs[reg] = val;
                pc += 3;
            } else if opcode == ADD {
                let r1 = bytecode[pc + 1] as usize;
                let r2 = bytecode[pc + 2] as usize;
                let r3 = bytecode[pc + 3] as usize;
                state.regs[r3] = state.regs[r1] + state.regs[r2];
                pc += 4;
            } else if opcode == RETURN {
                let r = bytecode[pc + 1] as usize;
                return state.regs[r];
            } else if opcode == JUMP_IF_ABOVE {
                let r1 = bytecode[pc + 1] as usize;
                let r2 = bytecode[pc + 2] as usize;
                let tgt = bytecode[pc + 3] as usize;
                if state.regs[r1] > state.regs[r2] {
                    // can_enter_jit: back-edge
                    if tgt < pc {
                        if self.driver.back_edge(tgt, &mut state, bytecode, || {}) {
                            pc = tgt;
                            continue;
                        }
                    }
                    pc = tgt;
                } else {
                    pc += 4;
                }
            } else {
                panic!("unsupported opcode in JIT path: {opcode}");
            }
        }

        panic!("fell off end of code");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    #[test]
    fn jit_loop_count_to_100() {
        let code = interp::compile(
            "
        main:
        LOAD 1 => r1
        LOAD 100 => r2
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        RETURN r0
        ",
        );
        let mut jit = JitTinyFrameInterp::new();
        let result = jit.run(&code, &[]);
        assert_eq!(result, 100);
    }

    #[test]
    fn jit_loop_sum() {
        // loop.tf: sum from 1 to N
        let code = interp::compile(
            "
        main:
        LOAD 0 => r1
        LOAD 1 => r2
        @add
        ADD r2 r1 => r1
        JUMP_IF_ABOVE r0 r1 @add
        RETURN r1
        ",
        );
        let mut jit = JitTinyFrameInterp::new();
        let result = jit.run(&code, &[(0, 100)]);
        assert_eq!(result, 100);
    }

    #[test]
    fn jit_matches_interp() {
        let code = interp::compile(
            "
        main:
        LOAD 1 => r1
        LOAD 0 => r0
        @l1
        ADD r0 r1 => r0
        JUMP_IF_ABOVE r2 r0 @l1
        RETURN r0
        ",
        );

        for n in [10, 50, 100, 255] {
            // Interpreter
            let mut frame = interp::Frame::new(&code);
            frame.registers[2] = Some(interp::Object::Int(n));
            let interp_result = frame.interpret(&code).as_int();

            // JIT
            let mut jit = JitTinyFrameInterp::new();
            let jit_result = jit.run(&code, &[(2, n)]);

            assert_eq!(jit_result, interp_result, "count_to({n}) mismatch");
        }
    }
}
