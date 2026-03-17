/// JIT-enabled TLA interpreter using JitDriver + JitState.
///
/// TLA uses wrapped objects (W_IntObject, W_StringObject), but the JIT
/// specializes for integers: the trace tracks OpRef values representing
/// raw i64 values, bypassing object allocation and dispatch.
///
/// Greens: [pc, bytecode]
/// Reds:   [stack]
use majit_ir::{OpCode, OpRef};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

use crate::interp::WObject;

const CONST_INT: u8 = 0;
const POP: u8 = 1;
const ADD: u8 = 2;
const RETURN: u8 = 3;
const JUMP_IF: u8 = 4;
const DUP: u8 = 5;
const SUB: u8 = 6;

const DEFAULT_THRESHOLD: u32 = 3;

// ── JitState types ──

/// Red variables: the integer stack.
pub struct TlaState {
    stack: Vec<i64>,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct TlaMeta {
    #[allow(dead_code)]
    header_pc: usize,
    /// Number of stack slots live at the loop header.
    num_stack_slots: usize,
}

/// Symbolic state during tracing — OpRef for each stack slot.
pub struct TlaSym {
    /// Symbolic stack: OpRef for each slot position.
    trace_stack: Vec<OpRef>,
    /// Number of stack slots at the loop header (for jump_args).
    num_stack_slots: usize,
}

impl JitState for TlaState {
    type Meta = TlaMeta;
    type Sym = TlaSym;
    type Env = [u8];

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> TlaMeta {
        TlaMeta {
            header_pc,
            num_stack_slots: self.stack.len(),
        }
    }

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64> {
        self.stack[..meta.num_stack_slots].to_vec()
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> TlaSym {
        let mut trace_stack = Vec::with_capacity(meta.num_stack_slots);
        for i in 0..meta.num_stack_slots {
            trace_stack.push(OpRef(i as u32));
        }
        TlaSym {
            trace_stack,
            num_stack_slots: meta.num_stack_slots,
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        self.stack.len() >= meta.num_stack_slots
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        self.stack.clear();
        self.stack.extend_from_slice(&values[..meta.num_stack_slots]);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.trace_stack[..sym.num_stack_slots].to_vec()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

/// Trace one instruction, recording IR into ctx.
fn trace_instruction(
    ctx: &mut TraceCtx,
    sym: &mut TlaSym,
    bytecode: &[u8],
    pc: usize,
    runtime_stack: &[i64],
    header_pc: usize,
) -> TraceAction {
    let opcode = bytecode[pc];

    if opcode == CONST_INT {
        let value = bytecode[pc + 1] as i64;
        let opref = ctx.const_int(value);
        sym.trace_stack.push(opref);
    } else if opcode == POP {
        sym.trace_stack.pop();
    } else if opcode == DUP {
        let top = *sym.trace_stack.last().unwrap();
        sym.trace_stack.push(top);
    } else if opcode == ADD {
        let b = sym.trace_stack.pop().unwrap();
        let a = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntAdd, &[a, b]);
        sym.trace_stack.push(result);
    } else if opcode == SUB {
        let b = sym.trace_stack.pop().unwrap();
        let a = sym.trace_stack.pop().unwrap();
        let result = ctx.record_op(OpCode::IntSub, &[a, b]);
        sym.trace_stack.push(result);
    } else if opcode == JUMP_IF {
        let target = bytecode[pc + 1] as usize;
        let cond = sym.trace_stack.pop().unwrap();
        let runtime_cond = *runtime_stack.last().unwrap();

        if runtime_cond != 0 && target == header_pc {
            // Back-edge to loop header: guard cond != 0, close loop.
            let num_live = sym.num_stack_slots;
            ctx.record_guard(OpCode::GuardTrue, &[cond], num_live);
            return TraceAction::CloseLoop;
        } else if runtime_cond != 0 {
            let num_live = sym.num_stack_slots;
            ctx.record_guard(OpCode::GuardTrue, &[cond], num_live);
        } else {
            let num_live = sym.num_stack_slots;
            ctx.record_guard(OpCode::GuardFalse, &[cond], num_live);
        }
    } else if opcode == RETURN {
        return TraceAction::Abort;
    } else {
        return TraceAction::Abort;
    }

    TraceAction::Continue
}

pub struct JitTlaInterp {
    driver: JitDriver<TlaState>,
}

impl JitTlaInterp {
    pub fn new() -> Self {
        JitTlaInterp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    /// Run the TLA interpreter with JIT support.
    /// Internally uses i64 for the stack (integer specialization).
    pub fn run(&mut self, bytecode: &[u8], w_arg: WObject) -> WObject {
        let initial = w_arg.int_value();
        let mut state = TlaState {
            stack: vec![initial],
        };
        let mut pc: usize = 0;

        loop {
            // jit_merge_point(bytecode=bytecode, pc=pc, stack=stack)
            let header_pc = self
                .driver
                .current_trace_green_key()
                .map(|k| k as usize)
                .unwrap_or(0);
            self.driver.merge_point(|ctx, sym| {
                trace_instruction(ctx, sym, bytecode, pc, &state.stack, header_pc)
            });

            if pc >= bytecode.len() {
                break;
            }

            let opcode = bytecode[pc];
            pc += 1;

            if opcode == CONST_INT {
                let value = bytecode[pc] as i64;
                pc += 1;
                state.stack.push(value);
            } else if opcode == POP {
                state.stack.pop();
            } else if opcode == DUP {
                let v = *state.stack.last().unwrap();
                state.stack.push(v);
            } else if opcode == ADD {
                let b = state.stack.pop().unwrap();
                let a = state.stack.pop().unwrap();
                state.stack.push(a + b);
            } else if opcode == SUB {
                let b = state.stack.pop().unwrap();
                let a = state.stack.pop().unwrap();
                state.stack.push(a - b);
            } else if opcode == JUMP_IF {
                let target = bytecode[pc] as usize;
                pc += 1;
                let cond = state.stack.pop().unwrap();
                if cond != 0 {
                    // can_enter_jit(bytecode=bytecode, pc=target, stack=stack)
                    if target < pc {
                        if self.driver.back_edge(target, &mut state, bytecode, || {}) {
                            pc = target;
                            continue;
                        }
                    }
                    pc = target;
                }
            } else if opcode == RETURN {
                break;
            }
        }

        WObject::Int(state.stack.pop().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    fn countdown_bytecode() -> Vec<u8> {
        vec![
            DUP, // 0
            CONST_INT, 1,   // 1, 2
            SUB, // 3
            DUP, // 4
            JUMP_IF, 1,      // 5, 6 → back to CONST_INT
            POP,    // 7
            RETURN, // 8
        ]
    }

    #[test]
    fn jit_countdown_5() {
        let bc = countdown_bytecode();
        let mut jit = JitTlaInterp::new();
        let res = jit.run(&bc, WObject::Int(5));
        assert_eq!(res.int_value(), 5);
    }

    #[test]
    fn jit_countdown_100() {
        let bc = countdown_bytecode();
        let mut jit = JitTlaInterp::new();
        let res = jit.run(&bc, WObject::Int(100));
        assert_eq!(res.int_value(), 100);
    }

    #[test]
    fn jit_matches_interp() {
        let bc = countdown_bytecode();
        for n in [1, 2, 5, 10, 50, 100, 200] {
            let expected = interp::run(&bc, WObject::Int(n)).int_value();
            let mut jit = JitTlaInterp::new();
            let got = jit.run(&bc, WObject::Int(n)).int_value();
            assert_eq!(got, expected, "mismatch for n={n}");
        }
    }

    #[test]
    fn jit_no_loop() {
        let prog = vec![CONST_INT, 42, ADD, RETURN];
        let mut jit = JitTlaInterp::new();
        let res = jit.run(&prog, WObject::Int(0));
        assert_eq!(res.int_value(), 42);
    }
}
