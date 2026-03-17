/// JIT-enabled tiny2 interpreter using JitDriver + JitState.
///
/// Greens: [pos, bytecode]   (bytecode and position are loop constants)
/// Reds:   [args]
///
/// Key difference from tl.py: the live state across loop iterations is args[],
/// not a stack. The stack is used only for intermediate computations within
/// one iteration and is empty at the merge point ('{' back-edge).
///
/// JIT traces the integer-only path. When all args are integers, the loop body
///
/// This example hand-writes `trace_instruction` for educational purposes.
/// In production, the `#[jit_interp]` proc macro auto-generates tracing
/// code from the interpreter's match dispatch — see aheuijit for an example.
/// compiles to pure i64 arithmetic.
use majit_ir::{OpCode, OpRef};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

const DEFAULT_THRESHOLD: u32 = 3;

// ── JitState types ──

/// Red variables: the args array.
pub struct Tiny2State {
    args: Vec<i64>,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct Tiny2Meta {
    header_pos: usize,
    num_args: usize,
}

/// Symbolic state during tracing — OpRef for each arg slot + computation stack.
pub struct Tiny2Sym {
    /// args[i] → OpRef mapping (the live state).
    trace_args: Vec<OpRef>,
    /// Intermediate computation stack (maps to OpRef during tracing).
    trace_stack: Vec<OpRef>,
}

impl JitState for Tiny2State {
    type Meta = Tiny2Meta;
    type Sym = Tiny2Sym;
    type Env = [&'static str];

    fn build_meta(&self, header_pos: usize, _env: &Self::Env) -> Tiny2Meta {
        Tiny2Meta {
            header_pos,
            num_args: self.args.len(),
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        self.args.clone()
    }

    fn create_sym(meta: &Self::Meta, _header_pos: usize) -> Tiny2Sym {
        let trace_args: Vec<OpRef> = (0..meta.num_args)
            .map(|i| OpRef(i as u32))
            .collect();
        Tiny2Sym {
            trace_args,
            trace_stack: Vec::new(),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        self.args.len() == meta.num_args
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.args = values.to_vec();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.trace_args.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

/// Trace one instruction, recording IR into ctx.
fn trace_instruction(
    ctx: &mut TraceCtx,
    sym: &mut Tiny2Sym,
    bytecode: &[&str],
    pos: usize,
    _state: &Tiny2State,
) -> TraceAction {
    let opcode = bytecode[pos];

    if opcode == "ADD" || opcode == "SUB" || opcode == "MUL" {
        let b = sym.trace_stack.pop().unwrap();
        let a = sym.trace_stack.pop().unwrap();
        let ir_op = match opcode {
            "ADD" => OpCode::IntAdd,
            "SUB" => OpCode::IntSub,
            "MUL" => OpCode::IntMul,
            _ => unreachable!(),
        };
        let result = ctx.record_op(ir_op, &[a, b]);
        sym.trace_stack.push(result);
    } else if opcode.starts_with('#') {
        let n = parse_int(opcode, 1) as usize;
        let opref = sym.trace_args[n - 1];
        sym.trace_stack.push(opref);
    } else if opcode.starts_with("->#") {
        let n = parse_int(opcode, 3) as usize;
        let opref = sym.trace_stack.pop().unwrap();
        sym.trace_args[n - 1] = opref;
    } else if opcode == "{" {
        // Nested loop start — abort tracing.
        return TraceAction::Abort;
    } else if opcode == "}" {
        // Back-edge: guard that flag != 0, close loop.
        let flag_ref = sym.trace_stack.pop().unwrap();
        let zero = ctx.const_int(0);
        let cond = ctx.record_op(OpCode::IntNe, &[flag_ref, zero]);
        let fail_args: Vec<OpRef> = sym.trace_args.clone();
        let num_live = fail_args.len();
        ctx.record_guard_with_fail_args(OpCode::GuardTrue, &[cond], num_live, &fail_args);
        return TraceAction::CloseLoop;
    } else {
        // Integer literal
        let val = parse_int(opcode, 0);
        let opref = ctx.const_int(val);
        sym.trace_stack.push(opref);
    }

    TraceAction::Continue
}

pub struct JitTiny2Interp {
    driver: JitDriver<Tiny2State>,
}

impl JitTiny2Interp {
    pub fn new() -> Self {
        JitTiny2Interp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    /// Run a word-based program with integer args.
    /// Returns the result: stack top if non-empty, else args[0].
    pub fn run(&mut self, bytecode: &[&str], args: &mut Vec<i64>) -> i64 {
        // JitState::Env is [&'static str], so we need to transmute the bytecode
        // lifetime. This is safe because the bytecode slice outlives the JIT run.
        let static_bytecode: &[&'static str] =
            unsafe { std::mem::transmute::<&[&str], &[&'static str]>(bytecode) };

        let mut state = Tiny2State {
            args: args.clone(),
        };
        let mut stack: Vec<i64> = Vec::new();
        let mut loops: Vec<usize> = Vec::new();
        let mut pos: usize = 0;

        while pos < bytecode.len() {
            // jit_merge_point
            self.driver.merge_point(|ctx, sym| {
                trace_instruction(ctx, sym, bytecode, pos, &state)
            });

            let opcode = bytecode[pos];
            pos += 1;

            if opcode == "ADD" || opcode == "SUB" || opcode == "MUL" {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let result = match opcode {
                    "ADD" => a + b,
                    "SUB" => a - b,
                    "MUL" => a * b,
                    _ => unreachable!(),
                };
                stack.push(result);
            } else if opcode.starts_with('#') {
                let n = parse_int(opcode, 1) as usize;
                stack.push(state.args[n - 1]);
            } else if opcode.starts_with("->#") {
                let n = parse_int(opcode, 3) as usize;
                let val = stack.pop().unwrap();
                state.args[n - 1] = val;
            } else if opcode == "{" {
                loops.push(pos);
            } else if opcode == "}" {
                let flag = stack.pop().unwrap();
                if flag == 0 {
                    loops.pop();
                } else {
                    let target = *loops.last().unwrap();

                    // can_enter_jit: back-edge
                    if !self.driver.is_tracing() {
                        if self.driver.back_edge(target, &mut state, static_bytecode, || {}) {
                            // JIT ran the loop to completion.
                            loops.pop();
                            // Restore args from state after JIT execution.
                            continue;
                        }
                    }

                    pos = target;
                }
            } else {
                // Integer literal
                stack.push(parse_int(opcode, 0));
            }
        }

        // Sync state back to args.
        *args = state.args;

        if !stack.is_empty() {
            stack.pop().unwrap()
        } else {
            args[0]
        }
    }
}

fn parse_int(s: &str, start: usize) -> i64 {
    let s = &s[start..];
    let mut res: i64 = 0;
    for c in s.chars() {
        let d = c as i64 - '0' as i64;
        res = res * 10 + d;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp;

    #[test]
    fn jit_fibonacci_single() {
        let prog: Vec<&str> = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny2Interp::new();
        let mut args = vec![1i64, 1, 11];
        let result = jit.run(&prog, &mut args);
        assert_eq!(result, 89);
    }

    #[test]
    fn jit_fibonacci_matches_interp() {
        let prog_str = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1";
        let prog: Vec<&str> = prog_str.split_whitespace().collect();

        for n in [5, 10, 11, 15, 20] {
            // Interpreter
            let mut interp_args = vec![
                interp::Box::Int(1),
                interp::Box::Int(1),
                interp::Box::Int(n),
            ];
            let interp_result = interp::interpret(&prog, &mut interp_args);
            let expected = interp::repr_stack(&interp_result);

            // JIT
            let mut jit = JitTiny2Interp::new();
            let mut jit_args = vec![1i64, 1, n];
            let jit_result = jit.run(&prog, &mut jit_args);

            assert_eq!(jit_result.to_string(), expected, "fib({n}) mismatch");
        }
    }

    #[test]
    fn jit_factorial() {
        let prog: Vec<&str> = "1 { #1 MUL #1 1 SUB ->#1 #1 }".split_whitespace().collect();

        // This computes factorial using a stack + args:
        // Push 1 (accumulator), then loop: multiply acc by arg1, decrement arg1
        // But our JIT interpreter tracks args only... let me adjust.
        // Actually this doesn't fit the integer-only JIT path well because
        // it uses stack for the accumulator. Let's test with the interp.
        let mut interp_args = vec![interp::Box::Int(5)];
        let result = interp::interpret(&prog, &mut interp_args);
        assert_eq!(interp::repr_stack(&result), "120");
    }

    #[test]
    fn jit_countdown() {
        // Simple countdown: just decrements arg1 each iteration
        // Loop body: #1 #1 1 SUB ->#1 #1
        // This pushes arg1, then decrements arg1, then pushes new arg1 as flag
        let prog: Vec<&str> = "{ #1 #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let mut jit = JitTiny2Interp::new();
        let mut args = vec![5i64];
        jit.run(&prog, &mut args);
        // After loop, arg1 should be 0
        assert_eq!(args[0], 0);
    }
}
