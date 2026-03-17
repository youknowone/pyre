/// JIT-enabled tiny3 interpreter using JitDriver + JitState.
///
/// Greens: [pos, bytecode]   (bytecode is constant per trace — not tracked)
/// Reds:   [args]
///
/// Identical JIT strategy to tiny2: trace the integer-only path.
/// When all args are integers, the loop body compiles to pure i64 arithmetic.
/// The difference from tiny2 is the interpreter: non-integer values are FloatBox
/// (not StrBox), and mixed int/float arithmetic casts to float.
use majit_ir::{OpCode, OpRef};
use majit_meta::{JitDriver, JitState, TraceAction, TraceCtx};

const DEFAULT_THRESHOLD: u32 = 3;

// ── JitState types ──

/// Red variables: the args array (integer-only path).
pub struct Tiny3State {
    args: Vec<i64>,
}

/// Trace shape captured at trace start.
#[derive(Clone)]
pub struct Tiny3Meta {
    num_args: usize,
}

/// Symbolic state during tracing — OpRef for each arg + computation stack.
pub struct Tiny3Sym {
    /// args[i] → current OpRef mapping.
    trace_args: Vec<OpRef>,
    /// Intermediate computation stack during tracing.
    trace_stack: Vec<OpRef>,
}

impl JitState for Tiny3State {
    type Meta = Tiny3Meta;
    type Sym = Tiny3Sym;
    type Env = [&'static str];

    fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Tiny3Meta {
        Tiny3Meta {
            num_args: self.args.len(),
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        self.args.clone()
    }

    fn create_sym(meta: &Self::Meta, _header_pc: usize) -> Tiny3Sym {
        let trace_args: Vec<OpRef> = (0..meta.num_args)
            .map(|i| OpRef(i as u32))
            .collect();
        Tiny3Sym {
            trace_args,
            trace_stack: Vec::new(),
        }
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        self.args.len() == meta.num_args
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.args.clear();
        self.args.extend_from_slice(values);
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
    sym: &mut Tiny3Sym,
    bytecode: &[&str],
    pos: usize,
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
        return TraceAction::Abort;
    } else if opcode == "}" {
        let flag_ref = sym.trace_stack.pop().unwrap();
        let zero = ctx.const_int(0);
        let cond = ctx.record_op(OpCode::IntNe, &[flag_ref, zero]);
        let num_live = sym.trace_args.len();
        let fail_args: Vec<OpRef> = sym.trace_args.clone();
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

pub struct JitTiny3Interp {
    driver: JitDriver<Tiny3State>,
}

impl JitTiny3Interp {
    pub fn new() -> Self {
        JitTiny3Interp {
            driver: JitDriver::new(DEFAULT_THRESHOLD),
        }
    }

    /// Run a word-based program with integer args.
    /// Returns the result: stack top if non-empty, else args[0].
    pub fn run(&mut self, bytecode: &[&str], args: &mut Vec<i64>) -> i64 {
        // Safety: bytecode references are valid for the duration of run().
        // We transmute to 'static to satisfy the Env bound; the driver
        // never stores the env reference beyond back_edge.
        let static_bytecode: &[&'static str] =
            unsafe { std::mem::transmute::<&[&str], &[&'static str]>(bytecode) };

        let mut state = Tiny3State {
            args: args.clone(),
        };
        let mut stack: Vec<i64> = Vec::new();
        let mut loops: Vec<usize> = Vec::new();
        let mut pos: usize = 0;

        while pos < bytecode.len() {
            // jit_merge_point
            {
                let bc = bytecode;
                let p = pos;
                self.driver.merge_point(|ctx, sym| {
                    trace_instruction(ctx, sym, bc, p)
                });
            }

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

                    // can_enter_jit
                    if target < pos
                        && self
                            .driver
                            .back_edge(target, &mut state, static_bytecode, || {})
                    {
                        // Compiled code ran — state.args is restored.
                        // The guard exits when flag == 0, so the loop is done.
                        loops.pop();
                        *args = state.args.clone();
                        continue;
                    }

                    pos = target;
                }
            } else {
                // Integer literal
                stack.push(parse_int(opcode, 0));
            }
        }

        // Sync final state back
        *args = state.args.clone();

        if !stack.is_empty() {
            stack.pop().unwrap()
        } else {
            state.args[0]
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
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![1i64, 1, 11];
        let result = jit.run(&prog, &mut args);
        assert_eq!(result, 89);
    }

    #[test]
    fn jit_fibonacci_matches_interp() {
        let prog_str = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1";
        let prog: Vec<&str> = prog_str.split_whitespace().collect();

        for n in [5, 10, 11, 15, 20] {
            let mut interp_args = vec![
                interp::Box::Int(1),
                interp::Box::Int(1),
                interp::Box::Int(n),
            ];
            let interp_result = interp::interpret(&prog, &mut interp_args);
            let expected = interp::repr_stack(&interp_result);

            let mut jit = JitTiny3Interp::new();
            let mut jit_args = vec![1i64, 1, n];
            let jit_result = jit.run(&prog, &mut jit_args);

            assert_eq!(jit_result.to_string(), expected, "fib({n}) mismatch");
        }
    }

    #[test]
    fn jit_countdown() {
        let prog: Vec<&str> = "{ #1 #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![5i64];
        jit.run(&prog, &mut args);
        assert_eq!(args[0], 0);
    }
}
