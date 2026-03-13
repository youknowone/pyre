//! Bytecode evaluation loop with JIT integration.
//!
//! The main dispatch loop processes RustPython instructions.
//! `JitDriver` merge points and back-edge hooks are integrated from day 1.

use pyre_bytecode::bytecode::{BinaryOperator, ComparisonOperator, Instruction, OpArgState};
use pyre_object::*;
use pyre_objspace::*;

use majit_meta::JitDriver;
use pyre_jit::state::{PyreEnv, PyreJitState};
use pyre_jit::trace::trace_bytecode;

use crate::frame::PyFrame;

/// JIT hot-count threshold. After this many back-edge hits, tracing starts.
const JIT_THRESHOLD: u32 = 1039;

/// Execute a Python code object and return its result.
///
/// The returned value is the result of `ReturnValue`,
/// or `None` if the code falls through without returning.
pub fn eval_frame(frame: &mut PyFrame) -> PyResult {
    let mut driver: JitDriver<PyreJitState> = JitDriver::new(JIT_THRESHOLD);
    let env = PyreEnv;
    let mut arg_state = OpArgState::default();
    // Clone Rc to avoid borrowing frame.code while mutating frame
    let code = frame.code.clone();

    loop {
        if frame.next_instr >= code.instructions.len() {
            // Fell off the end of the code — return None
            return Ok(w_none());
        }

        // ── JIT merge point ──────────────────────────────────────
        // When tracing is active, record IR for the current instruction.
        // The closure captures only immutable data (code, pc, stack snapshot).
        if driver.is_tracing() {
            let pc = frame.next_instr;
            let stack_snapshot: Vec<PyObjectRef> =
                frame.value_stack_w[..frame.stack_depth].to_vec();
            driver.merge_point(|ctx, sym| trace_bytecode(ctx, sym, &code, pc, &stack_snapshot));
        }

        let code_unit = code.instructions[frame.next_instr];
        let (instruction, op_arg) = arg_state.get(code_unit);
        frame.next_instr += 1;

        match instruction {
            Instruction::ExtendedArg => {
                // OpArgState handles this internally; just continue to next instruction
            }

            Instruction::Resume { .. }
            | Instruction::Nop
            | Instruction::Cache
            | Instruction::NotTaken => {
                // No-op
            }

            // ── Constants & variables ─────────────────────────────────
            Instruction::LoadConst { consti } => {
                let const_idx = consti.get(op_arg);
                let value = PyFrame::load_const(&code.constants[const_idx]);
                frame.push(value);
            }

            Instruction::LoadSmallInt { i } => {
                let value = i.get(op_arg) as i64;
                frame.push(w_int_new(value));
            }

            Instruction::LoadFast { var_num } => {
                let idx = var_num.get(op_arg).as_usize();
                let value = frame.locals_w[idx];
                if value.is_null() {
                    let name = &code.varnames[idx];
                    return Err(PyError {
                        kind: PyErrorKind::NameError,
                        message: format!("local variable '{name}' referenced before assignment"),
                    });
                }
                frame.push(value);
            }

            Instruction::StoreFast { var_num } => {
                let idx = var_num.get(op_arg).as_usize();
                let value = frame.pop();
                frame.locals_w[idx] = value;
            }

            Instruction::StoreName { namei } | Instruction::StoreGlobal { namei } => {
                let idx = namei.get(op_arg) as usize;
                let name = code.names[idx].to_string();
                let value = frame.pop();
                frame.namespace.insert(name, value);
            }

            Instruction::LoadName { namei } | Instruction::LoadGlobal { namei } => {
                let idx = namei.get(op_arg) as usize;
                let name: &str = code.names[idx].as_ref();
                match frame.namespace.get(name) {
                    Some(&value) => frame.push(value),
                    None => {
                        return Err(PyError {
                            kind: PyErrorKind::NameError,
                            message: format!("name '{name}' is not defined"),
                        });
                    }
                }
            }

            // ── Stack manipulation ────────────────────────────────────
            Instruction::PopTop => {
                frame.pop();
            }

            Instruction::PushNull => {
                frame.push(PY_NULL);
            }

            Instruction::Copy { i } => {
                let depth = i.get(op_arg) as usize;
                let val = frame.peek_at(depth - 1);
                frame.push(val);
            }

            Instruction::Swap { i } => {
                let depth = i.get(op_arg) as usize;
                let top_idx = frame.stack_depth - 1;
                let other_idx = frame.stack_depth - depth;
                frame.value_stack_w.swap(top_idx, other_idx);
            }

            // ── Binary operations ─────────────────────────────────────
            Instruction::BinaryOp { op } => {
                let op = op.get(op_arg);
                let b = frame.pop();
                let a = frame.pop();
                let result = binary_op(a, b, op)?;
                frame.push(result);
            }

            // ── Comparison operations ─────────────────────────────────
            Instruction::CompareOp { opname } => {
                let op = opname.get(op_arg);
                let b = frame.pop();
                let a = frame.pop();
                let cmp_op = match op {
                    ComparisonOperator::Less => CompareOp::Lt,
                    ComparisonOperator::LessOrEqual => CompareOp::Le,
                    ComparisonOperator::Greater => CompareOp::Gt,
                    ComparisonOperator::GreaterOrEqual => CompareOp::Ge,
                    ComparisonOperator::Equal => CompareOp::Eq,
                    ComparisonOperator::NotEqual => CompareOp::Ne,
                };
                let result = py_compare(a, b, cmp_op)?;
                frame.push(result);
            }

            // ── Unary operations ──────────────────────────────────────
            Instruction::UnaryNegative => {
                let a = frame.pop();
                let result = py_negative(a)?;
                frame.push(result);
            }

            Instruction::UnaryNot => {
                let a = frame.pop();
                frame.push(w_bool_from(!py_is_true(a)));
            }

            Instruction::UnaryInvert => {
                let a = frame.pop();
                if unsafe { is_int(a) } {
                    frame.push(w_int_new(unsafe { !w_int_get_value(a) }));
                } else {
                    return Err(PyError::type_error("bad operand type for unary ~"));
                }
            }

            // ── Control flow ──────────────────────────────────────────
            Instruction::JumpForward { delta } => {
                let d = delta.get(op_arg).as_usize();
                frame.next_instr += d;
            }

            Instruction::JumpBackward { delta } => {
                let d = delta.get(op_arg).as_usize();
                // Skip cache entries after this instruction to align with CPython's
                // delta base (deltas are relative to after instruction + its caches)
                let base = skip_caches(&code.instructions, frame.next_instr);
                let target_idx = base - d;

                // JIT back-edge detection: backward jumps trigger hot counting
                let mut jit_state = build_jit_state(frame);
                if driver.back_edge(target_idx, &mut jit_state, &env, || {}) {
                    sync_jit_state_to_frame(&jit_state, frame);
                    continue;
                }

                frame.next_instr = target_idx;
            }

            Instruction::PopJumpIfFalse { delta } => {
                let top = frame.pop();
                if !py_is_true(top) {
                    let d = delta.get(op_arg).as_usize();
                    frame.next_instr += d;
                }
            }

            Instruction::PopJumpIfTrue { delta } => {
                let top = frame.pop();
                if py_is_true(top) {
                    let d = delta.get(op_arg).as_usize();
                    frame.next_instr += d;
                }
            }

            // ── Function calls ────────────────────────────────────────
            Instruction::Call { argc } => {
                let nargs = argc.get(op_arg) as usize;
                // Pop arguments (top of stack = last arg)
                let mut args = Vec::with_capacity(nargs);
                for _ in 0..nargs {
                    args.push(frame.pop());
                }
                args.reverse();
                // Stack: [callable, null_or_self] (bottom to top after args popped)
                let _null_or_self = frame.pop();
                let callable = frame.pop();
                if unsafe { is_builtin_func(callable) } {
                    let func = unsafe { w_builtin_func_get(callable) };
                    let result = func(&args);
                    frame.push(result);
                } else {
                    return Err(PyError::type_error(format!(
                        "'{}' object is not callable",
                        unsafe { (*(*callable).ob_type).tp_name }
                    )));
                }
            }

            // ── Return ────────────────────────────────────────────────
            Instruction::ReturnValue => {
                let result = frame.pop();
                return Ok(result);
            }

            // ── Not yet implemented ───────────────────────────────────
            other => {
                return Err(PyError::type_error(format!(
                    "unimplemented instruction: {other:?}"
                )));
            }
        }
    }
}

/// Skip past any Cache instructions at `pos`, returning the index of the next
/// non-Cache instruction. Jump deltas are relative to after instruction + caches.
fn skip_caches(instructions: &[pyre_bytecode::bytecode::CodeUnit], mut pos: usize) -> usize {
    while pos < instructions.len() {
        let mut s = OpArgState::default();
        let (ins, _) = s.get(instructions[pos]);
        if matches!(ins, Instruction::Cache) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// Dispatch a binary operation based on RustPython's BinaryOperator enum.
fn binary_op(a: PyObjectRef, b: PyObjectRef, op: BinaryOperator) -> PyResult {
    match op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => py_add(a, b),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => py_sub(a, b),
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => py_mul(a, b),
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => py_floordiv(a, b),
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => py_mod(a, b),
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => {
            // Python's `/` produces float, but Phase 1 only has int
            py_floordiv(a, b)
        }
        _ => Err(PyError::type_error(format!(
            "binary operation {op:?} not yet implemented"
        ))),
    }
}

// ── JitState ↔ PyFrame conversion ────────────────────────────────────

/// Build a `PyreJitState` from the current frame state.
///
/// Namespace entries are sorted by key for deterministic ordering.
fn build_jit_state(frame: &PyFrame) -> PyreJitState {
    let mut ns_keys: Vec<String> = frame.namespace.keys().cloned().collect();
    ns_keys.sort();
    let ns_values: Vec<PyObjectRef> = ns_keys
        .iter()
        .map(|k| *frame.namespace.get(k).unwrap())
        .collect();

    PyreJitState {
        next_instr: frame.next_instr,
        locals: frame.locals_w.clone(),
        ns_keys,
        ns_values,
        stack: frame.value_stack_w[..frame.stack_depth].to_vec(),
        stack_depth: frame.stack_depth,
    }
}

/// Restore frame state from a `PyreJitState` after JIT code execution.
fn sync_jit_state_to_frame(jit_state: &PyreJitState, frame: &mut PyFrame) {
    frame.next_instr = jit_state.next_instr;

    // Restore fast locals
    let local_count = jit_state.locals.len().min(frame.locals_w.len());
    frame.locals_w[..local_count].copy_from_slice(&jit_state.locals[..local_count]);

    // Restore namespace
    frame.namespace.clear();
    for (k, &v) in jit_state.ns_keys.iter().zip(&jit_state.ns_values) {
        frame.namespace.insert(k.clone(), v);
    }

    // Restore stack
    let sd = jit_state.stack_depth;
    frame.value_stack_w[..sd].copy_from_slice(&jit_state.stack[..sd]);
    frame.stack_depth = sd;
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_bytecode::*;

    fn run_eval(source: &str) -> PyResult {
        let code = compile_eval(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        eval_frame(&mut frame)
    }

    #[allow(dead_code)]
    fn run_exec(source: &str) -> PyResult {
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        eval_frame(&mut frame)
    }

    #[test]
    fn test_literal() {
        let result = run_eval("42").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_addition() {
        let result = run_eval("1 + 2").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 3) };
    }

    #[test]
    fn test_subtraction() {
        let result = run_eval("10 - 3").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_multiplication() {
        let result = run_eval("6 * 7").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_complex_expr() {
        let result = run_eval("(2 + 3) * 4 - 1").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 19) };
    }

    #[test]
    fn test_comparison() {
        let result = run_eval("3 < 5").unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_comparison_false() {
        let result = run_eval("5 < 3").unwrap();
        unsafe { assert!(!w_bool_get_value(result)) };
    }

    #[test]
    fn test_store_load_namespace() {
        let source = "x = 5\ny = x * x";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let x = *frame.namespace.get("x").unwrap();
            let y = *frame.namespace.get("y").unwrap();
            assert_eq!(w_int_get_value(x), 5);
            assert_eq!(w_int_get_value(y), 25);
        }
    }

    #[test]
    fn test_while_loop() {
        let source = "i = 0\nwhile i < 10:\n    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let i = *frame.namespace.get("i").unwrap();
            assert_eq!(w_int_get_value(i), 10);
        }
    }

    #[test]
    fn test_none_result() {
        let result = run_eval("None").unwrap();
        unsafe { assert!(is_none(result)) };
    }

    #[test]
    fn test_bool_result() {
        let result = run_eval("True").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }
}
