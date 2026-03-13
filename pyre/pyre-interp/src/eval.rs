//! Bytecode evaluation loop with JIT integration.
//!
//! The main dispatch loop processes RustPython instructions.
//! `JitDriver` merge points and back-edge hooks are integrated from day 1.

use pyre_bytecode::bytecode::{BinaryOperator, ComparisonOperator, Instruction, OpArgState};
use pyre_object::*;
use pyre_objspace::*;
use pyre_runtime::{
    PyError, PyErrorKind, PyResult, is_builtin_func, is_func, w_builtin_func_get, w_code_get_ptr,
    w_func_get_code_ptr, w_func_new,
};

use majit_meta::JitDriver;
use pyre_jit::state::{PyreEnv, PyreJitState};
use pyre_jit::trace::trace_bytecode;

use crate::frame::{PyFrame, build_pyframe_virtualizable_info};

/// JIT hot-count threshold. After this many back-edge hits, tracing starts.
const JIT_THRESHOLD: u32 = 1039;

/// Execute a Python code object and return its result.
///
/// The returned value is the result of `ReturnValue`,
/// or `None` if the code falls through without returning.
pub fn eval_frame(frame: &mut PyFrame) -> PyResult {
    let mut driver: JitDriver<PyreJitState> = JitDriver::new(JIT_THRESHOLD);

    // Register virtualizable info so the JIT knows which PyFrame fields
    // can live in CPU registers during compiled code execution.
    driver.set_virtualizable_info(build_pyframe_virtualizable_info());

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

            Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
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

            Instruction::LoadFastBorrowLoadFastBorrow { var_nums } => {
                let pair = var_nums.get(op_arg);
                let idx1 = (pair >> 4) as usize;
                let idx2 = (pair & 15) as usize;
                let v1 = frame.locals_w[idx1];
                let v2 = frame.locals_w[idx2];
                if v1.is_null() {
                    let name = &code.varnames[idx1];
                    return Err(PyError {
                        kind: PyErrorKind::NameError,
                        message: format!("local variable '{name}' referenced before assignment"),
                    });
                }
                if v2.is_null() {
                    let name = &code.varnames[idx2];
                    return Err(PyError {
                        kind: PyErrorKind::NameError,
                        message: format!("local variable '{name}' referenced before assignment"),
                    });
                }
                frame.push(v1);
                frame.push(v2);
            }

            Instruction::StoreFast { var_num } => {
                let idx = var_num.get(op_arg).as_usize();
                let value = frame.pop();
                frame.locals_w[idx] = value;
            }

            Instruction::LoadFastCheck { var_num } => {
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

            Instruction::LoadFastLoadFast { var_nums } => {
                let pair = var_nums.get(op_arg);
                let idx1 = (pair >> 4) as usize;
                let idx2 = (pair & 15) as usize;
                frame.push(frame.locals_w[idx1]);
                frame.push(frame.locals_w[idx2]);
            }

            Instruction::StoreFastLoadFast { var_nums } => {
                let pair = var_nums.get(op_arg);
                let store_idx = pair.store_idx() as usize;
                let load_idx = pair.load_idx() as usize;
                let val = frame.pop();
                frame.locals_w[store_idx] = val;
                frame.push(frame.locals_w[load_idx]);
            }

            Instruction::StoreFastStoreFast { var_nums } => {
                let pair = var_nums.get(op_arg);
                let idx1 = (pair >> 4) as usize;
                let idx2 = (pair & 15) as usize;
                let v1 = frame.pop();
                frame.locals_w[idx1] = v1;
                let v2 = frame.pop();
                frame.locals_w[idx2] = v2;
            }

            Instruction::StoreName { namei } | Instruction::StoreGlobal { namei } => {
                let idx = namei.get(op_arg) as usize;
                let name = code.names[idx].to_string();
                let value = frame.pop();
                frame.namespace.insert(name, value);
            }

            Instruction::LoadName { namei } => {
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

            Instruction::LoadGlobal { namei } => {
                // In 3.14, oparg encodes: name_idx = oparg >> 1,
                // push_null = oparg & 1 (for the call protocol).
                // Push order: value first, then NULL (callable at bottom,
                // self_or_null above it).
                let raw = namei.get(op_arg) as usize;
                let name_idx = raw >> 1;
                let push_null = (raw & 1) != 0;
                let name: &str = code.names[name_idx].as_ref();
                match frame.namespace.get(name) {
                    Some(&value) => frame.push(value),
                    None => {
                        return Err(PyError {
                            kind: PyErrorKind::NameError,
                            message: format!("name '{name}' is not defined"),
                        });
                    }
                }
                if push_null {
                    frame.push(PY_NULL);
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
                let base = skip_caches(&code.instructions, frame.next_instr);
                frame.next_instr = base + d;
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
                    let base = skip_caches(&code.instructions, frame.next_instr);
                    frame.next_instr = base + d;
                }
            }

            Instruction::PopJumpIfTrue { delta } => {
                let top = frame.pop();
                if py_is_true(top) {
                    let d = delta.get(op_arg).as_usize();
                    let base = skip_caches(&code.instructions, frame.next_instr);
                    frame.next_instr = base + d;
                }
            }

            // ── Function creation ─────────────────────────────────────
            Instruction::MakeFunction => {
                // Stack: code object (TOS)
                // In CPython 3.14, MakeFunction pops a code object and
                // pushes a function object. The code object was pushed
                // via LoadConst of a ConstantData::Code.
                let code_obj = frame.pop();
                let code_ptr = unsafe { w_code_get_ptr(code_obj) };
                // Extract the function name from the CodeObject's qualname
                let code = unsafe { &*(code_ptr as *const pyre_bytecode::CodeObject) };
                let name = code.qualname.to_string();
                let func = w_func_new(code_ptr, name);
                frame.push(func);
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
                } else if unsafe { is_func(callable) } {
                    // User-defined function call
                    let code_ptr = unsafe { w_func_get_code_ptr(callable) };
                    let func_code = unsafe { &*(code_ptr as *const pyre_bytecode::CodeObject) };
                    let mut func_frame = PyFrame::new_for_call(
                        func_code.clone(),
                        &args,
                        &frame.namespace,
                        frame.execution_context.clone(),
                    );
                    let result = eval_frame(&mut func_frame)?;
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

            // ── Container construction ───────────────────────────────
            Instruction::BuildList { count } => {
                let size = count.get(op_arg) as usize;
                let mut items = Vec::with_capacity(size);
                for _ in 0..size {
                    items.push(frame.pop());
                }
                items.reverse();
                frame.push(w_list_new(items));
            }

            Instruction::BuildTuple { count } => {
                let size = count.get(op_arg) as usize;
                let mut items = Vec::with_capacity(size);
                for _ in 0..size {
                    items.push(frame.pop());
                }
                items.reverse();
                frame.push(w_tuple_new(items));
            }

            Instruction::BuildMap { count } => {
                let size = count.get(op_arg) as usize;
                let dict = w_dict_new();
                // Stack has key-value pairs: key1, val1, key2, val2, ...
                // We need to pop them in reverse order
                let mut pairs = Vec::with_capacity(size);
                for _ in 0..size {
                    let value = frame.pop();
                    let key = frame.pop();
                    pairs.push((key, value));
                }
                pairs.reverse();
                for (key, value) in pairs {
                    unsafe {
                        if is_int(key) {
                            w_dict_setitem(dict, w_int_get_value(key), value);
                        }
                    }
                }
                frame.push(dict);
            }

            // ── Subscript operations ─────────────────────────────────
            Instruction::StoreSubscr => {
                // Stack: [value, obj, key] (TOS=key, TOS1=obj, TOS2=value)
                let key = frame.pop();
                let obj = frame.pop();
                let value = frame.pop();
                py_setitem(obj, key, value)?;
            }

            // ── List operations ──────────────────────────────────────
            Instruction::ListAppend { i } => {
                let value = frame.pop();
                let depth = i.get(op_arg) as usize;
                let list = frame.peek_at(depth - 1);
                unsafe { w_list_append(list, value) };
            }

            // ── Sequence unpacking ───────────────────────────────────
            Instruction::UnpackSequence { count } => {
                let size = count.get(op_arg) as usize;
                let seq = frame.pop();
                unsafe {
                    if is_tuple(seq) {
                        let len = w_tuple_len(seq);
                        if len != size {
                            return Err(PyError::type_error(format!(
                                "not enough values to unpack (expected {size}, got {len})"
                            )));
                        }
                        // Push in reverse order so first element ends up on top
                        for idx in (0..size).rev() {
                            frame.push(w_tuple_getitem(seq, idx as i64).unwrap());
                        }
                    } else if is_list(seq) {
                        let len = w_list_len(seq);
                        if len != size {
                            return Err(PyError::type_error(format!(
                                "not enough values to unpack (expected {size}, got {len})"
                            )));
                        }
                        for idx in (0..size).rev() {
                            frame.push(w_list_getitem(seq, idx as i64).unwrap());
                        }
                    } else {
                        return Err(PyError::type_error(format!(
                            "cannot unpack non-sequence {}",
                            (*(*seq).ob_type).tp_name
                        )));
                    }
                }
            }

            // ── Iteration ────────────────────────────────────────────
            Instruction::GetIter => {
                let obj = frame.peek();
                if unsafe { is_range_iter(obj) } {
                    // range_iter is its own iterator, leave on stack
                } else {
                    return Err(PyError::type_error(format!(
                        "'{}' object is not iterable",
                        unsafe { (*(*obj).ob_type).tp_name }
                    )));
                }
            }

            Instruction::ForIter { delta } => {
                let iter = frame.peek();
                if unsafe { is_range_iter(iter) } {
                    match unsafe { w_range_iter_next(iter) } {
                        Some(value) => {
                            // Push next value on top of the iterator
                            frame.push(value);
                        }
                        None => {
                            // Iterator exhausted: leave iterator on stack,
                            // jump forward by delta (relative to after caches).
                            let base = skip_caches(&code.instructions, frame.next_instr);
                            let d = delta.get(op_arg).as_usize();
                            frame.next_instr = base + d;
                        }
                    }
                } else {
                    return Err(PyError::type_error("not an iterator"));
                }
            }

            Instruction::EndFor => {
                // End-of-for marker. In 3.14, this is a no-op placeholder;
                // the actual iterator cleanup is done by PopIter.
            }

            Instruction::PopIter => {
                // Pop the iterator from the stack after the for loop ends.
                frame.pop();
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
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => py_truediv(a, b),
        BinaryOperator::Subscr => py_getitem(a, b),
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
    use pyre_runtime::PyExecutionContext;
    use std::rc::Rc;

    fn run_eval(source: &str) -> PyResult {
        let code = compile_eval(source).expect("compile failed");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        eval_frame(&mut frame)
    }

    fn run_exec(source: &str) -> PyResult {
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        eval_frame(&mut frame)
    }

    fn run_exec_frame(source: &str) -> (PyResult, PyFrame) {
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        let result = eval_frame(&mut frame);
        (result, frame)
    }

    #[test]
    fn test_dump_for_bytecode() {
        let source = "s = 0\nfor i in range(10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        eprintln!("Instructions for 'for i in range(10)':");
        for (i, cu) in code.instructions.iter().enumerate() {
            let mut s = pyre_bytecode::bytecode::OpArgState::default();
            let (ins, oparg) = s.get(*cu);
            eprintln!("{i:3}: {ins:?}  (arg={})", u32::from(oparg));
        }
        eprintln!("\nNames: {:?}", code.names);
        eprintln!("Varnames: {:?}", code.varnames);
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

    #[test]
    fn test_float_literal() {
        let result = run_eval("1.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 1.5);
        }
    }

    #[test]
    fn test_float_addition() {
        let result = run_eval("1.5 + 2.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 4.0);
        }
    }

    #[test]
    fn test_float_truediv() {
        let result = run_eval("10 / 4").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 2.5);
        }
    }

    #[test]
    fn test_float_comparison() {
        let result = run_eval("1.5 < 2.5").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_int_mixed() {
        let result = run_eval("1.5 + 2").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 3.5);
        }
    }

    #[test]
    fn test_float_negation() {
        let result = run_eval("-3.14").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), -3.14);
        }
    }

    #[test]
    fn test_float_truthiness() {
        // Test via py_is_true directly since `not` uses ToBool instruction
        assert!(!py_is_true(w_float_new(0.0)));
        assert!(py_is_true(w_float_new(1.5)));
        assert!(py_is_true(w_float_new(-0.1)));
    }

    // ── str tests ────────────────────────────────────────────────────

    #[test]
    fn test_str_literal() {
        let result = run_eval("'hello'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello");
        }
    }

    #[test]
    fn test_str_concat() {
        let result = run_eval("'hello' + ' world'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello world");
        }
    }

    #[test]
    fn test_str_repeat() {
        let result = run_eval("'ab' * 3").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "ababab");
        }
    }

    #[test]
    fn test_str_comparison() {
        let result = run_eval("'abc' < 'abd'").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    // ── for loop / range tests ──────────────────────────────────────

    #[test]
    fn test_for_range() {
        let source = "s = 0\nfor i in range(10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *frame.namespace.get("s").unwrap();
            assert_eq!(w_int_get_value(s), 45);
        }
    }

    #[test]
    fn test_for_range_start_stop() {
        let source = "s = 0\nfor i in range(5, 10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *frame.namespace.get("s").unwrap();
            assert_eq!(w_int_get_value(s), 35);
        }
    }

    #[test]
    fn test_for_range_step() {
        let source = "s = 0\nfor i in range(0, 10, 2):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *frame.namespace.get("s").unwrap();
            // 0 + 2 + 4 + 6 + 8 = 20
            assert_eq!(w_int_get_value(s), 20);
        }
    }

    #[test]
    fn test_for_range_empty() {
        let source = "s = 42\nfor i in range(0):\n    s = 0";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *frame.namespace.get("s").unwrap();
            assert_eq!(w_int_get_value(s), 42);
        }
    }

    #[test]
    fn test_builtin_range_print() {
        let source = "s = 0\nfor i in range(5):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let s = *frame.namespace.get("s").unwrap();
            // 0 + 1 + 2 + 3 + 4 = 10
            assert_eq!(w_int_get_value(s), 10);
        }
    }

    // ── builtin tests ───────────────────────────────────────────────

    #[test]
    fn test_builtin_len() {
        let source = "x = len([1, 2, 3])";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let x = *frame.namespace.get("x").unwrap();
            assert_eq!(w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_builtin_abs() {
        let source = "x = abs(-5)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let x = *frame.namespace.get("x").unwrap();
            assert_eq!(w_int_get_value(x), 5);
        }
    }

    #[test]
    fn test_builtin_min_max() {
        let source = "a = min(3, 7)\nb = max(3, 7)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame(&mut frame);
        unsafe {
            let a = *frame.namespace.get("a").unwrap();
            let b = *frame.namespace.get("b").unwrap();
            assert_eq!(w_int_get_value(a), 3);
            assert_eq!(w_int_get_value(b), 7);
        }
    }

    // ── container tests ────────────────────────────────────────────

    #[test]
    fn test_list_literal() {
        let source = "x = [1, 2, 3]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *frame.namespace.get("x").unwrap();
            assert!(is_list(x));
            assert_eq!(w_list_len(x), 3);
            assert_eq!(w_int_get_value(w_list_getitem(x, 0).unwrap()), 1);
            assert_eq!(w_int_get_value(w_list_getitem(x, 1).unwrap()), 2);
            assert_eq!(w_int_get_value(w_list_getitem(x, 2).unwrap()), 3);
        }
    }

    #[test]
    fn test_tuple_unpack() {
        let source = "a, b = 1, 2";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let a = *frame.namespace.get("a").unwrap();
            let b = *frame.namespace.get("b").unwrap();
            assert_eq!(w_int_get_value(a), 1);
            assert_eq!(w_int_get_value(b), 2);
        }
    }

    #[test]
    fn test_list_subscr() {
        let source = "lst = [10, 20, 30]\nx = lst[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *frame.namespace.get("x").unwrap();
            assert_eq!(w_int_get_value(x), 20);
        }
    }

    #[test]
    fn test_list_store_subscr() {
        let source = "lst = [1, 2, 3]\nlst[0] = 99\nx = lst[0]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *frame.namespace.get("x").unwrap();
            assert_eq!(w_int_get_value(x), 99);
        }
    }

    #[test]
    fn test_dict_literal_and_subscr() {
        let source = "d = {1: 10, 2: 20}\nx = d[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *frame.namespace.get("x").unwrap();
            assert_eq!(w_int_get_value(x), 10);
        }
    }

    // ── function definition and call tests ──────────────────────────

    #[test]
    fn test_simple_function() {
        let source = "def double(x):\n    return x * 2\nresult = double(21)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *frame.namespace.get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_function_with_locals() {
        let source = "\
def add_squares(a, b):
    aa = a * a
    bb = b * b
    return aa + bb
result = add_squares(3, 4)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *frame.namespace.get("result").unwrap();
            assert_eq!(w_int_get_value(result), 25);
        }
    }

    #[test]
    fn test_recursive_function() {
        let source = "\
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n - 1)
result = factorial(5)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *frame.namespace.get("result").unwrap();
            assert_eq!(w_int_get_value(result), 120);
        }
    }
}
