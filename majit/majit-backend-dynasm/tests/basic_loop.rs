/// Basic test: compile and execute a simple int_add loop via dynasm backend.
///
/// Trace: i0 = input
///   label(i0)
///   i1 = int_add(i0, 1)
///   i2 = int_lt(i1, 10)
///   guard_true(i2)  [fail_args: i1]
///   jump(i1)        → label
///   finish(i1)      [on guard failure]
use std::collections::HashMap;

use majit_backend::{Backend, JitCellToken};
use majit_ir::{InputArg, Op, OpCode, OpRef, Type, Value};

use majit_backend_dynasm::runner::DynasmBackend;

#[test]
fn test_just_finish() {
    // Simplest possible trace: just FINISH with no args
    let mut backend = DynasmBackend::new();
    let mut token = JitCellToken::new(1);

    let inputargs = vec![];

    let mut finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos = OpRef(0);
    finish_op.fail_arg_types = Some(vec![]);
    finish_op.fail_args = Some(vec![].into());

    let ops = vec![finish_op];

    let result = backend.compile_loop(&inputargs, &ops, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
}

#[test]
fn test_simple_int_add() {
    let mut backend = DynasmBackend::new();
    let mut token = JitCellToken::new(1);

    // Simple trace: i1 = int_add(i0, CONST_1)
    // finish(i1)  [fail_arg_types: [Int], fail_args: [i1]]
    let i0 = OpRef(0);
    let const_1 = OpRef(10001);

    // Set constant: OpRef(10001) = 1
    let mut constants = HashMap::new();
    constants.insert(10001u32, 1i64);
    backend.set_constants(constants);

    let inputargs = vec![InputArg {
        tp: Type::Int,
        index: 0,
    }];

    let mut add_op = Op::new(OpCode::IntAdd, &[i0, const_1]);
    add_op.pos = OpRef(1); // result is OpRef(1)

    let mut finish_op = Op::new(OpCode::Finish, &[OpRef(1)]);
    finish_op.pos = OpRef(2);
    finish_op.fail_arg_types = Some(vec![Type::Int]);
    finish_op.fail_args = Some(vec![OpRef(1)].into());

    let ops = vec![add_op, finish_op];

    // Compile
    let result = backend.compile_loop(&inputargs, &ops, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Execute with input i0 = 42
    let args = vec![Value::Int(42)];
    let frame = backend.execute_token(&token, &args);

    // Check result
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());

    let result_val = backend.get_int_value(&frame, 0);
    assert_eq!(result_val, 43, "42 + 1 should be 43");
}

#[test]
fn test_float_add() {
    let mut backend = DynasmBackend::new();
    let mut token = JitCellToken::new(1);

    let i0 = OpRef(0); // input: f64
    let const_half = OpRef(10001);

    // constant 0.5 as raw bits
    let mut constants = HashMap::new();
    constants.insert(10001u32, 0.5f64.to_bits() as i64);
    backend.set_constants(constants);

    let inputargs = vec![InputArg {
        tp: Type::Float,
        index: 0,
    }];

    let mut add_op = Op::new(OpCode::FloatAdd, &[i0, const_half]);
    add_op.pos = OpRef(1);

    let mut finish_op = Op::new(OpCode::Finish, &[OpRef(1)]);
    finish_op.pos = OpRef(2);
    finish_op.fail_arg_types = Some(vec![Type::Float]);
    finish_op.fail_args = Some(vec![OpRef(1)].into());

    let ops = vec![add_op, finish_op];

    let result = backend.compile_loop(&inputargs, &ops, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Execute with input = 1.5
    let args = vec![Value::Float(1.5)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());

    let result_val = backend.get_float_value(&frame, 0);
    assert!(
        (result_val - 2.0).abs() < 1e-10,
        "1.5 + 0.5 should be 2.0, got {}",
        result_val
    );
}

#[test]
fn test_guard_and_loop() {
    // Trace: loop that adds 1 until >= 5, then guard fails
    // i0 = input
    // label(i0)
    // i1 = int_add(i0, CONST_1)
    // i2 = int_lt(i1, CONST_5)
    // guard_true(i2)   [fail_args: i1]
    // jump(i1)
    let mut backend = DynasmBackend::new();
    let mut token = JitCellToken::new(1);

    let mut constants = HashMap::new();
    constants.insert(10001u32, 1i64);
    constants.insert(10005u32, 5i64);
    backend.set_constants(constants);

    let inputargs = vec![InputArg {
        tp: Type::Int,
        index: 0,
    }];

    let mut label_op = Op::new(OpCode::Label, &[OpRef(0)]);
    label_op.pos = OpRef(100);

    let mut add_op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(10001)]);
    add_op.pos = OpRef(1);

    let mut lt_op = Op::new(OpCode::IntLt, &[OpRef(1), OpRef(10005)]);
    lt_op.pos = OpRef(2);

    let mut guard_op = Op::new(OpCode::GuardTrue, &[OpRef(2)]);
    guard_op.pos = OpRef(3);
    guard_op.fail_arg_types = Some(vec![Type::Int]);
    guard_op.fail_args = Some(vec![OpRef(1)].into());

    let mut jump_op = Op::new(OpCode::Jump, &[OpRef(1)]);
    jump_op.pos = OpRef(4);

    let ops = vec![label_op, add_op, lt_op, guard_op, jump_op];

    let result = backend.compile_loop(&inputargs, &ops, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Start with 0: should loop 0→1→2→3→4→5, guard fails at 5
    let args = vec![Value::Int(0)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "should be guard failure, not finish");

    // RPython parity: values stay in their allocated slots.
    // OpRef(1) was allocated to slot 1 by genop_int_add.
    // After Label remap, OpRef(0)=slot 0 (input), OpRef(1)=slot 1.
    let result_val = backend.get_int_value(&frame, 1);
    assert_eq!(result_val, 5, "loop should stop at 5, value at slot 1");
}
