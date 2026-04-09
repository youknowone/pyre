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
    let const_1 = OpRef::from_const(1);

    // Set constant: OpRef::from_const(1) = 1
    let mut constants = HashMap::new();
    constants.insert(OpRef::from_const(1).0, 1i64);
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
    let const_half = OpRef::from_const(1);

    // constant 0.5 as raw bits
    let mut constants = HashMap::new();
    constants.insert(OpRef::from_const(1).0, 0.5f64.to_bits() as i64);
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
    constants.insert(OpRef::from_const(1).0, 1i64);
    constants.insert(OpRef::from_const(5).0, 5i64);
    backend.set_constants(constants);

    let inputargs = vec![InputArg {
        tp: Type::Int,
        index: 0,
    }];

    let mut label_op = Op::new(OpCode::Label, &[OpRef(0)]);
    label_op.pos = OpRef(100);

    let mut add_op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef::from_const(1)]);
    add_op.pos = OpRef(1);

    let mut lt_op = Op::new(OpCode::IntLt, &[OpRef(1), OpRef::from_const(5)]);
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

    // fail_args = [OpRef(1)], so fail_arg index 0 = the IntAdd result.
    let result_val = backend.get_int_value(&frame, 0);
    assert_eq!(result_val, 5, "loop should stop at 5, fail_arg[0]");
}

#[test]
fn test_float_loop_carried_across_jump() {
    let mut backend = DynasmBackend::new();
    let mut token = JitCellToken::new(1);

    let mut constants = HashMap::new();
    constants.insert(OpRef::from_const(5).0, 5i64);
    constants.insert(OpRef::from_const(10).0, 0.5f64.to_bits() as i64);
    constants.insert(OpRef::from_const(1).0, 1i64);
    backend.set_constants(constants);

    let inputargs = vec![
        InputArg {
            tp: Type::Float,
            index: 0,
        },
        InputArg {
            tp: Type::Int,
            index: 1,
        },
    ];

    let mut label_op = Op::new(OpCode::Label, &[OpRef(0), OpRef(1)]);
    label_op.pos = OpRef(100);

    let mut lt_op = Op::new(OpCode::IntLt, &[OpRef(1), OpRef::from_const(5)]);
    lt_op.pos = OpRef(2);

    let mut guard_op = Op::new(OpCode::GuardTrue, &[OpRef(2)]);
    guard_op.pos = OpRef(3);
    guard_op.fail_arg_types = Some(vec![Type::Float, Type::Int]);
    guard_op.fail_args = Some(vec![OpRef(0), OpRef(1)].into());

    let mut cast_op = Op::new(OpCode::CastIntToFloat, &[OpRef(1)]);
    cast_op.pos = OpRef(4);

    let mut mul_op = Op::new(OpCode::FloatMul, &[OpRef(4), OpRef::from_const(10)]);
    mul_op.pos = OpRef(5);

    let mut add_op = Op::new(OpCode::FloatAdd, &[OpRef(0), OpRef(5)]);
    add_op.pos = OpRef(6);

    let mut inc_op = Op::new(OpCode::IntAdd, &[OpRef(1), OpRef::from_const(1)]);
    inc_op.pos = OpRef(7);

    let mut jump_op = Op::new(OpCode::Jump, &[OpRef(6), OpRef(7)]);
    jump_op.pos = OpRef(8);

    let ops = vec![
        label_op, lt_op, guard_op, cast_op, mul_op, add_op, inc_op, jump_op,
    ];

    let result = backend.compile_loop(&inputargs, &ops, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let args = vec![Value::Float(0.0), Value::Int(0)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "should exit via guard failure");

    let sum = backend.get_float_value(&frame, 0);
    let index = backend.get_int_value(&frame, 1);
    assert!(
        (sum - 5.0).abs() < 1e-10,
        "expected carried float sum to be 5.0, got {}",
        sum
    );
    assert_eq!(index, 5, "expected guard failure at i=5");
}
