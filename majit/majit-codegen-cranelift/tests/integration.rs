//! Integration tests: trace recording -> optimization -> Cranelift compilation -> execution.
//!
//! These tests exercise the full majit pipeline from end to end.

use std::collections::HashMap;
use std::sync::Arc;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{Descr, DescrRef, FailDescr, OpCode, OpRef, Type, Value};
use majit_opt::intdiv::magic_numbers;
use majit_opt::optimizer::Optimizer;
use majit_opt::pure::OptPure;
use majit_opt::rewrite::OptRewrite;
use majit_opt::simplify::OptSimplify;
use majit_trace::recorder::TraceRecorder;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Minimal FailDescr for use in Finish ops.
#[derive(Debug)]
struct TestFailDescr {
    index: u32,
}

impl Descr for TestFailDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for TestFailDescr {
    fn fail_index(&self) -> u32 {
        self.index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &[]
    }
}

fn make_descr(index: u32) -> DescrRef {
    Arc::new(TestFailDescr { index })
}

fn new_optimizer() -> Optimizer {
    let mut opt = Optimizer::new();
    opt.add_pass(Box::new(OptRewrite::new()));
    opt.add_pass(Box::new(OptPure::new()));
    opt.add_pass(Box::new(OptSimplify::new()));
    opt
}

// ---------------------------------------------------------------------------
// Test 1: Simple arithmetic (trace -> optimize -> compile -> execute)
// ---------------------------------------------------------------------------

#[test]
fn test_simple_arithmetic() {
    // Record: input(i) -> result = i + CONST_1 -> finish(result)
    // Execute with i=41, expect 42.
    let mut rec = TraceRecorder::new();
    let i0 = rec.record_input_arg(Type::Int);

    // Use a high OpRef index for the constant so it doesn't collide with
    // variable indices used by the recorder.
    let const_one = OpRef(1000);
    let result = rec.record_op(OpCode::IntAdd, &[i0, const_one]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    // Optimize
    let mut opt = new_optimizer();
    let optimized = opt.optimize(&trace.ops);

    // Compile
    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 1i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(0);
    backend
        .compile_loop(&trace.inputargs, &optimized, &mut token)
        .expect("compilation should succeed");

    // Execute
    let frame = backend.execute_token(&token, &[Value::Int(41)]);
    assert_eq!(backend.get_int_value(&frame, 0), 42);
}

// ---------------------------------------------------------------------------
// Test 2: Sum loop (trace -> optimize -> compile -> execute)
// ---------------------------------------------------------------------------

#[test]
fn test_sum_loop() {
    // Record a loop: input(i, sum) -> sum2 = sum + i -> i2 = i - 1
    //   -> cmp = i2 > 0 -> guard_true(cmp) -> jump(i2, sum2)
    // Execute with i=100, sum=0. Guard fails when i2=0, at which point
    // the accumulated sum in the frame should be 5050.
    let mut rec = TraceRecorder::new();
    let i = rec.record_input_arg(Type::Int);
    let sum = rec.record_input_arg(Type::Int);

    let const_one = OpRef(1000);
    let const_zero = OpRef(1001);

    let sum2 = rec.record_op(OpCode::IntAdd, &[sum, i]);
    let i2 = rec.record_op(OpCode::IntSub, &[i, const_one]);
    let cmp = rec.record_op(OpCode::IntGt, &[i2, const_zero]);
    rec.record_guard(OpCode::GuardTrue, &[cmp], make_descr(0));
    rec.close_loop(&[i2, sum2]);
    let trace = rec.get_trace();

    // Optimize
    let mut opt = new_optimizer();
    let optimized = opt.optimize(&trace.ops);

    // Compile
    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 1i64);
    constants.insert(1001, 0i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(1);
    backend
        .compile_loop(&trace.inputargs, &optimized, &mut token)
        .expect("compilation should succeed");

    // Execute: i=100, sum=0
    let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(0)]);

    // The loop runs: i=100..1, guard fails when i2=0.
    // At guard failure, the saved frame contains the current input arg values.
    // When i2 becomes 0, we've done sum = 100+99+...+1 = 5050,
    // but the guard fires *before* the jump, so the saved values are i=1, sum=4950+100=5050? No.
    //
    // Let's trace carefully:
    //   Iteration 1: i=100, sum=0 -> sum2=100, i2=99, cmp=true -> jump(99, 100)
    //   Iteration 2: i=99, sum=100 -> sum2=199, i2=98, cmp=true -> jump(98, 199)
    //   ...
    //   Iteration 99: i=2, sum=4949 -> sum2=4951, i2=1, cmp=true -> jump(1, 4951)
    //   Iteration 100: i=1, sum=4951 -> sum2=4952, i2=0, cmp=false -> guard_true fails
    //
    // Wait: 100+99+...+2 = 5050 - 1 = 5049. And then sum2 = 4951 + 1 = 4952? Let me recalculate.
    //   sum after iter 1: 0 + 100 = 100
    //   sum after iter 2: 100 + 99 = 199
    //   ...
    //   sum after iter k: sum of (100, 99, ..., 100-k+1)
    //   After 99 iters: sum = 100+99+...+2 = (100*101/2) - 1 = 5050 - 1 = 5049
    //   Wait: 100+99+...+2 = sum(1..100) - 1 = 5050 - 1 = 5049.
    //   The jump args are (i2=1, sum2=5049).
    //   Iteration 100: i=1, sum=5049 -> sum2=5049+1=5050, i2=0, cmp=false -> guard fails
    //
    // At guard failure, the backend saves the *input arg* values (i, sum), not (i2, sum2).
    // Looking at collect_guards: for guards, fail_arg_refs = (0..num_inputs).map(OpRef).
    // So it saves var(0)=i=1 and var(1)=sum=5049.
    //
    // The sum2 and i2 have NOT been written back to the input vars yet (that happens at Jump).
    assert_eq!(backend.get_int_value(&frame, 0), 1); // i at guard failure
    assert_eq!(backend.get_int_value(&frame, 1), 5049); // sum at guard failure

    // The total sum 1+2+...+100 = 5050.
    // At guard failure we have i=1, sum=5049. The "missing" addition is sum + i = 5049 + 1 = 5050.
    let final_i = backend.get_int_value(&frame, 0);
    let final_sum = backend.get_int_value(&frame, 1);
    assert_eq!(final_sum + final_i, 5050);
}

// ---------------------------------------------------------------------------
// Test 3: Constant folding through pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_identity_operations() {
    // Record: input(x) -> y = x + 0 -> z = y * 1 -> finish(z)
    // The optimizer's OptRewrite pass can eliminate identity operations
    // when it knows the constant values. Since the optimizer doesn't have
    // access to backend constants, we verify correctness through execution.
    // The backend handles constants at codegen time regardless.
    let mut rec = TraceRecorder::new();
    let x = rec.record_input_arg(Type::Int);

    let const_zero = OpRef(1000);
    let const_one = OpRef(1001);

    let y = rec.record_op(OpCode::IntAdd, &[x, const_zero]);
    let z = rec.record_op(OpCode::IntMul, &[y, const_one]);
    rec.finish(&[z], make_descr(0));
    let trace = rec.get_trace();

    // Optimize
    let mut opt = new_optimizer();
    let optimized = opt.optimize(&trace.ops);

    // The optimized trace should pass through (optimizer doesn't know
    // about backend constants), but execution must still be correct.

    // Compile
    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 0i64);
    constants.insert(1001, 1i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(2);
    backend
        .compile_loop(&trace.inputargs, &optimized, &mut token)
        .expect("compilation should succeed");

    // Execute: x=99 -> result should be 99
    let frame = backend.execute_token(&token, &[Value::Int(99)]);
    assert_eq!(backend.get_int_value(&frame, 0), 99);

    // Also test with other values
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 0);

    let frame = backend.execute_token(&token, &[Value::Int(-42)]);
    assert_eq!(backend.get_int_value(&frame, 0), -42);
}

// ---------------------------------------------------------------------------
// Test 4: Guard failure path
// ---------------------------------------------------------------------------

#[test]
fn test_guard_failure_path() {
    // Record: input(x) -> cmp = x > 0 -> guard_true(cmp)
    //   -> result = x * 2 -> finish(result)
    let mut rec = TraceRecorder::new();
    let x = rec.record_input_arg(Type::Int);

    let const_zero = OpRef(1000);
    let const_two = OpRef(1001);

    let cmp = rec.record_op(OpCode::IntGt, &[x, const_zero]);
    rec.record_guard(OpCode::GuardTrue, &[cmp], make_descr(0));
    let result = rec.record_op(OpCode::IntMul, &[x, const_two]);
    rec.finish(&[result], make_descr(1));
    let trace = rec.get_trace();

    // Optimize
    let mut opt = new_optimizer();
    let optimized = opt.optimize(&trace.ops);

    // Compile
    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 0i64);
    constants.insert(1001, 2i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(3);
    backend
        .compile_loop(&trace.inputargs, &optimized, &mut token)
        .expect("compilation should succeed");

    // Execute with x=5: guard passes, result = 5 * 2 = 10
    let frame = backend.execute_token(&token, &[Value::Int(5)]);
    let descr = backend.get_latest_descr(&frame);
    // fail_index=1 means Finish was reached (second guard/finish in the trace)
    assert_eq!(descr.fail_index(), 1);
    assert_eq!(backend.get_int_value(&frame, 0), 10);

    // Execute with x=-1: guard fails, DeadFrame has input arg values
    let frame = backend.execute_token(&token, &[Value::Int(-1)]);
    let descr = backend.get_latest_descr(&frame);
    // fail_index=0 means the guard_true failed (first guard/finish in the trace)
    assert_eq!(descr.fail_index(), 0);
    // Guard failure saves input args: x=-1
    assert_eq!(backend.get_int_value(&frame, 0), -1);

    // Execute with x=0: 0 > 0 is false, guard fails
    let frame = backend.execute_token(&token, &[Value::Int(0)]);
    let descr = backend.get_latest_descr(&frame);
    assert_eq!(descr.fail_index(), 0);
    assert_eq!(backend.get_int_value(&frame, 0), 0);
}

// ---------------------------------------------------------------------------
// Test 5: Multiple passes working together (CSE + algebraic simplification)
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_optimization_passes() {
    // Record a trace with redundant operations:
    //   input(a, b)
    //   c = a + b          (first computation)
    //   d = a + b          (duplicate -- CSE by OptPure)
    //   e = c * d          (uses both; after CSE d is forwarded to c)
    //   finish(e)
    //
    // After OptPure CSE, d is replaced by c, so e = c * c.
    // The duplicate IntAdd is removed, reducing the trace by 1 op.
    let mut rec = TraceRecorder::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);

    let c = rec.record_op(OpCode::IntAdd, &[a, b]);
    let d = rec.record_op(OpCode::IntAdd, &[a, b]); // duplicate of c
    let e = rec.record_op(OpCode::IntMul, &[c, d]);
    rec.finish(&[e], make_descr(0));
    let trace = rec.get_trace();

    let original_len = trace.ops.len();

    // Optimize
    let mut opt = new_optimizer();
    let optimized = opt.optimize(&trace.ops);

    // CSE should have eliminated the duplicate IntAdd(a, b).
    // Original: IntAdd, IntAdd, IntMul, Finish = 4 ops
    // Optimized: IntAdd, IntMul(c, c), Finish = 3 ops (one IntAdd removed)
    assert!(
        optimized.len() < original_len,
        "optimizer should reduce op count: {} < {}",
        optimized.len(),
        original_len
    );

    // Compile and execute to verify correctness
    let mut backend = CraneliftBackend::new();
    let mut token = LoopToken::new(4);
    backend
        .compile_loop(&trace.inputargs, &optimized, &mut token)
        .expect("compilation should succeed");

    // a=3, b=7 -> c = 10, d = 10 (CSE'd to c), e = c * c = 100
    let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(7)]);
    assert_eq!(backend.get_int_value(&frame, 0), 100);

    // a=0, b=5 -> c = 5, e = 5 * 5 = 25
    let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(5)]);
    assert_eq!(backend.get_int_value(&frame, 0), 25);

    // a=-3, b=3 -> c = 0, e = 0 * 0 = 0
    let frame = backend.execute_token(&token, &[Value::Int(-3), Value::Int(3)]);
    assert_eq!(backend.get_int_value(&frame, 0), 0);
}

// ---------------------------------------------------------------------------
// Test 6: Bridge compilation end-to-end
// ---------------------------------------------------------------------------

#[test]
fn test_bridge_end_to_end() {
    // Main trace: sum loop counting down from N.
    //   input(i, sum) -> sum2 = sum + i -> i2 = i - 1
    //   -> cmp = i2 > 0 -> guard_true(cmp) -> jump(i2, sum2)
    //
    // When guard fails (i2 <= 0), compile a bridge that doubles the sum.
    let mut rec = TraceRecorder::new();
    let i = rec.record_input_arg(Type::Int);
    let sum = rec.record_input_arg(Type::Int);

    let const_one = OpRef(1000);
    let const_zero = OpRef(1001);

    let sum2 = rec.record_op(OpCode::IntAdd, &[sum, i]);
    let i2 = rec.record_op(OpCode::IntSub, &[i, const_one]);
    let cmp = rec.record_op(OpCode::IntGt, &[i2, const_zero]);
    rec.record_guard(OpCode::GuardTrue, &[cmp], make_descr(0));
    rec.close_loop(&[i2, sum2]);
    let trace = rec.get_trace();

    // Optimize
    let mut opt = new_optimizer();
    let optimized = opt.optimize(&trace.ops);

    // Compile main loop
    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 1i64);
    constants.insert(1001, 0i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(10);
    backend
        .compile_loop(&trace.inputargs, &optimized, &mut token)
        .expect("compilation should succeed");

    // Execute without bridge: i=5, sum=0
    // Loop counts 5,4,3,2,1 -> guard fails when i2=0
    // At guard failure we get: i=1, sum=5049? No, with small N=5:
    //   iter1: i=5,sum=0 -> sum2=5, i2=4, pass -> jump(4,5)
    //   iter2: i=4,sum=5 -> sum2=9, i2=3, pass -> jump(3,9)
    //   iter3: i=3,sum=9 -> sum2=12, i2=2, pass -> jump(2,12)
    //   iter4: i=2,sum=12 -> sum2=14, i2=1, pass -> jump(1,14)
    //   iter5: i=1,sum=14 -> sum2=15, i2=0, fail
    // Guard saves input args (i=1, sum=14)
    let frame = backend.execute_token(&token, &[Value::Int(5), Value::Int(0)]);
    let descr = backend.get_latest_descr(&frame);
    assert_eq!(descr.fail_index(), 0);
    assert_eq!(backend.get_int_value(&frame, 0), 1); // i
    assert_eq!(backend.get_int_value(&frame, 1), 14); // sum

    // Now compile a bridge for the guard failure.
    // Bridge takes (i, sum) and returns sum * 2.
    let mut bridge_rec = TraceRecorder::new();
    let bi = bridge_rec.record_input_arg(Type::Int);
    let bsum = bridge_rec.record_input_arg(Type::Int);
    let _ = bi; // bridge ignores i

    let bridge_const_two = OpRef(1000);
    let result = bridge_rec.record_op(OpCode::IntMul, &[bsum, bridge_const_two]);
    bridge_rec.finish(&[result], make_descr(1));
    let bridge_trace = bridge_rec.get_trace();

    let mut bridge_opt = new_optimizer();
    let bridge_optimized = bridge_opt.optimize(&bridge_trace.ops);

    let mut bridge_constants = HashMap::new();
    bridge_constants.insert(1000, 2i64);
    backend.set_constants(bridge_constants);

    // We need a CraneliftFailDescr to pass to compile_bridge.
    // The fail_index matches the guard's index in the original loop.
    let bridge_fail_descr =
        majit_codegen_cranelift::guard::CraneliftFailDescr::new(0, vec![Type::Int, Type::Int]);

    let bridge_info = backend
        .compile_bridge(
            &bridge_fail_descr,
            &bridge_trace.inputargs,
            &bridge_optimized,
            &token,
        )
        .expect("bridge compilation should succeed");
    assert!(bridge_info.code_addr != 0);

    // Execute with bridge: i=5, sum=0
    // Loop runs same as before, guard fails with i=1, sum=14
    // Bridge runs: sum * 2 = 14 * 2 = 28
    let frame = backend.execute_token(&token, &[Value::Int(5), Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 28);

    // Also test with different initial values: i=3, sum=0
    //   iter1: i=3,sum=0 -> sum2=3, i2=2, pass -> jump(2,3)
    //   iter2: i=2,sum=3 -> sum2=5, i2=1, pass -> jump(1,5)
    //   iter3: i=1,sum=5 -> sum2=6, i2=0, fail
    // Guard saves i=1, sum=5 -> bridge: 5 * 2 = 10
    let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(0)]);
    assert_eq!(backend.get_int_value(&frame, 0), 10);
}

// ---------------------------------------------------------------------------
// Helpers for intdiv pipeline tests
// ---------------------------------------------------------------------------

/// Floor division (towards negative infinity).
fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    let r = a % b;
    if (r != 0) && ((r ^ b) < 0) { d - 1 } else { d }
}

/// Floor modulo: a - floor_div(a, b) * b.
fn floor_mod(a: i64, b: i64) -> i64 {
    a - floor_div(a, b) * b
}

/// Build a magic-number division trace directly (no optimizer), compile it,
/// and return (backend, token).
///
/// The trace implements: result = floor_div(input, m)
///   t = input >> 63
///   nt = input ^ t
///   mul = UINT_MUL_HIGH(nt, k)
///   sh = UINT_RSHIFT(mul, i)
///   result = sh ^ t
///   finish(result)
fn build_magic_div_trace(m: i64, token_id: u64) -> (CraneliftBackend, LoopToken) {
    let (k, i) = magic_numbers(m);

    // Record the magic-number division sequence using TraceRecorder
    // so that OpRef indexing is handled correctly.
    let mut rec = TraceRecorder::new();
    let x = rec.record_input_arg(Type::Int);

    // Constants via high OpRef indices.
    let const_k = OpRef(1000);
    let const_i = OpRef(1001);
    let const_63 = OpRef(1002);

    // t = x >> 63
    let t = rec.record_op(OpCode::IntRshift, &[x, const_63]);
    // nt = x ^ t
    let nt = rec.record_op(OpCode::IntXor, &[x, t]);
    // mul = UINT_MUL_HIGH(nt, k)
    let mul = rec.record_op(OpCode::UintMulHigh, &[nt, const_k]);
    // sh = UINT_RSHIFT(mul, i)
    let sh = rec.record_op(OpCode::UintRshift, &[mul, const_i]);
    // result = sh ^ t
    let result = rec.record_op(OpCode::IntXor, &[sh, t]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    let mut constants = HashMap::new();
    constants.insert(1000, k as i64);
    constants.insert(1001, i as i64);
    constants.insert(1002, 63i64);

    let mut backend = CraneliftBackend::new();
    backend.set_constants(constants);

    let mut token = LoopToken::new(token_id);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    (backend, token)
}

/// Build a magic-number modulo trace: result = n - (n // m) * m.
fn build_magic_mod_trace(m: i64, token_id: u64) -> (CraneliftBackend, LoopToken) {
    let (k, i) = magic_numbers(m);

    let mut rec = TraceRecorder::new();
    let x = rec.record_input_arg(Type::Int);

    let const_k = OpRef(1000);
    let const_i = OpRef(1001);
    let const_63 = OpRef(1002);
    let const_m = OpRef(1003);

    // Division: floor_div(x, m)
    let t = rec.record_op(OpCode::IntRshift, &[x, const_63]);
    let nt = rec.record_op(OpCode::IntXor, &[x, t]);
    let mul = rec.record_op(OpCode::UintMulHigh, &[nt, const_k]);
    let sh = rec.record_op(OpCode::UintRshift, &[mul, const_i]);
    let div = rec.record_op(OpCode::IntXor, &[sh, t]);
    // Modulo: x - div * m
    let product = rec.record_op(OpCode::IntMul, &[div, const_m]);
    let remainder = rec.record_op(OpCode::IntSub, &[x, product]);
    rec.finish(&[remainder], make_descr(0));
    let trace = rec.get_trace();

    let mut constants = HashMap::new();
    constants.insert(1000, k as i64);
    constants.insert(1001, i as i64);
    constants.insert(1002, 63i64);
    constants.insert(1003, m);

    let mut backend = CraneliftBackend::new();
    backend.set_constants(constants);

    let mut token = LoopToken::new(token_id);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    (backend, token)
}

/// Build a power-of-2 division trace:
///   sign = x >> 63
///   correction = sign & (divisor - 1)
///   adjusted = x + correction
///   result = adjusted >> shift
///   finish(result)
fn build_power_of_two_div_trace(divisor: i64, token_id: u64) -> (CraneliftBackend, LoopToken) {
    assert!(divisor > 1 && divisor.count_ones() == 1);
    let shift = divisor.trailing_zeros();

    let mut rec = TraceRecorder::new();
    let x = rec.record_input_arg(Type::Int);

    let const_63 = OpRef(1000);
    let const_mask = OpRef(1001);
    let const_shift = OpRef(1002);

    let sign = rec.record_op(OpCode::IntRshift, &[x, const_63]);
    let correction = rec.record_op(OpCode::IntAnd, &[sign, const_mask]);
    let adjusted = rec.record_op(OpCode::IntAdd, &[x, correction]);
    let result = rec.record_op(OpCode::IntRshift, &[adjusted, const_shift]);
    rec.finish(&[result], make_descr(0));
    let trace = rec.get_trace();

    let mut constants = HashMap::new();
    constants.insert(1000, 63i64);
    constants.insert(1001, divisor - 1);
    constants.insert(1002, shift as i64);

    let mut backend = CraneliftBackend::new();
    backend.set_constants(constants);

    let mut token = LoopToken::new(token_id);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    (backend, token)
}

// ---------------------------------------------------------------------------
// Test 7: IntFloorDiv magic-number pipeline (divisor = 7)
//
// The magic-number algorithm produces floor division (towards -inf).
// ---------------------------------------------------------------------------

#[test]
fn test_intdiv_magic_number_pipeline() {
    let (backend, token) = build_magic_div_trace(7, 100);

    let test_cases: &[(i64, i64)] = &[
        (0, 0),
        (1, 0),
        (6, 0),
        (7, 1),
        (14, 2),
        (100, 14),
        (-1, -1),
        (-6, -1),
        (-7, -1),
        (-8, -2),
        (-100, -15),
        (i64::MAX, floor_div(i64::MAX, 7)),
        (i64::MIN + 1, floor_div(i64::MIN + 1, 7)),
    ];

    for &(input, expected) in test_cases {
        let frame = backend.execute_token(&token, &[Value::Int(input)]);
        let actual = backend.get_int_value(&frame, 0);
        assert_eq!(
            actual, expected,
            "IntFloorDiv: {input} // 7 = expected {expected}, got {actual}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 8: IntMod magic-number pipeline (divisor = 7)
// ---------------------------------------------------------------------------

#[test]
fn test_intmod_magic_number_pipeline() {
    let (backend, token) = build_magic_mod_trace(7, 101);

    let test_cases: &[(i64, i64)] = &[
        (0, 0),
        (1, 1),
        (6, 6),
        (7, 0),
        (14, 0),
        (100, floor_mod(100, 7)),
        (-1, floor_mod(-1, 7)),
        (-7, 0),
        (-8, floor_mod(-8, 7)),
        (-100, floor_mod(-100, 7)),
        (i64::MAX, floor_mod(i64::MAX, 7)),
        (i64::MIN + 1, floor_mod(i64::MIN + 1, 7)),
    ];

    for &(input, expected) in test_cases {
        let frame = backend.execute_token(&token, &[Value::Int(input)]);
        let actual = backend.get_int_value(&frame, 0);
        assert_eq!(
            actual, expected,
            "IntMod: {input} % 7 = expected {expected}, got {actual}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 9: IntFloorDiv power-of-2 pipeline (divisor = 8)
//
// The power-of-2 strength reduction uses the Hacker's Delight formula:
//   result = (x + ((x >> 63) & (2^n - 1))) >> n
// This produces truncation division (towards zero), matching Cranelift sdiv.
// ---------------------------------------------------------------------------

#[test]
fn test_intdiv_power_of_two_pipeline() {
    let (backend, token) = build_power_of_two_div_trace(8, 102);

    let test_cases: &[(i64, i64)] = &[
        (0, 0),
        (1, 0),
        (7, 0),
        (8, 1),
        (16, 2),
        (100, 12),
        (-1, 0),       // truncation: -1/8 = 0
        (-7, 0),       // truncation: -7/8 = 0
        (-8, -1),      // exact: -8/8 = -1
        (-9, -1),      // truncation: -9/8 = -1
        (-100, -12),   // truncation: -100/8 = -12
        (i64::MAX, i64::MAX / 8),
        (i64::MIN + 1, (i64::MIN + 1) / 8),
    ];

    for &(input, expected) in test_cases {
        let frame = backend.execute_token(&token, &[Value::Int(input)]);
        let actual = backend.get_int_value(&frame, 0);
        assert_eq!(
            actual, expected,
            "IntFloorDiv(pow2): {input} / 8 = expected {expected}, got {actual}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 10: IntFloorDiv magic-number various divisors
// ---------------------------------------------------------------------------

#[test]
fn test_intdiv_various_divisors() {
    for (tid, divisor) in [3i64, 5, 10, 13, 100, 127].iter().enumerate() {
        let (backend, token) = build_magic_div_trace(*divisor, 200 + tid as u64);

        for input in [0i64, 1, *divisor - 1, *divisor, *divisor + 1, 999, -1, -*divisor, -999] {
            let expected = floor_div(input, *divisor);
            let frame = backend.execute_token(&token, &[Value::Int(input)]);
            let actual = backend.get_int_value(&frame, 0);
            assert_eq!(
                actual, expected,
                "IntFloorDiv: {input} // {divisor} = expected {expected}, got {actual}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 11: VecIntAdd native SIMD (pack + add + unpack)
// ---------------------------------------------------------------------------

#[test]
fn test_vec_int_add_simd() {
    // input(a, b, c, d)
    // vec0 = VecI()
    // vec1 = VecPackI(vec0, a, 0, 2)
    // vec2 = VecPackI(vec1, b, 1, 2)
    // vec3 = VecI()
    // vec4 = VecPackI(vec3, c, 0, 2)
    // vec5 = VecPackI(vec4, d, 1, 2)
    // vec6 = VecIntAdd(vec2, vec5)
    // r0 = VecUnpackI(vec6, 0, 2)
    // r1 = VecUnpackI(vec6, 1, 2)
    // finish(r0, r1)
    let mut rec = TraceRecorder::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);

    let const_0 = OpRef(1000);
    let const_1 = OpRef(1001);
    let const_2 = OpRef(1002);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec3 = rec.record_op(OpCode::VecI, &[]);
    let vec4 = rec.record_op(OpCode::VecPackI, &[vec3, c, const_0, const_2]);
    let vec5 = rec.record_op(OpCode::VecPackI, &[vec4, d, const_1, const_2]);
    let vec6 = rec.record_op(OpCode::VecIntAdd, &[vec2, vec5]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 0i64);
    constants.insert(1001, 1i64);
    constants.insert(1002, 2i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(300);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    // a=10, b=20, c=3, d=7 -> r0 = 10+3 = 13, r1 = 20+7 = 27
    let frame = backend.execute_token(
        &token,
        &[Value::Int(10), Value::Int(20), Value::Int(3), Value::Int(7)],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 13);
    assert_eq!(backend.get_int_value(&frame, 1), 27);

    // Negative values: a=-5, b=100, c=15, d=-200
    let frame = backend.execute_token(
        &token,
        &[
            Value::Int(-5),
            Value::Int(100),
            Value::Int(15),
            Value::Int(-200),
        ],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 10); // -5 + 15
    assert_eq!(backend.get_int_value(&frame, 1), -100); // 100 + (-200)
}

// ---------------------------------------------------------------------------
// Test 12: VecIntSub native SIMD
// ---------------------------------------------------------------------------

#[test]
fn test_vec_int_sub_simd() {
    let mut rec = TraceRecorder::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);

    let const_0 = OpRef(1000);
    let const_1 = OpRef(1001);
    let const_2 = OpRef(1002);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec3 = rec.record_op(OpCode::VecI, &[]);
    let vec4 = rec.record_op(OpCode::VecPackI, &[vec3, c, const_0, const_2]);
    let vec5 = rec.record_op(OpCode::VecPackI, &[vec4, d, const_1, const_2]);
    let vec6 = rec.record_op(OpCode::VecIntSub, &[vec2, vec5]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 0i64);
    constants.insert(1001, 1i64);
    constants.insert(1002, 2i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(301);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    // a=10, b=20, c=3, d=7 -> r0 = 10-3 = 7, r1 = 20-7 = 13
    let frame = backend.execute_token(
        &token,
        &[Value::Int(10), Value::Int(20), Value::Int(3), Value::Int(7)],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 7);
    assert_eq!(backend.get_int_value(&frame, 1), 13);
}

// ---------------------------------------------------------------------------
// Test 13: VecIntMul native SIMD
// ---------------------------------------------------------------------------

#[test]
fn test_vec_int_mul_simd() {
    let mut rec = TraceRecorder::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);

    let const_0 = OpRef(1000);
    let const_1 = OpRef(1001);
    let const_2 = OpRef(1002);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec3 = rec.record_op(OpCode::VecI, &[]);
    let vec4 = rec.record_op(OpCode::VecPackI, &[vec3, c, const_0, const_2]);
    let vec5 = rec.record_op(OpCode::VecPackI, &[vec4, d, const_1, const_2]);
    let vec6 = rec.record_op(OpCode::VecIntMul, &[vec2, vec5]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 0i64);
    constants.insert(1001, 1i64);
    constants.insert(1002, 2i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(302);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    // a=5, b=6, c=7, d=8 -> r0 = 5*7 = 35, r1 = 6*8 = 48
    let frame = backend.execute_token(
        &token,
        &[Value::Int(5), Value::Int(6), Value::Int(7), Value::Int(8)],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 35);
    assert_eq!(backend.get_int_value(&frame, 1), 48);
}

// ---------------------------------------------------------------------------
// Test 14: VecExpandI + VecIntAdd (broadcast + vector add)
// ---------------------------------------------------------------------------

#[test]
fn test_vec_expand_add_simd() {
    // input(a, b, s)
    // vec_ab = pack(a, b)
    // vec_s = VecExpandI(s)       -- broadcast s to both lanes
    // vec_r = VecIntAdd(vec_ab, vec_s)
    // r0 = unpack(vec_r, 0)
    // r1 = unpack(vec_r, 1)
    // finish(r0, r1)
    let mut rec = TraceRecorder::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let s = rec.record_input_arg(Type::Int);

    let const_0 = OpRef(1000);
    let const_1 = OpRef(1001);
    let const_2 = OpRef(1002);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec_s = rec.record_op(OpCode::VecExpandI, &[s]);
    let vec_r = rec.record_op(OpCode::VecIntAdd, &[vec2, vec_s]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec_r, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec_r, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 0i64);
    constants.insert(1001, 1i64);
    constants.insert(1002, 2i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(303);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    // a=10, b=20, s=100 -> r0 = 10+100 = 110, r1 = 20+100 = 120
    let frame = backend.execute_token(
        &token,
        &[Value::Int(10), Value::Int(20), Value::Int(100)],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 110);
    assert_eq!(backend.get_int_value(&frame, 1), 120);
}

// ---------------------------------------------------------------------------
// Test 15: VecFloatAdd native SIMD
// ---------------------------------------------------------------------------

#[test]
fn test_vec_float_add_simd() {
    // We use CastIntToFloat to convert i64 inputs to f64, then pack into
    // F64X2 vectors, add, unpack, and CastFloatToInt to get back i64.
    // This avoids type mismatch in Finish fail_arg_types.
    //
    // Use VecPackI/VecUnpackI with VecFloatAdd: the VecFloatAdd reads
    // I64X2 operands and reinterprets them as F64X2 internally.
    // Instead, use Int-typed inputs, CastIntToFloat for each, then pack
    // into F64X2 and unpack as Float, then CastFloatToInt.
    //
    // Simpler: just use Int paths throughout, packing f64 bit patterns
    // into I64X2 via VecPackI, then use VecFloatAdd (which handles the
    // bitcast internally), then VecUnpackI to get scalar i64 back.
    let mut rec = TraceRecorder::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);

    let const_0 = OpRef(1000);
    let const_1 = OpRef(1001);
    let const_2 = OpRef(1002);

    // Pack f64 bit patterns into I64X2 vectors
    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec1 = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec2 = rec.record_op(OpCode::VecPackI, &[vec1, b, const_1, const_2]);
    let vec3 = rec.record_op(OpCode::VecI, &[]);
    let vec4 = rec.record_op(OpCode::VecPackI, &[vec3, c, const_0, const_2]);
    let vec5 = rec.record_op(OpCode::VecPackI, &[vec4, d, const_1, const_2]);
    // VecFloatAdd operates on the I64X2 as if they contain f64 bit patterns
    let vec6 = rec.record_op(OpCode::VecFloatAdd, &[vec2, vec5]);
    // Unpack using VecUnpackI to get scalar i64 (containing f64 bits)
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec6, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 0i64);
    constants.insert(1001, 1i64);
    constants.insert(1002, 2i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(304);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    // Pass f64 values as i64 bit patterns
    let a_f = 1.5f64;
    let b_f = 2.5f64;
    let c_f = 3.0f64;
    let d_f = 4.0f64;

    let frame = backend.execute_token(
        &token,
        &[
            Value::Int(a_f.to_bits() as i64),
            Value::Int(b_f.to_bits() as i64),
            Value::Int(c_f.to_bits() as i64),
            Value::Int(d_f.to_bits() as i64),
        ],
    );
    let r0_bits = backend.get_int_value(&frame, 0) as u64;
    let r1_bits = backend.get_int_value(&frame, 1) as u64;
    assert_eq!(f64::from_bits(r0_bits), 4.5); // 1.5 + 3.0
    assert_eq!(f64::from_bits(r1_bits), 6.5); // 2.5 + 4.0
}

// ---------------------------------------------------------------------------
// Test 16: Chained VecIntAdd + VecIntMul (vector add then multiply)
// ---------------------------------------------------------------------------

#[test]
fn test_vec_chained_add_mul_simd() {
    // input(a, b, c, d, e, f)
    // vec_ab = pack(a, b)
    // vec_cd = pack(c, d)
    // vec_ef = pack(e, f)
    // vec_sum = VecIntAdd(vec_ab, vec_cd)
    // vec_result = VecIntMul(vec_sum, vec_ef)
    // r0 = unpack(vec_result, 0)
    // r1 = unpack(vec_result, 1)
    // finish(r0, r1)
    let mut rec = TraceRecorder::new();
    let a = rec.record_input_arg(Type::Int);
    let b = rec.record_input_arg(Type::Int);
    let c = rec.record_input_arg(Type::Int);
    let d = rec.record_input_arg(Type::Int);
    let e = rec.record_input_arg(Type::Int);
    let f = rec.record_input_arg(Type::Int);

    let const_0 = OpRef(1000);
    let const_1 = OpRef(1001);
    let const_2 = OpRef(1002);

    let vec0 = rec.record_op(OpCode::VecI, &[]);
    let vec_a = rec.record_op(OpCode::VecPackI, &[vec0, a, const_0, const_2]);
    let vec_ab = rec.record_op(OpCode::VecPackI, &[vec_a, b, const_1, const_2]);

    let vec1 = rec.record_op(OpCode::VecI, &[]);
    let vec_c = rec.record_op(OpCode::VecPackI, &[vec1, c, const_0, const_2]);
    let vec_cd = rec.record_op(OpCode::VecPackI, &[vec_c, d, const_1, const_2]);

    let vec2 = rec.record_op(OpCode::VecI, &[]);
    let vec_e = rec.record_op(OpCode::VecPackI, &[vec2, e, const_0, const_2]);
    let vec_ef = rec.record_op(OpCode::VecPackI, &[vec_e, f, const_1, const_2]);

    let vec_sum = rec.record_op(OpCode::VecIntAdd, &[vec_ab, vec_cd]);
    let vec_result = rec.record_op(OpCode::VecIntMul, &[vec_sum, vec_ef]);
    let r0 = rec.record_op(OpCode::VecUnpackI, &[vec_result, const_0, const_2]);
    let r1 = rec.record_op(OpCode::VecUnpackI, &[vec_result, const_1, const_2]);
    rec.finish(&[r0, r1], make_descr(0));
    let trace = rec.get_trace();

    let mut backend = CraneliftBackend::new();
    let mut constants = HashMap::new();
    constants.insert(1000, 0i64);
    constants.insert(1001, 1i64);
    constants.insert(1002, 2i64);
    backend.set_constants(constants);

    let mut token = LoopToken::new(305);
    backend
        .compile_loop(&trace.inputargs, &trace.ops, &mut token)
        .expect("compilation should succeed");

    // a=2, b=3, c=4, d=5, e=10, f=20
    // sum = (2+4, 3+5) = (6, 8)
    // result = (6*10, 8*20) = (60, 160)
    let frame = backend.execute_token(
        &token,
        &[
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
            Value::Int(5),
            Value::Int(10),
            Value::Int(20),
        ],
    );
    assert_eq!(backend.get_int_value(&frame, 0), 60);
    assert_eq!(backend.get_int_value(&frame, 1), 160);
}
