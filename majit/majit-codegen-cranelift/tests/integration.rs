//! Integration tests: trace recording -> optimization -> Cranelift compilation -> execution.
//!
//! These tests exercise the full majit pipeline from end to end.

use std::collections::HashMap;
use std::sync::Arc;

use majit_codegen::{Backend, LoopToken};
use majit_codegen_cranelift::CraneliftBackend;
use majit_ir::{Descr, DescrRef, FailDescr, OpCode, OpRef, Type, Value};
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
    assert_eq!(backend.get_int_value(&frame, 0), 1);   // i at guard failure
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
    assert_eq!(backend.get_int_value(&frame, 0), 1);  // i
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
    let bridge_fail_descr = majit_codegen_cranelift::guard::CraneliftFailDescr::new(
        0,
        vec![Type::Int, Type::Int],
    );

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
