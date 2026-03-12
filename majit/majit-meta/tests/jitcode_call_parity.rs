use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};

use majit_ir::OpCode;
use majit_meta::{
    assert_trace_parity, trace_jitcode, BackEdgeAction, JitCodeBuilder, JitCodeMachine, JitCodeSym,
    MIFrame, MetaInterp, SymbolicStack, TraceAction, TraceParityCase,
};

static LAST_RAW_VOID_SUM: AtomicI64 = AtomicI64::new(-1);

extern "C" fn weighted_sum4(a: i64, b: i64, c: i64, d: i64) -> i64 {
    a + 10 * b + 100 * c + 1000 * d
}

extern "C" fn sink_sum4(a: i64, b: i64, c: i64, d: i64) {
    LAST_RAW_VOID_SUM.store(a + b + c + d, Ordering::SeqCst);
}

struct TestSym {
    stacks: HashMap<usize, SymbolicStack>,
    current_selected: usize,
    loop_header_pc: usize,
}

impl TestSym {
    fn new(input_depth: usize) -> Self {
        let mut stacks = HashMap::new();
        stacks.insert(0, SymbolicStack::from_input_args(0, input_depth));
        Self {
            stacks,
            current_selected: 0,
            loop_header_pc: 0,
        }
    }
}

impl JitCodeSym for TestSym {
    fn current_selected(&self) -> usize {
        self.current_selected
    }

    fn set_current_selected(&mut self, selected: usize) {
        self.current_selected = selected;
    }

    fn stack(&self, selected: usize) -> Option<&SymbolicStack> {
        self.stacks.get(&selected)
    }

    fn stack_mut(&mut self, selected: usize) -> Option<&mut SymbolicStack> {
        self.stacks.get_mut(&selected)
    }

    fn total_slots(&self) -> usize {
        self.stacks.values().map(SymbolicStack::len).sum()
    }

    fn loop_header_pc(&self) -> usize {
        self.loop_header_pc
    }

    fn ensure_stack(&mut self, selected: usize, offset: usize, len: usize) {
        self.stacks
            .entry(selected)
            .or_insert_with(|| SymbolicStack::from_input_args(offset, len));
    }
}

fn start_trace(live: &[i64]) -> MetaInterp<()> {
    let mut interp = MetaInterp::new(1);
    let action = interp.on_back_edge(0, live);
    assert!(matches!(action, BackEdgeAction::StartedTracing));
    interp
}

#[test]
fn jitcode_inline_call_moves_int_args_and_return_value() {
    let mut callee = JitCodeBuilder::new();
    callee.record_binop_i(2, OpCode::IntAdd, 0, 1);

    let mut root = JitCodeBuilder::new();
    root.pop_i(0);
    root.pop_i(1);
    let sub_idx = root.add_sub_jitcode(callee.finish());
    root.inline_call_i(sub_idx, &[(0, 0), (1, 1)], Some((2, 2)));
    root.push_i(2);
    let jitcode = root.finish();

    let mut interp = start_trace(&[3, 4]);
    let mut sym = TestSym::new(2);
    let initial = [3_i64, 4_i64];
    let action = {
        let ctx = interp.trace_ctx().unwrap();
        trace_jitcode(
            ctx,
            &mut sym,
            &jitcode,
            0,
            |_| initial.len(),
            |_, pos| initial[pos],
            |_| 0,
        )
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stack(0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jitcode_inline_call_moves_args_and_return",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: MIFrame argument passing and return",
        expected_lines: &["v2 = IntAdd(v1, v0)", "Finish(v2)"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jitcode_nested_inline_call_feeds_returned_register_into_next_call() {
    let mut callee_add = JitCodeBuilder::new();
    callee_add.record_binop_i(2, OpCode::IntAdd, 0, 1);

    let mut callee_neg = JitCodeBuilder::new();
    callee_neg.record_unary_i(1, OpCode::IntNeg, 0);

    let mut root = JitCodeBuilder::new();
    root.pop_i(0);
    root.pop_i(1);
    let add_idx = root.add_sub_jitcode(callee_add.finish());
    let neg_idx = root.add_sub_jitcode(callee_neg.finish());
    root.inline_call_i(add_idx, &[(0, 0), (1, 1)], Some((2, 2)));
    root.inline_call_i(neg_idx, &[(2, 0)], Some((1, 3)));
    root.push_i(3);
    let jitcode = root.finish();

    let mut interp = start_trace(&[3, 4]);
    let mut sym = TestSym::new(2);
    let initial = [3_i64, 4_i64];
    let action = {
        let ctx = interp.trace_ctx().unwrap();
        trace_jitcode(
            ctx,
            &mut sym,
            &jitcode,
            0,
            |_| initial.len(),
            |_, pos| initial[pos],
            |_| 0,
        )
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stack(0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jitcode_nested_inline_call_return_chain",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: nested MIFrame return propagation",
        expected_lines: &["v2 = IntAdd(v1, v0)", "v3 = IntNeg(v2)", "Finish(v3)"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jitcode_inline_call_updates_trace_ctx_inline_depth() {
    let mut callee = JitCodeBuilder::new();
    callee.record_unary_i(1, OpCode::IntNeg, 0);

    let mut root = JitCodeBuilder::new();
    root.pop_i(0);
    let sub_idx = root.add_sub_jitcode(callee.finish());
    root.inline_call_i(sub_idx, &[(0, 0)], Some((1, 1)));
    root.push_i(1);
    let jitcode = root.finish();

    let mut interp = start_trace(&[5]);
    let mut sym = TestSym::new(1);
    let initial = [5_i64];
    let root = MIFrame::new(&jitcode, 0);
    let mut machine =
        JitCodeMachine::<TestSym, _>::new(root, &jitcode.sub_jitcodes, &jitcode.fn_ptrs);
    let runtime = majit_meta::ClosureRuntime::new(|_| initial.len(), |_, pos| initial[pos], |_| 0);

    let ctx = interp.trace_ctx().unwrap();
    assert_eq!(ctx.inline_depth(), 0);
    assert!(matches!(
        machine.run_one_step(ctx, &mut sym, &runtime),
        TraceAction::Continue
    ));
    assert_eq!(ctx.inline_depth(), 0);
    assert!(matches!(
        machine.run_one_step(ctx, &mut sym, &runtime),
        TraceAction::Continue
    ));
    assert_eq!(ctx.inline_depth(), 1);
    assert!(matches!(
        machine.run_one_step(ctx, &mut sym, &runtime),
        TraceAction::Continue
    ));
    assert_eq!(ctx.inline_depth(), 1);
    assert!(matches!(
        machine.run_one_step(ctx, &mut sym, &runtime),
        TraceAction::Continue
    ));
    assert_eq!(ctx.inline_depth(), 0);
}

#[test]
fn jitcode_call_pure_int_supports_four_args() {
    let mut root = JitCodeBuilder::new();
    root.pop_i(0);
    root.pop_i(1);
    root.pop_i(2);
    root.pop_i(3);
    let fn_idx = root.add_fn_ptr(weighted_sum4 as *const ());
    root.call_pure_int(fn_idx, &[0, 1, 2, 3], 4);
    root.push_i(4);
    let jitcode = root.finish();

    let mut interp = start_trace(&[1, 2, 3, 4]);
    let mut sym = TestSym::new(4);
    let initial = [1_i64, 2_i64, 3_i64, 4_i64];
    let action = {
        let ctx = interp.trace_ctx().unwrap();
        trace_jitcode(
            ctx,
            &mut sym,
            &jitcode,
            0,
            |_| initial.len(),
            |_, pos| initial[pos],
            |_| 0,
        )
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stack(0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops[0].opcode, OpCode::CallPureI);
}

#[test]
fn jitcode_residual_call_void_supports_four_args() {
    LAST_RAW_VOID_SUM.store(-1, Ordering::SeqCst);

    let mut root = JitCodeBuilder::new();
    root.pop_i(0);
    root.pop_i(1);
    root.pop_i(2);
    root.pop_i(3);
    let fn_idx = root.add_fn_ptr(sink_sum4 as *const ());
    root.residual_call_void_args(fn_idx, &[0, 1, 2, 3]);
    root.push_i(0);
    let jitcode = root.finish();

    let mut interp = start_trace(&[1, 2, 3, 4]);
    let mut sym = TestSym::new(4);
    let initial = [1_i64, 2_i64, 3_i64, 4_i64];
    let action = {
        let ctx = interp.trace_ctx().unwrap();
        trace_jitcode(
            ctx,
            &mut sym,
            &jitcode,
            0,
            |_| initial.len(),
            |_, pos| initial[pos],
            |_| 0,
        )
    };
    assert!(matches!(action, TraceAction::Continue));
    assert_eq!(LAST_RAW_VOID_SUM.load(Ordering::SeqCst), 10);

    let result = sym.stack(0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops[0].opcode, OpCode::CallN);
}
