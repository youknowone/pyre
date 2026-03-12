use std::sync::atomic::{AtomicI64, Ordering};

use majit_ir::OpCode;
use majit_macros::{dont_look_inside, elidable, jit_inline, jit_interp};
use majit_meta::{BackEdgeAction, JitState, MetaInterp, TraceAction};

const UNTRACEABLE: usize = 99;
const OP_INLINE_CHAIN: u8 = 1;
const OP_INLINE_NESTED: u8 = 2;
const OP_INLINE_ELIDABLE: u8 = 3;
const OP_INLINE_RESIDUAL_VOID: u8 = 4;
const OP_INLINE_FOUR_ARGS: u8 = 5;

static LAST_INLINE_SINK_SUM: AtomicI64 = AtomicI64::new(-1);

#[jit_inline]
fn inline_sum_plus_one(lhs: i64, rhs: i64) -> i64 {
    let sum = lhs + rhs;
    sum + 1
}

#[jit_inline]
fn inline_negate(value: i64) -> i64 {
    -value
}

#[jit_inline]
fn inline_sum_four(a: i64, b: i64, c: i64, d: i64) -> i64 {
    let ab = a + b;
    let cd = c + d;
    ab + cd
}

#[jit_inline]
fn inline_nested_chain(lhs: i64, rhs: i64) -> i64 {
    inline_negate(inline_sum_plus_one(lhs, rhs))
}

#[elidable]
extern "C" fn inline_leaf_elidable(value: i64) -> i64 {
    value * value + 1
}

#[jit_inline]
fn inline_calls_elidable(value: i64) -> i64 {
    inline_leaf_elidable(value) + 2
}

#[dont_look_inside]
extern "C" fn inline_sink_sum(lhs: i64, rhs: i64) {
    LAST_INLINE_SINK_SUM.store(lhs + rhs, Ordering::SeqCst);
}

#[jit_inline]
fn inline_with_sink(lhs: i64, rhs: i64) -> i64 {
    inline_sink_sum(lhs, rhs);
    lhs + rhs
}

#[derive(Clone, Default)]
struct Store {
    values: Vec<i64>,
}

impl Store {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn push(&mut self, value: i64) {
        self.values.push(value);
    }

    fn pop(&mut self) -> Option<i64> {
        self.values.pop()
    }

    fn peek_at(&self, index: usize) -> i64 {
        self.values[index]
    }

    fn clear(&mut self) {
        self.values.clear();
    }
}

#[derive(Clone)]
struct StoragePool {
    stores: Vec<Store>,
}

impl StoragePool {
    fn new(stacks: &[&[i64]]) -> Self {
        Self {
            stores: stacks
                .iter()
                .map(|stack| Store {
                    values: stack.to_vec(),
                })
                .collect(),
        }
    }

    fn get(&self, index: usize) -> &Store {
        &self.stores[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut Store {
        &mut self.stores[index]
    }
}

struct Program {
    ops: Vec<u8>,
}

impl Program {
    fn get_op(&self, pc: usize) -> u8 {
        self.ops[pc]
    }
}

struct TestState {
    storage: StoragePool,
    selected: usize,
}

impl TestState {
    fn pop(&mut self) -> Option<i64> {
        self.storage.get_mut(self.selected).pop()
    }

    fn push(&mut self, value: i64) {
        self.storage.get_mut(self.selected).push(value);
    }
}

fn scan_used_storages(_program: &Program, _header_pc: usize, selected: usize) -> Vec<usize> {
    vec![selected]
}

#[jit_interp(
    state = TestState,
    env = Program,
    storage = {
        pool: state.storage,
        pool_type: StoragePool,
        selector: state.selected,
        untraceable: [UNTRACEABLE],
        scan: scan_used_storages,
    },
    binops = {},
    io_shims = {},
    calls = {
        inline_sum_plus_one,
        inline_negate,
        inline_sum_four,
        inline_nested_chain,
        inline_calls_elidable,
        inline_with_sink => inline_int,
    },
)]
fn trace_inline_step(program: &Program, pc: usize, state: &mut TestState) -> i64 {
    let op = program.get_op(pc);
    match op {
        OP_INLINE_CHAIN => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            let sum = inline_sum_plus_one(lhs, rhs);
            state.push(inline_negate(sum));
        }
        OP_INLINE_NESTED => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            state.push(inline_nested_chain(lhs, rhs));
        }
        OP_INLINE_ELIDABLE => {
            let value = state.pop().unwrap();
            state.push(inline_calls_elidable(value));
        }
        OP_INLINE_RESIDUAL_VOID => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            state.push(inline_with_sink(lhs, rhs));
        }
        OP_INLINE_FOUR_ARGS => {
            let d = state.pop().unwrap();
            let c = state.pop().unwrap();
            let b = state.pop().unwrap();
            let a = state.pop().unwrap();
            state.push(inline_sum_four(a, b, c, d));
        }
        _ => {}
    }
    state.storage.get(state.selected).peek_at(0)
}

fn start_trace(
    program: &Program,
    state: &TestState,
) -> (
    MetaInterp<<TestState as JitState>::Meta>,
    <TestState as JitState>::Sym,
) {
    let meta = state.build_meta(0, program);
    let live = state.extract_live(&meta);
    let mut interp = MetaInterp::new(1);
    let result = interp.on_back_edge(0, &live);
    assert!(matches!(result, BackEdgeAction::StartedTracing));
    let sym = TestState::create_sym(&meta, 0);
    (interp, sym)
}

#[test]
fn jit_interp_inline_helpers_lower_to_inlined_arith_ops() {
    let program = Program {
        ops: vec![OP_INLINE_CHAIN],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[3, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_inline_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(
        trace.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![
            OpCode::IntAdd,
            OpCode::IntAdd,
            OpCode::IntNeg,
            OpCode::Finish
        ]
    );
}

#[test]
fn jit_inline_helpers_keep_interpreter_behavior() {
    let program = Program {
        ops: vec![OP_INLINE_CHAIN],
    };
    let mut state = TestState {
        storage: StoragePool::new(&[&[2, 5]]),
        selected: 0,
    };

    assert_eq!(trace_inline_step(&program, 0, &mut state), -8);
    assert_eq!(
        trace_inline_step(
            &Program {
                ops: vec![OP_INLINE_FOUR_ARGS]
            },
            0,
            &mut TestState {
                storage: StoragePool::new(&[&[1, 2, 3, 4]]),
                selected: 0,
            }
        ),
        10
    );
}

#[test]
fn jit_inline_helpers_can_nest_other_inline_helpers() {
    let program = Program {
        ops: vec![OP_INLINE_NESTED],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[4, 6]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_inline_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(
        trace.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![
            OpCode::IntAdd,
            OpCode::IntAdd,
            OpCode::IntNeg,
            OpCode::Finish
        ]
    );
}

#[test]
fn jit_inline_helpers_can_issue_nested_call_pure_ops() {
    let program = Program {
        ops: vec![OP_INLINE_ELIDABLE],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[5]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_inline_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(
        trace.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![OpCode::CallPureI, OpCode::IntAdd, OpCode::Finish]
    );
}

#[test]
fn jit_inline_helpers_can_issue_nested_call_n_ops() {
    LAST_INLINE_SINK_SUM.store(-1, Ordering::SeqCst);
    let program = Program {
        ops: vec![OP_INLINE_RESIDUAL_VOID],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[7, 9]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_inline_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(
        trace.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![OpCode::CallN, OpCode::IntAdd, OpCode::Finish]
    );
}

#[test]
fn jit_inline_nested_call_n_keeps_interpreter_side_effects() {
    LAST_INLINE_SINK_SUM.store(-1, Ordering::SeqCst);
    let program = Program {
        ops: vec![OP_INLINE_RESIDUAL_VOID],
    };
    let mut state = TestState {
        storage: StoragePool::new(&[&[4, 8]]),
        selected: 0,
    };

    assert_eq!(trace_inline_step(&program, 0, &mut state), 12);
    assert_eq!(LAST_INLINE_SINK_SUM.load(Ordering::SeqCst), 12);
}

#[test]
fn jit_interp_inline_helper_supports_four_args() {
    let program = Program {
        ops: vec![OP_INLINE_FOUR_ARGS],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[1, 2, 3, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_inline_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(
        trace.ops.iter().map(|op| op.opcode).collect::<Vec<_>>(),
        vec![
            OpCode::IntAdd,
            OpCode::IntAdd,
            OpCode::IntAdd,
            OpCode::Finish
        ]
    );
}
