use std::sync::atomic::{AtomicI64, Ordering};

use majit_ir::{OpCode, OpRef};
use majit_macros::jit_interp;
use majit_meta::{
    assert_trace_parity, BackEdgeAction, JitState, MetaInterp, TraceAction, TraceParityCase,
};

const UNTRACEABLE: usize = 99;
const OP_COMPOUND_ADD: u8 = 1;
const OP_IO_WRITE: u8 = 2;
const OP_ROUTE_PUSH: u8 = 3;

static LAST_WRITTEN: AtomicI64 = AtomicI64::new(-1);

fn test_write_number(value: i64) {
    LAST_WRITTEN.store(value, Ordering::SeqCst);
}

extern "C" fn jit_write_number_shim(value: i64) {
    LAST_WRITTEN.store(value, Ordering::SeqCst);
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

    fn add(&mut self) {
        let rhs = self.pop().unwrap();
        let lhs = self.pop().unwrap();
        self.push(lhs + rhs);
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

fn scan_used_storages(_program: &Program, _header_pc: usize, _selected: usize) -> Vec<usize> {
    vec![0, 1]
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
    binops = {
        add => IntAdd,
    },
    io_shims = {
        test_write_number => jit_write_number_shim,
    },
)]
fn trace_config_step(program: &Program, pc: usize, state: &mut TestState) -> i64 {
    let op = program.get_op(pc);
    match op {
        OP_COMPOUND_ADD => {
            state.storage.get_mut(state.selected).add();
        }
        OP_IO_WRITE => {
            let value = state.storage.get_mut(state.selected).pop().unwrap();
            test_write_number(value);
            state.storage.get_mut(state.selected).push(value);
        }
        OP_ROUTE_PUSH => {
            let value = state.storage.get_mut(state.selected).pop().unwrap();
            state.selected = 1;
            state.storage.get_mut(1).push(value);
            state.selected = 0;
            state.storage.get_mut(0).push(99);
        }
        _ => {}
    }

    let stack = state.storage.get(state.selected);
    stack.peek_at(stack.len() - 1)
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
fn jit_interp_compound_storage_method_inlines_sub_jitcode() {
    let program = Program {
        ops: vec![OP_COMPOUND_ADD],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[3, 4], &[]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_config_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_compound_storage_method",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: inlined helper stack binop",
        expected_lines: &["v2 = IntAdd(v0, v1)", "Finish(v2)"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_io_shim_records_residual_call_and_runs_side_effect() {
    LAST_WRITTEN.store(-1, Ordering::SeqCst);

    let program = Program {
        ops: vec![OP_IO_WRITE],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[9], &[]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_config_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));
    assert_eq!(LAST_WRITTEN.load(Ordering::SeqCst), 9);

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallN);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
}

#[test]
fn jit_interp_selector_and_push_to_route_value_to_other_stack() {
    let program = Program {
        ops: vec![OP_ROUTE_PUSH],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[7], &[]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_config_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let moved = sym.stacks.get(&1).unwrap().peek().unwrap();
    assert_eq!(moved, OpRef(0));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_selector_and_push_to",
        rpython_reference:
            "rpython/jit/metainterp/pyjitpl.py: stateful helper side effects without IR ops",
        expected_lines: &["Finish(99)"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn annotated_config_function_still_runs_interpreter_path() {
    LAST_WRITTEN.store(-1, Ordering::SeqCst);

    let program = Program {
        ops: vec![OP_IO_WRITE],
    };
    let mut state = TestState {
        storage: StoragePool::new(&[&[11], &[]]),
        selected: 0,
    };

    assert_eq!(trace_config_step(&program, 0, &mut state), 11);
    assert_eq!(LAST_WRITTEN.load(Ordering::SeqCst), 11);
}
