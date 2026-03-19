use majit_ir::OpCode;
use majit_macros::jit_interp;
use majit_meta::{
    BackEdgeAction, JitState, MetaInterp, TraceAction, TraceParityCase, assert_trace_parity,
};
use std::sync::Mutex;

mod aheui {
    pub const OP_BRPOP1: u8 = 1;
    pub const OP_BRPOP2: u8 = 2;
    pub const OP_BRZ: u8 = 3;
    pub const OP_JMP: u8 = 4;
}

const UNTRACEABLE: usize = 99;
static BRANCH_PARITY_TEST_LOCK: Mutex<()> = Mutex::new(());

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
    labels: Vec<usize>,
}

impl Program {
    fn get_op(&self, pc: usize) -> u8 {
        self.ops[pc]
    }

    fn get_label(&self, pc: usize) -> usize {
        self.labels[pc]
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
)]
fn branch_step(program: &Program, pc: usize, state: &mut TestState) -> i64 {
    let op = program.get_op(pc);
    match op {
        aheui::OP_BRPOP1 | aheui::OP_BRPOP2 | aheui::OP_BRZ | aheui::OP_JMP => {
            let branch = match op {
                aheui::OP_BRPOP1 | aheui::OP_BRPOP2 | aheui::OP_BRZ | aheui::OP_JMP => op,
                _ => unreachable!(),
            };

            match branch {
                aheui::OP_BRPOP1 => {
                    if state.storage.get(state.selected).len() < 1 {
                        return -1;
                    }
                }
                aheui::OP_BRPOP2 => {
                    if state.storage.get(state.selected).len() < 2 {
                        return -2;
                    }
                }
                aheui::OP_BRZ => {
                    let _target = program.get_label(pc);
                    let _ = state.pop().unwrap();
                }
                aheui::OP_JMP => {
                    let _ = program.get_label(pc);
                }
                _ => unreachable!(),
            }
        }
        _ => {}
    }
    state.storage.get(state.selected).len() as i64
}

fn start_trace(
    program: &Program,
    state: &TestState,
    header_pc: usize,
) -> (
    MetaInterp<<TestState as JitState>::Meta>,
    <TestState as JitState>::Sym,
) {
    let meta = state.build_meta(header_pc, program);
    let live = state.extract_live(&meta);
    let mut interp = MetaInterp::new(1);
    let result = interp.on_back_edge(header_pc as u64, &live);
    assert!(matches!(result, BackEdgeAction::StartedTracing));
    let sym = TestState::create_sym(&meta, header_pc);
    (interp, sym)
}

#[test]
fn jit_interp_branch_group_false_path_records_guard_false() {
    let _guard = BRANCH_PARITY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let program = Program {
        ops: vec![aheui::OP_BRZ],
        labels: vec![1],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[0]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state, 0);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_branch_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let (trace, constants) = interp.finish_trace_for_parity(&[]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_branch_group_guard_false",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: generate_guard(rop.GUARD_FALSE)",
        expected_lines: &["GuardFalse(v0) [fail_args=0, 0, 1]", "Finish()"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_branch_group_true_path_records_guard_true() {
    let _guard = BRANCH_PARITY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let program = Program {
        ops: vec![aheui::OP_BRZ],
        labels: vec![1],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[7]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state, 0);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_branch_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let (trace, constants) = interp.finish_trace_for_parity(&[]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_branch_group_guard_true",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: generate_guard(rop.GUARD_TRUE)",
        expected_lines: &["GuardTrue(v0) [fail_args=0, 0, 1]", "Finish()"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_branch_group_closes_loop_on_backedge() {
    let _guard = BRANCH_PARITY_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let program = Program {
        ops: vec![aheui::OP_BRZ],
        labels: vec![0],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[0]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state, 0);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_branch_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::CloseLoop));

    let (trace, constants) = interp.finish_trace_for_parity(&[]).unwrap();
    let normalized = majit_meta::normalize_trace(&trace, &constants);
    assert_eq!(trace.ops[0].opcode, OpCode::GuardFalse);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
    assert_eq!(normalized.last().unwrap(), "Finish()");
}
