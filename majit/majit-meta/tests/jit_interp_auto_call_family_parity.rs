use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

use majit_ir::OpCode;
use majit_macros::{jit_interp, jit_loop_invariant, jit_may_force, jit_release_gil};
use majit_meta::{BackEdgeAction, JitState, MetaInterp, TraceAction};

const UNTRACEABLE: usize = 99;
const OP_AUTO_MAY_FORCE_CALL: u8 = 1;
const OP_AUTO_RELEASE_GIL_VOID_CALL: u8 = 2;
const OP_AUTO_RELEASE_GIL_INT_CALL: u8 = 3;
const OP_AUTO_LOOPINVARIANT_INT_CALL: u8 = 4;

static LAST_AUTO_FAMILY_VOID_SUM: AtomicI64 = AtomicI64::new(-1);
static AUTO_CALL_FAMILY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[jit_may_force]
extern "C" fn may_force_add(x: i64, y: i64) -> i64 {
    x + y
}

#[jit_release_gil]
extern "C" fn release_sink_sum(x: i64, y: i64) {
    LAST_AUTO_FAMILY_VOID_SUM.store(x + y, Ordering::SeqCst);
}

#[jit_release_gil]
extern "C" fn release_add(x: i64, y: i64) -> i64 {
    x + y
}

#[jit_loop_invariant]
extern "C" fn loop_add_one(x: i64) -> i64 {
    x + 1
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

struct AutoFamilyState {
    storage: StoragePool,
    selected: usize,
}

impl AutoFamilyState {
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
    state = AutoFamilyState,
    env = Program,
    auto_calls = true,
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
fn trace_auto_call_family_step(program: &Program, pc: usize, state: &mut AutoFamilyState) -> i64 {
    let op = program.get_op(pc);
    match op {
        OP_AUTO_MAY_FORCE_CALL => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            state.push(may_force_add(lhs, rhs));
        }
        OP_AUTO_RELEASE_GIL_VOID_CALL => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            release_sink_sum(lhs, rhs);
        }
        OP_AUTO_RELEASE_GIL_INT_CALL => {
            let delta = state.pop().unwrap();
            let value = state.pop().unwrap();
            state.push(release_add(value, delta));
        }
        OP_AUTO_LOOPINVARIANT_INT_CALL => {
            let value = state.pop().unwrap();
            state.push(loop_add_one(value));
        }
        _ => {}
    }
    state.storage.get(state.selected).peek_at(0)
}

fn start_trace(
    program: &Program,
    state: &AutoFamilyState,
) -> (
    MetaInterp<<AutoFamilyState as JitState>::Meta>,
    <AutoFamilyState as JitState>::Sym,
) {
    let meta = state.build_meta(0, program);
    let live = state.extract_live(&meta);
    let mut interp = MetaInterp::new(1);
    let result = interp.on_back_edge(0, &live);
    assert!(matches!(result, BackEdgeAction::StartedTracing));
    let sym = AutoFamilyState::create_sym(&meta, 0);
    (interp, sym)
}

#[test]
fn jit_interp_auto_release_gil_void_call_infers_call_release_gil_n() {
    let _guard = AUTO_CALL_FAMILY_TEST_LOCK.lock().unwrap();
    LAST_AUTO_FAMILY_VOID_SUM.store(-1, Ordering::SeqCst);

    let program = Program {
        ops: vec![OP_AUTO_RELEASE_GIL_VOID_CALL],
    };
    let state = AutoFamilyState {
        storage: StoragePool::new(&[&[99, 5, 6]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_auto_call_family_step(
            ctx,
            &mut sym,
            &program,
            0,
            &state.storage,
            state.selected,
        )
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallReleaseGilN);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);

    let mut interp_state = AutoFamilyState {
        storage: StoragePool::new(&[&[99, 5, 6]]),
        selected: 0,
    };
    assert_eq!(
        trace_auto_call_family_step(&program, 0, &mut interp_state),
        99
    );
    assert_eq!(LAST_AUTO_FAMILY_VOID_SUM.load(Ordering::SeqCst), 11);
}

#[test]
fn jit_interp_auto_release_gil_int_call_infers_call_release_gil_i() {
    let program = Program {
        ops: vec![OP_AUTO_RELEASE_GIL_INT_CALL],
    };
    let state = AutoFamilyState {
        storage: StoragePool::new(&[&[7, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_auto_call_family_step(
            ctx,
            &mut sym,
            &program,
            0,
            &state.storage,
            state.selected,
        )
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallReleaseGilI);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);

    let mut interp_state = AutoFamilyState {
        storage: StoragePool::new(&[&[7, 4]]),
        selected: 0,
    };
    assert_eq!(
        trace_auto_call_family_step(&program, 0, &mut interp_state),
        11
    );
}

#[test]
fn jit_interp_auto_loopinvariant_int_call_infers_call_loopinvariant_i() {
    let program = Program {
        ops: vec![OP_AUTO_LOOPINVARIANT_INT_CALL],
    };
    let state = AutoFamilyState {
        storage: StoragePool::new(&[&[11]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_auto_call_family_step(
            ctx,
            &mut sym,
            &program,
            0,
            &state.storage,
            state.selected,
        )
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallLoopinvariantI);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);

    let mut interp_state = AutoFamilyState {
        storage: StoragePool::new(&[&[11]]),
        selected: 0,
    };
    assert_eq!(
        trace_auto_call_family_step(&program, 0, &mut interp_state),
        12
    );
}
