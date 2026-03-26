use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

use majit_ir::{GcRef, OpCode, Type};
use majit_macros::{dont_look_inside, elidable, jit_interp};
use majit_metainterp::{BackEdgeAction, JitState, MetaInterp, TraceAction};

const UNTRACEABLE: usize = 99;
const OP_ELIDABLE_CALL: u8 = 1;
const OP_OPAQUE_CALL: u8 = 2;
const OP_OPAQUE_VOID_CALL: u8 = 3;
const OP_ELIDABLE_CALL4: u8 = 4;
const OP_OPAQUE_CALL4: u8 = 5;
const OP_OPAQUE_VOID_CALL4: u8 = 6;
const OP_OPAQUE_REF_CALL: u8 = 7;
const OP_ELIDABLE_FLOAT_CALL: u8 = 8;
const OP_OPAQUE_REF_INT_CALL: u8 = 9;
const OP_ELIDABLE_FLOAT_INT_CALL: u8 = 10;

static LAST_VOID_SUM: AtomicI64 = AtomicI64::new(-1);
static LAST_VOID_SUM4: AtomicI64 = AtomicI64::new(-1);
static CALL_HELPER_TEST_LOCK: Mutex<()> = Mutex::new(());

#[elidable]
extern "C" fn compute_square_plus_one(x: i64) -> i64 {
    x * x + 1
}

#[elidable]
extern "C" fn compute_weighted_sum4(a: i64, b: i64, c: i64, d: i64) -> i64 {
    a + 10 * b + 100 * c + 1000 * d
}

#[dont_look_inside]
extern "C" fn opaque_add(x: i64, y: i64) -> i64 {
    x + y
}

#[dont_look_inside]
extern "C" fn opaque_add4(a: i64, b: i64, c: i64, d: i64) -> i64 {
    a + b + c + d
}

#[dont_look_inside]
extern "C" fn opaque_sink_sum(x: i64, y: i64) {
    LAST_VOID_SUM.store(x + y, Ordering::SeqCst);
}

#[dont_look_inside]
extern "C" fn opaque_sink_sum4(a: i64, b: i64, c: i64, d: i64) {
    LAST_VOID_SUM4.store(a + b + c + d, Ordering::SeqCst);
}

#[dont_look_inside]
extern "C" fn opaque_make_ref(x: i64) -> GcRef {
    GcRef((x + 1000) as usize)
}

#[elidable]
extern "C" fn elidable_make_float(x: i64) -> f64 {
    (x as f64) + 0.5
}

#[dont_look_inside]
extern "C" fn opaque_ref_plus_int(value: GcRef, delta: i64) -> i64 {
    value.as_usize() as i64 + delta
}

#[elidable]
extern "C" fn elidable_float_plus_int(value: f64, delta: i64) -> i64 {
    value.floor() as i64 + delta
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

    fn push_ref(&mut self, value: GcRef) {
        self.storage
            .get_mut(self.selected)
            .push(value.as_usize() as i64);
    }

    fn push_float(&mut self, value: f64) {
        self.storage
            .get_mut(self.selected)
            .push(value.to_bits() as i64);
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
        compute_square_plus_one,
        compute_weighted_sum4,
        opaque_add => residual_int,
        opaque_add4 => residual_int,
        opaque_sink_sum,
        opaque_sink_sum4,
        opaque_make_ref => residual_ref_wrapped,
        elidable_make_float => elidable_float_wrapped,
        opaque_ref_plus_int => residual_int_wrapped,
        elidable_float_plus_int => elidable_int_wrapped,
    },
)]
fn trace_call_step(program: &Program, pc: usize, state: &mut TestState) -> i64 {
    let op = program.get_op(pc);
    match op {
        OP_ELIDABLE_CALL => {
            let value = state.pop().unwrap();
            state.push(compute_square_plus_one(value));
        }
        OP_OPAQUE_CALL => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            state.push(opaque_add(lhs, rhs));
        }
        OP_OPAQUE_VOID_CALL => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            opaque_sink_sum(lhs, rhs);
        }
        OP_ELIDABLE_CALL4 => {
            let d = state.pop().unwrap();
            let c = state.pop().unwrap();
            let b = state.pop().unwrap();
            let a = state.pop().unwrap();
            state.push(compute_weighted_sum4(a, b, c, d));
        }
        OP_OPAQUE_CALL4 => {
            let d = state.pop().unwrap();
            let c = state.pop().unwrap();
            let b = state.pop().unwrap();
            let a = state.pop().unwrap();
            state.push(opaque_add4(a, b, c, d));
        }
        OP_OPAQUE_VOID_CALL4 => {
            let d = state.pop().unwrap();
            let c = state.pop().unwrap();
            let b = state.pop().unwrap();
            let a = state.pop().unwrap();
            opaque_sink_sum4(a, b, c, d);
        }
        OP_OPAQUE_REF_CALL => {
            let value = state.pop().unwrap();
            state.push_ref(opaque_make_ref(value));
        }
        OP_ELIDABLE_FLOAT_CALL => {
            let value = state.pop().unwrap();
            state.push_float(elidable_make_float(value));
        }
        OP_OPAQUE_REF_INT_CALL => {
            let delta = state.pop().unwrap();
            let value = state.pop().unwrap();
            let reference = opaque_make_ref(value);
            state.push(opaque_ref_plus_int(reference, delta));
        }
        OP_ELIDABLE_FLOAT_INT_CALL => {
            let delta = state.pop().unwrap();
            let value = state.pop().unwrap();
            let float_value = elidable_make_float(value);
            state.push(elidable_float_plus_int(float_value, delta));
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
fn jit_interp_elidable_call_lowers_to_call_pure_i() {
    let program = Program {
        ops: vec![OP_ELIDABLE_CALL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[5]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallPureI);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
}

#[test]
fn jit_interp_opaque_call_lowers_to_call_i() {
    let program = Program {
        ops: vec![OP_OPAQUE_CALL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[3, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallI);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
}

#[test]
fn jit_interp_high_arity_elidable_call_lowers_to_call_pure_i() {
    let program = Program {
        ops: vec![OP_ELIDABLE_CALL4],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[1, 2, 3, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallPureI);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
}

#[test]
fn jit_interp_high_arity_opaque_call_lowers_to_call_i() {
    let program = Program {
        ops: vec![OP_OPAQUE_CALL4],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[1, 2, 3, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallI);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
}

#[test]
fn annotated_call_helpers_still_run_interpreter_path() {
    let _guard = CALL_HELPER_TEST_LOCK.lock().unwrap();
    LAST_VOID_SUM.store(-1, Ordering::SeqCst);
    LAST_VOID_SUM4.store(-1, Ordering::SeqCst);
    let program = Program {
        ops: vec![OP_OPAQUE_CALL, OP_ELIDABLE_CALL],
    };
    let mut state = TestState {
        storage: StoragePool::new(&[&[2, 3]]),
        selected: 0,
    };

    assert_eq!(trace_call_step(&program, 0, &mut state), 5);
    assert_eq!(
        trace_call_step(
            &Program {
                ops: vec![OP_ELIDABLE_CALL]
            },
            0,
            &mut state
        ),
        26
    );
    assert_eq!(
        trace_call_step(
            &Program {
                ops: vec![OP_OPAQUE_VOID_CALL]
            },
            0,
            &mut TestState {
                storage: StoragePool::new(&[&[7, 2, 3]]),
                selected: 0,
            }
        ),
        7
    );
    assert_eq!(LAST_VOID_SUM.load(Ordering::SeqCst), 5);
    assert_eq!(
        trace_call_step(
            &Program {
                ops: vec![OP_ELIDABLE_CALL4]
            },
            0,
            &mut TestState {
                storage: StoragePool::new(&[&[1, 2, 3, 4]]),
                selected: 0,
            }
        ),
        4321
    );
    assert_eq!(
        trace_call_step(
            &Program {
                ops: vec![OP_OPAQUE_CALL4]
            },
            0,
            &mut TestState {
                storage: StoragePool::new(&[&[1, 2, 3, 4]]),
                selected: 0,
            }
        ),
        10
    );
    assert_eq!(
        trace_call_step(
            &Program {
                ops: vec![OP_OPAQUE_VOID_CALL4]
            },
            0,
            &mut TestState {
                storage: StoragePool::new(&[&[11, 7, 2, 3, 5]]),
                selected: 0,
            }
        ),
        11
    );
    assert_eq!(LAST_VOID_SUM4.load(Ordering::SeqCst), 17);
}

#[test]
fn jit_interp_opaque_void_call_lowers_to_call_n_and_keeps_stack_result() {
    let _guard = CALL_HELPER_TEST_LOCK.lock().unwrap();
    LAST_VOID_SUM.store(-1, Ordering::SeqCst);

    let program = Program {
        ops: vec![OP_OPAQUE_VOID_CALL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[7, 2, 3]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));
    assert_eq!(LAST_VOID_SUM.load(Ordering::SeqCst), 5);

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallN);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
}

#[test]
fn jit_interp_high_arity_opaque_void_call_lowers_to_call_n_and_keeps_stack_result() {
    let _guard = CALL_HELPER_TEST_LOCK.lock().unwrap();
    LAST_VOID_SUM4.store(-1, Ordering::SeqCst);

    let program = Program {
        ops: vec![OP_OPAQUE_VOID_CALL4],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[11, 7, 2, 3, 5]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));
    assert_eq!(LAST_VOID_SUM4.load(Ordering::SeqCst), 17);

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallN);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
}

#[test]
fn jit_interp_wrapped_ref_call_lowers_to_call_r() {
    let program = Program {
        ops: vec![OP_OPAQUE_REF_CALL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[23]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallR);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);

    let mut interp_state = TestState {
        storage: StoragePool::new(&[&[23]]),
        selected: 0,
    };
    assert_eq!(trace_call_step(&program, 0, &mut interp_state), 1023);
}

#[test]
fn jit_interp_wrapped_float_call_lowers_to_call_pure_f() {
    let program = Program {
        ops: vec![OP_ELIDABLE_FLOAT_CALL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[7]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallPureF);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);

    let mut interp_state = TestState {
        storage: StoragePool::new(&[&[7]]),
        selected: 0,
    };
    assert_eq!(
        f64::from_bits(trace_call_step(&program, 0, &mut interp_state) as u64),
        7.5
    );
}

#[test]
fn jit_interp_wrapped_ref_arg_call_records_ref_and_int_arg_types() {
    let program = Program {
        ops: vec![OP_OPAQUE_REF_INT_CALL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[23, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 3);
    assert_eq!(trace.ops[0].opcode, OpCode::CallR);
    assert_eq!(trace.ops[1].opcode, OpCode::CallI);
    let call_descr = trace.ops[1]
        .descr
        .as_ref()
        .and_then(|descr| descr.as_call_descr())
        .expect("call op should carry CallDescr");
    assert_eq!(call_descr.arg_types(), &[Type::Ref, Type::Int]);
    assert_eq!(trace.ops[2].opcode, OpCode::Finish);

    let mut interp_state = TestState {
        storage: StoragePool::new(&[&[23, 4]]),
        selected: 0,
    };
    assert_eq!(trace_call_step(&program, 0, &mut interp_state), 1027);
}

#[test]
fn jit_interp_wrapped_float_arg_call_records_float_and_int_arg_types() {
    let program = Program {
        ops: vec![OP_ELIDABLE_FLOAT_INT_CALL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[7, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 3);
    assert_eq!(trace.ops[0].opcode, OpCode::CallPureF);
    assert_eq!(trace.ops[1].opcode, OpCode::CallPureI);
    let call_descr = trace.ops[1]
        .descr
        .as_ref()
        .and_then(|descr| descr.as_call_descr())
        .expect("call op should carry CallDescr");
    assert_eq!(call_descr.arg_types(), &[Type::Float, Type::Int]);
    assert_eq!(trace.ops[2].opcode, OpCode::Finish);

    let mut interp_state = TestState {
        storage: StoragePool::new(&[&[7, 4]]),
        selected: 0,
    };
    assert_eq!(trace_call_step(&program, 0, &mut interp_state), 11);
}
