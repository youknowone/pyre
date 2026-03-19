use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

use majit_ir::{GcRef, OpCode, Type};
use majit_macros::{dont_look_inside, elidable, jit_interp};
use majit_meta::{BackEdgeAction, JitState, MetaInterp, TraceAction};

const UNTRACEABLE: usize = 99;
const OP_MAY_FORCE_CALL: u8 = 1;
const OP_RELEASE_GIL_VOID_CALL: u8 = 2;
const OP_LOOPINVARIANT_REF_CALL: u8 = 3;
const OP_RELEASE_GIL_REF_INT_CALL: u8 = 4;
const OP_LOOPINVARIANT_FLOAT_CALL: u8 = 5;

static LAST_VOID_SUM: AtomicI64 = AtomicI64::new(-1);
static CALL_FAMILY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[dont_look_inside]
extern "C" fn opaque_add(x: i64, y: i64) -> i64 {
    x + y
}

#[dont_look_inside]
extern "C" fn opaque_sink_sum(x: i64, y: i64) {
    LAST_VOID_SUM.store(x + y, Ordering::SeqCst);
}

#[dont_look_inside]
extern "C" fn opaque_make_ref(x: i64) -> GcRef {
    GcRef((x + 1000) as usize)
}

#[dont_look_inside]
extern "C" fn opaque_ref_plus_int(value: GcRef, delta: i64) -> i64 {
    value.as_usize() as i64 + delta
}

#[elidable]
extern "C" fn elidable_make_float(x: i64) -> f64 {
    (x as f64) + 0.5
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

struct SpecialState {
    storage: StoragePool,
    selected: usize,
}

impl SpecialState {
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
    state = SpecialState,
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
        opaque_add => may_force_int,
        opaque_sink_sum => release_gil_void,
        opaque_make_ref => loopinvariant_ref_wrapped,
        opaque_ref_plus_int => release_gil_int_wrapped,
        elidable_make_float => loopinvariant_float_wrapped,
    },
)]
fn trace_special_call_step(program: &Program, pc: usize, state: &mut SpecialState) -> i64 {
    let op = program.get_op(pc);
    match op {
        OP_MAY_FORCE_CALL => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            state.push(opaque_add(lhs, rhs));
        }
        OP_RELEASE_GIL_VOID_CALL => {
            let rhs = state.pop().unwrap();
            let lhs = state.pop().unwrap();
            opaque_sink_sum(lhs, rhs);
        }
        OP_LOOPINVARIANT_REF_CALL => {
            let value = state.pop().unwrap();
            state.push_ref(opaque_make_ref(value));
        }
        OP_RELEASE_GIL_REF_INT_CALL => {
            let delta = state.pop().unwrap();
            let value = state.pop().unwrap();
            let reference = opaque_make_ref(value);
            state.push(opaque_ref_plus_int(reference, delta));
        }
        OP_LOOPINVARIANT_FLOAT_CALL => {
            let value = state.pop().unwrap();
            state.push_float(elidable_make_float(value));
        }
        _ => {}
    }
    state.storage.get(state.selected).peek_at(0)
}

fn start_trace(
    program: &Program,
    state: &SpecialState,
) -> (
    MetaInterp<<SpecialState as JitState>::Meta>,
    <SpecialState as JitState>::Sym,
) {
    let meta = state.build_meta(0, program);
    let live = state.extract_live(&meta);
    let mut interp = MetaInterp::new(1);
    let result = interp.on_back_edge(0, &live);
    assert!(matches!(result, BackEdgeAction::StartedTracing));
    let sym = SpecialState::create_sym(&meta, 0);
    (interp, sym)
}

#[test]
fn jit_interp_may_force_call_lowers_to_call_may_force_i() {
    let program = Program {
        ops: vec![OP_MAY_FORCE_CALL],
    };
    let state = SpecialState {
        storage: StoragePool::new(&[&[3, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_special_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallMayForceI);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);

    let mut interp_state = SpecialState {
        storage: StoragePool::new(&[&[3, 4]]),
        selected: 0,
    };
    assert_eq!(trace_special_call_step(&program, 0, &mut interp_state), 7);
}

#[test]
fn jit_interp_release_gil_void_call_lowers_to_call_release_gil_n() {
    let _guard = CALL_FAMILY_TEST_LOCK.lock().unwrap();
    LAST_VOID_SUM.store(-1, Ordering::SeqCst);

    let program = Program {
        ops: vec![OP_RELEASE_GIL_VOID_CALL],
    };
    let state = SpecialState {
        storage: StoragePool::new(&[&[7, 2, 3]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_special_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));
    assert_eq!(LAST_VOID_SUM.load(Ordering::SeqCst), 5);

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallReleaseGilN);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);
}

#[test]
fn jit_interp_loopinvariant_ref_call_lowers_to_call_loopinvariant_r() {
    let program = Program {
        ops: vec![OP_LOOPINVARIANT_REF_CALL],
    };
    let state = SpecialState {
        storage: StoragePool::new(&[&[23]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_special_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallLoopinvariantR);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);

    let mut interp_state = SpecialState {
        storage: StoragePool::new(&[&[23]]),
        selected: 0,
    };
    assert_eq!(
        trace_special_call_step(&program, 0, &mut interp_state),
        1023
    );
}

#[test]
fn jit_interp_release_gil_wrapped_ref_arg_call_records_ref_and_int_arg_types() {
    let program = Program {
        ops: vec![OP_RELEASE_GIL_REF_INT_CALL],
    };
    let state = SpecialState {
        storage: StoragePool::new(&[&[23, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_special_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 3);
    assert_eq!(trace.ops[0].opcode, OpCode::CallLoopinvariantR);
    assert_eq!(trace.ops[1].opcode, OpCode::CallReleaseGilI);
    let call_descr = trace.ops[1]
        .descr
        .as_ref()
        .and_then(|descr| descr.as_call_descr())
        .expect("call op should carry CallDescr");
    assert_eq!(call_descr.arg_types(), &[Type::Ref, Type::Int]);
    assert_eq!(trace.ops[2].opcode, OpCode::Finish);

    let mut interp_state = SpecialState {
        storage: StoragePool::new(&[&[23, 4]]),
        selected: 0,
    };
    assert_eq!(
        trace_special_call_step(&program, 0, &mut interp_state),
        1027
    );
}

#[test]
fn jit_interp_loopinvariant_float_call_lowers_to_call_loopinvariant_f() {
    let program = Program {
        ops: vec![OP_LOOPINVARIANT_FLOAT_CALL],
    };
    let state = SpecialState {
        storage: StoragePool::new(&[&[7]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_special_call_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, _constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    assert_eq!(trace.ops.len(), 2);
    assert_eq!(trace.ops[0].opcode, OpCode::CallLoopinvariantF);
    assert_eq!(trace.ops[1].opcode, OpCode::Finish);

    let mut interp_state = SpecialState {
        storage: StoragePool::new(&[&[7]]),
        selected: 0,
    };
    assert_eq!(
        f64::from_bits(trace_special_call_step(&program, 0, &mut interp_state) as u64),
        7.5
    );
}
