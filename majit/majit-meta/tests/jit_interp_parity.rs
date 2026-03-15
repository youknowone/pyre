use majit_macros::jit_interp;
use majit_meta::{
    assert_trace_parity, BackEdgeAction, JitState, MetaInterp, TraceAction, TraceParityCase,
};

const UNTRACEABLE: usize = 99;
const OP_INLINE_ADD: u8 = 1;
const OP_INLINE_COMPLEX: u8 = 2;
const OP_NOP1: u8 = 3;
const OP_NOP2: u8 = 4;
const OP_NOP3: u8 = 5;
const OP_SHIFT_CMP: u8 = 6;
const OP_EQ_BOOL: u8 = 7;
const OP_NONZERO_BOOL: u8 = 8;
const OP_IF_NONZERO_PUSH: u8 = 9;
const OP_IF_EXPR_VALUE: u8 = 10;

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
)]
fn trace_step(program: &Program, pc: usize, state: &mut TestState) -> i64 {
    let op = program.get_op(pc);
    match op {
        OP_INLINE_ADD => {
            let a = state.pop().unwrap();
            let b = state.pop().unwrap();
            let c = a + b;
            state.push(c);
        }
        OP_INLINE_COMPLEX => {
            let a = state.pop().unwrap();
            let b = state.pop().unwrap();
            let c = (a + 1) * (b - 2);
            state.push(c);
        }
        OP_SHIFT_CMP => {
            let a = state.pop().unwrap();
            let b = state.pop().unwrap();
            let c = (((a << 1) | (b >> 2)) >= 10) as i64;
            state.push(c);
        }
        OP_EQ_BOOL => {
            let a = state.pop().unwrap();
            let b = state.pop().unwrap();
            state.push(if b == a { 1 } else { 0 });
        }
        OP_NONZERO_BOOL => {
            let cond = state.pop().unwrap();
            state.push(if cond != 0 { 1 } else { 0 });
        }
        OP_IF_NONZERO_PUSH => {
            let a = state.pop().unwrap();
            if a != 0 {
                state.push(a + 10);
            } else {
                state.push(42);
            }
        }
        OP_IF_EXPR_VALUE => {
            let a = state.pop().unwrap();
            let out = if a != 0 { a + 1 } else { a - 1 };
            state.push(out);
        }
        OP_NOP1 => {}
        OP_NOP2 => {}
        OP_NOP3 => {}
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
fn jit_interp_inline_add_matches_runtime_parity_seam() {
    let program = Program {
        ops: vec![OP_INLINE_ADD],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[3, 4]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_inline_add",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: opimpl_int_add",
        expected_lines: &["v2 = IntAdd(v1, v0)", "Finish(v2)"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_inline_complex_expr_matches_runtime_parity_seam() {
    let program = Program {
        ops: vec![OP_INLINE_COMPLEX],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[4, 7]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_inline_complex_expr",
        rpython_reference:
            "rpython/jit/metainterp/pyjitpl.py: opimpl_int_add / opimpl_int_sub / opimpl_int_mul",
        expected_lines: &[
            "v2 = IntAdd(v1, 1)",
            "v3 = IntSub(v0, 2)",
            "v4 = IntMul(v2, v3)",
            "Finish(v4)",
        ],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_shift_and_compare_matches_runtime_parity_seam() {
    let program = Program {
        ops: vec![OP_SHIFT_CMP],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[8, 3]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_shift_and_compare",
        rpython_reference:
            "rpython/jit/metainterp/pyjitpl.py: opimpl_int_lshift / opimpl_int_rshift / opimpl_int_ge",
        expected_lines: &[
            "v2 = IntLshift(v1, 1)",
            "v3 = IntRshift(v0, 2)",
            "v4 = IntOr(v2, v3)",
            "v5 = IntGe(v4, 10)",
            "Finish(v5)",
        ],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_eq_booleanization_matches_runtime_parity_seam() {
    let program = Program {
        ops: vec![OP_EQ_BOOL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[5, 5]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_eq_booleanization",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: opimpl_int_eq",
        expected_lines: &["v2 = IntEq(v0, v1)", "Finish(v2)"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_nonzero_booleanization_matches_runtime_parity_seam() {
    let program = Program {
        ops: vec![OP_NONZERO_BOOL],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[9]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_nonzero_booleanization",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: opimpl_int_ne",
        expected_lines: &["v1 = IntNe(v0, 0)", "Finish(v1)"],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_internal_if_taken_path_records_guard_true() {
    let program = Program {
        ops: vec![OP_IF_NONZERO_PUSH],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[7]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_internal_if_taken",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: bytecode branch on traced bool",
        expected_lines: &[
            "v1 = IntNe(v0, 0)",
            "GuardTrue(v1) [fail_args=0, 0, 0]",
            "v3 = IntAdd(v0, 10)",
            "Finish(v3)",
        ],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_internal_if_fallthrough_records_guard_false() {
    let program = Program {
        ops: vec![OP_IF_NONZERO_PUSH],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[0]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_internal_if_fallthrough",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: bytecode branch on traced bool",
        expected_lines: &[
            "v1 = IntNe(v0, 0)",
            "GuardFalse(v1) [fail_args=0, 0, 0]",
            "Finish(42)",
        ],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_if_expr_taken_path_reuses_branch_value() {
    let program = Program {
        ops: vec![OP_IF_EXPR_VALUE],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[5]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_if_expr_taken",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: register-valued branch result",
        expected_lines: &[
            "v1 = IntNe(v0, 0)",
            "GuardTrue(v1) [fail_args=0, 0, 0]",
            "v3 = IntAdd(v0, 1)",
            "Finish(v3)",
        ],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn jit_interp_if_expr_fallthrough_reuses_branch_value() {
    let program = Program {
        ops: vec![OP_IF_EXPR_VALUE],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[0]]),
        selected: 0,
    };
    let (mut interp, mut sym) = start_trace(&program, &state);

    let action = {
        let ctx = interp.trace_ctx().unwrap();
        __trace_trace_step(ctx, &mut sym, &program, 0, &state.storage, state.selected)
    };
    assert!(matches!(action, TraceAction::Continue));

    let result = sym.stacks.get(&0).unwrap().peek().unwrap();
    let (trace, constants) = interp.finish_trace_for_parity(&[result]).unwrap();
    let case = TraceParityCase {
        name: "jit_interp_if_expr_fallthrough",
        rpython_reference: "rpython/jit/metainterp/pyjitpl.py: register-valued branch result",
        expected_lines: &[
            "v1 = IntNe(v0, 0)",
            "GuardFalse(v1) [fail_args=0, 0, 0]",
            "v3 = IntSub(v0, 1)",
            "Finish(v3)",
        ],
    };
    assert_trace_parity(&trace, &constants, &case);
}

#[test]
fn generated_jit_state_preserves_storage_layout_order() {
    let program = Program {
        ops: vec![OP_INLINE_ADD],
    };
    let state = TestState {
        storage: StoragePool::new(&[&[10, 20, 30]]),
        selected: 0,
    };
    let meta = state.build_meta(0, &program);
    let sym = TestState::create_sym(&meta, 0);
    assert_eq!(
        TestState::collect_jump_args(&sym),
        vec![majit_ir::OpRef(0), majit_ir::OpRef(1), majit_ir::OpRef(2)]
    );
    assert!(TestState::validate_close(&sym, &meta));
}

#[test]
fn annotated_function_still_runs_interpreter_path() {
    let program = Program {
        ops: vec![OP_INLINE_COMPLEX],
    };
    let mut state = TestState {
        storage: StoragePool::new(&[&[4, 7]]),
        selected: 0,
    };
    assert_eq!(trace_step(&program, 0, &mut state), (7 + 1) * (4 - 2));
}
