use majit_ir::GreenKey;
use majit_macros::jit_interp;
use majit_meta::{TraceAction, TraceCtx};

const UNTRACEABLE: usize = 99;
const OP_BACKEDGE: u8 = 1;
const OP_STOP: u8 = 2;

#[derive(Clone, Default)]
struct Store {
    values: Vec<i64>,
}

impl Store {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn peek_at(&self, index: usize) -> i64 {
        self.values[index]
    }

    fn push(&mut self, value: i64) {
        self.values.push(value);
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
    targets: Vec<usize>,
}

impl Program {
    fn get_op(&self, pc: usize) -> u8 {
        self.ops[pc]
    }

    fn get_target(&self, pc: usize) -> usize {
        self.targets[pc]
    }
}

fn scan_used_storages(_program: &Program, _header_pc: usize, selected: usize) -> Vec<usize> {
    vec![selected]
}

#[derive(Default)]
struct FakeDriver {
    merge_calls: usize,
    back_edge_calls: usize,
    pre_run_calls: usize,
    run_compiled_once: bool,
    last_structured_green_key: Option<GreenKey>,
}

impl FakeDriver {
    fn merge_point<Sym, F>(&mut self, _trace_fn: F)
    where
        F: FnOnce(&mut TraceCtx, &mut Sym) -> TraceAction,
    {
        self.merge_calls += 1;
    }

    fn back_edge<State, Env, F>(
        &mut self,
        _target_pc: usize,
        _state: &mut State,
        _env: &Env,
        pre_run: F,
    ) -> bool
    where
        F: FnOnce(),
    {
        self.back_edge_calls += 1;
        self.pre_run_calls += 1;
        pre_run();
        if self.run_compiled_once {
            self.run_compiled_once = false;
            true
        } else {
            false
        }
    }

    fn back_edge_structured<State, Env, F>(
        &mut self,
        green_key: GreenKey,
        _target_pc: usize,
        _state: &mut State,
        _env: &Env,
        pre_run: F,
    ) -> bool
    where
        F: FnOnce(),
    {
        self.last_structured_green_key = Some(green_key);
        self.back_edge_calls += 1;
        self.pre_run_calls += 1;
        pre_run();
        if self.run_compiled_once {
            self.run_compiled_once = false;
            true
        } else {
            false
        }
    }
}

mod explicit_case {
    use super::*;

    struct TestState {
        storage: StoragePool,
        selected: usize,
    }

    impl TestState {
        fn push(&mut self, value: i64) {
            self.storage.get_mut(self.selected).push(value);
        }
    }

    #[jit_interp(
        state = TestState,
        env = Program,
        storage = {
            pool: vm_state.storage,
            pool_type: StoragePool,
            selector: vm_state.selected,
            untraceable: [UNTRACEABLE],
            scan: scan_used_storages,
        },
        binops = {},
        io_shims = {},
    )]
    fn marker_loop_explicit(
        env: &Program,
        start_ip: usize,
        vm_state: &mut TestState,
        jit_drv: &mut FakeDriver,
    ) -> (usize, i32) {
        let mut ip = start_ip;
        let mut sp_len = -1;
        while ip < env.ops.len() {
            majit_runtime::jit_merge_point!(jit_drv, env, ip);
            let op = env.get_op(ip);
            match op {
                OP_BACKEDGE => {
                    vm_state.push(77);
                    let target = env.get_target(ip);
                    majit_runtime::can_enter_jit!(
                        jit_drv,
                        target,
                        vm_state,
                        env,
                        || {},
                        ip,
                        sp_len
                    );
                    ip += 1;
                }
                OP_STOP => break,
                _ => break,
            }
        }
        (ip, sp_len)
    }

    pub(super) fn run(program: &Program, driver: &mut FakeDriver) -> (usize, i32, usize) {
        let mut state = TestState {
            storage: StoragePool::new(&[&[]]),
            selected: 0,
        };
        let (ip, sp_len) = marker_loop_explicit(program, 0, &mut state, driver);
        (ip, sp_len, state.storage.get(0).len())
    }
}

mod legacy_case {
    use super::*;

    struct TestState {
        storage: StoragePool,
        selected: usize,
    }

    impl TestState {
        fn push(&mut self, value: i64) {
            self.storage.get_mut(self.selected).push(value);
        }
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
    fn marker_loop_legacy(
        program: &Program,
        start_pc: usize,
        state: &mut TestState,
        driver: &mut FakeDriver,
    ) -> (usize, i32) {
        let mut pc = start_pc;
        let mut stacksize = -1;
        while pc < program.ops.len() {
            majit_runtime::jit_merge_point!();
            let op = program.get_op(pc);
            match op {
                OP_BACKEDGE => {
                    state.push(11);
                    let target = program.get_target(pc);
                    majit_runtime::can_enter_jit!(driver, target, state, program, || {});
                    pc += 1;
                }
                OP_STOP => break,
                _ => break,
            }
        }
        (pc, stacksize)
    }

    pub(super) fn run(program: &Program, driver: &mut FakeDriver) -> (usize, i32, usize) {
        let mut state = TestState {
            storage: StoragePool::new(&[&[]]),
            selected: 0,
        };
        let (pc, stacksize) = marker_loop_legacy(program, 0, &mut state, driver);
        (pc, stacksize, state.storage.get(0).len())
    }
}

mod structured_green_case {
    use super::*;

    struct TestState {
        storage: StoragePool,
        selected: usize,
    }

    impl TestState {
        fn push(&mut self, value: i64) {
            self.storage.get_mut(self.selected).push(value);
        }
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
        greens = [state.selected, 123_i64],
    )]
    fn marker_loop_structured(
        program: &Program,
        start_pc: usize,
        state: &mut TestState,
        driver: &mut FakeDriver,
    ) -> (usize, i32) {
        let mut pc = start_pc;
        let mut stacksize = -1;
        while pc < program.ops.len() {
            majit_runtime::jit_merge_point!();
            let op = program.get_op(pc);
            match op {
                OP_BACKEDGE => {
                    state.push(33);
                    let target = program.get_target(pc);
                    majit_runtime::can_enter_jit!(driver, target, state, program, || {});
                    pc += 1;
                }
                OP_STOP => break,
                _ => break,
            }
        }
        (pc, stacksize)
    }

    pub(super) fn run(program: &Program, driver: &mut FakeDriver) -> (usize, i32, usize) {
        let mut state = TestState {
            storage: StoragePool::new(&[&[], &[], &[], &[], &[], &[], &[], &[]]),
            selected: 7,
        };
        let (pc, stacksize) = marker_loop_structured(program, 0, &mut state, driver);
        (pc, stacksize, state.storage.get(7).len())
    }
}

mod marker_green_tuple_case {
    use super::*;

    struct TestState {
        storage: StoragePool,
        selected: usize,
    }

    impl TestState {
        fn push(&mut self, value: i64) {
            self.storage.get_mut(self.selected).push(value);
        }
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
    fn marker_loop_structured_from_marker(
        program: &Program,
        start_pc: usize,
        state: &mut TestState,
        driver: &mut FakeDriver,
    ) -> (usize, i32) {
        let mut pc = start_pc;
        let mut stacksize = -1;
        while pc < program.ops.len() {
            majit_runtime::jit_merge_point!(driver, program, pc; state.selected, 321_i64);
            let op = program.get_op(pc);
            match op {
                OP_BACKEDGE => {
                    state.push(44);
                    let target = program.get_target(pc);
                    majit_runtime::can_enter_jit!(
                        driver,
                        target,
                        state,
                        program,
                        || {},
                        pc,
                        stacksize;
                        state.selected,
                        321_i64
                    );
                    pc += 1;
                }
                OP_STOP => break,
                _ => break,
            }
        }
        (pc, stacksize)
    }

    pub(super) fn run(program: &Program, driver: &mut FakeDriver) -> (usize, i32, usize) {
        let mut state = TestState {
            storage: StoragePool::new(&[&[], &[], &[], &[]]),
            selected: 3,
        };
        let (pc, stacksize) = marker_loop_structured_from_marker(program, 0, &mut state, driver);
        (pc, stacksize, state.storage.get(3).len())
    }
}

fn sample_program() -> Program {
    Program {
        ops: vec![OP_BACKEDGE, OP_STOP],
        targets: vec![1, 1],
    }
}

#[test]
fn jit_interp_marker_rewrite_accepts_explicit_nonstandard_variables() {
    let program = sample_program();
    let mut driver = FakeDriver {
        run_compiled_once: true,
        ..FakeDriver::default()
    };

    let (ip, sp_len, stack_len) = explicit_case::run(&program, &mut driver);

    assert_eq!(ip, 1);
    assert_eq!(sp_len, 1);
    assert_eq!(stack_len, 1);
    assert_eq!(driver.merge_calls, 2);
    assert_eq!(driver.back_edge_calls, 1);
    assert_eq!(driver.pre_run_calls, 1);
}

#[test]
fn jit_interp_marker_rewrite_keeps_legacy_defaults() {
    let program = sample_program();
    let mut driver = FakeDriver {
        run_compiled_once: true,
        ..FakeDriver::default()
    };

    let (pc, stacksize, stack_len) = legacy_case::run(&program, &mut driver);

    assert_eq!(pc, 1);
    assert_eq!(stacksize, 1);
    assert_eq!(stack_len, 1);
    assert_eq!(driver.merge_calls, 2);
    assert_eq!(driver.back_edge_calls, 1);
    assert_eq!(driver.pre_run_calls, 1);
}

#[test]
fn jit_interp_marker_rewrite_passes_structured_green_keys() {
    let program = sample_program();
    let mut driver = FakeDriver {
        run_compiled_once: true,
        ..FakeDriver::default()
    };

    let (pc, stacksize, stack_len) = structured_green_case::run(&program, &mut driver);

    assert_eq!(pc, 1);
    assert_eq!(stacksize, 1);
    assert_eq!(stack_len, 1);
    assert_eq!(
        driver.last_structured_green_key,
        Some(GreenKey::new(vec![1, 7, 123]))
    );
}

#[test]
fn jit_interp_marker_rewrite_accepts_marker_local_green_tuple() {
    let program = sample_program();
    let mut driver = FakeDriver {
        run_compiled_once: true,
        ..FakeDriver::default()
    };

    let (pc, stacksize, stack_len) = marker_green_tuple_case::run(&program, &mut driver);

    assert_eq!(pc, 1);
    assert_eq!(stacksize, 1);
    assert_eq!(stack_len, 1);
    assert_eq!(
        driver.last_structured_green_key,
        Some(GreenKey::new(vec![1, 3, 321]))
    );
}
