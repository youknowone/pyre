/// Tests that codegen produces valid wasm modules.
///
/// Most tests use wasmparser to validate emitted bytes. The terminal-decline
/// regression additionally executes the full wasm host and compares its Python
/// output with dynasm, because the old failure was a runtime pointer miscast.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use majit_backend_wasm::codegen;
use majit_ir::operand::Operand;
use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
use smallvec::smallvec;
use wasmi::{Engine, Linker, Memory, MemoryType, Module, Store};

fn validate_wasm(bytes: &[u8]) {
    wasmparser::validate(bytes).expect("generated wasm should be valid");
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("majit/majit-backend-wasm must be below the workspace root")
        .to_path_buf()
}

fn run_runtime_program(
    binary: &Path,
    script: &Path,
    envs: &[(&str, &str)],
) -> std::process::Output {
    let mut command = Command::new(binary);
    command.arg(script);
    for &(key, value) in envs {
        command.env(key, value);
    }
    command
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", binary.display()))
}

fn stat_value(stderr: &str, name: &str) -> u64 {
    stderr
        .split_whitespace()
        .find_map(|field| field.strip_prefix(&format!("{name}=")))
        .unwrap_or_else(|| panic!("missing {name}= in wasm JIT stats:\n{stderr}"))
        .parse()
        .unwrap_or_else(|err| panic!("invalid {name}= in wasm JIT stats: {err}\n{stderr}"))
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn global_reassign_retraces_non_last_label_backedge_at_runtime() {
    let root = workspace_root();
    let dynasm = root.join("target/release/pyre-dynasm");
    let wasm_runner = root.join("target/release/pyre-wasm-runner");
    let host_module = root.join("target/wasm32-unknown-unknown/release/pyre_wasm.wasm-host.wasm");
    let plain_module = root.join("target/wasm32-unknown-unknown/release/pyre_wasm.wasm");
    let wasm_module = if host_module.exists() {
        host_module
    } else {
        plain_module
    };

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime global-reassign regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    for bench in ["global_reassign.py", "global_reassign_obj.py"] {
        let script = root.join("pyre/bench/synth").join(bench);
        let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
        assert!(
            dynasm_run.status.success(),
            "dynasm failed for {bench}:\n{}",
            String::from_utf8_lossy(&dynasm_run.stderr)
        );
        let wasm_run = run_runtime_program(
            &wasm_runner,
            &script,
            &[
                ("PYRE_WASM_MODULE", module),
                ("PYRE_WASM_ENGINE", "wasmtime"),
                ("PYRE_WASM_JIT_STATS", "1"),
            ],
        );
        let stderr = String::from_utf8_lossy(&wasm_run.stderr);
        assert!(
            wasm_run.status.success(),
            "wasm failed for {bench}:\n{stderr}"
        );
        assert_eq!(
            wasm_run.stdout, dynasm_run.stdout,
            "wasm output diverged from dynasm for {bench}:\n{stderr}"
        );
        assert!(
            stat_value(&stderr, "compiles") > 1,
            "{bench} did not recompile after its global invalidation:\n{stderr}"
        );
        assert!(
            stat_value(&stderr, "gc_majors") < 10,
            "{bench} fell back to the allocating interpreter loop:\n{stderr}"
        );
    }
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn terminal_declined_call_assembler_matches_dynasm_at_runtime() {
    let root = workspace_root();
    let dynasm = root.join("target/release/pyre-dynasm");
    let wasm_runner = root.join("target/release/pyre-wasm-runner");
    let host_module = root.join("target/wasm32-unknown-unknown/release/pyre_wasm.wasm-host.wasm");
    let plain_module = root.join("target/wasm32-unknown-unknown/release/pyre_wasm.wasm");
    let wasm_module = if host_module.exists() {
        host_module
    } else {
        plain_module
    };
    let script = root.join("pyre/bench/ca_terminal_decline.py");

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime CA regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert!(
        dynasm_run.status.success(),
        "dynasm failed:\n{}",
        String::from_utf8_lossy(&dynasm_run.stderr)
    );
    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    let wasm_run = run_runtime_program(
        &wasm_runner,
        &script,
        &[
            ("PYRE_WASM_MODULE", module),
            ("PYRE_WASM_ENGINE", "wasmtime"),
            ("PYRE_WASM_JIT_STATS", "1"),
            ("PYRE_WASM_FORCE_CA_TERMINAL_DECLINE", "1"),
        ],
    );
    let stderr = String::from_utf8_lossy(&wasm_run.stderr);
    assert!(wasm_run.status.success(), "wasm failed:\n{stderr}");
    assert_eq!(
        wasm_run.stdout, dynasm_run.stdout,
        "forced terminal-decline wasm output diverged from dynasm\nwasm stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("accepted_ca=") && !stderr.contains("accepted_ca=0"),
        "fixture did not compile its outer CALL_ASSEMBLER trace:\n{stderr}"
    );
    assert!(
        stderr.contains("forced_ca_terminal_decline=1"),
        "terminal-decline hook did not run after CA admission:\n{stderr}"
    );
}

fn make_op(opcode: OpCode, args: &[OpRef], pos: OpRef) -> Op {
    let bx: Vec<Operand> = args.iter().map(|a| rb(*a)).collect();
    let op = Op::new(opcode, &bx);
    op.pos.set(pos);
    op
}

use majit_ir::forwarding::bound_operand_from_opref as rb;

fn make_guard(opcode: OpCode, args: &[OpRef], fail_args: &[OpRef]) -> Op {
    let bx: Vec<Operand> = args.iter().map(|a| rb(*a)).collect();
    let op = Op::new(opcode, &bx);
    op.setfailargs(smallvec![rb(fail_args[0]); 0]);
    let mut fa: smallvec::SmallVec<[Operand; 3]> = smallvec::SmallVec::new();
    for &a in fail_args {
        fa.push(rb(a));
    }
    op.setfailargs(fa);
    op
}

/// Calls `codegen::build_wasm_module` with the fixed test defaults (no
/// classptr map, no allocator/nursery, zero chaining slots, CA off) and
/// returns just the emitted bytes and guard exits — the only outputs the
/// tests assert on. `vtable_offset` and `gc_info` stay explicit because a
/// few tests vary them.
fn build_module(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    vtable_offset: Option<usize>,
    gc_info: &codegen::GuardGcTypeInfo,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    let (bytes, guards, _, _, _) = codegen::build_wasm_module(
        inputargs,
        ops,
        constants,
        vtable_offset,
        &HashMap::new(),
        gc_info,
        0,
        0,
        0,
        None, // nursery
        0,    // invalidated_flag_addr
        0,    // fail_index_base
        0,    // external_jump_slot
        0,    // external_jump_key
        codegen::FrameGeometry::fixed(),
        codegen::CaParams::default(),
    )
    .expect("wasm codegen should succeed");
    (bytes, guards)
}

/// `build_module` with the most common variant: entry vtable_offset `Some(0)`
/// and a default (disabled) `GuardGcTypeInfo`.
fn build_module_default(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    build_module(
        inputargs,
        ops,
        constants,
        Some(0),
        &codegen::GuardGcTypeInfo::default(),
    )
}

fn execute_ovf_trace_with_guard(
    opcode: OpCode,
    guard_opcode: OpCode,
    a: i64,
    b: i64,
) -> (i64, i64) {
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Int, 1),
    ];
    let guard = Op::new(guard_opcode, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]);
    finish.setfailargs(smallvec![rb(OpRef::int_op(2))]);
    let ops = vec![
        make_op(
            opcode,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        guard,
        finish,
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(1, None)).expect("test memory should allocate");
    memory
        .write(
            &mut store,
            codegen::FRAME_SLOT_BASE as usize,
            &a.to_le_bytes(),
        )
        .unwrap();
    memory
        .write(
            &mut store,
            (codegen::FRAME_SLOT_BASE + 8) as usize,
            &b.to_le_bytes(),
        )
        .unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("generated trace should instantiate");
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .expect("generated trace should execute");

    let mut fail_index = [0; 8];
    let mut result = [0; 8];
    memory.read(&store, 0, &mut fail_index).unwrap();
    memory
        .read(&store, codegen::FRAME_SLOT_BASE as usize, &mut result)
        .unwrap();
    (i64::from_le_bytes(fail_index), i64::from_le_bytes(result))
}

fn execute_ovf_trace(opcode: OpCode, a: i64, b: i64) -> (i64, i64) {
    execute_ovf_trace_with_guard(opcode, OpCode::GuardNoOverflow, a, b)
}

#[test]
fn test_int_add_ovf_guards_overflow() {
    for (a, b, expected) in [(10, 20, 30), (i64::MIN, 1, i64::MIN + 1)] {
        assert_eq!(execute_ovf_trace(OpCode::IntAddOvf, a, b), (1, expected));
    }
    assert_eq!(execute_ovf_trace(OpCode::IntAddOvf, i64::MAX, 1).0, 0);
}

#[test]
fn test_int_sub_ovf_guards_overflow() {
    for (a, b, expected) in [(100, 58, 42), (i64::MAX, 1, i64::MAX - 1)] {
        assert_eq!(execute_ovf_trace(OpCode::IntSubOvf, a, b), (1, expected));
    }
    assert_eq!(execute_ovf_trace(OpCode::IntSubOvf, i64::MIN, 1).0, 0);
}

#[test]
fn test_int_mul_ovf_guards_overflow() {
    for (a, b, expected) in [(6, 7, 42), (i64::MIN, 1, i64::MIN), (-9, -7, 63)] {
        assert_eq!(execute_ovf_trace(OpCode::IntMulOvf, a, b), (1, expected));
    }
    for (a, b) in [(i64::MIN, -1), (i64::MAX, 2), (1_i64 << 62, 3)] {
        assert_eq!(execute_ovf_trace(OpCode::IntMulOvf, a, b).0, 0);
    }
}

#[test]
fn test_guard_overflow_uses_pending_flag() {
    assert_eq!(
        execute_ovf_trace_with_guard(OpCode::IntAddOvf, OpCode::GuardOverflow, i64::MAX, 1,).0,
        1
    );
    assert_eq!(
        execute_ovf_trace_with_guard(OpCode::IntAddOvf, OpCode::GuardOverflow, 1, 2).0,
        0
    );
}

#[test]
fn test_empty_trace() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let ops = vec![{
        let op = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
        op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
        op
    }];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(guards[0].is_finish);
}

#[test]
fn test_int_add_loop() {
    // Label(i, sum) -> IntAdd(sum, i) -> IntAdd(i, 1) -> IntLt(i, 100)
    // -> GuardTrue -> Jump(new_i, new_sum)
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0), // i
        InputArg::from_type(Type::Int, 1), // sum
    ];

    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(1), OpRef::input_arg_int(0)],
            OpRef::int_op(2),
        ), // sum + i
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(3),
        ), // i + 1
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), const_100],
            OpRef::int_op(4),
        ), // i+1 < 100
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(3), OpRef::int_op(2)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3)), rb(OpRef::int_op(2))]),
    ];

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1); // one guard
    assert!(!guards[0].is_finish);
}

#[test]
fn test_float_ops() {
    let inputargs = vec![
        InputArg::from_type(Type::Float, 0),
        InputArg::from_type(Type::Float, 1),
    ];

    let ops = vec![
        make_op(
            OpCode::FloatAdd,
            &[OpRef::input_arg_float(0), OpRef::input_arg_float(1)],
            OpRef::float_op(2),
        ),
        make_op(
            OpCode::FloatSub,
            &[OpRef::input_arg_float(0), OpRef::input_arg_float(1)],
            OpRef::float_op(3),
        ),
        make_op(
            OpCode::FloatMul,
            &[OpRef::float_op(2), OpRef::float_op(3)],
            OpRef::float_op(4),
        ),
        make_op(
            OpCode::FloatTrueDiv,
            &[OpRef::float_op(4), OpRef::input_arg_float(0)],
            OpRef::float_op(5),
        ),
        make_op(OpCode::FloatNeg, &[OpRef::float_op(5)], OpRef::float_op(6)),
        make_op(OpCode::FloatAbs, &[OpRef::float_op(6)], OpRef::float_op(7)),
        make_op(
            OpCode::FloatLt,
            &[OpRef::input_arg_float(0), OpRef::input_arg_float(1)],
            OpRef::int_op(8),
        ),
        {
            let op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(7))]);
            op.setfailargs(smallvec![rb(OpRef::float_op(7))]);
            op
        },
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

#[test]
fn test_call_generates_import() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];

    let func_ptr = OpRef::const_int(42); // fake func_ptr
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        make_op(
            OpCode::CallI,
            &[func_ptr, OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        {
            let op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
            op.setfailargs(smallvec![rb(OpRef::int_op(1))]);
            op
        },
    ];

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);

    // Verify the module has jit_call import
    let parser = wasmparser::Parser::new(0);
    let mut has_jit_call = false;
    for payload in parser.parse_all(&bytes) {
        if let Ok(wasmparser::Payload::ImportSection(imports)) = payload {
            for import in imports {
                if let Ok(import) = import {
                    if import.name == "jit_call" {
                        has_jit_call = true;
                    }
                }
            }
        }
    }
    assert!(has_jit_call, "module should import jit_call");
}

#[test]
fn test_guard_types() {
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Int, 1),
    ];

    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
        // GuardTrue
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::input_arg_int(0)],
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        ),
        // GuardFalse
        make_guard(
            OpCode::GuardFalse,
            &[OpRef::input_arg_int(1)],
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        ),
        // GuardValue
        make_guard(
            OpCode::GuardValue,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            &[OpRef::input_arg_int(0)],
        ),
        // GuardNonnull
        make_guard(
            OpCode::GuardNonnull,
            &[OpRef::input_arg_int(0)],
            &[OpRef::input_arg_int(0)],
        ),
        // GuardIsnull
        make_guard(
            OpCode::GuardIsnull,
            &[OpRef::input_arg_int(1)],
            &[OpRef::input_arg_int(1)],
        ),
        // GuardNoOverflow (0 args)
        {
            let op = Op::new(OpCode::GuardNoOverflow, &[]);
            op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
            op
        },
        // GuardNotInvalidated (0 args; address 0 in this generic smoke test)
        {
            let op = Op::new(OpCode::GuardNotInvalidated, &[]);
            op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
            op
        },
        Op::new(
            OpCode::Jump,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 7);
}

/// GUARD_NOT_INVALIDATED must read the owning token's live AtomicBool rather
/// than being folded to an always-passing guard. The runtime regression is
/// covered by global_quasiimmut_invalidation.py; this pins the emitted load.
#[test]
fn test_guard_not_invalidated_loads_runtime_flag() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let guard = Op::new(OpCode::GuardNotInvalidated, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let ops = vec![
        guard,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let constants = indexmap::IndexMap::new();
    let (bytes, guards, _, _, _) = codegen::build_wasm_module(
        &inputargs,
        &ops,
        &constants,
        None,
        &HashMap::new(),
        &codegen::GuardGcTypeInfo::default(),
        0,
        0,
        0,
        None,
        0x1000,
        0,
        0,
        0,
        codegen::FrameGeometry::fixed(),
        codegen::CaParams::default(),
    )
    .expect("wasm codegen should succeed");

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2);
    let mut saw_flag_load = false;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                if matches!(
                    operators.read().unwrap(),
                    wasmparser::Operator::I32Load8U { .. }
                ) {
                    saw_flag_load = true;
                }
            }
        }
    }
    assert!(saw_flag_load, "GUARD_NOT_INVALIDATED omitted its flag load");
}

/// GuardNoException loads the global exception slot and fails when it is set;
/// GuardException compares the pending exception type against the expected one
/// and, on match, binds the caught value into its result var and clears both
/// slots. Validates the emitted bytecode is well-formed (stack-balanced).
#[test]
fn test_exception_guards() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];

    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        // GuardNoException — 0 args, fails when an exception is pending.
        {
            let op = Op::new(OpCode::GuardNoException, &[]);
            op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
            op
        },
        // GuardException(expected_type) — caught value bound to int_op(1).
        {
            let op = Op::new(OpCode::GuardException, &[rb(OpRef::input_arg_int(0))]);
            op.pos.set(OpRef::int_op(1));
            op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
            op
        },
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]),
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2);
}

/// GuardGcType contract in majit: arg0 = object ref, arg1 = expected
/// type_id. The wasm backend reads the GC header word at
/// `obj - GcHeader::SIZE` (matching the cranelift backend and
/// `majit_gc::header::GcHeader` layout) and compares the low
/// `TYPE_ID_BITS` against arg1. arg1 is an immediate type_id, NOT a
/// classptr — no `mem32[obj + 0]` read, no classptr→typeid lookup.
#[test]
fn test_guard_gc_type_uses_immediate_typeid() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];

    // Inline-Const carrying the immediate typeid 0x42
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        make_guard(
            OpCode::GuardGcType,
            &[OpRef::input_arg_int(0), OpRef::const_int(0x42)],
            &[OpRef::input_arg_int(0)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]),
    ];

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

/// Build a `GuardGcTypeInfo` matching what `WasmBackend::compile_loop`
/// derives from a real `GcLLDescr_framework`-equivalent allocator.
/// Mirrors `gc.py:592 get_translated_info_for_typeinfo` /
/// `gc.py:619 get_translated_info_for_guard_is_object` /
/// `x86/assembler.py:1951 cpu.subclassrange_min_offset`.
fn enabled_guard_gc_type_info() -> codegen::GuardGcTypeInfo {
    let mut info = codegen::GuardGcTypeInfo::default();
    info.supports_guard_gc_type = true;
    // Pretend the TYPE_INFO table sits at a small in-memory address;
    // wasm validation only checks the bytecode shape, not the actual
    // load addresses, so any value works for codegen testing.
    info.base_type_info = 0x1000;
    // majit `TypeEntry` stride = 32 bytes (TypeInfoLayout 16 + ClassTypeLayout 16).
    // shift_by = log2(32) = 5, sizeof_ti = rffi.sizeof(TYPE_INFO) = 16.
    info.shift_by = 5;
    info.sizeof_ti = 16; // size_of::<TypeInfoLayout>()
    // gc.py:603-622 _setup_guard_is_object: T_IS_RPYTHON_INSTANCE
    // = 0x100000 (gctypelayout.py:196), packed little-endian into a
    // Signed word — byte at offset +2 carries the flag, mask = 0x10.
    info.infobits_offset = 2;
    info.is_object_flag = 0x10;
    info.subclassrange_min_offset = 0; // offset within ClassTypeLayout
    info
}

/// x86/assembler.py:1924-1943 `genop_guard_guard_is_object` lowering —
/// the wasm backend's GUARD_IS_OBJECT arm emits the same MOV+addr_add
/// +TEST8+branch sequence. With `supports_guard_gc_type` enabled the
/// `assert` at line 1925 falls through and the rest of the lowering
/// runs; the resulting module must validate as legal wasm.
#[test]
fn test_guard_is_object_lowers_to_typeinfo_test() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];

    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        make_guard(
            OpCode::GuardIsObject,
            &[OpRef::input_arg_int(0)],
            &[OpRef::input_arg_int(0)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]),
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module(
        &inputargs,
        &ops,
        &constants,
        Some(0),
        &enabled_guard_gc_type_info(),
    );
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

/// x86/assembler.py:1945-1980 `genop_guard_guard_subclass` lowering —
/// the wasm backend's GUARD_SUBCLASS arm emits the gcremovetypeptr
/// branch (cpu.vtable_offset = None) when `vtable_offset` is `None`,
/// otherwise the vtable-load branch. With `supports_guard_gc_type`
/// enabled and the constant classptr's `(min, max)` pre-fetched, the
/// lowering runs to completion.
#[test]
fn test_guard_subclass_lowers_to_subclassrange_check() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];

    // model.py:199-201 `cls_of_box()` returns `ConstInt(ptr2int(typeptr))` —
    // the emitted guard-class operand is the vtable address carried as a raw
    // integer (read with `op.getarg(1).getint()`, rewrite.py:247). Use the
    // inline ConstInt factory so the variant tag matches the backend reader.
    let class_constant = OpRef::const_int(0xCAFE);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        make_guard(
            OpCode::GuardSubclass,
            &[OpRef::input_arg_int(0), class_constant],
            &[OpRef::input_arg_int(0)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]),
    ];

    let mut info = enabled_guard_gc_type_info();
    // assembler.py:1971-1974: codegen-time
    // (vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max).
    info.subclass_ranges.insert(0xCAFE, (10, 20));

    // gcremovetypeptr branch: vtable_offset = None.
    let (bytes, guards) = build_module(&inputargs, &ops, &constants, None, &info);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);

    // vtable-load branch: vtable_offset = Some(...).
    let (bytes2, _) = build_module(&inputargs, &ops, &constants, Some(8), &info);
    validate_wasm(&bytes2);
}

#[test]
fn test_sameas_and_conversions() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];

    let ops = vec![
        make_op(
            OpCode::SameAsI,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        make_op(
            OpCode::CastIntToFloat,
            &[OpRef::input_arg_int(0)],
            OpRef::float_op(2),
        ),
        make_op(
            OpCode::CastFloatToInt,
            &[OpRef::float_op(2)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::CastIntToPtr,
            &[OpRef::input_arg_int(0)],
            OpRef::ref_op(4),
        ),
        make_op(OpCode::CastPtrToInt, &[OpRef::ref_op(4)], OpRef::int_op(5)),
        make_op(OpCode::IntNeg, &[OpRef::input_arg_int(0)], OpRef::int_op(6)),
        make_op(
            OpCode::IntInvert,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(7),
        ),
        make_op(
            OpCode::IntIsTrue,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(8),
        ),
        make_op(
            OpCode::IntIsZero,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(9),
        ),
        {
            let op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(9))]);
            op.setfailargs(smallvec![rb(OpRef::int_op(9))]);
            op
        },
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
}

#[test]
fn test_overflow_ops() {
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Int, 1),
    ];

    let ops = vec![
        make_op(
            OpCode::IntAddOvf,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        {
            let op = Op::new(OpCode::GuardNoOverflow, &[]);
            op.setfailargs(smallvec![rb(OpRef::int_op(2))]);
            op
        },
        make_op(
            OpCode::IntSubOvf,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(3),
        ),
        {
            let op = Op::new(OpCode::GuardNoOverflow, &[]);
            op.setfailargs(smallvec![rb(OpRef::int_op(3))]);
            op
        },
        {
            let op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]);
            op.setfailargs(smallvec![rb(OpRef::int_op(2))]);
            op
        },
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 3); // 2 GuardNoOverflow + 1 Finish
}

#[test]
fn test_single_label_peeled_loop_validates() {
    // A single-label PEELED loop: a preamble op (the unrolled first iteration)
    // precedes the LABEL, so codegen wraps it in the resume-at-LABEL preamble-
    // skip dispatch (block $exit / $past_loader / $skip_preamble + br_if, with
    // the preamble at br-depth 2 and the body at 1). This validates the new
    // control-flow nesting and br depths via wasmparser.
    let inputargs = vec![InputArg::from_type(Type::Int, 0)]; // i
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        // preamble (unrolled first iteration): i + 1 -> v1
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(1),
        ),
        // loop header carrying v1 (single LABEL, with the preamble before it)
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        // body: v1 + 1 -> v2 ; v2 < 100 -> v3 ; guard ; jump v2 back to LABEL
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), const_1],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(2), const_100],
            OpRef::int_op(3),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(3)], &[OpRef::int_op(2)]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]),
    ];

    // Must be classified as single-label peeled (exercises the dispatch wrapper).
    assert!(codegen::is_single_label_peeled(&ops));

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(!guards[0].is_finish);
}

#[test]
fn test_multi_label_peeled_resumes_at_last_label_validates() {
    // A MULTI-label peeled loop: a preamble precedes an outer entry LABEL and
    // the inner loop-header LABEL. `is_single_label_peeled` is false (two
    // labels) but `is_resumable_peeled` is true, so codegen emits the SAME
    // resume-at-LABEL 3-block wrapper, resuming at the LAST label (where the
    // `loop` is). This proves the wrapper + br depths stay valid for a
    // multi-label source — the case `compile_bridge` newly accepts when a
    // loop-closing bridge targets that last label.
    let inputargs = vec![InputArg::from_type(Type::Int, 0)]; // i
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        // preamble (unrolled first iteration): i + 1 -> v1
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(1),
        ),
        // outer entry LABEL carrying v1 (no `loop` — a codegen no-op)
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        // inner loop-header LABEL carrying v1 (the LAST label — `loop` here)
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        // body: v1 + 1 -> v2 ; v2 < 100 -> v3 ; guard ; jump v2 back to LABEL
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), const_1],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(2), const_100],
            OpRef::int_op(3),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(3)], &[OpRef::int_op(2)]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]),
    ];

    // Multi-label: NOT the single-label subset, but still resumable-peeled, so
    // the wrapper is emitted and resumes at the last label.
    assert!(!codegen::is_single_label_peeled(&ops));
    assert!(codegen::is_resumable_peeled(&ops));

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(!guards[0].is_finish);
}

#[test]
fn test_non_last_label_backedge_validates() {
    // Quasi-immutable invalidation can re-trace a loop with a wide entry
    // LABEL followed by a narrower peeled header, while the closing JUMP
    // targets the earlier entry label.  The LABEL/JUMP descr identity, not
    // source position, is the loop target and determines the parallel move.
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let wide_descr = majit_ir::make_loop_target_descr(10, false);
    let narrow_descr = majit_ir::make_loop_target_descr(11, false);

    let wide_label = Op::new(
        OpCode::Label,
        &[rb(OpRef::int_op(1)), rb(OpRef::input_arg_int(0))],
    );
    wide_label.setdescr(wide_descr.clone());
    let narrow_label = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    narrow_label.setdescr(narrow_descr);
    let jump = Op::new(
        OpCode::Jump,
        &[rb(OpRef::int_op(3)), rb(OpRef::input_arg_int(0))],
    );
    jump.setdescr(wide_descr);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        wide_label,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        narrow_label,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        jump,
    ];

    // The key-dispatch wrapper intentionally remains restricted to a last-label
    // target, but ordinary local lowering must accept this shape.
    assert!(!codegen::is_resumable_peeled(&ops));
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
}

#[test]
fn test_registration_loop_stamps_label_block_id() {
    // The bridge-side target-ordinal recovery rests on one fact: a LABEL and the
    // closing JUMP that targets it share their loop-target descr by Arc identity,
    // so the `label_block_id` `compile_loop` stamps on the LABEL is readable from
    // the JUMP's descr in `compile_bridge`. Reproduce that here: build two LABELs
    // each with its own descr, run the registration loop's stamping (ordinals 0,
    // 1), and confirm the JUMP that shares the second LABEL's descr reads back 1.
    let descr0 = majit_ir::make_loop_target_descr(10, false);
    let descr1 = majit_ir::make_loop_target_descr(11, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label1.setdescr(descr1.clone());
    // A loop-closing bridge's terminal JUMP carries the SAME descr Arc as the
    // label it targets (here, the second/last label).
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]);
    jump.setdescr(descr1.clone());

    // Registration loop (mirrors compile_loop): stamp each LABEL with its ordinal.
    for (ordinal, label) in [&label0, &label1].iter().enumerate() {
        let d = label.getdescr().expect("label has a descr");
        d.as_loop_target_descr()
            .expect("loop-target descr")
            .set_label_block_id(ordinal as u32);
    }

    // Recover the JUMP's target ordinal — it must equal the LAST label's (1),
    // via the shared Arc, NOT 0 (the default) or the first label's ordinal.
    let recovered = jump
        .getdescr()
        .and_then(|d| d.as_loop_target_descr().map(|t| t.label_block_id()));
    assert_eq!(recovered, Some(1));
}
