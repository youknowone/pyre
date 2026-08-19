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
use majit_ir::{EffectInfo, InputArg, Op, OpCode, OpRef, PyreHelperKind, Type};
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

/// The wasm-host module these runtime tests measure.
///
/// Only the snapshot path, never the raw `pyre_wasm.wasm` cargo output: the
/// `web` and `wasm-host` features of `pyre-wasm` build to that one filename and
/// overwrite each other, so a tree that last built `web` leaves a module there
/// which loads and runs but is not the one whose counters these tests pin.
/// `check.py` copies the wasm-host build here for exactly that reason.
/// A release runtime binary, named the way the host names an executable.
fn runtime_binary(root: &Path, name: &str) -> PathBuf {
    root.join("target/release")
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

fn wasm_host_module(root: &Path) -> PathBuf {
    root.join("target/wasm32-unknown-unknown/release/pyre_wasm.wasm-host.wasm")
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
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", binary.display()));
    // A child killed by a signal leaves every assertion below nothing to show:
    // both runtimes report their own failures before exiting — the wasm runner
    // keeps the run result precisely so a guest trap still prints its JIT
    // stats — so a failed status next to an empty stderr means the process
    // never reached any of that. Name the signal and the run that took it
    // here, once, instead of letting each call site blame output the child
    // never had the chance to write.
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = output.status.signal() {
            let env = envs
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(" ");
            panic!(
                "{} died on signal {signal} running {}\nenv: {env}\nstderr:\n{}",
                binary.display(),
                script.display(),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }
    output
}

/// Assert a runtime subprocess exited 0, reporting enough to diagnose the
/// failure from a CI log alone.
///
/// `run_runtime_program` has already claimed a death by signal, so what
/// reaches here is a child that chose its own nonzero status. Which run it
/// was, what status it chose, and what it had written to stdout are all part
/// of that answer, and stderr alone carries none of them — a runner that
/// reports on stdout and exits 1 otherwise fails with an empty message.
#[track_caller]
fn assert_ran_ok(label: &str, output: &std::process::Output) {
    assert!(
        output.status.success(),
        "{label} failed: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Assert the two runtimes printed the same thing, up to the line ending
/// their hosts spell a newline with.
///
/// What is being compared is what the two backends computed. `pyre-dynasm`
/// runs on the host, where a text stream translates `\n` on the way out and
/// Windows spells the result `\r\n`; the wasm guest is its own platform and
/// spells it `\n`. Comparing the bytes would make every one of these tests
/// fail on a Windows host over the separator alone.
#[track_caller]
fn assert_same_stdout(label: &str, wasm: &std::process::Output, dynasm: &std::process::Output) {
    fn lines(stdout: &[u8]) -> String {
        String::from_utf8_lossy(stdout).replace("\r\n", "\n")
    }
    assert_eq!(
        lines(&wasm.stdout),
        lines(&dynasm.stdout),
        "{label} output diverged from dynasm:\n{}",
        String::from_utf8_lossy(&wasm.stderr),
    );
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
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime global-reassign regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    {
        let bench = "global_reassign.py";
        let script = root.join("pyre/bench/synth").join(bench);
        let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
        assert_ran_ok(&format!("dynasm {bench}"), &dynasm_run);
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
        assert_ran_ok(&format!("wasm {bench}"), &wasm_run);
        assert_same_stdout(&format!("wasm {bench}"), &wasm_run, &dynasm_run);
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
fn raise_catch_clear_root_does_not_cross_the_host_per_exception() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);
    let script = root.join("pyre/bench/raise_catch_loop.py");

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime raise/catch regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert_ran_ok("dynasm raise/catch", &dynasm_run);
    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
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
    assert_ran_ok("wasm raise/catch", &wasm_run);
    assert_same_stdout("wasm raise/catch", &wasm_run, &dynasm_run);
    assert!(
        stat_value(&stderr, "jit_calls") < 100_000,
        "caught-exception root clearing crossed the host trampoline per iteration:\n{stderr}"
    );
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn recursive_call_assembler_does_not_refill_zeroed_nursery_frames() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);
    let script = root.join("pyre/bench/fib_recursive.py");

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime recursive-CA regression needs {}; build the requested artifacts first",
            artifact.display()
        );
    }
    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert_ran_ok("dynasm recursive fib", &dynasm_run);
    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    let wasm_run = run_runtime_program(
        &wasm_runner,
        &script,
        &[
            ("PYRE_WASM_MODULE", module),
            ("PYRE_WASM_ENGINE", "wasmtime"),
            ("PYRE_WASM_JIT_STATS", "1"),
            ("PYRE_WASM_DUMP_ALL_TRACES", "1"),
            ("PYRE_WASM_NO_CACHE", "1"),
        ],
    );
    let stderr = String::from_utf8_lossy(&wasm_run.stderr);
    assert_ran_ok("wasm recursive fib", &wasm_run);
    assert_same_stdout("wasm recursive fib", &wasm_run, &dynasm_run);
    // `compiles` is the host's module-compile tally, one per loop and one per
    // bridge, and `BRIDGE_OK` counts the bridges the backend accepted — the
    // same event `bridges_compiled` counts, since both are bumped only on the
    // `Ok` side of `compile_bridge`. So both follow from the committed
    // `pyre/bench/fib_recursive.wasm.jitstats`: `loops_compiled=1` +
    // `bridges_compiled=8`. Re-record these two alongside that baseline.
    assert_eq!(stat_value(&stderr, "compiles"), 9);
    assert_eq!(stat_value(&stderr, "BRIDGE_OK"), 8);
    assert!(
        !stderr.contains("memory.fill"),
        "recursive CA still refills a nursery that is already zeroed:\n{stderr}"
    );
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn fannkuch_blackhole_helpers_do_not_reflect_through_the_host() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);
    let script = root.join("pyre/bench/fannkuch.py");
    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime fannkuch regression needs {}; build the requested artifacts first",
            artifact.display()
        );
    }

    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert_ran_ok("dynasm fannkuch", &dynasm_run);
    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
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
    assert_ran_ok("wasm fannkuch", &wasm_run);
    assert_same_stdout("wasm fannkuch", &wasm_run, &dynasm_run);
    assert_eq!(stat_value(&stderr, "compiles"), 28);
    assert!(
        stat_value(&stderr, "jit_calls") < 100,
        "uniform-i64 blackhole helpers still reflected through the host:\n{stderr}"
    );
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn terminal_declined_call_assembler_matches_dynasm_at_runtime() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);
    let script = root.join("pyre/bench/ca_terminal_decline.py");

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime CA regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert_ran_ok("dynasm terminal-decline", &dynasm_run);
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
    assert_ran_ok("wasm terminal-decline", &wasm_run);
    assert_same_stdout("forced terminal-decline wasm", &wasm_run, &dynasm_run);
    assert!(
        stderr.contains("accepted_ca=") && !stderr.contains("accepted_ca=0"),
        "fixture did not compile its outer CALL_ASSEMBLER trace:\n{stderr}"
    );
    assert!(
        stderr.contains("forced_ca_terminal_decline=1"),
        "terminal-decline hook did not run after CA admission:\n{stderr}"
    );
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn wasm_outlier_bridges_stay_compiled_at_runtime() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime outlier regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    // A region that is no longer declined reaches compiled steady state through
    // one of two counters: an out-of-line bridge of its own (`BRIDGE_OK`), or,
    // when it closes back onto its owner's loop, a merge into the owner
    // (`inline_ok`). Inlining is the default and takes `exception_oserror_fields`,
    // so only the pair is a stable statement of "not declined".
    for (bench, compiled_counters) in [
        (
            "exception_oserror_fields.py",
            &["BRIDGE_OK", "inline_ok"][..],
        ),
        ("generator_tree_recursion.py", &["accepted_ca"][..]),
    ] {
        let script = root.join("pyre/bench/synth").join(bench);
        let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
        assert_ran_ok(&format!("dynasm {bench}"), &dynasm_run);
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
        assert_ran_ok(&format!("wasm {bench}"), &wasm_run);
        assert_same_stdout(&format!("wasm {bench}"), &wasm_run, &dynasm_run);
        let compiled: u64 = compiled_counters
            .iter()
            .map(|counter| stat_value(&stderr, counter))
            .sum();
        assert!(
            compiled > 0,
            "{bench} did not compile its formerly-declined bridge \
             (none of {compiled_counters:?} moved):\n{stderr}"
        );
        assert_eq!(
            stat_value(&stderr, "ml_unsafe_label"),
            0,
            "{bench} declined a LABEL resume:\n{stderr}"
        );
        assert_eq!(
            stat_value(&stderr, "decl_callasm"),
            0,
            "{bench} declined a CALL_ASSEMBLER bridge:\n{stderr}"
        );
    }
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
    build_module_with_frame(
        inputargs,
        ops,
        constants,
        vtable_offset,
        gc_info,
        codegen::FrameGeometry::fixed(),
    )
}

fn build_module_with_frame(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    vtable_offset: Option<usize>,
    gc_info: &codegen::GuardGcTypeInfo,
    frame: codegen::FrameGeometry,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset,
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: gc_info.clone(),
        alloc: codegen::AllocHelpers::default(),
        wb_fn_ptr: 0,
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        fail_index_base: 0,
        bridge_cells_base: 0,
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        external_jump_slot: 0,
        external_jump_key: 0,
        frame,
        ca: codegen::CaParams::default(),
    };
    let (bytes, guards, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
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

/// Locals of the module's FIRST code entry.
///
/// A trace that emits a label-parameter entry puts the narrow shim first and
/// the real body second, and the shim declares no locals. Use this only for
/// shapes `has_label_param_entry` rejects, or it reports the shim's zero.
fn emitted_local_count(bytes: &[u8]) -> u32 {
    wasmparser::Parser::new(0)
        .parse_all(bytes)
        .find_map(|payload| match payload.unwrap() {
            wasmparser::Payload::CodeSectionEntry(body) => Some(
                body.get_locals_reader()
                    .unwrap()
                    .into_iter()
                    .map(|local| local.unwrap().0)
                    .sum(),
            ),
            _ => None,
        })
        .expect("generated module must contain its trace function")
}

/// `(type arities, defined-function type indices, exported name -> func index)`.
///
/// Each type is reduced to `(params, results)` because that is all the entry
/// shape assertions need, and it keeps them readable next to the wasm text.
fn module_shape(bytes: &[u8]) -> (Vec<(usize, usize)>, Vec<u32>, HashMap<String, u32>) {
    let mut types = Vec::new();
    let mut functions = Vec::new();
    let mut exports = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        match payload.unwrap() {
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader {
                    for ty in group.unwrap().into_types() {
                        let func = ty.unwrap_func();
                        types.push((func.params().len(), func.results().len()));
                    }
                }
            }
            wasmparser::Payload::FunctionSection(reader) => {
                for idx in reader {
                    functions.push(idx.unwrap());
                }
            }
            wasmparser::Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.unwrap();
                    if export.kind == wasmparser::ExternalKind::Func {
                        exports.insert(export.name.to_string(), export.index);
                    }
                }
            }
            _ => {}
        }
    }
    (types, functions, exports)
}

#[test]
fn sparse_value_ids_declare_only_addressable_value_locals() {
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Int, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(40),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(40), OpRef::const_int(1)],
            OpRef::int_op(900),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(900))]),
    ];

    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(
        emitted_local_count(&bytes),
        inputargs.len() as u32 + 3 + 6 + 1,
        "two input values and three sparse ids, plus fixed i64 and i32 locals"
    );
}

#[test]
fn peephole_folds_adjacent_local_set_get_to_local_tee() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let tee_local = 2;
    let ops = vec![
        make_op(
            OpCode::SameAsI,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];

    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);

    let mut has_tee = false;
    let mut has_unfolded_pair = false;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut reader = body.get_operators_reader().unwrap();
            let mut previous_local_set = None;
            while !reader.eof() {
                match reader.read().unwrap() {
                    wasmparser::Operator::LocalSet { local_index } => {
                        previous_local_set = Some(local_index);
                    }
                    wasmparser::Operator::LocalGet { local_index } => {
                        has_unfolded_pair |=
                            previous_local_set == Some(tee_local) && local_index == tee_local;
                        previous_local_set = None;
                    }
                    wasmparser::Operator::LocalTee { local_index } => {
                        has_tee |= local_index == tee_local;
                        previous_local_set = None;
                    }
                    _ => previous_local_set = None,
                }
            }
        }
    }

    assert!(has_tee);
    assert!(
        !has_unfolded_pair,
        "the local.set/local.get pair must be folded"
    );
}

#[test]
fn unbound_pool_float_operand_declares_an_f64_local() {
    let folded_float = OpRef::float_op(7);
    let ops = vec![Op::new(OpCode::Finish, &[rb(folded_float)])];
    let constants = indexmap::IndexMap::from([(folded_float.raw(), 3.5_f64.to_bits() as i64)]);

    let (bytes, _) = build_module_default(&[], &ops, &constants);
    validate_wasm(&bytes);
    let local_types = wasmparser::Parser::new(0)
        .parse_all(&bytes)
        .find_map(|payload| match payload.unwrap() {
            wasmparser::Payload::CodeSectionEntry(body) => Some(
                body.get_locals_reader()
                    .unwrap()
                    .into_iter()
                    .map(|local| local.unwrap().1)
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .expect("generated module must contain its trace function");
    assert!(
        local_types.contains(&wasmparser::ValType::F64),
        "the producer-less Float operand must declare an f64 local"
    );
}

/// Count the direct `wasm_jit_write_barrier` table calls by their unique table
/// target immediate.  The direct lowering places that `i32.const` immediately
/// before its `call_indirect`.
fn direct_write_barrier_call_count(bytes: &[u8], target: i32) -> usize {
    let mut count = 0;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            let mut target_on_stack = false;
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I32Const { value } if value == target => {
                        target_on_stack = true;
                    }
                    wasmparser::Operator::CallIndirect { .. } if target_on_stack => {
                        count += 1;
                        target_on_stack = false;
                    }
                    _ => target_on_stack = false,
                }
            }
        }
    }
    count
}

fn build_module_with_write_barrier_target(
    inputargs: &[InputArg],
    ops: &[Op],
    write_barrier_target: i64,
) -> Vec<u8> {
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb_fn_ptr: write_barrier_target,
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        fail_index_base: 0,
        bridge_cells_base: 0,
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        external_jump_slot: 0,
        external_jump_key: 0,
        // The allocated trace keeps both Ref inputs live across New; reserve
        // their homes in the shared helper geometry.
        frame: codegen::FrameGeometry::compact(4, 2, 0),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _) = codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    bytes
}

#[test]
fn write_barrier_elision_keeps_one_barrier_per_base() {
    use majit_ir::descr::{SimpleFieldDescr, SimpleSizeDescr};
    use std::sync::Arc;

    const WB_TARGET: i64 = 0x4a11;
    let pointer_field = Arc::new(SimpleFieldDescr::new(0, 0, 8, Type::Ref, false));
    let finish = Op::new(OpCode::Finish, &[]);

    let new_obj = make_op(OpCode::New, &[], OpRef::ref_op(2));
    new_obj.setdescr(Arc::new(SimpleSizeDescr::new(0, 16, 1)));
    let new_store_a = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::ref_op(2)), rb(OpRef::input_arg_ref(0))],
    );
    new_store_a.setdescr(pointer_field.clone());
    let new_store_b = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::ref_op(2)), rb(OpRef::input_arg_ref(1))],
    );
    new_store_b.setdescr(pointer_field.clone());
    let allocated = build_module_with_write_barrier_target(
        &[
            InputArg::from_type(Type::Ref, 0),
            InputArg::from_type(Type::Ref, 1),
        ],
        &[new_obj, new_store_a, new_store_b, finish.clone()],
        WB_TARGET,
    );
    validate_wasm(&allocated);
    assert_eq!(
        direct_write_barrier_call_count(&allocated, WB_TARGET as i32),
        1,
        "an allocation result is not seeded into the applied set — its generation \
         is a runtime choice — so the first store barriers and the second is elided"
    );

    let live_store_a = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(1))],
    );
    live_store_a.setdescr(pointer_field.clone());
    let live_store_b = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(2))],
    );
    live_store_b.setdescr(pointer_field);
    let repeated_livein = build_module_with_write_barrier_target(
        &[
            InputArg::from_type(Type::Ref, 0),
            InputArg::from_type(Type::Ref, 1),
            InputArg::from_type(Type::Ref, 2),
        ],
        &[live_store_a, live_store_b, finish],
        WB_TARGET,
    );
    validate_wasm(&repeated_livein);
    assert_eq!(
        direct_write_barrier_call_count(&repeated_livein, WB_TARGET as i32),
        1,
        "repeated stores into one live-in base must emit one write-barrier call"
    );
}

#[test]
fn write_barrier_elision_follows_same_as_r_base() {
    use majit_ir::descr::SimpleFieldDescr;
    use std::sync::Arc;

    const WB_TARGET: i64 = 0x4a11;
    let pointer_field = Arc::new(SimpleFieldDescr::new(0, 0, 8, Type::Ref, false));
    let store_before_alias = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(1))],
    );
    store_before_alias.setdescr(pointer_field.clone());
    let alias = make_op(
        OpCode::SameAsR,
        &[OpRef::input_arg_ref(0)],
        OpRef::ref_op(3),
    );
    let store_through_alias = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::ref_op(3)), rb(OpRef::input_arg_ref(2))],
    );
    store_through_alias.setdescr(pointer_field);

    let bytes = build_module_with_write_barrier_target(
        &[
            InputArg::from_type(Type::Ref, 0),
            InputArg::from_type(Type::Ref, 1),
            InputArg::from_type(Type::Ref, 2),
        ],
        &[
            store_before_alias,
            alias,
            store_through_alias,
            Op::new(OpCode::Finish, &[]),
        ],
        WB_TARGET,
    );
    validate_wasm(&bytes);
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_TARGET as i32),
        1,
        "stores through a SameAsR base share one applied write barrier"
    );
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

/// Same trace as [`execute_ovf_trace`], but the constant `c` is an operand
/// rather than a second input argument, so the overflow check takes the
/// folded-bound form instead of the sign-comparison one. `const_first` puts
/// it on the left, which only addition accepts.
fn execute_ovf_trace_const(opcode: OpCode, a: i64, c: i64, const_first: bool) -> (i64, i64) {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let guard = Op::new(OpCode::GuardNoOverflow, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.setfailargs(smallvec![rb(OpRef::int_op(1))]);
    let args = if const_first {
        [OpRef::const_int(c), OpRef::input_arg_int(0)]
    } else {
        [OpRef::input_arg_int(0), OpRef::const_int(c)]
    };
    let ops = vec![make_op(opcode, &args, OpRef::int_op(1)), guard, finish];
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

/// A constant operand takes the folded-bound overflow check, so it needs the
/// same verdicts as the general form at the extremes — including the two
/// bounds that sit closest to overflowing themselves, `a - i64::MIN` and
/// `a + i64::MIN`.
#[test]
fn test_ovf_against_a_constant_matches_the_general_form() {
    // (opcode, a, c, overflows)
    let cases = [
        (OpCode::IntAddOvf, 10, 20, false),
        (OpCode::IntAddOvf, 5, 0, false),
        (OpCode::IntAddOvf, i64::MAX, 1, true),
        (OpCode::IntAddOvf, i64::MAX - 1, 1, false),
        (OpCode::IntAddOvf, 10, -20, false),
        (OpCode::IntAddOvf, i64::MIN, -1, true),
        (OpCode::IntAddOvf, -1, i64::MIN, true),
        (OpCode::IntAddOvf, 0, i64::MIN, false),
        (OpCode::IntSubOvf, 100, 58, false),
        (OpCode::IntSubOvf, 5, 0, false),
        (OpCode::IntSubOvf, i64::MIN, 1, true),
        (OpCode::IntSubOvf, i64::MIN + 1, 1, false),
        (OpCode::IntSubOvf, 10, -5, false),
        (OpCode::IntSubOvf, i64::MAX, -1, true),
        (OpCode::IntSubOvf, 0, i64::MIN, true),
        (OpCode::IntSubOvf, -1, i64::MIN, false),
    ];
    for (opcode, a, c, overflows) in cases {
        let (fail_index, result) = execute_ovf_trace_const(opcode, a, c, false);
        if overflows {
            assert_eq!(fail_index, 0, "{opcode:?}: {a} op {c} should guard-exit");
        } else {
            let expected = match opcode {
                OpCode::IntAddOvf => a.wrapping_add(c),
                _ => a.wrapping_sub(c),
            };
            assert_eq!(
                (fail_index, result),
                (1, expected),
                "{opcode:?}: {a} op {c}"
            );
        }
    }

    // Addition is commutative, so the constant is also accepted on the left.
    assert_eq!(
        execute_ovf_trace_const(OpCode::IntAddOvf, 10, 20, true),
        (1, 30)
    );
    assert_eq!(
        execute_ovf_trace_const(OpCode::IntAddOvf, i64::MAX, 1, true).0,
        0
    );
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
    for (a, b, expected) in [
        (6, 7, 42),
        (-9, -7, 63),
        (i32::MIN as i64, i32::MIN as i64, 1_i64 << 62),
        (
            i32::MAX as i64,
            i32::MAX as i64,
            (i32::MAX as i64) * (i32::MAX as i64),
        ),
        (i64::MIN, 1, i64::MIN),
        (i32::MAX as i64 + 1, 2, 1_i64 << 32),
    ] {
        assert_eq!(execute_ovf_trace(OpCode::IntMulOvf, a, b), (1, expected));
    }
    for (a, b) in [
        (i64::MIN, -1),
        (i64::MAX, 2),
        (1_i64 << 62, 3),
        (i32::MAX as i64 + 1, 1_i64 << 32),
    ] {
        assert_eq!(execute_ovf_trace(OpCode::IntMulOvf, a, b).0, 0);
    }
}

#[test]
fn test_int_mul_ovf_emits_signed32_fast_path_and_full_width_fallback() {
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Int, 1),
    ];
    let guard = Op::new(OpCode::GuardNoOverflow, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let ops = vec![
        make_op(
            OpCode::IntMulOvf,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        guard,
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);

    let mut extend32_s = 0;
    let mut i64_mul = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I64Extend32S => extend32_s += 1,
                    wasmparser::Operator::I64Mul => i64_mul += 1,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(extend32_s, 2, "both factors need an exact signed-32 check");
    assert!(
        i64_mul > 1,
        "the software full-width overflow fallback must remain in the module"
    );
}

/// Each guard spills its own fail arguments before it branches to the shared
/// bridge-dispatch epilogue.
#[test]
fn test_guard_fail_args_spill_in_their_own_failure_arms() {
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Int, 1),
    ];
    let first_guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::input_arg_int(0))]);
    first_guard.setfailargs(smallvec![
        rb(OpRef::input_arg_int(0)),
        rb(OpRef::input_arg_int(1)),
    ]);
    let second_guard = Op::new(OpCode::GuardFalse, &[rb(OpRef::input_arg_int(1))]);
    second_guard.setfailargs(smallvec![
        rb(OpRef::input_arg_int(1)),
        rb(OpRef::input_arg_int(0)),
    ]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
    let (bytes, guards) = build_module_default(
        &inputargs,
        &[first_guard, second_guard, finish],
        &indexmap::IndexMap::new(),
    );
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 3);

    let mut control_stack = Vec::new();
    let mut stores_per_guard_arm = Vec::new();
    let mut br_tables = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::If { .. } => control_stack.push(Some(0usize)),
                    wasmparser::Operator::Block { .. } | wasmparser::Operator::Loop { .. } => {
                        control_stack.push(None);
                    }
                    wasmparser::Operator::End => {
                        if let Some(Some(stores)) = control_stack.pop() {
                            stores_per_guard_arm.push(stores);
                        }
                    }
                    wasmparser::Operator::I64Store { .. } => {
                        if let Some(Some(stores)) =
                            control_stack.iter_mut().rev().find(|frame| frame.is_some())
                        {
                            *stores += 1;
                        }
                    }
                    wasmparser::Operator::BrTable { .. } => br_tables += 1,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(stores_per_guard_arm, [3, 3]);
    assert_eq!(
        br_tables, 0,
        "guard exits must not use a selector dispatcher"
    );
}

#[test]
fn test_fused_integer_guard_true_uses_inverse_comparison_directly() {
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Int, 1),
    ];
    let compare = make_op(
        OpCode::IntLt,
        &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        OpRef::int_op(2),
    );
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(2)],
        &[OpRef::input_arg_int(0)],
    );
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(1))]);
    let (bytes, _) = build_module_default(
        &inputargs,
        &[compare, guard, finish],
        &indexmap::IndexMap::new(),
    );
    validate_wasm(&bytes);

    let mut ge_s = 0;
    let mut i32_eqz = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I64GeS => ge_s += 1,
                    wasmparser::Operator::I32Eqz => i32_eqz += 1,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(ge_s, 1, "IntLt guard failure should be emitted as IntGe");
    assert_eq!(i32_eqz, 0, "integer inverse must not materialize i32.eqz");
}

#[test]
fn test_cold_guard_recovery_preserves_nonzero_base_and_typed_bits() {
    const FAIL_INDEX_BASE: u32 = 37;
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Ref, 1),
        InputArg::from_type(Type::Float, 2),
    ];
    let fail_args = smallvec![
        rb(OpRef::input_arg_ref(1)),
        rb(OpRef::input_arg_float(2)),
        rb(OpRef::input_arg_int(0)),
    ];
    let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::input_arg_int(0))]);
    guard.setfailargs(fail_args.clone());
    let finish = Op::new(OpCode::Finish, &fail_args);
    finish.setfailargs(fail_args);
    let ops = [guard, finish];
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb_fn_ptr: 0,
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        fail_index_base: FAIL_INDEX_BASE,
        bridge_cells_base: 0,
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        external_jump_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, guards, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    assert_eq!(guards[0].fail_index, FAIL_INDEX_BASE);

    let ref_bits = 0x1234_5678_i64;
    let float_bits = (-13.25_f64).to_bits() as i64;
    let mut store = Store::new(&Engine::default(), ());
    let engine = store.engine().clone();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let memory = Memory::new(&mut store, MemoryType::new(1, None)).unwrap();
    for (slot, bits) in [0_i64, ref_bits, float_bits].into_iter().enumerate() {
        memory
            .write(
                &mut store,
                codegen::FRAME_SLOT_BASE as usize + slot * 8,
                &bits.to_le_bytes(),
            )
            .unwrap();
    }
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .unwrap();

    let mut word = [0; 8];
    memory.read(&store, 0, &mut word).unwrap();
    assert_eq!(i64::from_le_bytes(word), FAIL_INDEX_BASE as i64);
    for (slot, expected) in [ref_bits, float_bits, 0].into_iter().enumerate() {
        memory
            .read(
                &store,
                codegen::FRAME_SLOT_BASE as usize + slot * 8,
                &mut word,
            )
            .unwrap();
        assert_eq!(i64::from_le_bytes(word), expected);
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

/// The `ModuleBuildInputs` shape every inline-region test shares: a fixed
/// frame, no nursery, no census, and every base at zero.  Only the owner trace
/// and the regions merged into it vary between them, so a new field lands here
/// once instead of at each call site.
fn inline_region_inputs(
    inputargs: &[InputArg],
    ops: Vec<Op>,
    inlined_bridges: Vec<codegen::InlinedBridge>,
) -> codegen::ModuleBuildInputs {
    codegen::ModuleBuildInputs {
        inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
        ops,
        inlined_bridges,
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb_fn_ptr: 0,
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        fail_index_base: 0,
        bridge_cells_base: 0,
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        external_jump_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    }
}

#[test]
fn inlined_bridge_without_owner_loop_label_declines() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::input_arg_int(0)],
        &[OpRef::input_arg_int(0)],
    );
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
    let inputs = inline_region_inputs(
        &inputargs,
        vec![guard, finish],
        vec![codegen::InlinedBridge {
            source_fail_index: 0,
            trace_id: 1,
            inputargs: vec![InputArg::from_type(Type::Int, 1)],
            ops: vec![Op::new(OpCode::Finish, &[])],
            gc_table_base: 0,
            constants: indexmap::IndexMap::new(),
        }],
    );

    let error = match codegen::build_wasm_module(&inputs) {
        Ok(_) => panic!("a label-less owner cannot accept an inlined bridge"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("no local loop LABEL"));
}

/// A CALL_ASSEMBLER descr carrying a callee token, which the stock
/// `SimpleCallDescr` cannot: its `call_target_token` answers `None`, and a
/// merge declines on the missing token before it ever consults `ca.targets`.
#[derive(Debug)]
struct TargetTokenCallDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: u64,
}

impl majit_ir::Descr for TargetTokenCallDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }

    fn as_call_descr(&self) -> Option<&dyn majit_ir::descr::CallDescr> {
        Some(self)
    }

    fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::LoopTokenDescr> {
        Some(self)
    }
}

impl majit_ir::LoopTokenDescr for TargetTokenCallDescr {
    fn loop_token_number(&self) -> u64 {
        self.target_token
    }
}

impl majit_ir::descr::CallDescr for TargetTokenCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }

    fn result_type(&self) -> Type {
        self.result_type
    }

    fn result_size(&self) -> usize {
        8
    }

    fn call_target_token(&self) -> Option<u64> {
        Some(self.target_token)
    }

    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(
            majit_ir::ExtraEffect::CanRaise,
            majit_ir::OopSpecIndex::None,
        );
        &INFO
    }
}

/// A region merged into an owner brings its own CALL_ASSEMBLER callee, but the
/// dedicated CA arm is selected by `ca.emit_ca` and bakes the callee geometry
/// out of `ca.targets` — both decided when the OWNER was compiled. An op that
/// misses that arm does not fail: it falls through to the ordinary
/// residual-call arm, which lowers arg 0 as an `__indirect_function_table`
/// slot, while a CALL_ASSEMBLER's arg 0 is the callee's first frame slot. That
/// calls whatever the slot happens to index and hands the answer back as the
/// callee's — a silent wrong result rather than a trap. Every trace's own ops
/// are screened for unsupported opcodes before compilation; the merged stream
/// is the one place that question is never re-asked, so the merge asks it.
///
/// Both ways of missing the arm are covered, and they are NOT the same test:
/// with no arm emitted at all the op never reaches `ca.targets`, while an owner
/// that does emit the arm still has no geometry for a callee it never compiled
/// against — and that second case would otherwise reach the CA arm's
/// `expect("CA op target must be registered")`.
#[test]
fn inlined_bridge_carrying_an_unarmed_call_assembler_declines() {
    fn build(
        region_ops: Vec<Op>,
        ca: codegen::CaParams,
    ) -> Result<Vec<u8>, majit_backend::BackendError> {
        let inputargs = vec![
            InputArg::from_type(Type::Int, 0),
            InputArg::from_type(Type::Int, 1),
        ];
        let owner_ops = vec![
            Op::new(
                OpCode::Label,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
            ),
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::const_int(1)],
                OpRef::int_op(2),
            ),
            make_guard(
                OpCode::GuardTrue,
                &[OpRef::int_op(2)],
                &[OpRef::int_op(2), OpRef::input_arg_int(1)],
            ),
            Op::new(
                OpCode::Jump,
                &[rb(OpRef::int_op(2)), rb(OpRef::input_arg_int(1))],
            ),
        ];
        let inputs = codegen::ModuleBuildInputs {
            inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
            ops: owner_ops,
            inlined_bridges: vec![codegen::InlinedBridge {
                source_fail_index: 0,
                trace_id: 7,
                inputargs: vec![
                    InputArg::from_type(Type::Int, 40),
                    InputArg::from_type(Type::Int, 41),
                ],
                ops: region_ops,
                gc_table_base: 0,
                constants: indexmap::IndexMap::new(),
            }],
            constants: indexmap::IndexMap::new(),
            vtable_offset: Some(0),
            classptr_to_typeid: HashMap::new(),
            guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
            alloc: codegen::AllocHelpers::default(),
            wb_fn_ptr: 0,
            nursery: None,
            invalidated_flag_addr: 0,
            gc_table_base: 0,
            fail_index_base: 0,
            bridge_cells_base: 0,
            bridge_entry_arity: None,
            bridge_param_dispatch: false,
            trace_entry_census: None,
            external_jump_slot: 0,
            external_jump_key: 0,
            frame: codegen::FrameGeometry::fixed(),
            ca,
        };
        codegen::build_wasm_module(&inputs).map(|(bytes, _, _)| bytes)
    }

    /// The region the decline arms use, with `opcode` producing its one value.
    /// The loop-closing JUMP carries the region's own inputs, so the result
    /// type never reaches the owner's label and only the opcode varies.
    ///
    /// `token` gives the op a callee the owner could in principle have an arm
    /// for; without it the merge declines on the missing token and never
    /// consults `ca.targets` at all.
    fn region_ops(opcode: OpCode, result: OpRef, token: Option<u64>) -> Vec<Op> {
        let call = make_op(
            opcode,
            &[OpRef::input_arg_int(40), OpRef::input_arg_int(41)],
            result,
        );
        if let Some(target_token) = token {
            call.setdescr(std::sync::Arc::new(TargetTokenCallDescr {
                arg_types: vec![Type::Int],
                result_type: opcode.result_type(),
                target_token,
            }));
        }
        vec![
            call,
            Op::new(
                OpCode::Jump,
                &[rb(OpRef::input_arg_int(40)), rb(OpRef::input_arg_int(41))],
            ),
        ]
    }

    // The same region under an ordinary opcode, to pin that what declines
    // below is the CALL_ASSEMBLER and not the fixture.
    let plain = build(
        region_ops(OpCode::IntAdd, OpRef::int_op(42), None),
        codegen::CaParams::default(),
    )
    .expect("a loop-closing region with no CALL_ASSEMBLER merges into its owner");
    validate_wasm(&plain);

    for (opcode, result) in [
        (OpCode::CallAssemblerI, OpRef::int_op(42)),
        (OpCode::CallAssemblerR, OpRef::ref_op(42)),
    ] {
        for (case, token, ca) in [
            // No CA arm at all, so no table to consult.
            ("no arm", Some(0x5a5a_u64), codegen::CaParams::default()),
            // The arm is emitted, but this callee is not one the owner
            // compiled against, so its geometry is absent from `ca.targets`.
            (
                "arm without this callee",
                Some(0x5a5a_u64),
                codegen::CaParams {
                    emit_ca: true,
                    ..codegen::CaParams::default()
                },
            ),
            // A CALL_ASSEMBLER that names no callee at all.
            ("no callee token", None, codegen::CaParams::default()),
        ] {
            let error = match build(region_ops(opcode, result, token), ca) {
                Ok(_) => {
                    panic!("{opcode:?} with {case} must decline, not build a module")
                }
                Err(error) => error,
            };
            assert!(
                error.to_string().contains("no CALL_ASSEMBLER arm for"),
                "{case}: declined for the wrong reason: {error}"
            );
        }
    }
}

/// A region's value ids are its own trace's, so they collide with the owner's.
/// The merged stream has one local namespace, so an unrebased collision makes
/// the region's entry moves land in locals the owner still holds live across
/// the back edge. Where a region's numbering happens to start must therefore
/// not be observable in the emitted code.
#[test]
fn inlined_bridge_emission_is_independent_of_the_regions_own_numbering() {
    fn owner_ops() -> Vec<Op> {
        vec![
            // Defined before the LABEL and read after it, so it is live across
            // the back edge and is restored only on preamble/resume entry.
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                OpRef::int_op(2),
            ),
            Op::new(
                OpCode::Label,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
            ),
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::const_int(1)],
                OpRef::int_op(3),
            ),
            make_op(
                OpCode::IntLt,
                &[OpRef::int_op(3), OpRef::const_int(10)],
                OpRef::int_op(4),
            ),
            make_guard(
                OpCode::GuardTrue,
                &[OpRef::int_op(4)],
                &[OpRef::int_op(3), OpRef::input_arg_int(1)],
            ),
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(1), OpRef::int_op(2)],
                OpRef::int_op(5),
            ),
            make_op(
                OpCode::IntLt,
                &[OpRef::int_op(5), OpRef::const_int(1000)],
                OpRef::int_op(6),
            ),
            make_guard(
                OpCode::GuardTrue,
                &[OpRef::int_op(6)],
                &[OpRef::int_op(3), OpRef::int_op(5)],
            ),
            Op::new(OpCode::Jump, &[rb(OpRef::int_op(3)), rb(OpRef::int_op(5))]),
        ]
    }

    // `base` picks where the region numbers its own values. `base = 2` makes
    // its first input arg share an id with the owner's loop-invariant
    // `int_op(2)`; `base = 40` clears every owner id.
    fn build(base: u32) -> Vec<u8> {
        let region_pool = indexmap::IndexMap::from([(base + 3, 0x5a5a_007)]);
        let first = build_with(base, region_pool.clone(), indexmap::IndexMap::new())
            .expect("a loop-closing region merges into its owner");
        // A region is RETAINED for the next re-emission, and `reemit_loop`
        // rebases the same retained copy every time, so the rebase must leave
        // it untouched. Building twice is what catches a rebase that wrote
        // through to the region it read.
        let second = build_with(base, region_pool, indexmap::IndexMap::new())
            .expect("a retained region rebases identically on re-emission");
        assert_eq!(first, second, "rebasing mutated the retained region");
        first
    }

    fn build_with(
        base: u32,
        region_constants: indexmap::IndexMap<u32, i64>,
        owner_constants: indexmap::IndexMap<u32, i64>,
    ) -> Result<Vec<u8>, majit_backend::BackendError> {
        let inputargs = vec![
            InputArg::from_type(Type::Int, 0),
            InputArg::from_type(Type::Int, 1),
        ];
        // `int_op(base + 3)` has no producing op: it is a folded value that
        // only the region's own constant pool binds, so the merge has to move
        // its pool key by the same offset it moves the read by.
        let region_ops = vec![
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(base), OpRef::input_arg_int(base + 1)],
                OpRef::int_op(base + 2),
            ),
            make_op(
                OpCode::IntAdd,
                &[OpRef::int_op(base + 2), OpRef::int_op(base + 3)],
                OpRef::int_op(base + 4),
            ),
            Op::new(
                OpCode::Jump,
                &[
                    rb(OpRef::int_op(base + 4)),
                    rb(OpRef::input_arg_int(base + 1)),
                ],
            ),
        ];
        let inputs = codegen::ModuleBuildInputs {
            inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
            ops: owner_ops(),
            inlined_bridges: vec![codegen::InlinedBridge {
                source_fail_index: 1,
                trace_id: 7,
                inputargs: vec![
                    InputArg::from_type(Type::Int, base),
                    InputArg::from_type(Type::Int, base + 1),
                ],
                ops: region_ops,
                gc_table_base: 0,
                constants: region_constants,
            }],
            constants: owner_constants,
            vtable_offset: Some(0),
            classptr_to_typeid: HashMap::new(),
            guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
            alloc: codegen::AllocHelpers::default(),
            wb_fn_ptr: 0,
            nursery: None,
            invalidated_flag_addr: 0,
            gc_table_base: 0,
            fail_index_base: 0,
            bridge_cells_base: 0,
            bridge_entry_arity: None,
            bridge_param_dispatch: false,
            trace_entry_census: None,
            external_jump_slot: 0,
            external_jump_key: 0,
            frame: codegen::FrameGeometry::fixed(),
            ca: codegen::CaParams::default(),
        };
        codegen::build_wasm_module(&inputs).map(|(bytes, _, _)| bytes)
    }

    let colliding = build(2);
    let disjoint = build(40);
    validate_wasm(&colliding);
    validate_wasm(&disjoint);
    assert_eq!(colliding, disjoint);

    // A key the owner pool carries inside the window the region is rebased
    // into names one of the REGION's ids once the merge is done, not the
    // owner value it was recorded for. Left in place it answers a read the
    // region's own pool declines, so the module builds on unrelated bits
    // instead of declining. Dropping the region's seed must therefore reach
    // the decline even though the owner pool has an entry at that position.
    let stale = build_with(
        2,
        indexmap::IndexMap::new(),
        indexmap::IndexMap::from([(12, 0x1234)]),
    );
    let error = match stale {
        Ok(_) => panic!("a stale owner key inside the region window answered the region's read"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("read with no producing op and no"),
        "unexpected decline: {error}"
    );
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

    // The residual-call trampoline reads its scratch at `base + offset`, and
    // the scratch is the module-static call area, so every emitting module
    // takes the two-argument import.
    let parser = wasmparser::Parser::new(0);
    let mut has_jit_call = false;
    for payload in parser.parse_all(&bytes) {
        if let Ok(wasmparser::Payload::ImportSection(imports)) = payload {
            for import in imports {
                if let Ok(import) = import
                    && import.name == "jit_call_compact"
                {
                    has_jit_call = true;
                }
            }
        }
    }
    assert!(has_jit_call, "module should import jit_call_compact");
}

fn void_call(arg_types: Vec<Type>, args: &[OpRef], result_size: usize) -> Op {
    void_call_with_helper(arg_types, args, result_size, PyreHelperKind::None)
}

fn void_call_with_helper(
    arg_types: Vec<Type>,
    args: &[OpRef],
    result_size: usize,
    helper: PyreHelperKind,
) -> Op {
    let mut operands = vec![rb(OpRef::const_int(42))];
    operands.extend(args.iter().copied().map(rb));
    let op = Op::new(OpCode::CallN, &operands);
    let effect = EffectInfo {
        pyre_helper: helper,
        ..EffectInfo::default()
    };
    op.setdescr(majit_ir::descr::make_call_descr_full(
        0,
        arg_types,
        Type::Void,
        false,
        result_size,
        effect,
    ));
    op
}

fn import_func_type(bytes: &[u8], name: &str) -> Option<u32> {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::ImportSection(imports) = payload.unwrap() {
            for import in imports {
                let import = import.unwrap();
                if import.name == name {
                    return match import.ty {
                        wasmparser::TypeRef::Func(type_idx) => Some(type_idx),
                        _ => None,
                    };
                }
            }
        }
    }
    None
}

fn has_table_import(bytes: &[u8]) -> bool {
    wasmparser::Parser::new(0).parse_all(bytes).any(|payload| {
        let Ok(wasmparser::Payload::ImportSection(imports)) = payload else {
            return false;
        };
        imports
            .into_iter()
            .any(|import| import.is_ok_and(|import| import.name == "__indirect_function_table"))
    })
}

fn indirect_call_types_and_drop_count(bytes: &[u8]) -> (Vec<(u32, u32)>, usize) {
    let mut indirect_calls = Vec::new();
    let mut drops = 0;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::CallIndirect {
                        type_index,
                        table_index,
                        ..
                    } => indirect_calls.push((type_index, table_index)),
                    wasmparser::Operator::Drop => drops += 1,
                    _ => {}
                }
            }
        }
    }
    (indirect_calls, drops)
}

fn function_type(
    bytes: &[u8],
    type_index: usize,
) -> (Vec<wasmparser::ValType>, Vec<wasmparser::ValType>) {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::TypeSection(types) = payload.unwrap() {
            let ty = types
                .into_iter_err_on_gc_types()
                .nth(type_index)
                .unwrap_or_else(|| panic!("missing function type {type_index}"))
                .unwrap();
            return (ty.params().to_vec(), ty.results().to_vec());
        }
    }
    panic!("module has no type section");
}

fn entry_function_type_index(bytes: &[u8]) -> usize {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::FunctionSection(functions) = payload.unwrap() {
            return functions
                .into_iter()
                .next()
                .expect("module has an entry function")
                .expect("entry function type is valid") as usize;
        }
    }
    panic!("module has no function section");
}

#[test]
fn zero_arity_parameter_entry_is_structurally_type_zero() {
    // A published LABEL target is entered through type 0, `(i32) -> i32`.
    // The separate type index emitted for a zero-arity parameter bridge must
    // retain that same structural signature.
    let inputargs = Vec::new();
    let ops = vec![Op::new(OpCode::Label, &[]), Op::new(OpCode::Finish, &[])];
    let inputs = codegen::ModuleBuildInputs {
        inputargs,
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb_fn_ptr: 0,
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        fail_index_base: 0,
        bridge_cells_base: 4,
        bridge_entry_arity: Some(0),
        bridge_param_dispatch: true,
        trace_entry_census: None,
        external_jump_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _) = codegen::build_wasm_module(&inputs).unwrap();

    validate_wasm(&bytes);
    assert_eq!(
        function_type(&bytes, entry_function_type_index(&bytes)),
        function_type(&bytes, 0),
        "a published LABEL target's module entry must be structurally `(i32) -> i32`"
    );
}

#[test]
fn test_nullary_true_void_call_uses_indirect_call_without_drop() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let ops = vec![
        void_call(vec![], &[], 0),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(has_table_import(&bytes));
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls, vec![(1, 0)]);
    assert_eq!(drops, 0, "a genuine () -> () helper needs no result drop");
}

#[test]
fn test_true_void_int_ref_call_uses_void_result_type_without_drop() {
    let inputargs = vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Ref, 1),
    ];
    let ops = vec![
        void_call(
            vec![Type::Int, Type::Ref],
            &[OpRef::input_arg_int(0), OpRef::input_arg_ref(1)],
            0,
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(has_table_import(&bytes));
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls, vec![(3, 0)]);
    assert_eq!(
        function_type(&bytes, 3),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![]
        )
    );
    assert_eq!(drops, 0, "a genuine void call has no result to drop");
}

#[test]
fn test_true_void_family_does_not_shift_new_call_type() {
    use majit_ir::descr::SimpleSizeDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type(Type::Ref, 0),
        InputArg::from_type(Type::Ref, 1),
        InputArg::from_type(Type::Int, 2),
    ];
    let new_op = make_op(OpCode::New, &[], OpRef::ref_op(3));
    new_op.setdescr(Arc::new(SimpleSizeDescr::new(0, 32, 53)));
    let ops = vec![
        void_call(
            vec![Type::Ref, Type::Ref],
            &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
            0,
        ),
        new_op,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(2))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls.len(), 4);
    assert_eq!(
        function_type(&bytes, indirect_calls[0].0 as usize),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![]
        )
    );
    assert_eq!(
        function_type(&bytes, indirect_calls[2].0 as usize),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        )
    );
    assert_eq!(drops, 0);
}

#[test]
fn test_list_append_word_abi_and_new_type_indices_match_declared_i64_types() {
    use majit_ir::descr::SimpleSizeDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type(Type::Ref, 0),
        InputArg::from_type(Type::Ref, 1),
        InputArg::from_type(Type::Int, 2),
    ];
    let new_op = make_op(OpCode::New, &[], OpRef::ref_op(3));
    new_op.setdescr(Arc::new(SimpleSizeDescr::new(0, 32, 53)));
    let ops = vec![
        void_call_with_helper(
            vec![Type::Ref, Type::Ref],
            &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
            8,
            PyreHelperKind::ListAppendValue,
        ),
        new_op,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(2))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls, vec![(3, 0), (1, 0), (3, 0), (1, 0)]);
    assert_eq!(
        function_type(&bytes, indirect_calls[0].0 as usize),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        ),
        "jit_list_append returns an ignored machine word"
    );
    assert_eq!(
        function_type(&bytes, indirect_calls[2].0 as usize),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        ),
        "New must reference its declared (i64, i64) -> i64 type"
    );
    assert_eq!(drops, 1, "only jit_list_append's result is discarded");
}

#[test]
fn test_true_void_float_arg_call_keeps_trampoline() {
    let inputargs = vec![
        InputArg::from_type(Type::Float, 0),
        InputArg::from_type(Type::Int, 1),
    ];
    let ops = vec![
        void_call(vec![Type::Float], &[OpRef::input_arg_float(0)], 0),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(1))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
    let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
    assert!(indirect_calls.is_empty());
}

#[test]
fn test_void_word_abi_call_uses_i64_result_type_and_drop() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let ops = vec![
        void_call(vec![Type::Int], &[OpRef::input_arg_int(0)], 8),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(has_table_import(&bytes));
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls, vec![(2, 0), (1, 0)]);
    assert_eq!(
        function_type(&bytes, 2),
        (
            vec![wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        )
    );
    assert_eq!(drops, 1, "the ignored word result must be dropped");
}

#[test]
fn test_true_void_type_index_accounts_for_trampoline_type() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::CallI,
            &[OpRef::const_int(43), OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        void_call(vec![], &[], 0),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls, vec![(2, 0)]);
    assert_eq!(drops, 0);
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
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset: None,
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb_fn_ptr: 0,
        nursery: None,
        invalidated_flag_addr: 0x1000,
        gc_table_base: 0,
        fail_index_base: 0,
        bridge_cells_base: 0,
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        external_jump_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, guards, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");

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
/// Mirrors `gc.py get_translated_info_for_typeinfo` /
/// `gc.py get_translated_info_for_guard_is_object` /
/// `x86/assembler.py cpu.subclassrange_min_offset`.
fn enabled_guard_gc_type_info() -> codegen::GuardGcTypeInfo {
    // Pretend the TYPE_INFO table sits at a small in-memory address;
    // wasm validation only checks the bytecode shape, not the actual
    // load addresses, so any value works for codegen testing.
    // majit `TypeEntry` stride = 32 bytes (TypeInfoLayout 16 + ClassTypeLayout 16).
    // shift_by = log2(32) = 5, sizeof_ti = rffi.sizeof(TYPE_INFO) = 16.
    // gc.py _setup_guard_is_object: T_IS_RPYTHON_INSTANCE
    // = 0x100000 (gctypelayout.py:196), packed little-endian into a
    // Signed word — byte at offset +2 carries the flag, mask = 0x10.
    codegen::GuardGcTypeInfo {
        supports_guard_gc_type: true,
        base_type_info: 0x1000,
        shift_by: 5,
        sizeof_ti: 16, // size_of::<TypeInfoLayout>()
        infobits_offset: 2,
        is_object_flag: 0x10,
        subclassrange_min_offset: 0, // offset within ClassTypeLayout
        ..Default::default()
    }
}

/// x86/assembler.py `genop_guard_guard_is_object` lowering —
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

/// x86/assembler.py `genop_guard_guard_subclass` lowering —
/// the wasm backend's GUARD_SUBCLASS arm emits the gcremovetypeptr
/// branch (cpu.vtable_offset = None) when `vtable_offset` is `None`,
/// otherwise the vtable-load branch. With `supports_guard_gc_type`
/// enabled and the constant classptr's `(min, max)` pre-fetched, the
/// lowering runs to completion.
#[test]
fn test_guard_subclass_lowers_to_subclassrange_check() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];

    // model.py `cls_of_box()` returns `ConstInt(ptr2int(typeptr))` —
    // the emitted guard-class operand is the vtable address carried as a raw
    // integer (read with `op.getarg(1).getint()`, rewrite.py). Use the
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

/// A `LoadFromGcTable` placed inside the loop body is emitted inside the loop.
///
/// `rewrite.py remove_constptr` caches one load per gc-table index,
/// but `rewrite.py emit_label` clears `gcrefs_recently_loaded` at
/// every LABEL, so a reference constant used after the LABEL is loaded again on
/// each iteration. The comment there rejects keeping the value alive across the
/// label ("don't spill it") as "the wrong level" — the backend emits the op
/// where the trace puts it and leaves that decision to the optimizer.
#[test]
fn gc_table_load_inside_a_loop_body_is_emitted_inside_the_loop() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        make_op(
            OpCode::LoadFromGcTable,
            &[OpRef::const_int(0)],
            OpRef::ref_op(2),
        ),
        make_guard(
            OpCode::GuardNonnull,
            &[OpRef::ref_op(2)],
            &[OpRef::int_op(1)],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]),
    ];
    let gc_table_base = 4096;
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb_fn_ptr: 0,
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base,
        fail_index_base: 0,
        bridge_cells_base: 0,
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        external_jump_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _) = codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);

    let mut control_stack = Vec::new();
    let mut gc_table_address_on_stack = false;
    let mut loads_inside_loop = 0usize;
    let mut loads_outside_loop = 0usize;
    let mut saw_loop = false;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::Loop { .. } => {
                        saw_loop = true;
                        control_stack.push(true);
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::Block { .. } | wasmparser::Operator::If { .. } => {
                        control_stack.push(false);
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::End => {
                        control_stack.pop();
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::I32Const { value } if value == gc_table_base as i32 => {
                        gc_table_address_on_stack = true;
                    }
                    wasmparser::Operator::I32Load { .. } if gc_table_address_on_stack => {
                        if control_stack.contains(&true) {
                            loads_inside_loop += 1;
                        } else {
                            loads_outside_loop += 1;
                        }
                        gc_table_address_on_stack = false;
                    }
                    _ => gc_table_address_on_stack = false,
                }
            }
        }
    }
    // Without this, the counts below also hold for a body that emitted no loop
    // at all.
    assert!(saw_loop, "codegen emitted no loop for a looping trace");
    assert_eq!(
        loads_inside_loop, 1,
        "the in-loop LoadFromGcTable must be emitted inside the loop"
    );
    assert_eq!(
        loads_outside_loop, 0,
        "no gc-table load belongs outside the loop for this trace"
    );
}

#[test]
fn test_peeled_label_captures_missing_ref_livein_in_frozen_frame() {
    let inputargs = vec![
        InputArg::from_type(Type::Ref, 0),
        InputArg::from_type(Type::Int, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        // The Ref input remains live in the body but is intentionally absent
        // from the semantic LABEL args: it must be restored from a GC-rooted
        // backend capture home on bridge re-entry.
        Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]),
        make_guard(
            OpCode::GuardNonnull,
            &[OpRef::input_arg_ref(0)],
            &[OpRef::input_arg_ref(0), OpRef::int_op(2)],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]),
    ];

    assert!(codegen::is_resumable_peeled(&ops));
    assert_eq!(codegen::label_ref_capture_slots(&inputargs, &ops), 1);
    let ordinary_homes = codegen::count_ref_homes(&inputargs, &ops);
    let frame = codegen::FrameGeometry::compact(
        codegen::frame_value_slots(&inputargs, &ops),
        ordinary_homes + 1,
        1,
    );
    assert_eq!(
        codegen::label_resume_info(&inputargs, &ops, frame),
        vec![(true, true)]
    );
    assert_eq!(frame.ordinary_home_slots(), ordinary_homes);
    let (bytes, guards) = build_module_with_frame(
        &inputargs,
        &ops,
        &indexmap::IndexMap::new(),
        Some(0),
        &codegen::GuardGcTypeInfo::default(),
        frame,
    );
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
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

/// A resumable-peeled loop within the fixed arity emits two functions: the
/// narrow `trace` shim the host and CALL_ASSEMBLER keep entering, and the
/// `trace_wide` body a loop-closing JUMP can pass its arguments to.
///
/// `trace` must stay structurally `(i32) -> i32`: a loop's table slot is also
/// what the CALL_ASSEMBLER path calls as type 0, so widening it in place would
/// turn every such call into a trap rather than a decline.
#[test]
fn peeled_loop_exports_a_narrow_shim_beside_its_wide_entry() {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(1),
        ),
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
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

    let frame = codegen::FrameGeometry::fixed();
    assert!(
        codegen::has_label_param_entry(&inputargs, &ops, frame, None),
        "this shape is the one the wide entry exists for; if the gate stops \
         accepting it the rest of these assertions prove nothing"
    );

    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);

    let (types, functions, exports) = module_shape(&bytes);
    let narrow = exports.get("trace").copied().expect("narrow entry export");
    let wide = exports
        .get("trace_wide")
        .copied()
        .expect("wide entry export");
    assert_eq!(wide, narrow + 1, "the wide entry follows the shim");

    // Exports index the whole function space, imports first; the type section
    // is indexed by defined function only.
    let imported = narrow;
    assert_eq!(
        types[functions[(narrow - imported) as usize] as usize],
        (1, 1),
        "trace must stay (i32) -> i32"
    );
    assert_eq!(
        types[functions[(wide - imported) as usize] as usize],
        (1 + majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY, 1),
        "trace_wide takes frame_ptr plus one word per fixed label parameter"
    );

    // The shim reads from `FRAME_SLOT_BASE`, so its loads land in slots
    // 1..=FROZEN_LABEL_PARAM_ARITY and a frame holding exactly that many slots
    // is one short — its last load would run off the end.
    let exact = codegen::FrameGeometry {
        value_slots: majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY,
        ..frame
    };
    assert!(
        !codegen::has_label_param_entry(&inputargs, &ops, exact, None),
        "a frame with FROZEN_LABEL_PARAM_ARITY slots must be rejected"
    );
    let one_more = codegen::FrameGeometry {
        value_slots: majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY + 1,
        ..frame
    };
    assert!(
        codegen::has_label_param_entry(&inputargs, &ops, one_more, None),
        "one slot past the arity is the smallest frame the shim can read"
    );
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

    // The wide entry label precedes the header, so it is a resume point; the
    // narrow one sits inside the `loop` and gets no block pair, which is what
    // keeps the wrapper's structured control flow well-formed here.
    assert!(codegen::is_resumable_peeled(&ops));
    assert_eq!(codegen::resumable_label_count(&ops), 1);
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

/// Emitted `loop` count and entry-`br_table` arm count, the two things that say
/// whether a module took the resume-`loop` shape.
fn loop_and_br_table_shape(bytes: &[u8]) -> (usize, Vec<usize>) {
    let mut loops = 0usize;
    let mut tables = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::Loop { .. } => loops += 1,
                    wasmparser::Operator::BrTable { targets } => {
                        tables.push(targets.len() as usize)
                    }
                    _ => {}
                }
            }
        }
    }
    (loops, tables)
}

/// A two-label peeled owner (`preamble; LABEL0; segment; LABEL1(header); body;
/// JUMP->LABEL1`) with one inlined region, whose closing JUMP names `which`.
fn build_owner_with_region_closing_at(
    which: usize,
) -> Result<Vec<u8>, majit_backend::BackendError> {
    let descr0 = majit_ir::make_loop_target_descr(10, false);
    let descr1 = majit_ir::make_loop_target_descr(11, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1.clone());

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(100)],
            OpRef::int_op(4),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(3)]),
        jump,
    ];
    // Both LABELs precede the header, so both are resume points.
    assert!(codegen::is_resumable_peeled(&ops));
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_jump.setdescr(if which == 0 { descr0 } else { descr1 });
    let region_ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(1)],
            OpRef::int_op(11),
        ),
        region_jump,
    ];

    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![codegen::InlinedBridge {
            source_fail_index: 0,
            trace_id: 1,
            inputargs: vec![InputArg::from_type(Type::Int, 10)],
            ops: region_ops,
            gc_table_base: 0,
            constants: indexmap::IndexMap::new(),
        }],
    );
    codegen::build_wasm_module(&inputs).map(|(bytes, _, _)| bytes)
}

/// A region closing at the loop HEADER `br`s to the `loop`, the long-standing
/// shape: one `loop`, and the entry `br_table` keeps its one arm per label plus
/// the fresh-entry bucket.
#[test]
fn region_closing_at_the_header_keeps_the_single_loop_shape() {
    let bytes = build_owner_with_region_closing_at(1).expect("header-closing region merges");
    validate_wasm(&bytes);
    let (loops, tables) = loop_and_br_table_shape(&bytes);
    assert_eq!(loops, 1, "only the loop header opens a `loop`");
    assert_eq!(
        tables,
        vec![3],
        "entry br_table: key 0 plus one arm per label"
    );
}

/// A region closing at a NON-header LABEL cannot `br` to the `loop` — that
/// would skip the segment between its label and the header. It re-enters the
/// entry dispatch instead, so the module grows a second `loop` around that
/// dispatch and a second `br_table` bucket per label (`num_labels + 1 + j`,
/// landing past label j's resume loader with the args already in locals).
#[test]
fn region_closing_at_a_non_header_label_wraps_the_dispatch_in_a_loop() {
    let bytes = build_owner_with_region_closing_at(0).expect("non-header region merges");
    validate_wasm(&bytes);
    let (loops, tables) = loop_and_br_table_shape(&bytes);
    assert_eq!(loops, 2, "the resume `loop` wraps the entry dispatch");
    assert_eq!(
        tables,
        vec![5],
        "entry br_table: key 0, one loader arm per label, one past-loader arm per label"
    );
}

/// Execute a two-label peeled loop whose inlined region closes at the NON-header
/// LABEL, and report the three values the exit guard spills. The region's
/// re-entry must land at label 0 and then run the segment between label 0 and
/// the header, exactly as an out-of-line bridge resuming at key 1 would.
///
///   preamble  v1 = v0 + 1
///   LABEL0    [v1]
///   segment   v2 = v1 + 1
///   LABEL1    [v2]                     <- header, the `loop`
///   body      v3 = v2 + 1
///             guard v3 > 10            <- fail 0, the region
///             guard v3 < 1000          <- fail 1, exits with [v3, v2, v1]
///             JUMP -> LABEL1 [v3]
///   region    v11 = v10 + 5000; JUMP -> LABEL0 [v11]
///
/// v0 = 0 makes v3 = 3, so the first guard fails into the region, which
/// re-enters at LABEL0 with 5003. The segment then makes v2 = 5004 and the body
/// v3 = 5005, which passes the first guard and fails the second.
#[test]
fn region_closing_at_a_non_header_label_reenters_and_runs_the_segment() {
    assert_eq!(run_non_header_region_repro(false), (5005, 5004, 5003));
}

/// The same trace with a Ref label arg carried through both LABELs and rebound
/// by the region's JUMP, so the region's back edge has to refresh a Ref home
/// exactly as the resume loader does.
#[test]
fn region_closing_at_a_non_header_label_carries_a_ref_label_arg() {
    assert_eq!(run_non_header_region_repro(true), (5005, 5004, 5003));
}

fn run_non_header_region_repro(with_ref: bool) -> (i64, i64, i64) {
    let descr0 = majit_ir::make_loop_target_descr(20, false);
    let descr1 = majit_ir::make_loop_target_descr(21, false);

    // v6 is a Ref inputarg threaded through both LABELs unchanged.
    let (label0_args, label1_args, jump_args, region_jump_args) = if with_ref {
        (
            vec![rb(OpRef::int_op(1)), rb(OpRef::ref_op(6))],
            vec![rb(OpRef::int_op(2)), rb(OpRef::ref_op(6))],
            vec![rb(OpRef::int_op(3)), rb(OpRef::ref_op(6))],
            vec![rb(OpRef::int_op(11)), rb(OpRef::ref_op(12))],
        )
    } else {
        (
            vec![rb(OpRef::int_op(1))],
            vec![rb(OpRef::int_op(2))],
            vec![rb(OpRef::int_op(3))],
            vec![rb(OpRef::int_op(11))],
        )
    };
    let label0 = Op::new(OpCode::Label, &label0_args);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &label1_args);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &jump_args);
    jump.setdescr(descr1);

    let region_failargs: Vec<_> = if with_ref {
        vec![OpRef::int_op(3), OpRef::ref_op(6)]
    } else {
        vec![OpRef::int_op(3)]
    };
    let guard_to_region = make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &region_failargs);
    let guard_exit = make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(5)],
        &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(1)],
    );
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        guard_to_region,
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(1000)],
            OpRef::int_op(5),
        ),
        guard_exit,
        jump,
    ];
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_jump = Op::new(OpCode::Jump, &region_jump_args);
    region_jump.setdescr(descr0);
    let mut region_ops = vec![make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
        OpRef::int_op(11),
    )];
    if with_ref {
        // The region rebinds the Ref label arg to its own live-in copy, so the
        // back edge must move it AND refresh its home.
        region_ops.push(make_op(
            OpCode::SameAsR,
            &[OpRef::ref_op(13)],
            OpRef::ref_op(12),
        ));
    }
    region_ops.push(region_jump);

    let mut inputargs = vec![InputArg::from_type(Type::Int, 0)];
    if with_ref {
        inputargs.push(InputArg::from_type(Type::Ref, 6));
    }
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![codegen::InlinedBridge {
            source_fail_index: 0,
            trace_id: 1,
            inputargs: if with_ref {
                vec![
                    InputArg::from_type(Type::Int, 10),
                    InputArg::from_type(Type::Ref, 13),
                ]
            } else {
                vec![InputArg::from_type(Type::Int, 10)]
            },
            ops: region_ops,
            gc_table_base: 0,
            constants: indexmap::IndexMap::new(),
        }],
    );
    let (bytes, _, _) = codegen::build_wasm_module(&inputs).expect("non-header region merges");
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(2, None)).expect("test memory should allocate");
    memory
        .write(
            &mut store,
            codegen::FRAME_SLOT_BASE as usize,
            &0i64.to_le_bytes(),
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

    let read = |off: u64| {
        let mut buf = [0u8; 8];
        memory.read(&store, off as usize, &mut buf).unwrap();
        i64::from_le_bytes(buf)
    };
    assert_eq!(read(0), 1, "the second guard is the one that exits");
    (
        read(codegen::FRAME_SLOT_BASE),
        read(codegen::FRAME_SLOT_BASE + 8),
        read(codegen::FRAME_SLOT_BASE + 16),
    )
}

/// Collect every `i32.const` / `i64.const` immediate in the emitted body, so a
/// test can assert which helper address the allocation arms baked in.
fn const_immediates(bytes: &[u8]) -> Vec<i64> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I32Const { value } => out.push(value as i64),
                    wasmparser::Operator::I64Const { value } => out.push(value),
                    _ => {}
                }
            }
        }
    }
    out
}

/// A `non_moving` descr must allocate through the old-generation helper, never
/// the nursery one. The native backends make that choice in the GC rewrite pass
/// (`handle_new` / `handle_new_array`); the wasm backend lowers `New*` itself,
/// so the flag has to be honoured here or it is silently dropped.
///
/// Dropping it is not a slowdown, it is a use-after-move: a `non_moving` descr
/// marks an object reached through a raw pointer nothing forwards, so a movable
/// copy leaves those pointers on the pre-move address.
#[test]
fn test_non_moving_descr_allocates_through_the_oldgen_helper() {
    use majit_ir::descr::{SimpleArrayDescr, SimpleSizeDescr};
    use std::sync::Arc;

    const NEW_FN: i64 = 0x11;
    const NEW_ARRAY_FN: i64 = 0x22;
    const NEW_OLDGEN_FN: i64 = 0x33;
    const NEW_ARRAY_OLDGEN_FN: i64 = 0x44;

    // One `New` and one `NewArrayClear`, both marked non-moving.
    let build = |non_moving: bool| {
        let size_descr = SimpleSizeDescr::new(0, 32, 53);
        size_descr.set_non_moving(non_moving);
        let array_descr = SimpleArrayDescr::new(1, 8, 8, 55, Type::Ref);
        array_descr.set_non_moving(non_moving);

        let new_op = make_op(OpCode::New, &[], OpRef::ref_op(1));
        new_op.setdescr(Arc::new(size_descr));
        let new_array_op = make_op(
            OpCode::NewArrayClear,
            &[OpRef::input_arg_int(0)],
            OpRef::ref_op(2),
        );
        new_array_op.setdescr(Arc::new(array_descr));
        let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
        finish.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);

        let inputs = codegen::ModuleBuildInputs {
            inputargs: vec![InputArg::from_type(Type::Int, 0)],
            ops: vec![new_op, new_array_op, finish],
            inlined_bridges: Vec::new(),
            constants: indexmap::IndexMap::new(),
            vtable_offset: Some(0),
            classptr_to_typeid: HashMap::new(),
            guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
            alloc: codegen::AllocHelpers {
                new_fn_ptr: NEW_FN,
                new_array_fn_ptr: NEW_ARRAY_FN,
                new_oldgen_fn_ptr: NEW_OLDGEN_FN,
                new_array_oldgen_fn_ptr: NEW_ARRAY_OLDGEN_FN,
            },
            wb_fn_ptr: 0,
            nursery: None, // the inline bump is off, so only the helper choice shows
            invalidated_flag_addr: 0,
            gc_table_base: 0,
            fail_index_base: 0,
            bridge_cells_base: 0,
            bridge_entry_arity: None,
            bridge_param_dispatch: false,
            trace_entry_census: None,
            external_jump_slot: 0,
            external_jump_key: 0,
            frame: codegen::FrameGeometry::fixed(),
            ca: codegen::CaParams::default(),
        };
        let (bytes, _, _) =
            codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
        validate_wasm(&bytes);
        const_immediates(&bytes)
    };

    let moving = build(false);
    assert!(moving.contains(&NEW_FN) && moving.contains(&NEW_ARRAY_FN));
    assert!(!moving.contains(&NEW_OLDGEN_FN) && !moving.contains(&NEW_ARRAY_OLDGEN_FN));

    let non_moving = build(true);
    assert!(
        non_moving.contains(&NEW_OLDGEN_FN),
        "non_moving New must call the old-gen helper"
    );
    assert!(
        non_moving.contains(&NEW_ARRAY_OLDGEN_FN),
        "non_moving NewArrayClear must call the old-gen helper"
    );
    assert!(
        !non_moving.contains(&NEW_FN) && !non_moving.contains(&NEW_ARRAY_FN),
        "non_moving descrs must not reach the nursery helpers"
    );
}

/// Minimal `FailDescr` for driving `Backend::compile_bridge` from the host.
/// `trace_id` defaults to 0, which is also a freshly compiled loop's, so the
/// bridge reads as a direct guard of the source loop.
#[derive(Debug)]
struct HostFailDescr {
    fail_index: u32,
    arg_types: Vec<Type>,
}

impl majit_ir::Descr for HostFailDescr {}

impl majit_ir::descr::FailDescr for HostFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.arg_types
    }
}

/// A two-input loop whose single LABEL sits at the entry, so
/// `stamp_and_publish_label_targets` publishes it as a re-enterable target and
/// a bridge closing onto `label_descr` resolves in-module.
fn host_loop_ops(label_descr: &std::sync::Arc<dyn majit_ir::Descr>) -> Vec<std::rc::Rc<Op>> {
    let label = std::rc::Rc::new(Op::new(
        OpCode::Label,
        &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
    ));
    label.setdescr(label_descr.clone());
    let advance = std::rc::Rc::new(make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(0), OpRef::const_int(1)],
        OpRef::int_op(2),
    ));
    let guard = std::rc::Rc::new(make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(2)],
        &[OpRef::int_op(2), OpRef::input_arg_int(1)],
    ));
    // Bind the JUMP to the real producer, not to a synthetic stand-in: the
    // loop-carried advance is read off the arg's producing opcode.
    let jump = std::rc::Rc::new(Op::new(
        OpCode::Jump,
        &[
            Operand::from_bound_op(&advance),
            rb(OpRef::input_arg_int(1)),
        ],
    ));
    jump.setdescr(label_descr.clone());
    vec![label, advance, guard, jump]
}

/// A loop-closing bridge for the loop above: it advances one loop-carried
/// value and jumps back onto the owner's published label.
fn host_bridge_ops(label_descr: &std::sync::Arc<dyn majit_ir::Descr>) -> Vec<std::rc::Rc<Op>> {
    let advance = std::rc::Rc::new(make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(40), OpRef::const_int(1)],
        OpRef::int_op(42),
    ));
    let jump = std::rc::Rc::new(Op::new(
        OpCode::Jump,
        &[
            Operand::from_bound_op(&advance),
            rb(OpRef::input_arg_int(41)),
        ],
    ));
    jump.setdescr(label_descr.clone());
    vec![advance, jump]
}

fn host_bridge_inputargs() -> Vec<InputArg> {
    vec![
        InputArg::from_type(Type::Int, 40),
        InputArg::from_type(Type::Int, 41),
    ]
}

/// The global fail-descr space is appended to under the assumption that no
/// other compile interleaves (`failguard.rs register_fail_descrs`), which the
/// single-threaded wasm host guarantees and a parallel test runner does not.
static HOST_COMPILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn host_loop_inputargs() -> Vec<InputArg> {
    vec![
        InputArg::from_type(Type::Int, 0),
        InputArg::from_type(Type::Int, 1),
    ]
}

/// The complement of the decline below. With the same owner, guard and bridge
/// but no invalidation, nothing short-circuits the inline arm: the trial runs
/// every accept precondition and reaches the merged install. On the host that
/// install cannot complete — there is no wasm host, so the owner's module was
/// never materialized and the rebuild refuses — which is what `bridge_diag(37)`
/// records. So this does not assert an accepted inline (only a wasm host can
/// produce one); it pins that the invalidation decline is the ONLY thing
/// separating the two tests.
#[test]
fn a_valid_owner_reaches_the_inline_trial() {
    use majit_backend::Backend;

    let _serialized = HOST_COMPILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut backend = majit_backend_wasm::WasmBackend::new();
    let token = majit_backend::JitCellToken::new(1);
    let label_descr = majit_ir::make_loop_target_descr(70, false);

    backend
        .compile_loop(&host_loop_inputargs(), &host_loop_ops(&label_descr), &token)
        .expect("the owner loop compiles");
    assert!(!token.is_invalidated());

    let fail_descr = HostFailDescr {
        fail_index: 0,
        arg_types: vec![Type::Int, Type::Int],
    };
    let declines_before = majit_backend_wasm::bridge_diag(50);
    let trials_before = majit_backend_wasm::bridge_diag(37);
    backend
        .compile_bridge(
            &fail_descr,
            &host_bridge_inputargs(),
            &host_bridge_ops(&label_descr),
            &token,
            &[],
            None,
        )
        .expect("the loop-closing bridge compiles");

    assert_eq!(
        majit_backend_wasm::bridge_diag(50),
        declines_before,
        "a valid owner is not declined by the invalidation arm"
    );
    assert!(
        majit_backend_wasm::bridge_diag(37) > trials_before,
        "the inline trial reached the merged install"
    );
}

/// `model.py:145-152`, pinned upstream by `runner_test.py
/// test_guard_not_invalidated` steps 3 and 4: a bridge compiled AFTER
/// `invalidate_loop` starts valid, and only a LATER invalidation activates its
/// GUARD_NOT_INVALIDATED.
///
/// A merged region cannot honour that. It runs from the owner's module, whose
/// `invalidated_flag_addr` is the owner's root flag — already set once the
/// owner is invalidated — so the region would be dead the moment it is
/// installed. The inline arm therefore declines on an invalidated owner and
/// leaves the out-of-line path to mint the fresh, clear generation.
#[test]
fn a_bridge_compiled_after_the_owner_was_invalidated_starts_valid() {
    use majit_backend::Backend;

    let _serialized = HOST_COMPILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut backend = majit_backend_wasm::WasmBackend::new();
    let token = majit_backend::JitCellToken::new(1);
    let label_descr = majit_ir::make_loop_target_descr(71, false);

    backend
        .compile_loop(&host_loop_inputargs(), &host_loop_ops(&label_descr), &token)
        .expect("the owner loop compiles");
    assert!(
        token.latest_bridge_invalidation_flag().is_none(),
        "no bridge generation exists before any bridge is compiled"
    );

    token.invalidate();
    assert!(token.is_invalidated());

    let fail_descr = HostFailDescr {
        fail_index: 0,
        arg_types: vec![Type::Int, Type::Int],
    };
    let declines_before = majit_backend_wasm::bridge_diag(50);
    backend
        .compile_bridge(
            &fail_descr,
            &host_bridge_inputargs(),
            &host_bridge_ops(&label_descr),
            &token,
            &[],
            None,
        )
        .expect("the loop-closing bridge compiles out of line");

    assert!(
        majit_backend_wasm::bridge_diag(50) > declines_before,
        "the inline arm must decline an invalidated owner"
    );
    let generation = token
        .latest_bridge_invalidation_flag()
        .expect("the out-of-line path mints a generation for this bridge");
    assert!(
        !generation.load(std::sync::atomic::Ordering::Acquire),
        "a bridge compiled after invalidate_loop starts valid"
    );
}

/// The same two-label shape, but with a backend-only live-in captured at the
/// label the region resumes at: `v7` is produced before LABEL0, is not one of
/// its args, and is read in the loop body. The region's back edge must restore
/// it from its capture slot exactly as the entry `br_table`'s resume loader
/// does, so the body's `v3 + v7` reports 5005 + 100.
#[test]
fn region_closing_at_a_non_header_label_restores_that_labels_captures() {
    assert_eq!(run_non_header_capture_repro(), (5005, 5004, 5105));
}

fn run_non_header_capture_repro() -> (i64, i64, i64) {
    let descr0 = majit_ir::make_loop_target_descr(30, false);
    let descr1 = majit_ir::make_loop_target_descr(31, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        // Produced before LABEL0, not one of its args, read after the header:
        // the capture plan must hold it across both resume paths.
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(100)],
            OpRef::int_op(7),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(3), OpRef::int_op(7)],
            OpRef::int_op(8),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(3)]),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(1000)],
            OpRef::int_op(5),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(5)],
            &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(8)],
        ),
        jump,
    ];
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_jump.setdescr(descr0);
    let region_ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
            OpRef::int_op(11),
        ),
        region_jump,
    ];

    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![codegen::InlinedBridge {
            source_fail_index: 0,
            trace_id: 1,
            inputargs: vec![InputArg::from_type(Type::Int, 10)],
            ops: region_ops,
            gc_table_base: 0,
            constants: indexmap::IndexMap::new(),
        }],
    );
    let (bytes, _, _) = codegen::build_wasm_module(&inputs).expect("non-header region merges");
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(2, None)).expect("test memory should allocate");
    memory
        .write(
            &mut store,
            codegen::FRAME_SLOT_BASE as usize,
            &0i64.to_le_bytes(),
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

    let read = |off: u64| {
        let mut buf = [0u8; 8];
        memory.read(&store, off as usize, &mut buf).unwrap();
        i64::from_le_bytes(buf)
    };
    assert_eq!(read(0), 1, "the second guard is the one that exits");
    (
        read(codegen::FRAME_SLOT_BASE),
        read(codegen::FRAME_SLOT_BASE + 8),
        read(codegen::FRAME_SLOT_BASE + 16),
    )
}

/// Two regions attached to the SAME owner, both closing at the NON-header
/// LABEL. Each region owns one of the blocks opened at the loop header, so the
/// guard that reaches it and the back edge it takes must both name that
/// region's own depth — region 0 innermost.
///
///   preamble  v1 = v0 + 1
///   LABEL0    [v1]
///   segment   v2 = v1 + 1
///   LABEL1    [v2]                     <- header, the `loop`
///   body      v3 = v2 + 1
///             guard v3 > 10            <- fail 0, region A
///             guard v3 > 6000          <- fail 1, region B
///             guard v3 < 100000        <- fail 2, exits with [v3, v2, v1]
///             JUMP -> LABEL1 [v3]
///   region A  v11 = v10 + 5000;  JUMP -> LABEL0 [v11]
///   region B  v21 = v20 + 50000; JUMP -> LABEL0 [v21]
#[test]
fn two_regions_closing_at_a_non_header_label_each_reenter_at_their_own_depth() {
    assert_eq!(run_two_non_header_regions_repro(), (100000, 99999, 55005));
}

fn run_two_non_header_regions_repro() -> (i64, i64, i64) {
    let descr0 = majit_ir::make_loop_target_descr(40, false);
    let descr1 = majit_ir::make_loop_target_descr(41, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(3)]),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(6000)],
            OpRef::int_op(5),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(5)], &[OpRef::int_op(3)]),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(100000)],
            OpRef::int_op(6),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(6)],
            &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(1)],
        ),
        jump,
    ];
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_a_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_a_jump.setdescr(descr0.clone());
    let region_a = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
            OpRef::int_op(11),
        ),
        region_a_jump,
    ];
    let region_b_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(21))]);
    region_b_jump.setdescr(descr0);
    let region_b = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(20), OpRef::const_int(50000)],
            OpRef::int_op(21),
        ),
        region_b_jump,
    ];

    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![
            codegen::InlinedBridge {
                source_fail_index: 0,
                trace_id: 1,
                inputargs: vec![InputArg::from_type(Type::Int, 10)],
                ops: region_a,
                gc_table_base: 0,
                constants: indexmap::IndexMap::new(),
            },
            codegen::InlinedBridge {
                source_fail_index: 1,
                trace_id: 2,
                inputargs: vec![InputArg::from_type(Type::Int, 20)],
                ops: region_b,
                gc_table_base: 0,
                constants: indexmap::IndexMap::new(),
            },
        ],
    );
    let (bytes, _, _) = codegen::build_wasm_module(&inputs).expect("two non-header regions merge");
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(2, None)).expect("test memory should allocate");
    memory
        .write(
            &mut store,
            codegen::FRAME_SLOT_BASE as usize,
            &0i64.to_le_bytes(),
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

    let read = |off: u64| {
        let mut buf = [0u8; 8];
        memory.read(&store, off as usize, &mut buf).unwrap();
        i64::from_le_bytes(buf)
    };
    assert_eq!(read(0), 2, "the third guard is the one that exits");
    (
        read(codegen::FRAME_SLOT_BASE),
        read(codegen::FRAME_SLOT_BASE + 8),
        read(codegen::FRAME_SLOT_BASE + 16),
    )
}

/// A region's guard must sit inside the `loop` whose blocks it branches to.
/// `InlineGuard::branch_depth` counts those blocks from loop-body statement
/// level; in the peeled preamble the same depth names a LABEL resume loader,
/// so a bridge sourced there has to keep the out-of-line path.
#[test]
fn a_preamble_guard_is_reported_as_unreachable_from_the_region_blocks() {
    let descr0 = majit_ir::make_loop_target_descr(50, false);
    let descr1 = majit_ir::make_loop_target_descr(51, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0);
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(1), OpRef::const_int(50)],
            OpRef::int_op(6),
        ),
        // exit 0: in the preamble, between the two LABELs.
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(6)], &[OpRef::int_op(1)]),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(100)],
            OpRef::int_op(4),
        ),
        // exit 1: in the loop body.
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(3)]),
        jump,
    ];

    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let inputs = inline_region_inputs(&inputargs, ops, vec![]);
    assert!(codegen::inline_source_guard_precedes_loop_label(&inputs, 0));
    assert!(!codegen::inline_source_guard_precedes_loop_label(
        &inputs, 1
    ));
}
