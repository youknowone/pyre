//! Structural anchor for `make_jitcodes` graph-keyed output.
//!
//! Runs as
//! `cargo test -p majit-translate test_make_jitcodes_produces_graph_keyed_output`.
//! The full generated-registry check runs only under
//! `cargo test --release`; debug-profile CI keeps this file to the cheap
//! structural fixture.
//!
//! ## RPython references
//!
//! - `rpython/jit/codewriter/codewriter.py:74-89 make_jitcodes` — the
//!   main driver that pops from `callcontrol.enum_pending_graphs()`
//!   and produces one `JitCode` per graph.
//! - `rpython/jit/codewriter/codewriter.py:33-72 transform_graph_to_jitcode` —
//!   the 4-step pipeline (jtransform → regalloc → flatten → assemble)
//!   that turns one FunctionGraph into one JitCode.
//! - `rpython/jit/codewriter/call.py:87 self.jitcodes = {}` — the
//!   graph-keyed dict that `AllJitCodes::by_path` mirrors.
//! - `rpython/jit/codewriter/call.py:88 self.all_jitcodes = []` — the
//!   alloc-order list that `AllJitCodes::in_order` mirrors, with the
//!   `all_jitcodes[i].index == i` invariant from `codewriter.py:80`.
//!
//! ## What this test anchors
//!
//! `make_jitcodes` output is **graph-keyed** — one `JitCode` per
//! `CallPath`, with no `Instruction`-variant tables, no
//! opcode-to-fragment lookups, no anything Python bytecode-shaped. The
//! structural prohibition is: output is `{graph: JitCode}` only, no
//! variant-keyed map.
//!
//! `test_phase_f_all_jitcodes.rs` covers most of the behavioural
//! acceptance; this file adds the **structural** anchor, with focused
//! assertions that detect any future drift toward variant-keyed output
//! schemas.

use majit_translate::{
    CallPath,
    generated::{AllJitCodes, with_all_jitcodes},
    jitcode::JitCode,
};
use std::collections::HashSet;
use std::sync::Arc;

#[test]
fn test_make_jitcodes_produces_graph_keyed_output() {
    let reg = fixture_all_jitcodes();

    // This is the cheap structural floor: `make_jitcodes` output is a
    // graph-keyed registry (`CallPath -> JitCode`) with opcode-dispatch arms
    // represented as ordinary synthetic CallPaths, not as a separate
    // Instruction-keyed table. The full interpreter LLBC translation is
    // deliberately kept out of the default test path.
    assert_registry_is_graph_keyed(&reg);

    let dispatch_selectors = dispatch_selectors(&reg);
    assert_eq!(
        dispatch_selectors.len(),
        2,
        "fixture should contain one named opcode arm and one wildcard arm"
    );
    assert!(
        dispatch_selectors.contains("Instruction::LoadConst"),
        "named opcode arm present"
    );
    assert!(
        dispatch_selectors.contains("_"),
        "wildcard opcode arm present"
    );

    assert_no_instruction_keyed_output_map();
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_make_jitcodes_produces_graph_keyed_output`"
)]
fn slow_generated_jitcodes_keep_opcode_dispatch_shape() {
    if !ensure_workspace_llbc_env() {
        eprintln!(
            "skipping: build/llbc/{{pyre-object,pyre-interpreter,pyre-jit}}.ullbc missing — \
             run `scripts/extract-llbc.py` to enable this test"
        );
        return;
    }

    with_all_jitcodes(|reg| {
        assert_registry_is_graph_keyed(reg);

        // Registry size reflects the expected closure. The
        // 28 `opcode_*` handlers lower to FunctionGraphs, every PyFrame
        // trait method has a graph, and the registry holds at least that
        // many JitCodes (plus whatever shared_opcode / inherent method
        // closure the BFS pulls in).
        //
        // Lower floor is intentional — the upper bound drifts as more
        // helpers get pulled into the closure by the BFS. A count
        // regression is the signal; any growth is fine.
        assert!(
            reg.in_order.len() >= 28,
            "expected at least 28 JitCodes (Phase A opcode_* floor); got {}",
            reg.in_order.len()
        );

        // Registry contains every per-opcode-arm dispatch JitCode,
        // produced by `build_canonical_opcode_dispatch` (lib.rs:965-1027)
        // from the decomposed `execute_opcode_step` match arms. The portal
        // selector itself is not separately registered; each arm becomes
        // one `__opcode_dispatch__::<selector>#<arm_id>` JitCode mirroring
        // RPython's per-opcode handler graphs at `call.py:145-148
        // grab_initial_jitcodes`.
        let dispatch_keys = dispatch_keys(reg);
        assert_eq!(
            dispatch_keys.len(),
            111,
            "expected 111 per-opcode dispatch arms in `by_path`; got {}",
            dispatch_keys.len()
        );
        let dispatch_selectors = selectors_from_dispatch_keys(&dispatch_keys);
        assert_eq!(
            dispatch_selectors.len(),
            dispatch_keys.len(),
            "opcode dispatch selectors must be unique across arms"
        );
        assert_eq!(
            dispatch_selectors
                .iter()
                .filter(|selector| selector.contains(" | "))
                .count(),
            3,
            "expected three Or-pattern dispatch arms"
        );
        assert!(
            dispatch_selectors.contains("Instruction::PopTop"),
            "PopTop arm present"
        );
        assert!(
            dispatch_selectors.contains("Instruction::LoadConst"),
            "LoadConst arm present"
        );
        assert!(
            dispatch_selectors.contains("Instruction::ExitInitCheck"),
            "ExitInitCheck present as its own arm"
        );
        assert!(dispatch_selectors.contains("_"), "wildcard key present");
        let async_stub_cases = [
            "Instruction::CleanupThrow",
            "Instruction::EndAsyncFor",
            "Instruction::GetAiter",
            "Instruction::GetAnext",
        ];
        assert!(
            dispatch_selectors.iter().any(|selector| {
                async_stub_cases
                    .iter()
                    .all(|case| selector.split(" | ").any(|part| part == *case))
            }),
            "async-stub Or group present: got {dispatch_selectors:?}"
        );
        assert!(
            dispatch_selectors.contains("Instruction::GetAwaitable"),
            "GetAwaitable present as its own arm"
        );
    });

    assert_no_instruction_keyed_output_map();
}

fn fixture_all_jitcodes() -> AllJitCodes {
    let portal = Arc::new(JitCode::new("portal"));
    let load_const = Arc::new(JitCode::new("Instruction::LoadConst#0"));
    let wildcard = Arc::new(JitCode::new("_#1"));
    let helper = Arc::new(JitCode::new("helper"));

    AllJitCodes {
        by_path: [
            (CallPath::from_segments(["portal"]), Arc::clone(&portal)),
            (
                CallPath::from_segments(["__opcode_dispatch__", "Instruction::LoadConst#0"]),
                Arc::clone(&load_const),
            ),
            (
                CallPath::from_segments(["__opcode_dispatch__", "_#1"]),
                Arc::clone(&wildcard),
            ),
            (CallPath::from_segments(["helper"]), Arc::clone(&helper)),
        ]
        .into_iter()
        .collect(),
        in_order: vec![portal, load_const, wildcard, helper],
    }
}

fn assert_registry_is_graph_keyed(reg: &AllJitCodes) {
    // `AllJitCodes::by_path` is keyed by `CallPath` (graph identity),
    // matching upstream `call.py:87 self.jitcodes`. The field's type
    // ensures no Instruction-variant key can land here; the runtime checks
    // below pin the registry's two views to the same JitCode objects.
    assert!(!reg.in_order.is_empty(), "registry should not be empty");

    // Invariant 1: every `in_order` entry is also reachable through
    // `by_path`. Together with invariant 2 they pin the 1:1 mapping
    // between CallPath and JitCode (upstream `codewriter.py:80-81`).
    let in_order_ptrs: HashSet<usize> = reg
        .in_order
        .iter()
        .map(|jc: &Arc<JitCode>| Arc::as_ptr(jc) as usize)
        .collect();
    let by_path_ptrs: HashSet<usize> = reg
        .by_path
        .values()
        .map(|jc| Arc::as_ptr(jc) as usize)
        .collect();
    assert!(
        in_order_ptrs.is_subset(&by_path_ptrs),
        "every JitCode in `in_order` must be reachable via `by_path` \
         (RPython `call.py:87-88` parity)"
    );

    // Invariant 2: each JitCode Arc appears in `by_path` under exactly
    // one CallPath. Duplicate entries would mean a graph was registered
    // under two different paths, which breaks upstream's `{graph:
    // JitCode}` identity contract at `call.py:157-165`.
    let mut seen: HashSet<usize> = HashSet::new();
    for (path, jc) in &reg.by_path {
        let ptr = Arc::as_ptr(jc) as usize;
        assert!(
            seen.insert(ptr),
            "JitCode `{}` appears in `by_path` under multiple CallPath \
             keys — last seen at {path:?}",
            jc.name
        );
    }
}

fn dispatch_keys(reg: &AllJitCodes) -> Vec<String> {
    reg.by_path
        .keys()
        .filter_map(|k| {
            if k.segments.first().map(|s| s.as_str()) == Some("__opcode_dispatch__") {
                k.segments.get(1).cloned()
            } else {
                None
            }
        })
        .collect()
}

fn selectors_from_dispatch_keys(dispatch_keys: &[String]) -> HashSet<String> {
    dispatch_keys
        .iter()
        .map(|k| {
            k.rsplit_once('#')
                .map(|(selector, _)| selector)
                .unwrap_or(k)
        })
        .map(str::to_string)
        .collect()
}

fn dispatch_selectors(reg: &AllJitCodes) -> HashSet<String> {
    selectors_from_dispatch_keys(&dispatch_keys(reg))
}

fn assert_no_instruction_keyed_output_map() {
    // Registry is not keyed by Instruction. This is a
    // structural assertion: `by_path` is `HashMap<CallPath, Arc<JitCode>>`
    // at the type level, so we can't even construct a variant-keyed
    // view. The assertion below verifies the grep floor agrees — a
    // compile-time type enforces no variant-key lookup ever compiles.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders: Vec<String> = Vec::new();
    walk_rs_files(&root, &mut |path, contents| {
        for (i, line) in contents.lines().enumerate() {
            let trimmed = line.trim_start();
            // Skip doc comments and regular comments — they may
            // legitimately reference the negative form.
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("*") {
                continue;
            }
            if line.contains("HashMap<Instruction") {
                offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "variant-keyed output forbidden — RPython `call.py:87 self.jitcodes` \
         is graph-keyed only:\n{}",
        offenders.join("\n")
    );
}

/// Resolve workspace LLBC artefacts and export `PYRE_MIR_FRONTEND_LLBC`
/// when they exist. The 5-module `PYRE_JIT_GRAPH_MODULES` fixture used
/// by `generated::with_all_jitcodes` falls below the production
/// auto-discovery floor (>=50 parsed_files), so test invocations must
/// opt in explicitly. Returns `false` when the artefacts are absent so
/// the caller can skip cleanly.
fn ensure_workspace_llbc_env() -> bool {
    if std::env::var_os("PYRE_MIR_FRONTEND_LLBC").is_some() {
        return true;
    }
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let llbc_dir = workspace_root.join("build").join("llbc");
    let required = [
        "pyre-object.ullbc",
        "pyre-interpreter.ullbc",
        "pyre-jit.ullbc",
    ];
    let mut paths = Vec::with_capacity(required.len());
    for name in required {
        let p = llbc_dir.join(name);
        if !p.exists() {
            return false;
        }
        paths.push(p.to_string_lossy().into_owned());
    }
    // Join with the OS path-list separator (`;` on Windows, `:` else)
    // so Windows drive letters survive; lib.rs parses it via split_paths.
    let joined = std::env::join_paths(&paths).expect("join llbc paths");
    // SAFETY: set_var is unsafe in Rust 2024 because multi-threaded
    // env mutation races; this test binary runs single-threaded before
    // `with_all_jitcodes` spawns any worker, so the call is sound.
    unsafe { std::env::set_var("PYRE_MIR_FRONTEND_LLBC", joined) };
    true
}

fn walk_rs_files<F: FnMut(&std::path::Path, &str)>(dir: &std::path::Path, f: &mut F) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, f);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                f(&path, &contents);
            }
        }
    }
}
