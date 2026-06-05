//! Structural anchor for `make_jitcodes` graph-keyed output.
//!
//! Runs as
//! `cargo test -p majit-translate test_make_jitcodes_produces_graph_keyed_output`.
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

use majit_translate::{generated::with_all_jitcodes, jitcode::JitCode};
use std::collections::HashSet;
use std::sync::Arc;

#[test]
fn test_make_jitcodes_produces_graph_keyed_output() {
    if !ensure_workspace_llbc_env() {
        eprintln!(
            "skipping: build/llbc/{{pyre-object,pyre-interpreter,pyre-jit}}.ullbc missing — \
             run `scripts/extract-llbc.sh` to enable this test"
        );
        return;
    }
    // `AllJitCodes::by_path` is keyed by `CallPath` (graph identity),
    // matching upstream `call.py:87 self.jitcodes`.
    // The `by_path` field's type ensures at compile time that no
    // Instruction-variant key can ever land here — this test only has
    // to verify that the live registry respects the contract without
    // structural surprises.
    with_all_jitcodes(|reg| {
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

        // Invariant 3: registry size reflects the expected closure. The
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

        // Invariant 4: registry contains per-opcode-arm dispatch JitCodes,
        // produced by `build_canonical_opcode_dispatch` (lib.rs:965-1027)
        // from the decomposed `execute_opcode_step` match arms. The portal
        // selector itself is not separately registered; each arm becomes
        // one `__opcode_dispatch__::<selector>#<arm_id>` JitCode mirroring
        // RPython's per-opcode handler graphs at `call.py:145-148
        // grab_initial_jitcodes`.
        let has_arm = reg.by_path.keys().any(|k| {
            k.segments
                .first()
                .map(|s| s == "__opcode_dispatch__")
                .unwrap_or(false)
        });
        assert!(
            has_arm,
            "per-opcode dispatch arms missing from `by_path` — \
             `build_canonical_opcode_dispatch` (lib.rs:965-1027) did not seed any \
             `__opcode_dispatch__::<selector>#<arm_id>` JitCode"
        );
    });

    // Invariant 5: registry is not keyed by Instruction. This is a
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
/// when they exist. The 5-source `PYRE_JIT_GRAPH_SOURCES` fixture used
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
