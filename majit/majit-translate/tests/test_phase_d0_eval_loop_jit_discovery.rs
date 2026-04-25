//! Phase D0 probe: verify that `pyre-jit/src/eval.rs` is syn-parseable and
//! that the `eval_loop_jit` portal runner can be located + lowered through
//! `build_function_graph_pub`.
//!
//! This is the *first* concrete Phase D0 step. It does not yet wire the
//! graph into the codewriter pipeline — it only establishes the baseline
//! question "can the Rust-source adapter read pyre-jit sources at all?"
//! If lowering fails for structural reasons (unsupported Rust patterns,
//! missing helper registrations) those failures show up here and the
//! remediation list becomes a concrete Phase D0 backlog.
//!
//! Positioning: Phase D0 is a Parity layer port — `warmspot.py` registers
//! a portal graph with the codewriter (`call.py:57 find_all_graphs(portal,
//! policy)`). pyre's equivalent graph lives in a sibling crate; this
//! probe confirms the bridge.

use std::path::PathBuf;

use majit_translate::front::ast::build_function_graph_pub;
use syn::{File, Item};

fn eval_rs_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("..");
    p.push("..");
    p.push("pyre");
    p.push("pyre-jit");
    p.push("src");
    p.push("eval.rs");
    p
}

fn parse_eval_rs() -> File {
    let path = eval_rs_path();
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    syn::parse_file(&src).unwrap_or_else(|e| panic!("failed to parse {}: {}", path.display(), e))
}

fn find_fn<'a>(file: &'a File, name: &str) -> Option<&'a syn::ItemFn> {
    file.items.iter().find_map(|item| match item {
        Item::Fn(func) if func.sig.ident == name => Some(func),
        _ => None,
    })
}

#[test]
fn eval_rs_is_syn_parseable() {
    // Minimal smoke: the file reads and parses. Structural discovery
    // (eval_loop_jit lowering) is exercised below so that a parse failure
    // here fails fast with a file-level diagnostic.
    let file = parse_eval_rs();
    let fn_count = file
        .items
        .iter()
        .filter(|item| matches!(item, Item::Fn(_)))
        .count();
    assert!(
        fn_count > 0,
        "pyre-jit/src/eval.rs parsed to zero free-function items — syn recognised \
         something but nothing Phase D0 cares about"
    );
}

#[test]
fn eval_loop_jit_lowers_to_function_graph() {
    // The portal-runner graph is the equivalent of RPython's
    // `warmspot.py::portal_runner` — `find_all_graphs(portal, policy)`
    // (`call.py:57`) seeds BFS from this graph in upstream. The first
    // step of Phase D0 is having any graph at all; later steps handle
    // JitDriver registration + green/red layout.
    let file = parse_eval_rs();
    let func = find_fn(&file, "eval_loop_jit").unwrap_or_else(|| {
        panic!(
            "fn eval_loop_jit not found at top level of pyre-jit/src/eval.rs — \
             did the portal runner get renamed? Phase D0 portal identity \
             depends on this name matching the `PipelineConfig::portal` value."
        )
    });

    // Lower through the adapter. A panic here is Phase D0's first concrete
    // blocker: an unsupported Rust pattern inside the portal runner that
    // `front::ast` cannot yet express in `FunctionGraph` form.
    let sf = build_function_graph_pub(func).expect("eval_loop_jit must lower");
    assert_eq!(
        sf.name, "eval_loop_jit",
        "SemanticFunction.name must match the source ident"
    );
    assert!(
        !sf.graph.blocks.is_empty(),
        "eval_loop_jit lowered to zero blocks — empty graph is a hard parity \
         violation vs upstream portal_runner"
    );
    eprintln!(
        "[phase-d0] eval_loop_jit lowered: {} blocks, {} ops",
        sf.graph.blocks.len(),
        sf.graph
            .blocks
            .iter()
            .map(|b| b.operations.len())
            .sum::<usize>(),
    );
}
