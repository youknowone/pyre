//! Phase D.1: exercise `CallControl::find_all_graphs` over the full
//! pyre-interpreter opcode handler set.
//!
//! RPython parity points:
//! - `rpython/jit/codewriter/codewriter.py:74 CodeWriter.make_jitcodes` calls
//!   `callcontrol.grab_initial_jitcodes()` which calls `find_all_graphs(policy)`
//!   via `call.py:49-92`.
//! - `rpython/jit/codewriter/call.py:145 find_all_graphs` BFSes the reachable
//!   graph set from a portal seed.
//!
//! This harness does the equivalent seeding for pyre: every freestanding
//! `opcode_*` handler in `pyopcode.rs` is a portal; every `impl <Trait> for
//! PyFrame` method is a candidate graph reachable by trait dispatch.
//! After `find_all_graphs` runs we assert that the candidate closure covers
//! both the portals and the PyFrame trait-method graphs that the super-
//! instruction helpers invoke.
//!
//! No new helpers on `CallControl` are introduced — the test uses the
//! existing `register_function_graph`, `register_trait_method`,
//! `mark_portal`, and `find_all_graphs` API surface. If Phase D reveals
//! that this wiring is repeatedly needed, a thin registration helper can
//! be added in a follow-up commit.

use std::path::PathBuf;

use majit_translate::{
    CallPath, ParsedInterpreter,
    call::CallControl,
    extract_trait_impls,
    front::{StructFieldRegistry, ast::build_function_graph_pub},
    parse_source,
    policy::DefaultJitPolicy,
};
use syn::{Item, ItemFn};

fn pyre_file_path(relative: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("..");
    p.push("..");
    for segment in relative.split('/') {
        p.push(segment);
    }
    p
}

fn parse_pyre_file(relative: &str) -> ParsedInterpreter {
    let path = pyre_file_path(relative);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    parse_source(&src)
}

fn iter_opcode_handler_fns(file: &syn::File) -> impl Iterator<Item = &ItemFn> {
    file.items.iter().filter_map(|item| match item {
        Item::Fn(func) if func.sig.ident.to_string().starts_with("opcode_") => Some(func),
        _ => None,
    })
}

#[test]
fn find_all_graphs_closure_reaches_pyframe_methods() {
    let pyopcode = parse_pyre_file("pyre/pyre-interpreter/src/pyopcode.rs");
    let eval = parse_pyre_file("pyre/pyre-interpreter/src/eval.rs");

    let mut cc = CallControl::new();

    // Register every opcode_* freestanding handler as a portal entry point.
    // This is the pyre analogue of RPython `warmspot.py` handing its
    // portal graphs to the callcontrol before `grab_initial_jitcodes` runs.
    let mut portal_names = Vec::new();
    for func in iter_opcode_handler_fns(&pyopcode.file) {
        let sf = build_function_graph_pub(func);
        let path = CallPath::from_segments([sf.name.clone()]);
        cc.register_function_graph(path.clone(), sf.graph);
        cc.mark_portal(path);
        portal_names.push(sf.name);
    }
    assert!(
        portal_names.len() >= 20,
        "expected at least 20 opcode_* handlers from pyopcode.rs, got {}",
        portal_names.len()
    );

    // Register trait impl graphs from both pyopcode.rs and eval.rs so BFS
    // can follow executor.method(...) calls into the concrete impls.
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = std::collections::HashMap::new();
    let empty_struct_names = std::collections::HashSet::new();
    let mut impls = Vec::new();
    impls.extend(extract_trait_impls(
        &pyopcode,
        &empty_registry,
        &empty_fn_ret,
        &empty_struct_names,
    ));
    impls.extend(extract_trait_impls(
        &eval,
        &empty_registry,
        &empty_fn_ret,
        &empty_struct_names,
    ));
    for imp in &impls {
        for method in &imp.methods {
            if let Some(graph) = method.graph.clone() {
                cc.register_trait_method(&method.name, Some(&imp.trait_name), &imp.for_type, graph);
            }
        }
    }

    // Run the BFS closure (call.py:49-92 parity).
    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    // Every portal must be in the candidate set — `find_all_graphs` seeds
    // them first (call.py:56).
    for name in &portal_names {
        let path = CallPath::from_segments([name.clone()]);
        assert!(
            cc.is_candidate(&path),
            "portal `{}` missing from candidate graphs after find_all_graphs",
            name
        );
    }

    // The function graph registry must contain at least one PyFrame trait
    // method graph.
    let pyframe_method_paths: Vec<&CallPath> = cc
        .function_graphs()
        .keys()
        .filter(|p| p.segments.first().map(|s| s.as_str()) == Some("PyFrame"))
        .collect();
    assert!(
        pyframe_method_paths.len() >= 9,
        "expected at least 9 `PyFrame::*` method graphs registered via \
         register_trait_method, got {}",
        pyframe_method_paths.len()
    );

    // BFS must reach the concrete PyFrame impls of every trait method
    // invoked by the four super-instruction helpers — these are the
    // atomic operations whose `-live-` emission will drive super-inst
    // parity once Phase E wires `make_jitcodes` end-to-end.
    let required_pyframe_methods = [
        "load_local_value",
        "load_local_checked_value",
        "store_local_value",
        "push_value",
        "pop_value",
    ];
    for method in required_pyframe_methods {
        let path = CallPath::from_segments(["PyFrame", method]);
        assert!(
            cc.is_candidate(&path),
            "`PyFrame::{}` not in candidate graphs after find_all_graphs — \
             BFS failed to walk from opcode_* portals into the trait impl",
            method
        );
    }

    // Emit the summary so the matrix is readable from `cargo test --nocapture`.
    let candidate_count = cc
        .function_graphs()
        .keys()
        .filter(|p| cc.is_candidate(p))
        .count();
    let pyframe_candidates: Vec<&CallPath> = cc
        .function_graphs()
        .keys()
        .filter(|p| p.segments.first().map(|s| s.as_str()) == Some("PyFrame") && cc.is_candidate(p))
        .collect();
    eprintln!(
        "[phase-d.1] portals={} pyframe_methods_registered={} candidates={} pyframe_in_candidates={}",
        portal_names.len(),
        pyframe_method_paths.len(),
        candidate_count,
        pyframe_candidates.len(),
    );

    if !pyframe_candidates.is_empty() {
        let mut names: Vec<String> = pyframe_candidates
            .iter()
            .map(|p| p.canonical_key())
            .collect();
        names.sort();
        eprintln!("[phase-d.1] PyFrame candidates (sample): ");
        for name in names.iter().take(15) {
            eprintln!("[phase-d.1]   {}", name);
        }
        if names.len() > 15 {
            eprintln!("[phase-d.1]   ... +{} more", names.len() - 15);
        }
    }
}
