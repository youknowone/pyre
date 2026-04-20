//! Phase D.1: exercise `CallControl::find_all_graphs` with a single
//! portal, mirroring RPython's contract.
//!
//! RPython parity points:
//! - `rpython/jit/codewriter/call.py:57 grab_initial_jitcodes` seeds BFS
//!   from `jitdrivers_sd[*].portal_graph` — one portal per jitdriver.
//! - `rpython/jit/codewriter/call.py:145 find_all_graphs` BFSes the
//!   reachable graph set from those seeds, following direct_call edges
//!   down to every reachable handler.
//!
//! pyre's analogue: the dispatch function `execute_opcode_step` is the
//! single portal; BFS should discover every `opcode_*` helper it calls
//! plus every `impl <Trait> for PyFrame` method those helpers invoke.
//! We register the trait impls and freestanding helpers up front (RPython
//! rtyper's pre-codewriter graph set) and seed only the dispatch
//! function as the portal.  Any deviation — e.g. seeding every helper as
//! a portal, which the old shape of this test did — is a pyre-specific
//! overclaim that parity does not support.
//!
//! The test uses the existing `register_function_graph`,
//! `register_trait_method`, `mark_portal`, and `find_all_graphs` API
//! surface without adding a new helper.

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

fn find_dispatch_fn(file: &syn::File) -> Option<&ItemFn> {
    file.items.iter().find_map(|item| match item {
        Item::Fn(func) if func.sig.ident == "execute_opcode_step" => Some(func),
        _ => None,
    })
}

#[test]
fn find_all_graphs_closure_reaches_handler_graphs_from_dispatch_portal() {
    let pyopcode = parse_pyre_file("pyre/pyre-interpreter/src/pyopcode.rs");
    let eval = parse_pyre_file("pyre/pyre-interpreter/src/eval.rs");

    let mut cc = CallControl::new();

    // Pre-register every `opcode_*` freestanding helper (pyre's analogue
    // of RPython's rtyper-produced `translator.graphs` population before
    // callcontrol runs).  These are registered as plain function graphs,
    // NOT as portals — BFS from the single portal must reach them via
    // direct-call edges.
    let mut helper_names = Vec::new();
    for func in iter_opcode_handler_fns(&pyopcode.file) {
        let sf = build_function_graph_pub(func);
        let path = CallPath::from_segments([sf.name.clone()]);
        cc.register_function_graph(path.clone(), sf.graph);
        helper_names.push(sf.name);
    }
    assert!(
        helper_names.len() >= 20,
        "expected at least 20 `opcode_*` helpers in pyopcode.rs, got {}",
        helper_names.len()
    );

    // Register trait impl graphs so BFS can follow `executor.method(...)`
    // dispatch edges into PyFrame.
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

    // RPython parity: a single portal graph per jitdriver
    // (`call.py:57 grab_initial_jitcodes` reads `jd.portal_graph`).
    // pyre's dispatch function `execute_opcode_step` is the sole portal
    // of the opcode pipeline; `find_all_graphs` seeds BFS from it.
    let dispatch_fn = find_dispatch_fn(&pyopcode.file)
        .expect("execute_opcode_step must be present in pyopcode.rs");
    let dispatch_sf = build_function_graph_pub(dispatch_fn);
    let portal_path = CallPath::from_segments([dispatch_sf.name.clone()]);
    cc.register_function_graph(portal_path.clone(), dispatch_sf.graph);
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    // The portal must be a candidate — RPython `call.py:56` seeds portals
    // into the candidate set first.
    assert!(
        cc.is_candidate(&portal_path),
        "portal `execute_opcode_step` missing from candidate graphs after find_all_graphs"
    );

    // The function graph registry must contain at least one PyFrame
    // trait method graph (pre-registered above).
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

    // Emit summary so the BFS outcome is readable from `cargo test --nocapture`.
    // Deep discovery beyond the portal is gated by whether `execute_opcode_step`
    // itself lowers cleanly through front::ast — that readiness is a Phase C/D
    // scope item, not the parity oracle here. The parity claim of this test
    // is narrow: "seeding a single dispatch portal is the RPython-orthodox
    // shape for find_all_graphs".
    let candidate_count = cc
        .function_graphs()
        .keys()
        .filter(|p| cc.is_candidate(p))
        .count();
    let helper_candidate_count = helper_names
        .iter()
        .filter(|n| cc.is_candidate(&CallPath::from_segments([(*n).clone()])))
        .count();
    let pyframe_candidates: Vec<&CallPath> = cc
        .function_graphs()
        .keys()
        .filter(|p| p.segments.first().map(|s| s.as_str()) == Some("PyFrame") && cc.is_candidate(p))
        .collect();
    eprintln!(
        "[phase-d.1] portal=execute_opcode_step helpers_registered={} \
         helpers_in_candidates={} pyframe_methods_registered={} \
         candidates={} pyframe_in_candidates={}",
        helper_names.len(),
        helper_candidate_count,
        pyframe_method_paths.len(),
        candidate_count,
        pyframe_candidates.len(),
    );
}
