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
    CallPath, CallTarget, OpKind, ParsedInterpreter, build_semantic_program_from_parsed_files,
    call::CallControl, extract_trait_impls, front::ast::build_function_graph_pub, parse_source,
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

/// Mirror `pyre-jit-trace/build.rs::collect_rs_files`: walk every `.rs`
/// under `pyre-object/src` and `pyre-interpreter/src`, parsing each so
/// `build_semantic_program_from_parsed_files` sees the same
/// whole-program scope as production.
fn collect_pyre_interpreter_program_inputs() -> Vec<ParsedInterpreter> {
    let mut out = Vec::new();
    for dir in ["pyre/pyre-object/src", "pyre/pyre-interpreter/src"] {
        let root = pyre_file_path(dir);
        collect_rs_under(&root, &mut out);
    }
    out
}

fn collect_rs_under(dir: &std::path::Path, out: &mut Vec<ParsedInterpreter>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_under(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        out.push(parse_source(&src));
    }
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
        let Ok(sf) = build_function_graph_pub(func) else {
            continue;
        };
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
    // dispatch edges into PyFrame.  Production at `lib.rs:317-342`
    // populates `program.fn_return_types` / `program.struct_fields` /
    // `program.known_struct_names` via
    // `build_semantic_program_from_parsed_files` BEFORE handing them to
    // `extract_trait_impls`.  Mirror that two-pass walk here so the
    // impl-body lowering can resolve user-source method-return types
    // (`bookkeeper.getdesc(...).find_method` upstream parity at
    // `unaryop.py:206-213`).  Without it, every `!self.method()`
    // surface where `method` is a user-defined predicate fails the
    // `UnaryNotUnknownOperand` fail-loud at `front/ast.rs:3713`.
    // Re-parse for the program build because `ParsedInterpreter`
    // does not derive `Clone`.  Walk the full pyre-interpreter and
    // pyre-object source trees so the walker discovers every
    // user-defined method's return type before any body lowering
    // touches `expr_unary_not_operand_kind`
    // (`bookkeeper.getdesc(...).find_method` upstream parity at
    // `unaryop.py:206-213`).  Mirrors `pyre-jit-trace/build.rs`'s
    // `collect_rs_files` walk over `pyre-object/src` +
    // `pyre-interpreter/src` so the test sees the same whole-program
    // visibility production runs with.
    let program_inputs = collect_pyre_interpreter_program_inputs();
    let program = build_semantic_program_from_parsed_files(&program_inputs)
        .expect("pyre-interpreter source must lower without FlowingError");
    let mut impls = Vec::new();
    impls.extend(
        extract_trait_impls(
            &pyopcode,
            &program.struct_fields,
            &program.fn_return_types,
            &program.known_struct_names,
            None,
        )
        .expect("pyopcode trait impls must lower"),
    );
    impls.extend(
        extract_trait_impls(
            &eval,
            &program.struct_fields,
            &program.fn_return_types,
            &program.known_struct_names,
            None,
        )
        .expect("eval trait impls must lower"),
    );
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
    let dispatch_sf =
        build_function_graph_pub(dispatch_fn).expect("dispatch_fn must lower without FlowingError");
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

    // Deep-BFS oracle (Phase M5).
    //
    // RPython `call.py:77-90 find_all_graphs` BFS visits every callee of
    // every candidate graph and inserts the callee into `candidate_graphs`
    // when `policy.look_inside_graph` accepts.  Two invariants follow for
    // pyre's closed-world dispatch:
    //
    // 1. Every `opcode_*` helper is a candidate after BFS — because the
    //    portal `execute_opcode_step` direct-calls each helper via the
    //    generated dispatch table.  This is the strict shape of
    //    `call.py:97 graphs_from(op)` returning `[funcobj.graph]` for a
    //    direct_call site: the callee graph MUST be reachable.
    //
    // 2. For every `OpKind::Call` site inside a candidate opcode helper
    //    whose target is `CallTarget::Method`, if the method name has a
    //    registered `PyFrame::<name>` impl, that impl must be a candidate.
    //    The closed-world set is OpcodeStepExecutor → only PyFrame, so the
    //    BFS-internal `target_to_path` lookup (`call.rs:3007-3056`)
    //    converges on `PyFrame::<name>` whether the resolver picks the
    //    receiver-root or trait-wildcard branch.
    for name in &helper_names {
        let p = CallPath::from_segments([name.clone()]);
        assert!(
            cc.is_candidate(&p),
            "find_all_graphs did not reach opcode helper `{name}` from portal"
        );
    }

    // `policy.py:71-83 look_inside_graph` rejects callees whose graphs
    // carry backedges — pyre's `DefaultJitPolicy` mirrors this at
    // `jit_codewriter/policy.rs:166-171`.  Excluding loopy PyFrame
    // graphs from the oracle is the strict parity stance: an opcode
    // helper that calls into a loopy PyFrame method still reaches the
    // method *as a residual call*, not as a BFS candidate, and that
    // matches upstream behaviour for `@dont_look_inside`-equivalent
    // sites.  Without this filter the assertion would fire on any
    // PyFrame method whose implementation contains a Rust `for` /
    // `while` / `loop` (e.g. `ensure_iter_value` at
    // `pyre-interpreter/src/eval.rs:1257`), which is not a BFS gap.
    let mut missing: Vec<(String, String)> = Vec::new();
    for helper_name in &helper_names {
        let helper_path = CallPath::from_segments([helper_name.clone()]);
        let helper_graph = cc
            .function_graphs()
            .get(&helper_path)
            .expect("helper graph registered above");
        for block in &helper_graph.blocks {
            for op in &block.operations {
                let OpKind::Call {
                    target: CallTarget::Method { name: m_name, .. },
                    ..
                } = &op.kind
                else {
                    continue;
                };
                let pyframe_path = CallPath::for_impl_method("PyFrame", m_name);
                let Some(pyframe_graph) = cc.function_graphs().get(&pyframe_path) else {
                    continue;
                };
                if !majit_translate::policy::find_backedges(pyframe_graph).is_empty() {
                    continue;
                }
                if !cc.is_candidate(&pyframe_path) {
                    missing.push((helper_name.clone(), m_name.clone()));
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "find_all_graphs missed loop-free PyFrame methods referenced from opcode helpers: {missing:?}"
    );

    // Diagnostic summary — keep `--nocapture` readable.
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
        "[phase-d.1] portal=execute_opcode_step helpers_registered={} \
         helpers_in_candidates={} pyframe_methods_registered={} \
         candidates={} pyframe_in_candidates={}",
        helper_names.len(),
        helper_names.len(),
        pyframe_method_paths.len(),
        candidate_count,
        pyframe_candidates.len(),
    );
}
