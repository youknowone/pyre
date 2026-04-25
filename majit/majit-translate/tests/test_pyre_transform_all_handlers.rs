//! Phase E.0.1: exercise `CodeWriter::transform_graph_to_jitcode` on every
//! pyre-interpreter `opcode_*` freestanding handler.
//!
//! This is Phase E.0 (single handler) scaled across the ~28 handlers.
//! Each handler is transformed independently through the full pipeline:
//!
//!     pyopcode.rs → front::ast → FunctionGraph
//!                 → CallControl (register + get_jitcode)
//!                 → CodeWriter::transform_graph_to_jitcode
//!                 → SSARepr + Arc<JitCode> body
//!
//! RPython parity point: `rpython/jit/codewriter/codewriter.py:74
//! CodeWriter.make_jitcodes` iterates `callcontrol.enum_pending_graphs()`
//! and calls `transform_graph_to_jitcode(graph, jitcode, verbose, idx)`
//! on each (graph, jitcode) pair.  We exercise the per-graph transform
//! step directly, matching the RPython test-harness idiom at
//! `rpython/jit/codewriter/codewriter.py:25 transform_func_to_jitcode`
//! (explicitly marked "For testing").
//!
//! Per-handler results are collected so failures become visible without
//! stopping the run.  The assertion is calibrated to the known-good
//! count; regressions against that count fail the test, and new
//! handler-specific gaps surface as `FAIL` rows in the matrix for
//! later phases to address.
//!
//! Each handler runs against its own fresh `CallControl` so cross-handler
//! state cannot leak.  The trait-impl registration mirrors Phase E.0.
//!
//! RPython `jtransform.py`/`flatten.py`/`regalloc.py`/`liveness.py`/
//! `assembler.py` are NOT modified — the test asserts that the existing
//! pipeline handles the Rust-source-derived graphs without per-arm
//! special cases, which is the structural claim of the epic.
//!
//! The test is READ-ONLY against `pyre-interpreter/src/` and does not
//! touch pyre-jit; baseline `./pyre/check.sh` 14/14 remains unaffected.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::Arc;

use majit_translate::codewriter::CodeWriter;
use majit_translate::{
    CallPath, ParsedInterpreter,
    call::CallControl,
    extract_trait_impls,
    front::{StructFieldRegistry, ast::build_function_graph_pub},
    jitcode::JitCode,
    jtransform::GraphTransformConfig,
    parse_source,
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

struct HandlerResult {
    name: String,
    ok: bool,
    ssarepr_len: Option<usize>,
    body_len: Option<usize>,
    error: Option<String>,
}

#[test]
fn transform_all_handlers_to_jitcode() {
    let pyopcode = parse_pyre_file("pyre/pyre-interpreter/src/pyopcode.rs");
    let eval = parse_pyre_file("pyre/pyre-interpreter/src/eval.rs");

    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = std::collections::HashMap::new();
    let empty_struct_names = std::collections::HashSet::new();

    let handlers: Vec<&ItemFn> = iter_opcode_handler_fns(&pyopcode.file).collect();
    assert!(
        handlers.len() >= 25,
        "expected at least 25 opcode_* handlers in pyopcode.rs, got {}",
        handlers.len()
    );

    // Pre-collect trait impls once — they do not change across handlers.
    let mut impls = Vec::new();
    impls.extend(
        extract_trait_impls(
            &pyopcode,
            &empty_registry,
            &empty_fn_ret,
            &empty_struct_names,
        )
        .expect("pyopcode trait impls must lower"),
    );
    impls.extend(
        extract_trait_impls(&eval, &empty_registry, &empty_fn_ret, &empty_struct_names)
            .expect("eval trait impls must lower"),
    );

    let mut results: Vec<HandlerResult> = Vec::new();

    for handler in &handlers {
        let name = handler.sig.ident.to_string();
        let sf =
            build_function_graph_pub(handler).expect("handler must lower without FlowingError");
        let path = CallPath::from_segments([sf.name.clone()]);

        let mut cc = CallControl::new();
        cc.register_function_graph(path.clone(), sf.graph.clone());
        for imp in &impls {
            for method in &imp.methods {
                if let Some(graph) = method.graph.clone() {
                    cc.register_trait_method(
                        &method.name,
                        Some(&imp.trait_name),
                        &imp.for_type,
                        graph,
                    );
                }
            }
        }

        let jitcode: Arc<JitCode> = cc.get_jitcode(&path);
        let graph_for_transform = sf.graph.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut cw = CodeWriter::new();
            let config = GraphTransformConfig::default();
            // RPython `codewriter.py:33 transform_graph_to_jitcode(self,
            // graph, jitcode, verbose, index)` mutates the JitCode in
            // place and returns None. Tests read `jitcode.body()._ssarepr`
            // afterwards (assembler.py:49 `jitcode._ssarepr = ssarepr`
            // stores it on the body for dump/diagnostic access).
            let idx = cc.finished_jitcodes_len();
            cw.transform_graph_to_jitcode(
                &graph_for_transform,
                &path,
                &mut cc,
                &config,
                &jitcode,
                /* verbose = */ false,
                idx,
            );
            let body = jitcode.body();
            let ssarepr_len = body._ssarepr.as_ref().map(|s| s.insns.len()).unwrap_or(0);
            (ssarepr_len, body.code.len())
        }));

        match outcome {
            Ok((ssarepr_len, body_len)) => results.push(HandlerResult {
                name,
                ok: true,
                ssarepr_len: Some(ssarepr_len),
                body_len: Some(body_len),
                error: None,
            }),
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<unknown panic>".to_string()
                };
                results.push(HandlerResult {
                    name,
                    ok: false,
                    ssarepr_len: None,
                    body_len: None,
                    error: Some(msg),
                });
            }
        }
    }

    let total = results.len();
    let ok_count = results.iter().filter(|r| r.ok).count();
    eprintln!(
        "[phase-e.0.1] {}/{} handlers transformed cleanly",
        ok_count, total
    );
    for r in &results {
        if r.ok {
            eprintln!(
                "[phase-e.0.1]   OK   {:48} ssarepr={:4} body={:5}",
                r.name,
                r.ssarepr_len.unwrap(),
                r.body_len.unwrap()
            );
        } else {
            eprintln!(
                "[phase-e.0.1]   FAIL {:48} {}",
                r.name,
                r.error.as_deref().unwrap_or("")
            );
        }
    }

    // Regression guard: first green run transformed all 28 handlers.
    // Any drop indicates a regression in front::ast, CallControl
    // trait resolution, jtransform, or a handler body pattern that
    // majit-translate no longer covers.
    assert_eq!(
        ok_count, total,
        "handler transform matrix regressed: {ok_count}/{total} clean"
    );
    assert!(
        total >= 28,
        "expected >=28 opcode_* handlers, got {total} (pyopcode.rs lost some?)"
    );
}
