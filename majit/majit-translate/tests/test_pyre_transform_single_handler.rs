//! Phase E.0: run `CodeWriter::transform_graph_to_jitcode` on a single
//! pyre-interpreter opcode handler.
//!
//! This is the narrowest end-to-end slice of the majit-translate codewriter
//! pipeline operating on real Rust source:
//!
//!     pyopcode.rs → front::ast → FunctionGraph
//!                 → CallControl (register + get_jitcode)
//!                 → CodeWriter::transform_graph_to_jitcode
//!                 →   annotate + rtype
//!                 →   jtransform
//!                 →   regalloc
//!                 →   flatten + liveness
//!                 →   assemble
//!                 → SSARepr + Arc<JitCode> body
//!
//! RPython parity point: `rpython/jit/codewriter/codewriter.py:33
//! transform_func_to_jitcode` — the same 5-step pipeline applied to one
//! function graph at a time.
//!
//! If this test panics we know where the pyre/Rust adaptation breaks the
//! pipeline. If it passes we have proof that the existing majit-translate
//! codewriter handles pyre-interpreter opcode handlers without further
//! modification — the thesis of the epic.

use std::path::PathBuf;
use std::sync::Arc;

use majit_translate::codewriter::CodeWriter;
use majit_translate::{
    CallPath, ParsedInterpreter, build_semantic_program_from_parsed_files, call::CallControl,
    extract_trait_impls, flatten::FlatOp, front::ast::build_function_graph_pub, jitcode::JitCode,
    jtransform::GraphTransformConfig, model::ExitSwitch, parse_source,
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
/// under `pyre-object/src` and `pyre-interpreter/src` so
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

fn find_opcode_handler<'a>(file: &'a syn::File, name: &str) -> Option<&'a ItemFn> {
    file.items.iter().find_map(|item| match item {
        Item::Fn(func) if func.sig.ident == name => Some(func),
        _ => None,
    })
}

#[test]
fn transform_opcode_load_fast_load_fast_to_jitcode() {
    // The super-instruction helper this test targets is the simplest
    // handler that still exercises two trait method types
    // (LocalOpcodeHandler::load_local_value + SharedOpcodeHandler::push_value)
    // and uses `?` propagation. If the pipeline can turn this into a
    // JitCode, every simpler handler will too.
    let pyopcode = parse_pyre_file("pyre/pyre-interpreter/src/pyopcode.rs");
    let eval = parse_pyre_file("pyre/pyre-interpreter/src/eval.rs");

    let handler = find_opcode_handler(&pyopcode.file, "opcode_load_fast_load_fast")
        .expect("opcode_load_fast_load_fast is present in pyopcode.rs");
    let sf = build_function_graph_pub(handler).expect("handler must lower without FlowingError");
    assert_eq!(
        sf.graph.block(sf.graph.exceptblock).inputargs.len(),
        2,
        "exception block must mirror RPython exceptblock arity `(etype, evalue)`"
    );
    let canraise_blocks: Vec<_> = sf.graph.blocks.iter().filter(|b| b.canraise()).collect();
    assert!(
        !canraise_blocks.is_empty(),
        "opcode_load_fast_load_fast should lower `?` to can-raise blocks"
    );
    for block in canraise_blocks {
        assert_eq!(block.exitswitch, Some(ExitSwitch::LastException));
        assert_eq!(block.exits.len(), 2);
        assert_eq!(block.exits[0].exitcase, None);
        assert!(block.exits[1].catches_all_exceptions());
        assert!(block.exits[1].last_exception.is_some());
        assert!(block.exits[1].last_exc_value.is_some());
    }
    let path = CallPath::from_segments([sf.name.clone()]);

    // Seed a CallControl with the target graph plus every PyFrame trait
    // impl method it may reach. This matches the RPython pipeline where
    // the rtyper has already produced every candidate graph before
    // make_jitcodes iterates.
    let mut cc = CallControl::new();
    cc.register_function_graph(path.clone(), sf.graph.clone());

    // Production at `lib.rs:317-342` populates `program.fn_return_types`
    // / `struct_fields` / `known_struct_names` via
    // `build_semantic_program_from_parsed_files` BEFORE
    // `extract_trait_impls` so the impl-body lowering can resolve
    // user-defined method-return types
    // (`bookkeeper.getdesc(...).find_method` upstream parity at
    // `unaryop.py:206-213`).
    let parsed_files = collect_pyre_interpreter_program_inputs();
    let program = build_semantic_program_from_parsed_files(&parsed_files)
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

    // RPython `call.py:grab_initial_jitcodes` + friends allocate a JitCode
    // shell per graph and hold it in `callcontrol.jitcodes`. We mimic that
    // directly for the one graph under test.
    let jitcode: Arc<JitCode> = cc.get_jitcode(&path);

    // Run the transformer. A panic here fails the test and localizes the
    // first Rust pattern majit-translate's pipeline cannot lower.
    //
    // RPython `codewriter.py:33 transform_graph_to_jitcode(self, graph,
    // jitcode, verbose, index)` returns None. The SSARepr is stashed on
    // the jitcode body by `assembler.py:49 jitcode._ssarepr = ssarepr`
    // and read back here for inspection, matching how upstream tests
    // dig into ssarepr via `jitcode._ssarepr`.
    let mut cw = CodeWriter::new();
    let config = GraphTransformConfig::default();
    let idx = cc.finished_jitcodes_len();
    cw.transform_graph_to_jitcode(
        &sf.graph, &path, &mut cc, &config, &jitcode, /* verbose = */ false, idx,
    );
    let body = jitcode.body();
    let ssarepr = body
        ._ssarepr
        .as_ref()
        .expect("assembler must stash SSARepr on jitcode body per assembler.py:49");

    // Minimal shape check: the SSARepr must carry at least as many
    // instructions as the original graph had operations. Transform passes
    // only add markers (`-live-`, labels), they never reduce count below
    // the original op count minus one (the terminator).
    let original_ops: usize = sf.graph.blocks.iter().map(|b| b.operations.len()).sum();
    assert!(
        ssarepr.insns.len() >= original_ops.saturating_sub(1),
        "SSARepr shrank below the original op count ({} vs {}); \
         a transform pass dropped operations",
        ssarepr.insns.len(),
        original_ops
    );
    assert!(
        ssarepr
            .insns
            .iter()
            .any(|op| matches!(op, FlatOp::CatchException { .. })),
        "lowered SSA must keep the can-raise edge as catch_exception"
    );
    assert!(
        ssarepr.insns.iter().any(|op| matches!(op, FlatOp::Reraise)),
        "lowered SSA must keep the exception arm as reraise"
    );

    // The JitCode shell must have been populated with a body.
    assert!(
        !body.code.is_empty(),
        "JitCode body code was empty after transform_graph_to_jitcode"
    );

    eprintln!(
        "[phase-e.0] opcode_load_fast_load_fast → ssarepr.insns={} body.code.len={}",
        ssarepr.insns.len(),
        body.code.len(),
    );
}
