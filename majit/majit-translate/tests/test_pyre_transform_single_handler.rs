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
    CallPath, ParsedInterpreter,
    call::CallControl,
    extract_trait_impls,
    flatten::FlatOp,
    front::{StructFieldRegistry, ast::build_function_graph_pub},
    jitcode::JitCode,
    jtransform::GraphTransformConfig,
    model::ExitSwitch,
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
    let sf = build_function_graph_pub(handler);
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

    // RPython `call.py:grab_initial_jitcodes` + friends allocate a JitCode
    // shell per graph and hold it in `callcontrol.jitcodes`. We mimic that
    // directly for the one graph under test.
    let jitcode: Arc<JitCode> = cc.get_jitcode(&path);

    // Run the transformer. A panic here fails the test and localizes the
    // first Rust pattern majit-translate's pipeline cannot lower.
    let mut cw = CodeWriter::new();
    let config = GraphTransformConfig::default();
    let ssarepr = cw.transform_graph_to_jitcode(&sf.graph, &mut cc, &config, &jitcode);

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
    let body = jitcode.body();
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
