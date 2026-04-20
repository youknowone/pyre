//! Phase A.1 harness: feed every `opcode_*` freestanding handler in
//! `pyre-interpreter/src/pyopcode.rs` through the existing Rust-source-to-
//! FunctionGraph path (`front::ast::build_function_graph_pub`).
//!
//! Positioning: this harness sits in the PRE-EXISTING-ADAPTATION layer —
//! RPython does not parse interpreter *source* to build graphs, its rtyper
//! produces `translator.graphs` before codewriter ever runs. Here pyre
//! bridges Rust source to the same `FunctionGraph` object that RPython
//! assumes exists. See `src/front/README.md` for the boundary.
//!
//! Output: for every handler the test prints `(name, blocks, ops, status)`
//! so the matrix is readable. A panic anywhere fails the test — that is
//! the discovery signal telling us which Rust patterns Phase B/C still
//! need to cover.
//!
//! Intentionally lax assertions: we only insist that a core set of super-
//! instruction helpers (`opcode_load_fast_load_fast`,
//! `opcode_load_fast_pair_checked`, `opcode_store_fast_load_fast`,
//! `opcode_store_fast_store_fast`) lower to at least one block. Any
//! stricter shape check belongs to later phases.

use std::path::PathBuf;

use majit_translate::front::ast::build_function_graph_pub;
use majit_translate::model::ExitSwitch;
use syn::{File, Item};

fn pyopcode_rs_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("..");
    p.push("..");
    p.push("pyre");
    p.push("pyre-interpreter");
    p.push("src");
    p.push("pyopcode.rs");
    p
}

fn parse_pyopcode_rs() -> File {
    let path = pyopcode_rs_path();
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    syn::parse_file(&src).unwrap_or_else(|e| panic!("failed to parse {}: {}", path.display(), e))
}

fn iter_opcode_handler_fns(file: &File) -> impl Iterator<Item = &syn::ItemFn> {
    file.items.iter().filter_map(|item| match item {
        Item::Fn(func) if func.sig.ident.to_string().starts_with("opcode_") => Some(func),
        _ => None,
    })
}

#[derive(Debug)]
struct HandlerGraphStats {
    name: String,
    blocks: usize,
    ops: usize,
    /// Number of blocks whose `exitswitch` is `LastException` — the
    /// direct port of RPython `Block.canraise`
    /// (`flowspace/model.py:214`). One per Rust `?` operator in the
    /// handler body.
    canraise_blocks: usize,
    exception_links_well_formed: usize,
    /// RPython parity: `FunctionGraph.exceptblock` has two inputargs,
    /// `(etype, evalue)` (`flowspace/model.py:21-25`).
    exception_block_arity: usize,
}

fn lower_handler(func: &syn::ItemFn) -> HandlerGraphStats {
    let sf = build_function_graph_pub(func);
    let blocks = sf.graph.blocks.len();
    let ops: usize = sf.graph.blocks.iter().map(|b| b.operations.len()).sum();
    let canraise_blocks = sf.graph.blocks.iter().filter(|b| b.canraise()).count();
    let exception_links_well_formed = sf
        .graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .filter(|b| {
            b.exits.len() == 2
                && b.exits[0].exitcase.is_none()
                && b.exits[1].catches_all_exceptions()
                && b.exits[1].last_exception.is_some()
                && b.exits[1].last_exc_value.is_some()
        })
        .count();
    let exception_block_arity = sf.graph.block(sf.graph.exceptblock).inputargs.len();
    HandlerGraphStats {
        name: sf.name,
        blocks,
        ops,
        canraise_blocks,
        exception_links_well_formed,
        exception_block_arity,
    }
}

#[test]
fn discover_pyre_opcode_handler_graphs() {
    let file = parse_pyopcode_rs();
    let handlers: Vec<&syn::ItemFn> = iter_opcode_handler_fns(&file).collect();
    assert!(
        !handlers.is_empty(),
        "expected at least one `fn opcode_*` in pyre-interpreter/src/pyopcode.rs"
    );

    let mut matrix: Vec<HandlerGraphStats> = Vec::with_capacity(handlers.len());
    for func in &handlers {
        let stats = lower_handler(func);
        matrix.push(stats);
    }

    // Emit the matrix so the phase reviewer can scan it from `cargo test --nocapture`.
    eprintln!("[phase-a.1] pyre-interpreter opcode handler lowering matrix:");
    eprintln!(
        "[phase-a.1]   {:<40} {:>7} {:>7} {:>10} {:>10} {:>10}",
        "name", "blocks", "ops", "canraise", "exc_links", "exc_arity"
    );
    for entry in &matrix {
        eprintln!(
            "[phase-a.1]   {:<40} {:>7} {:>7} {:>10} {:>10} {:>10}",
            entry.name,
            entry.blocks,
            entry.ops,
            entry.canraise_blocks,
            entry.exception_links_well_formed,
            entry.exception_block_arity,
        );
    }

    // Every lowering must produce a graph with at least one block.
    for entry in &matrix {
        assert!(
            entry.blocks >= 1,
            "{} lowered to zero blocks — front::ast produced an empty graph",
            entry.name
        );
    }

    // The core super-instruction helpers are the reason this epic exists.
    // If any of these cannot lower, super-inst parity cannot be closed.
    let required_super_inst_helpers = [
        "opcode_load_fast_load_fast",
        "opcode_load_fast_pair_checked",
        "opcode_store_fast_load_fast",
        "opcode_store_fast_store_fast",
    ];
    for required in required_super_inst_helpers {
        let entry = matrix
            .iter()
            .find(|m| m.name == required)
            .unwrap_or_else(|| {
                panic!(
                    "super-instruction helper `{}` missing from pyopcode.rs — the \
                     epic's premise (each super-inst expressed as an atomic-op chain) \
                     no longer holds",
                    required
                )
            });
        // Every super-instruction helper uses `?` on each trait method call
        // (load_local_value / push_value / pop_value / store_local_value),
        // so the graph must contain can-raise blocks (one per `?`) plus
        // a shared exception block. RPython parity:
        // `flowspace/model.py:214 Block.canraise` + the "raise block" at
        // `model.py:198`.  A regression here means `Expr::Try` lowering
        // silently dropped the exception edge, violating RPython parity in
        // `jtransform.py:456 rewrite_op_direct_call` (which needs the
        // CFG-level exception exit to emit `residual_call_*` + `-live-`).
        assert!(
            entry.canraise_blocks > 0,
            "{} has zero canraise blocks — `?` lowering lost the exception edge",
            entry.name
        );
        assert_eq!(
            entry.exception_links_well_formed, entry.canraise_blocks,
            "{} has malformed can-raise Link metadata",
            entry.name
        );
        assert_eq!(
            entry.exception_block_arity, 2,
            "{} exception block must mirror RPython exceptblock arity `(etype, evalue)`",
            entry.name
        );
    }
}
