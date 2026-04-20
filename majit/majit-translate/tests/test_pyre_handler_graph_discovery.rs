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
}

fn lower_handler(func: &syn::ItemFn) -> HandlerGraphStats {
    let sf = build_function_graph_pub(func);
    let blocks = sf.graph.blocks.len();
    let ops: usize = sf.graph.blocks.iter().map(|b| b.operations.len()).sum();
    HandlerGraphStats {
        name: sf.name,
        blocks,
        ops,
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
    eprintln!("[phase-a.1]   {:<40} {:>7} {:>7}", "name", "blocks", "ops");
    for entry in &matrix {
        eprintln!(
            "[phase-a.1]   {:<40} {:>7} {:>7}",
            entry.name, entry.blocks, entry.ops
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
        let found = matrix.iter().any(|m| m.name == required);
        assert!(
            found,
            "super-instruction helper `{}` missing from pyopcode.rs — the \
             epic's premise (each super-inst expressed as an atomic-op chain) \
             no longer holds",
            required
        );
    }
}
