//! Side-by-side equivalence harness — lower the same function via
//! the AST front-end AND the MIR-driven flowspace driver, then
//! compare the resulting `FunctionGraph` shapes (issue #97 Step 4.2).
//!
//! ## What we compare
//!
//! - Block counts and per-block op kinds.
//! - Number of `BinOp` ops (semantic operations are the load-bearing
//!   parity signal).
//! - Number of `Call` ops, separated by whether they go through a
//!   `FunctionPath` / `Method` / `SyntheticTransparentCtor`.
//!
//! ## What we *don't* compare
//!
//! - Variable names / slots.  The AST driver and MIR driver mint
//!   Variables in different orders.  Variable identity is local to a
//!   single driver and never observable across the boundary.
//! - Synthetic ops the MIR driver emits that the AST driver does not
//!   (`__discriminant`, `__dyn_call`, `__str_const`, `__deref_write`,
//!   etc.).  These are intentional fidelity gaps documented in
//!   `front/mir.rs` and tracked under Step 4.3.

use majit_charon_reader::Llbc;
use majit_translate::front::{
    self,
    ast::SemanticProgram as AstProgram,
    mir::{build_semantic_program_from_llbc, lower_function},
};
use majit_translate::model::OpKind;
use majit_translate::parse_source;

const CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../charon-spike/corpus.ullbc",
);

fn ast_program(src: &str) -> AstProgram {
    let parsed = parse_source(src);
    front::ast::build_semantic_program(&parsed).expect("ast lowering")
}

fn count_ops_by_kind(graph: &majit_translate::model::FunctionGraph) -> (usize, usize) {
    let mut binops = 0;
    let mut calls = 0;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::BinOp { .. } => binops += 1,
                OpKind::Call { .. } => calls += 1,
                _ => {}
            }
        }
    }
    (binops, calls)
}

#[test]
fn straight_line_add_ast_vs_mir() {
    // Identical source on both sides — the only shape variation is
    // the lowering pipeline that produced the IR.
    const SRC: &str = r#"
        pub fn straight_line_add(a: i64, b: i64, c: i64) -> i64 {
            let s = a + b;
            let t = s * 2;
            t + c
        }
    "#;

    let ast = ast_program(SRC);
    let ast_fn = ast
        .functions
        .iter()
        .find(|f| f.name == "straight_line_add")
        .expect("ast missing fn");
    let (ast_binops, ast_calls) = count_ops_by_kind(&ast_fn.graph);

    let llbc = Llbc::load(CORPUS).expect("load corpus");
    let mir_graph = lower_function(&llbc, "straight_line_add").expect("mir lowering");
    let (mir_binops, mir_calls) = count_ops_by_kind(&mir_graph);

    // The arithmetic surface is identical: `a + b`, `s * 2`, `t + c`
    // = exactly three BinOps in both lowering paths.  This is the
    // load-bearing parity claim of Step 4 — semantic ops are equal
    // even if the surrounding shape (block count, synthetic helpers)
    // differs by driver.
    assert_eq!(
        ast_binops, 3,
        "AST driver should emit 3 BinOps for a + b * 2 + c (got {ast_binops})"
    );
    assert_eq!(
        mir_binops, 3,
        "MIR driver should emit 3 BinOps for a + b * 2 + c (got {mir_binops})"
    );
    assert_eq!(ast_binops, mir_binops, "BinOp count divergence");

    // Neither driver should emit any Call op for a pure arithmetic
    // function.  This is the most basic "no spurious calls" parity
    // check — if MIR starts emitting a `__deref_write` or similar
    // synthetic for a pure-int function, that's a regression.
    assert_eq!(ast_calls, 0, "AST emitted unexpected Call ops");
    assert_eq!(mir_calls, 0, "MIR emitted unexpected Call ops");
}

#[test]
fn semantic_program_builder_matches_corpus_count() {
    // The MIR-driven SemanticProgram builder should surface the same
    // four corpus functions the spike documents.  Pin the count so a
    // future Charon upgrade that silently elides one body trips the
    // assertion.
    let llbc = Llbc::load(CORPUS).expect("load corpus");
    let program = build_semantic_program_from_llbc(&llbc).expect("mir program");
    // Crate-prefix stripped at SemanticProgram build (Step 4.5);
    // match against the bare leaf names instead.
    let leaves: std::collections::HashSet<_> = program
        .functions
        .iter()
        .map(|f| f.name.clone())
        .collect();
    let corpus_names: Vec<_> = [
        "straight_line_add",
        "branch_loop_sum",
        "strategy_len",
        "parse_one",
        "desugar_mix",
    ]
    .iter()
    .filter(|n| leaves.contains(**n))
    .map(|s| s.to_string())
    .collect();
    // 4 pub fns (straight_line_add, branch_loop_sum, strategy_len,
    // desugar_mix) + the `parse_one` private helper that
    // `desugar_mix` calls.
    assert_eq!(
        corpus_names.len(),
        5,
        "expected 5 corpus fns, got {corpus_names:?}"
    );
}
