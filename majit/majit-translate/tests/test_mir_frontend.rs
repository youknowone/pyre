//! End-to-end smoke tests for the MIR-driven flowspace driver
//! (issue #97 Step 3).
//!
//! The corpus snapshot at `majit/charon-spike/corpus.ullbc` is the
//! input — the same artefact the spike's prototype consumes.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, build_semantic_program_from_llbc, lower_function};

const CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../charon-spike/corpus.ullbc",
);

fn load_corpus() -> Llbc {
    Llbc::load(CORPUS).expect("load corpus.ullbc")
}

#[test]
fn lowers_straight_line_add() {
    let llbc = load_corpus();
    let graph = lower_function(&llbc, "straight_line_add").expect("lowering");
    assert_eq!(graph.name, "charon_spike_corpus::straight_line_add");

    let startblock = graph.block(graph.startblock);
    assert_eq!(
        startblock.inputargs.len(),
        3,
        "straight_line_add takes three i64 args"
    );
    // straight_line_add has 5 MIR BBs; the FunctionGraph adds
    // startblock(0)/returnblock(1)/exceptblock(2) as canonical
    // sentinels but the MIR bb0 maps onto startblock, so the total
    // block count is 5 (MIR bbs) + 2 (returnblock + exceptblock) = 7.
    assert_eq!(
        graph.blocks.len(),
        7,
        "5 MIR bbs + returnblock + exceptblock"
    );

    // At least one of the MIR blocks should carry a BinOp operation
    // (the AddChecked / MulChecked / AddChecked sequence collapses to
    // three BinOp ops once the overflow asserts are stripped).
    use majit_translate::model::OpKind;
    let mut binop_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            if matches!(op.kind, OpKind::BinOp { .. }) {
                binop_count += 1;
            }
        }
    }
    assert_eq!(
        binop_count, 3,
        "expected 3 BinOps for the a + b * 2 + c chain"
    );
}

#[test]
fn lowers_branch_loop_sum_with_calls_and_discriminant() {
    // `branch_loop_sum` exercises three surfaces that the early
    // skeleton refused: `Call` terminators (`slice.iter()` /
    // `Iterator::next`), `Drop` terminators, and `Rvalue::Discriminant`
    // on the iterator's `Option<&i64>` step result. Each individually
    // landed in Steps 3.5 / 3.6 / 3.7 — this is the integration
    // smoke test.
    let llbc = load_corpus();
    let graph = lower_function(&llbc, "branch_loop_sum").expect("lowering");
    assert_eq!(graph.name, "charon_spike_corpus::branch_loop_sum");

    use majit_translate::model::OpKind;
    let mut call_count = 0usize;
    let mut discr_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call { .. } => call_count += 1,
                OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant" =>
                {
                    discr_count += 1
                }
                _ => {}
            }
        }
    }
    // `branch_loop_sum` calls `<[i64]>::iter` once and
    // `Iterator::next` once per loop iteration; the second call sits
    // inside the loop body so there's exactly one `Call` op for it
    // in the static IR.
    assert_eq!(call_count, 2, "expected 2 Call ops");
    assert_eq!(
        discr_count, 1,
        "expected 1 __discriminant FieldRead for the Option step"
    );
}

#[test]
fn lowers_strategy_len_with_discriminant_switch() {
    let llbc = load_corpus();
    let graph = lower_function(&llbc, "strategy_len").expect("lowering");
    assert_eq!(graph.name, "charon_spike_corpus::strategy_len");
    // bb0 Discriminant + Switch, bb1/bb2/bb3 arm bodies + Return,
    // bb4 Abort → 5 MIR bbs + returnblock + exceptblock = 7.
    assert_eq!(graph.blocks.len(), 7);
}

#[test]
fn lowers_desugar_mix_with_aggregate_and_question_mark() {
    // `desugar_mix` exercises every surface the corpus carries: `?`
    // desugaring (Call + Match + Discriminant on `Result`), enum
    // construction (`Rvalue::Aggregate` for `PyResult::Ok`), iterator
    // calls, and `break`. Landing this is the final per-shape Step 3
    // milestone for the corpus.
    let llbc = load_corpus();
    let graph = lower_function(&llbc, "desugar_mix").expect("lowering");
    assert_eq!(graph.name, "charon_spike_corpus::desugar_mix");

    use majit_translate::model::{CallTarget, OpKind};
    let mut ctor_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            if let OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            } = &op.kind
            {
                ctor_count += 1;
            }
        }
    }
    assert!(
        ctor_count >= 1,
        "expected at least one SyntheticTransparentCtor for PyResult::Ok"
    );
}

#[test]
fn unknown_function_name_errors() {
    let llbc = load_corpus();
    let err = lower_function(&llbc, "no_such_function_anywhere").unwrap_err();
    assert!(matches!(err, LowerError::FunctionNotFound(_)));
}

#[test]
fn semantic_program_builder_lowers_every_corpus_function() {
    // Step 4.1 smoke test: building a SemanticProgram from the
    // corpus.ullbc should succeed and surface every local function
    // as a SemanticFunction with a populated FunctionGraph. The
    // whole-program metadata (struct_fields etc.) is left empty by
    // design — populating it comes from Step 4.3 (Charon type_decls).
    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(&llbc).expect("builder");
    assert!(
        program.functions.len() >= 4,
        "expected at least the 4 corpus shapes, got {}",
        program.functions.len()
    );
    let names: std::collections::HashSet<_> =
        program.functions.iter().map(|f| f.name.as_str()).collect();
    for required in [
        "charon_spike_corpus::straight_line_add",
        "charon_spike_corpus::branch_loop_sum",
        "charon_spike_corpus::strategy_len",
        "charon_spike_corpus::desugar_mix",
    ] {
        assert!(names.contains(required), "missing {required}");
    }
    // Step 4.3 deliverable: these should NOT be empty after
    // type_decls derivation lands. Pin them at empty for now so the
    // first non-empty population triggers a test-update event.
    assert!(program.known_struct_names.is_empty());
    assert!(program.fn_return_types.is_empty());
}
