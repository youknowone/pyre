//! End-to-end smoke tests for the MIR-driven flowspace driver.
//!
//! The corpus snapshot at `majit/charon-corpus/corpus.ullbc` is the
//! input and the regression fixture for the production MIR frontend.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, build_semantic_program_from_llbc, lower_function};

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc",);

fn load_corpus() -> Llbc {
    Llbc::load(CORPUS).expect("load corpus.ullbc")
}

#[test]
fn lowers_straight_line_add() {
    let llbc = load_corpus();
    let graph = lower_function(&llbc, "straight_line_add").expect("lowering");
    // FunctionGraph.name keeps the full Charon-qualified path
    // because it identifies the LLBC source — only the
    // SemanticFunction.name has the crate-prefix stripping applied
    // at SemanticProgram build time.
    assert_eq!(graph.name, "charon_corpus::straight_line_add");

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
    // `branch_loop_sum` exercises three surfaces together: `Call`
    // terminators (`slice.iter()` / `Iterator::next`), `Drop`
    // terminators, and `Rvalue::Discriminant` on the iterator's
    // `Option<&i64>` step result.
    let llbc = load_corpus();
    let graph = lower_function(&llbc, "branch_loop_sum").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::branch_loop_sum");

    use majit_translate::model::{CallTarget, OpKind};
    let mut call_count = 0usize;
    let mut discr_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                // An Abort terminator lowers the `exc_from_raise` op
                // pair (`simple_call(const(exc_class))` + `type(evalue)`)
                // into its block; exclude those raise-machinery ops so
                // the count characterizes the body's own calls.
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if matches!(
                    segments.first().map(String::as_str),
                    Some("simple_call" | "type")
                ) => {}
                OpKind::Call { .. } => call_count += 1,
                OpKind::FieldRead { field, .. } if field.name == "__discriminant" => {
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
    assert_eq!(call_count, 2, "expected 2 body Call ops");
    assert_eq!(
        discr_count, 1,
        "expected 1 __discriminant FieldRead for the Option step"
    );
}

#[test]
fn lowers_strategy_len_with_discriminant_switch() {
    let llbc = load_corpus();
    let graph = lower_function(&llbc, "strategy_len").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::strategy_len");
    // bb0 Discriminant + Switch, bb1/bb2/bb3 arm bodies + Return,
    // bb4 Abort → 5 MIR bbs + returnblock + exceptblock = 7.
    assert_eq!(graph.blocks.len(), 7);
}

#[test]
fn lowers_desugar_mix_with_aggregate_and_question_mark() {
    // `desugar_mix` exercises every surface the corpus carries: `?`
    // desugaring (Call + Match + Discriminant on `Result`), enum
    // construction (`Rvalue::Aggregate` for `PyResult::Ok`), iterator
    // calls, and `break`.
    let llbc = load_corpus();
    let graph = lower_function(&llbc, "desugar_mix").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::desugar_mix");

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
fn lowers_tuple_roundtrip_with_symmetric_positional_field_reads() {
    // `tuple_roundtrip` constructs a real tuple `(a + b, a - b)` and
    // reads both `.0` / `.1` in the same function.  The lowering must
    // emit a `FieldRead __pos_<idx>` for those reads — symmetric with
    // the construction-side `FieldWrite __pos_<idx>` chain and carrying
    // the *same* `owner_root` — rather than collapsing every `.N` to
    // the synthetic-ctor base Variable.
    //
    // The same function also exercises the case that MUST still
    // collapse: each `a + b` / `a - b` / `pair.0 * pair.1` lowers
    // through a `*Checked` `(value, bool)` `BinaryOp`, whose `.0` reads
    // are `Field` projections of a `(i64, bool)` local.  Those locals
    // are bound by `Rvalue::BinaryOp`, never an `Aggregate`, so they
    // are absent from `positional_aggregate_locals` and their `.0`
    // reads fall through.  Asserting the FieldRead count is exactly the
    // two genuine tuple reads (not five) pins that boundary.
    use majit_translate::model::{CallTarget, OpKind};

    let llbc = load_corpus();
    let graph = lower_function(&llbc, "tuple_roundtrip").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::tuple_roundtrip");

    let mut field_reads: Vec<(String, Option<String>)> = Vec::new();
    let mut field_writes: Vec<(String, Option<String>)> = Vec::new();
    let mut ctor_count = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::FieldRead { field, .. } => {
                    field_reads.push((field.name.clone(), field.owner_root.clone()));
                }
                OpKind::FieldWrite { field, .. } => {
                    field_writes.push((field.name.clone(), field.owner_root.clone()));
                }
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                } => ctor_count += 1,
                _ => {}
            }
        }
    }

    // Exactly one synthetic ctor (the genuine tuple) and its two-field
    // `__pos_0` / `__pos_1` construction chain.
    assert_eq!(ctor_count, 1, "expected one tuple SyntheticTransparentCtor");
    field_writes.sort();
    assert_eq!(
        field_writes,
        vec![
            ("__pos_0".to_string(), Some("Tuple".to_string())),
            ("__pos_1".to_string(), Some("Tuple".to_string())),
        ],
        "tuple construction must emit a __pos_0 / __pos_1 FieldWrite chain"
    );

    // Exactly the two genuine tuple reads become FieldReads. The three
    // `*Checked` `.0` reads collapse, so a count of 2 (not 5) proves the
    // boundary holds.
    field_reads.sort();
    assert_eq!(
        field_reads,
        vec![
            ("__pos_0".to_string(), Some("Tuple".to_string())),
            ("__pos_1".to_string(), Some("Tuple".to_string())),
        ],
        "tuple reads must emit __pos_0 / __pos_1 FieldReads (owner_root \
         matching the FieldWrite chain) and *Checked .0 reads must collapse"
    );

    // Symmetry: every FieldRead pairs with an identically-keyed
    // FieldWrite (same name AND owner_root), so the read resolves the
    // value the construction stored.
    assert_eq!(
        field_reads, field_writes,
        "FieldRead keys must match the FieldWrite chain exactly"
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
    // Building a SemanticProgram from the corpus.ullbc should succeed
    // and surface every local function as a SemanticFunction with a
    // populated FunctionGraph.
    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(&llbc).expect("builder");
    assert!(
        program.functions.len() >= 4,
        "expected at least the 4 corpus shapes, got {}",
        program.functions.len()
    );
    let names: std::collections::HashSet<_> =
        program.functions.iter().map(|f| f.name.as_str()).collect();
    // Names are crate-prefix-stripped (lib.rs:444
    // register_function_graph_alias walks bare leaf + crate aliases
    // off this shape).
    for required in [
        "straight_line_add",
        "branch_loop_sum",
        "strategy_len",
        "desugar_mix",
    ] {
        assert!(names.contains(required), "missing {required}");
    }
    // The corpus declares one struct-shaped enum (Strategy + Token),
    // one type alias (PyResult), so we expect Strategy/Token and their
    // variant paths plus the leaf names.
    assert!(
        program.known_struct_names.contains("Strategy"),
        "expected Strategy in known_struct_names, got {:?}",
        program.known_struct_names
    );
    assert!(
        program
            .known_struct_names
            .contains("charon_corpus::Strategy::IntKeyed")
    );
    assert!(program.known_struct_names.contains("Token"));
}

#[test]
fn enum_variant_by_discriminant_round_trips_against_variant_paths() {
    // The discriminant→variant-name map must parse Charon's
    // `{"Scalar":{"Signed"|"Unsigned":[w,"K"]}}` discriminants and key
    // each enum under both its qualified path and bare leaf. Validate
    // against the corpus' Strategy enum without hard-coding variant
    // counts: every name the map produced must have a matching
    // `Strategy::<name>` variant path in known_struct_names, and the
    // leaf key must mirror the qualified key.
    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(&llbc).expect("builder");

    let by_leaf = program
        .enum_variant_by_discriminant
        .get("Strategy")
        .expect("Strategy discriminant map present under bare leaf");
    assert!(!by_leaf.is_empty(), "Strategy must carry discriminants");

    // Discriminant 0 .. N-1 are distinct (HashMap keys) and every value
    // names a real Strategy variant.
    for name in by_leaf.values() {
        let path = format!("charon_corpus::Strategy::{name}");
        assert!(
            program.known_struct_names.contains(&path),
            "discriminant map produced {name:?} with no matching variant path {path:?}"
        );
    }
    // At least the variant the sibling test pins must round-trip.
    assert!(
        by_leaf.values().any(|n| n == "IntKeyed"),
        "expected IntKeyed among Strategy discriminants, got {by_leaf:?}"
    );

    // Qualified-path key mirrors the bare-leaf key.
    let by_qualified = program
        .enum_variant_by_discriminant
        .get("charon_corpus::Strategy")
        .expect("Strategy discriminant map present under qualified path");
    assert_eq!(by_leaf, by_qualified, "leaf and qualified maps must match");
}

#[test]
fn front_graph_carries_no_synthesized_exception_edges() {
    // The MIR driver drops every Call / Assert / Drop `on_unwind`
    // successor (a Rust panic-cleanup path) and routes only to the
    // success continuation, because Python exceptions ride the
    // `Result<_, PyError>` Switch/Return edges as ordinary control flow —
    // never a Rust unwind. Lock that structurally on the FRONT flow graph
    // (NOT the jitcode, where can-raise is re-derived op-locally as
    // guard_no_exception and is orthogonally correct):
    //
    //   A. No lowered block carries a `LastException` exitswitch — the
    //      driver never synthesizes a typed try/except handler dispatch.
    //   B. Every edge into the canonical exceptblock is a bare
    //      panic-propagation raise (`UnwindResume` / `Abort` -> set_raise),
    //      so the count of blocks linking to the exceptblock equals the
    //      count of `UnwindResume` / `Abort` MIR terminators. A Call /
    //      Assert / Drop success block contributes zero such edges.
    use majit_charon_reader::ullbc::{TermKind, Unstructured};
    use majit_translate::front::mir::lower_fun_decl;
    use majit_translate::model::ExitSwitch;

    let llbc = load_corpus();
    let mut checked = 0usize;
    for fd in llbc.iter_local_fns() {
        let Some(body): Option<Unstructured> = fd.unstructured() else {
            continue;
        };
        let graph = lower_fun_decl(&llbc, fd)
            .unwrap_or_else(|e| panic!("{} failed to lower: {e}", fd.item_meta.name_path()));

        // Invariant A.
        for b in &graph.blocks {
            assert!(
                b.exitswitch != Some(ExitSwitch::LastException),
                "{}: block {:?} carries a LastException exitswitch — a typed \
                 exception-handler edge was synthesized; the MIR driver must \
                 drop on_unwind, not lower it as try/except",
                graph.name,
                b.id,
            );
        }

        // Invariant B.
        let raises_in_mir = body
            .body
            .iter()
            .filter(|blk| {
                matches!(
                    blk.term(),
                    Ok(TermKind::UnwindResume) | Ok(TermKind::Abort(_))
                )
            })
            .count();
        let edges_into_exceptblock = graph
            .blocks
            .iter()
            .filter(|b| b.exits.iter().any(|l| l.target == graph.exceptblock))
            .count();
        assert_eq!(
            edges_into_exceptblock, raises_in_mir,
            "{}: {} block(s) link to the exceptblock but the MIR has {} \
             UnwindResume/Abort terminator(s) — a Call/Assert/Drop on_unwind \
             edge leaked into the front graph",
            graph.name, edges_into_exceptblock, raises_in_mir,
        );
        checked += 1;
    }
    assert!(
        checked >= 4,
        "expected to lower at least the 4 corpus shapes, got {checked}",
    );
}
