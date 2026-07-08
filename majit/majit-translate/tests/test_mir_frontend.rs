//! End-to-end smoke tests for the MIR-driven flowspace driver.
//!
//! The corpus snapshot at `majit/charon-corpus/corpus.ullbc` is the
//! input and the regression fixture for the production MIR frontend.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, build_semantic_program_from_llbc, lower_function};
use std::sync::OnceLock;

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc",);

/// Load `corpus.ullbc` once and share it across every test. `Llbc` is
/// read-only after `load`, so a single parse behind a `OnceLock` is
/// sufficient: `get_or_init` runs the load exactly once even under the
/// concurrent test threads, and the lowering entry points only borrow it.
fn load_corpus() -> &'static Llbc {
    static LLBC: OnceLock<Llbc> = OnceLock::new();
    LLBC.get_or_init(|| Llbc::load(CORPUS).expect("load corpus.ullbc"))
}

/// BFS the set of blocks reachable from the graph's startblock. The
/// `bool_then` / `option_question_mark` rewrites can leave the pre-split
/// framestate merge block unreachable, so the reachable-only assertions
/// filter against this set.
fn reachable_blocks(
    graph: &majit_translate::model::FunctionGraph,
) -> std::collections::HashSet<majit_translate::model::BlockId> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![graph.startblock];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        for l in &graph.block(id).exits {
            stack.push(l.target);
        }
    }
    seen
}

#[test]
fn lowers_straight_line_add() {
    let llbc = load_corpus();
    let graph = lower_function(llbc, "straight_line_add").expect("lowering");
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
    let graph = lower_function(llbc, "branch_loop_sum").expect("lowering");
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
    // `branch_loop_sum` calls `<[i64]>::iter` once (the `iter` op) and
    // `Iterator::next` once (lifted to the `[__iter_next]` op); both are
    // `Call` ops in the static IR.
    assert_eq!(call_count, 2, "expected 2 body Call ops (iter + next)");
    // The `next`-diamond rewrite (`front::iter_next`) replaces the
    // `Option` step's `__discriminant` switch with the `next` op's
    // StopIteration exception edge, so the discriminant read is consumed
    // and its (now-unreachable) block dropped.
    assert_eq!(
        discr_count, 0,
        "the Option __discriminant read is consumed by the next rewrite"
    );
}

#[test]
fn lowers_strategy_len_with_discriminant_switch() {
    let llbc = load_corpus();
    let graph = lower_function(llbc, "strategy_len").expect("lowering");
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
    let graph = lower_function(llbc, "desugar_mix").expect("lowering");
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
    let graph = lower_function(llbc, "tuple_roundtrip").expect("lowering");
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
    // `__pos_0` / `__pos_1` construction chain.  The per-shape tuple classdef
    // (default-ON) keys the owner on the tuple's element types, so the owner is
    // the suffixed `Tuple<i64,i64>` — the construction and projection sides must
    // agree on that exact spelling (the symmetry asserted below).
    assert_eq!(ctor_count, 1, "expected one tuple SyntheticTransparentCtor");
    field_writes.sort();
    assert_eq!(
        field_writes,
        vec![
            ("__pos_0".to_string(), Some("Tuple<i64,i64>".to_string())),
            ("__pos_1".to_string(), Some("Tuple<i64,i64>".to_string())),
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
            ("__pos_0".to_string(), Some("Tuple<i64,i64>".to_string())),
            ("__pos_1".to_string(), Some("Tuple<i64,i64>".to_string())),
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
    let err = lower_function(llbc, "no_such_function_anywhere").unwrap_err();
    assert!(matches!(err, LowerError::FunctionNotFound(_)));
}

#[test]
fn semantic_program_builder_lowers_every_corpus_function() {
    // Building a SemanticProgram from the corpus.ullbc should succeed
    // and surface every local function as a SemanticFunction with a
    // populated FunctionGraph.
    let llbc = load_corpus();
    let program = build_semantic_program_from_llbc(llbc).expect("builder");
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
    let program = build_semantic_program_from_llbc(llbc).expect("builder");

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
    use majit_translate::model::{CallTarget, ExitSwitch, OpKind};

    let llbc = load_corpus();
    let mut checked = 0usize;
    for fd in llbc.iter_local_fns() {
        let Some(body): Option<Unstructured> = fd.unstructured() else {
            continue;
        };
        let graph = lower_fun_decl(llbc, fd)
            .unwrap_or_else(|e| panic!("{} failed to lower: {e}", fd.item_meta.name_path()));

        // Invariant A.  The iterator `next`-diamond rewrite
        // (`front::iter_next`) is the one sanctioned synthesized
        // exception edge: a `for x in it` loop's `StopIteration` catch
        // (the RPython `next` op raising at exhaustion).  Its block — the
        // one carrying the `[__iter_next]` op — legitimately closes with
        // `LastException`; every other block must still drop on_unwind
        // rather than lower a try/except.
        for b in &graph.blocks {
            let is_next_handler = b.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments },
                        ..
                    } if segments.len() == 1 && segments[0] == "__iter_next"
                )
            });
            if is_next_handler {
                continue;
            }
            assert!(
                b.exitswitch != Some(ExitSwitch::LastException),
                "{}: block {:?} carries a LastException exitswitch — a typed \
                 exception-handler edge was synthesized; the MIR driver must \
                 drop on_unwind, not lower it as try/except",
                graph.name,
                b.id,
            );
        }

        // Invariant B: no Call/Assert/Drop on_unwind edge leaks into the
        // front graph, i.e. every live edge into the exceptblock is a bare
        // panic-propagation raise (`UnwindResume` / `Abort` -> set_raise),
        // so the live count never EXCEEDS the MIR's raise terminators.  It
        // may be fewer: a graph that runs `clear_unreachable_blocks` (the
        // iterator `next`-diamond and `?` rewrites do) prunes the dead
        // panic-cleanup blocks the driver leaves unreachable — those
        // pruned blocks were already dead exceptblock edges, never live
        // control flow.  A leak, by contrast, sits in a REACHABLE success
        // block and would push the live count above the raise count.
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
        assert!(
            edges_into_exceptblock <= raises_in_mir,
            "{}: {} block(s) link to the exceptblock but the MIR has only {} \
             UnwindResume/Abort terminator(s) — a Call/Assert/Drop on_unwind \
             edge leaked into the front graph",
            graph.name,
            edges_into_exceptblock,
            raises_in_mir,
        );
        checked += 1;
    }
    assert!(
        checked >= 4,
        "expected to lower at least the 4 corpus shapes, got {checked}",
    );
}

/// `branch_loop_sum`'s `for &v in slice` lifts to the native `iter` +
/// `[__iter_next]` ops: Layer 3 of the iterator vertical replaces the
/// residual `Iterator::next()` call (an unregistered callee that would
/// make the rtyper census Skip) with the `next` op + a `LastException`
/// block, the way `front::iter_next` rewrites the `Option` match diamond.
#[test]
fn branch_loop_sum_lifts_next_to_iter_next_op() {
    use majit_translate::model::{CallTarget, ExitSwitch, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "branch_loop_sum").expect("lowering");

    let mut iter_next_blocks = Vec::new();
    let mut residual_next = 0usize;
    for (i, b) in graph.blocks.iter().enumerate() {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.len() == 1 && segments[0] == "__iter_next" => {
                    iter_next_blocks.push(i);
                }
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "next" => residual_next += 1,
                _ => {}
            }
        }
    }

    assert_eq!(
        iter_next_blocks.len(),
        1,
        "expected exactly one `[__iter_next]` op after the rewrite",
    );
    assert_eq!(
        residual_next, 0,
        "the residual `Iterator::next()` call must be replaced",
    );

    // The `[__iter_next]` block is a `canraise` block: `LastException`
    // exitswitch with a normal (Some) exit and a StopIteration (break)
    // exit.  No catch-all propagation edge — list `next` raises only
    // StopIteration.
    let a = iter_next_blocks[0];
    assert!(
        matches!(graph.blocks[a].exitswitch, Some(ExitSwitch::LastException)),
        "the next block must close with LastException exits",
    );
    assert_eq!(
        graph.blocks[a].exits.len(),
        2,
        "normal -> Some, StopIteration -> break",
    );
    // Exactly one exit (the normal/Some arm) carries no exitcase.
    let normal = graph.blocks[a]
        .exits
        .iter()
        .filter(|l| l.exitcase.is_none())
        .count();
    assert_eq!(normal, 1, "exactly one non-exception (Some) exit");
    // The exceptblock gains no edge from the rewrite.
    assert!(
        !graph.blocks[a]
            .exits
            .iter()
            .any(|l| l.target == graph.exceptblock),
        "the next rewrite must not link to the exceptblock",
    );
}

/// `bool_then_closure`'s `c.then(|| x + 1)` lifts to the short-circuit
/// `Option` diamond (`front::bool_then`): the residual `core::bool::then`
/// call — an unregistered callee that would make the rtyper census Skip —
/// is replaced by a `bool(c)` branch whose arms build `Some(closure())` and
/// `None` and call the closure's transparent `call_once` inherent method.
#[test]
fn bool_then_closure_lifts_to_short_circuit_diamond() {
    use majit_translate::model::{CallTarget, ExitSwitch, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "bool_then_closure").expect("lowering");

    // Only reachable blocks matter — the rewrite can leave the pre-split
    // framestate merge block unreachable (dropped by later consumers).
    let reachable = reachable_blocks(&graph);

    let mut residual_then = 0usize;
    let mut call_once = 0usize;
    let mut bool_branches = 0usize;
    let mut some_ctor = 0usize;
    let mut none_ctor = 0usize;
    for b in &graph.blocks {
        if !reachable.contains(&b.id) {
            continue;
        }
        if matches!(b.exitswitch, Some(ExitSwitch::Value(_))) {
            bool_branches += 1;
        }
        // Track the discriminant a block writes so a Some/None aggregate is
        // classified by its `__discriminant` constant.
        let mut last_disc: Option<i64> = None;
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.last().map(String::as_str) == Some("then")
                    && segments.iter().any(|s| s == "bool") =>
                {
                    residual_then += 1
                }
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "call_once" => call_once += 1,
                OpKind::ConstInt(d) => last_disc = Some(*d),
                OpKind::FieldWrite { field, .. } if field.name == "__discriminant" => {
                    match last_disc {
                        Some(1) => some_ctor += 1,
                        Some(0) => none_ctor += 1,
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    assert_eq!(
        residual_then, 0,
        "the residual `core::bool::then` call must be replaced by the diamond",
    );
    assert_eq!(call_once, 1, "the then arm calls the closure's `call_once`");
    assert_eq!(
        bool_branches, 1,
        "the call block closes with a single `bool(cond)` branch",
    );
    assert_eq!(some_ctor, 1, "the then arm builds `Some(payload)`");
    assert_eq!(none_ctor, 1, "the else arm builds `None`");
}

/// `option_question_mark`'s `let v = opt?` lifts the residual
/// `Try::branch(opt)` / `ControlFlow` diamond into a direct switch on
/// `opt.__discriminant`: `Some` extracts `opt.__pos_0` and continues,
/// `None` builds a normal `None` return value.
#[test]
fn option_question_mark_lifts_to_direct_option_switch() {
    use majit_translate::model::{CallTarget, ExitCase, ExitSwitch, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "option_question_mark").expect("lowering");

    let reachable = reachable_blocks(&graph);

    let mut residual_branch = 0usize;
    let mut direct_option_switch = 0usize;
    let mut some_payload_reads = 0usize;
    let mut none_ctor = 0usize;
    for b in &graph.blocks {
        if !reachable.contains(&b.id) {
            continue;
        }
        let mut last_disc: Option<i64> = None;
        let mut option_disc_read = None;
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "branch" => residual_branch += 1,
                OpKind::FieldRead { field, .. }
                    if field.name == "__discriminant"
                        && field.owner_root.as_deref() == Some("core::option::Option") =>
                {
                    option_disc_read = op.result.clone();
                }
                OpKind::FieldRead { field, .. }
                    if field.name == "__pos_0"
                        && field.owner_root.as_deref() == Some("core::option::Option::Some") =>
                {
                    some_payload_reads += 1;
                }
                OpKind::ConstInt(d) => last_disc = Some(*d),
                OpKind::FieldWrite { field, .. }
                    if field.name == "__discriminant"
                        && field.owner_root.as_deref() == Some("core::option::Option") =>
                {
                    if last_disc == Some(0) {
                        none_ctor += 1;
                    }
                }
                _ => {}
            }
        }
        if let (Some(ExitSwitch::Value(sw)), Some(disc)) = (&b.exitswitch, option_disc_read)
            && *sw == disc
        {
            let mut cases: Vec<i64> = b
                .exits
                .iter()
                .filter_map(|l| match &l.exitcase {
                    Some(ExitCase::Const(majit_translate::flowspace::model::ConstValue::Int(
                        i,
                    ))) => Some(*i),
                    _ => None,
                })
                .collect();
            cases.sort_unstable();
            if cases == [0, 1] {
                direct_option_switch += 1;
            }
        }
    }

    assert_eq!(
        residual_branch, 0,
        "the residual `Try::branch` call must be replaced",
    );
    assert_eq!(
        direct_option_switch, 1,
        "the rewrite must leave one direct Option discriminant switch",
    );
    assert!(
        some_payload_reads >= 1,
        "the Some arm extracts the Option payload",
    );
    assert_eq!(none_ctor, 1, "the None arm builds the normal None return");
}
