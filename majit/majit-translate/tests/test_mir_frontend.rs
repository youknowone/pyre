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
    // Names are crate-prefix-stripped (lib.rs
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

/// `branch_loop_sum` iterates `&[i64]`, so the element `[__iter_next]`
/// yields is an `i64` — the list's item repr, the way
/// `rlist.py ll_listnext` hands one back.
///
/// This is the corpus half of the element-type fix.  The fold used to
/// answer `Ref` for every container that was not `front::range_iter`'s
/// `range()` builtin, because the `iter` op carries the iterator and not
/// the container's item type — so this graph typed a raw `i64` into the
/// ref register bank.  `result_ty` is not a hint the rtyper can overrule:
/// `resolve_call_result_kind` consults `concretetype` only when
/// `result_ty` is `Unknown`, and `authoritative_result_types` stamps the
/// derived kind back over it.
#[test]
fn branch_loop_sum_next_yields_an_int_element() {
    use majit_translate::model::{CallTarget, OpKind, ValueType};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "branch_loop_sum").expect("lowering");

    let element_types: Vec<ValueType> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                result_ty,
                ..
            } if segments.len() == 1 && segments[0] == "__iter_next" => Some(result_ty.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(
        element_types,
        vec![ValueType::Int],
        "the `&[i64]` element must keep its own kind, not be typed as a GC reference",
    );
}

/// The element `[__iter_next]` yields is a `&i64` for both of these, and the
/// two get there differently: `slice_of_refs_sum` iterates `&[&i64]`, whose
/// `core::slice::iter::Iter` yields `Option<&&i64>` — one reference the
/// iterator added over one the element owns — while `array_of_refs_sum`
/// iterates `[&i64; 3]` by value, whose `core::array::iter::IntoIter` yields
/// `Option<&i64>` with no reference of its own.
///
/// So neither a blanket peel nor a blanket keep answers both: peeling every
/// reference types the first element `Int`, and peeling one unconditionally
/// types the second `Int`.  Either way a pointer lands in the integer
/// register bank, which is why the decision reads the receiver's iterator
/// ADT rather than the payload's shape alone.
#[test]
fn a_reference_element_stays_a_reference_through_either_iterator() {
    use majit_translate::model::{CallTarget, OpKind, ValueType};
    let llbc = load_corpus();

    for name in ["slice_of_refs_sum", "array_of_refs_sum"] {
        let graph = lower_function(llbc, name).expect("lowering");
        let element_types: Vec<ValueType> = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    result_ty,
                    ..
                } if segments.len() == 1 && segments[0] == "__iter_next" => Some(result_ty.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(
            element_types,
            vec![ValueType::Ref(None)],
            "{name}: a `&i64` element is a pointer and must keep the ref bank",
        );
    }
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

/// `bool_then_some`'s `c.then_some(x + 1)` lifts to the same short-circuit
/// `Option` diamond (`front::bool_then`): the residual `core::bool::then_some`
/// call — an unregistered callee that would make the rtyper census Skip — is
/// replaced by a `bool(c)` branch whose arms build `Some(value)` and `None`.
/// Unlike `then`, `then_some` takes an already-evaluated value, so the then
/// arm wraps it directly — it emits **no** `call_once`.
#[test]
fn bool_then_some_lifts_to_short_circuit_diamond() {
    use majit_translate::model::{CallTarget, ExitSwitch, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "bool_then_some").expect("lowering");

    let reachable = reachable_blocks(&graph);

    let mut residual_then_some = 0usize;
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
        let mut last_disc: Option<i64> = None;
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.last().map(String::as_str) == Some("then_some")
                    && segments.iter().any(|s| s == "bool") =>
                {
                    residual_then_some += 1
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
        residual_then_some, 0,
        "the residual `core::bool::then_some` call must be replaced by the diamond",
    );
    assert_eq!(
        call_once, 0,
        "the then_some arm wraps the eager value directly — no `call_once`",
    );
    assert_eq!(
        bool_branches, 1,
        "the call block closes with a single `bool(cond)` branch",
    );
    assert_eq!(some_ctor, 1, "the then arm builds `Some(value)`");
    assert_eq!(none_ctor, 1, "the else arm builds `None`");
}

/// `option_question_mark`'s `let v = opt?` lifts the residual
/// `Try::branch(opt)` / `ControlFlow` diamond into a direct switch on
/// `opt.__discriminant`: `Some` extracts `opt.__pos_0` and continues,
/// `None` builds a normal `None` return value.
///
/// The owners are the per-instantiation `Option<i64>` root, not the bare
/// template: the fixture's own `Some(v + addend)` writes `__pos_0` under the
/// suffixed root, so a bare read would take the payload off a different
/// classdef than the one the producer wrote.
#[test]
fn option_question_mark_lifts_to_direct_option_switch() {
    use majit_translate::model::{CallTarget, ExitCase, ExitSwitch, OpKind};
    const OPTION_ROOT: &str = "core::option::Option<i64>";
    const SOME_ROOT: &str = "core::option::Option<i64>::Some";
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
                        && field.owner_root.as_deref() == Some(OPTION_ROOT) =>
                {
                    option_disc_read = op.result.clone();
                }
                OpKind::FieldRead { field, .. }
                    if field.name == "__pos_0"
                        && field.owner_root.as_deref() == Some(SOME_ROOT) =>
                {
                    some_payload_reads += 1;
                }
                OpKind::ConstInt(d) => last_disc = Some(*d),
                OpKind::FieldWrite { field, .. }
                    if field.name == "__discriminant"
                        && field.owner_root.as_deref() == Some(OPTION_ROOT)
                        && last_disc == Some(0) =>
                {
                    none_ctor += 1;
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

/// A field read through a raw object pointer retains the pointee's declared
/// type. The frontend represents the narrowing explicitly so descriptor
/// lookup does not receive a classless instance.
#[test]
fn header_read_narrows_to_a_typed_field_read() {
    use majit_translate::model::{CallTarget, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "w_object_type").expect("lowering");
    assert_eq!(graph.name, "charon_corpus::w_object_type");

    let mut narrows = 0usize;
    let mut typed_header_reads = 0usize;
    let mut untyped_field_reads = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.as_slice()
                    == [
                        "__cast_instance_intrinsic".to_string(),
                        "ObjectHeader".to_string(),
                    ] =>
                {
                    narrows += 1
                }
                OpKind::FieldRead { field, .. } if field.name == "ob_type" => {
                    if field.owner_root.as_deref() == Some("ObjectHeader")
                        && field.owner_id.is_some()
                    {
                        typed_header_reads += 1;
                    } else {
                        untyped_field_reads += 1;
                    }
                }
                _ => {}
            }
        }
    }
    assert_eq!(narrows, 1, "the deref base is narrowed exactly once");
    assert_eq!(typed_header_reads, 1, "ob_type reads as a typed FieldRead");
    assert_eq!(
        untyped_field_reads, 0,
        "no classdef-less header read survives the narrow",
    );
}

/// A boxing allocation becomes `NewWithVtable` only after its class-static
/// address is known. Exercise the fusion directly so the unresolved and
/// resolved cases differ by that input alone.
#[test]
fn boxing_cluster_fuses_once_the_class_address_resolves() {
    use majit_translate::model::{CallTarget, OpKind, ValueType};

    const CLASS_ADDR: i64 = 0x00C0_FFEE;
    let attrs = std::collections::HashMap::from([(
        "W_IntObject".to_string(),
        vec![
            ("ob_header".to_string(), ValueType::Ref(None)),
            ("intval".to_string(), ValueType::Int),
        ],
    )]);

    let llbc = load_corpus();

    // Without a resolvable class address the cluster is left alone.
    let mut graph = lower_function(llbc, "w_new_int").expect("lowering");
    assert_eq!(
        majit_translate::model::fuse_boxing_alloc(&mut graph, &attrs),
        0,
        "an unresolvable class-static address declines the fuse, silently",
    );

    // Stand in for the driver-supplied static address.
    let mut graph = lower_function(llbc, "w_new_int").expect("lowering");
    let mut substituted = 0usize;
    for b in &mut graph.blocks {
        for op in &mut b.operations {
            let is_class_static = matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.last().map(String::as_str) == Some("INT_CLASS")
            );
            if is_class_static {
                op.kind = OpKind::ConstRefAddr(CLASS_ADDR);
                substituted += 1;
            }
        }
    }
    assert_eq!(
        substituted, 2,
        "ob_type and w_class each read the class static",
    );

    assert_eq!(
        majit_translate::model::fuse_boxing_alloc(&mut graph, &attrs),
        1,
        "the cluster fuses once the class address is a constant",
    );

    let mut fused = Vec::new();
    let mut payload_stores = 0usize;
    let mut residual_mallocs = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::NewWithVtable { owner, vtable } => fused.push((owner.clone(), *vtable)),
                OpKind::FieldWrite { field, .. } if field.name == "intval" => payload_stores += 1,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.last().map(String::as_str) == Some("malloc_typed") => {
                    residual_mallocs += 1
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        fused,
        vec![("W_IntObject".to_string(), CLASS_ADDR)],
        "one NewWithVtable carrying the real class pointer",
    );
    // Two: the re-emitted store after the `NewWithVtable`, plus the original
    // aggregate store, which is dead but not yet swept — `fuse_boxing_alloc`
    // leaves that to the `remove_dead_aggregates` pass in
    // `simplify_lowered_graph`.
    assert_eq!(payload_stores, 2, "the intval payload store is re-emitted");
    assert_eq!(
        residual_mallocs, 0,
        "the malloc_typed call is consumed, not left residual",
    );
}

/// The production lowering receives class-static addresses through
/// `HostStaticAddrs`. It must both fuse the allocation and preserve the
/// static's declared `ClassObject` root for pointer-identity comparisons.
#[test]
fn boxing_cluster_fuses_from_the_host_supplied_class_address() {
    use majit_translate::front::mir::lower_fun_decl_with_static_addrs;
    use majit_translate::model::{CallTarget, OpKind, ValueType};

    const CLASS_ADDR: i64 = 0x00C0_FFEE;
    let llbc = load_corpus();
    let static_addrs = majit_translate::HostStaticAddrs {
        pytypes: &[("INT_CLASS", CLASS_ADDR)],
        ..Default::default()
    };

    let fd = llbc.local_fn("w_new_int").expect("w_new_int in corpus");
    let graph = lower_fun_decl_with_static_addrs(llbc, fd, static_addrs).expect("lowering");

    let mut fused = Vec::new();
    let mut payload_stores = 0usize;
    let mut residual_mallocs = 0usize;
    let mut residual_class_reads = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::NewWithVtable { owner, vtable } => fused.push((owner.clone(), *vtable)),
                OpKind::FieldWrite { field, .. } if field.name == "intval" => payload_stores += 1,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } => match segments.last().map(String::as_str) {
                    Some("malloc_typed") => residual_mallocs += 1,
                    Some("INT_CLASS") => residual_class_reads += 1,
                    _ => {}
                },
                _ => {}
            }
        }
    }
    assert_eq!(
        fused,
        vec![("W_IntObject".to_string(), CLASS_ADDR)],
        "the real lowering fuses to one NewWithVtable carrying the class pointer",
    );
    assert_eq!(
        residual_mallocs, 0,
        "no residual lltype::malloc_typed survives",
    );
    assert_eq!(
        residual_class_reads, 0,
        "the class-static read resolved to the host address, not a residual call",
    );
    // One, not the two the hand-substituted sibling sees: the whole-graph
    // lowering runs `remove_dead_aggregates` after the fuse, so the
    // orphaned aggregate store is already swept here.
    assert_eq!(payload_stores, 1, "the intval payload store is re-emitted");

    // `w_number_add` keeps the class-static narrowing live after lowering.
    // Collect every narrowing so an incorrectly stamped root is reported as
    // its own entry rather than disappearing from an expected-root search.
    let add = llbc
        .local_fn("w_number_add")
        .expect("w_number_add in corpus");
    let add_graph = lower_fun_decl_with_static_addrs(llbc, add, static_addrs).expect("lowering");
    let mut narrow_roots = std::collections::BTreeMap::new();
    let mut class_addr_narrowed = 0usize;
    for b in &add_graph.blocks {
        for op in &b.operations {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args,
                result_ty,
            } = &op.kind
            else {
                continue;
            };
            if segments.first().map(String::as_str) != Some("__cast_instance_intrinsic") {
                continue;
            }
            let root = segments.get(1).cloned().unwrap_or_default();
            *narrow_roots.entry(root.clone()).or_insert(0usize) += 1;
            assert_eq!(
                result_ty,
                &ValueType::Ref(Some(root.clone())),
                "a narrow's result type is its own root",
            );
            // The narrow whose operand is the host-supplied class address
            // is the one this test is about.
            let narrows_class_addr = add_graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .any(|p| {
                    p.result.as_ref() == args.first()
                        && matches!(p.kind, OpKind::ConstRefAddr(a) if a == CLASS_ADDR)
                });
            if narrows_class_addr {
                class_addr_narrowed += 1;
                assert_eq!(
                    root, "ClassObject",
                    "the class-static address narrows to the corpus's own class root",
                );
            }
        }
    }
    assert_eq!(
        class_addr_narrowed, 1,
        "the host-supplied class address is narrowed exactly once",
    );
    assert_eq!(
        narrow_roots,
        std::collections::BTreeMap::from([
            ("ClassObject".to_string(), 1usize),
            ("ObjectHeader".to_string(), 2),
            ("W_IntObject".to_string(), 2),
        ]),
        "every narrow in the corpus carries a root the corpus itself declares",
    );
}

/// A header matching RPython's root object layout has a type pointer but no
/// per-instance class word. Lower it through the production metadata path so
/// both layout registration and allocation fusion are covered.
#[test]
fn boxing_cluster_fuses_where_the_header_declares_no_class_word() {
    use majit_translate::front::mir::lower_fun_decl_with_static_addrs;
    use majit_translate::model::{CallTarget, OpKind};

    const CLASS_ADDR: i64 = 0x00C0_FFEE;
    let llbc = load_corpus();
    let static_addrs = majit_translate::HostStaticAddrs {
        pytypes: &[("INT_CLASS", CLASS_ADDR)],
        ..Default::default()
    };

    let fd = llbc
        .local_fn("w_new_type_only_int")
        .expect("w_new_type_only_int in corpus");
    let graph = lower_fun_decl_with_static_addrs(llbc, fd, static_addrs).expect("lowering");

    let mut fused = Vec::new();
    let mut payload_stores = 0usize;
    let mut class_word_stores = 0usize;
    let mut residual_mallocs = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::NewWithVtable { owner, vtable } => fused.push((owner.clone(), *vtable)),
                OpKind::FieldWrite { field, .. } => match field.name.as_str() {
                    "intval" => payload_stores += 1,
                    "w_class" => class_word_stores += 1,
                    _ => {}
                },
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.last().map(String::as_str) == Some("malloc_typed") => {
                    residual_mallocs += 1
                }
                _ => {}
            }
        }
    }
    // The premise the arm rests on, asserted rather than assumed: nothing in
    // this cluster writes a class word, so a fuse here can only have come
    // through the no-class-word arm.
    assert_eq!(
        class_word_stores, 0,
        "the one-word header's cluster stores no class word",
    );
    assert_eq!(
        fused,
        vec![("W_TypeOnlyIntObject".to_string(), CLASS_ADDR)],
        "the one-word header's cluster fuses to one NewWithVtable",
    );
    assert_eq!(
        residual_mallocs, 0,
        "no residual lltype::malloc_typed survives",
    );
    assert_eq!(payload_stores, 1, "the intval payload store is re-emitted");
}

/// A pointer-identity type dispatch keeps its concrete arm as a direct call,
/// allowing graph discovery and inlining to continue through it.
#[test]
fn narrowing_chain_arm_lowers_to_a_direct_call() {
    use majit_translate::model::{CallTarget, OpKind};
    let llbc = load_corpus();
    let graph = lower_function(llbc, "w_number_add").expect("lowering");

    let mut direct_arm_calls = 0usize;
    let mut dyn_calls = 0usize;
    let mut header_reads = 0usize;
    let mut identity_eqs = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } => match segments.last().map(String::as_str) {
                    Some("w_int_add") => direct_arm_calls += 1,
                    Some("__dyn_call") => dyn_calls += 1,
                    _ => {}
                },
                OpKind::FieldRead { field, .. } if field.name == "ob_type" => header_reads += 1,
                OpKind::BinOp { op, .. } if op == "eq" => identity_eqs += 1,
                _ => {}
            }
        }
    }
    assert_eq!(direct_arm_calls, 1, "the taken arm is a direct call");
    assert_eq!(
        dyn_calls, 0,
        "no arm degrades to the __dyn_call placeholder"
    );
    assert_eq!(header_reads, 2, "both receivers' class words are read");
    assert_eq!(
        identity_eqs, 2,
        "type(a) is type(b), then the per-class shortcut",
    );
}

/// Counts of the three lowerings a scalar `v[i]` can end in — the eager
/// `ArrayRead` `front::mir`'s `is_vec_index_call` emits, a residual
/// `Index::index` where that arm's width proof declined, or a residual
/// `<[T]>::get` — plus the element-bank and classdef facts that say whether
/// the projections off the element resolved.
///
/// `array_descr_keys` is parallel to `array_reads`: the `(array_type_id,
/// nolength)` pair the same `ArrayRead` carries. Together with the item bank
/// that pair is the whole descr key `codewriter::assembler`'s `ArrayRead` arm
/// hands to `arraydescrof`, so recording it lets a test mint the very descr
/// the bytecode emit would.
///
/// `residual_indexes` separates "declined" from "never reached the arm":
/// both leave no `ArrayRead`, and only the surviving call says which.
struct SlotReadShape {
    array_reads: Vec<majit_translate::model::ValueType>,
    array_descr_keys: Vec<(Option<String>, bool)>,
    residual_gets: usize,
    residual_indexes: usize,
    typed_discriminant_reads: usize,
    classdefless_discriminant_reads: usize,
}

fn slot_read_shape(name: &str) -> SlotReadShape {
    use majit_translate::model::{CallTarget, OpKind};
    let graph = lower_function(load_corpus(), name).expect("lowering");
    let mut shape = SlotReadShape {
        array_reads: Vec::new(),
        array_descr_keys: Vec::new(),
        residual_gets: 0,
        residual_indexes: 0,
        typed_discriminant_reads: 0,
        classdefless_discriminant_reads: 0,
    };
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::ArrayRead {
                    item_ty,
                    array_type_id,
                    nolength,
                    ..
                } => {
                    shape.array_reads.push(item_ty.clone());
                    shape
                        .array_descr_keys
                        .push((array_type_id.clone(), *nolength));
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.last().map(String::as_str) == Some("get") => shape.residual_gets += 1,
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "get" => shape.residual_gets += 1,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if matches!(
                    segments.last().map(String::as_str),
                    Some("index" | "index_mut")
                ) =>
                {
                    shape.residual_indexes += 1
                }
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "index" || name == "index_mut" => shape.residual_indexes += 1,
                OpKind::FieldRead { field, .. } if field.name == "__discriminant" => {
                    if field.owner_root.is_some() && field.owner_id.is_some() {
                        shape.typed_discriminant_reads += 1;
                    } else {
                        shape.classdefless_discriminant_reads += 1;
                    }
                }
                _ => {}
            }
        }
    }
    shape
}

/// `&v[i]` on a `Vec<T>` whose `T` is a multi-word by-value ADT stored inline
/// declines rather than lowering to an `ArrayRead`. An `ArrayRead` here would
/// carry a descr that strides by **one word**: every index > 0 would address
/// the wrong element, and even index 0 would load the element's first word —
/// whichever of the tag or a payload field `repr(Rust)` puts there — and bank
/// it as a GC reference.
///
/// `front::mir`'s index arm admits only an element whose true width reaches the
/// descr, by one of two proofs: an ARRAY identity minted from the receiver
/// spelling (`narrow_item_array_type_id`, the narrow-int case), or an
/// addressable element — a scalar naming its own spelling, or a thin pointer,
/// which IS the one target word the identity-less descr assumes. A multi-word
/// ADT answers neither, so the call stays residual and real Rust computes the
/// element address at the real `size_of::<SlotValue>()` stride.
///
/// The descr behind the declined shape is minted below, from the key the leg
/// would have carried, and asserted to be one word wide. That width is the
/// whole reason the decline is required, so asserting it keeps the two facts
/// tied together.
///
/// `aggregate_slot_get` is the control. Both spellings are residual, so the
/// pair separates *which* call survives rather than a lowered read from a
/// residual one; the sibling
/// `a_scalar_element_indexes_to_an_int_banked_array_read` supplies the positive
/// case where the index arm does emit its `ArrayRead`.
#[test]
fn an_aggregate_element_index_declines_instead_of_striding_by_one_word() {
    use majit_translate::model::ValueType;

    // `charon_corpus` declares its own `[workspace]` table and is not a
    // dependency of this crate, so `size_of::<SlotValue>()` is not callable
    // here. This mirrors the corpus declaration field for field — an `i64`
    // variant, a thin raw-pointer variant and a two-`i64` variant, none of
    // which leaves a niche free — so the `Pair` payload alone is two words
    // before any tag. The load-bearing claim is only "wider than one word",
    // which holds under any `repr(Rust)` layout choice.
    #[allow(dead_code)]
    enum SlotValueLayout {
        Int(i64),
        Object(*const u8),
        Pair { lhs: i64, rhs: i64 },
    }
    let elem_size = std::mem::size_of::<SlotValueLayout>();
    let word = majit_translate::layout::target_word_size();
    assert!(
        elem_size > word,
        "the fixture element must be wider than one word for the decline to \
         be the load-bearing outcome, got {elem_size} vs {word}",
    );

    let indexed = slot_read_shape("aggregate_slot_index");
    // The width proof declines, so no `ArrayRead` is emitted at all — and the
    // surviving `Vec::index` call says the arm was *reached* and refused,
    // rather than never matched.
    assert!(
        indexed.array_reads.is_empty(),
        "the aggregate element reaches no ArrayRead, got {:?}",
        indexed.array_reads,
    );
    assert_eq!(
        indexed.residual_indexes, 1,
        "the declined element leaves its `Index::index` call residual",
    );
    assert_eq!(
        indexed.residual_gets, 0,
        "the index spelling reaches no `get`",
    );
    // The discriminant read downstream of the element still resolves against
    // `SlotValue`'s own classdef rather than arriving as a bare pointer: the
    // decline costs the eager read, not the typing.
    assert_eq!(
        indexed.typed_discriminant_reads, 1,
        "the match reads __discriminant once, against a resolved owner",
    );
    assert_eq!(
        indexed.classdefless_discriminant_reads, 0,
        "no classdef-less discriminant read survives the residual call",
    );

    // The descr the leg would have carried, minted directly. `codewriter::
    // assembler`'s `ArrayRead` arm calls the module-level
    // `arraydescrof(item_ty, array_type_id, len_offset, callcontrol)` with
    // `len_offset = None` when `nolength` and `Some(0)` otherwise, and that
    // routes straight through `CallControl::arraydescrof_for_type`. The
    // `ir_type` it passes comes from the private
    // `value_type_to_ir_type_for_descr`, whose wildcard arm answers
    // `Type::Ref` for `ValueType::Ref(_)`.
    //
    // With `array_type_id: None`, `arraydescrof_concrete` never consults
    // `is_known_struct`, so no registered struct layout can reach this descr's
    // item size and the else arm sets `item_size = target_word_size()`. Both
    // assertions below still hold; they are why the arm above has to decline
    // rather than emit.
    let callcontrol = majit_translate::codewriter::call::CallControl::new();
    let descr = callcontrol.arraydescrof_for_type(
        &ValueType::Ref(None),
        &None,
        majit_ir::value::Type::Ref,
        Some(0),
    );
    let array_descr = descr
        .as_array_descr()
        .expect("arraydescrof_for_type must answer an ArrayDescr");
    assert_eq!(
        array_descr.item_size(),
        word,
        "an identity-less descr still strides by one word ({word}) while the \
         element is {elem_size} bytes wide — which is why no ArrayRead may \
         carry it over this element",
    );
    assert_eq!(
        array_descr.item_type(),
        majit_ir::value::Type::Ref,
        "and the single word it would load is banked as a GC reference, so \
         even at index 0 the backend would hand a non-pointer word to the \
         ref bank",
    );

    // The control: the `get` spelling is residual independently of the index
    // arm's width proof, so it pins the same safe lowering from the other side.
    let got = slot_read_shape("aggregate_slot_get");
    assert_eq!(
        got.residual_gets, 1,
        "the `get` spelling leaves its call residual, so real Rust computes \
         the element address at the real stride",
    );
    assert!(
        got.array_reads.is_empty(),
        "the `get` spelling emits no ArrayRead, got {:?}",
        got.array_reads,
    );
}

/// The same pair over `Vec<i64>`, an element bank the index arm is already
/// known to serve. It separates the two ways the sibling test could read: an
/// aggregate element that failed to lower would differ from this baseline,
/// while a fixture that failed to reach the index arm at all would match it
/// in the `get` column and miss the `ArrayRead` in both.
#[test]
fn a_scalar_element_indexes_to_an_int_banked_array_read() {
    use majit_translate::model::ValueType;

    let indexed = slot_read_shape("scalar_slot_index");
    assert_eq!(
        indexed.array_reads,
        vec![ValueType::Int],
        "an i64 element reads as one ArrayRead in the int bank",
    );
    assert_eq!(indexed.residual_gets, 0, "the index spelling has no `get`");

    let got = slot_read_shape("scalar_slot_get");
    assert_eq!(
        got.residual_gets, 1,
        "the `get` spelling leaves its call residual for a scalar element too",
    );
    assert!(
        got.array_reads.is_empty(),
        "the `get` spelling emits no ArrayRead, got {:?}",
        got.array_reads,
    );
}

/// A borrowed primitive banks by its container, not by its own type.
///
/// `charon-corpus` §10's three shapes each put a shared borrow of a primitive
/// in a payload position, and all three serialize that borrow identically, so
/// no predicate over the payload's own type separates them:
///
/// | shape                        | payload         | reached through            |
/// |------------------------------|-----------------|----------------------------|
/// | `slice_get_tag_dispatch`     | `Option<&u8>`   | `<[T]>::get` then `?`      |
/// | `range_start_index`          | `Bound<&usize>` | `RangeBounds::start_bound` |
/// | `borrowed_byte_fields_alias` | `&u8`           | a struct field             |
///
/// The first two are enum-variant payloads, reached by matching on them, so
/// the borrow belongs to the match rather than to the program — and a sibling
/// arm supplying the merged value by value (`Bound::Unbounded => 0`) forces
/// one bank across the merge.  The third is a reference the program declared
/// and stores, which `ptr::eq` compares by address, so it keeps the ref bank.
///
/// Each shape falls to a different wrong answer, which is why all three are
/// asserted together: never peeling types the first `Ref`; peeling only the
/// `?`-desugaring shells types the second `Ref`, because `Bound` is not one;
/// peeling every borrowed primitive types the third `Unsigned`.
#[test]
fn a_borrowed_primitive_banks_by_its_container() {
    use majit_translate::model::{OpKind, ValueType};
    let llbc = load_corpus();

    // `__discriminant` is the tag read the match itself needs, not a payload.
    let payloads = |name: &str| -> Vec<(String, ValueType)> {
        let graph = lower_function(llbc, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldRead { field, ty, .. } if field.name != "__discriminant" => {
                    Some((field.owner_root.clone().unwrap_or_default(), ty.clone()))
                }
                _ => None,
            })
            .collect()
    };

    assert_eq!(
        payloads("slice_get_tag_dispatch"),
        vec![("core::option::Option::Some".to_string(), ValueType::Unsigned)],
        "the `?` payload of an `Option<&u8>` is the byte, not a pointer to it",
    );
    assert_eq!(
        payloads("range_start_index"),
        vec![
            ("Bound::Included".to_string(), ValueType::Unsigned),
            ("Bound::Excluded".to_string(), ValueType::Unsigned),
        ],
        "a `Bound` payload is an enum variant's too, though no `?` produces it",
    );
    assert_eq!(
        payloads("borrowed_byte_fields_alias"),
        vec![
            ("BorrowedByte".to_string(), ValueType::Ref(None)),
            ("BorrowedByte".to_string(), ValueType::Ref(None)),
        ],
        "a struct's `&u8` field is a pointer the program stores and compares",
    );
}
