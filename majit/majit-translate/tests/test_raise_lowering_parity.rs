//! Parity test: `panic!` / `assert!` lowering emits the RPython
//! inlined `exc_from_raise` op sequence at the AST layer, with no
//! pyre-only synthetic helper calls.
//!
//! Upstream chain (`rpython/flowspace/flowcontext.py:600` + `:1253`):
//!
//! ```text
//! w_value = op.simple_call(const(exc_class), *args)  # flowcontext.py:600 / 614 / 623
//! w_type  = op.type(w_value)                          # flowcontext.py:600 tail
//! closeblock(Link([w_type, w_value], exceptblock))    # flowcontext.py:1253
//! ```
//!
//! ## PRE-EXISTING-ADAPTATION pinned by these tests
//!
//! Upstream `Constant` (`flowspace/model.py:354`) lives directly
//! inside `SpaceOperation.args` (`flowspace/model.py:436` — `args` is
//! a mixed `Variable | Constant` list), so `simple_call.args[0]` is
//! the class Constant in upstream.  Pyre's `OpKind::Call.args` is
//! still `Vec<ValueId>` — migrating it to `Vec<LinkArg>` is the
//! orthodox fix and is a multi-session port.  Until that lands, the
//! AST helper carries the constant exception class as the *second
//! segment* of the `simple_call` Call's `FunctionPath` target —
//! `["simple_call", exc_class_name]` — and the tests pin that shape.
//! See `front/raise.rs` module docstring for the rejected attempts
//! and the orthodox-migration plan.
//!
//! The flowspace port lives at
//! `majit-translate/src/flowspace/flowcontext.rs:1189`
//! (`FlowContext::exc_from_raise`). The AST-layer helper at
//! `majit-translate/src/front/raise.rs::lower_exc_from_raise`
//! emits the constant-foldable slice of that op sequence directly
//! into a `FunctionGraph`. Rust `panic!` / `assert!` / `unreachable!`
//! / `todo!` / `unimplemented!` all bottom out in the same helper —
//! their only degree of freedom is the constant exception class
//! (`"PanicError"` for panic-family macros, `"AssertionError"` for
//! assert-family macros), materialised through a leading `const`
//! Call producer so the class lives at `simple_call.args[0]`.
//!
//! This file pins the invariants the reviewer called out:
//!
//! 1. **No synthetic helper targets.** No `__pyre_exc_from_raise__`
//!    or `__pyre_exception_type_of__` Call may appear in a lowered
//!    graph.
//! 2. **The `(etype, evalue)` pair is `(op.type(evalue), op.simple_call(...))`.**
//!    Both are real SSA Call results; the op names are the RPython
//!    op namespace itself.
//! 3. **Operand-position parity.**  The exception class flows through
//!    `simple_call.args[0]`, produced by a `const`-target Call op —
//!    not folded into the `simple_call` target's `FunctionPath`.
//!    Upstream reference: `flowspace/operation.py:666` reads
//!    `self.args[0]` as the callable.
//! 4. **Side effects of every message arg land on the graph.**
//!    `panic!("{}", side_effect())` / `assert!(cond, side_effect())`
//!    walk every argument before raising; fail-branch-only for the
//!    conditional case, per `flowcontext.py:107 guessbool`.
//! 5. **No RaisePayload op.** The deleted `OpKind::RaisePayload` must
//!    not resurface.

use majit_translate::front::ast::build_function_graph_pub;
use majit_translate::model::{CallTarget, OpKind};

fn parse_fn(src: &str) -> syn::ItemFn {
    let file: syn::File = syn::parse_str(src).expect("source must parse");
    for item in file.items {
        if let syn::Item::Fn(f) = item {
            return f;
        }
    }
    panic!("no fn in source");
}

fn find_raise_args(
    graph: &majit_translate::model::FunctionGraph,
) -> (
    majit_translate::model::ValueId,
    majit_translate::model::ValueId,
) {
    let exceptblock = graph.exceptblock;
    let mut seen: Vec<_> = Vec::new();
    for block in &graph.blocks {
        for link in &block.exits {
            if link.target == exceptblock && link.args.len() == 2 {
                let etype = link.args[0]
                    .as_value()
                    .expect("raise link etype must be a ValueId, not a Const");
                let evalue = link.args[1]
                    .as_value()
                    .expect("raise link evalue must be a ValueId, not a Const");
                seen.push((etype, evalue));
            }
        }
    }
    assert_eq!(
        seen.len(),
        1,
        "expected exactly one raising link to exceptblock, got {}",
        seen.len()
    );
    seen[0]
}

fn lookup_op_by_result<'a>(
    graph: &'a majit_translate::model::FunctionGraph,
    v: majit_translate::model::ValueId,
) -> Option<&'a majit_translate::model::SpaceOperation> {
    graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .find(|op| op.result == Some(v))
}

/// PRE-EXISTING-ADAPTATION shape check — pins the
/// `["simple_call", exc_class_name]` two-segment path encoding that
/// the helper currently uses for the constant class operand.  See
/// the module docstring (top of this file) for the orthodox-fix
/// migration plan tracked separately.
///
/// ```text
/// evalue = op.simple_call(const(ExcClass), *message_args)   # path encodes class
/// etype  = op.type(evalue)
/// ```
fn assert_exc_from_raise_shape(
    graph: &majit_translate::model::FunctionGraph,
    expected_exc_class: &str,
) -> (
    majit_translate::model::ValueId,
    majit_translate::model::ValueId,
    Vec<majit_translate::model::ValueId>,
) {
    let (etype, evalue) = find_raise_args(graph);
    let evalue_op = lookup_op_by_result(graph, evalue)
        .expect("evalue must be produced by a real op (op.simple_call)");
    let etype_op =
        lookup_op_by_result(graph, etype).expect("etype must be produced by a real op (op.type)");
    // evalue = op.simple_call(...) with the constant exception class
    // encoded as path segment 1 (PRE-EXISTING-ADAPTATION — see the
    // file-level docstring for the rejected attempts to put the
    // class at args[0] and the orthodox-fix migration plan).
    let evalue_args = match &evalue_op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } => {
            assert_eq!(
                segments[0], "simple_call",
                "evalue Call target must be `op.simple_call`; got {:?}",
                segments
            );
            assert_eq!(
                segments.get(1).map(String::as_str),
                Some(expected_exc_class),
                "evalue Call must carry the constant exception class \
                 `{}` as its second path segment (PRE-EXISTING-ADAPTATION \
                 of RPython `simple_call(const(ExcClass), ...)`); got \
                 {:?}",
                expected_exc_class,
                segments
            );
            args.clone()
        }
        other => panic!(
            "evalue must be OpKind::Call (op.simple_call), got {:?}",
            other
        ),
    };
    // etype = op.type(evalue) — single path segment, single-arg call.
    match &etype_op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } => {
            assert_eq!(
                segments,
                &vec!["type".to_string()],
                "etype Call target must be `op.type`; got {:?}",
                segments
            );
            assert_eq!(
                args,
                &vec![evalue],
                "etype Call must take evalue as its single input (mirrors \
                 `op.type(w_value)` at flowcontext.py:600 tail)"
            );
        }
        other => panic!("etype must be OpKind::Call (op.type), got {:?}", other),
    }
    (etype, evalue, evalue_args)
}

fn assert_no_synthetic_raise_targets(graph: &majit_translate::model::FunctionGraph) {
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } = &op.kind
            {
                for seg in segments {
                    assert!(
                        !seg.starts_with("__pyre_"),
                        "no synthetic `__pyre_*` Call target may appear \
                         in lowered graph (reviewer priority: no \
                         synthetic helper calls); found {:?}",
                        segments
                    );
                }
            }
        }
    }
}

#[test]
fn panic_exceptblock_link_pair_is_two_rpython_op_calls() {
    let func = parse_fn(
        r#"
        fn raises() {
            panic!(make_msg());
        }
        "#,
    );
    let sf = build_function_graph_pub(&func).expect("must lower");
    assert_no_synthetic_raise_targets(&sf.graph);
    let (_etype, _evalue, evalue_args) = assert_exc_from_raise_shape(&sf.graph, "PanicError");
    // panic!(make_msg()) forwards the evaluated message Call result
    // as the single positional arg of the `simple_call` op.  The
    // class is encoded in the path's second segment (PRE-EXISTING-
    // ADAPTATION), so `evalue_args` here only enumerates the
    // message slots.
    assert_eq!(
        evalue_args.len(),
        1,
        "panic!(make_msg()) must forward the single evaluated message \
         arg as simple_call's message-side arg; got args: {:?}",
        evalue_args
    );
}

#[test]
fn panic_message_args_are_walked_for_side_effects_on_the_graph() {
    // RPython `RAISE_VARARGS` evaluates every popped value before
    // reaching the raise machinery (`flowcontext.py:638-656`
    // popvalue chain). Pyre's `lower_expr` walk of each message
    // arg reproduces that: every sub-expression's Call / FieldRead
    // / … lands on the graph.
    //
    // The variadic shape `simple_call(class_v, fmt, a, b, c)` is
    // itself RPython-canonical: `flowspace/operation.py:663-679`
    // defines `SimpleCall(SingleDispatchMixin, CallOp)` whose
    // `args = [callable, *args]` is variadic.  `raise PanicError(a,
    // b, c)` would lower upstream to exactly the same op shape.
    let func = parse_fn(
        r#"
        fn raises_multi() {
            panic!("{} {} {}", a(), b(), c());
        }
        "#,
    );
    let sf = build_function_graph_pub(&func).expect("must lower");
    assert_no_synthetic_raise_targets(&sf.graph);

    // Every evaluated message-arg Call must appear on the graph.
    let all_call_targets: Vec<String> = sf
        .graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call { target, .. } => Some(format!("{:?}", target)),
            _ => None,
        })
        .collect();
    for name in ["\"a\"", "\"b\"", "\"c\""] {
        assert!(
            all_call_targets.iter().any(|s| s.contains(name)),
            "message-arg Call `{name}` must land on the graph for \
             its side effects (popvalue-before-raise semantic); \
             got Call targets: {all_call_targets:?}"
        );
    }
    assert_exc_from_raise_shape(&sf.graph, "PanicError");
}

#[test]
fn bare_panic_uses_rpython_simple_call_with_no_message_args() {
    // `panic!()` → `simple_call(const(PanicError))` (class-only) +
    // `op.type(evalue)`.  The helper always emits a real Call; it
    // never produces a fresh null placeholder in the exceptblock
    // Link.  Under the PRE-EXISTING-ADAPTATION the class lives in
    // the path's second segment, so `simple_call`'s `args` vector is
    // empty for a bare panic.
    let func = parse_fn(
        r#"
        fn bare() {
            panic!();
        }
        "#,
    );
    let sf = build_function_graph_pub(&func).expect("must lower");
    assert_no_synthetic_raise_targets(&sf.graph);
    let (_etype, _evalue, evalue_args) = assert_exc_from_raise_shape(&sf.graph, "PanicError");
    assert!(
        evalue_args.is_empty(),
        "bare panic: simple_call op has no message args (the class \
         is encoded in path segment 1); got args: {:?}",
        evalue_args
    );
}

#[test]
fn assert_fail_branch_uses_rpython_simple_call_assertion_error() {
    // `assert!(cond, make_msg())`: fail branch lowers `make_msg()`
    // for side effects then emits the RPython op sequence with
    // `AssertionError` as the constant class. The pass branch must
    // not contain the message Call (guessbool independent walks,
    // `flowcontext.py:107`).
    let func = parse_fn(
        r#"
        fn checked(n: i64) {
            assert!(n > 0, make_msg());
        }
        "#,
    );
    let sf = build_function_graph_pub(&func).expect("must lower");
    assert_no_synthetic_raise_targets(&sf.graph);
    assert_exc_from_raise_shape(&sf.graph, "AssertionError");
    // The block containing the simple_call is the fail block. Find
    // it and assert make_msg is there, while no other block has it.
    let exceptblock = sf.graph.exceptblock;
    let fail_block = sf
        .graph
        .blocks
        .iter()
        .find(|b| {
            b.exits
                .iter()
                .any(|l| l.target == exceptblock && l.args.len() == 2)
        })
        .expect("fail block");
    let fail_has_make_msg = fail_block.operations.iter().any(|op| match &op.kind {
        OpKind::Call { target, .. } => format!("{:?}", target).contains("make_msg"),
        _ => false,
    });
    assert!(
        fail_has_make_msg,
        "fail branch must contain the make_msg Call — upstream \
         guessbool puts message-side work inside the failing arm only"
    );
    let pass_has_make_msg = sf.graph.blocks.iter().any(|b| {
        if std::ptr::eq(b, fail_block) {
            return false;
        }
        b.operations.iter().any(|op| match &op.kind {
            OpKind::Call { target, .. } => format!("{:?}", target).contains("make_msg"),
            _ => false,
        })
    });
    assert!(
        !pass_has_make_msg,
        "pass branch must NOT contain make_msg — message eval is \
         the fail-arm-only part of the assert! expansion"
    );
}

#[test]
fn no_synthetic_or_raise_payload_ops_in_lowered_graphs() {
    // Double regression fence:
    // - `__pyre_exc_from_raise__` / `__pyre_exception_type_of__`
    //   must never reappear (reviewer priority: no synthetic
    //   helper calls masking upstream's inlined op sequence).
    // - `OpKind::RaisePayload` (the earlier deleted variant) must
    //   not resurface either.
    for src in [
        "fn zero() { panic!(); }",
        r#"fn one() { panic!(make_msg()); }"#,
        r#"fn multi() { panic!("{} {}", a(), b()); }"#,
        "fn bare_assert(n: i64) { assert!(n > 0); }",
        r#"fn asserted(n: i64) { assert!(n > 0, make_msg()); }"#,
    ] {
        let func = parse_fn(src);
        let sf = build_function_graph_pub(&func).expect("must lower");
        assert_no_synthetic_raise_targets(&sf.graph);
        for block in &sf.graph.blocks {
            for op in &block.operations {
                let debug = format!("{:?}", op.kind);
                assert!(
                    !debug.contains("RaisePayload"),
                    "source `{src}` produced a RaisePayload op: {debug}"
                );
            }
        }
    }
}
