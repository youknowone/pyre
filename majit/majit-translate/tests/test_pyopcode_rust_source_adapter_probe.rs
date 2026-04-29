//! Probe: run the Rust-AST adapter on the real
//! `pyre-interpreter::execute_opcode_step` portal and capture the
//! current rejection point.
//!
//! Serves the `M2.5e — pass the real pyopcode.rs through the adapter`
//! milestone from the annotator-monomorphization plan (see
//! `~/.claude/plans/annotator-monomorphization-tier1-abstract-lake.md`).
//! The plan's acceptance criterion for M2.5e is that the adapter
//! produces a complete `FunctionGraph` for `execute_opcode_step<E>`,
//! with every opcode branch represented and every method call carrying
//! a resolvable receiver classdef.
//!
//! Today the adapter stops at the first `AdapterError::Unsupported`
//! that comes out of walking the function body. This test pins the
//! exact category of that stop so regressions surface early, and so
//! future adapter extensions have a visible "does it get further now?"
//! signal: every M2.5d/e slice that lands should either move the
//! rejection point deeper into the body or eliminate it.
//!
//! RPython parity note: upstream `flowspace/objspace.py:38-53
//! build_flow(func)` consumes Python bytecode end-to-end. The Rust-AST
//! adapter is the Position-2 adaptation for pyre's Rust-source
//! interpreter; the rejection surface is inherent to the
//! "implementation incomplete" state, not a parity gap on the
//! flowspace side.

use majit_translate::flowspace::rust_source::{AdapterError, build_flow_from_rust};
use syn::{File, Item};

const PYOPCODE_SRC: &str = include_str!("../../../pyre/pyre-interpreter/src/pyopcode.rs");

fn parse_pyopcode() -> File {
    syn::parse_file(PYOPCODE_SRC).expect("pyopcode.rs must parse")
}

fn find_fn<'a>(file: &'a File, name: &str) -> &'a syn::ItemFn {
    file.items
        .iter()
        .find_map(|item| match item {
            Item::Fn(func) if func.sig.ident == name => Some(func),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected `fn {name}` in pyopcode.rs"))
}

#[test]
fn adapter_rejects_execute_opcode_step_on_composite_match_pattern() {
    let file = parse_pyopcode();
    let func = find_fn(&file, "execute_opcode_step");

    // The function parses and runs through `validate_signature`
    // successfully — generics + where-clause on `E: …OpcodeHandler`
    // traits are accepted per `build_flow.rs:134-143` (the annotator's
    // `FunctionDesc.specialize` is what monomorphizes `E` into a
    // classdef, so the adapter itself can admit the generic shape).
    //
    // ### Rejection timeline
    //
    // - Before M2.5d slice 1 (or-pattern splitting): the first match
    //   arm
    //   `Instruction::ExtendedArg | Instruction::Resume {..} | ...`
    //   rejected at the outer `Pat::Or` classifier
    //   (`build_flow.rs:classify_pattern` — or-pattern arm).
    // - After M2.5d slice 1: or-pattern flattens, surfacing the
    //   first composite / variant sub-pattern. `Instruction::ExtendedArg`
    //   is a unit enum variant (`Pat::Path`), rejected today via the
    //   `_` catch-all of `classify_pattern` with
    //   "match arm pattern not in M2.5b subset".
    // - After M2.5d slice 2c (`Pat::Path` accepted): the first
    //   rejection moves to `Pat::Struct {..}` (e.g.
    //   `Instruction::Resume {..}`) with
    //   "composite pattern (enum/tuple/struct — lands in M2.5d)".
    // - After M2.5d slice 2d (rest-only `Pat::Struct {..}` and
    //   `Pat::TupleStruct(..)` accepted): the first rejection moves
    //   to `Pat::Struct { field, .. }` (a struct variant whose match
    //   arm binds at least one field, e.g.
    //   `Instruction::LoadConst { consti }`) with
    //   "match arm struct-variant pattern with field bindings (…) —
    //   field-binding extraction lands in M2.5d slice 2e".
    // - After M2.5d slice 2e (struct-variant named-Ident field
    //   bindings accepted): the cascade lowers every match-arm
    //   pattern in `execute_opcode_step`. Lowering then progresses
    //   INTO the arm bodies and rejects on the first un-resolved
    //   identifier — the `Result::Ok(...)` constructor reference at
    //   `Ok(StepResult::Continue)`. Surfaces as
    //   `AdapterError::UnboundLocal { name: "Ok" }` because the
    //   adapter has no host-environment registry for the standard
    //   library `Result` constructors. Resolving these is a separate
    //   M2.5g intake task.
    //
    // The assertion accepts any of these states so the probe
    // continues to pin the "rejection depth" even as slices land. A
    // landing slice that moves the rejection to a category NOT listed
    // here must update this test with the new expected state.
    let err = build_flow_from_rust(func)
        .err()
        .expect("adapter is expected to reject today — see M2.5d/e");
    match err {
        AdapterError::Unsupported { reason } => {
            eprintln!("adapter rejection at M2.5e probe: Unsupported: {reason}");
            let accepts_or = reason.contains("or-pattern");
            let accepts_variant_path =
                reason.contains("not in M2.5b subset") || reason.contains("unit variant");
            let accepts_composite =
                reason.contains("composite pattern") || reason.contains("enum/tuple/struct");
            let accepts_field_bindings =
                reason.contains("field bindings") || reason.contains("slice 2e");
            assert!(
                accepts_or || accepts_variant_path || accepts_composite || accepts_field_bindings,
                "unexpected rejection category — did a new M2.5d slice land? reason: {reason}"
            );
        }
        AdapterError::UnboundLocal { name } => {
            // Slice 2e landed: the adapter walked every match arm and
            // is now rejecting on a body-level identifier. The
            // expected first hit is `Ok` (the `Result::Ok` ctor used
            // by every `Ok(StepResult::Continue)` arm tail). Other
            // unresolved standard-library identifiers (`Err`, etc.)
            // are equally valid milestones — they all signal that the
            // pattern lowering is fully covered and the next epic is
            // the M2.5g host-environment intake for stdlib ctors.
            eprintln!("adapter rejection at M2.5e probe: UnboundLocal({name})");
        }
        other => panic!("expected AdapterError::Unsupported or UnboundLocal, got {other:?}"),
    }
}

#[test]
fn adapter_accepts_execute_opcode_step_signature_shape() {
    // Sanity partition: the signature shape alone is fine (generic
    // `<E: Trait>`, where-clause, plain-identifier params). If this
    // ever fails, the regression is in `validate_signature` +
    // `collect_params`, not in the body walker.
    //
    // Exercised indirectly via a synthetic fixture that copies only
    // the outer signature shape so the test stays independent of the
    // large body's content churn.
    let synthetic: syn::ItemFn = syn::parse_str(
        "fn execute_opcode_step<E>(
             executor: E,
             code: i64,
             instruction: i64,
             op_arg: i64,
             next_instr: i64,
         ) -> i64 where E: Handler { 0 }",
    )
    .expect("synthetic fixture parses");
    let g = build_flow_from_rust(&synthetic).expect("signature shape must be accepted");
    majit_translate::flowspace::model::checkgraph(&g);
}
