//! Probe: run the Rust-AST adapter on the real
//! `pyre-interpreter::execute_opcode_step` portal.
//!
//! Serves the "pass the real pyopcode.rs through the adapter" milestone
//! from the annotator-monomorphization plan.
//! The acceptance criterion is that the adapter
//! produces a complete `FunctionGraph` for `execute_opcode_step<E>`,
//! with every opcode branch represented and every method call carrying
//! a resolvable receiver classdef.
//!
//! Status (2026-05-17, post `Expr::If` statement-position
//! generalization): the with-walker oracle now asserts the
//! adapter lowers `execute_opcode_step` end-to-end. The without-
//! walker oracle still rejects at the cast-removal helper layer
//! (no per-module registry to resolve `u32_as_i64` &c through), and
//! the signature-shape oracle continues to verify `validate_signature`
//! independently. Regression detection: any of the three failing
//! signals an upstream slice silently broke.
//!
//! RPython parity note: upstream `flowspace/objspace.py:38-53
//! build_flow(func)` consumes Python bytecode end-to-end. The Rust-AST
//! adapter is the Position-2 adaptation for pyre's Rust-source
//! interpreter; the rejection surface is inherent to the
//! "implementation incomplete" state, not a parity gap on the
//! flowspace side.

use majit_translate::flowspace::rust_source::{
    AdapterError, build_flow_from_rust, build_flow_from_rust_in_module, register_rust_module,
};
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
fn adapter_accepts_execute_opcode_step_when_walker_registers_module() {
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
    // - Before or-pattern splitting: the first match
    //   arm
    //   `Instruction::ExtendedArg | Instruction::Resume {..} | ...`
    //   rejected at the outer `Pat::Or` classifier
    //   (`build_flow.rs:classify_pattern` — or-pattern arm).
    // - After or-pattern splitting: or-pattern flattens, surfacing the
    //   first composite / variant sub-pattern. `Instruction::ExtendedArg`
    //   is a unit enum variant (`Pat::Path`), rejected today via the
    //   `_` catch-all of `classify_pattern` with
    //   "match arm pattern not in supported subset".
    // - After `Pat::Path` accepted: the first rejection moves to
    //   `Pat::Struct {..}` (e.g. `Instruction::Resume {..}`) with
    //   "composite pattern (enum/tuple/struct)".
    // - After rest-only `Pat::Struct {..}` and
    //   `Pat::TupleStruct(..)` accepted): the first rejection moves
    //   to `Pat::Struct { field, .. }` (a struct variant whose match
    //   arm binds at least one field, e.g.
    //   `Instruction::LoadConst { consti }`) with
    //   "match arm struct-variant pattern with field bindings (…)".
    // - After struct-variant named-Ident field
    //   bindings accepted): the cascade lowers every match-arm
    //   pattern in `execute_opcode_step`. Lowering then progresses
    //   INTO the arm bodies and rejects on the first un-resolved
    //   identifier — the `Result::Ok(...)` constructor reference at
    //   `Ok(StepResult::Continue)`. Surfaces as
    //   `AdapterError::UnboundLocal { name: "Ok" }` because the
    //   adapter has no host-environment registry for the standard
    //   library `Result` constructors. Resolving these requires
    //   `Bookkeeper::register_rust_function`.
    // - The Result/Option wrapper-transparency rewrite trio
    //   (Ok/Some/None value-position rewrite, qualified-path
    //   expression-position sentinel, terminator-position Err raise
    //   edge) landed in `e7e168c29f7` and was REVERTED 2026-05-03
    //   per Codex parity audit. Each rewrite was a deviation:
    //   value-position `Ok(x)` collapse erased the
    //   `simple_call(<host>, x)` op upstream emits; qualified-path
    //   ByteStr sentinel produced a graph that did not match
    //   `getattr(…, attr)` cascade upstream emits; terminator Err
    //   raise rewrite was an incomplete `exc_from_raise` shape
    //   missing the isinstance check, optional class instantiation
    //   and `ll_assert_not_none` from
    //   `flowcontext.py:632-636 exc_from_raise`. The orthodox
    //   replacement plan is at
    //   `~/.claude/plans/m2_5e_orthodox_host_env_resolution.md`.
    // - 2026-05-03 — Orthodox replacement landed:
    //   `Builder::resolve_path_constant` mirrors upstream
    //   `flowcontext.py:856 LOAD_GLOBAL` + `:861 LOAD_ATTR` chain.
    //   Closed-world `host_env::PYRE_STDLIB` registry resolves bare
    //   `Ok` / `Some` / `Err` / `Result` / `Option` to
    //   `Constant(HostObject(<class>))`; bare `None` resolves to
    //   `Constant(ConstValue::None)`; multi-segment paths emit a
    //   `getattr` cascade per `operation.py:618 getattr`, with the
    //   leftmost segment minted on demand and cached on the Builder
    //   so two cascade steps that name the same class share
    //   identity. Probe rejection advanced from
    //   `UnboundLocal { name: "Ok" }` to
    //   `UnboundLocal { name: "u32_as_i64" }` — the
    //   pyre-as-cast-removal helper that
    //   `Bookkeeper::register_rust_function` resolves.
    // - 2026-05-04 — Further orthodox replacement
    //   landed (err-raise attempted+reverted+re-landed same day with
    //   the fork-elision deviation addressed):
    //     * `lower_match_variant_cascade` isinstance arg2 routes
    //       through `Builder::resolve_path_constant`. Each cascade
    //       step block emits its own `getattr` op per non-leftmost
    //       segment of the variant path, then `isinstance(scrutinee,
    //       <leaf>)` per `operation.py:449`; identity sharing across
    //       cascade steps (and across graphs) via the process-global
    //       `host_env::HOST_CLASS_MINTS` registry. Replaces the
    //       prior `Constant(ByteStr(joined_path))` sentinel.
    //     * `lower_value_boundary` collapses `Ok(x)` / `Some(x)`
    //       / `None` AT BOUNDARY positions only (function/arm tail,
    //       `return` operand). Documented adaptation;
    //       value-position calls keep `simple_call(<host>, …)` per
    //       the orthodox replacement above.
    //     * `emit_err_raise_boundary` (full fork, PARITY) lowers
    //       boundary-position `Err(e)` to the upstream
    //       `flowcontext.py:600-636 exc_from_raise` op sequence with
    //       the 2-exit `guessbool(isinstance(arg, type))` fork at
    //       `flowcontext.py:610` preserved. True arm:
    //       `w_value = simple_call(evalue)` (`flowcontext.py:614`,
    //       instantiate). False arm:
    //       `w_value = ll_assert_not_none(evalue)`
    //       (`flowcontext.py:632-634`, instance shape; the TypeError
    //       sub-arm is constant-folded out by upstream's
    //       `guessbool(is_(w_arg2, const(None)))` since w_arg2 is
    //       `const(None)` from `RAISE_VARARGS(1)`). Both arms
    //       converge on a join block that emits `type(w_value)` and
    //       Links `[etype, w_value]` to `graph.exceptblock` per
    //       `flowcontext.py:1259 Raise.nomoreblocks`. The prior
    //       reverted attempt (`16ebcd497b0`) elided the fork on the
    //       unenforced "Err always carries an instance" claim;
    //       re-landed via `f296dfdc490` after the orthodox port.
    //   Probe rejection unchanged: still
    //   `UnboundLocal { name: "u32_as_i64" }`. The cascade walks
    //   INTO arm bodies; the first body-level free identifier is the
    //   pyre-as-cast-removal helper, which resolves through the
    //   `Bookkeeper::register_rust_function` (a separate effort).
    // - 2026-05-04 — module-globals walker landed:
    //   `register_rust_module(&syn::File)` walks `pyopcode.rs` once
    //   and registered every top-level `Item::Fn` into the per-process
    //   `host_env::HOST_RUST_MODULE_FUNCS` registry as a
    //   `HostObject::UserFunction` whose `GraphFunc.prebuilt_flow_graph`
    //   stays `None`. With the helpers resolving, the adapter walked
    //   deeper into `execute_opcode_step` and surfaced a closure
    //   expression as the next un-roadmapped construct.
    // - 2026-05-05 — Issue 1.2 TODO: the module-walker's
    //   `Item::Fn` registration is REVERTED. The deferred-body
    //   `HostObject` had no path back to the Rust-AST adapter
    //   (`FunctionDesc.buildgraph` at `description.py:140` only
    //   knows how to call `build_flow(GraphFunc)` against
    //   `func.__code__.co_code`, but pyre's `HostCode` for an
    //   Item::Fn carries empty bytecode), so registered sibling
    //   fns would supply empty bodies at lowering time. The
    //   walker now skips Item::Fn entirely; the entry-point fn is
    //   located via `file.items.iter().find_map(...)` in
    //   `build_host_function_from_rust_file` instead. **Probe
    //   rejection rolls back** from `Unsupported(closure)` to the
    //   pre-walker `UnboundLocal { name: "u32_as_i64" }` state, since
    //   the cast-removal helpers no longer resolve through the
    //   registry. Convergence path: side-table walker that
    //   pairs the metadata HostObject with a stored `&syn::ItemFn`
    //   for replay, OR eager prebuilt-graph
    //   construction at walker time.
    // - 2026-05-07 — `d126c8d16d7` re-introduced eager Item::Fn
    //   registration (the eager prebuilt-graph path
    //   convergence option). Walker now does try-build-then-
    //   register-on-success: `Item::Fn`s whose bodies lower
    //   cleanly register as `HostObject::UserFunction` carrying
    //   the prebuilt PyGraph; bodies the walker rejects (e.g.
    //   `as T`) stay unregistered, falling back to the resolver's
    //   mint-or-fail path. Helpers' `as T` cast bodies continue to
    //   fail registration → probe stayed at
    //   `UnboundLocal { name: "u32_as_i64" }` (the first cascade-
    //   driven helper reference encountered in
    //   `execute_opcode_step`).
    // - 2026-05-08 — cast-removal helpers rewrite:
    //   `u32_as_i64` body rewritten from `x as i64` to
    //   `i64::from(x)`. The lossless `From<u32> for i64` impl
    //   lowers as `simple_call(getattr(<i64>, "from"), x)` per
    //   `lower_call`, mirroring upstream's
    //   `LOAD_GLOBAL r_longlong; LOAD_FAST x; CALL_FUNCTION 1`
    //   (the RPython idiom for explicit widening at
    //   `rlib/rarithmetic.py:303`). Walker registers `u32_as_i64`
    //   with its prebuilt graph; the cascade in
    //   `execute_opcode_step` now resolves the helper and walks
    //   deeper, surfacing the closure expression as the new
    //   un-roadmapped construct. The 3 sibling helpers
    //   (`u32_as_usize` / `op_arg_as_usize` / `raise_kind_as_usize`)
    //   stay un-rewritten this session because their u32 → usize
    //   cast has no const-stable `From` impl on 32-bit hosts; the
    //   helper bodies' `as usize` survives until a follow-up
    //   slice replaces it with `usize::try_from(x).expect(...)`
    //   or similar (no-`as`) idiom. Reaching either of those
    //   helpers requires the closure rejection to lift first, so
    //   the probe re-pins to the closure stuck point.

    // 2026-05-17 — closure-free LoadFast/LoadFastCheck
    // varnames lookup + `let _ = expr?;` rewrite + LoadSpecial wildcard
    // tail arm) plus the `Expr::If` statement-position generalization in
    // `lower_block` advanced the walker past every previously-pinned
    // stuck point. The Position-2 adapter now lowers
    // `execute_opcode_step` end-to-end when invoked alongside the
    // module-globals walker. This is the end-to-end adapter milestone.
    //
    // Timeline (the rejection history this oracle previously pinned —
    // each entry was a slice that moved the stuck point deeper, see
    // earlier commits for the full chain): or-pattern → unit variant
    // (`Pat::Path`) → rest-only composite → struct-variant field-
    // binding cascade → wrapper-transparency rewrites → cast-
    // removal helpers (`u32_as_i64` via `i64::from`) →
    // closure in LoadFast → composite let-pattern → if-else as
    // statement → `let _ = expr?` → variant-cascade wildcard tail
    // arm. Closing the last items advanced the walker past the
    // function's terminal arm.
    //
    // The probe pins the **success** state strictly:
    //
    // - `build_flow_from_rust_in_module` returns Ok(graph) for
    //   `execute_opcode_step` after `register_rust_module` has
    //   populated the per-module registry.
    // - The resulting `FunctionGraph` carries multiple blocks (the
    //   outer `match instruction` cascade) and `checkgraph` passes.
    // - If this regresses to an `Err(_)`, an intermediate slice
    //   silently broke; treat as a parity regression and locate the
    //   slice via the error message.
    //
    // Per-module scoping (Issue 1.3, 2026-05-05):
    // `register_rust_module` mints a fresh `ModuleId` and returns it;
    // `build_flow_from_rust_in_module` threads the same id through
    // body lowering so the cascade's `LOAD_GLOBAL` resolutions hit
    // the just-walked partition.
    let file_for_walker = parse_pyopcode();
    let module_id = register_rust_module(&file_for_walker).expect("walker must succeed");
    let graph = build_flow_from_rust_in_module(func, module_id)
        .expect("Position-2 adapter lowers execute_opcode_step end-to-end");
    majit_translate::flowspace::model::checkgraph(&graph);
    // Structural sanity: the outer `match instruction` produces a
    // cascade of blocks (one isinstance fork per non-wildcard arm),
    // so the resulting graph must carry strictly more than the
    // startblock + returnblock pair a constant-return function
    // produces.
    let block_count = graph.iterblocks().len();
    assert!(
        block_count > 2,
        "execute_opcode_step's outer match cascade must materialize \
         multiple blocks; got block_count={block_count} (a 2-block \
         graph would mean the body lowered to a single constant)"
    );
    // The exact statement-position `if/else` join shape is pinned by
    // `statement_position_if_else_join_has_only_merged_local_inputargs`
    // in `test_rust_source_adapter_through_build_types.rs`. Keep this
    // real-portal probe focused on end-to-end adapter acceptance so it
    // does not reject legitimate value-position joins whose result is
    // unused by later code.
}

#[test]
fn adapter_rejects_execute_opcode_step_without_walker_at_cast_removal_helper() {
    // Sister oracle: WITHOUT the walker call, the rejection state is a
    // pre-walker `UnboundLocal { name }`. Since the dispatch arms were
    // lifted to per-opcode `execute_<op>` tail-calls, the first symbol
    // the adapter cannot resolve is now the first lifted handler
    // (`execute_load_const`, the `LoadConst` arm) rather than a
    // cast-removal helper inlined inside an arm body (`u32_as_i64` /
    // `u32_as_usize` / `op_arg_as_usize` / `raise_kind_as_usize`, now
    // reachable only past the unresolved handler). Documented in
    // `pyre/pyre-interpreter/src/pyopcode.rs:1302..1336`.
    //
    // Per-module scoping (Issue 1.3, 2026-05-05): `build_flow_from_rust`
    // mints a fresh `ModuleId` internally, so this test's lookup
    // partition is isolated from any other test's
    // `register_rust_module` walk. The rejection is therefore
    // strictly pre-walker — sibling tests' registry writes live
    // under different ids and cannot leak in. The pre-Issue-1.3
    // process-global pollution caveat (which forced this test to
    // accept a post-walker `Unsupported(closure)` outcome too) no
    // longer applies.
    let file = parse_pyopcode();
    let func = find_fn(&file, "execute_opcode_step");
    let err = build_flow_from_rust(func)
        .err()
        .expect("adapter still has un-roadmapped constructs to walk past");
    match err {
        AdapterError::UnboundLocal { name } => {
            // The first lifted per-opcode handler (`LoadConst` arm) is
            // the new pre-walker frontier; the cast-removal helpers stay
            // listed because they remain the rejection point inside any
            // arm body the walker does step into.
            const PRE_WALKER_UNRESOLVED_SYMBOLS: &[&str] = &[
                "execute_load_const",
                "u32_as_i64",
                "u32_as_usize",
                "op_arg_as_usize",
                "raise_kind_as_usize",
            ];
            assert!(
                PRE_WALKER_UNRESOLVED_SYMBOLS.contains(&name.as_str()),
                "without walker: expected an unresolved lifted handler or \
                 cast-removal helper from {PRE_WALKER_UNRESOLVED_SYMBOLS:?}, \
                 got {name:?}",
            );
        }
        other => panic!(
            "post-Issue-1.3 per-module scoping: a fresh ModuleId means no cross-test \
             pollution from sibling walks, so the rejection MUST be \
             `UnboundLocal(<cast-removal helper>)`. Got {other:?}"
        ),
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
