//! Probe: run the Rust-AST adapter on the real
//! `pyre-interpreter::execute_opcode_step` portal.
//!
//! The adapter produces a complete `FunctionGraph` for
//! `execute_opcode_step<E>`, with every opcode branch represented and
//! every method call carrying a resolvable receiver classdef.
//!
//! The with-walker oracle asserts the adapter lowers
//! `execute_opcode_step` end-to-end. The without-walker oracle rejects
//! at the first lifted per-opcode handler (no per-module registry to
//! resolve it through), and the signature-shape oracle verifies
//! `validate_signature` independently. If any of the three fails, an
//! intermediate change silently broke; treat it as a parity regression.
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
    // ### How the adapter lowers the body end-to-end
    //
    // Match arms: or-patterns flatten into separate arms; unit enum
    // variants (`Pat::Path`), rest-only `Pat::Struct {..}` /
    // `Pat::TupleStruct(..)`, and struct-variant patterns with named-Ident
    // field bindings (e.g. `Instruction::LoadConst { consti }`) all lower.
    //
    // Host-environment resolution of standard-library constructors:
    // `Builder::resolve_path_constant` mirrors upstream
    // `flowcontext.py:856 LOAD_GLOBAL` + `:861 LOAD_ATTR` chain. The
    // closed-world `host_env::PYRE_STDLIB` registry resolves bare
    // `Ok` / `Some` / `Err` / `Result` / `Option` to
    // `Constant(HostObject(<class>))`; bare `None` resolves to
    // `Constant(ConstValue::None)`; multi-segment paths emit a `getattr`
    // cascade per `operation.py:618 getattr`, with the leftmost segment
    // minted on demand and cached on the Builder so two cascade steps that
    // name the same class share identity.
    //
    // `lower_match_variant_cascade` isinstance arg2 routes through
    // `Builder::resolve_path_constant`. Each cascade step block emits its
    // own `getattr` op per non-leftmost segment of the variant path, then
    // `isinstance(scrutinee, <leaf>)` per `operation.py:449`; identity
    // sharing across cascade steps (and across graphs) via the
    // process-global `host_env::HOST_CLASS_MINTS` registry.
    //
    // `lower_value_boundary` collapses `Ok(x)` / `Some(x)` / `None` AT
    // BOUNDARY positions only (function/arm tail, `return` operand);
    // value-position calls keep `simple_call(<host>, …)` per the
    // resolution above.
    //
    // `emit_err_raise_boundary` (full fork, PARITY) lowers
    // boundary-position `Err(e)` to the upstream
    // `flowcontext.py:600-636 exc_from_raise` op sequence with the 2-exit
    // `guessbool(isinstance(arg, type))` fork at `flowcontext.py:610`
    // preserved. True arm: `w_value = simple_call(evalue)`
    // (`flowcontext.py:614`, instantiate). False arm:
    // `w_value = ll_assert_not_none(evalue)` (`flowcontext.py:632-634`,
    // instance shape; the TypeError sub-arm is constant-folded out by
    // upstream's `guessbool(is_(w_arg2, const(None)))` since w_arg2 is
    // `const(None)` from `RAISE_VARARGS(1)`). Both arms converge on a join
    // block that emits `type(w_value)` and Links `[etype, w_value]` to
    // `graph.exceptblock` per `flowcontext.py:1259 Raise.nomoreblocks`.
    //
    // Module-globals walker: `register_rust_module(&syn::File)` walks
    // `pyopcode.rs` once with a try-build-then-register-on-success policy.
    // `Item::Fn`s whose bodies lower cleanly register as
    // `HostObject::UserFunction` carrying the prebuilt PyGraph; bodies the
    // walker rejects stay unregistered, falling back to the resolver's
    // mint-or-fail path. A deferred-body registration (register now, build
    // lazily) cannot work: `FunctionDesc.buildgraph` at `description.py:140`
    // only knows how to call `build_flow(GraphFunc)` against
    // `func.__code__.co_code`, but pyre's `HostCode` for an `Item::Fn`
    // carries empty bytecode, so a deferred sibling would supply an empty
    // body at lowering time — hence the eager prebuilt-graph path. The
    // entry-point fn is located via `file.items.iter().find_map(...)` in
    // `build_host_function_from_rust_file`, not registered as a sibling.
    //
    // Cast-removal helpers: `u32_as_i64`'s body uses `i64::from(x)`. The
    // lossless `From<u32> for i64` impl lowers as
    // `simple_call(getattr(<i64>, "from"), x)` per `lower_call`, mirroring
    // upstream's `LOAD_GLOBAL r_longlong; LOAD_FAST x; CALL_FUNCTION 1`
    // (the RPython idiom for explicit widening at
    // `rlib/rarithmetic.py:303`). The walker registers it with its
    // prebuilt graph so the cascade in `execute_opcode_step` resolves it.
    //
    // Together with closure-free LoadFast/LoadFastCheck varnames lookup,
    // `let _ = expr?;` lowering, the LoadSpecial wildcard tail arm, and the
    // `Expr::If` statement-position handling in `lower_block`, the
    // Position-2 adapter lowers `execute_opcode_step` end-to-end when
    // invoked alongside the module-globals walker.
    //
    // The probe pins the **success** state strictly:
    //
    // - `build_flow_from_rust_in_module` returns Ok(graph) for
    //   `execute_opcode_step` after `register_rust_module` has
    //   populated the per-module registry.
    // - The resulting `FunctionGraph` carries multiple blocks (the
    //   outer `match instruction` cascade) and `checkgraph` passes.
    // - If this regresses to an `Err(_)`, an intermediate change
    //   silently broke; treat as a parity regression and locate the
    //   cause via the error message.
    //
    // Per-module scoping: `register_rust_module` mints a fresh
    // `ModuleId` and returns it; `build_flow_from_rust_in_module`
    // threads the same id through body lowering so the cascade's
    // `LOAD_GLOBAL` resolutions hit the just-walked partition.
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
    // the adapter cannot resolve is the first lifted handler:
    // `execute_load_const`, the `LoadConst` arm (the arms before it —
    // `ExtendedArg`/`Resume`/`Nop`/`Cache`/`NotTaken` — return
    // `Ok(StepResult::Continue)` and need no symbol resolution). The
    // dispatch lives in `pyre/pyre-interpreter/src/pyopcode.rs`'s
    // `execute_opcode_step`. The cast-removal helpers (`u32_as_i64`
    // etc.) only appear inside arm bodies the walker never enters in
    // this without-walker path, so they cannot be the frontier here.
    //
    // Per-module scoping: `build_flow_from_rust` mints a fresh
    // `ModuleId` internally, so this test's lookup partition is
    // isolated from any other test's `register_rust_module` walk. The
    // rejection is therefore strictly pre-walker — sibling tests'
    // registry writes live under different ids and cannot leak in.
    let file = parse_pyopcode();
    let func = find_fn(&file, "execute_opcode_step");
    let err = build_flow_from_rust(func)
        .err()
        .expect("adapter still has un-roadmapped constructs to walk past");
    match err {
        AdapterError::UnboundLocal { name } => {
            // The first lifted per-opcode handler (`LoadConst` arm) is
            // the without-walker frontier: the adapter resolves the
            // leading `Ok(StepResult::Continue)` arms cleanly and stops
            // at the first body-bearing arm's tail call. Pinning the
            // exact symbol keeps the oracle from silently passing if the
            // adapter ever regressed past the lifted-handler boundary
            // into an arm body.
            assert_eq!(
                name, "execute_load_const",
                "without walker: the first lifted per-opcode handler \
                 (`LoadConst` arm) is the pre-walker frontier; got {name:?}",
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
