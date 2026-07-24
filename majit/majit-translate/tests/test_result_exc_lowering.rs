//! Result-of-PyError → exception-link lowering: production-LLBC
//! regression tests for `front::result_exc`.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, ExitSwitch, OpKind};
use std::sync::OnceLock;

const INTERP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc",
);

/// Load `pyre-interpreter.ullbc` once and share it across every test.
///
/// The corpus is ~224MB on disk and its parsed `serde_json` form is
/// several GB resident.  Loading it per test means the four tests here
/// — run concurrently by the default test harness — each hold a full
/// parse resident at the same time, several times the runner's RAM on
/// the 16GB CI hosts; the resulting OOM/swap-thrash gets the job killed
/// (Linux), where a developer machine with more headroom only runs
/// slowly.  `Llbc` is read-only after `load`, so a single shared parse
/// behind a `OnceLock` is sufficient: `get_or_init` runs the load
/// exactly once even under the concurrent test threads, and
/// `lower_function` only borrows it.
fn interp() -> &'static Llbc {
    static LLBC: OnceLock<Llbc> = OnceLock::new();
    LLBC.get_or_init(|| Llbc::load(INTERP).expect("load pyre-interpreter.ullbc"))
}

#[test]
fn unit_result_callee_declares_void_return() {
    // A `Result<(), PyError>` scoped callee returns void after the
    // exception-link lowering, so `front::mir` stamps `return_type =
    // "()"` (`FUNC.RESULT = void`); the codewriter then collapses the
    // returnblock to a genuine void return post-annotation.  Without the
    // stamp the call descriptor would read `r` for the `Ref`-typed unit
    // `()` shell.  Covered: the callee-rule `Ok(())` (`store_local_value`)
    // and the tail-forward `f(...)?` (`store_fast` / `store_fast_store_fast`).
    for name in [
        "pyre_interpreter::eval::<Impl>::store_local_value",
        "pyre_interpreter::pyopcode::OpcodeStepExecutor::store_fast",
        "pyre_interpreter::pyopcode::OpcodeStepExecutor::store_fast_store_fast",
    ] {
        let g = lower_function(interp(), name).expect("lower");
        assert_eq!(
            g.return_type.as_deref(),
            Some("()"),
            "{name}: Result<(), PyError> callee must declare FUNC.RESULT = void"
        );
    }

    // A non-unit `Result<T, PyError>` callee is not void-widened:
    // `pop_value` returns `Result<PyObjectRef, PyError>`, so it carries
    // no void stamp and keeps its single ref return variable.
    let pv = lower_function(interp(), "pyre_interpreter::eval::<Impl>::pop_value")
        .expect("lower pop_value");
    assert_ne!(
        pv.return_type.as_deref(),
        Some("()"),
        "pop_value returns a value, not void"
    );
    assert_eq!(
        pv.block(pv.returnblock).inputargs.len(),
        1,
        "pop_value returns a value; its returnblock keeps the return var"
    );
}

#[test]
fn pop_value_lowers_to_raise_links() {
    let llbc = interp();
    let graph =
        lower_function(llbc, "pyre_interpreter::eval::<Impl>::pop_value").expect("lower pop_value");
    let mut result_ctors = 0usize;
    let mut to_exc_object_calls = 0usize;
    let mut except_links = 0usize;
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { owner_path, .. },
                    ..
                } if owner_path.last().map(String::as_str) == Some("Result") => {
                    result_ctors += 1;
                }
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } if name == "to_exc_object" => to_exc_object_calls += 1,
                _ => {}
            }
        }
        for link in &b.exits {
            if link.target == graph.exceptblock {
                except_links += 1;
            }
        }
    }
    assert_eq!(result_ctors, 0, "Result shells must be gone");
    assert_eq!(
        to_exc_object_calls, 1,
        "Err arm materialises the exception object"
    );
    assert!(except_links >= 1, "Err arm raises towards exceptblock");
    eprintln!("pop_value: to_exc_object={to_exc_object_calls} except_links={except_links}");
}

#[test]
fn pop_value_caller_gets_lastexception_exits() {
    let llbc = interp();
    // The SFSF chain's free-fn body pops twice via `?`
    // (pyopcode.rs `opcode_store_fast_store_fast`).
    let graph = lower_function(llbc, "opcode_store_fast_store_fast").expect("lower caller");
    eprintln!("caller graph = {}", graph.name);
    let lastexc_blocks = graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .count();
    let branch_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(&op.kind, OpKind::Call { target: CallTarget::Method { name, .. }, .. } if name == "branch")
        })
        .count();
    eprintln!("caller: lastexc_blocks={lastexc_blocks} branch_calls={branch_calls}");
    assert!(
        lastexc_blocks >= 1,
        "pop_value call sites get LastException exits"
    );
}

/// Count Result-shell ctors (`SyntheticTransparentCtor` with owner
/// `core::result::Result`) in a lowered graph.
fn count_result_ctors(graph: &majit_translate::model::FunctionGraph) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { owner_path, .. },
                    ..
                } if owner_path.last().map(String::as_str) == Some("Result")
            )
        })
        .count()
}

#[test]
fn execute_wrapper_family_lowers_to_raise_links() {
    let llbc = interp();
    // The arm wrapper's `Ok(StepResult::Continue)` shell must be gone
    // and the `?` on the scoped `store_fast_store_fast` method must be
    // a LastException diamond.
    let graph = lower_function(
        llbc,
        "pyre_interpreter::pyopcode::execute_store_fast_store_fast",
    )
    .expect("lower wrapper");
    assert_eq!(
        count_result_ctors(&graph),
        0,
        "wrapper Result shells must be gone"
    );
    let lastexc_blocks = graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .count();
    assert!(lastexc_blocks >= 1, "wrapper `?` gets LastException exits");
}

/// Facet A firing guard — the jd1 drain-loop `match next()` fusion.
///
/// `_unpackiterable_unknown_length`'s StopIteration drain loop is a
/// hand-written `match next() { Ok(w) => append, Err(e) if e.kind ==
/// StopIteration => break, Err(e) => return Err(e) }`.  Lowered naively it
/// materialises a `Result` shell whose Err arm holds a `StopIteration`
/// `SyntheticTransparentCtor` + a `PyErrorKind::eq` — residuals with no host
/// funcptr that SIGBUS the jd1 walk.  `try_fuse_drain_match`
/// (`front::result_exc`) replaces that shell with a `LastException`
/// exception-edge whose handler is the exact-kind test
/// `exc_kind_discriminant(evalue) == 10` (`ExcKind::StopIteration`).
///
/// The fusion is FAIL-SAFE: on any shape it does not recognise it silently
/// falls back to `catch_and_rewrap`, which leaves the `StopIteration` ctor in
/// place.  That silent decline is invisible to the correctness suite — the
/// default (non-jd1) run never executes the fusion, and the
/// `unpack_drain_exact_kind` parity test only guards the default path — yet it
/// reintroduces the unwalkable ctors and reopens the jd1 SIGBUS.  A drain
/// rework that perturbs the recognised shape (or a recognizer regression) is
/// exactly such a silent decline.  This lowers the REAL drain and asserts the
/// fused signature is present and the ctor is gone, so a decline fails loud.
#[test]
fn unpackiterable_drain_match_fuses_to_kind_test() {
    let llbc = interp();
    let graph = lower_function(
        llbc,
        "pyre_interpreter::baseobjspace::_unpackiterable_unknown_length",
    )
    .expect("lower _unpackiterable_unknown_length");

    // Positive firing signal: the fusion synthesises the exc_kind_discriminant
    // kind-test call — the only site in the tree that emits this FunctionPath,
    // so its presence proves `try_fuse_drain_match` fired (not declined).
    let exc_kind_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().map(String::as_str) == Some("exc_kind_discriminant")
            )
        })
        .count();
    assert!(
        exc_kind_calls >= 1,
        "drain fusion must synthesise the exc_kind_discriminant kind-test \
         (0 = recognizer silently declined to catch_and_rewrap → the \
         StopIteration ctor/eq residuals remain and SIGBUS the jd1 walk)"
    );

    // Elimination signal: the StopIteration guard ctor survives ONLY on the
    // decline (catch_and_rewrap) path, so a fired fusion leaves none.
    let stopiteration_ctors = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, .. },
                    ..
                } if name == "StopIteration"
            )
        })
        .count();
    assert_eq!(
        stopiteration_ctors, 0,
        "the StopIteration guard ctor must be gone after the drain fusion"
    );

    // The fused next() call site carries a LastException exit.
    let lastexc_blocks = graph
        .blocks
        .iter()
        .filter(|b| matches!(b.exitswitch, Some(ExitSwitch::LastException)))
        .count();
    assert!(
        lastexc_blocks >= 1,
        "the drain next() site must become a LastException exception-edge"
    );
    eprintln!(
        "drain fusion: exc_kind_discriminant={exc_kind_calls} \
         stopiteration_ctors={stopiteration_ctors} lastexc_blocks={lastexc_blocks}"
    );
}

#[test]
fn eval_loop_custom_match_gets_catch_and_rewrap() {
    let llbc = interp();
    let graph = lower_function(llbc, "pyre_interpreter::eval::eval_loop").expect("lower eval_loop");
    // The execute_opcode_step call block must carry LastException exits.
    let call_block = graph
        .blocks
        .iter()
        .find(|b| {
            b.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                        if segments.last().map(String::as_str) == Some("execute_opcode_step")
                )
            })
        })
        .expect("eval_loop calls execute_opcode_step");
    assert!(
        matches!(call_block.exitswitch, Some(ExitSwitch::LastException)),
        "custom-match call site gets catch-and-rewrap LastException exits"
    );
    // The exception arm re-binds the caught value into the PyError
    // domain before rebuilding the Err shell.
    let from_exc_calls = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                    if name == "from_exc_object"
            )
        })
        .count();
    assert!(
        from_exc_calls >= 1,
        "rewrap exception arm binds PyError::from_exc_object(last_exc_value)"
    );
}
