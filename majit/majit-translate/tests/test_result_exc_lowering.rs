//! Result-of-PyError → exception-link lowering: production-LLBC
//! regression tests for `front::result_exc`.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, ExitSwitch, OpKind};

const INTERP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc",
);

#[test]
fn pop_value_lowers_to_raise_links() {
    let llbc = Llbc::load(INTERP).expect("load pyre-interpreter.ullbc");
    let graph = lower_function(&llbc, "pyre_interpreter::eval::<Impl>::pop_value")
        .expect("lower pop_value");
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
    let llbc = Llbc::load(INTERP).expect("load pyre-interpreter.ullbc");
    // The SFSF chain's free-fn body pops twice via `?`
    // (pyopcode.rs `opcode_store_fast_store_fast`).
    let graph = lower_function(&llbc, "opcode_store_fast_store_fast").expect("lower caller");
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
    let llbc = Llbc::load(INTERP).expect("load pyre-interpreter.ullbc");
    // The arm wrapper's `Ok(StepResult::Continue)` shell must be gone
    // and the `?` on the scoped `store_fast_store_fast` method must be
    // a LastException diamond.
    let graph = lower_function(
        &llbc,
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

#[test]
fn eval_loop_custom_match_gets_catch_and_rewrap() {
    let llbc = Llbc::load(INTERP).expect("load pyre-interpreter.ullbc");
    let graph =
        lower_function(&llbc, "pyre_interpreter::eval::eval_loop").expect("lower eval_loop");
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
