//! A trait associated-type projection must reach the SAME register bank
//! wherever it is spelled: production-LLBC regression for
//! `front::mir`'s unique-impl resolution of `TraitType`.
//!
//! `<Self as TruthOpcodeHandler>::Truth` is spelled two ways in the
//! interpreter.  A *required* method (`truth_value`,
//! `concrete_truth_as_bool`) is lowered from `impl BranchOpcodeHandler for
//! PyFrame`, where `Self` is concrete and the projection is already `bool`.
//! A generic caller (`opcode_pop_jump_if<H>`) and a *provided* default body
//! (`record_branch_guard`) are lowered from the trait, where it is not.
//! Without the resolution the two disagree on the bank of one value, and no
//! kind for the caller's variable satisfies both callees.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function_with_static_addrs;
use majit_translate::model::{CallTarget, OpKind, ValueType};
use majit_translate::{ErrorCarrierSpec, HostStaticAddrs};
use std::sync::OnceLock;

const INTERP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc",
);

/// Shared parse — see the same note on `test_result_exc_lowering.rs`: the
/// corpus is several GB resident, so one parse behind a `OnceLock` is what
/// keeps concurrent tests off the runner's swap.
fn interp() -> &'static Llbc {
    static L: OnceLock<Llbc> = OnceLock::new();
    L.get_or_init(|| Llbc::load(INTERP).expect("load pyre-interpreter.ullbc"))
}

fn lower(name: &str) -> majit_translate::model::FunctionGraph {
    lower_function_with_static_addrs(
        interp(),
        name,
        HostStaticAddrs {
            error_carrier: ErrorCarrierSpec {
                carrier_path: "pyre_interpreter::error::PyError",
                carrier_wrappers: &[],
                to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
                from_exc_object: Some(("PyError", "from_exc_object")),
            },
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("lower {name}: {e:?}"))
}

#[test]
fn a_generic_caller_types_an_assoc_type_result_through_the_unique_impl() {
    let graph = lower("pyre_interpreter::pyopcode::opcode_pop_jump_if");
    let mut seen = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target, result_ty, ..
            } = &op.kind
            else {
                continue;
            };
            let CallTarget::FunctionPath { segments } = target else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("truth_value") {
                continue;
            }
            seen += 1;
            assert_eq!(
                *result_ty,
                ValueType::Bool,
                "`truth_value` hands back `Self::Truth`, which the trait's only \
                 impl binds to `bool`.  Left unresolved it reads `Ref(None)` and \
                 the caller emits `inline_call_*_r` against a callee that \
                 `int_return`s"
            );
        }
    }
    assert_eq!(seen, 1, "opcode_pop_jump_if calls truth_value once");
}

#[test]
fn the_same_projection_resolves_in_a_provided_default_body() {
    // `opcode_unary_not` is the second caller of the same projection, and
    // `record_branch_guard` is the provided body that reaches it from the
    // trait side rather than from the impl.
    let graph = lower("pyre_interpreter::pyopcode::opcode_unary_not");
    let mut seen = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target, result_ty, ..
            } = &op.kind
            else {
                continue;
            };
            let CallTarget::FunctionPath { segments } = target else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("truth_value") {
                continue;
            }
            seen += 1;
            assert_eq!(*result_ty, ValueType::Bool);
        }
    }
    assert_eq!(seen, 1, "opcode_unary_not calls truth_value once");
    // The provided default body lowers at all and keeps its own call.
    lower("pyre_interpreter::pyopcode::BranchOpcodeHandler::record_branch_guard");
}
