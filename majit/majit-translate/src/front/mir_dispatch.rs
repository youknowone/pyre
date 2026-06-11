//! MIR-sourced opcode-dispatch arm extraction.
//!
//! Builds the `Vec<ExtractedOpcodeArm>` JIT dispatch table from the
//! *lowered* MIR `FunctionGraph` of `execute_opcode_step`, so the table
//! is built without parsing the interpreter source.
//!
//! ## Shape of the lowered dispatcher
//!
//! `lookup_free("execute_opcode_step")` resolves the dispatcher graph.
//! Its startblock holds the five `OpKind::Input` dispatcher params and a
//! discriminant switch (`ExitSwitch::Value`).  Each switch exit carries
//! `ExitCase::Const(Int(k))` where `k` is the `Instruction` variant's
//! *discriminant* (not its index — they diverge), plus one default exit.
//! Exits that share a target block are the one source `A | B`-pattern arm
//! flattened by match lowering, so grouping by target block reproduces
//! the one-arm-per-source-arm structure.
//!
//! Each target block is one of three closed shapes:
//! - **tail-call** — a single `Call` to a `FunctionPath` whose leaf is
//!   `execute_<op>`; the forwarded dispatcher params are the call args.
//! - **const-return** — `Ok(StepResult::Continue)` (the meta opcodes and
//!   `ExitInitCheck`); lowering erases the variant names to `Adt` ctors,
//!   so the fixed template is emitted by shape recognition.
//! - **raise-stub** — `Err(PyError::type_error("…").into())` (the async
//!   stubs); likewise emitted from a fixed template.
//!
//! The body graphs are built through [`crate::front::opcode_wrapper`].

use crate::flowspace::model::{ConstValue, Variable};
use crate::front::opcode_wrapper::{
    RaiseStubReturn, WrappedUnitVariantReturn, build_raise_stub_wrapper, build_tail_call_wrapper,
    build_wrapped_unit_variant_wrapper,
};
use crate::front::semantic::{MirGraphLookup, SemanticProgram};
use crate::model::{
    Block, CallTarget, ExitCase, ExitSwitch, FunctionGraph, LinkArg, OpKind,
    UnsupportedLiteralKind, ValueType,
};
use crate::parse::{CallPath, ExtractedOpcodeArm, OpcodeDispatchSelector};

/// The dispatcher prologue `(name, ValueType)` for `execute_opcode_step`,
/// matching its signature
/// `(executor: &mut E, code: &CodeObject, instruction: Instruction,
///   op_arg: OpArg, next_instr: usize)`.  Hard-coded rather than read
/// from the lowered Input ops because Charon erases the generic `E` and
/// the concrete reference targets to `Ref(None)` / `Int`, which would
/// not match the expected wrapper prologue.
fn dispatcher_params() -> Vec<(String, ValueType)> {
    vec![
        (
            "executor".to_string(),
            ValueType::Ref(Some("E".to_string())),
        ),
        (
            "code".to_string(),
            ValueType::Ref(Some("CodeObject".to_string())),
        ),
        (
            "instruction".to_string(),
            ValueType::Ref(Some("Instruction".to_string())),
        ),
        (
            "op_arg".to_string(),
            ValueType::Ref(Some("OpArg".to_string())),
        ),
        ("next_instr".to_string(), ValueType::Unsigned),
    ]
}

/// Fixed `Ok(StepResult::Continue)` const-return template.
fn continue_return_template() -> WrappedUnitVariantReturn {
    WrappedUnitVariantReturn {
        wrapper_owner: Vec::new(),
        wrapper_name: "Ok".to_string(),
        inner_owner: vec!["StepResult".to_string()],
        inner_name: "Continue".to_string(),
    }
}

/// Fixed `Err(crate::PyError::type_error("…").into())` raise-stub template.
fn async_raise_stub_template() -> RaiseStubReturn {
    RaiseStubReturn {
        fn_segments: vec![
            "crate".to_string(),
            "PyError".to_string(),
            "type_error".to_string(),
        ],
        method_name: "into".to_string(),
        wrapper_owner: Vec::new(),
        wrapper_name: "Err".to_string(),
        literal: UnsupportedLiteralKind::Str,
    }
}

/// Build the [`OpcodeDispatchSelector`] for a target block's discriminant
/// set.  A single discriminant yields `Path(["Instruction", variant])`;
/// multiple yield an `Or` of those (the source `A | B` arm), emitted in
/// ascending-discriminant order (the within-`Or` source order is not
/// represented in MIR).
fn selector_for(
    ks: &[i64],
    disc_to_name: &std::collections::HashMap<i64, String>,
) -> OpcodeDispatchSelector {
    let mut sorted = ks.to_vec();
    sorted.sort_unstable();
    let mut paths: Vec<OpcodeDispatchSelector> = sorted
        .iter()
        .map(|k| {
            let name = disc_to_name.get(k).unwrap_or_else(|| {
                panic!("opcode dispatch: no Instruction variant for discriminant {k}")
            });
            OpcodeDispatchSelector::Path(CallPath::from_segments([
                "Instruction".to_string(),
                name.clone(),
            ]))
        })
        .collect();
    if paths.len() == 1 {
        paths.pop().unwrap()
    } else {
        OpcodeDispatchSelector::Or(paths)
    }
}

/// Classify a switch-target block and build its dispatcher-shaped wrapper
/// graph via the shared [`crate::front::opcode_wrapper`] builders.
fn build_arm_body_graph(
    name: &str,
    blk: &Block,
    params: &[(String, ValueType)],
    input_names: &[(Variable, String)],
) -> FunctionGraph {
    // tail-call: a single Call to FunctionPath `…::execute_<op>`.
    let tail_call = blk.operations.iter().find_map(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if segments
            .last()
            .is_some_and(|leaf| leaf.starts_with("execute_")) =>
        {
            Some((segments.clone(), args.clone()))
        }
        _ => None,
    });
    if let Some((handler_segments, args)) = tail_call {
        let handler_leaf = handler_segments
            .last()
            .cloned()
            .expect("tail-call segments are non-empty (leaf matched above)");
        let forwarded: Vec<String> = args
            .iter()
            .map(|arg| {
                input_names
                    .iter()
                    .find(|(var, _)| var == arg)
                    .map(|(_, n)| n.clone())
                    .unwrap_or_else(|| {
                        panic!(
                            "opcode arm `{name}`: handler `{handler_leaf}` receives an argument \
                             that is not a dispatcher parameter"
                        )
                    })
            })
            .collect();
        // Keep the dispatcher callsite's full Charon-qualified segments.
        // `CallControl::find_all_graphs_bfs` walked this same callsite
        // (the switch-target block of `execute_opcode_step`) and inserted
        // the qualified spelling into `candidate_graphs`; truncating to
        // the bare leaf here made `guess_call_kind` resolve the wrapper's
        // tail-call through a non-candidate alias spelling and classify it
        // `Residual` — every arm degenerated to a blackbox
        // `residual_call_r_r(<symbolic hash>, …)` instead of the
        // `inline_call` into the already-drained per-opcode jitcode
        // (`call.py:117-139 guess_call_kind` keys candidacy on graph
        // identity; pyre's path-keyed set needs the spelling to match).
        let handler_path = CallPath::from_segments(handler_segments);
        return build_tail_call_wrapper(name, params, &handler_path, &forwarded);
    }

    // raise-stub: a `PyError::type_error(…)` call.  An associated
    // function lowers as `FunctionPath` (no `self` receiver); the
    // `Method` form is kept for receiver-shaped lowerings.
    let is_raise_stub = blk.operations.iter().any(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            ..
        } => m == "type_error",
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            ..
        } => segments.last().is_some_and(|leaf| leaf == "type_error"),
        _ => false,
    });
    if is_raise_stub {
        return build_raise_stub_wrapper(name, params, &async_raise_stub_template());
    }

    // const-return: `Ok(StepResult::Continue)` lowered to Adt ctors.
    let is_const_return = blk.operations.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            }
        )
    });
    if is_const_return {
        return build_wrapped_unit_variant_wrapper(name, params, &continue_return_template());
    }

    panic!("opcode arm `{name}`: target block matches no known dispatch shape");
}

/// Reconstruct the opcode-dispatch arms from the lowered MIR program.
///
/// Returns an empty vector when the dispatcher graph is absent or does
/// not have the discriminant-switch shape (e.g. an LLBC set that does
/// not include the interpreter).
pub fn extract_opcode_dispatch_arms_from_mir(program: &SemanticProgram) -> Vec<ExtractedOpcodeArm> {
    let lookup = MirGraphLookup::from_program(program);
    let Some(graph) = lookup.lookup_free("execute_opcode_step") else {
        return Vec::new();
    };
    let start = graph.block(graph.startblock);
    if !matches!(start.exitswitch, Some(ExitSwitch::Value(_))) {
        return Vec::new();
    }
    let Some(disc_to_name) = program.enum_variant_by_discriminant.get("Instruction") else {
        return Vec::new();
    };

    // startblock Input result var → param name, for forwarded-arg mapping.
    let input_names: Vec<(Variable, String)> = start
        .operations
        .iter()
        .filter_map(|op| match &op.kind {
            OpKind::Input { name, .. } => op.result.clone().map(|res| (res, name.clone())),
            _ => None,
        })
        .collect();

    let params = dispatcher_params();

    // Group case exits by target block (first-encounter = discriminant
    // order); the single non-Int exit is the `_ =>` default.
    let mut groups: Vec<(crate::model::BlockId, Vec<i64>)> = Vec::new();
    let mut default_target: Option<crate::model::BlockId> = None;
    for link in &start.exits {
        match &link.exitcase {
            Some(ExitCase::Const(ConstValue::Int(k))) => {
                if let Some(slot) = groups.iter_mut().find(|(t, _)| *t == link.target) {
                    slot.1.push(*k);
                } else {
                    groups.push((link.target, vec![*k]));
                }
            }
            _ => default_target = Some(link.target),
        }
    }

    let mut arms: Vec<ExtractedOpcodeArm> = Vec::with_capacity(groups.len() + 1);
    for (target, ks) in groups {
        let selector = selector_for(&ks, disc_to_name);
        let names = arm_input_names(&input_names, start, target, graph);
        let body_graph = build_arm_body_graph(
            &selector.canonical_key(),
            graph.block(target),
            &params,
            &names,
        );
        arms.push(ExtractedOpcodeArm {
            selector,
            body_graph: Some(body_graph),
            mir_handler_path: None,
        });
    }
    if let Some(target) = default_target {
        let selector = OpcodeDispatchSelector::Wildcard;
        let names = arm_input_names(&input_names, start, target, graph);
        let body_graph = build_arm_body_graph(
            &selector.canonical_key(),
            graph.block(target),
            &params,
            &names,
        );
        arms.push(ExtractedOpcodeArm {
            selector,
            body_graph: Some(body_graph),
            mir_handler_path: None,
        });
    }
    arms
}

/// Extend the startblock parameter map with a switch-target block's own
/// `inputargs`, resolved through the dispatch link that feeds them.
///
/// A `Link` renames each source `args[i]` into the target block's
/// `inputargs[i]` (`flowspace/model.py:114-116`).  An arm whose handler
/// forwards an arm-local inputarg — e.g. a single-`&mut`-receiver call whose
/// executor reborrow is threaded as a block parameter rather than referencing
/// the startblock Input directly — is forwarding the same dispatcher parameter
/// under a renamed `Variable`.  Resolving the feeding link restores the
/// parameter name so the wrapper builder forwards the correct slot instead of
/// rejecting the inputarg as unknown.
fn arm_input_names(
    base: &[(Variable, String)],
    start: &Block,
    target: crate::model::BlockId,
    graph: &FunctionGraph,
) -> Vec<(Variable, String)> {
    let mut names = base.to_vec();
    let Some(link) = start.exits.iter().find(|l| l.target == target) else {
        return names;
    };
    for (i, inputarg) in graph.block(target).inputargs.iter().enumerate() {
        if let Some(LinkArg::Value(src)) = link.args.get(i)
            && let Some((_, name)) = base.iter().find(|(v, _)| v == src)
        {
            names.push((inputarg.clone(), name.clone()));
        }
    }
    names
}
