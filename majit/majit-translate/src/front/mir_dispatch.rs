//! MIR-sourced opcode-dispatch arm extraction.
//!
//! The syn extractor (`parse::extract_opcode_dispatch_arms`) walks the
//! `execute_opcode_step` Rust `match` AST.  This module reconstructs the
//! same `Vec<ExtractedOpcodeArm>` from the *lowered* MIR `FunctionGraph`
//! of `execute_opcode_step`, so the JIT dispatch table can be built
//! without parsing the interpreter source.
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
//! the syn one-arm-per-source-arm structure.
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
//! The body graphs are built through [`crate::front::opcode_wrapper`],
//! the same builders the syn synthesizers use, so the two extractors
//! produce identical wrapper graphs.

use crate::flowspace::model::{ConstValue, Variable};
use crate::front::opcode_wrapper::{
    RaiseStubReturn, WrappedUnitVariantReturn, build_raise_stub_wrapper, build_tail_call_wrapper,
    build_wrapped_unit_variant_wrapper,
};
use crate::front::semantic::{MirGraphLookup, SemanticProgram};
use crate::model::{
    Block, CallTarget, ExitCase, ExitSwitch, FunctionGraph, OpKind, UnsupportedLiteralKind,
    ValueType,
};
use crate::parse::{CallPath, ExtractedOpcodeArm, OpcodeDispatchSelector};

/// The dispatcher prologue `(name, ValueType)` for `execute_opcode_step`,
/// matching `classify_fn_arg_ty` on its signature
/// `(executor: &mut E, code: &CodeObject, instruction: Instruction,
///   op_arg: OpArg, next_instr: usize)`.  Hard-coded rather than read
/// from the lowered Input ops because Charon erases the generic `E` and
/// the concrete reference targets to `Ref(None)` / `Int`, which would
/// not match the syn-built wrapper prologue.
fn dispatcher_params() -> Vec<(String, ValueType)> {
    vec![
        ("executor".to_string(), ValueType::Ref(Some("E".to_string()))),
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
fn selector_for(ks: &[i64], disc_to_name: &std::collections::HashMap<i64, String>) -> OpcodeDispatchSelector {
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
            Some((segments.last().unwrap().clone(), args.clone()))
        }
        _ => None,
    });
    if let Some((handler_leaf, args)) = tail_call {
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
        let handler_path = CallPath::from_segments([handler_leaf]);
        return build_tail_call_wrapper(name, params, &handler_path, &forwarded);
    }

    // raise-stub: a `PyError::type_error(…)` method call.
    let is_raise_stub = blk.operations.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name: m, .. },
                ..
            } if m == "type_error"
        )
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
/// not include the interpreter).  Mirrors the contract of
/// [`crate::parse::extract_opcode_dispatch_arms`].
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
        let body_graph =
            build_arm_body_graph(&selector.canonical_key(), graph.block(target), &params, &input_names);
        arms.push(ExtractedOpcodeArm {
            selector,
            handler_calls: Vec::new(),
            body_graph: Some(body_graph),
            mir_handler_path: None,
        });
    }
    if let Some(target) = default_target {
        let selector = OpcodeDispatchSelector::Wildcard;
        let body_graph =
            build_arm_body_graph(&selector.canonical_key(), graph.block(target), &params, &input_names);
        arms.push(ExtractedOpcodeArm {
            selector,
            handler_calls: Vec::new(),
            body_graph: Some(body_graph),
            mir_handler_path: None,
        });
    }
    arms
}

#[cfg(test)]
mod equivalence_tests {
    use super::*;
    use crate::model::LinkArg;

    /// Canonical key with `Or` cases sorted — the order-insensitive form
    /// for cross-checking against the syn extractor (within-`Or` source
    /// order is unrepresented in MIR).
    fn sorted_key(selector: &OpcodeDispatchSelector) -> String {
        match selector {
            OpcodeDispatchSelector::Or(cases) => {
                let mut parts: Vec<String> = cases.iter().map(sorted_key).collect();
                parts.sort();
                parts.join(" | ")
            }
            other => other.canonical_key(),
        }
    }

    fn var_index(idx: &mut std::collections::HashMap<Variable, usize>, v: &Variable) -> usize {
        let n = idx.len();
        *idx.entry(v.clone()).or_insert(n)
    }

    /// Structural fingerprint of a wrapper body graph, independent of
    /// `Variable` identity (fresh allocs differ between the syn and MIR
    /// builds) and of `graph.name` (the two extractors order `Or` cases
    /// differently).  Two wrappers with equal fingerprints lower
    /// identically.
    fn fingerprint(g: &FunctionGraph) -> String {
        use std::fmt::Write;
        let start = g.block(g.startblock);
        let mut idx: std::collections::HashMap<Variable, usize> = std::collections::HashMap::new();
        let mut out = String::new();
        for op in &start.operations {
            match &op.kind {
                OpKind::Input { name, ty, .. } => {
                    let r = op.result.as_ref().map(|v| var_index(&mut idx, v));
                    let _ = write!(out, "IN({name},{ty:?})->#{r:?};");
                }
                OpKind::Call {
                    target,
                    args,
                    result_ty,
                } => {
                    let aidx: Vec<usize> = args.iter().map(|a| var_index(&mut idx, a)).collect();
                    let r = op.result.as_ref().map(|v| var_index(&mut idx, v));
                    let t = match target {
                        CallTarget::FunctionPath { segments } => format!("FN[{}]", segments.join("::")),
                        CallTarget::SyntheticTransparentCtor { name, owner_path } => {
                            format!("CTOR[{}|{name}]", owner_path.join("::"))
                        }
                        CallTarget::Method {
                            name, receiver_root, ..
                        } => format!("M[{}::{name}]", receiver_root.clone().unwrap_or_default()),
                        other => format!("{other:?}"),
                    };
                    let _ = write!(out, "CALL({t};{aidx:?};{result_ty:?})->#{r:?};");
                }
                OpKind::Abort { kind } => {
                    let r = op.result.as_ref().map(|v| var_index(&mut idx, v));
                    let _ = write!(out, "ABORT({kind:?})->#{r:?};");
                }
                other => {
                    let _ = write!(out, "OP({other:?});");
                }
            }
        }
        if let Some(LinkArg::Value(v)) = start.exits.first().and_then(|l| l.args.first()) {
            let _ = write!(out, "RET(#{});", var_index(&mut idx, v));
        }
        out
    }

    fn signature_map(arms: &[ExtractedOpcodeArm]) -> std::collections::BTreeMap<String, String> {
        arms.iter()
            .map(|a| {
                let body = a.body_graph.as_ref().expect("real dispatch arm has a body graph");
                (sorted_key(&a.selector), fingerprint(body))
            })
            .collect()
    }

    /// The MIR extractor reproduces, arm-for-arm, the syn extractor's
    /// output: identical selector set (`Or` order aside) and identical
    /// body-graph structure per arm.  This is the parallel-equivalence
    /// anchor for flipping the dispatch source from syn to MIR.
    ///
    /// Skips when the interpreter LLBC artefact is absent (fresh checkout
    /// without `scripts/extract-llbc.sh`).
    #[test]
    fn syn_and_mir_dispatch_arms_are_equivalent() {
        let ullbc = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../build/llbc/pyre-interpreter.ullbc"
        );
        if !std::path::Path::new(ullbc).exists() {
            eprintln!("SKIP equivalence: {ullbc} absent (run scripts/extract-llbc.sh)");
            return;
        }
        let pyopcode = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../pyre/pyre-interpreter/src/pyopcode.rs"
        ))
        .expect("read pyopcode.rs");

        let parsed = crate::parse::parse_source(&pyopcode);
        let syn_arms = crate::parse::extract_opcode_dispatch_arms(&parsed);
        assert!(!syn_arms.is_empty(), "syn extractor found no arms");

        let llbc = majit_charon_reader::Llbc::load(ullbc).expect("load ullbc");
        let program = crate::front::mir::build_semantic_program_from_llbc(&llbc).expect("program");
        let mir_arms = extract_opcode_dispatch_arms_from_mir(&program);

        let syn_map = signature_map(&syn_arms);
        let mir_map = signature_map(&mir_arms);

        let syn_keys: Vec<&String> = syn_map.keys().collect();
        let mir_keys: Vec<&String> = mir_map.keys().collect();
        assert_eq!(
            syn_keys, mir_keys,
            "selector sets differ\n  syn-only: {:?}\n  mir-only: {:?}",
            syn_map.keys().filter(|k| !mir_map.contains_key(*k)).collect::<Vec<_>>(),
            mir_map.keys().filter(|k| !syn_map.contains_key(*k)).collect::<Vec<_>>(),
        );
        for (k, syn_fp) in &syn_map {
            assert_eq!(
                syn_fp, &mir_map[k],
                "body graph differs for selector `{k}`",
            );
        }
    }
}
