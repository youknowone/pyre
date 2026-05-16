//! Legacy end-to-end analysis pipeline — transitional production driver.
//!
//! TODO(retire-legacy-pipeline): majit-local driver that sequences the
//! ad-hoc `legacy_annotator::annotate → legacy_resolve::resolve_types →
//! flatten_with_types` chain.  No upstream `rpython/` counterpart —
//! orthodox `translator/driver.py` (TranslationDriver +
//! SimpleTaskEngine) drives a different shape (per-function build_types
//! after BFS-from-entry, no separate flatten driver), so this file is
//! not a port target.
//!
//! The legacy chain remains the production analysis driver for
//! `lib.rs::analyze_pipeline_from_parsed` — the BFS-from-portal
//! candidate set drives the per-function annotate → rtype → jtransform
//! → flatten chain that feeds `build_canonical_opcode_dispatch`.
//! Retirement requires migrating the production caller to the
//! real-rtyper path ([`crate::translator::rtyper::cutover`]) end-to-end.
//!
//! RPython-orthodox chain (when fully ported):
//!   flowspace → annotator → rtyper → jtransform → flatten
//!
//! This module provides a single entry point that runs all passes
//! in sequence on a SemanticProgram.

use crate::call::CallControl;
use crate::flatten;
use crate::front::SemanticFunction;
use crate::pipeline::{PipelineConfig, PipelineResult, ProgramPipelineResult};
use crate::translator::rtyper::legacy_annotator::annotate;
use crate::translator::rtyper::legacy_resolve::resolve_types;

/// Run the full analysis pipeline on a single function.
///
/// RPython equivalent: translate a single function through
/// flowspace → annotator → rtyper → jtransform → flatten.
pub fn analyze_function(func: &SemanticFunction, config: &PipelineConfig) -> PipelineResult {
    let graph = &func.graph;
    let original_blocks = graph.blocks.len();

    // Pass 1: Annotation (RPython annotator)
    let annotations = annotate(graph);
    let annotations_count = annotations.types.len();

    // Pass 2: Type resolution (RPython rtyper)
    let mut types = resolve_types(graph, &annotations);

    // Pass 2b: rtyper-equivalent indirect_call lowering. RPython's rtyper
    // (rpbc.py:199-217) always emits `indirect_call(funcptr, *args,
    // c_graphs)` before jtransform sees the graph. Pyre's canonical
    // `codewriter::transform_graph_to_jitcode` runs this pass before
    // `rewrite_graph`; the legacy driver must do the same so callers that
    // consume `&dyn Trait` receivers (e.g. `pyre-jit/src/eval.rs`'s
    // `allocate_struct(typedescr: &dyn majit_ir::SizeDescr)`) do not trip
    // the `assert_no_indirect_call_targets` debug invariant inside
    // `rewrite_graph`. Legacy analyze is not plugged into CallControl, so
    // pass an empty one — `lower_indirect_calls` treats the resulting
    // empty `all_impls_for_indirect` family as "unknown" (graphs = None),
    // which is the conservative RPython-orthodox fallback.
    let mut legacy_callcontrol = CallControl::new();
    let mut graph_owned = graph.clone();
    crate::translator::rtyper::rpbc::lower_indirect_calls(
        &mut graph_owned,
        &mut types,
        &legacy_callcontrol,
    );

    // PyPy parity: hydrate every Variable's `concretetype` cell from the
    // rtyper-produced `types` table BEFORE jtransform runs, so jtransform
    // reads kinds via `graph.concretetype(v)` directly (the same
    // `getkind(v.concretetype)` path as upstream).  `with_type_state` is
    // still threaded as a belt-and-suspenders fallback for any slot that
    // the rtyper left Unknown — without it the legacy snapshot path
    // defaulted those slots to `'r'`, forcing `jtransform`'s
    // kind-coercion arms to manufacture `cast_ptr_to_int` ops.
    crate::jit_codewriter::type_state::apply_to_graph(&types, &mut graph_owned);

    // Pass 3: JIT transform (RPython jtransform) — thread the same empty
    // CallControl so `lower_indirect_call_op` has access to `getcalldescr`
    // / `guess_call_kind` / `graphs_from`. With no registered candidates
    // the op resolves to `CallKind::Residual`, matching upstream's
    // conservative fallback for `indirect_call` with unknown family.
    let transform_result = {
        let mut transformer = crate::jtransform::Transformer::new(&config.transform)
            .with_callcontrol(&mut legacy_callcontrol)
            .with_type_state(&types);
        transformer.transform(&graph_owned)
    };
    let vable_rewrites = transform_result.vable_rewrites;
    let transform_notes = transform_result.notes.clone();
    let rewritten_types = crate::translator::rtyper::legacy_resolve::resolve_rewritten_types(
        &types,
        &transform_result.graph,
        &annotations,
    );

    // Hydrate per-value `concretetype` cells on each backing
    // Variable held in `graph.value_variables` — the upstream
    // `Variable.concretetype` access pattern verbatim.  The
    // transitional `TypeResolutionState` then becomes a scratch the
    // regalloc projection still consumes; downstream consumers read
    // kinds via `graph.concretetype(v)`.
    let mut transform_result = transform_result;
    crate::jit_codewriter::type_state::apply_to_graph(
        &rewritten_types,
        &mut transform_result.graph,
    );

    // Pass 4: Flatten with type info (RPython flatten + regalloc)
    // Reads kinds straight off `graph.concretetype(v)` after the
    // canonical exceptblock stamp.  No `value_kinds` HashMap surface
    // any more — the graph IS the kind table.
    crate::regalloc::augment_canonical_exceptblock_on_graph(&mut transform_result.graph);
    let mut regallocs = crate::regalloc::perform_all_register_allocations(&transform_result.graph);
    // `flatten_graph` runs `enforce_input_args` (flatten.py:88-100)
    // internally as part of the upstream `flatten.py:63-66`
    // invocation order, so the startblock inputargs occupy the
    // dense `0..N` color prefix per kind and the rotation persists
    // into the assembler call.
    let flattened = flatten::flatten_graph(&transform_result.graph, &mut regallocs);

    PipelineResult {
        name: func.name.clone(),
        original_blocks,
        annotations_count,
        vable_rewrites,
        transform_notes,
        flattened,
    }
}

/// Run the full pipeline on all functions in a program.
pub fn analyze_program(
    program: &crate::front::SemanticProgram,
    config: &PipelineConfig,
) -> ProgramPipelineResult {
    analyze_program_filtered(&program.functions, config, |_| true)
}

/// Run the full pipeline on a filtered subset of functions.
///
/// RPython parity: `RPythonAnnotator.build_types(entry_point, args)` only
/// annotates / rtypes / lowers functions reachable from `entry_point` via
/// the call graph; unreachable helpers stay outside the translator
/// (`translator/translator.py:55 buildflowgraph` is invoked from BFS in
/// `annrpython.py:215 schedule_pending`).  Pyre's pre-cutover whole-
/// program iteration was a deviation: it dragged runtime-only resume
/// helpers (`materialize_virtual_from_rd`, `replay_pending_fields`)
/// through annotator → rtyper → flatten even though no JIT trace ever
/// reaches them, surfacing strict typing failures the JIT does not
/// actually need.  The filter closure lets the canonical caller pass a
/// `CallControl::is_candidate`-driven test so legacy_pipeline iterates
/// the same BFS-reachable set the canonical jitcode emitter consumes.
pub fn analyze_program_filtered<F>(
    functions: &[crate::front::SemanticFunction],
    config: &PipelineConfig,
    keep: F,
) -> ProgramPipelineResult
where
    F: Fn(&crate::front::SemanticFunction) -> bool,
{
    let mut analyzed = Vec::new();
    let mut total_blocks = 0;
    let mut total_ops = 0;
    let mut total_vable_rewrites = 0;

    for func in functions {
        if !keep(func) {
            continue;
        }
        let result = analyze_function(func, config);
        total_blocks += result.original_blocks;
        total_ops += result.flattened.insns.len();
        total_vable_rewrites += result.vable_rewrites;
        analyzed.push(result);
    }

    ProgramPipelineResult {
        functions: analyzed,
        opcode_dispatch: Vec::new(),
        jitcodes: Vec::new(),
        jitcodes_by_path: std::collections::HashMap::new(),
        insns: majit_ir::vec_assoc::VecAssoc::new(),
        descrs: Vec::new(),
        total_blocks,
        total_ops,
        total_vable_rewrites,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::OpcodeDispatchSelector;
    use crate::front;
    use crate::jitcode::JitCode;
    use crate::jtransform::GraphTransformConfig;
    use crate::opcode_dispatch::PipelineOpcodeArm;
    use crate::pipeline::{PipelineConfig, PipelineResult, ProgramPipelineResult};
    use crate::{
        flatten::{FlatOp, SSARepr},
        flowspace::model::ConstValue,
        model::LinkArg,
    };

    #[test]
    fn pipeline_e2e_simple_function() {
        let parsed = crate::parse::parse_source(
            r#"
            fn add(a: i64, b: i64) -> i64 {
                a + b
            }
        "#,
        );
        let program = front::build_semantic_program(&parsed).expect("source must lower");
        let config = PipelineConfig::default();
        let result = analyze_program(&program, &config);

        assert_eq!(result.functions.len(), 1);
        assert_eq!(result.functions[0].name, "add");
        assert!(result.functions[0].annotations_count > 0);
        assert!(result.functions[0].flattened.insns.len() > 0);
    }

    #[test]
    fn pipeline_e2e_with_virtualizable() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Frame { next_instr: usize, locals_w: Vec<i64> }
            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
            }
        "#,
        );
        let program = front::build_semantic_program(&parsed).expect("source must lower");
        let config = PipelineConfig {
            transform: GraphTransformConfig {
                vable_fields: vec![crate::jtransform::VirtualizableFieldDescriptor::new(
                    "next_instr",
                    Some("Frame".into()),
                    0,
                )],
                vable_arrays: vec![crate::jtransform::VirtualizableFieldDescriptor::new(
                    "locals_w",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let result = analyze_program(&program, &config);
        assert_eq!(result.functions.len(), 1);
        assert!(
            result.total_vable_rewrites > 0,
            "should rewrite next_instr to VableFieldRead, notes: {:?}",
            result.functions[0].transform_notes
        );
        // Should rewrite BOTH the field read (next_instr) AND the array read (locals_w[idx])
        let notes_str = format!("{:?}", result.functions[0].transform_notes);
        assert!(
            notes_str.contains("VableFieldRead"),
            "should contain VableFieldRead rewrite note"
        );
        assert!(
            notes_str.contains("VableArrayRead"),
            "should contain VableArrayRead rewrite note, got: {notes_str}"
        );
    }

    #[test]
    fn pipeline_e2e_with_control_flow() {
        let parsed = crate::parse::parse_source(
            r#"
            fn fib(n: i64) -> i64 {
                if n <= 1 {
                    return n;
                }
                let a = n - 1;
                let b = n - 2;
                a + b
            }
        "#,
        );
        let program = front::build_semantic_program(&parsed).expect("source must lower");
        let config = PipelineConfig::default();
        let result = analyze_program(&program, &config);

        let func = &result.functions[0];
        assert!(
            func.original_blocks >= 4,
            "if/else should create >=4 blocks, got {}",
            func.original_blocks
        );
        // Flattened should have jumps
        let has_jump = func.flattened.insns.iter().any(|op| {
            matches!(
                op,
                crate::flatten::FlatOp::Jump(_) | crate::flatten::FlatOp::GotoIfNot { .. }
            )
        });
        assert!(has_jump, "flattened fib should have conditional jumps");
    }

    #[test]
    fn serialized_program_pipeline_skips_flattened_ssa_consts() {
        let flattened = SSARepr {
            name: "consts".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::byte_str("hello")),
            ))],
            num_values: 0,
            num_blocks: 1,
            insns_pos: None,
        };
        let program = ProgramPipelineResult {
            functions: vec![PipelineResult {
                name: "consts".into(),
                original_blocks: 1,
                annotations_count: 0,
                vable_rewrites: 0,
                transform_notes: Vec::new(),
                flattened: flattened.clone(),
            }],
            opcode_dispatch: vec![PipelineOpcodeArm {
                arm_id: 7,
                selector: OpcodeDispatchSelector::Unsupported,
                entry_jitcode_index: Some(0),
                flattened: Some(flattened),
            }],
            jitcodes: vec![Arc::new(JitCode::new("consts"))],
            jitcodes_by_path: std::collections::HashMap::new(),
            insns: majit_ir::vec_assoc::VecAssoc::new(),
            descrs: Vec::new(),
            total_blocks: 1,
            total_ops: 1,
            total_vable_rewrites: 0,
        };

        let json = serde_json::to_string(&program).expect("program pipeline should serialize");
        assert!(
            !json.contains("flattened"),
            "serialized artifact should not persist debug SSA payloads"
        );
        serde_json::to_string(&program.opcode_dispatch)
            .expect("opcode dispatch artifact should serialize without SSARepr");
    }
}
