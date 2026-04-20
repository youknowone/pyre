//! End-to-end analysis pipeline.
//!
//! **LEGACY.** Majit-local driver that sequences the ad-hoc
//! `annotate → resolve_types → flatten_with_types` chain. Not the
//! same as RPython's `translator/driver.py` (TranslationDriver +
//! SimpleTaskEngine); re-identification was considered and rejected.
//! The rtyper-pipeline cutover (roadmap Phase 8) routes pyre compiles
//! through `majit-rtyper` directly and bypasses this file; this file
//! is deleted at roadmap commit P8.11.
//!
//! RPython-orthodox chain (when fully ported):
//!   flowspace → annotator → rtyper → jtransform → flatten
//!
//! This module provides a single entry point that runs all passes
//! in sequence on a SemanticProgram.

use serde::{Deserialize, Serialize};

use crate::OpcodeDispatchSelector;
use crate::flatten::{self, SSARepr};
use crate::front::SemanticFunction;
use crate::jtransform::{GraphTransformConfig, rewrite_graph};
use crate::translate_legacy::annotator::annrpython::annotate;
use crate::translate_legacy::rtyper::rtyper::resolve_types;

/// JitDriver portal binding.
///
/// RPython equivalent: `JitDriverStaticData.portal_graph` + the driver's
/// `greens=[...]`/`reds=[...]`/`virtualizables=[...]` declarations
/// (`rlib/jit.py::JitDriver`).
/// `CallControl.setup_jitdriver` consumes this to register the portal
/// entry point and its green/red layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortalSpec {
    /// Function name of the portal entry point (e.g. `"mainloop"` or
    /// `"execute_opcode_step"`).
    pub name: String,
    pub greens: Vec<String>,
    pub reds: Vec<String>,
    /// Optional explicit virtualizable red names. Empty means no
    /// virtualizable, which matches the common non-pyre case.
    #[serde(default)]
    pub virtualizables: Vec<String>,
    /// Optional red-type identities parallel to `reds`, mirroring the
    /// `_JIT_ENTER_FUNCTYPE.ARGS` information warmspot uses upstream.
    #[serde(default)]
    pub red_types: Vec<String>,
}

/// Configuration for the full analysis pipeline.
///
/// RPython: implicit in `CodeWriter.__init__` + `CallControl.__init__`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// jtransform configuration (virtualizable fields, call classification).
    pub transform: GraphTransformConfig,
    /// Portal binding for `CallControl.setup_jitdriver`. When `None` the
    /// pipeline falls back to the default pyre portal
    /// (`execute_opcode_step`) for backwards compatibility with the
    /// pyre-specific entry path.
    pub portal: Option<PortalSpec>,
}

/// Result of running the full pipeline on a single function.
///
/// RPython: the result of `transform_graph_to_jitcode()` — one per function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub name: String,
    pub original_blocks: usize,
    pub annotations_count: usize,
    pub concrete_types_count: usize,
    pub vable_rewrites: usize,
    pub calls_classified: usize,
    pub transform_notes: Vec<crate::jtransform::GraphTransformNote>,
    /// RPython: the SSARepr produced by flatten_graph().
    pub flattened: SSARepr,
}

/// Canonical opcode dispatch metadata.
///
/// RPython parity: PyPy's interpreter has one Python method per opcode
/// (e.g. `def LOAD_FAST(self, varindex, next_instr)`), and each method
/// is registered with `CallControl.get_jitcode(graph)` which assigns it
/// a slot in `all_jitcodes[]`. pyre's interpreter does the same dispatch
/// inside one big `match` instead of separate methods, so the parser
/// extracts each match arm body as its own synthetic graph and the
/// canonical pipeline registers it under
/// `CallPath::["__opcode_dispatch__", "<selector>#<arm_id>"]`. The result
/// is the same: each arm gets its own indexed jitcode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOpcodeArm {
    /// Sequential id assigned at extract time. Stable across runs.
    /// Identity for cross-references; selector string is display only.
    pub arm_id: usize,
    /// Display label only. Multi-pattern arms keep their `A | B` shape;
    /// the manifest layer expands variants downstream.
    pub selector: OpcodeDispatchSelector,
    /// Index into `ProgramPipelineResult.jitcodes` once the arm has been
    /// processed by `CodeWriter::drain_pending_graphs`. None if the arm
    /// has no body graph (rare).
    pub entry_jitcode_index: Option<usize>,
    /// Flattened SSARepr — kept for debug / snapshot diff. The orthodox
    /// pipeline produces its own flattened repr inside
    /// `transform_graph_to_jitcode`; this field is the parser-level view.
    pub flattened: Option<SSARepr>,
}

/// Result of running the pipeline on a full program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramPipelineResult {
    pub functions: Vec<PipelineResult>,
    pub opcode_dispatch: Vec<PipelineOpcodeArm>,
    /// RPython: all_jitcodes returned by CodeWriter.make_jitcodes() (codewriter.py:89).
    /// Assembled JitCode bytecode for each transformed graph. `Arc` so the
    /// shells handed out earlier (e.g. into
    /// `JitDriverStaticData.mainjitcode` or `IndirectCallTargets`) share
    /// identity with the values appearing here.
    pub jitcodes: Vec<std::sync::Arc<crate::jitcode::JitCode>>,
    /// RPython: `Assembler.insns` (assembler.py:?). The opcode-key → u8
    /// table grown on-demand by `write_insn`. Persisted alongside the
    /// jitcodes so the runtime can map bytecode bytes back to opnames —
    /// without it, the u8 opcodes embedded in `JitCode.code` are opaque
    /// (the mapping is local to the build-time assembler instance).
    /// Consumed by `BlackholeInterpBuilder::setup_insns` at runtime.
    #[serde(default)]
    pub insns: std::collections::HashMap<String, u8>,
    /// RPython: `Assembler.descrs` (assembler.py:23), consumed by
    /// `BlackholeInterpBuilder.setup_descrs(asm.descrs)`
    /// (blackhole.py:59, 102-103). Each 'd'/'j' argcode in a
    /// `JitCode.code` byte stream indexes into this shared descr pool
    /// to read field offsets / call descrs / sub-JitCodes.
    ///
    /// Persisted alongside `insns` so `BlackholeInterpBuilder` at
    /// runtime can call `setup_descrs(descrs)` and dispatch any 'd'/'j'
    /// argcode opname through the shared pool — matches RPython's
    /// single-store descr model.
    #[serde(default)]
    pub descrs: Vec<crate::jitcode::BhDescr>,
    pub total_blocks: usize,
    pub total_ops: usize,
    pub total_vable_rewrites: usize,
}

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
    let types = resolve_types(graph, &annotations);
    let concrete_types_count = types.concrete_types.len();

    // Pass 3: JIT transform (RPython jtransform)
    let transform_result = rewrite_graph(graph, &config.transform);
    let vable_rewrites = transform_result.vable_rewrites;
    let calls_classified = transform_result.calls_classified;
    let transform_notes = transform_result.notes.clone();
    let rewritten_types = resolve_types(&transform_result.graph, &annotations);

    // Pass 4: Flatten with type info (RPython flatten + regalloc)
    let value_kinds = crate::translate_legacy::rtyper::rtyper::build_value_kinds(&rewritten_types);
    let regallocs =
        crate::regalloc::perform_all_register_allocations(&transform_result.graph, &value_kinds);
    let flattened =
        flatten::flatten_with_types(&transform_result.graph, &rewritten_types, &regallocs);

    PipelineResult {
        name: func.name.clone(),
        original_blocks,
        annotations_count,
        concrete_types_count,
        vable_rewrites,
        calls_classified,
        transform_notes,
        flattened,
    }
}

/// Run the full pipeline on all functions in a program.
pub fn analyze_program(
    program: &crate::front::SemanticProgram,
    config: &PipelineConfig,
) -> ProgramPipelineResult {
    let mut functions = Vec::new();
    let mut total_blocks = 0;
    let mut total_ops = 0;
    let mut total_vable_rewrites = 0;

    for func in &program.functions {
        let result = analyze_function(func, config);
        total_blocks += result.original_blocks;
        total_ops += result.flattened.insns.len();
        total_vable_rewrites += result.vable_rewrites;
        functions.push(result);
    }

    ProgramPipelineResult {
        functions,
        opcode_dispatch: Vec::new(),
        jitcodes: Vec::new(),
        insns: std::collections::HashMap::new(),
        descrs: Vec::new(),
        total_blocks,
        total_ops,
        total_vable_rewrites,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::front;
    use crate::jtransform::GraphTransformConfig;

    #[test]
    fn pipeline_e2e_simple_function() {
        let parsed = crate::parse::parse_source(
            r#"
            fn add(a: i64, b: i64) -> i64 {
                a + b
            }
        "#,
        );
        let program = front::build_semantic_program(&parsed);
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
        let program = front::build_semantic_program(&parsed);
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
        let program = front::build_semantic_program(&parsed);
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
}
