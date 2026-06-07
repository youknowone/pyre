//! Pipeline result types.
//!
//! Data carriers consumed by codegen + downstream tooling. The
//! production producer is `lib::analyze_pipeline_from_module_paths` (which
//! drives `build_canonical_opcode_dispatch` to fill `opcode_dispatch`,
//! `jitcodes`, `insns`, `descrs`).

use serde::{Deserialize, Serialize};

use crate::flatten::SSARepr;
use crate::jtransform::{GraphTransformConfig, GraphTransformNote};
use crate::opcode_dispatch::PipelineOpcodeArm;

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
#[derive(Debug, Clone, Serialize)]
pub struct PipelineResult {
    pub name: String,
    pub original_blocks: usize,
    pub annotations_count: usize,
    pub vable_rewrites: usize,
    pub transform_notes: Vec<GraphTransformNote>,
    /// RPython: the SSARepr produced by flatten_graph().
    ///
    /// This stays in-memory only. Build artifacts persist the assembled
    /// jitcodes and arm table, not the debug SSA dump.
    #[serde(skip, default)]
    pub flattened: SSARepr,
}

/// Result of running the pipeline on a full program.
#[derive(Debug, Clone, Serialize)]
pub struct ProgramPipelineResult {
    pub functions: Vec<PipelineResult>,
    pub opcode_dispatch: Vec<PipelineOpcodeArm>,
    /// RPython: all_jitcodes returned by CodeWriter.make_jitcodes() (codewriter.py:89).
    /// Assembled JitCode bytecode for each transformed graph. `Arc` so the
    /// shells handed out earlier (e.g. into
    /// `JitDriverStaticData.mainjitcode` or `IndirectCallTargets`) share
    /// identity with the values appearing here.
    pub jitcodes: Vec<std::sync::Arc<crate::jitcode::JitCode>>,
    /// RPython: `rpython/jit/codewriter/call.py:87 self.jitcodes`
    /// (graph-keyed dict). Pyre uses `CallPath` as graph identity at the
    /// module boundary. Paired with `jitcodes` (which mirrors
    /// `self.all_jitcodes` from `call.py:88`) so consumers can look up a
    /// JitCode either by alloc-order index or by graph key.
    ///
    /// Skipped by serde because serde_json cannot serialize a map
    /// keyed by a struct (`CallPath` is not a `String`). The
    /// `jit_metadata.json` round-trip used by `pyre-jit-trace/build.rs`
    /// does not need this view — it reads the alloc-ordered `jitcodes`
    /// vector directly. Consumers that require `by_path` read it from
    /// the live in-memory `ProgramPipelineResult`, not from the JSON
    /// artifact.
    #[serde(skip)]
    pub jitcodes_by_path:
        indexmap::IndexMap<crate::parse::CallPath, std::sync::Arc<crate::jitcode::JitCode>>,
    /// RPython: `Assembler.insns` (assembler.py:?). The opcode-key → u8
    /// table grown on-demand by `write_insn`. Persisted alongside the
    /// jitcodes so the runtime can map bytecode bytes back to opnames —
    /// without it, the u8 opcodes embedded in `JitCode.code` are opaque
    /// (the mapping is local to the build-time assembler instance).
    /// Consumed by `BlackholeInterpBuilder::setup_insns` at runtime.
    #[serde(default)]
    pub insns: majit_ir::vec_assoc::VecAssoc<String, u8>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::OpcodeDispatchSelector;
    use crate::flatten::{FlatOp, SSARepr};
    use crate::flowspace::model::ConstValue;
    use crate::jitcode::JitCode;
    use crate::opcode_dispatch::PipelineOpcodeArm;

    #[test]
    fn serialized_program_pipeline_skips_flattened_ssa_consts() {
        let flattened = SSARepr {
            name: "consts".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::byte_str("hello")),
            ))],
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
            jitcodes_by_path: indexmap::IndexMap::new(),
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
