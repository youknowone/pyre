//! Opcode dispatch arm metadata.
//!
//! PRE-EXISTING-ADAPTATION. PyPy's interpreter has one Python method per
//! opcode (`def LOAD_FAST(self, varindex, next_instr)`); each method is
//! registered with `CallControl.get_jitcode(graph)` and assigned a slot
//! in `all_jitcodes[]`. Pyre's interpreter dispatches opcodes inside one
//! big Rust `match` statement, so the parser extracts each match arm
//! body as its own synthetic graph and registers it under
//! `CallPath::["__opcode_dispatch__", "<selector>#<arm_id>"]`. The arm
//! table lives alongside the alloc-ordered jitcode vector so the runtime
//! (`pyre-jit-trace::jitcode_runtime`) can resolve a variant name →
//! `arm_id` → `entry_jitcode_index` → JitCode. There is no RPython
//! counterpart because PyPy's per-method model makes an explicit arm
//! table unnecessary.

use serde::{Deserialize, Serialize};

use crate::OpcodeDispatchSelector;
use crate::flatten::SSARepr;

/// Canonical opcode dispatch metadata.
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
    #[serde(skip, default)]
    pub flattened: Option<SSARepr>,
}
