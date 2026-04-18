//! `PyJitCode`: pyre's per-CodeObject JitCode wrapper.
//!
//! RPython's `JitCode` (jitcode.py:9) is a single class that owns
//! both the bytecode (`code` / `constants_*` / `c_num_regs_*`) and
//! the per-graph metadata (`name`, `fnaddr`, `calldescr`,
//! `jitdriver_sd`). pyre's runtime layer carries the same shape but
//! splits the storage:
//!
//!   * `majit_metainterp::jitcode::JitCode` holds the canonical
//!     bytecode container (`code`, `constants_*`, `num_regs_*`).
//!     It is the "compiled bytecode" half.
//!   * `PyJitCode` (this struct) wraps that JitCode together with
//!     pyre-only translation metadata — `pc_map` (Python PC → byte
//!     offset), `merge_point_pc`, register layout — that RPython
//!     does not need because RPython's bytecode PCs are already
//!     JitCode PCs.
//!
//! The struct lives in `pyre-jit-trace` (the lower crate) so that
//! both the codewriter (`pyre-jit::jit::codewriter`) and the
//! trace/blackhole runtime (this crate) can hold the same
//! `Arc<PyJitCode>` instances. RPython's `MetaInterpStaticData.jitcodes`
//! list and `CallControl.jitcodes` dict reference identical
//! `JitCode` Python objects via Python's reference semantics; pyre
//! mirrors that with shared `Arc` ownership of the unified shape
//! defined here.

use majit_metainterp::jitcode::{JitCode, LivenessInfo};

/// Per-Python-PC mapping from "logical PyFrame slot" to "JIT register
/// index". The blackhole consults this when materialising registers
/// from a concrete `PyFrame` on resume.
///
/// `Pinned` is the legacy/initial layout where every local at slot `i`
/// occupies JIT register `i` and every stack value at depth `d`
/// occupies register `nlocals + d`. After full regalloc lands
/// (`PerPc` variant), the JIT register file is decoupled from the
/// `PyFrame.locals_cells_stack_w` slot space — locals/stack values
/// can sit in any register, and the mapping varies per PC.
///
/// RPython parity: jitcode.py treats register indices as opaque
/// colors produced by `regalloc.py::perform_register_allocation`;
/// blackhole.py receives them through resume data + JitCode metadata
/// and never assumes "register == frame slot".
pub enum RegisterMapping {
    /// Initial layout: register index == PyFrame slot index. Used
    /// before regalloc decouples the spaces.
    Pinned {
        /// Number of fast locals; the boundary between "local" and
        /// "stack" register ranges in the legacy layout.
        nlocals: u16,
    },
    /// Per-PC variable layout produced by chordal coloring.
    PerPc {
        /// `local_to_reg[py_pc][local_idx]` = JIT register holding
        /// the value of local `local_idx` at PC `py_pc`, or
        /// `u16::MAX` if the local is dead at that PC.
        local_to_reg: Vec<Vec<u16>>,
        /// `stack_to_reg[py_pc][depth]` = JIT register holding the
        /// value of stack slot `depth` at PC `py_pc`. Length matches
        /// `depth_at_py_pc[py_pc]`.
        stack_to_reg: Vec<Vec<u16>>,
    },
}

impl RegisterMapping {
    /// JIT register that holds the value of fast local `local_idx`
    /// at Python PC `py_pc`. Returns `u16::MAX` if the local is
    /// dead at that PC (only possible in `PerPc` mode).
    pub fn register_for_local(&self, py_pc: usize, local_idx: usize) -> u16 {
        match self {
            RegisterMapping::Pinned { .. } => local_idx as u16,
            RegisterMapping::PerPc { local_to_reg, .. } => local_to_reg
                .get(py_pc)
                .and_then(|row| row.get(local_idx).copied())
                .unwrap_or(u16::MAX),
        }
    }

    /// JIT register that holds the value of stack slot `depth` at
    /// Python PC `py_pc`. Returns `u16::MAX` if no value is on the
    /// stack at that depth at that PC.
    pub fn register_for_stack(&self, py_pc: usize, depth: usize) -> u16 {
        match self {
            RegisterMapping::Pinned { nlocals } => *nlocals + depth as u16,
            RegisterMapping::PerPc { stack_to_reg, .. } => stack_to_reg
                .get(py_pc)
                .and_then(|row| row.get(depth).copied())
                .unwrap_or(u16::MAX),
        }
    }
}

/// Pyre-only metadata attached to a Python CodeObject's compiled JitCode.
///
/// RPython does not need these fields because its bytecode PCs are already
/// JitCode PCs. Pyre translates CPython bytecode to JitCode lazily, so the
/// translation maps live here instead of polluting the canonical JitCode.
pub struct PyJitCodeMetadata {
    /// py_pc → jitcode byte offset. Named for RPython's `frame.pc →
    /// jitcode position` flow; the runtime side reads this to map
    /// the Python frame's `next_instr` to the JitCode entry point
    /// for blackhole resume / inline call tracing.
    pub pc_map: Vec<usize>,
    /// Value-stack depth at each Python PC, in slots above stack_base.
    pub depth_at_py_pc: Vec<u16>,
    /// Register allocated for the portal's frame red argument.
    pub portal_frame_reg: u16,
    /// Register allocated for the portal's execution-context red argument.
    pub portal_ec_reg: u16,
    /// Absolute start index of the operand stack in PyFrame.locals_cells_stack_w.
    pub stack_base: usize,
    /// Pyre-local decoded liveness view used by `resume_in_blackhole`'s
    /// per-section register fill. The canonical packed bytes live on
    /// `MetaInterpStaticData.liveness_info` and are read via inline
    /// `-live-` offsets embedded in `JitCode.code`.
    pub liveness: Vec<LivenessInfo>,
    /// Per-PC mapping from PyFrame slot space to JIT register space.
    /// See [`RegisterMapping`] for the staged migration path.
    pub register_mapping: RegisterMapping,
}

/// Compiled JitCode plus pyre-only metadata.
///
/// Held by `Arc` so the same instance can sit in both
/// `MetaInterpStaticData.jitcodes` (the runtime list) and
/// `CallControl.jitcodes` (the codewriter dict) without duplicating
/// the bytecode buffer or metadata vectors. RPython's
/// `JitCode` references are shared the same way through Python's
/// refcount semantics.
pub struct PyJitCode {
    pub jitcode: std::sync::Arc<JitCode>,
    pub metadata: PyJitCodeMetadata,
    /// True if the jitcode contains BC_ABORT opcodes (unsupported bytecodes).
    /// Precomputed at compile time to avoid repeated bytecode scanning.
    pub has_abort: bool,
    /// Python PC of the jit_merge_point opcode (trace entry header).
    pub merge_point_pc: Option<usize>,
}

impl PyJitCode {
    /// Check if this jitcode has BC_ABORT opcodes.
    pub fn has_abort_opcode(&self) -> bool {
        self.has_abort
    }

    /// Empty `PyJitCode` slot inserted by `CallControl::get_jitcode`
    /// (call.py:168 `jitcode = JitCode(graph.name, fnaddr, calldescr, ...)`).
    ///
    /// In RPython the `JitCode` constructor returns a fresh object whose
    /// `code` / `descrs` / `liveness` arrays are all empty until
    /// `assembler.assemble(...)` populates them later in
    /// `make_jitcodes`'s drain loop (codewriter.py:80).  The skeleton
    /// gives the dict an entry with a stable identity so re-entrant
    /// `get_jitcode` calls (or pyre's `merge_point_pc` refinement
    /// shortcut) can find an existing key without recompiling.
    ///
    /// Until the drain replaces the slot, the only field with meaningful
    /// content is `merge_point_pc` (the refinement hint passed in by
    /// `get_jitcode`).
    pub fn skeleton(merge_point_pc: Option<usize>) -> Self {
        Self {
            jitcode: std::sync::Arc::new(JitCode::default()),
            metadata: PyJitCodeMetadata {
                pc_map: Vec::new(),
                depth_at_py_pc: Vec::new(),
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                stack_base: 0,
                liveness: Vec::new(),
                register_mapping: RegisterMapping::Pinned { nlocals: 0 },
            },
            has_abort: false,
            merge_point_pc,
        }
    }
}
