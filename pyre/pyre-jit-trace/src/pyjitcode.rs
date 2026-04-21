//! `PyJitCode`: pyre's per-CodeObject JitCode wrapper.
//!
//! RPython's `JitCode` (jitcode.py:9) is a single class that owns
//! both the bytecode (`code` / `constants_*` / `c_num_regs_*`) and
//! the per-graph metadata (`name`, `fnaddr`, `calldescr`,
//! `jitdriver_sd`). pyre still has a split runtime representation:
//!
//!   * `majit_metainterp::jitcode::JitCode` is the current runtime
//!     adapter bytecode container (`code`, `constants_*`, `num_regs_*`,
//!     plus pyre-only `exec.*` pools). It is not the canonical
//!     codewriter `majit_translate::jitcode::JitCode`.
//!   * `PyJitCode` (this struct) wraps that JitCode together with
//!     pyre-only translation metadata — `pc_map` (Python PC → byte
//!     offset), `merge_point_pc`, the runtime `w_code` wrapper, and
//!     register layout — that RPython
//!     does not need because RPython's bytecode PCs are already
//!     JitCode PCs.
//!
//! The struct lives in `pyre-jit-trace` (the lower crate) so that
//! both the codewriter (`pyre-jit::jit::codewriter`) and the
//! trace/blackhole runtime (this crate) can hold the same
//! `Arc<PyJitCode>` instances. RPython's `MetaInterpStaticData.jitcodes`
//! list and `CallControl.jitcodes` dict reference identical
//! `JitCode` Python objects via Python's reference semantics; pyre
//! mirrors the shared-identity part with `Arc<PyJitCode>`, but still
//! keeps the extra runtime adapter split described above.

use majit_metainterp::jitcode::JitCode as RuntimeJitCode;

/// Pyre-only metadata attached to a Python CodeObject's compiled JitCode.
///
/// RPython does not need these fields because its bytecode PCs are already
/// JitCode PCs. Pyre translates CPython bytecode to JitCode lazily, so the
/// translation maps live here instead of polluting either upstream's
/// canonical `JitCode` or pyre's eventual single-store replacement.
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
    pub jitcode: std::sync::Arc<RuntimeJitCode>,
    pub metadata: PyJitCodeMetadata,
    /// pyre-only: the `W_CodeObject` wrapper this jitcode was compiled
    /// for. Stored on the shared payload so `CallControl` can recover
    /// the runtime code identity during its `unfinished_graphs` drain
    /// without carrying a parallel `(graph, w_code, ...)` queue.
    pub w_code: *const (),
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

    /// "Has `assembler.assemble` been run on this jitcode yet?" A
    /// freshly-constructed RPython `JitCode(name, fnaddr, calldescr,
    /// ...)` (jitcode.py:14, call.py:168) leaves `self.code` unset
    /// until `setup` (jitcode.py:22) is invoked by
    /// `assembler.assemble(ssarepr, jitcode, num_regs)`
    /// (codewriter.py:67); pyre's split wrapper uses `pc_map.is_empty()`
    /// as the same "still a shell" test.
    pub fn is_populated(&self) -> bool {
        !self.metadata.pc_map.is_empty()
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
    /// Until the drain replaces the slot, the only fields with
    /// meaningful content are `w_code` and `merge_point_pc`.
    pub fn skeleton(w_code: *const (), merge_point_pc: Option<usize>) -> Self {
        Self {
            jitcode: std::sync::Arc::new(RuntimeJitCode::default()),
            metadata: PyJitCodeMetadata {
                pc_map: Vec::new(),
                depth_at_py_pc: Vec::new(),
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                stack_base: 0,
            },
            w_code,
            has_abort: false,
            merge_point_pc,
        }
    }
}
