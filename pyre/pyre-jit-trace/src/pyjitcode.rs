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
    /// Phase 2 commit 2.1 (Tasks #158/#159/#122 epic, plan
    /// `~/.claude/plans/staged-sauteeing-koala.md`): post-regalloc
    /// color of each Python-semantic stack slot.
    /// `stack_slot_color_map[d]` = `apply_rename(Kind::Ref, stack_base + d)`
    /// for `d in 0..max_stackdepth`. Populated in `finalize_jitcode`
    /// after `apply_rename` runs.
    ///
    /// Currently with stack-slot input-arg pinning (regalloc.rs:455-466),
    /// this is identical to `[stack_base, stack_base+1, ..., stack_base+max-1]`
    /// — i.e. `stack_slot_color_map[d] == nlocals + d`. The map exists as
    /// a side channel so the decoder can stop assuming this invariant
    /// before commit 2.1 step C removes the pinning.
    pub stack_slot_color_map: Vec<u16>,
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
    /// pyre's graph identity for the cached jitcode slot.
    ///
    /// RPython indexes `CallControl.jitcodes` and `unfinished_graphs`
    /// directly by graph object. pyre still keys the public cache by
    /// `w_code` when available, but the cached object carries the raw
    /// CodeObject pointer so the queue can stay a bare graph list.
    pub code_ptr: *const pyre_interpreter::CodeObject,
    /// pyre-only wrapper identity for trace-side jitcode lookup.
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

    /// Is this `PyJitCode` a portal-bridged install (G.3a
    /// `canonical_bridge::install_portal_for`)?
    ///
    /// Discriminator:
    ///   * `jitcode.code` non-empty (rules out `PyJitCode::skeleton`,
    ///     which clones `Arc::new(RuntimeJitCode::default())` whose
    ///     `code` is empty).
    ///   * `metadata.pc_map` empty (rules out per-CodeObject installs
    ///     produced by `compile_jitcode_for_callee`, whose drain
    ///     populates `pc_map` to `code.instructions.len()`).
    ///
    /// Used by readers that have to branch on portal-mode semantics —
    /// portal entry has no per-Python-PC `pc_map` because the portal
    /// jitcode dispatches on `pycode.instructions[pc]` at runtime via
    /// its own dispatch arms.  See
    /// `canonical_bridge::install_portal_for` for the full reader
    /// audit (G.3a).
    ///
    /// G.3b commit lands this discriminator only.  No production
    /// reader currently calls it — readers will pick it up site-by-site
    /// in G.3c when concrete callers flip onto `install_portal_for`.
    pub fn is_portal_bridge(&self) -> bool {
        !self.jitcode.code.is_empty() && self.metadata.pc_map.is_empty()
    }

    /// True when the entry is "ready to dispatch" — either fully
    /// compiled (per-CodeObject install via the codewriter drain) or
    /// portal-bridged (`canonical_bridge::install_portal_for` clone of
    /// the portal canonical jitcode).  Both have non-empty
    /// `jitcode.code`; the discriminator between them is whether
    /// `metadata.pc_map` carries a per-PC mapping (per-CodeObject) or
    /// stays empty (portal-bridge dispatches on `pycode.instructions[pc]`
    /// at runtime via its own arms — no per-PC mapping needed).
    ///
    /// The mutually-exclusive `is_populated()` / `is_portal_bridge()`
    /// pair is preserved as the structural discriminator; this method
    /// is the union and exists so CallControl readers can ask "is this
    /// entry usable?" without needing to know which install kind it
    /// is.  `PyJitCode::skeleton()` (call.py:168 — fresh shell awaiting
    /// the drain) returns false for both predicates and therefore
    /// `is_dispatchable() == false`, preserving the skeleton-filter
    /// invariant `find_compiled_jitcode_arc` relies on
    /// (`PRE-EXISTING-ADAPTATION` doc on
    /// `pyre/pyre-jit/src/jit/call.rs:256-265`).
    ///
    /// G.4.4 prereq #2 step 1: introduced as the gate
    /// `find_compiled_jitcode_arc` switches to so that future
    /// `register_portal_bridge_in_callcontrol` activation
    /// (`pyre-jit/src/jit/codewriter.rs:5854`) can route portal-bridge
    /// entries through CallControl without the readers triggering
    /// compile loops on what the narrow `is_populated()` gate would
    /// otherwise classify as "not yet compiled".
    pub fn is_dispatchable(&self) -> bool {
        self.is_populated() || self.is_portal_bridge()
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
    pub fn skeleton(
        code_ptr: *const pyre_interpreter::CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) -> Self {
        Self {
            jitcode: std::sync::Arc::new(RuntimeJitCode::default()),
            metadata: PyJitCodeMetadata {
                pc_map: Vec::new(),
                depth_at_py_pc: Vec::new(),
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                stack_base: 0,
                stack_slot_color_map: Vec::new(),
            },
            code_ptr,
            w_code,
            has_abort: false,
            merge_point_pc,
        }
    }
}
