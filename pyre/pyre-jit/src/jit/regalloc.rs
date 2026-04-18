//! Post-pass register allocation for pyre's per-CodeObject SSARepr.
//!
//! Runs after `transform_graph_to_jitcode`'s dispatch loop has filled
//! the `SSARepr` with `Insn::Op` entries that reference registers by
//! pinned PyFrame-slot indices (locals at `0..nlocals`, stack values at
//! `nlocals + d`). The pass scans the SSARepr to build per-kind
//! `DependencyGraph`s of simultaneously-live registers, runs the
//! chordal-coloring routine in
//! `majit_codewriter::regalloc::DependencyGraph::find_node_coloring`
//! (line-by-line port of `rpython/tool/algo/color.py`), and produces a
//! rename map that compacts register indices into the smallest color set.
//!
//! RPython parity: `rpython/jit/codewriter/regalloc.py:8` invokes
//! `perform_register_allocation` once per kind ('int' / 'ref' / 'float')
//! over a `FunctionGraph`. Pyre's input is a CPython `CodeObject` rather
//! than a `FunctionGraph`, so the dependency-graph build operates on
//! the post-flatten `SSARepr` instead. The coloring algorithm is the
//! same — chordal greedy over a lexicographic-BFS order
//! (`color.py:31-85`) — and is shared with the algorithm exercised by
//! `majit-codewriter`'s flow-graph regalloc tests.
//!
//! ## Staging
//!
//! Step 1 (committed b3e85483d9) added [`RegisterMapping`] on
//! `PyJitCodeMetadata` with a `Pinned` variant that preserves the
//! legacy "register == PyFrame slot" layout. This module is the entry
//! point that future steps grow into:
//!
//! - **Step 2 (this commit)** — wire the pass; identity rename, no
//!   coloring. `RegisterMapping::Pinned` still produced. Behavior is
//!   unchanged but the dispatch finalisation now flows through the
//!   single seam where coloring will plug in.
//! - **Step 3** — populate the dependency graph from `SSARepr` def/use
//!   walks and run `find_node_coloring` to produce the rename map.
//! - **Step 4** — rewrite SSARepr `Insn::Op` register operands using
//!   the rename map; populate `RegisterMapping::PerPc`.
//! - **Step 5** — drop the `liveness_regs_to_u8_sorted` `has_abort`
//!   fallback now that `num_regs_*` is bounded.

use std::collections::HashMap;

use pyre_jit_trace::RegisterMapping;

use super::flatten::{Kind, SSARepr};

/// Output of [`allocate_registers`].
///
/// `rename` is a per-kind table mapping the SSARepr's pre-allocation
/// register index to the post-allocation index. The dispatch
/// finalisation in `codewriter.rs` applies the rename to the
/// `SSARepr` in place before handing it to `Assembler::assemble`.
///
/// `mapping` is the `RegisterMapping` value the dispatch finalisation
/// stores on `PyJitCodeMetadata` so blackhole resume
/// (`call_jit::blackhole_from_jit_frame`) knows which register to
/// load each PyFrame slot into.
pub(super) struct RegallocResult {
    /// `rename[(kind, pre_index)] = post_index`. Entries are present
    /// only when `pre_index != post_index`; missing entries are
    /// implicitly identity. The `Pinned` no-op pass returns an empty
    /// map.
    pub rename: HashMap<(Kind, u16), u16>,
    /// Register layout consumed by blackhole resume.
    pub mapping: RegisterMapping,
}

/// Allocate register indices for the `SSARepr` produced by pyre's
/// dispatch finalisation.
///
/// `nlocals` is the number of CPython fast locals (`code.varnames.len()`).
///
/// **Current implementation:** returns the identity rename plus the
/// pinned `RegisterMapping`. Step 3+ replace the body with the
/// dependency-graph build and chordal-coloring call.
pub(super) fn allocate_registers(_ssarepr: &SSARepr, nlocals: usize) -> RegallocResult {
    RegallocResult {
        rename: HashMap::new(),
        mapping: RegisterMapping::Pinned {
            nlocals: nlocals as u16,
        },
    }
}

/// Apply [`RegallocResult::rename`] to the `SSARepr` in place.
///
/// `Insn::Op` and `Insn::Live` operand registers (and `result`
/// registers on `Insn::Op`) get rewritten through the rename table.
/// Constants, labels, descrs, and indirect-call-target operands are
/// passed through unchanged.
///
/// **Current implementation:** the identity rename produced by
/// [`allocate_registers`] makes this a no-op. Kept as a separate
/// function so Step 4 can fill in the rewrite logic without altering
/// the call site in `codewriter.rs`.
pub(super) fn apply_rename(_ssarepr: &mut SSARepr, _rename: &HashMap<(Kind, u16), u16>) {
    // Identity rename — nothing to do.
}
