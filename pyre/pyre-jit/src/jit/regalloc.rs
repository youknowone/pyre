//! Register allocation: PyPy-orthodox port + pyre-only SSARepr scanner.
//!
//! ## PyPy-orthodox surface
//!
//! Mirrors the two-file split in PyPy:
//!
//!   * `rpython/jit/codewriter/regalloc.py:6-8
//!     perform_register_allocation(graph, kind)` — thin 2-arg wrapper.
//!     Pyre's analog is `perform_register_allocation(graph, kind)`
//!     below.
//!   * `rpython/tool/algo/regalloc.py:8-15
//!     perform_register_allocation(graph, consider_var, ListOfKind)`:
//!     ```python
//!     regalloc = RegAllocator(graph, consider_var, ListOfKind)
//!     regalloc.make_dependencies()    # interference graph
//!     regalloc.coalesce_variables()   # union-find on jump edges
//!     regalloc.find_node_coloring()   # chordal coloring
//!     ```
//!     Pyre's analog is the `RegAllocator` struct below plus its three
//!     private methods of the same name.
//!   * `rpython/jit/codewriter/flatten.py:88-100 enforce_input_args` —
//!     after coloring, `swapcolors` rotates inputarg colors into
//!     `0..n-1`. Pyre's analog is the `enforce_input_args` free
//!     function below (called by `GraphFlattener::enforce_input_args`
//!     at `flatten.rs` to mirror `flatten.py:68 flattener.enforce_input_args()`,
//!     and directly by the walker production path until the walker
//!     defers SSARepr emission to
//!     `codewriter.py:53 flatten_graph(graph, regallocs, cpu)`).
//!   * `rpython/jit/codewriter/codewriter.py:62-67` —
//!     `num_regs[kind] = max(coloring)+1` per kind, packed into the
//!     `JitCode`. Pyre's analog is `RegAllocator::find_num_colors`
//!     plus the `AllocationResult.num_regs` field.
//!
//! ## Pyre-only deviation (SSARepr-side companion allocator)
//!
//! Pyre's walker emits SSARepr inline rather than building a graph it
//! later flattens.  Until the walker defers SSARepr emission to the
//! canonical `codewriter.py:53 flatten_graph(graph, regallocs, cpu)`
//! driver, an SSARepr-side companion allocator is needed:
//!
//!   * `SSAReprRegAllocator` (declared further down) — same
//!     `make_dependencies` / `coalesce_variables` /
//!     `find_node_coloring` / `getcolor` / `swapcolors` /
//!     `find_num_colors` contract as upstream `RegAllocator`, but
//!     driven by a backward live-set walk over the populated SSARepr
//!     instead of `graph.iterblocks()`.
//!   * `perform_ssarepr_register_allocation` — builds an
//!     `SSAReprRegAllocator` and runs the three-stage pipeline,
//!     mirroring the upstream `perform_register_allocation` body.
//!   * `enforce_ssarepr_input_args` — variant of `enforce_input_args`
//!     keyed on u16 register indices instead of Variable identities.
//!
//! The chordal coloring algorithm itself is shared with
//! `majit-translate`'s flow-graph regalloc through
//! `majit_translate::regalloc::DependencyGraph::find_node_coloring`
//! (line-by-line port of `rpython/tool/algo/color.py:31-85`).

use std::collections::{HashMap, HashSet};

use majit_translate::regalloc::DependencyGraph;
use majit_translate::tool::algo::unionfind::UnionFind;

use super::flatten::{DescrOperand, Insn, Kind, Operand, Register, SSARepr, TLabel};
use super::flow::{ExitSwitch, ExitSwitchElement, FlowValue, FunctionGraph as FlowGraph, Variable};

#[derive(Debug, Clone)]
pub struct GraphAllocationResult {
    pub coloring: HashMap<super::flow::VariableId, u16>,
    pub num_colors: u16,
}

impl GraphAllocationResult {
    /// `rpython/tool/algo/regalloc.py:129-130 RegAllocator.getcolor` —
    /// return the post-coloring color for a Variable.  Panics when
    /// `v` is not colored (matches PyPy `_coloring[...]` KeyError).
    /// Pyre's `enforce_input_args` short-circuits via direct
    /// `coloring.get` to skip the inputargs-never-referenced case (the
    /// pyre walker can produce that shape; PyPy's `make_dependencies`
    /// always adds inputargs as nodes, so the case can't arise there).
    pub fn getcolor(&self, v: super::flow::VariableId) -> u16 {
        *self.coloring.get(&v).unwrap_or_else(|| {
            panic!("GraphAllocationResult::getcolor: missing color for {v:?}");
        })
    }

    /// `rpython/tool/algo/regalloc.py:138-143 RegAllocator.swapcolors`
    /// — swap every occurrence of `col1` and `col2` across the coloring
    /// dict.  Called by `enforce_input_args` (`flatten.py:88-100`) when
    /// an inputarg's coloring lands on a higher color than its
    /// positional `realcol`.
    pub fn swapcolors(&mut self, col1: u16, col2: u16) {
        for color in self.coloring.values_mut() {
            if *color == col1 {
                *color = col2;
            } else if *color == col2 {
                *color = col1;
            }
        }
    }
}

/// Field names follow `rpython/tool/algo/regalloc.py:21-24` —
/// `self.graph = graph` (py:22), `self.consider_var = consider_var`
/// (py:23 — pyre uses a `kind: Kind` filter because `Kind` is a closed
/// enum), `_depgraph` (`make_dependencies`, py:77), `_unionfind`
/// (`coalesce_variables`, py:80), `_coloring` (`find_node_coloring`,
/// py:115).  `self.ListOfKind` is omitted because pyre has exactly one
/// such type (`FlowListOfKind`).  The union-find is the real
/// `tool/algo/unionfind.py UnionFind` port (`()` info, matching
/// upstream `info_factory=None`).
struct RegAllocator<'a> {
    graph: &'a FlowGraph,
    kind: Kind,
    _depgraph: DependencyGraph<super::flow::VariableId>,
    _unionfind: UnionFind<super::flow::VariableId, ()>,
    _coloring: HashMap<super::flow::VariableId, u16>,
}

impl<'a> RegAllocator<'a> {
    fn new(graph: &'a FlowGraph, kind: Kind) -> Self {
        Self {
            graph,
            kind,
            _depgraph: DependencyGraph::new(),
            _unionfind: UnionFind::new(|_| ()),
            _coloring: HashMap::new(),
        }
    }

    fn make_dependencies(&mut self) {
        let kind = self.kind;
        for block in self.graph.iterblocks() {
            let block_borrow = block.borrow();
            let mut die_at: HashMap<super::flow::VariableId, usize> = HashMap::new();
            for arg in &block_borrow.inputargs {
                if let Some(v) = arg.as_variable() {
                    if v.kind == Some(kind) {
                        // ADAPTATION: project each Variable ID through
                        // `_unionfind.find_rep` so pre-merged pairs
                        // (from `perform_register_allocation_with_pairs`'s
                        // `extra_coalesce_pairs`) share a single
                        // live-set identity.  Walker scratch variables
                        // pinned to a local-i inputarg slot otherwise
                        // get distinct entries here and an interference
                        // edge gets recorded between them — preventing
                        // the later `try_coalesce` from merging them
                        // (regalloc.py:106 `has_edge` early return).
                        // When `_unionfind` has no pre-merges, find_rep
                        // returns the input ID unchanged so this matches
                        // upstream `regalloc.py:26-77` exactly.
                        let rep = self._unionfind.find_rep(v.id);
                        die_at.insert(rep, 0);
                    }
                }
            }
            for (i, op) in block_borrow.operations.iter().enumerate() {
                for arg in &op.args {
                    for v in arg.variables() {
                        if v.kind == Some(kind) {
                            let rep = self._unionfind.find_rep(v.id);
                            die_at.insert(rep, i);
                        }
                    }
                }
                if let Some(v) = op.result.as_ref().and_then(FlowValue::as_variable) {
                    if v.kind == Some(kind) {
                        let rep = self._unionfind.find_rep(v.id);
                        die_at.insert(rep, i + 1);
                    }
                }
            }
            match &block_borrow.exitswitch {
                Some(ExitSwitch::Value(value)) => {
                    if let Some(v) = value.as_variable() {
                        let rep = self._unionfind.find_rep(v.id);
                        die_at.remove(&rep);
                    }
                }
                Some(ExitSwitch::Tuple(values)) => {
                    for value in values {
                        if let ExitSwitchElement::Value(value) = value {
                            if let Some(v) = value.as_variable() {
                                let rep = self._unionfind.find_rep(v.id);
                                die_at.remove(&rep);
                            }
                        }
                    }
                }
                None => {}
            }
            for link in &block_borrow.exits {
                for arg in &link.borrow().args {
                    if let Some(v) = arg.as_ref().and_then(FlowValue::as_variable) {
                        let rep = self._unionfind.find_rep(v.id);
                        die_at.remove(&rep);
                    }
                }
            }
            let mut die_list: Vec<(usize, super::flow::VariableId)> =
                die_at.into_iter().map(|(var, time)| (time, var)).collect();
            die_list.sort_by_key(|(time, _)| *time);
            die_list.push((usize::MAX, super::flow::VariableId(u32::MAX)));

            let livevar_reps: Vec<super::flow::VariableId> = block_borrow
                .inputargs
                .iter()
                .filter_map(FlowValue::as_variable)
                .filter(|v| v.kind == Some(kind))
                .map(|v| self._unionfind.find_rep(v.id))
                .collect();
            for (i, &v) in livevar_reps.iter().enumerate() {
                self._depgraph.add_node(v);
                for j in 0..i {
                    // `add_edge` is a no-op for self-edges, so distinct
                    // inputargs that pre-merge into the same rep don't
                    // trigger a phantom interference.
                    self._depgraph.add_edge(livevar_reps[j], v);
                }
            }
            // upstream: `livevars = set(livevars)` — shadow the list
            // with the set rather than renaming to `alive`.
            let mut livevars: HashSet<super::flow::VariableId> = livevar_reps.into_iter().collect();
            let mut die_index = 0;
            for (i, op) in block_borrow.operations.iter().enumerate() {
                while die_list[die_index].0 == i {
                    livevars.remove(&die_list[die_index].1);
                    die_index += 1;
                }
                if let Some(result) = op.result.as_ref().and_then(FlowValue::as_variable) {
                    if result.kind == Some(kind) {
                        let rep = self._unionfind.find_rep(result.id);
                        self._depgraph.add_node(rep);
                        // upstream (`regalloc.py:73`): add an edge from
                        // every live var to `result`.  `result` is added
                        // to `livevars` only *after* the loop, so no
                        // self-edge guard is needed.
                        for &v in &livevars {
                            self._depgraph.add_edge(v, rep);
                        }
                        livevars.insert(rep);
                    }
                }
            }
        }
    }

    fn coalesce_variables(&mut self) {
        let kind = self.kind;
        let mut pendingblocks = self.graph.iterblocks();
        while let Some(block) = pendingblocks.pop() {
            // Match `rpython/tool/algo/regalloc.py:82-86`: walk from the
            // end of the graph first because resume/blackhole execution
            // typically restarts in the middle rather than at the entry.
            let block_borrow = block.borrow();
            for link in &block_borrow.exits {
                let link_borrow = link.borrow();
                if let Some(v) = link_borrow.last_exception {
                    if v.kind == Some(kind) {
                        self._depgraph.add_node(v.id);
                    }
                }
                if let Some(v) = link_borrow.last_exc_value {
                    if v.kind == Some(kind) {
                        self._depgraph.add_node(v.id);
                    }
                }
                let Some(target) = link_borrow.target.clone() else {
                    continue;
                };
                let target_borrow = target.borrow();
                for (arg, target_input) in
                    link_borrow.args.iter().zip(target_borrow.inputargs.iter())
                {
                    let Some(src) = arg.as_ref().and_then(FlowValue::as_variable) else {
                        continue;
                    };
                    let Some(dst) = target_input.as_variable() else {
                        continue;
                    };
                    self.try_coalesce(src, dst);
                }
            }
        }
    }

    /// `rpython/tool/algo/regalloc.py:98-112 _try_coalesce` — kind
    /// check + identity short-circuit + interference check + union.
    /// Both endpoints are assumed to be in `_depgraph` already because
    /// `make_dependencies` registered every op result, inputarg, and
    /// link-arg-derived Variable; upstream does no `add_node` here.
    fn try_coalesce(&mut self, v: Variable, w: Variable) {
        if v.kind != Some(self.kind) || w.kind != Some(self.kind) {
            return;
        }
        if v.id == w.id {
            return;
        }
        let v0 = self._unionfind.find_rep(v.id);
        let w0 = self._unionfind.find_rep(w.id);
        if v0 == w0 {
            return;
        }
        if self._depgraph.has_edge(&v0, &w0) {
            return;
        }
        let (_, rep) = self._unionfind.union(v0, w0);
        debug_assert_eq!(self._unionfind.find_rep(v0), rep);
        debug_assert_eq!(self._unionfind.find_rep(w0), rep);
        if rep == v0 {
            self._depgraph.coalesce(w0, v0);
        } else {
            debug_assert_eq!(rep, w0);
            self._depgraph.coalesce(v0, w0);
        }
    }

    /// Variable-id-keyed pin coalesce for `walker_pin_pairs` —
    /// pyre-only ADAPTATION called from
    /// `perform_register_allocation_with_pairs` to honour the walker's
    /// scratch↔inputarg slot pinning.  Walker scratch variables that
    /// don't appear as op operands/results in the canonical graph
    /// aren't registered via `make_dependencies`, so they're absent
    /// from `_depgraph.all_nodes`.  `DependencyGraph::coalesce` only
    /// modifies `neighbours` (not `all_nodes`), so
    /// `find_node_coloring`'s `getnodes` filter would skip a coalesced
    /// surviving node that was never explicitly added — yielding
    /// `None` for `getcolor` and dropping the chain's inputarg from
    /// the final coloring map.  This defensive `add_node` can be
    /// dropped once coalesced survivors are always registered via
    /// `make_dependencies`.
    fn try_coalesce_pin_ids(
        &mut self,
        v_id: super::flow::VariableId,
        w_id: super::flow::VariableId,
    ) {
        if v_id == w_id {
            return;
        }
        self._depgraph.add_node(v_id);
        self._depgraph.add_node(w_id);
        let v0 = self._unionfind.find_rep(v_id);
        let w0 = self._unionfind.find_rep(w_id);
        if v0 == w0 {
            return;
        }
        if self._depgraph.has_edge(&v0, &w0) {
            return;
        }
        let (_, rep) = self._unionfind.union(v0, w0);
        debug_assert_eq!(self._unionfind.find_rep(v0), rep);
        debug_assert_eq!(self._unionfind.find_rep(w0), rep);
        if rep == v0 {
            self._depgraph.coalesce(w0, v0);
        } else {
            debug_assert_eq!(rep, w0);
            self._depgraph.coalesce(v0, w0);
        }
    }

    /// Record an interference edge between two frame-local slot
    /// representatives so the chordal coloring assigns them DISTINCT
    /// colors, even when their SSA register live ranges are disjoint.
    ///
    /// Complement of [`try_coalesce_pin_ids`]: that merges Variables onto
    /// one color, this forces them apart.  Used by the splice regalloc to
    /// reproduce the walker's bijective slot→register assignment so the
    /// per-slot resume reverse map is injective.  Both endpoints are
    /// projected through `_unionfind.find_rep`, so a slot whose Variables
    /// were already coalesced into one rep (by the same-slot coalesce
    /// pairs) contributes a single node; `add_edge` is a no-op for
    /// self-edges, so two reps that happen to coincide do not trigger a
    /// phantom interference.  `add_node` registers each rep in
    /// `_depgraph.all_nodes` so `find_node_coloring`'s `getnodes` filter
    /// keeps it (matching `try_coalesce_pin_ids`).
    fn add_interference_pin_ids(
        &mut self,
        v_id: super::flow::VariableId,
        w_id: super::flow::VariableId,
    ) {
        let v0 = self._unionfind.find_rep(v_id);
        let w0 = self._unionfind.find_rep(w_id);
        self._depgraph.add_node(v0);
        self._depgraph.add_node(w0);
        self._depgraph.add_edge(v0, w0);
    }

    fn find_node_coloring(&mut self) {
        self._coloring = self
            ._depgraph
            .find_node_coloring()
            .into_iter()
            .map(|(value, color)| (value, color as u16))
            .collect();
    }

    fn getcolor(&mut self, v: Variable) -> Option<u16> {
        let rep = self._unionfind.find_rep(v.id);
        self._coloring.get(&rep).copied()
    }

    fn find_num_colors(&self) -> u16 {
        self._coloring.values().copied().max().map_or(0, |m| m + 1)
    }
}

/// `rpython/jit/codewriter/regalloc.py:6 perform_register_allocation(graph, kind)`
/// — thin wrapper over `rpython/tool/algo/regalloc.py:8-15
/// perform_register_allocation(graph, consider_var, ListOfKind=())`.
///
/// Pyre bakes `consider_var` into the single `kind` filter because
/// `Kind` is a closed enum (Int/Ref/Float) whereas upstream's
/// `consider_var` is an open predicate over lltype concreteness.
/// `ListOfKind` is not a parameter because pyre has exactly one such
/// class (`FlowListOfKind`).
///
/// Invoked from production via `perform_register_allocation_all_kinds`
/// at `codewriter.rs:transform_graph_to_jitcode`, where its result
/// feeds the canonical `flatten_graph` splice regalloc.  The pyre-only
/// SSARepr-side `perform_ssarepr_register_allocation` further down in
/// this module still runs alongside this CFG allocator because pyre
/// has two coalesce sources that cover non-overlapping work:
///
///   1. This CFG allocator runs `coalesce_variables` over
///      `link.args ↔ target.inputargs` pairs.  Every pair is trivially
///      `(slot, slot)` because `FrameState::getoutputargs` builds
///      `link.args` by positional walk over
///      `targetstate.mergeable()`, so `try_coalesce` is a no-op at the
///      CFG level today.  When the walker eventually defers SSARepr
///      emission to the canonical `flatten_graph` driver, the CFG
///      allocator will become the load-bearing coalesce source.
///   2. The SSARepr scanner handles **intra-block** `*_copy`
///      sequences (emitted by `emit_ref_copy!`/`emit_int_copy!` inline
///      for STORE_FAST-LOAD_FAST fusions) that have no Link-level
///      representation and therefore cannot be coalesced at the CFG
///      level.
pub(super) fn perform_register_allocation(graph: &FlowGraph, kind: Kind) -> GraphAllocationResult {
    perform_register_allocation_with_pairs(graph, kind, &[])
}

/// ADAPTATION variant: runs the same `RegAllocator` pipeline as
/// [`perform_register_allocation`] but applies `extra_coalesce_pairs`
/// between the upstream-orthodox `coalesce_variables` and
/// `find_node_coloring` steps.
///
/// Each pair `(scratch_id, inputarg_id)` requests that
/// `scratch_id`'s post-coloring color equal `inputarg_id`'s color.
/// The mechanism is `try_coalesce`: the two variables are unioned in
/// the regalloc union-find (if no interference edge blocks it), so
/// the subsequent chordal coloring assigns them the same color.
/// `enforce_input_args` then rotates the unified cluster onto the
/// inputarg's `0..nlocals-1` slot.
///
/// Upstream RPython has no analog because PyPy's flowgraph never
/// produces "scratch local-i" Variables disjoint from the
/// `startblock.inputargs[i]` variable — the same Variable flows
/// through every read/write of local i.  Pyre's walker
/// (`codewriter.rs::transform_graph_to_jitcode`) emits fresh
/// scratch Variables for each `LOAD_FAST` / `STORE_FAST` and pins
/// them to slot=i via `walker_slot_for_variable`; this helper lets
/// the canonical graph regalloc honor that same pin so the bytes it
/// emits match the walker's inline emission slot-for-slot.
pub fn perform_register_allocation_with_pairs(
    graph: &FlowGraph,
    kind: Kind,
    extra_coalesce_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
) -> GraphAllocationResult {
    perform_register_allocation_with_pairs_and_interference(graph, kind, extra_coalesce_pairs, &[])
}

/// Splice adaptation: like [`perform_register_allocation_with_pairs`] but
/// also records an interference edge between the union-find reps named in
/// each `interference_pairs` entry, forcing those reps onto DISTINCT
/// colors.
///
/// `interference_pairs` is the complement of `extra_coalesce_pairs`:
/// where the coalesce pairs merge same-slot Variables onto one color, the
/// interference pairs separate distinct frame-local slots whose SSA
/// register live ranges happen to be disjoint (each `LOAD_FAST` re-reads
/// the local, so a local's SSA value dies between reads).  Without this
/// the chordal coloring is free to give two frame-live locals one color,
/// and the splice resume reverse map (`pyre_color_for_semantic_local` →
/// `semantic_ref_slot_for_reg_color`) collapses them onto one slot.
/// The edges are added after `make_dependencies` (the base liveness graph
/// must exist) and before `coalesce_variables` (so a cross-slot coalesce
/// is blocked by the `try_coalesce` `has_edge` guard) and
/// `find_node_coloring`.  Splice-only — production callers pass `&[]`,
/// leaving the coloring byte-identical.
pub fn perform_register_allocation_with_pairs_and_interference(
    graph: &FlowGraph,
    kind: Kind,
    extra_coalesce_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
    interference_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
) -> GraphAllocationResult {
    // `rpython/tool/algo/regalloc.py:11-15`:
    //     regalloc = RegAllocator(graph, consider_var, ListOfKind)
    //     regalloc.make_dependencies()
    //     regalloc.coalesce_variables()
    //     regalloc.find_node_coloring()
    let mut allocator = RegAllocator::new(graph, kind);
    // ADAPTATION: pre-merge external pairs into `_unionfind` BEFORE
    // `make_dependencies` so the live-set tracking (which projects
    // every Variable ID through `_unionfind.find_rep`) treats each
    // pinned scratch↔inputarg pair as a single node.  Without the
    // pre-merge, walker scratch and the corresponding canonical
    // inputarg get separate live entries and `make_dependencies`
    // records an interference edge between them; the post-coalesce
    // `try_coalesce_ids` then early-returns at the `has_edge` check
    // (regalloc.py:106) and the pin has no effect on coloring.
    // `find_rep` auto-creates a singleton partition for IDs not yet
    // tracked, so unknown scratch IDs are handled safely.
    for &(v_id, w_id) in extra_coalesce_pairs {
        let v0 = allocator._unionfind.find_rep(v_id);
        let w0 = allocator._unionfind.find_rep(w_id);
        if v0 != w0 {
            allocator._unionfind.union(v0, w0);
        }
    }
    allocator.make_dependencies();
    // Record interference between the named slot reps so the
    // chordal coloring keeps distinct frame-local slots on distinct
    // colors.  Added after `make_dependencies` so the base graph exists,
    // before `coalesce_variables`/`find_node_coloring` so both honour it.
    for &(a_id, b_id) in interference_pairs {
        allocator.add_interference_pin_ids(a_id, b_id);
    }
    allocator.coalesce_variables();
    // External pins — re-apply via `try_coalesce_pin_ids` after
    // `make_dependencies` so the surviving rep is explicitly added to
    // `_depgraph.all_nodes` even when neither endpoint appeared as an
    // op result/arg in the canonical graph.  With the union-find
    // pre-merge above these calls are no-ops on the union-find side
    // (`find_rep` already returns a common rep), but `add_node` still
    // matters for `find_node_coloring`'s `getnodes` filter.
    for &(v_id, w_id) in extra_coalesce_pairs {
        allocator.try_coalesce_pin_ids(v_id, w_id);
    }
    allocator.find_node_coloring();

    let mut coloring = HashMap::new();
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        for variable in block_borrow.getvariables() {
            if variable.kind == Some(kind) {
                if let Some(color) = allocator.getcolor(variable) {
                    coloring.insert(variable.id, color);
                }
            }
        }
        for link in &block_borrow.exits {
            let link_borrow = link.borrow();
            if let Some(v) = link_borrow.last_exception {
                if v.kind == Some(kind) {
                    if let Some(color) = allocator.getcolor(v) {
                        coloring.insert(v.id, color);
                    }
                }
            }
            if let Some(v) = link_borrow.last_exc_value {
                if v.kind == Some(kind) {
                    if let Some(color) = allocator.getcolor(v) {
                        coloring.insert(v.id, color);
                    }
                }
            }
            for arg in &link_borrow.args {
                if let Some(v) = arg.as_ref().and_then(FlowValue::as_variable) {
                    if v.kind == Some(kind) {
                        if let Some(color) = allocator.getcolor(v) {
                            coloring.insert(v.id, color);
                        }
                    }
                }
            }
        }
    }

    GraphAllocationResult {
        coloring,
        num_colors: allocator.find_num_colors(),
    }
}

/// Run `perform_register_allocation` once per `Kind` and collect
/// the per-kind `GraphAllocationResult`s, mirroring
/// `rpython/jit/codewriter/codewriter.py:44-46`:
///
/// ```python
/// regallocs = {}
/// for kind in KINDS:
///     regallocs[kind] = perform_register_allocation(graph, kind)
/// ```
///
/// The resulting `[GraphAllocationResult; 3]` is indexed by
/// `Kind::index()` (`Int=0, Ref=1, Float=2`).  Upstream uses a Python
/// dict; pyre uses `[T; 3]`: the
/// RPython `KINDS` list has 3 statically-known entries so the dict
/// degenerates to a position-indexed array in any RPython-orthodox
/// port.  This is the input shape that the canonical
/// `flatten_graph(graph, regallocs, ...)` driver consumes.
pub fn perform_register_allocation_all_kinds(graph: &FlowGraph) -> [GraphAllocationResult; 3] {
    perform_register_allocation_all_kinds_with_pairs(graph, &[])
}

/// ADAPTATION variant: invokes the per-kind
/// `perform_register_allocation_with_pairs` for `Kind::Ref` with
/// `ref_coalesce_pairs`.  Int and Float kinds use the empty-pair
/// path because walker's `walker_slot_for_variable` only tracks Ref
/// slots (every `FrameState.mergeable()` position is Ref-kind:
/// locals, stack, last_exc pair).
pub fn perform_register_allocation_all_kinds_with_pairs(
    graph: &FlowGraph,
    ref_coalesce_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
) -> [GraphAllocationResult; 3] {
    [
        perform_register_allocation(graph, Kind::Int),
        perform_register_allocation_with_pairs(graph, Kind::Ref, ref_coalesce_pairs),
        perform_register_allocation(graph, Kind::Float),
    ]
}

/// Splice adaptation: like [`perform_register_allocation_all_kinds_with_pairs`]
/// but threads `ref_interference_pairs` into the `Kind::Ref` allocation so
/// distinct frame-local slots receive distinct Ref colors.  Int and Float
/// use the empty path (walker tracks only Ref slots).  Splice-only.
pub fn perform_register_allocation_all_kinds_with_pairs_and_interference(
    graph: &FlowGraph,
    ref_coalesce_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
    ref_interference_pairs: &[(super::flow::VariableId, super::flow::VariableId)],
) -> [GraphAllocationResult; 3] {
    [
        perform_register_allocation(graph, Kind::Int),
        perform_register_allocation_with_pairs_and_interference(
            graph,
            Kind::Ref,
            ref_coalesce_pairs,
            ref_interference_pairs,
        ),
        perform_register_allocation(graph, Kind::Float),
    ]
}

/// Mirrors `rpython/jit/codewriter/flatten.py:88-100 enforce_input_args`
/// at the graph level (sibling to the SSA-side private
/// `enforce_ssarepr_input_args` further down that handles per-
/// `SSAReprRegAllocator` swapcolors).
///
/// Walks the startblock's inputargs in source order; for each inputarg
/// of kind `K` whose current color in `regallocs[K]` does not equal
/// the next "real" color for that kind (`0, 1, 2, ...` per appearance),
/// invokes `swapcolors(realcol, curcol)` over the entire
/// `coloring` map (`rpython/tool/algo/regalloc.py:138-143`).
///
/// Upstream `flatten_graph` runs this immediately after
/// `regallocs[kind] = perform_register_allocation(graph, kind)` and
/// before `generate_ssa_form` walks links, so every downstream
/// observer sees the post-swap coloring.  Pyre's canonical
/// `flatten_graph` entry (`flatten.rs::flatten_graph`) and the
/// walker-side post-walk path both call this free function rather
/// than a `GraphFlattener` method: pyre's `get_register` closure
/// captures `&regallocs` immutably, so the `&mut regallocs`
/// swap must run BEFORE the closure is constructed.
pub fn enforce_input_args(graph: &FlowGraph, regallocs: &mut [GraphAllocationResult; 3]) {
    let inputargs = graph.startblock.borrow().inputargs.clone();
    // RPython `numkinds = {}` (flatten.py:91); pyre stores the per-kind
    // counter in a `[u16; 3]` array indexed by `Kind::index()`.
    let mut numkinds: [u16; 3] = [0; 3];
    for arg in &inputargs {
        let Some(v) = arg.as_variable() else { continue };
        let Some(kind) = v.kind else { continue };
        let kind_idx = kind.index();
        let realcol = numkinds[kind_idx];
        numkinds[kind_idx] = realcol + 1;
        let alloc = &mut regallocs[kind_idx];
        // Inputarg never appeared in any instruction — coloring
        // skipped it. Swap is unnecessary because no register refers
        // to its color (mirrors the SSARepr-side `enforce_input_args`
        // shortcut).
        let Some(&curcol) = alloc.coloring.get(&v.id) else {
            continue;
        };
        if curcol == realcol {
            continue;
        }
        assert!(
            curcol > realcol,
            "enforce_input_args: inputarg color {} must be >= realcol {} \
             (regalloc.py invariant)",
            curcol,
            realcol,
        );
        // `flatten.py:100 self.regallocs[kind].swapcolors(realcol, curcol)`.
        alloc.swapcolors(realcol, curcol);
    }
}

/// External-input registers preserved across coloring.
///
/// RPython parity: `regalloc.py:54-60` adds pairwise interference
/// edges between every variable in `block.inputargs`, and
/// `flatten.py:88-100` enforces that those same inputargs land on
/// colors `0..n-1` via `swapcolors`. Pyre's analog is the set of
/// registers that arrive pre-populated:
///
/// - The `nlocals` low Ref registers (locals 0..nlocals) are the
///   trace-side analog of `block.inputargs`. Both `trace_opcode.rs`
///   (the LiveVars expansion) and the bytecode walker decode
///   `register_idx < nlocals` as "this register holds Python local
///   `register_idx`'s value", so the post-regalloc colors of these
///   registers must equal their pre-coloring indices. This is
///   guaranteed by `enforce_input_args` (not by interference
///   heuristics), in line with `flatten.py:88-100`.
/// - Portal red args (`frame_reg`, `ec_reg`) are pre-populated by
///   `BlackholeInterpreter::fill_portal_registers`
///   (blackhole.rs:1133-1140) at compile-time-fixed register slots
///   produced by `RegisterLayout::compute`; they get colors
///   `nlocals` and `nlocals+1` after `enforce_input_args` runs.
#[derive(Clone, Copy)]
pub(super) struct ExternalInputs {
    pub portal_frame_reg: u16,
    pub portal_ec_reg: u16,
    pub portal_inputs: bool,
}

/// Result of `allocate_registers`.
///
/// `rename` carries the per-kind pre→post coloring map applied by
/// `apply_rename`.  `[Vec<u16>; 3]` indexed by `Kind::index()`: each
/// inner `Vec<u16>` is indexed by
/// the pre-coloring slot and yields the post-coloring color.  Entries
/// past the vector's length implicitly map to identity (no rename
/// occurred for that slot).  Mirrors RPython's `(kind, pre) → post`
/// dict at `codewriter.py:62-67` projected onto pyre's u16 slot space.
///
/// `num_regs` carries the per-kind `max(color)+1` value the assembler
/// stores in `JitCode.num_regs_*` (codewriter.py:62-67).
pub(super) struct AllocationResult {
    pub rename: [Vec<u16>; 3],
    /// Per-kind `max(coloring)+1` indexed by `Kind::index()`.
    /// Mirrors RPython
    /// `codewriter.py:62-67 num_regs[kind]` — pyre's `KINDS` array
    /// of 3 statically-known kinds collapses the dict to `[u16; 3]`.
    pub num_regs: [u16; 3],
}

/// Lookup helper for the kind-indexed rename vec: returns the post
/// coloring for `pre`, falling back to identity when no rename was
/// recorded.
#[inline]
pub(super) fn rename_lookup(rename: &[Vec<u16>; 3], kind: Kind, pre: u16) -> u16 {
    rename[kind.index()]
        .get(pre as usize)
        .copied()
        .filter(|&p| p != u16::MAX)
        .unwrap_or(pre)
}

/// Run register allocation on `ssarepr` and produce the rename map
/// plus per-kind `num_regs`.
///
/// `nlocals` is the number of CPython fast locals (`code.varnames.len()`).
///
/// `cfg_coalesce_pairs` is the output of
/// `codewriter::collect_cfg_coalesce_pairs` — `(source_slot,
/// target_slot)` pairs from CFG link boundaries (`regalloc.py:79-96`
/// `link.args[i] ↔ link.target.inputargs[i]`), all of Ref kind
/// because every `FrameState.mergeable()` position in pyre holds a
/// Ref-kind Variable (locals, stack, last_exc pair).
///
/// RPython parity: `codewriter.py:45-47, 62-67`.
pub(super) fn allocate_registers(
    ssarepr: &SSARepr,
    nlocals: usize,
    inputs: ExternalInputs,
    cfg_coalesce_pairs: &[(u16, u16)],
) -> AllocationResult {
    // codewriter.py:45-47 `for kind in KINDS:
    //   regallocs[kind] = perform_register_allocation(graph, kind)`.
    // `[SSAReprRegAllocator; 3]` indexed by `Kind::index()`.
    let mut allocators: [SSAReprRegAllocator; 3] =
        std::array::from_fn(|_| SSAReprRegAllocator::new());
    for &kind in &Kind::ALL {
        let mut external: Vec<u16> = Vec::new();
        if kind == Kind::Ref {
            for i in 0..nlocals as u16 {
                external.push(i);
            }
            // Stack slots are not pinned to fixed colors;
            // `stack_slot_color_map` (PyJitCodeMetadata) records the
            // post-rename color so decoders / blackhole resume can
            // translate slot → color without assuming identity.
            if inputs.portal_inputs {
                if inputs.portal_frame_reg != u16::MAX {
                    external.push(inputs.portal_frame_reg);
                }
                if inputs.portal_ec_reg != u16::MAX {
                    external.push(inputs.portal_ec_reg);
                }
            }
        }
        // CFG pairs are projected from `FrameState.mergeable()` Variables,
        // which are uniformly Ref-kind in pyre.  Int / Float allocators
        // see an empty slice.
        let cfg_pairs_for_kind: &[(u16, u16)] = if kind == Kind::Ref {
            cfg_coalesce_pairs
        } else {
            &[]
        };
        allocators[kind.index()] =
            perform_ssarepr_register_allocation(ssarepr, kind, &external, cfg_pairs_for_kind);
    }

    // flatten.py:88-100 `enforce_input_args` — rotate inputarg colors
    // into 0..n-1 via swapcolors so the trace-side `idx < nlocals`
    // decode is guaranteed by code rather than by an interference
    // heuristic.
    enforce_ssarepr_input_args(&mut allocators, nlocals, &inputs);

    // codewriter.py:62-67 `num_regs = {kind: max(coloring)+1 if coloring else 0}`.
    // Per-kind rename map: `[Vec<u16>; 3]` indexed by `Kind::index()`,
    // inner Vec keyed by pre-coloring slot.  Identity entries (no
    // rename) are left as the slot's own index; `rename_lookup` returns
    // identity for indices past the vector length.
    let mut rename: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut num_regs: [u16; 3] = [0; 3];
    for &kind in &Kind::ALL {
        let alloc = &allocators[kind.index()];
        let max_pre = alloc.coloring.keys().copied().max().unwrap_or(0);
        let kind_rename = &mut rename[kind.index()];
        // Initialize to identity: index i → value i.
        kind_rename.reserve_exact((max_pre as usize) + 1);
        for i in 0..=max_pre {
            kind_rename.push(i);
        }
        for (&pre, &post) in alloc.coloring.iter() {
            if pre != post {
                kind_rename[pre as usize] = post;
            }
        }
        num_regs[kind.index()] = alloc.find_num_colors();
    }
    AllocationResult { rename, num_regs }
}

/// SSARepr-side companion to upstream
/// `rpython/jit/codewriter/flatten.py:88-100
/// GraphFlattener.enforce_input_args`.
///
/// Upstream:
/// ```python
/// def enforce_input_args(self):
///     inputargs = self.graph.startblock.inputargs
///     numkinds = {}
///     for v in inputargs:
///         kind = getkind(v.concretetype)
///         if kind == 'void':
///             continue
///         curcol = self.regallocs[kind].getcolor(v)
///         realcol = numkinds.get(kind, 0)
///         numkinds[kind] = realcol + 1
///         if curcol != realcol:
///             assert curcol > realcol
///             self.regallocs[kind].swapcolors(realcol, curcol)
/// ```
///
/// Pyre's `inputargs` are all of kind `Ref`: locals `0..nlocals` (the
/// trace-side Python-local mirror) followed by the portal red args
/// (frame, ec). Int and Float kinds have no inputargs — see
/// `ExternalInputs` docstring.
///
/// TODO: the SSARepr-side variant operates on
/// `SSAReprRegAllocator` keyed by u16 register indices instead of
/// Variable identities.  The PyPy-orthodox graph-side sibling lives
/// at `enforce_input_args` (free function) above and retires this
/// variant when the walker defers SSARepr emission to
/// `codewriter.py:53 flatten_graph(graph, regallocs, cpu)`.
fn enforce_ssarepr_input_args(
    allocators: &mut [SSAReprRegAllocator; 3],
    nlocals: usize,
    inputs: &ExternalInputs,
) {
    let alloc = &mut allocators[Kind::Ref.index()];
    let mut input_indices: Vec<u16> = (0..nlocals as u16).collect();
    // Stack slots are not rotated into fixed colors; the decoder
    // consults `stack_slot_color_map` to recover the post-rename
    // color.
    if inputs.portal_inputs {
        if inputs.portal_frame_reg != u16::MAX {
            input_indices.push(inputs.portal_frame_reg);
        }
        if inputs.portal_ec_reg != u16::MAX {
            input_indices.push(inputs.portal_ec_reg);
        }
    }
    for (realcol, &v) in input_indices.iter().enumerate() {
        let realcol = realcol as u16;
        let curcol = match alloc.getcolor(v) {
            Some(c) => c,
            // Inputarg never appeared in any instruction — coloring
            // skipped it. swap is unnecessary because no register
            // refers to its color.
            None => continue,
        };
        if curcol != realcol {
            assert!(
                curcol > realcol,
                "enforce_ssarepr_input_args: inputarg color {} must be >= realcol {} (regalloc.py invariant)",
                curcol,
                realcol
            );
            alloc.swapcolors(realcol, curcol);
        }
    }
}

/// SSARepr-side companion to PyPy's
/// `rpython/jit/codewriter/regalloc.py:6 perform_register_allocation(graph, kind)`
/// + `rpython/tool/algo/regalloc.py:8-15`.  Builds an
/// `SSAReprRegAllocator` and runs the three-stage pipeline.
///
/// TODO: pyre's walker emits SSARepr inline, so
/// the consumer is `SSAReprRegAllocator`, not the orthodox
/// `RegAllocator`.  The graph-side sibling (`perform_register_allocation`)
/// is the PyPy-orthodox entry; this variant retires when the walker
/// defers SSARepr emission to
/// `codewriter.py:53 flatten_graph(graph, regallocs, cpu)`.
///
/// Dual coalesce source, ordered to give CFG link priority over the
/// pyre-only SSARepr scanner when an interference edge forces a
/// choice between two `try_coalesce` candidates:
///   1. CFG-level `(source_slot, target_slot)` pairs from
///      `link.args[i] ↔ link.target.inputargs[i]` per
///      `regalloc.py:79-96 coalesce_variables`.  The caller derives
///      these from `graph.iterblocks()` and the walker's slot
///      assignment for each Variable; passing them in keeps the
///      upstream iteration shape even when the walker's chosen
///      slots make most pairs trivially equal (`try_coalesce(v, v)`
///      returns immediately).  Runs FIRST so a link-driven union
///      always wins over an SSA-copy union when both would target
///      the same interference-graph cluster — matching upstream
///      where CFG `coalesce_variables` is the only coalesce source.
///   2. SSARepr `*_copy` scanner — pyre's walker emits intra-block
///      `int_copy` / `ref_copy` / `float_copy` ops for stack
///      shuffling / STORE_FAST sequences directly into the SSARepr;
///      RPython has no analog because `flatten.py:306-334`
///      `insert_renamings` places its copies post-coalesce at
///      flatten time.  The scanner unions each `*_copy`'s src and
///      dst so the chordal coloring reuses one color.  Runs after
///      the CFG pass so the pyre-only source defers to upstream's
///      link-driven priority on conflict.
fn perform_ssarepr_register_allocation(
    ssarepr: &SSARepr,
    kind: Kind,
    external_inputs: &[u16],
    cfg_coalesce_pairs: &[(u16, u16)],
) -> SSAReprRegAllocator {
    let mut alloc = SSAReprRegAllocator::new();
    alloc.make_dependencies(ssarepr, kind, external_inputs);
    // `regalloc.py:79-96` CFG-level coalesce — every Link's
    // `link.args[i] ↔ link.target.inputargs[i]` pair, projected to
    // the walker's u16 slots.  Runs BEFORE the SSARepr `*_copy`
    // scanner so the upstream link-driven coalesce wins on
    // interference-graph conflict.
    for &(src, dst) in cfg_coalesce_pairs {
        alloc.try_coalesce(src, dst);
    }
    alloc.coalesce_variables(ssarepr, kind);
    alloc.find_node_coloring();
    alloc
}

/// Pyre-only SSARepr-side allocator (TODO: bring to parity).
///
/// PyPy has exactly one allocator at `rpython/tool/algo/regalloc.py:18`
/// (`RegAllocator`), driven by a `FunctionGraph`.  Pyre's walker emits
/// SSARepr inline rather than building a graph it later flattens, so
/// this companion allocator works directly over the populated
/// `SSARepr` via a backward live-set walk.  Method bodies still follow
/// `RegAllocator`'s contract line-by-line (`make_dependencies`,
/// `coalesce_variables`, `find_node_coloring`, `find_num_colors`,
/// `getcolor`, `swapcolors`) so that — once the walker defers SSARepr
/// emission to the canonical
/// `codewriter.py:53 flatten_graph(graph, regallocs, cpu)` driver —
/// this whole struct retires and the sibling `RegAllocator` at the
/// top of this module is the sole allocator, matching upstream
/// exactly.
///
/// The `SSARepr` prefix marks this as the deviation pending retirement.
///
/// RPython equivalent (`rpython/tool/algo/regalloc.py:18-143
/// RegAllocator`):
/// ```python
/// class RegAllocator(object):
///     def __init__(self, graph, consider_var, ListOfKind): ...
///     def make_dependencies(self): ...
///     def coalesce_variables(self): ...
///     def find_node_coloring(self): ...
///     def find_num_colors(self): ...
///     def getcolor(self, v): ...
///     def swapcolors(self, col1, col2): ...
/// ```
struct SSAReprRegAllocator {
    depgraph: DependencyGraph<u16>,
    /// Union-find over register indices (RPython
    /// `tool.algo.unionfind.UnionFind.link_to_parent`). Created
    /// lazily; missing nodes self-rep.
    unionfind: HashMap<u16, u16>,
    /// RPython `UnionFind.weight` — only roots carry entries.
    unionfind_weight: HashMap<u16, usize>,
    coloring: HashMap<u16, u16>,
}

impl SSAReprRegAllocator {
    fn new() -> Self {
        Self {
            depgraph: DependencyGraph::new(),
            unionfind: HashMap::new(),
            unionfind_weight: HashMap::new(),
            coloring: HashMap::new(),
        }
    }

    /// `unionfind.find_rep` with path compression.
    fn find_rep(&mut self, v: u16) -> u16 {
        if !self.unionfind.contains_key(&v) {
            self.unionfind.insert(v, v);
            self.unionfind_weight.insert(v, 1);
            return v;
        }
        let mut root = v;
        while self.unionfind[&root] != root {
            root = self.unionfind[&root];
        }
        let mut current = v;
        while current != root {
            let next = self.unionfind[&current];
            self.unionfind.insert(current, root);
            current = next;
        }
        root
    }

    /// `unionfind.union` — weighted union, matching
    /// `rpython/tool/algo/unionfind.py:67-91`.
    fn union(&mut self, v0: u16, w0: u16) -> u16 {
        let r1 = self.find_rep(v0);
        let r2 = self.find_rep(w0);
        if r1 == r2 {
            return r1;
        }
        let w1 = self.unionfind_weight.get(&r1).copied().unwrap_or(1);
        let w2 = self.unionfind_weight.get(&r2).copied().unwrap_or(1);
        let (winner, loser) = if w1 >= w2 { (r1, r2) } else { (r2, r1) };
        self.unionfind.insert(loser, winner);
        self.unionfind_weight.remove(&loser);
        *self.unionfind_weight.entry(winner).or_insert(0) = w1 + w2;
        winner
    }

    /// `regalloc.py:26-77` `RegAllocator.make_dependencies`.
    ///
    /// RPython walks each block forward computing per-variable
    /// `die_at` (last-use index), then forward again killing dead
    /// variables and adding interference edges between every result
    /// register and the currently-alive set. Pyre's input is a flat
    /// `SSARepr`, so the equivalent live-set computation is done in
    /// a backward sweep with a fixpoint over labels (analogous to
    /// `liveness.py`'s alive-set propagation).
    fn make_dependencies(&mut self, ssarepr: &SSARepr, kind: Kind, external_inputs: &[u16]) {
        // regalloc.py:54-60 `for i, v in enumerate(livevars):
        //   ... for j in range(i): dg.add_edge(livevars[j], v)`.
        for (i, &v) in external_inputs.iter().enumerate() {
            self.depgraph.add_node(v);
            for j in 0..i {
                if external_inputs[j] != v {
                    self.depgraph.add_edge(external_inputs[j], v);
                }
            }
        }

        // Backward live-set walk over the SSARepr with a label
        // fixpoint. Equivalent in effect to `regalloc.py:62-77`'s
        // forward `die_at` driven loop, just expressed in the
        // direction natural to pyre's flat instruction list (matches
        // `super::liveness::_compute_liveness_must_continue`).
        let mut label2alive: HashMap<String, HashSet<u16>> = HashMap::new();
        loop {
            let mut alive: HashSet<u16> = HashSet::new();
            let mut must_continue = false;

            for insn in ssarepr.insns.iter().rev() {
                let label_name = match insn {
                    Insn::Label(label) => Some(label.name.clone()),
                    _ => None,
                };
                if let Some(name) = label_name {
                    let alive_at_point = label2alive.entry(name).or_default();
                    let prevlength = alive_at_point.len();
                    alive_at_point.extend(alive.iter().copied());
                    if prevlength != alive_at_point.len() {
                        must_continue = true;
                    }
                    continue;
                }
                match insn {
                    Insn::Unreachable => {
                        alive.clear();
                    }
                    Insn::Label(_) => unreachable!("handled above"),
                    Insn::Op { args, result, .. } => {
                        // Defs: `'->' result` interferes with everything
                        // currently alive (regalloc.py:70-76).
                        if let Some(reg) = result {
                            if reg.kind == kind {
                                self.depgraph.add_node(reg.index);
                                for &a in &alive {
                                    if a != reg.index {
                                        self.depgraph.add_node(a);
                                        self.depgraph.add_edge(reg.index, a);
                                    }
                                }
                                alive.remove(&reg.index);
                            }
                        }
                        // Uses: every Register / ListOfKind in args
                        // becomes alive for preceding instructions.
                        for x in args {
                            match x {
                                Operand::Register(reg) if reg.kind == kind => {
                                    alive.insert(reg.index);
                                    self.depgraph.add_node(reg.index);
                                }
                                Operand::ListOfKind(lst) if lst.kind == kind => {
                                    for y in &lst.content {
                                        if let Operand::Register(reg) = y {
                                            if reg.kind == kind {
                                                alive.insert(reg.index);
                                                self.depgraph.add_node(reg.index);
                                            }
                                        }
                                    }
                                }
                                Operand::TLabel(lbl) => follow_label(&mut alive, &label2alive, lbl),
                                Operand::Descr(rc) => {
                                    if let DescrOperand::SwitchDict(descr) = &**rc {
                                        for (_, label) in &descr.labels {
                                            follow_label(&mut alive, &label2alive, label);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            if !must_continue {
                break;
            }
        }
    }

    /// `regalloc.py:79-112` `RegAllocator.coalesce_variables`.
    ///
    /// RPython coalesces `link.args[i]` with `link.target.inputargs[i]`
    /// for every block-exit link. Pyre's `SSARepr` is post-flatten
    /// and has no `link.args ↔ inputargs` pairing — the flatten step
    /// dropped it in favor of pinned PyFrame slot indices. The
    /// SSARepr-level remnant of jump-edge unification is the `*_copy`
    /// instruction, which expresses `target_register := source`.
    /// Coalescing `*_copy`'s source and result lets the chordal
    /// coloring assign them the same color, turning the copy into a
    /// runtime no-op when src == dst.
    ///
    /// Note: SSARepr-level copy coalescing instead
    /// of FunctionGraph-level link.args coalescing. The effect is a
    /// strict subset of RPython's because pyre still does not see the
    /// original cross-block link representation.
    fn coalesce_variables(&mut self, ssarepr: &SSARepr, kind: Kind) {
        let copy_op = match kind {
            Kind::Int => "int_copy",
            Kind::Ref => "ref_copy",
            Kind::Float => "float_copy",
        };
        for insn in &ssarepr.insns {
            if let Insn::Op {
                opname,
                args,
                result,
            } = insn
            {
                if opname != copy_op {
                    continue;
                }
                let dst = match result {
                    Some(r) if r.kind == kind => *r,
                    _ => continue,
                };
                let src = match args.first() {
                    Some(Operand::Register(r)) if r.kind == kind => *r,
                    _ => continue,
                };
                self.try_coalesce(src.index, dst.index);
            }
        }
    }

    /// `regalloc.py:98-112` `RegAllocator._try_coalesce`.
    fn try_coalesce(&mut self, v: u16, w: u16) {
        let v0 = self.find_rep(v);
        let w0 = self.find_rep(w);
        if v0 == w0 {
            return;
        }
        if self.depgraph.has_edge(&v0, &w0) {
            return;
        }
        let rep = self.union(v0, w0);
        if rep == v0 {
            self.depgraph.coalesce(w0, v0);
        } else {
            self.depgraph.coalesce(v0, w0);
        }
    }

    /// `regalloc.py:114-120` `RegAllocator.find_node_coloring`.
    fn find_node_coloring(&mut self) {
        let coloring = self.depgraph.find_node_coloring();
        // RPython stores coloring keyed by union-find rep; pyre
        // expands back to all per-register entries by composing with
        // the union-find map so getcolor / rename can dereference any
        // original register index.
        self.coloring = HashMap::new();
        // Collect all registers seen (both in unionfind and in
        // depgraph rep set).
        let all_regs: HashSet<u16> = self
            .unionfind
            .keys()
            .copied()
            .chain(coloring.keys().copied())
            .collect();
        for v in all_regs {
            let rep = self.find_rep(v);
            if let Some(&color) = coloring.get(&rep) {
                self.coloring.insert(v, color as u16);
            }
        }
    }

    /// `regalloc.py:129-130` `RegAllocator.getcolor`.
    fn getcolor(&mut self, v: u16) -> Option<u16> {
        let rep = self.find_rep(v);
        // The expanded coloring always carries per-register entries
        // (see find_node_coloring), but defend against unused regs.
        self.coloring
            .get(&v)
            .copied()
            .or_else(|| self.coloring.get(&rep).copied())
    }

    /// `regalloc.py:138-143` `RegAllocator.swapcolors`.
    fn swapcolors(&mut self, col1: u16, col2: u16) {
        for color in self.coloring.values_mut() {
            if *color == col1 {
                *color = col2;
            } else if *color == col2 {
                *color = col1;
            }
        }
    }

    /// `rpython/tool/algo/regalloc.py:122-127` `RegAllocator.find_num_colors`:
    /// `max(self._coloring.values())+1 if self._coloring else 0`.
    fn find_num_colors(&self) -> u16 {
        self.coloring.values().copied().max().map_or(0, |m| m + 1)
    }
}

#[inline]
fn follow_label(
    alive: &mut HashSet<u16>,
    label2alive: &HashMap<String, HashSet<u16>>,
    lbl: &TLabel,
) {
    if let Some(alive_at_point) = label2alive.get(&lbl.name) {
        alive.extend(alive_at_point.iter().copied());
    }
}

/// Apply the rename table to the `SSARepr` in place.
///
/// Walks every `Insn::Op` (including `-live-` markers), rewriting
/// `Register` operands and `result` registers through the rename
/// table. Leaves constants, labels, descrs, and indirect-call-target
/// operands untouched.
///
/// `-live-` markers happen to be empty at the point this function
/// runs in the current pipeline (dispatch emits placeholders,
/// `filter_liveness_in_place` only populates them AFTER, per
/// `codewriter.py:44-56` parity), so the walk is a no-op on them —
/// but the handling is order-agnostic: if a `-live-` marker ever
/// arrives here with registers, they'd be remapped consistently with
/// the surrounding ops.
pub(super) fn apply_rename(ssarepr: &mut SSARepr, rename: &[Vec<u16>; 3]) {
    if rename.iter().all(|v| v.is_empty()) {
        return;
    }
    for insn in ssarepr.insns.iter_mut() {
        match insn {
            Insn::Op { args, result, .. } => {
                if let Some(reg) = result {
                    rename_register(reg, rename);
                }
                for op in args.iter_mut() {
                    rename_operand(op, rename);
                }
            }
            Insn::Label(_) | Insn::Unreachable => {}
        }
    }
}

fn rename_operand(op: &mut Operand, rename: &[Vec<u16>; 3]) {
    match op {
        Operand::Register(reg) => rename_register(reg, rename),
        Operand::ListOfKind(lst) => {
            for inner in lst.content.iter_mut() {
                rename_operand(inner, rename);
            }
        }
        _ => {}
    }
}

#[inline]
fn rename_register(reg: &mut Register, rename: &[Vec<u16>; 3]) {
    reg.index = rename_lookup(rename, reg.kind, reg.index);
}

#[cfg(test)]
mod tests {
    use super::super::flatten::{Insn, Kind, ListOfKind, Operand, Register, SSARepr};
    use super::super::flow::{
        Block, Constant, FlowListOfKind, FunctionGraph, Link, SpaceOperation, Variable, VariableId,
        push_op,
    };
    use super::*;

    fn op_def(name: &str, args: Vec<Operand>, result: Register) -> Insn {
        Insn::op_with_result(name, args, result)
    }

    fn op_use(name: &str, args: Vec<Operand>) -> Insn {
        Insn::op(name, args)
    }

    fn reg(kind: Kind, idx: u16) -> Operand {
        Operand::reg(kind, idx)
    }

    fn r(kind: Kind, idx: u16) -> Register {
        Register::new(kind, idx)
    }

    fn flow_var(id: u32, kind: Kind) -> Variable {
        Variable::new(VariableId(id), kind)
    }

    #[test]
    fn all_kinds_driver_produces_regalloc_for_every_kind() {
        // Build a graph with one variable per kind as startblock inputs,
        // and a returnblock that takes a single Int so the link has
        // matching arity.
        let v0 = flow_var(0, Kind::Int);
        let vr = flow_var(1, Kind::Ref);
        let vf = flow_var(2, Kind::Float);
        let start = Block::shared(vec![v0.into(), vr.into(), vf.into()]);
        let graph = FunctionGraph::new("all_kinds", start.clone(), Some(v0));
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let regallocs = perform_register_allocation_all_kinds(&graph);
        for &kind in &Kind::ALL {
            let result = &regallocs[kind.index()];
            // Each kind has at least one variable (Int: v0 twice via
            // return link; Ref: vr in startblock inputargs; Float: vf).
            // Colorings never exceed num_colors.
            assert!(
                result.num_colors >= 1,
                "kind {kind:?} expected at least one color, got {}",
                result.num_colors
            );
            for (_id, color) in &result.coloring {
                assert!(
                    *color < result.num_colors,
                    "kind {kind:?} color {color} exceeds num_colors {}",
                    result.num_colors
                );
            }
        }
    }

    #[test]
    fn graph_regalloc_reuses_color_for_non_overlapping_values() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let graph = FunctionGraph::new("graph_regalloc", start.clone(), Some(v1));
        push_op(
            &start,
            SpaceOperation::new("same_as", vec![v0.into()], Some(v1.into()), 0),
        );
        start.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn perform_register_allocation_with_pairs_shares_color_for_pinned_scratch() {
        // Two INTERFERING Ref variables: v0 (inputarg) is kept live
        // past v1's definition by carrying both on the outgoing link,
        // so the unpinned chordal coloring must assign them different
        // colors.  `extra_coalesce_pairs` pre-merges them in the
        // union-find before `make_dependencies` so they collapse into
        // a single node and share a color — bypassing the interference
        // edge that `_try_coalesce` (regalloc.py:106) would otherwise
        // honour.
        let build_graph = || {
            let v0 = flow_var(0, Kind::Ref);
            let v1 = flow_var(1, Kind::Ref);
            let start = Block::shared(vec![v0.into()]);
            let mut graph = FunctionGraph::new("pin_share_color", start.clone(), None);
            push_op(
                &start,
                SpaceOperation::new("ref_copy", vec![v0.into()], Some(v1.into()), 0),
            );
            let v2 = flow_var(2, Kind::Ref);
            let v3 = flow_var(3, Kind::Ref);
            let next = graph.new_block(vec![v2.into(), v3.into()]);
            // Both v0 and v1 carried forward so the live-set at the
            // outgoing link contains both, forcing an interference
            // edge under the unpinned allocator.
            start.closeblock(vec![
                Link::new(vec![v0.into(), v1.into()], Some(next.clone()), None).into_ref(),
            ]);
            // returnblock arity is always 1 (a fresh untyped variable
            // when `return_var = None` was passed to FunctionGraph::new).
            next.closeblock(vec![
                Link::new(vec![v2.into()], Some(graph.returnblock.clone()), None).into_ref(),
            ]);
            (graph, v0, v1)
        };

        // Baseline: without pins, the interference forces distinct colors.
        let (graph_unpinned, v0_u, v1_u) = build_graph();
        let unpinned = perform_register_allocation_with_pairs(&graph_unpinned, Kind::Ref, &[]);
        let unpinned_v0 = unpinned.coloring.get(&v0_u.id).copied();
        let unpinned_v1 = unpinned.coloring.get(&v1_u.id).copied();
        assert!(unpinned_v0.is_some() && unpinned_v1.is_some());
        assert_ne!(
            unpinned_v0, unpinned_v1,
            "without pins, interfering v0 and v1 must get distinct colors"
        );

        // Pinned: pre-merge unifies them into one node before
        // make_dependencies so the interference edge never gets recorded.
        let (graph_pinned, v0_p, v1_p) = build_graph();
        let pin_pairs = vec![(v1_p.id, v0_p.id)];
        let pinned = perform_register_allocation_with_pairs(&graph_pinned, Kind::Ref, &pin_pairs);
        let pinned_v0 = pinned.coloring.get(&v0_p.id).copied();
        let pinned_v1 = pinned.coloring.get(&v1_p.id).copied();
        assert!(pinned_v0.is_some() && pinned_v1.is_some());
        assert_eq!(
            pinned_v0, pinned_v1,
            "pin must unify v0 and v1 even across an interference edge"
        );
    }

    #[test]
    fn perform_register_allocation_with_pairs_handles_unknown_scratch_id() {
        // Walker may produce scratch Variable IDs that never appear in
        // the canonical graph (walker-only emit sites).  Pin pairs
        // containing such IDs must not panic and must not strip the
        // inputarg's color entry.
        let v0 = flow_var(0, Kind::Ref);
        let start = Block::shared(vec![v0.into()]);
        let graph = FunctionGraph::new("pin_unknown_scratch", start.clone(), None);
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        // v_99 is not in the graph; pin (99, 0) should be benign.
        let pin_pairs = vec![(VariableId(99), v0.id)];
        let result = perform_register_allocation_with_pairs(&graph, Kind::Ref, &pin_pairs);
        assert!(
            result.coloring.get(&v0.id).is_some(),
            "inputarg v0 must retain a color even when pinned scratch ID 99 is absent from the graph"
        );
    }

    #[test]
    fn graph_regalloc_coalesces_goto_link_args_with_target_inputargs() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let mut graph = FunctionGraph::new("graph_goto", start.clone(), None);
        let next = graph.new_block(vec![v1.into()]);
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(next.clone()), None).into_ref(),
        ]);
        next.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn graph_regalloc_seeds_exception_extravars_as_colorable_nodes() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("graph_exc", start.clone(), None);
        let target = graph.new_block(Vec::new());
        let exc_type = flow_var(10, Kind::Int);
        let mut link = Link::new(Vec::new(), Some(target), None);
        link.extravars(Some(exc_type), None);
        start.closeblock(vec![link.into_ref()]);

        let result = perform_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&exc_type.id), Some(&0));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn graph_regalloc_marks_listofkind_args_as_uses() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let graph = FunctionGraph::new("graph_listofkind", start.clone(), Some(v1));
        push_op(
            &start,
            SpaceOperation::new(
                "same_as",
                vec![Constant::signed(1).into()],
                Some(v1.into()),
                0,
            ),
        );
        push_op(
            &start,
            SpaceOperation::new(
                "consume",
                vec![FlowListOfKind::new(Kind::Int, vec![v0.into()]).into()],
                None,
                0,
            ),
        );
        start.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_register_allocation(&graph, Kind::Int);
        assert_ne!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 2);
    }

    /// `flatten.py:88-100 enforce_input_args` parity at the graph
    /// allocator level: after the swap, every kind's startblock
    /// inputargs occupy colors `0, 1, 2, …` in source order.
    #[test]
    fn enforce_input_args_graph_side_normalises_inputarg_colors() {
        // 2 Ref inputargs + 1 Int inputarg, all live across an op
        // that defines fresh Variables of each kind so the chordal
        // coloring has to place every node on its own color.
        let a = flow_var(0, Kind::Ref);
        let b = flow_var(1, Kind::Ref);
        let i = flow_var(2, Kind::Int);
        let r0 = flow_var(10, Kind::Ref);
        let i0 = flow_var(11, Kind::Int);
        let start = Block::shared(vec![a.into(), b.into(), i.into()]);
        let mut graph = FunctionGraph::new("enforce_sim", start.clone(), None);
        push_op(
            &start,
            SpaceOperation::new("consume_ref", vec![a.into(), b.into()], Some(r0.into()), 0),
        );
        push_op(
            &start,
            SpaceOperation::new("consume_int", vec![i.into()], Some(i0.into()), 0),
        );
        let next = graph.new_block(vec![r0.into(), i0.into()]);
        start.closeblock(vec![
            Link::new(vec![r0.into(), i0.into()], Some(next), None).into_ref(),
        ]);

        let mut regallocs = perform_register_allocation_all_kinds(&graph);
        enforce_input_args(&graph, &mut regallocs);

        let ref_colors = &regallocs[Kind::Ref.index()].coloring;
        let int_colors = &regallocs[Kind::Int.index()].coloring;
        assert_eq!(
            ref_colors.get(&a.id).copied(),
            Some(0),
            "first Ref inputarg must occupy color 0 post-enforce_input_args"
        );
        assert_eq!(
            ref_colors.get(&b.id).copied(),
            Some(1),
            "second Ref inputarg must occupy color 1 post-enforce_input_args"
        );
        assert_eq!(
            int_colors.get(&i.id).copied(),
            Some(0),
            "first Int inputarg must occupy color 0 post-enforce_input_args"
        );
    }

    /// (a) inputargs land on consecutive colors `0..n-1` regardless
    /// of how the chordal coloring picks initial colors.
    #[test]
    fn enforce_input_args_normalises_inputarg_colors() {
        // 2 locals + 2 portal regs; introduce an op that defines a
        // fresh Ref register and has all 4 inputargs alive. The
        // chordal coloring assigns 5 different colors; without
        // enforce_input_args nothing constrains *which* color goes
        // to which register.
        let mut ssarepr = SSARepr::new("t");
        ssarepr.insns.push(op_def(
            "consume",
            vec![
                reg(Kind::Ref, 0),
                reg(Kind::Ref, 1),
                reg(Kind::Ref, 100),
                reg(Kind::Ref, 101),
            ],
            r(Kind::Ref, 200),
        ));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 200)]));

        let inputs = ExternalInputs {
            portal_frame_reg: 100,
            portal_ec_reg: 101,
            portal_inputs: true,
        };
        let result = allocate_registers(&ssarepr, 2, inputs, &[]);
        let new = |old: u16| rename_lookup(&result.rename, Kind::Ref, old);
        // locals 0,1 → colors 0,1; portal regs → 2,3.
        assert_eq!(new(0), 0, "local 0 must keep color 0 after enforce");
        assert_eq!(new(1), 1, "local 1 must keep color 1 after enforce");
        assert_eq!(new(100), 2, "portal_frame_reg must land on color 2");
        assert_eq!(new(101), 3, "portal_ec_reg must land on color 3");
    }

    /// (b) RPython `regalloc.py:54-60` parity: a dead local's color is
    /// reused by an SSA temp whose live range starts after the local
    /// dies.  Resume data is Box-by-identity, not register-slot indexed,
    /// so temp/local color sharing is sound.
    #[test]
    fn dead_local_color_reused() {
        let mut ssarepr = SSARepr::new("t");
        ssarepr
            .insns
            .push(op_use("read_local", vec![reg(Kind::Ref, 0)]));
        ssarepr
            .insns
            .push(op_def("make_value", vec![], r(Kind::Ref, 100)));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 100)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 1, inputs, &[]);
        let new = |old: u16| rename_lookup(&result.rename, Kind::Ref, old);
        assert_eq!(new(0), 0, "local 0 stays at color 0 (enforce_input_args)");
        assert_eq!(
            new(100),
            0,
            "temp 100 reuses local 0's color (local dies before temp's def)"
        );
        assert_eq!(
            result.num_regs[Kind::Ref.index()],
            1,
            "single color for the disjoint live ranges"
        );
    }

    /// (c) `num_regs[kind]` equals the number of distinct colors
    /// used after coloring (codewriter.py:62-67).
    #[test]
    fn num_regs_matches_max_color_plus_one() {
        // Force 3 simultaneously-live Ref registers + 1 Int.
        let mut ssarepr = SSARepr::new("t");
        ssarepr.insns.push(op_def("a", vec![], r(Kind::Ref, 50)));
        ssarepr.insns.push(op_def("b", vec![], r(Kind::Ref, 51)));
        ssarepr.insns.push(op_def("c", vec![], r(Kind::Ref, 52)));
        ssarepr.insns.push(op_use(
            "use_all",
            vec![reg(Kind::Ref, 50), reg(Kind::Ref, 51), reg(Kind::Ref, 52)],
        ));
        ssarepr.insns.push(op_def("i0", vec![], r(Kind::Int, 7)));
        ssarepr
            .insns
            .push(op_use("use_int", vec![reg(Kind::Int, 7)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 0, inputs, &[]);
        assert_eq!(result.num_regs[Kind::Ref.index()], 3);
        assert_eq!(result.num_regs[Kind::Int.index()], 1);
        assert_eq!(result.num_regs[Kind::Float.index()], 0);
    }

    #[test]
    fn union_keeps_heavier_partition_as_representative() {
        let mut alloc = SSAReprRegAllocator::new();

        assert_eq!(alloc.union(10, 11), 10);
        assert_eq!(alloc.find_rep(10), 10);
        assert_eq!(alloc.find_rep(11), 10);

        assert_eq!(alloc.union(12, 13), 12);
        assert_eq!(alloc.find_rep(12), 12);
        assert_eq!(alloc.find_rep(13), 12);

        assert_eq!(alloc.union(10, 12), 10);
        assert_eq!(alloc.find_rep(10), 10);
        assert_eq!(alloc.find_rep(11), 10);
        assert_eq!(alloc.find_rep(12), 10);
        assert_eq!(alloc.find_rep(13), 10);
    }

    /// (d) `coalesce_variables` unifies a `*_copy dst <- src` source
    /// and target into the same color when they don't interfere.
    #[test]
    fn move_source_and_target_coalesce_to_same_color() {
        // r5 = produce; ref_copy r6 <- r5; use r6.
        // r5 dies at the copy; r6 takes over. With coalescing they
        // should share a color.
        let mut ssarepr = SSARepr::new("t");
        ssarepr
            .insns
            .push(op_def("produce", vec![], r(Kind::Ref, 5)));
        ssarepr
            .insns
            .push(op_def("ref_copy", vec![reg(Kind::Ref, 5)], r(Kind::Ref, 6)));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 6)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 0, inputs, &[]);
        let new5 = rename_lookup(&result.rename, Kind::Ref, 5);
        let new6 = rename_lookup(&result.rename, Kind::Ref, 6);
        assert_eq!(
            new5, new6,
            "coalesce_variables should give ref_copy src and dst the same color (got {} vs {})",
            new5, new6
        );
        assert_eq!(
            result.num_regs[Kind::Ref.index()],
            1,
            "after coalesce only one Ref color is needed"
        );
    }

    /// (e) RPython `flatten.py:88-100` parity: non-inputarg registers
    /// reuse a dead inputarg's slot when their live range starts after
    /// the inputarg's last use.  Resume data is Box-by-identity, so the
    /// reuse is sound — the same color identifying two non-overlapping
    /// values does not break the resume contract.
    #[test]
    fn non_inputarg_can_reuse_inputarg_color() {
        let mut ssarepr = SSARepr::new("t");
        ssarepr.insns.push(op_use(
            "read_local",
            vec![Operand::ListOfKind(ListOfKind::new(
                Kind::Ref,
                vec![reg(Kind::Ref, 0)],
            ))],
        ));
        ssarepr
            .insns
            .push(op_def("make_value", vec![], r(Kind::Ref, 50)));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 50)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 1, inputs, &[]);
        let new50 = rename_lookup(&result.rename, Kind::Ref, 50);
        assert_eq!(
            new50, 0,
            "non-inputarg reg 50 reuses dead inputarg 0's color (RPython parity)"
        );
    }

    #[test]
    fn weighted_union_prefers_heavier_partition() {
        let mut alloc = SSAReprRegAllocator::new();
        assert_eq!(alloc.union(1, 2), 1);
        assert_eq!(alloc.union(3, 4), 3);
        assert_eq!(alloc.union(3, 5), 3);
        assert_eq!(
            alloc.union(1, 3),
            3,
            "RPython UnionFind keeps the heavier partition's representative"
        );
        assert_eq!(alloc.find_rep(1), 3);
        assert_eq!(alloc.find_rep(2), 3);
        assert_eq!(alloc.find_rep(4), 3);
        assert_eq!(alloc.find_rep(5), 3);
    }
}
