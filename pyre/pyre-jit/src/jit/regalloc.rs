//! Post-pass register allocation for pyre's per-CodeObject SSARepr.
//!
//! Mirrors the two-file split in RPython:
//!
//!   * `rpython/jit/codewriter/regalloc.py:6-8`
//!     `perform_register_allocation(graph, kind)` — thin wrapper around
//!     `tool.algo.regalloc.perform_register_allocation`. Pyre's analog is
//!     `perform_register_allocation` below.
//!   * `rpython/tool/algo/regalloc.py:8-15`
//!     ```python
//!     regalloc = RegAllocator(graph, consider_var, ListOfKind)
//!     regalloc.make_dependencies()    # interference graph
//!     regalloc.coalesce_variables()   # union-find on jump edges
//!     regalloc.find_node_coloring()   # chordal coloring
//!     ```
//!     Pyre's analog is `RegAllocator` + the three private methods of
//!     the same name.
//!   * `rpython/jit/codewriter/flatten.py:88-100` `enforce_input_args` —
//!     after coloring, `swapcolors` rotates inputarg colors into
//!     `0..n-1`. Pyre's analog is `enforce_input_args` below.
//!   * `rpython/jit/codewriter/codewriter.py:62-67` —
//!     `num_regs[kind] = max(coloring)+1` per kind, packed into the
//!     `JitCode`. Pyre's analog is `RegAllocator::num_colors` plus the
//!     `AllocationResult.num_regs` field.
//!
//! Architecture difference (PRE-EXISTING-ADAPTATION): RPython's
//! `RegAllocator` consumes a `FunctionGraph` (block + link.args
//! structure). Pyre's input is a CPython `CodeObject` translated into
//! a flat `SSARepr` by the dispatch loop, so `make_dependencies` works
//! over the populated `SSARepr` via a backward live-set walk and
//! `coalesce_variables` operates on `move_X` instructions (the
//! SSARepr-level remnant of jump-edge `link.args ↔ inputargs`
//! pairings). The chordal coloring algorithm itself is shared with
//! `majit-translate`'s flow-graph regalloc through
//! `majit_translate::regalloc::DependencyGraph::find_node_coloring`
//! (line-by-line port of `rpython/tool/algo/color.py:31-85`).

use std::collections::{HashMap, HashSet};

use majit_translate::model::ValueId;
use majit_translate::regalloc::DependencyGraph;

use super::flatten::{DescrOperand, Insn, Kind, Operand, Register, SSARepr, TLabel};

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
pub(super) struct ExternalInputs {
    pub portal_frame_reg: u16,
    pub portal_ec_reg: u16,
    pub portal_inputs: bool,
    /// Pre-rename stack base (== `nlocals` in the walker's layout).
    ///
    /// PRE-EXISTING-ADAPTATION. The trace-side decoder in
    /// `pyre/pyre-jit-trace/src/trace_opcode.rs` reads
    /// `register_idx >= nlocals` as "register holds stack slot
    /// `register_idx - nlocals`", so post-regalloc stack colors must
    /// equal `stack_base + slot_index`. Without pinning, the chordal
    /// coloring is free to place stack regs at any color, which breaks
    /// the tracer's `stack_values[idx - nlocals]` lookup — see
    /// `memory/phase1_step_d1_regalloc_rename_finding.md`.
    pub stack_base: u16,
    /// Number of stack slots to pin (clamp of `depth_at_pc.iter().max()`).
    ///
    /// RPython has no analog: its FunctionGraph decode reads PyFrame
    /// slots via `enumerate_vars` and never reuses Python-semantic
    /// stack indices as register names.
    pub max_stack_depth: u16,
}

/// Result of `allocate_registers`.
///
/// `rename` carries the per-kind `(pre_index → post_index)` map
/// applied by `apply_rename`. `num_regs` carries the per-kind
/// `max(color)+1` value the assembler stores in `JitCode.num_regs_*`
/// (codewriter.py:62-67).
pub(super) struct AllocationResult {
    pub rename: HashMap<(Kind, u16), u16>,
    pub num_regs: HashMap<Kind, u16>,
}

/// Run register allocation on `ssarepr` and produce the rename map
/// plus per-kind `num_regs`.
///
/// `nlocals` is the number of CPython fast locals (`code.varnames.len()`).
///
/// `cfg_coalesce_pairs` is the output of
/// `codewriter::collect_link_slot_pairs` — `(source_slot,
/// target_slot)` pairs from CFG link boundaries, all of Ref kind
/// because every `FrameState.mergeable()` position in pyre holds a
/// Ref-kind Variable (locals, stack, last_exc pair).  See
/// `collect_link_slot_pairs` docstring + `regalloc.py:79-96`.
///
/// RPython parity: `codewriter.py:45-47, 62-67`.
pub(super) fn allocate_registers(
    ssarepr: &SSARepr,
    nlocals: usize,
    inputs: ExternalInputs,
    cfg_coalesce_pairs: &[(u16, u16)],
) -> AllocationResult {
    // codewriter.py:45-47 `for kind in KINDS:
    //   regallocs[kind] = perform_register_allocation(graph, kind)`
    let mut allocators: HashMap<Kind, RegAllocator> = HashMap::new();
    for &kind in &Kind::ALL {
        let mut external: Vec<u16> = Vec::new();
        if kind == Kind::Ref {
            for i in 0..nlocals as u16 {
                external.push(i);
            }
            // Pyre-only: stack regs are Python-semantic pinned slots
            // (see `ExternalInputs::stack_base` docstring). Pin them
            // before portal red args so `enforce_input_args` assigns
            // them colors `nlocals..nlocals+max_stack_depth`, matching
            // the trace-side `stack_values[idx - nlocals]` decode.
            for d in 0..inputs.max_stack_depth {
                external.push(inputs.stack_base + d);
            }
            if inputs.portal_inputs {
                if inputs.portal_frame_reg != u16::MAX {
                    external.push(inputs.portal_frame_reg);
                }
                if inputs.portal_ec_reg != u16::MAX {
                    external.push(inputs.portal_ec_reg);
                }
            }
        }
        // `cfg_coalesce_pairs` are CFG-level Variable pairs from Link
        // boundaries (regalloc.py:79-96).  All mergeable positions in
        // pyre's FrameState are Ref-kind, so pass them only to the
        // Ref allocator; Int / Float regalloc sees an empty slice.
        let cfg_pairs_for_kind: &[(u16, u16)] = if kind == Kind::Ref {
            cfg_coalesce_pairs
        } else {
            &[]
        };
        let alloc = perform_register_allocation(ssarepr, kind, &external, cfg_pairs_for_kind);
        allocators.insert(kind, alloc);
    }

    // flatten.py:88-100 `enforce_input_args` — rotate inputarg colors
    // into 0..n-1 via swapcolors so the trace-side `idx < nlocals`
    // decode is guaranteed by code rather than by an interference
    // heuristic.
    enforce_input_args(&mut allocators, nlocals, &inputs);

    // codewriter.py:62-67 `num_regs = {kind: max(coloring)+1 if coloring else 0}`.
    let mut rename: HashMap<(Kind, u16), u16> = HashMap::new();
    let mut num_regs: HashMap<Kind, u16> = HashMap::new();
    for (&kind, alloc) in allocators.iter() {
        for (&pre, &post) in alloc.coloring.iter() {
            if pre != post {
                rename.insert((kind, pre), post);
            }
        }
        num_regs.insert(kind, alloc.find_num_colors());
    }
    AllocationResult { rename, num_regs }
}

/// `flatten.py:88-100` `GraphFlattener.enforce_input_args`.
///
/// RPython:
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
fn enforce_input_args(
    allocators: &mut HashMap<Kind, RegAllocator>,
    nlocals: usize,
    inputs: &ExternalInputs,
) {
    let alloc = allocators
        .get_mut(&Kind::Ref)
        .expect("Ref allocator must exist");
    let mut input_indices: Vec<u16> = (0..nlocals as u16).collect();
    for d in 0..inputs.max_stack_depth {
        input_indices.push(inputs.stack_base + d);
    }
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
                "enforce_input_args: inputarg color {} must be >= realcol {} (regalloc.py invariant)",
                curcol,
                realcol
            );
            alloc.swapcolors(realcol, curcol);
        }
    }
}

/// RPython `regalloc.py:6` `perform_register_allocation(graph, kind)`
/// + `tool/algo/regalloc.py:8-15`. Builds a `RegAllocator` and runs
/// the three-stage pipeline.
///
/// Dual coalesce sources:
///   1. `cfg_coalesce_pairs` — pre-computed `(source_slot,
///      target_slot)` pairs from `codewriter::collect_link_slot_pairs`.
///      These are the upstream `tool/algo/regalloc.py:79-96`
///      `link.args ↔ link.target.inputargs` pairs, projected from
///      Variables onto pyre's u16 register slots via the walker's
///      positional alignment (see `collect_link_slot_pairs`
///      docstring).  Trivially-equal pairs by `getoutputargs`
///      construction — their `try_coalesce` is a no-op when
///      `src_slot == dst_slot`, but the call preserves RPython's
///      exact iteration shape.
///   2. SSARepr `*_copy` scanner (PRE-EXISTING-ADAPTATION) — pyre's
///      walker emits intra-block `int_copy` / `ref_copy` /
///      `float_copy` ops for stack shuffling / STORE_FAST sequences
///      that have no CFG / Link representation.  The scanner unions
///      each copy's src and dst so the chordal coloring reuses one
///      color, turning the runtime copy into a no-op when
///      `src_color == dst_color`.  Upstream does not have this
///      SSARepr-level pass because `flatten.py:306-334`
///      `insert_renamings` places its copies post-coalesce, so
///      RPython's CFG-level `regalloc.py:79-96` is the sole coalesce
///      source.  In pyre both sources are orthogonal — the CFG
///      pairs cover cross-block link boundaries, the scanner covers
///      intra-block shuffles — so both run.
fn perform_register_allocation(
    ssarepr: &SSARepr,
    kind: Kind,
    external_inputs: &[u16],
    cfg_coalesce_pairs: &[(u16, u16)],
) -> RegAllocator {
    let mut alloc = RegAllocator::new();
    alloc.make_dependencies(ssarepr, kind, external_inputs);
    // regalloc.py:79-96 CFG-level coalesce: union link.args[i] with
    // link.target.inputargs[i] for every block exit.  Fed in from
    // `codewriter::collect_link_slot_pairs` (Ref kind only).
    for &(src_slot, dst_slot) in cfg_coalesce_pairs {
        alloc.try_coalesce(src_slot, dst_slot);
    }
    alloc.coalesce_variables(ssarepr, kind);
    alloc.find_node_coloring();
    alloc
}

/// `tool/algo/regalloc.py:18-143` `RegAllocator`.
///
/// RPython:
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
struct RegAllocator {
    depgraph: DependencyGraph,
    /// Union-find over register indices (RPython
    /// `tool.algo.unionfind.UnionFind`). Created lazily; missing
    /// nodes self-rep.
    unionfind: HashMap<u16, u16>,
    /// RPython `UnionFind.weight` — only roots carry entries.
    unionfind_weight: HashMap<u16, usize>,
    coloring: HashMap<u16, u16>,
}

impl RegAllocator {
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
            self.depgraph.add_node(ValueId(v as usize));
            for j in 0..i {
                if external_inputs[j] != v {
                    self.depgraph
                        .add_edge(ValueId(external_inputs[j] as usize), ValueId(v as usize));
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
                match insn {
                    // PRE-EXISTING-ADAPTATION (see `Insn::PcAnchor` in
                    // `flatten.rs`). Anchors carry no operand or
                    // liveness — pure SSARepr-position markers for
                    // post-assemble pc_map build. Skip in the
                    // interference walk.
                    Insn::PcAnchor(_) => {}
                    Insn::Label(label) => {
                        let alive_at_point = label2alive.entry(label.name.clone()).or_default();
                        let prevlength = alive_at_point.len();
                        alive_at_point.extend(alive.iter().copied());
                        if prevlength != alive_at_point.len() {
                            must_continue = true;
                        }
                    }
                    Insn::Unreachable => {
                        alive.clear();
                    }
                    Insn::Op { args, result, .. } => {
                        // Defs: `'->' result` interferes with everything
                        // currently alive (regalloc.py:70-76).
                        if let Some(reg) = result {
                            if reg.kind == kind {
                                self.depgraph.add_node(ValueId(reg.index as usize));
                                for &a in &alive {
                                    if a != reg.index {
                                        self.depgraph.add_node(ValueId(a as usize));
                                        self.depgraph.add_edge(
                                            ValueId(reg.index as usize),
                                            ValueId(a as usize),
                                        );
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
                                    self.depgraph.add_node(ValueId(reg.index as usize));
                                }
                                Operand::ListOfKind(lst) if lst.kind == kind => {
                                    for y in &lst.content {
                                        if let Operand::Register(reg) = y {
                                            if reg.kind == kind {
                                                alive.insert(reg.index);
                                                self.depgraph.add_node(ValueId(reg.index as usize));
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
    /// PRE-EXISTING-ADAPTATION: SSARepr-level copy coalescing instead
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
        if self
            .depgraph
            .has_edge(ValueId(v0 as usize), ValueId(w0 as usize))
        {
            return;
        }
        let rep = self.union(v0, w0);
        if rep == v0 {
            self.depgraph
                .coalesce(ValueId(w0 as usize), ValueId(v0 as usize));
        } else {
            self.depgraph
                .coalesce(ValueId(v0 as usize), ValueId(w0 as usize));
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
            .chain(coloring.keys().map(|vid| vid.0 as u16))
            .collect();
        for v in all_regs {
            let rep = self.find_rep(v);
            if let Some(&color) = coloring.get(&ValueId(rep as usize)) {
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
pub(super) fn apply_rename(ssarepr: &mut SSARepr, rename: &HashMap<(Kind, u16), u16>) {
    if rename.is_empty() {
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
            Insn::Label(_) | Insn::Unreachable | Insn::PcAnchor(_) => {}
        }
    }
}

fn rename_operand(op: &mut Operand, rename: &HashMap<(Kind, u16), u16>) {
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
fn rename_register(reg: &mut Register, rename: &HashMap<(Kind, u16), u16>) {
    if let Some(&new) = rename.get(&(reg.kind, reg.index)) {
        reg.index = new;
    }
}

#[cfg(test)]
mod tests {
    use super::super::flatten::{Insn, Kind, ListOfKind, Operand, Register, SSARepr};
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
            stack_base: 2,
            max_stack_depth: 0,
        };
        let result = allocate_registers(&ssarepr, 2, inputs, &[]);
        let mut new = |old: u16| result.rename.get(&(Kind::Ref, old)).copied().unwrap_or(old);
        // locals 0,1 → colors 0,1; portal regs → 2,3.
        assert_eq!(new(0), 0, "local 0 must keep color 0 after enforce");
        assert_eq!(new(1), 1, "local 1 must keep color 1 after enforce");
        assert_eq!(new(100), 2, "portal_frame_reg must land on color 2");
        assert_eq!(new(101), 3, "portal_ec_reg must land on color 3");
    }

    /// (b) A dead local's color is reused by a later temp value at
    /// the chordal-coloring stage (RPython `regalloc.py:54-60`).
    /// `enforce_input_args` only swaps the inputarg colors into the
    /// 0..n_inputs slots; non-inputarg colors are left where the
    /// chordal allocator placed them, so a temp whose live range
    /// starts after the inputarg dies legitimately reuses color 0.
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
            stack_base: 1,
            max_stack_depth: 0,
        };
        let result = allocate_registers(&ssarepr, 1, inputs, &[]);
        let mut new = |old: u16| result.rename.get(&(Kind::Ref, old)).copied().unwrap_or(old);
        assert_eq!(new(0), 0, "local 0 stays at color 0 (enforce_input_args)");
        assert_eq!(
            new(100),
            0,
            "temp 100 reuses dead local 0's color (chordal coloring)"
        );
        assert_eq!(
            result.num_regs.get(&Kind::Ref).copied(),
            Some(1),
            "single color shared between inputarg and dead-range temp"
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
            stack_base: 0,
            max_stack_depth: 0,
        };
        let result = allocate_registers(&ssarepr, 0, inputs, &[]);
        assert_eq!(result.num_regs.get(&Kind::Ref).copied(), Some(3));
        assert_eq!(result.num_regs.get(&Kind::Int).copied(), Some(1));
        assert_eq!(result.num_regs.get(&Kind::Float).copied(), Some(0));
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
            stack_base: 0,
            max_stack_depth: 0,
        };
        let result = allocate_registers(&ssarepr, 0, inputs, &[]);
        let new5 = result.rename.get(&(Kind::Ref, 5)).copied().unwrap_or(5);
        let new6 = result.rename.get(&(Kind::Ref, 6)).copied().unwrap_or(6);
        assert_eq!(
            new5, new6,
            "coalesce_variables should give ref_copy src and dst the same color (got {} vs {})",
            new5, new6
        );
        assert_eq!(
            result.num_regs.get(&Kind::Ref).copied(),
            Some(1),
            "after coalesce only one Ref color is needed"
        );
    }

    /// (e) RPython `enforce_input_args` only swaps inputargs into
    /// 0..n_inputs; non-inputarg registers can reuse those colors
    /// when the inputarg's live range has ended. This mirrors
    /// `flatten.py:88-100` exactly.
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
            stack_base: 1,
            max_stack_depth: 0,
        };
        let result = allocate_registers(&ssarepr, 1, inputs, &[]);
        let new50 = result.rename.get(&(Kind::Ref, 50)).copied().unwrap_or(50);
        assert_eq!(
            new50, 0,
            "non-inputarg reg 50 reuses dead inputarg 0's color (no shift)"
        );
    }

    #[test]
    fn weighted_union_prefers_heavier_partition() {
        let mut alloc = RegAllocator::new();
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
