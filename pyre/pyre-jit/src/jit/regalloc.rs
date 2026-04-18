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
//! `majit-codewriter`'s flow-graph regalloc through
//! `majit_codewriter::regalloc::DependencyGraph::find_node_coloring`
//! (line-by-line port of `rpython/tool/algo/color.py:31-85`).

use std::collections::{HashMap, HashSet};

use majit_codewriter::model::ValueId;
use majit_codewriter::regalloc::DependencyGraph;

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
/// RPython parity: `codewriter.py:45-47, 62-67`.
pub(super) fn allocate_registers(
    ssarepr: &SSARepr,
    nlocals: usize,
    inputs: ExternalInputs,
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
            if inputs.portal_inputs {
                if inputs.portal_frame_reg != u16::MAX {
                    external.push(inputs.portal_frame_reg);
                }
                if inputs.portal_ec_reg != u16::MAX {
                    external.push(inputs.portal_ec_reg);
                }
            }
        }
        let alloc = perform_register_allocation(ssarepr, kind, &external);
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
        num_regs.insert(kind, alloc.num_colors());
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

    // PRE-EXISTING-ADAPTATION (no RPython counterpart). The
    // trace-side decode in `pyre/pyre-jit-trace/src/trace_opcode.rs`
    // (and the bytecode walker) reads `register_idx < nlocals` as
    // "register holds local `register_idx`'s value", which requires
    // an INJECTIVE mapping in BOTH directions: not only must each
    // inputarg land at color `realcol` (handled above), but no
    // non-inputarg register may share a color in `0..n_inputs`
    // either. RPython has no equivalent invariant because its
    // blackhole resume reads PyFrame slots via `enumerate_vars`
    // rather than reusing register indices.
    //
    // Push every non-inputarg register's color above the inputarg
    // range by adding `n_inputs`. This preserves the chordal
    // coloring's correctness (any two non-inputargs that previously
    // shared a color still share `color + n_inputs`) and only
    // separates non-inputargs that previously collided with an
    // inputarg color. The cost is `num_regs_r` growing by
    // `n_inputs`, which is acceptable until the trace-side decode
    // is rewritten in terms of an explicit register_mapping (see
    // reviewer plan step #5 — multi-session refactor).
    let input_set: HashSet<u16> = input_indices.iter().copied().collect();
    let n_inputs = input_indices.len() as u16;
    if n_inputs > 0 {
        for (v, color) in alloc.coloring.iter_mut() {
            if !input_set.contains(v) {
                *color = color.checked_add(n_inputs).expect(
                    "enforce_input_args: color overflow when shifting non-inputargs above input range",
                );
            }
        }
    }
}

/// RPython `regalloc.py:6` `perform_register_allocation(graph, kind)`
/// + `tool/algo/regalloc.py:8-15`. Builds a `RegAllocator` and runs
/// the three-stage pipeline.
fn perform_register_allocation(
    ssarepr: &SSARepr,
    kind: Kind,
    external_inputs: &[u16],
) -> RegAllocator {
    let mut alloc = RegAllocator::new();
    alloc.make_dependencies(ssarepr, kind, external_inputs);
    let input_set: HashSet<u16> = external_inputs.iter().copied().collect();
    alloc.coalesce_variables(ssarepr, kind, &input_set);
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
    coloring: HashMap<u16, u16>,
}

impl RegAllocator {
    fn new() -> Self {
        Self {
            depgraph: DependencyGraph::new(),
            unionfind: HashMap::new(),
            coloring: HashMap::new(),
        }
    }

    /// `unionfind.find_rep` with path compression.
    fn find_rep(&mut self, v: u16) -> u16 {
        if !self.unionfind.contains_key(&v) {
            self.unionfind.insert(v, v);
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

    /// `unionfind.union` — picks `v0` as the new representative
    /// (matches `regalloc.py:106-112` `_, rep, _ = uf.union(v0, w0)`
    /// + the if/else that picks `dg.coalesce(loser, rep)`).
    fn union(&mut self, v0: u16, w0: u16) -> u16 {
        let r1 = self.find_rep(v0);
        let r2 = self.find_rep(w0);
        if r1 == r2 {
            return r1;
        }
        // RPython UnionFind picks by weight; a deterministic
        // smaller-index rep keeps coalesce results identical across
        // runs without affecting correctness.
        let (rep, loser) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
        self.unionfind.insert(loser, rep);
        rep
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
                    Insn::Label(label) => {
                        let alive_at_point = label2alive.entry(label.name.clone()).or_default();
                        let prevlength = alive_at_point.len();
                        alive_at_point.extend(alive.iter().copied());
                        if prevlength != alive_at_point.len() {
                            must_continue = true;
                        }
                    }
                    Insn::Live(args) => {
                        for x in args {
                            match x {
                                Operand::Register(reg) if reg.kind == kind => {
                                    alive.insert(reg.index);
                                }
                                Operand::TLabel(lbl) => follow_label(&mut alive, &label2alive, lbl),
                                _ => {}
                            }
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
    /// SSARepr-level remnant of jump-edge unification is the
    /// `move_X` instruction, which expresses
    /// `target_register := source_register`. Coalescing `move_X`'s
    /// source and result lets the chordal coloring assign them the
    /// same color, turning the move into a NOP at runtime
    /// (assembler.rs `move_*` emission becomes a no-op when src==dst).
    ///
    /// PRE-EXISTING-ADAPTATION: SSARepr-level move coalescing
    /// instead of FunctionGraph-level link.args coalescing. The
    /// effect is a strict subset of RPython's because pyre never
    /// sees the cross-block link representation.
    ///
    /// Additionally skip coalescing across the inputarg / non-inputarg
    /// boundary: the trace-side `idx < nlocals` decode requires
    /// inputarg colors to remain disjoint from non-inputarg colors
    /// after `enforce_input_args` shifts the latter. Collapsing the
    /// two via coalesce would put a non-inputarg into the same
    /// union as an inputarg, after which the shift would split the
    /// union and corrupt the rename map.
    fn coalesce_variables(&mut self, ssarepr: &SSARepr, kind: Kind, input_set: &HashSet<u16>) {
        let move_op = match kind {
            Kind::Int => "move_i",
            Kind::Ref => "move_r",
            Kind::Float => "move_f",
        };
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
                if opname != move_op && opname != copy_op {
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
                let src_in = input_set.contains(&src.index);
                let dst_in = input_set.contains(&dst.index);
                if src_in != dst_in {
                    // Skip the cross-boundary case (see fn docstring).
                    continue;
                }
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

    /// `regalloc.py:122-127` `RegAllocator.find_num_colors` →
    /// `max(coloring.values())+1` or 0.
    fn num_colors(&self) -> u16 {
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
/// Walks every `Insn::Op` and `Insn::Live`, rewriting `Register`
/// operands and `result` registers through the rename table. Leaves
/// constants, labels, descrs, and indirect-call-target operands
/// untouched.
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
            Insn::Live(args) => {
                for op in args.iter_mut() {
                    rename_operand(op, rename);
                }
            }
            Insn::Label(_) | Insn::Unreachable => {}
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
        };
        let result = allocate_registers(&ssarepr, 2, inputs);
        let mut new = |old: u16| result.rename.get(&(Kind::Ref, old)).copied().unwrap_or(old);
        // locals 0,1 → colors 0,1; portal regs → 2,3.
        assert_eq!(new(0), 0, "local 0 must keep color 0 after enforce");
        assert_eq!(new(1), 1, "local 1 must keep color 1 after enforce");
        assert_eq!(new(100), 2, "portal_frame_reg must land on color 2");
        assert_eq!(new(101), 3, "portal_ec_reg must land on color 3");
    }

    /// (b) A dead local's color is reused by a later temp value at
    /// the chordal-coloring stage. The pyre-only enforce_input_args
    /// shift then lifts the temp's color above the inputarg range
    /// so the trace-side `idx < nlocals` decode stays injective.
    /// Tests both that the chordal coloring is willing to reuse
    /// (no spurious cross-edge in the dependency graph) and that
    /// the shift preserves the trace-side invariant.
    #[test]
    fn dead_local_color_reused_then_shifted() {
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
        let result = allocate_registers(&ssarepr, 1, inputs);
        let mut new = |old: u16| result.rename.get(&(Kind::Ref, old)).copied().unwrap_or(old);
        assert_eq!(new(0), 0, "local 0 stays at color 0 (enforce_input_args)");
        // Pre-shift, temp 100 would have color 0 (reuses dead local 0).
        // Post-shift, it's pushed to color 0 + n_inputs = 1.
        assert_eq!(
            new(100),
            1,
            "temp 100 reuses dead local 0's color but is shifted above inputarg range"
        );
        assert_eq!(
            result.num_regs.get(&Kind::Ref).copied(),
            Some(2),
            "1 inputarg color + 1 shifted temp color = 2"
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
        let result = allocate_registers(&ssarepr, 0, inputs);
        assert_eq!(result.num_regs.get(&Kind::Ref).copied(), Some(3));
        assert_eq!(result.num_regs.get(&Kind::Int).copied(), Some(1));
        assert_eq!(result.num_regs.get(&Kind::Float).copied(), Some(0));
    }

    /// (d) `coalesce_variables` unifies a `move_X dst <- src` source
    /// and target into the same color when they don't interfere.
    #[test]
    fn move_source_and_target_coalesce_to_same_color() {
        // r5 = produce; move_r r6 <- r5; use r6
        // r5 dies at the move; r6 takes over. With coalescing they
        // should share a color.
        let mut ssarepr = SSARepr::new("t");
        ssarepr
            .insns
            .push(op_def("produce", vec![], r(Kind::Ref, 5)));
        ssarepr
            .insns
            .push(op_def("move_r", vec![reg(Kind::Ref, 5)], r(Kind::Ref, 6)));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 6)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 0, inputs);
        let new5 = result.rename.get(&(Kind::Ref, 5)).copied().unwrap_or(5);
        let new6 = result.rename.get(&(Kind::Ref, 6)).copied().unwrap_or(6);
        assert_eq!(
            new5, new6,
            "coalesce_variables should give move_r src and dst the same color (got {} vs {})",
            new5, new6
        );
        assert_eq!(
            result.num_regs.get(&Kind::Ref).copied(),
            Some(1),
            "after coalesce only one Ref color is needed"
        );
    }

    /// (e) PRE-EXISTING-ADAPTATION invariant: every non-inputarg
    /// register's color lies STRICTLY above the inputarg range
    /// (`color >= n_inputs`). The trace-side `idx < nlocals` decode
    /// in `trace_opcode.rs` requires this injective separation.
    /// (RPython has no equivalent because its blackhole resume reads
    /// PyFrame slots via `enumerate_vars` rather than reusing
    /// register indices.)
    #[test]
    fn non_inputarg_color_is_above_inputarg_range() {
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
        let result = allocate_registers(&ssarepr, 1, inputs);
        let new50 = result.rename.get(&(Kind::Ref, 50)).copied().unwrap_or(50);
        assert!(
            new50 >= 1,
            "non-inputarg reg 50 must land at color >= n_inputs (1), got {}",
            new50
        );
    }
}
