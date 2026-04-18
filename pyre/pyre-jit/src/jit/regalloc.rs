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

use std::collections::{HashMap, HashSet};

use majit_codewriter::model::ValueId;
use majit_codewriter::regalloc::DependencyGraph;
use pyre_jit_trace::RegisterMapping;

use super::flatten::{DescrOperand, Insn, Kind, Operand, Register, SSARepr, TLabel};

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
    /// implicitly identity.
    pub rename: HashMap<(Kind, u16), u16>,
    /// Register layout consumed by blackhole resume.
    pub mapping: RegisterMapping,
}

/// External-input registers preserved across coloring.
///
/// RPython parity: `regalloc.py:54-60` adds pairwise interference
/// edges between every variable in `block.inputargs`. Pyre's analog
/// is the set of registers populated externally by the blackhole at
/// resume time:
///
/// - `frame.locals_w()[i]` → Ref register `i` (for `i in 0..nlocals`),
///   loaded by `blackhole_from_jit_frame` (call_jit.rs:543-567).
/// - `virtualizable_ptr` → portal `frame_reg`, loaded by
///   `BlackholeInterpreter::fill_portal_registers`
///   (blackhole.rs:1133-1140).
/// - `execution_context` → portal `ec_reg`, same site.
///
/// These registers must keep distinct color slots because the
/// blackhole writes each one to a fixed target after rename. Returning
/// them through `external_input_registers` lets `color_kind` add the
/// pairwise edges before the chordal-coloring pass runs.
pub(super) struct ExternalInputs {
    pub portal_frame_reg: u16,
    pub portal_ec_reg: u16,
    pub portal_inputs: bool,
}

/// Run register allocation on `ssarepr` and produce the rename map +
/// blackhole `RegisterMapping`.
///
/// `nlocals` is the number of CPython fast locals (`code.varnames.len()`).
/// `depth_at_py_pc[py_pc]` is the operand-stack depth at each Python
/// PC (above `stack_base`), supplied by the dispatch loop.
pub(super) fn allocate_registers(
    ssarepr: &SSARepr,
    nlocals: usize,
    depth_at_py_pc: &[u16],
    inputs: ExternalInputs,
) -> RegallocResult {
    // codewriter.py:45-47 `for kind in KINDS:
    //   regallocs[kind] = perform_register_allocation(graph, kind)`
    let mut rename: HashMap<(Kind, u16), u16> = HashMap::new();
    for &kind in &Kind::ALL {
        let mut external: Vec<u16> = Vec::new();
        if kind == Kind::Ref {
            // RPython regalloc.py:54-60 inputargs analog. Locals
            // 0..nlocals are externally populated by blackhole resume
            // (call_jit.rs:543-567); they must keep distinct colors so
            // the per-PC table maps each PyFrame slot to a distinct
            // JIT register.
            for i in 0..nlocals as u16 {
                external.push(i);
            }
            // Portal red args (frame, ec) are externally populated by
            // BlackholeInterpreter::fill_portal_registers
            // (blackhole.rs:1133-1140); they need distinct colors too.
            if inputs.portal_inputs {
                if inputs.portal_frame_reg != u16::MAX {
                    external.push(inputs.portal_frame_reg);
                }
                if inputs.portal_ec_reg != u16::MAX {
                    external.push(inputs.portal_ec_reg);
                }
            }
        }
        let coloring = color_kind(ssarepr, kind, &external);
        for (pre_index, post_index) in coloring {
            if pre_index != post_index {
                rename.insert((kind, pre_index), post_index);
            }
        }
    }

    // Build the PerPc mapping from the renamed pinned layout.
    //
    // Pre-rename layout (current dispatch loop convention):
    //   local i lives in Ref register `i` (for `i < nlocals`)
    //   stack slot d lives in Ref register `nlocals + d`
    //
    // After rename, look each pre-index up through `rename` (or fall
    // back to identity); the resulting per-PC tables tell blackhole
    // resume which renamed register to load each PyFrame slot into.
    let num_pcs = depth_at_py_pc.len();
    let mut local_to_reg: Vec<Vec<u16>> = Vec::with_capacity(num_pcs);
    let mut stack_to_reg: Vec<Vec<u16>> = Vec::with_capacity(num_pcs);
    for py_pc in 0..num_pcs {
        let mut row_locals: Vec<u16> = Vec::with_capacity(nlocals);
        for local_idx in 0..nlocals {
            let pre = local_idx as u16;
            row_locals.push(rename.get(&(Kind::Ref, pre)).copied().unwrap_or(pre));
        }
        local_to_reg.push(row_locals);

        let depth = depth_at_py_pc[py_pc] as usize;
        let mut row_stack: Vec<u16> = Vec::with_capacity(depth);
        for d in 0..depth {
            let pre = (nlocals + d) as u16;
            row_stack.push(rename.get(&(Kind::Ref, pre)).copied().unwrap_or(pre));
        }
        stack_to_reg.push(row_stack);
    }

    let mapping = RegisterMapping::PerPc {
        local_to_reg,
        stack_to_reg,
    };

    RegallocResult { rename, mapping }
}

/// Run chordal coloring on the registers of one kind referenced by
/// `ssarepr`. Returns the (pre_index → post_index) mapping for that kind.
///
/// `external_inputs` is the list of register indices that are populated
/// externally (block.inputargs analog) and therefore must keep distinct
/// colors. RPython parity: `regalloc.py:54-60`.
///
/// **Identity constraint for external inputs.** Pyre's blackhole resume
/// path (`call_jit::blackhole_from_jit_frame`) pre-loads each
/// PyFrame slot at index `i` into JIT register `i`. To keep this
/// channel correct the colors assigned to the external inputs MUST
/// equal their pre-coloring indices. The chordal coloring algorithm
/// can violate this if e.g. local 0 receives color 5 because some
/// stack register sits at color 0.
///
/// We therefore add cross-edges from every external input to every
/// other register that appears in the SSARepr def-use chain. This
/// pushes non-input colors above the input range, so locals keep
/// their original indices and stack/scratch fall into colors
/// `≥ external_inputs.len()`.
fn color_kind(ssarepr: &SSARepr, kind: Kind, external_inputs: &[u16]) -> HashMap<u16, u16> {
    // ── Pass 1: compute interference graph via backward walk ──
    //
    // Adapted from `super::liveness::_compute_liveness_must_continue`:
    // walks instructions in reverse, maintains the per-kind alive set,
    // propagates the set across labels via fixpoint iteration, and adds
    // an interference edge between every (def, currently-alive) pair.
    let mut graph = DependencyGraph::new();

    // RPython regalloc.py:54-60 `for i, v in enumerate(livevars): ... for j in range(i): dg.add_edge(livevars[j], v)`.
    // Pyre input args (locals + portal frame/ec) get pairwise interference.
    for (i, &v) in external_inputs.iter().enumerate() {
        graph.add_node(ValueId(v as usize));
        for j in 0..i {
            if external_inputs[j] != v {
                graph.add_edge(ValueId(external_inputs[j] as usize), ValueId(v as usize));
            }
        }
    }

    // Collect every register that appears in the SSARepr so that the
    // identity-constraint loop below can add cross-edges from each
    // external input to every other register. RPython has no analog
    // because RPython jitcode register indices are abstract colors;
    // pyre's `blackhole_from_jit_frame` pre-loads PyFrame slot `i`
    // into JIT register `i`, which the coloring must preserve.
    let mut all_regs: HashSet<u16> = HashSet::new();
    fn visit_op(all_regs: &mut HashSet<u16>, kind: Kind, op: &Operand) {
        match op {
            Operand::Register(r) if r.kind == kind => {
                all_regs.insert(r.index);
            }
            Operand::ListOfKind(lst) if lst.kind == kind => {
                for inner in &lst.content {
                    if let Operand::Register(r) = inner {
                        if r.kind == kind {
                            all_regs.insert(r.index);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    for insn in &ssarepr.insns {
        match insn {
            Insn::Op { args, result, .. } => {
                if let Some(reg) = result {
                    if reg.kind == kind {
                        all_regs.insert(reg.index);
                    }
                }
                for op in args {
                    visit_op(&mut all_regs, kind, op);
                }
            }
            Insn::Live(args) => {
                for op in args {
                    visit_op(&mut all_regs, kind, op);
                }
            }
            _ => {}
        }
    }
    let external_set: HashSet<u16> = external_inputs.iter().copied().collect();
    for &ext in external_inputs {
        for &other in &all_regs {
            if other != ext && !external_set.contains(&other) {
                graph.add_node(ValueId(other as usize));
                graph.add_edge(ValueId(ext as usize), ValueId(other as usize));
            }
        }
    }

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
                    // Defs: trailing `'->' result`. Result interferes
                    // with everything currently alive (post-def
                    // liveness).
                    if let Some(reg) = result {
                        if reg.kind == kind {
                            graph.add_node(ValueId(reg.index as usize));
                            for &a in &alive {
                                if a != reg.index {
                                    graph.add_node(ValueId(a as usize));
                                    graph
                                        .add_edge(ValueId(reg.index as usize), ValueId(a as usize));
                                }
                            }
                            alive.remove(&reg.index);
                        }
                    }
                    // Uses: every Register / ListOfKind in args. Add to
                    // alive so the next (preceding) instruction sees
                    // them as live.
                    for x in args {
                        match x {
                            Operand::Register(reg) if reg.kind == kind => {
                                alive.insert(reg.index);
                                graph.add_node(ValueId(reg.index as usize));
                            }
                            Operand::ListOfKind(lst) if lst.kind == kind => {
                                for y in &lst.content {
                                    if let Operand::Register(reg) = y {
                                        if reg.kind == kind {
                                            alive.insert(reg.index);
                                            graph.add_node(ValueId(reg.index as usize));
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

    // ── Pass 2: chordal coloring ──
    //
    // RPython tool/algo/color.py:70-85 `find_node_coloring`. Returns
    // `ValueId → color` (small non-negative integers).
    let coloring = graph.find_node_coloring();
    coloring
        .into_iter()
        .map(|(vid, color)| (vid.0 as u16, color as u16))
        .collect()
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

/// Apply [`RegallocResult::rename`] to the `SSARepr` in place.
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
