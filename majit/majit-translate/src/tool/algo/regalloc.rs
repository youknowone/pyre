//! Register allocation on the control flow graph.
//!
//! RPython equivalent: `rpython/tool/algo/regalloc.py`.
//!
//! Operates on `FunctionGraph` (Block structure), NOT on flattened ops.
//! RPython runs regalloc BEFORE flatten: codewriter.py:45-47.
//!
//! 1. Build interference graph per-block (die_at analysis)
//! 2. Coalesce variables connected by Goto link args
//! 3. Greedy graph coloring via lexicographic BFS

use std::collections::{HashMap, HashSet};

use crate::flatten::RegKind;
use crate::model::{Block, ConcreteType, FunctionGraph, OpKind};
pub use crate::tool::algo::color::DependencyGraph;

// ── UnionFind (RPython tool/algo/unionfind.py) ────────────────────

#[derive(Debug, Clone)]
struct UnionFind<N: Eq + std::hash::Hash + Clone> {
    parent: HashMap<N, N>,
    weight: HashMap<N, usize>,
}

impl<N: Eq + std::hash::Hash + Clone> UnionFind<N> {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            weight: HashMap::new(),
        }
    }

    fn find_rep(&mut self, v: N) -> N {
        if !self.parent.contains_key(&v) {
            self.parent.insert(v.clone(), v.clone());
            self.weight.insert(v.clone(), 1);
            return v;
        }
        let mut root = v.clone();
        while self.parent[&root] != root {
            root = self.parent[&root].clone();
        }
        let mut current = v;
        while current != root {
            let next = self.parent[&current].clone();
            self.parent.insert(current, root.clone());
            current = next;
        }
        root
    }

    fn union(&mut self, v1: N, v2: N) -> N {
        let rep1 = self.find_rep(v1);
        let rep2 = self.find_rep(v2);
        if rep1 == rep2 {
            return rep1;
        }
        let w1 = self.weight.get(&rep1).copied().unwrap_or(1);
        let w2 = self.weight.get(&rep2).copied().unwrap_or(1);
        let (winner, loser) = if w1 >= w2 { (rep1, rep2) } else { (rep2, rep1) };
        self.parent.insert(loser.clone(), winner.clone());
        self.weight.remove(&loser);
        *self.weight.entry(winner.clone()).or_insert(0) = w1 + w2;
        winner
    }
}

// ── RegAllocator (RPython tool/algo/regalloc.py) ──────────────────

/// Private register-allocation work state on FunctionGraph.
///
/// RPython: `regalloc.py::RegAllocator`.
/// Runs BEFORE flatten, on Block/SpaceOperation structure.
#[derive(Debug)]
struct RegAllocatorState {
    depgraph: DependencyGraph<crate::flowspace::model::Variable>,
    unionfind: UnionFind<crate::flowspace::model::Variable>,
    coloring: HashMap<crate::flowspace::model::Variable, usize>,
}

impl RegAllocatorState {
    fn new() -> Self {
        Self {
            depgraph: DependencyGraph::new(),
            unionfind: UnionFind::new(),
            coloring: HashMap::new(),
        }
    }

    /// RPython: `RegAllocator.make_dependencies()` — regalloc.py:26-77.
    /// Per-block die_at analysis.
    fn make_dependencies(
        &mut self,
        graph: &FunctionGraph,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
    ) {
        for block in &graph.blocks {
            self.process_block(block, consider);
        }
    }

    /// Process one block: compute die_at, build interference edges.
    fn process_block(
        &mut self,
        block: &Block,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
    ) {
        // die_at: last usage index of each variable in this block.
        // Keyed on the backing Variable so the coalesce / coloring
        // passes downstream operate on the upstream-orthodox identity
        // (`tool/algo/regalloc.py:31 coloring: dict[Variable, int]`).
        // Pyre models a block's entry values either as `Block.inputargs` or
        // as `OpKind::Input` operations at block entry; both are alive from
        // entry and must be colored like upstream `block.inputargs`.  Collect
        // both so an `Input`-op result used only as an operand still gets a
        // node (otherwise regalloc never colors it and the codewriter
        // liveness pass panics on the uncolored operand).
        let block_input_vars: Vec<crate::flowspace::model::Variable> = block
            .input_variables()
            .cloned()
            .chain(block.operations.iter().filter_map(|op| match &op.kind {
                OpKind::Input { .. } => op.result.clone(),
                _ => None,
            }))
            .collect();
        let mut die_at: HashMap<crate::flowspace::model::Variable, usize> = HashMap::new();
        for var in &block_input_vars {
            die_at.insert(var.clone(), 0);
        }
        for (i, op) in block.operations.iter().enumerate() {
            if matches!(op.kind, OpKind::Input { .. }) {
                // Pyre's `OpKind::Input` mirrors `Block.inputargs`; upstream
                // RPython has no operation for these values, so they must not
                // extend liveness or look like a second definition.  Their
                // result is already seeded as a block input above.
                continue;
            }
            for var in crate::inline::op_variable_refs(&op.kind) {
                die_at.insert(var, i);
            }
            if let Some(result_var) = op.result.clone() {
                die_at.insert(result_var, i + 1);
            }
        }
        // Variables used in exit links stay alive until block end.
        // RPython `rpython/jit/codewriter/regalloc.py:71-78 compute_liveness`:
        // iterate `block.exits` for `link.args` + `block.exitswitch` for the
        // branch condition.
        for link in &block.exits {
            for arg in &link.args {
                if let Some(var) = arg.as_variable() {
                    die_at.remove(var);
                }
            }
        }
        match &block.exitswitch {
            Some(crate::model::ExitSwitch::Value(cond)) => {
                die_at.remove(cond);
            }
            Some(crate::model::ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    die_at.remove(arg);
                }
            }
            Some(crate::model::ExitSwitch::LastException) | None => {}
        }
        let mut die_list: Vec<(usize, crate::flowspace::model::Variable)> =
            die_at.into_iter().map(|(v, t)| (t, v)).collect();
        die_list.sort_by_key(|(t, _)| *t);

        // inputargs all interfere with each other
        let livevars: Vec<crate::flowspace::model::Variable> = block_input_vars
            .iter()
            .filter(|var| consider(var))
            .cloned()
            .collect();
        for (i, v) in livevars.iter().enumerate() {
            self.depgraph.add_node(v.clone());
            for j in 0..i {
                // Pyre can carry duplicate block input Variables after
                // parity-preserving flow rewrites.  RPython's shared
                // DependencyGraph asserts against self-edges, so skip them
                // locally instead of weakening the color.py port.
                if livevars[j] != *v {
                    self.depgraph.add_edge(livevars[j].clone(), v.clone());
                }
            }
        }
        let mut alive: HashSet<crate::flowspace::model::Variable> = livevars.into_iter().collect();

        // Scan ops, kill at die_at, add interference edges
        let mut die_index = 0;
        for (i, op) in block.operations.iter().enumerate() {
            while die_index < die_list.len() && die_list[die_index].0 == i {
                alive.remove(&die_list[die_index].1);
                die_index += 1;
            }
            if matches!(op.kind, OpKind::Input { .. }) {
                continue;
            }
            if let Some(result_var) = op.result.clone() {
                if consider(&result_var) {
                    self.depgraph.add_node(result_var.clone());
                    for v in &alive {
                        // The result can already be represented in `alive`
                        // when upstream-shaped exception/control-flow
                        // rewrites reuse a Variable.  Keep the shared
                        // color.py invariant by not asking it for a
                        // self-edge.
                        if *v != result_var {
                            self.depgraph.add_edge(v.clone(), result_var.clone());
                        }
                    }
                    alive.insert(result_var);
                }
            }
        }
    }

    /// RPython: `RegAllocator.coalesce_variables()` — regalloc.py:79-96.
    /// Coalesce link.args[i] with target.inputargs[i] for every exit
    /// link.  Upstream materialises `list(self.graph.iterblocks())` and
    /// `pop()`s from the END (regalloc.py:81-83), coalescing from the
    /// tail of the graph because the tail typically runs more often
    /// during blackholing; this visit order is load-bearing for which
    /// equally-valid coloring — and hence which interior register
    /// numbering — wins.  We reproduce it by walking
    /// [`FunctionGraph::iterblocks_order`] in reverse, NOT `graph.blocks`
    /// storage order (which is not guaranteed to equal `iterblocks()`
    /// order).  Upstream also pre-seeds the depgraph with nodes for
    /// `link.last_exception` / `link.last_exc_value` so any downstream
    /// `getcolor(v)` against those extravars finds a colored node.
    fn coalesce_variables(
        &mut self,
        graph: &FunctionGraph,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
    ) {
        let order = graph.iterblocks_order();
        for &bid in order.iter().rev() {
            let block = graph.block(bid);
            for link in &block.exits {
                // RPython `regalloc.py:92-95`: add `link.last_exception` and
                // `link.last_exc_value` to the dep graph so subsequent
                // `_try_coalesce` calls find a node for them in
                // `DependencyGraph.coalesce`'s `neighbours[vnew]`
                // lookup.  Without this, coalescing a link arg into an
                // exception-target inputarg leaves the rep outside
                // `neighbours`, and `find_node_coloring` silently skips
                // it (`color.py:25-27 getnodes()` filters
                // `_all_nodes` by `neighbours.contains_key`).
                if let Some(arg) = &link.last_exception {
                    if let Some(var) = arg.as_variable() {
                        self.depgraph.add_node(var.clone());
                    }
                }
                if let Some(arg) = &link.last_exc_value {
                    if let Some(var) = arg.as_variable() {
                        self.depgraph.add_node(var.clone());
                    }
                }
                let target_block = graph.block(link.target);
                let target_input_vars: Vec<crate::flowspace::model::Variable> =
                    target_block.input_variables().cloned().collect();
                for (arg, target_var) in link.args.iter().zip(target_input_vars.iter()) {
                    if let Some(arg_var) = arg.as_variable() {
                        if consider(arg_var) {
                            self.depgraph.add_node(arg_var.clone());
                        }
                        self.try_coalesce(arg_var, target_var, consider);
                    }
                }
            }
        }
    }

    /// `regalloc.py:_try_coalesce` direct port — operands are
    /// `Variable` instances (matching upstream's `for v, w in
    /// zip(link.args, target.inputargs)`), and the `consider`
    /// predicate reads off the same Variable handle.
    fn try_coalesce(
        &mut self,
        v: &crate::flowspace::model::Variable,
        w: &crate::flowspace::model::Variable,
        consider: &dyn Fn(&crate::flowspace::model::Variable) -> bool,
    ) {
        if !consider(v) || !consider(w) {
            return;
        }
        let v0 = self.unionfind.find_rep(v.clone());
        let w0 = self.unionfind.find_rep(w.clone());
        if v0 == w0 {
            return;
        }
        if self
            .depgraph
            .neighbours
            .get(&w0)
            .map_or(false, |ns| ns.contains(&v0))
        {
            return;
        }
        let rep = self.unionfind.union(v0.clone(), w0.clone());
        if rep == v0 {
            self.depgraph.coalesce(w0, v0);
        } else {
            self.depgraph.coalesce(v0, w0);
        }
    }

    fn find_node_coloring(&mut self) {
        self.coloring = self.depgraph.find_node_coloring();
    }

    fn getcolor(&mut self, var: &crate::flowspace::model::Variable) -> Option<usize> {
        let rep = self.unionfind.find_rep(var.clone());
        self.coloring.get(&rep).copied()
    }
}

// ── Public API ────────────────────────────────────────────────────

/// Result of register allocation for one kind.
///
/// RPython: `regalloc.py::RegAllocator`, returned by
/// `perform_register_allocation`.
///
/// `coloring` is keyed on the backing
/// [`crate::flowspace::model::Variable`] —
/// matching upstream RPython's `coloring: dict[Variable, int]`
/// (`tool/algo/regalloc.py:31`).  Consumers hold `&Variable` directly
/// (`flatten.rs:GraphFlattener::getcolor(&Variable)`,
/// `liveness::variable_to_register(&Variable, regallocs)`) and call
/// [`Self::color_for_variable`] / [`Self::contains_variable`].
#[derive(Debug, Clone)]
pub struct RegAllocator {
    pub coloring: HashMap<crate::flowspace::model::Variable, usize>,
    pub num_regs: usize,
}

impl RegAllocator {
    /// Look up the register color assigned to `var` — matches upstream
    /// `coloring: dict[Variable, int]` (`tool/algo/regalloc.py:31`).
    /// Returns `None` when the Variable has no coloring (Void /
    /// Unknown / different kind class).
    pub fn color_for_variable(&self, var: &crate::flowspace::model::Variable) -> Option<usize> {
        self.coloring.get(var).copied()
    }

    /// `true` iff `var` has a coloring in this kind class.
    pub fn contains_variable(&self, var: &crate::flowspace::model::Variable) -> bool {
        self.coloring.contains_key(var)
    }

    /// `tool/algo/regalloc.py:138-143 swapcolors(col1, col2)` — swap
    /// every Variable holding `col1` with `col2` and vice versa.
    /// Used by `flatten.py:88-100 enforce_input_args` to renumber
    /// the startblock inputargs into the dense `0..N` prefix of
    /// each kind's color range.
    pub fn swapcolors(&mut self, col1: usize, col2: usize) {
        for color in self.coloring.values_mut() {
            if *color == col1 {
                *color = col2;
            } else if *color == col2 {
                *color = col1;
            }
        }
    }
}

// `perform_register_allocation` reads kinds directly from
// `FunctionGraph::concretetype_of(&v)`, matching upstream
// `regalloc.py::perform_register_allocation(graph, kind)` where
// every Variable's kind comes from `getkind(v.concretetype)`.
// See [`perform_register_allocation`] below.

/// Stamp the canonical `exceptblock.inputargs` kinds onto the graph
/// when they are still `Unknown`.
///
/// Upstream `rpython/rtyper/rclass.py` assigns `(etype, evalue)`
/// concretetypes `Ptr(OBJECT_VTABLE)` / `Ptr(OBJECT)` so
/// `flatten.py:143 raise %r`, `flatten.py:220-231 last_exception/>i`
/// + `goto_if_exception_mismatch/i` see canonical kinds.  Pyre's
/// codewriter creates the canonical exceptblock eagerly in
/// `FunctionGraph::new` with `Unknown` placeholders; this helper
/// stamps the canonical Signed / GcRef kinds whenever the rtyper
/// hand-off (`apply_to_graph` / `apply_from_flowspace_variables`)
/// did not — equivalent to the previous `augment_value_kinds_*`
/// helper but written directly through to each backing
/// `Variable.concretetype` cell via
/// `FunctionGraph::set_concretetype_of_inline` instead of returning
/// a transitional HashMap.
pub(crate) fn augment_canonical_exceptblock_on_graph(graph: &mut FunctionGraph) {
    let except_args = &graph.block(graph.exceptblock).inputargs;
    if except_args.len() == 2 {
        if matches!(
            FunctionGraph::concretetype_of(&except_args[0]),
            ConcreteType::Unknown
        ) {
            FunctionGraph::set_concretetype_of_inline(&except_args[0], ConcreteType::Signed);
        }
        if matches!(
            FunctionGraph::concretetype_of(&except_args[1]),
            ConcreteType::Unknown
        ) {
            FunctionGraph::set_concretetype_of_inline(&except_args[1], ConcreteType::GcRef);
        }
    }
}

/// Perform register allocation for all three kinds — `&FunctionGraph`-only.
///
/// RPython parity: every `Variable.concretetype` is the source of
/// kind; pyre reads each per-value kind via
/// `FunctionGraph::concretetype_of(&v)`, projecting the
/// [`ConcreteType`] enum onto the JIT codewriter's
/// [`RegKind`] partitioning axis.  Canonical exceptblock inputargs
/// are stamped on the graph up-front via
/// [`augment_canonical_exceptblock_on_graph`].
pub(crate) fn perform_all_register_allocations(
    graph: &FunctionGraph,
) -> HashMap<RegKind, RegAllocator> {
    // Fail loud if the canonical exceptblock inputargs are still
    // `Unknown` — `variable_regkind` silently drops `Unknown` so a
    // missed call to [`augment_canonical_exceptblock_on_graph`] would
    // leave `last_exception` / `last_exc_value` un-coloured (no
    // register class), and any later flatten/assembler pass would
    // emit ops that reference uncolored values without any diagnostic.
    let except_args = &graph.block(graph.exceptblock).inputargs;
    if except_args.len() == 2 {
        assert!(
            !matches!(
                FunctionGraph::concretetype_of(&except_args[0]),
                ConcreteType::Unknown
            ) && !matches!(
                FunctionGraph::concretetype_of(&except_args[1]),
                ConcreteType::Unknown
            ),
            "perform_all_register_allocations: canonical exceptblock inputargs are still \
             Unknown — caller must run augment_canonical_exceptblock_on_graph() before \
             register allocation (graph: {})",
            graph.name,
        );
    }
    let mut result = HashMap::new();
    for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
        result.insert(kind, perform_register_allocation(graph, kind));
    }
    result
}

/// `regalloc.py::perform_register_allocation(graph, kind)` direct
/// port.  Runs on FunctionGraph (Block structure), BEFORE flatten.
/// Reads kind from `FunctionGraph::concretetype_of(&v)` exactly like
/// upstream reads `getkind(v.concretetype)`.
pub fn perform_register_allocation(graph: &FunctionGraph, kind: RegKind) -> RegAllocator {
    let consider =
        |var: &crate::flowspace::model::Variable| -> bool { variable_regkind(var) == Some(kind) };
    let mut allocator = RegAllocatorState::new();
    allocator.make_dependencies(graph, &consider);
    allocator.coalesce_variables(graph, &consider);
    allocator.find_node_coloring();

    let mut coloring: HashMap<crate::flowspace::model::Variable, usize> = HashMap::new();
    let mut max_reg = 0usize;
    // Walk every Variable minted on the graph and pick those whose
    // concretetype lands in `kind`.  `getcolor` projects through the
    // unionfind rep to recover the chordal coloring entry — matches
    // upstream `regalloc.py:118 self.coloring[self.unionfind.find_rep(v)]`.
    for var in graph.iter_variables() {
        if variable_regkind(&var) == Some(kind) {
            if let Some(color) = allocator.getcolor(&var) {
                coloring.insert(var.clone(), color);
                if color + 1 > max_reg {
                    max_reg = color + 1;
                }
            }
        }
    }
    RegAllocator {
        coloring,
        num_regs: max_reg,
    }
}

/// [`crate::flowspace::model::Variable`] → [`RegKind`] projection,
/// reading the Variable's inline `concretetype` cell directly.  Mirrors
/// upstream RPython's `getkind(v.concretetype)` (`history.py:46-71`).
fn variable_regkind(var: &crate::flowspace::model::Variable) -> Option<RegKind> {
    let ct = match var.concretetype.borrow().as_ref() {
        Some(lltype) => crate::model::getkind(lltype),
        None => ConcreteType::Unknown,
    };
    concretetype_to_regkind(&ct)
}

/// `getkind`'s [`ConcreteType`] → [`RegKind`] projection:
/// Signed → Int, GcRef → Ref, Float → Float.  Void / Unknown have
/// no register class (the same way RPython's regalloc skips Void
/// Variables, `flatten.py:325`).
fn concretetype_to_regkind(ty: &ConcreteType) -> Option<RegKind> {
    match ty {
        ConcreteType::Signed => Some(RegKind::Int),
        ConcreteType::GcRef => Some(RegKind::Ref),
        ConcreteType::Float => Some(RegKind::Float),
        ConcreteType::Void | ConcreteType::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ExitCase, ExitSwitch, FunctionGraph, Link, OpKind, ValueType};

    fn push_int_input(
        graph: &mut FunctionGraph,
        block: crate::model::BlockId,
        name: &str,
    ) -> crate::flowspace::model::Variable {
        let var = graph
            .push_op_var(
                block,
                OpKind::Input {
                    name: name.into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(block, var.clone());
        var
    }

    #[test]
    fn non_overlapping_lifetimes_share_register() {
        // v0 = Input; v1 = BinOp(v0, v0); Return v1
        // v0 dies when v1 is defined → no interference → can share register.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0_var = push_int_input(&mut graph, entry, "a");
        let v1_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0_var.clone(),
                    rhs: v0_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v1_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1_var, ConcreteType::Signed);
        let result = perform_register_allocation(&graph, RegKind::Int);
        // v0 and v1 don't overlap → can share
        assert_eq!(result.num_regs, 1);
    }

    #[test]
    fn overlapping_lifetimes_need_different_registers() {
        // v0 = Input; v1 = Input; v2 = BinOp(v0, v1); Return v2
        // v0 and v1 are both alive when v2 is defined → v0 and v1 interfere
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0_var = push_int_input(&mut graph, entry, "a");
        let v1_var = push_int_input(&mut graph, entry, "b");
        let v2_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0_var.clone(),
                    rhs: v1_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v2_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v2_var, ConcreteType::Signed);
        let result = perform_register_allocation(&graph, RegKind::Int);
        assert_ne!(
            result.color_for_variable(&v0_var),
            result.color_for_variable(&v1_var),
            "v0 and v1 are simultaneously alive → different registers"
        );
        // v2 can share with v0 or v1 (they die before v2's definition)
        assert!(result.num_regs >= 2);
    }

    #[test]
    fn goto_link_coalescing() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0_var = push_int_input(&mut graph, entry, "a");
        let (block1, block1_args) = graph.create_block_with_arg_vars(1);
        let v1_var = block1_args[0].clone();
        graph.set_goto(entry, block1, vec![v0_var.clone()]);
        graph.set_return(block1, Some(v1_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1_var, ConcreteType::Signed);
        let result = perform_register_allocation(&graph, RegKind::Int);
        assert_eq!(
            result.color_for_variable(&v0_var),
            result.color_for_variable(&v1_var),
        );
        assert_eq!(result.num_regs, 1);
    }

    #[test]
    fn fused_exitswitch_args_stay_live_until_branch() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let seed_var = push_int_input(&mut graph, entry, "seed");
        let x_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: seed_var.clone(),
                    rhs: seed_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let y_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "sub".into(),
                    lhs: seed_var.clone(),
                    rhs: seed_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let false_block = graph.create_block();
        let true_block = graph.create_block();
        graph.set_return(false_block, None);
        graph.set_return(true_block, None);
        let false_link =
            Link::from_variables(&graph, vec![], false_block, Some(ExitCase::Bool(false)));
        let true_link =
            Link::from_variables(&graph, vec![], true_block, Some(ExitCase::Bool(true)));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Fused {
                opname: "int_lt".into(),
                args: vec![x_var.clone(), y_var.clone()],
            }),
            vec![false_link, true_link],
        );

        FunctionGraph::set_concretetype_of_inline(&seed_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&x_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&y_var, ConcreteType::Signed);
        let result = perform_register_allocation(&graph, RegKind::Int);
        assert_ne!(
            result.color_for_variable(&x_var),
            result.color_for_variable(&y_var),
            "fused exitswitch operands are both read by GotoIfNotOp"
        );
    }

    #[test]
    fn coloring_unbounded() {
        let mut dg = DependencyGraph::<u16>::new();
        for i in 0..100u16 {
            dg.add_node(i);
        }
        for i in 0..99u16 {
            dg.add_edge(i, i + 1);
        }
        let coloring = dg.find_node_coloring();
        assert_eq!(coloring.len(), 100);
        let max_color = coloring.values().max().copied().unwrap_or(0);
        assert!(
            max_color <= 1,
            "chain needs at most 2 colors, got {}",
            max_color + 1
        );
    }
}
