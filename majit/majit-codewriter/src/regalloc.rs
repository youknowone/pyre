//! Register allocation for flattened JitCode.
//!
//! RPython equivalent: `rpython/jit/codewriter/regalloc.py` +
//! `rpython/tool/algo/regalloc.py` + `rpython/tool/algo/color.py`.
//!
//! Builds an interference graph from the flattened instruction sequence,
//! coalesces variables connected by Move ops, then finds a minimal
//! graph coloring to assign register indices.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::model::ValueId;
use crate::passes::flatten::{FlatOp, RegKind, SSARepr};

// ── DependencyGraph (RPython tool/algo/color.py) ──────────────────

/// Interference graph for register allocation.
///
/// RPython: `color.py::DependencyGraph`.
/// Two nodes are neighbours if they are simultaneously alive and thus
/// cannot share a register.
#[derive(Debug, Clone)]
struct DependencyGraph {
    /// RPython: `DependencyGraph._all_nodes`
    all_nodes: Vec<ValueId>,
    /// RPython: `DependencyGraph.neighbours`
    neighbours: HashMap<ValueId, HashSet<ValueId>>,
}

impl DependencyGraph {
    fn new() -> Self {
        Self {
            all_nodes: Vec::new(),
            neighbours: HashMap::new(),
        }
    }

    /// RPython: `DependencyGraph.add_node(v)`
    fn add_node(&mut self, v: ValueId) {
        if !self.neighbours.contains_key(&v) {
            self.all_nodes.push(v);
            self.neighbours.insert(v, HashSet::new());
        }
    }

    /// RPython: `DependencyGraph.add_edge(v1, v2)`
    fn add_edge(&mut self, v1: ValueId, v2: ValueId) {
        if v1 == v2 {
            return;
        }
        self.neighbours.entry(v1).or_default().insert(v2);
        self.neighbours.entry(v2).or_default().insert(v1);
    }

    /// RPython: `DependencyGraph.coalesce(vold, vnew)`
    /// Remove vold from the graph, attach all its edges to vnew.
    fn coalesce(&mut self, vold: ValueId, vnew: ValueId) {
        if let Some(old_neighbours) = self.neighbours.remove(&vold) {
            for n in old_neighbours {
                if let Some(ns) = self.neighbours.get_mut(&n) {
                    ns.remove(&vold);
                    if n != vnew {
                        ns.insert(vnew);
                        self.neighbours.entry(vnew).or_default().insert(n);
                    }
                }
            }
        }
    }

    /// RPython: `DependencyGraph.getnodes()`
    fn getnodes(&self) -> Vec<ValueId> {
        self.all_nodes
            .iter()
            .filter(|v| self.neighbours.contains_key(v))
            .copied()
            .collect()
    }

    /// RPython: `DependencyGraph.lexicographic_order()`
    /// Lexicographic breadth-first ordering for chordal graph coloring.
    fn lexicographic_order(&self) -> Vec<ValueId> {
        let nodes = self.getnodes();
        if nodes.is_empty() {
            return Vec::new();
        }
        let mut sigma: Vec<Vec<ValueId>> = vec![nodes.into_iter().rev().collect()];
        let mut result = Vec::new();

        while !sigma.is_empty() && !sigma[0].is_empty() {
            let v = sigma[0].pop().unwrap();
            result.push(v);
            let neighb = self.neighbours.get(&v).cloned().unwrap_or_default();
            let mut new_sigma = Vec::new();
            for s in sigma {
                let mut s1 = Vec::new();
                let mut s2 = Vec::new();
                for x in s {
                    if neighb.contains(&x) {
                        s1.push(x);
                    } else {
                        s2.push(x);
                    }
                }
                if !s1.is_empty() {
                    new_sigma.push(s1);
                }
                if !s2.is_empty() {
                    new_sigma.push(s2);
                }
            }
            sigma = new_sigma;
        }
        result
    }

    /// RPython: `DependencyGraph.find_node_coloring()`
    /// Greedy coloring using lexicographic BFS order.
    fn find_node_coloring(&self) -> HashMap<ValueId, usize> {
        let mut result = HashMap::new();
        for v in self.lexicographic_order() {
            let mut forbidden: u64 = 0;
            if let Some(neighbours) = self.neighbours.get(&v) {
                for &n in neighbours {
                    if let Some(&color) = result.get(&n) {
                        if color < 64 {
                            forbidden |= 1u64 << color;
                        }
                    }
                }
            }
            // Find lowest 0 bit
            let mut num = 0;
            while forbidden & (1u64 << num) != 0 {
                num += 1;
            }
            result.insert(v, num);
        }
        result
    }
}

// ── UnionFind (RPython tool/algo/unionfind.py) ────────────────────

/// Union-Find data structure for variable coalescing.
///
/// RPython: `unionfind.py::UnionFind`.
#[derive(Debug, Clone)]
struct UnionFind {
    parent: HashMap<ValueId, ValueId>,
    weight: HashMap<ValueId, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            weight: HashMap::new(),
        }
    }

    /// RPython: `UnionFind.find_rep(obj)`
    fn find_rep(&mut self, v: ValueId) -> ValueId {
        if !self.parent.contains_key(&v) {
            self.parent.insert(v, v);
            self.weight.insert(v, 1);
            return v;
        }
        // Path compression
        let mut root = v;
        while self.parent[&root] != root {
            root = self.parent[&root];
        }
        // Compress path
        let mut current = v;
        while current != root {
            let next = self.parent[&current];
            self.parent.insert(current, root);
            current = next;
        }
        root
    }

    /// RPython: `UnionFind.union(obj1, obj2)` → returns representative
    fn union(&mut self, v1: ValueId, v2: ValueId) -> ValueId {
        let rep1 = self.find_rep(v1);
        let rep2 = self.find_rep(v2);
        if rep1 == rep2 {
            return rep1;
        }
        let w1 = self.weight.get(&rep1).copied().unwrap_or(1);
        let w2 = self.weight.get(&rep2).copied().unwrap_or(1);
        let (winner, loser) = if w1 >= w2 { (rep1, rep2) } else { (rep2, rep1) };
        self.parent.insert(loser, winner);
        self.weight.remove(&loser);
        *self.weight.entry(winner).or_insert(0) = w1 + w2;
        winner
    }
}

// ── RegAllocator (RPython tool/algo/regalloc.py) ──────────────────

/// Register allocator for a single kind (int/ref/float).
///
/// RPython: `regalloc.py::RegAllocator`.
///
/// 1. `make_dependencies()` — build interference graph from liveness
/// 2. `coalesce_variables()` — merge Move src/dst when non-interfering
/// 3. `find_node_coloring()` — greedy graph coloring
#[derive(Debug, Clone)]
struct RegAllocator {
    depgraph: DependencyGraph,
    unionfind: UnionFind,
    coloring: HashMap<ValueId, usize>,
}

impl RegAllocator {
    fn new() -> Self {
        Self {
            depgraph: DependencyGraph::new(),
            unionfind: UnionFind::new(),
            coloring: HashMap::new(),
        }
    }

    /// RPython: `RegAllocator.make_dependencies()`
    ///
    /// Build the interference graph: two values interfere if one is
    /// defined while the other is alive.
    fn make_dependencies(
        &mut self,
        ops: &[FlatOp],
        target_kind: RegKind,
        value_kinds: &HashMap<ValueId, RegKind>,
    ) {
        let consider = |v: &ValueId| -> bool { value_kinds.get(v).copied() == Some(target_kind) };

        // Compute liveness intervals and build interference edges.
        // Walk forward: track alive set, add edges when a new value is defined.
        let mut alive: HashSet<ValueId> = HashSet::new();

        for op in ops {
            match op {
                FlatOp::Op(inner) => {
                    // Operands are used — add to alive
                    for v in crate::inline::op_value_refs(&inner.kind) {
                        if consider(&v) {
                            self.depgraph.add_node(v);
                            alive.insert(v);
                        }
                    }
                    // Result is defined — interferes with all alive, then add to alive
                    if let Some(result) = inner.result {
                        if consider(&result) {
                            self.depgraph.add_node(result);
                            for &v in &alive {
                                self.depgraph.add_edge(v, result);
                            }
                            alive.insert(result);
                        }
                    }
                }
                FlatOp::Move { dst, src } => {
                    if consider(src) {
                        self.depgraph.add_node(*src);
                        alive.insert(*src);
                    }
                    if consider(dst) {
                        self.depgraph.add_node(*dst);
                        alive.insert(*dst);
                    }
                }
                FlatOp::JumpIfTrue { cond, .. } | FlatOp::JumpIfFalse { cond, .. } => {
                    if consider(cond) {
                        alive.insert(*cond);
                    }
                }
                FlatOp::Label(_) => {
                    // Block boundary — reset alive set (simplified).
                    // Full implementation would track across blocks via labels.
                    alive.clear();
                }
                FlatOp::Jump(_) | FlatOp::Live { .. } | FlatOp::Unreachable => {}
            }
        }
    }

    /// RPython: `RegAllocator.coalesce_variables()`
    ///
    /// For each Move(dst, src), try to coalesce src and dst if they
    /// don't interfere. This reduces unnecessary copies.
    fn coalesce_variables(
        &mut self,
        ops: &[FlatOp],
        value_kinds: &HashMap<ValueId, RegKind>,
        target_kind: RegKind,
    ) {
        let consider = |v: &ValueId| -> bool { value_kinds.get(v).copied() == Some(target_kind) };

        for op in ops {
            if let FlatOp::Move { dst, src } = op {
                if consider(src) && consider(dst) {
                    self.try_coalesce(*src, *dst);
                }
            }
        }
    }

    /// RPython: `RegAllocator._try_coalesce(v, w)`
    fn try_coalesce(&mut self, v: ValueId, w: ValueId) {
        let v0 = self.unionfind.find_rep(v);
        let w0 = self.unionfind.find_rep(w);
        if v0 == w0 {
            return;
        }
        // Check if they interfere
        if self
            .depgraph
            .neighbours
            .get(&w0)
            .map_or(false, |ns| ns.contains(&v0))
        {
            return; // can't coalesce — they interfere
        }
        let rep = self.unionfind.union(v0, w0);
        if rep == v0 {
            self.depgraph.coalesce(w0, v0);
        } else {
            self.depgraph.coalesce(v0, w0);
        }
    }

    /// RPython: `RegAllocator.find_node_coloring()`
    fn find_node_coloring(&mut self) {
        self.coloring = self.depgraph.find_node_coloring();
    }

    /// RPython: `RegAllocator.getcolor(v)`
    fn getcolor(&mut self, v: ValueId) -> Option<usize> {
        let rep = self.unionfind.find_rep(v);
        self.coloring.get(&rep).copied()
    }
}

// ── Public API ────────────────────────────────────────────────────

/// Result of register allocation for one kind.
///
/// RPython: the `RegAllocator` object with `_coloring` dict.
#[derive(Debug, Clone)]
pub struct RegAllocResult {
    /// RPython: `RegAlloc._coloring` — maps ValueId → register index.
    pub coloring: HashMap<ValueId, usize>,
    /// Number of registers used for this kind.
    pub num_regs: usize,
}

/// Perform register allocation for a single kind on a flattened function.
///
/// RPython: `regalloc.py::perform_register_allocation(graph, kind)`.
///
/// 1. Build interference graph from instruction liveness
/// 2. Coalesce Move src/dst pairs
/// 3. Find minimal graph coloring
pub fn perform_register_allocation(flattened: &SSARepr, kind: RegKind) -> RegAllocResult {
    let mut allocator = RegAllocator::new();
    allocator.make_dependencies(&flattened.ops, kind, &flattened.value_kinds);
    allocator.coalesce_variables(&flattened.ops, &flattened.value_kinds, kind);
    allocator.find_node_coloring();

    let mut coloring = HashMap::new();
    let mut max_reg = 0usize;
    for (&vid, &vkind) in &flattened.value_kinds {
        if vkind == kind {
            if let Some(color) = allocator.getcolor(vid) {
                coloring.insert(vid, color);
                if color + 1 > max_reg {
                    max_reg = color + 1;
                }
            }
        }
    }

    RegAllocResult {
        coloring,
        num_regs: max_reg,
    }
}

/// Perform register allocation for all three kinds.
///
/// RPython codewriter.py:45-47:
/// ```python
/// for kind in KINDS:
///     regallocs[kind] = perform_register_allocation(graph, kind)
/// ```
pub fn perform_all_register_allocations(flattened: &SSARepr) -> HashMap<RegKind, RegAllocResult> {
    let mut result = HashMap::new();
    for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
        result.insert(kind, perform_register_allocation(flattened, kind));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{OpKind, SpaceOperation, ValueType};

    #[test]
    fn allocate_separates_by_kind() {
        // Three values defined in ops so make_dependencies sees them
        let flat = SSARepr {
            name: "test".into(),
            ops: vec![
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(0)),
                    kind: OpKind::Input {
                        name: "a".into(),
                        ty: ValueType::Int,
                    },
                }),
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(1)),
                    kind: OpKind::Input {
                        name: "b".into(),
                        ty: ValueType::Ref,
                    },
                }),
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(2)),
                    kind: OpKind::Input {
                        name: "c".into(),
                        ty: ValueType::Int,
                    },
                }),
            ],
            num_values: 3,
            num_blocks: 1,
            value_kinds: {
                let mut m = HashMap::new();
                m.insert(ValueId(0), RegKind::Int);
                m.insert(ValueId(1), RegKind::Ref);
                m.insert(ValueId(2), RegKind::Int);
                m
            },
        };

        let allocs = perform_all_register_allocations(&flat);
        // Int kind has 2 values, Ref has 1, Float has 0.
        // But v0 and v2 don't interfere (no overlap), so they may coalesce to 1 reg.
        assert!(allocs[&RegKind::Int].num_regs >= 1);
        assert_eq!(allocs[&RegKind::Ref].num_regs, 1);
        assert_eq!(allocs[&RegKind::Float].num_regs, 0);
    }

    #[test]
    fn coalesce_reduces_registers() {
        // v0 = Input
        // v1 = Move(v0) — should coalesce with v0
        let mut flat = SSARepr {
            name: "test".into(),
            ops: vec![
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(0)),
                    kind: OpKind::Input {
                        name: "a".into(),
                        ty: ValueType::Int,
                    },
                }),
                FlatOp::Move {
                    dst: ValueId(1),
                    src: ValueId(0),
                },
            ],
            num_values: 2,
            num_blocks: 1,
            value_kinds: {
                let mut m = HashMap::new();
                m.insert(ValueId(0), RegKind::Int);
                m.insert(ValueId(1), RegKind::Int);
                m
            },
        };

        let result = perform_register_allocation(&flat, RegKind::Int);
        // v0 and v1 should coalesce to the same register
        assert_eq!(
            result.coloring.get(&ValueId(0)),
            result.coloring.get(&ValueId(1)),
            "coalesced values should share a register"
        );
        assert_eq!(
            result.num_regs, 1,
            "should need only 1 register after coalescing"
        );
    }

    #[test]
    fn interfering_values_get_different_registers() {
        // v0 = Input
        // v1 = BinOp(v0, v0) — v1 is defined while v0 is alive → they interfere
        let flat = SSARepr {
            name: "test".into(),
            ops: vec![
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(0)),
                    kind: OpKind::Input {
                        name: "a".into(),
                        ty: ValueType::Int,
                    },
                }),
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(1)),
                    kind: OpKind::BinOp {
                        op: "add".into(),
                        lhs: ValueId(0),
                        rhs: ValueId(0),
                        result_ty: ValueType::Int,
                    },
                }),
            ],
            num_values: 2,
            num_blocks: 1,
            value_kinds: {
                let mut m = HashMap::new();
                m.insert(ValueId(0), RegKind::Int);
                m.insert(ValueId(1), RegKind::Int);
                m
            },
        };

        let result = perform_register_allocation(&flat, RegKind::Int);
        assert_ne!(
            result.coloring.get(&ValueId(0)),
            result.coloring.get(&ValueId(1)),
            "interfering values should get different registers"
        );
        assert_eq!(result.num_regs, 2);
    }

    #[test]
    fn dependency_graph_coloring() {
        let mut dg = DependencyGraph::new();
        dg.add_node(ValueId(0));
        dg.add_node(ValueId(1));
        dg.add_node(ValueId(2));
        dg.add_edge(ValueId(0), ValueId(1));
        // 0-1 interfere, 2 is independent
        let coloring = dg.find_node_coloring();
        assert_ne!(coloring[&ValueId(0)], coloring[&ValueId(1)]);
        // 2 can share a color with either 0 or 1
        assert!(coloring[&ValueId(2)] <= 1);
    }
}
