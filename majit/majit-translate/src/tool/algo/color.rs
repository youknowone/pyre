//! Chordal graph coloring helper from `rpython/tool/algo/color.py`.

use std::collections::{HashMap, HashSet};

/// Interference graph for register allocation.
///
/// RPython: `color.py::DependencyGraph`.
///
/// Generic over the node identity type `N` so both
/// majit-translate (which keys nodes on
/// [`crate::flowspace::model::Variable`] — upstream-orthodox
/// `tool/algo/regalloc.py:31 coloring: dict[Variable, int]`) and
/// the pyre CPython-bytecode codewriter
/// (`pyre/pyre-jit/src/jit/regalloc.rs`, which keys on
/// `pyre-jit-trace::flow::VariableId` for its detached-index IR)
/// can share the chordal coloring engine. Per-kind callers run the
/// coloring independently per kind (see `regalloc.py:8`).
///
/// Node identity must be `Eq + Hash + Clone`; the chordal walk
/// requires deterministic identity ordering from insertion order,
/// matching upstream's `_all_nodes` list.
#[derive(Debug, Clone)]
pub struct DependencyGraph<N: Eq + std::hash::Hash + Clone> {
    _all_nodes: Vec<N>,
    pub neighbours: HashMap<N, HashSet<N>>,
}

impl<N: Eq + std::hash::Hash + Clone> DependencyGraph<N> {
    pub fn new() -> Self {
        Self {
            _all_nodes: Vec::new(),
            neighbours: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, v: N) {
        if !self.neighbours.contains_key(&v) {
            self._all_nodes.push(v.clone());
            self.neighbours.insert(v, HashSet::new());
        }
    }

    pub fn add_edge(&mut self, v1: N, v2: N) {
        assert!(v1 != v2);
        // `neighbours[v1].add(v2)` — both endpoints must already be `add_node`d
        // (absent key panics, mirroring upstream's KeyError); `add_edge` does
        // not register nodes itself.
        self.neighbours.get_mut(&v1).unwrap().insert(v2.clone());
        self.neighbours.get_mut(&v2).unwrap().insert(v1);
    }

    /// RPython: `color.py::DependencyGraph.coalesce(vold, vnew)`.
    /// Folds `vold`'s adjacency list into `vnew` and removes `vold`.
    /// Used by `RegAllocator.coalesce_variables` after a successful
    /// union so the chordal coloring sees a single combined node.
    pub fn coalesce(&mut self, vold: N, vnew: N) {
        let old_neighbours = self
            .neighbours
            .remove(&vold)
            .expect("DependencyGraph.coalesce: old node must exist");
        for n in old_neighbours {
            self.neighbours
                .get_mut(&n)
                .expect("DependencyGraph.coalesce: neighbour node must exist")
                .remove(&vold);
            assert!(vnew != n);
            self.neighbours
                .get_mut(&n)
                .expect("DependencyGraph.coalesce: neighbour node must exist")
                .insert(vnew.clone());
            self.neighbours
                .get_mut(&vnew)
                .expect("DependencyGraph.coalesce: new node must exist")
                .insert(n);
        }
    }

    /// RPython: `regalloc.py:105` `v0 not in dg.neighbours[w0]`.
    /// Returns true iff there is an interference edge between `v1` and `v2`.
    pub fn has_edge(&self, v1: &N, v2: &N) -> bool {
        self.neighbours.get(v1).map_or(false, |ns| ns.contains(v2))
    }

    pub fn getnodes(&self) -> Vec<N> {
        self._all_nodes
            .iter()
            .filter(|v| self.neighbours.contains_key(*v))
            .cloned()
            .collect()
    }

    /// RPython: `DependencyGraph.lexicographic_order()`.
    ///
    /// O(n²): each of the n iterations re-partitions every remaining node
    /// across `sigma`, independent of edge density. This is the codewriter-side
    /// super-linear compile cost identified as #203 gap-6 facet A. The O(V+E)
    /// lex-BFS (partition-refinement) rewrite that would remove it is
    /// deliberately declined to keep this a line-by-line port of `color.py`,
    /// which is identically O(n²); the quadratic term only manifests on
    /// pathologically large single-function graphs, not the small per-function
    /// graphs real code produces.
    pub fn lexicographic_order(&self) -> Vec<N> {
        let nodes = self.getnodes();
        if nodes.is_empty() {
            return Vec::new();
        }
        let mut sigma: Vec<Vec<N>> = vec![nodes.into_iter().rev().collect()];
        let mut result = Vec::new();
        while !sigma.is_empty() && !sigma[0].is_empty() {
            let v = sigma[0].pop().unwrap();
            let neighb = self.neighbours.get(&v).cloned().unwrap_or_default();
            result.push(v);
            let mut new_sigma = Vec::new();
            for s in sigma {
                let (s1, s2): (Vec<_>, Vec<_>) = s.into_iter().partition(|x| neighb.contains(x));
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

    /// RPython: `DependencyGraph.size_of_largest_clique()`.
    ///
    /// Assumes the graph is chordal, as upstream does.
    pub fn size_of_largest_clique(&self) -> usize {
        let mut result = 0;
        let mut seen = HashSet::new();
        for v in self.lexicographic_order() {
            let mut num = 1;
            if let Some(neighbours) = self.neighbours.get(&v) {
                for n in neighbours {
                    if seen.contains(n) {
                        num += 1;
                    }
                }
            }
            result = result.max(num);
            seen.insert(v);
        }
        result
    }

    /// RPython: `DependencyGraph.find_node_coloring()`.
    /// Uses `HashSet<usize>` — no color limit (fixes u64 overflow).
    pub fn find_node_coloring(&self) -> HashMap<N, usize> {
        let mut result = HashMap::new();
        for v in self.lexicographic_order() {
            let mut forbidden: HashSet<usize> = HashSet::new();
            if let Some(neighbours) = self.neighbours.get(&v) {
                for n in neighbours {
                    if let Some(&color) = result.get(n) {
                        forbidden.insert(color);
                    }
                }
            }
            let mut num = 0;
            while forbidden.contains(&num) {
                num += 1;
            }
            result.insert(v, num);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph1() -> DependencyGraph<char> {
        let mut dg = DependencyGraph::new();
        for node in ['a', 'b', 'c', 'd', 'e'] {
            dg.add_node(node);
        }
        dg.add_edge('a', 'b');
        dg.add_edge('a', 'd');
        dg.add_edge('d', 'b');
        dg.add_edge('d', 'e');
        dg.add_edge('b', 'c');
        dg.add_edge('b', 'e');
        dg.add_edge('e', 'c');
        dg
    }

    #[test]
    fn lexicographic_order_matches_rpython_tests() {
        let dg = graph1();
        let order: String = dg.lexicographic_order().into_iter().collect();
        assert_eq!(order, "abdec");
    }

    #[test]
    fn lexicographic_order_empty_matches_rpython_tests() {
        let dg = DependencyGraph::<char>::new();
        assert_eq!(dg.lexicographic_order(), Vec::<char>::new());
    }

    #[test]
    #[should_panic]
    fn add_edge_requires_preregistered_nodes() {
        // `neighbours[v1].add(v2)` raises KeyError when a node was not
        // `add_node`d first; the caller must register nodes before edges.
        let mut dg = DependencyGraph::new();
        dg.add_edge('a', 'b');
    }

    #[test]
    fn size_of_largest_clique_matches_rpython_tests() {
        let dg = graph1();
        assert_eq!(dg.size_of_largest_clique(), 3);
    }

    #[test]
    fn find_node_coloring_matches_rpython_tests() {
        let dg = graph1();
        let coloring = dg.find_node_coloring();
        assert_eq!(coloring.len(), 5);
        let mut keys: Vec<_> = coloring.keys().copied().collect();
        keys.sort();
        assert_eq!(keys, vec!['a', 'b', 'c', 'd', 'e']);
        let mut values: Vec<_> = coloring.values().copied().collect();
        values.sort();
        values.dedup();
        assert_eq!(values, vec![0, 1, 2]);
        for (v1, v2s) in &dg.neighbours {
            for v2 in v2s {
                assert_ne!(coloring[v1], coloring[v2]);
            }
        }
    }

    #[test]
    fn find_node_coloring_empty_matches_rpython_tests() {
        let dg = DependencyGraph::<char>::new();
        assert!(dg.find_node_coloring().is_empty());
    }
}
