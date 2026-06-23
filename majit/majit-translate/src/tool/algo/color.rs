//! Re-export of `rpython/tool/algo/color.py`.
//!
//! The concrete `DependencyGraph` implementation is shared with
//! `jit_codewriter::regalloc`, where PyPy's register allocator consumes it.

pub use crate::jit_codewriter::regalloc::DependencyGraph;

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
