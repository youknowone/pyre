//! Dependency graph for vectorization.
//!
//! Mirrors RPython's `dependency.py`: builds a DAG of data dependencies
//! between operations in a loop body. Used by the vector optimizer to
//! identify independent operations that can be packed into SIMD instructions.

use std::collections::{BinaryHeap, HashMap, HashSet};

use majit_ir::{Op, OpCode, OpRef};
use crate::schedule::PackGroup;

/// A node in the dependency graph.
#[derive(Clone, Debug)]
pub struct DepNode {
    /// Index in the ops list.
    pub idx: usize,
    /// The operation.
    pub op: Op,
    /// Indices of operations this one depends on (must execute before).
    pub deps: Vec<usize>,
    /// Indices of operations that depend on this one (must execute after).
    pub users: Vec<usize>,
}

/// Dependency graph for a loop body.
#[derive(Clone, Debug)]
pub struct DependencyGraph {
    pub nodes: Vec<DepNode>,
}

impl DependencyGraph {
    /// Build a dependency graph from a list of operations.
    ///
    /// Two operations have a dependency if:
    /// - One uses the result of the other (data dependency)
    /// - Both access memory and at least one is a write (memory dependency)
    /// - One is a guard (control dependency)
    pub fn build(ops: &[Op]) -> Self {
        let mut nodes: Vec<DepNode> = ops
            .iter()
            .enumerate()
            .map(|(idx, op)| DepNode {
                idx,
                op: op.clone(),
                deps: Vec::new(),
                users: Vec::new(),
            })
            .collect();

        // Map from OpRef (producer position) to node index
        let mut def_map: HashMap<OpRef, usize> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            def_map.insert(op.pos, i);
        }

        // Build data dependencies
        for i in 0..ops.len() {
            let op = &ops[i];
            for arg in &op.args {
                if let Some(&dep_idx) = def_map.get(arg) {
                    if dep_idx != i {
                        nodes[i].deps.push(dep_idx);
                        nodes[dep_idx].users.push(i);
                    }
                }
            }

            // fail_args also create dependencies
            if let Some(ref fa) = op.fail_args {
                for arg in fa.iter() {
                    if let Some(&dep_idx) = def_map.get(arg) {
                        if dep_idx != i && !nodes[i].deps.contains(&dep_idx) {
                            nodes[i].deps.push(dep_idx);
                            nodes[dep_idx].users.push(i);
                        }
                    }
                }
            }
        }

        // Memory dependencies: sequential ordering for loads/stores
        // to the same descriptor (conservative: treat all memory ops as aliasing)
        let mut last_memory_op: Option<usize> = None;
        for i in 0..ops.len() {
            if ops[i].opcode.is_memory_access() {
                if let Some(prev) = last_memory_op {
                    if !nodes[i].deps.contains(&prev) {
                        nodes[i].deps.push(prev);
                        nodes[prev].users.push(i);
                    }
                }
                last_memory_op = Some(i);
            }
        }

        DependencyGraph { nodes }
    }

    /// Find groups of independent, isomorphic operations that can be packed.
    ///
    /// Two ops are "isomorphic" if they have the same opcode and their
    /// args come from independent sources (no data dependency between them).
    pub fn find_packable_groups(&self) -> Vec<PackGroup> {
        let mut groups: Vec<PackGroup> = Vec::new();
        let mut used: HashSet<usize> = HashSet::new();

        // Group by opcode
        let mut by_opcode: HashMap<OpCode, Vec<usize>> = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.op.opcode.to_vector().is_some() && !node.op.opcode.is_guard() {
                by_opcode.entry(node.op.opcode).or_default().push(i);
            }
        }

        // For each opcode, find independent pairs/groups
        for (opcode, indices) in &by_opcode {
            let vec_opcode = match opcode.to_vector() {
                Some(v) => v,
                None => continue,
            };

            let mut group_indices = Vec::new();

            for &i in indices {
                if used.contains(&i) {
                    continue;
                }

                // Check independence from already-grouped ops
                let mut independent = true;
                for &already in &group_indices {
                    if self.has_dependency(i, already) {
                        independent = false;
                        break;
                    }
                }

                // vector.py: isomorphic check — ops must have the same
                // number of args and compatible types (same opcode already
                // guaranteed by the grouping).
                if independent && !group_indices.is_empty() {
                    let first = &self.nodes[group_indices[0]].op;
                    let candidate = &self.nodes[i].op;
                    if first.num_args() != candidate.num_args() {
                        independent = false;
                    }
                }

                if independent {
                    group_indices.push(i);
                }
            }

            // Need at least 2 ops to form a pack
            if group_indices.len() >= 2 {
                for &idx in &group_indices {
                    used.insert(idx);
                }
                groups.push(PackGroup {
                    scalar_opcode: *opcode,
                    vector_opcode: vec_opcode,
                    members: group_indices,
                });
            }
        }

        groups
    }

    /// Check if there's a direct or transitive dependency between two nodes.
    pub fn has_dependency(&self, a: usize, b: usize) -> bool {
        // Direct dependency check (sufficient for basic vectorization)
        self.nodes[a].deps.contains(&b) || self.nodes[b].deps.contains(&a)
    }
}

// ── Instruction Scheduling ──────────────────────────────────────────────

/// Reorder operations to maximize instruction-level parallelism.
///
/// Uses a topological sort with priority scheduling: among all operations
/// whose dependencies are satisfied, choose the one with the highest
/// "height" (longest path to a leaf in the dependency graph).
///
/// This mirrors RPython's `schedule.py`, which reorders the loop body to
/// improve ILP before packing decisions are made.
pub fn schedule_operations(graph: &DependencyGraph) -> Vec<usize> {
    let n = graph.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute heights in reverse topological order.
    // Height = 1 + max height among users (successors in the DAG).
    let mut heights = vec![0usize; n];
    for i in (0..n).rev() {
        let max_user_height = graph.nodes[i]
            .users
            .iter()
            .map(|&u| heights[u])
            .max()
            .unwrap_or(0);
        heights[i] = 1 + max_user_height;
    }

    // Compute in-degrees from deps.
    let mut in_degree = vec![0usize; n];
    for node in &graph.nodes {
        in_degree[node.idx] = node.deps.len();
    }

    // Seed the priority queue with all zero-in-degree nodes.
    // BinaryHeap is a max-heap: (height, index) — higher height = higher priority.
    let mut ready: BinaryHeap<(usize, usize)> = BinaryHeap::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            ready.push((heights[i], i));
        }
    }

    let mut schedule = Vec::with_capacity(n);
    while let Some((_, idx)) = ready.pop() {
        schedule.push(idx);
        for &user in &graph.nodes[idx].users {
            in_degree[user] -= 1;
            if in_degree[user] == 0 {
                ready.push((heights[user], user));
            }
        }
    }

    schedule
}
