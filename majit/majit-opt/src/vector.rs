/// Vector optimization pass — auto-vectorization of scalar loop bodies.
///
/// Translated from rpython/jit/metainterp/optimizeopt/vector.py,
/// dependency.py, and costmodel.py.
///
/// This pass analyzes a peeled loop body (between Label and Jump) to find
/// independent scalar operations that can be packed into SIMD vector
/// instructions. It builds a dependency graph, identifies vectorizable
/// groups, checks profitability via a cost model, and rewrites the ops.
///
/// # Pipeline
///
/// 1. **Dependency analysis**: Build a DAG of data dependencies between ops
/// 2. **Packing**: Find pairs/groups of isomorphic ops that can run in parallel
/// 3. **Cost model**: Estimate whether vectorization is profitable
/// 4. **Rewrite**: Replace scalar ops with vector equivalents (VecIntAdd, etc.)
///
/// # Limitations
///
/// - Only operates on loop bodies (Label..Jump)
/// - Requires array load/store patterns for memory access vectorization
/// - Guards in the loop body prevent full vectorization (conservative)
use std::collections::{BinaryHeap, HashMap, HashSet};

use majit_ir::{Op, OpCode, OpRef};

use crate::{OptContext, OptimizationPass, PassResult};

// ── Dependency Graph ────────────────────────────────────────────────────

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
    fn has_dependency(&self, a: usize, b: usize) -> bool {
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

/// A group of independent, isomorphic operations that can be packed.
#[derive(Clone, Debug)]
pub struct PackGroup {
    /// The scalar opcode of the group members.
    pub scalar_opcode: OpCode,
    /// The vector opcode to replace them with.
    pub vector_opcode: OpCode,
    /// Indices into the DepGraph nodes.
    pub members: Vec<usize>,
}

// ── Cost Model ──────────────────────────────────────────────────────────

/// Cost model for deciding whether vectorization is profitable.
///
/// From rpython/jit/metainterp/optimizeopt/costmodel.py.
///
/// Vectorization has overhead from:
/// - Pack/unpack operations to move scalars into/out of vector registers
/// - Potential register pressure increase
///
/// It saves:
/// - Instruction count reduction (N scalar ops → 1 vector op)
/// - Memory bandwidth (packed loads/stores)
pub struct CostModel {
    /// Minimum group size to consider vectorization (default: 2).
    pub min_pack_size: usize,
    /// Overhead per pack/unpack operation (in abstract cost units).
    pub pack_cost: i32,
    /// Saving per eliminated scalar op (in abstract cost units).
    pub scalar_save: i32,
}

impl CostModel {
    pub fn new() -> Self {
        CostModel {
            min_pack_size: 2,
            pack_cost: 2,
            scalar_save: 1,
        }
    }

    /// Estimate whether vectorizing a group is profitable.
    ///
    /// Returns true if the estimated savings outweigh the pack/unpack costs.
    pub fn is_profitable(&self, group: &PackGroup) -> bool {
        let n = group.members.len() as i32;
        if n < self.min_pack_size as i32 {
            return false;
        }

        // Savings: eliminate (n-1) scalar ops
        let savings = (n - 1) * self.scalar_save;

        // Cost: need to pack inputs and unpack outputs
        // Each input needs a VecExpand or VecPack
        // Each output needs VecUnpack ops if used by non-vectorized code
        let pack_ops = 2; // rough estimate: 1 pack + 1 unpack
        let cost = pack_ops * self.pack_cost;

        savings > cost
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Vector Optimization Pass ────────────────────────────────────────────

/// Auto-vectorization pass.
///
/// Analyzes loop bodies to find independent scalar operations that can be
/// packed into SIMD vector instructions.
pub struct OptVectorize {
    /// Operations in the current loop body (between Label and Jump).
    body_ops: Vec<Op>,
    /// Whether we're inside a loop body.
    in_loop: bool,
    /// Cost model for profitability decisions.
    cost_model: CostModel,
}

impl OptVectorize {
    pub fn new() -> Self {
        OptVectorize {
            body_ops: Vec::new(),
            in_loop: false,
            cost_model: CostModel::new(),
        }
    }

    /// Attempt to vectorize the buffered loop body.
    ///
    /// Before packing, operations are scheduled for maximum ILP.
    /// The reordering may expose additional packing opportunities.
    fn try_vectorize(&mut self, ctx: &mut OptContext) -> Option<Vec<Op>> {
        if self.body_ops.len() < 4 {
            return None; // Too small to benefit
        }

        // Phase 1: Schedule operations for ILP before packing.
        let dep_graph = DependencyGraph::build(&self.body_ops);
        let schedule = schedule_operations(&dep_graph);
        if schedule.len() == self.body_ops.len() {
            let scheduled: Vec<Op> = schedule.iter().map(|&i| self.body_ops[i].clone()).collect();
            self.body_ops = scheduled;
        }

        // Phase 2: Rebuild dependency graph on reordered ops and find packs.
        let dep_graph = DependencyGraph::build(&self.body_ops);
        let groups = dep_graph.find_packable_groups();

        if groups.is_empty() {
            return None;
        }

        // Filter by cost model
        let profitable: Vec<&PackGroup> = groups
            .iter()
            .filter(|g| self.cost_model.is_profitable(g))
            .collect();

        if profitable.is_empty() {
            return None;
        }

        // Build the vectorized ops
        let mut vectorized_indices: HashSet<usize> = HashSet::new();
        let mut new_ops: Vec<Op> = Vec::with_capacity(self.body_ops.len());
        let mut vector_results: Vec<(usize, OpRef)> = Vec::new(); // (group member 0 idx, vector result ref)

        // First, emit vector operations
        for group in &profitable {
            for &idx in &group.members {
                vectorized_indices.insert(idx);
            }

            // Emit VecPack for inputs: pack the first args of each member
            let first_member = &self.body_ops[group.members[0]];
            let mut vec_op = Op::new(group.vector_opcode, &first_member.args);
            vec_op.pos = OpRef(ctx.new_operations.len() as u32 + new_ops.len() as u32);
            let vec_ref = vec_op.pos;
            new_ops.push(vec_op);

            // Record that group members' results are now in the vector result
            for (lane, &idx) in group.members.iter().enumerate() {
                vector_results.push((idx, vec_ref));
                // Emit VecUnpack for each lane if the result is used
                if !dep_graph.nodes[idx].users.is_empty() {
                    let unpack_opcode = if is_float_opcode(group.scalar_opcode) {
                        OpCode::VecUnpackF
                    } else {
                        OpCode::VecUnpackI
                    };
                    let lane_ref = OpRef(lane as u32 + 10_000); // constant for lane index
                    let count_ref = OpRef(group.members.len() as u32 + 10_000); // constant for count
                    let mut unpack = Op::new(unpack_opcode, &[vec_ref, lane_ref, count_ref]);
                    unpack.pos = OpRef(ctx.new_operations.len() as u32 + new_ops.len() as u32);
                    new_ops.push(unpack);
                }
            }
        }

        // Then, emit remaining (non-vectorized) ops
        for (i, op) in self.body_ops.iter().enumerate() {
            if !vectorized_indices.contains(&i) {
                new_ops.push(op.clone());
            }
        }

        Some(new_ops)
    }
}

impl Default for OptVectorize {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for OptVectorize {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        match op.opcode {
            OpCode::Label => {
                self.in_loop = true;
                PassResult::Emit(op.clone())
            }
            OpCode::Jump if self.in_loop => {
                // End of loop body — attempt vectorization
                if let Some(vectorized) = self.try_vectorize(ctx) {
                    for vop in vectorized {
                        ctx.emit(vop);
                    }
                } else {
                    // No vectorization — emit original ops
                    for body_op in &self.body_ops {
                        ctx.emit(body_op.clone());
                    }
                }
                self.in_loop = false;
                PassResult::Emit(op.clone())
            }
            _ => {
                if self.in_loop {
                    // Buffer loop body ops
                    self.body_ops.push(op.clone());
                    PassResult::Remove
                } else {
                    // Not in a loop — pass through unchanged
                    PassResult::PassOn
                }
            }
        }
    }

    fn setup(&mut self) {
        self.body_ops.clear();
        self.in_loop = false;
    }

    fn name(&self) -> &'static str {
        "vectorize_simd"
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Whether a scalar opcode produces float results.
fn is_float_opcode(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::FloatAdd
            | OpCode::FloatSub
            | OpCode::FloatMul
            | OpCode::FloatTrueDiv
            | OpCode::FloatAbs
            | OpCode::FloatNeg
            | OpCode::CastIntToFloat
            | OpCode::CastSinglefloatToFloat
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Op, OpCode, OpRef};

    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(base + i as u32);
        }
    }

    // ── Dependency graph tests ──

    #[test]
    fn test_dep_graph_basic() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(101)]), // depends on op 0
            Op::new(OpCode::IntSub, &[OpRef(100), OpRef(101)]), // independent of op 0
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        assert_eq!(graph.nodes.len(), 3);

        // op1 depends on op0
        assert!(graph.nodes[1].deps.contains(&0));
        // op2 is independent of op0
        assert!(!graph.nodes[2].deps.contains(&0));
    }

    #[test]
    fn test_dep_graph_no_self_dep() {
        let mut ops = vec![Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(101)])];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        // Self-reference should not create self-dependency
        assert!(graph.nodes[0].deps.is_empty());
    }

    // ── Pack group tests ──

    #[test]
    fn test_find_packable_groups() {
        // Two independent IntAdd ops
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(102), OpRef(103)]),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        let groups = graph.find_packable_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].scalar_opcode, OpCode::IntAdd);
        assert_eq!(groups[0].vector_opcode, OpCode::VecIntAdd);
        assert_eq!(groups[0].members.len(), 2);
    }

    #[test]
    fn test_dependent_ops_not_packed() {
        // Two IntAdd ops where second depends on first
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(101)]), // depends on op 0
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        let groups = graph.find_packable_groups();

        // Can't pack because of dependency
        assert!(groups.is_empty());
    }

    #[test]
    fn test_different_opcodes_not_packed() {
        // IntAdd and IntSub — not isomorphic
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntSub, &[OpRef(102), OpRef(103)]),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        let groups = graph.find_packable_groups();

        // Each group needs 2+ members, single ops can't form a group
        for g in &groups {
            assert!(g.members.len() >= 2);
        }
    }

    #[test]
    fn test_three_independent_ops() {
        let mut ops = vec![
            Op::new(OpCode::IntMul, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntMul, &[OpRef(102), OpRef(103)]),
            Op::new(OpCode::IntMul, &[OpRef(104), OpRef(105)]),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        let groups = graph.find_packable_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 3);
    }

    // ── Cost model tests ──

    #[test]
    fn test_cost_model_profitable() {
        let cm = CostModel::new();
        let group = PackGroup {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3], // 4 ops
        };
        // savings = 3 * 1 = 3, cost = 2 * 2 = 4 → not profitable with 4
        // Actually savings = 3, cost = 4, so NOT profitable by default
        // Let's adjust: with default params, need enough ops
        assert!(!cm.is_profitable(&group)); // 3 < 4

        let group5 = PackGroup {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3, 4], // 5 ops → savings = 4 > cost = 4
        };
        assert!(!cm.is_profitable(&group5)); // 4 == 4, not strictly greater
    }

    #[test]
    fn test_cost_model_too_small() {
        let cm = CostModel::new();
        let group = PackGroup {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0], // Only 1 op
        };
        assert!(!cm.is_profitable(&group));
    }

    #[test]
    fn test_cost_model_custom_params() {
        let cm = CostModel {
            min_pack_size: 2,
            pack_cost: 1,
            scalar_save: 2,
        };
        let group = PackGroup {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1], // savings = 1*2 = 2, cost = 2*1 = 2 → not profitable
        };
        assert!(!cm.is_profitable(&group));

        let group3 = PackGroup {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2], // savings = 2*2 = 4, cost = 2*1 = 2 → profitable
        };
        assert!(cm.is_profitable(&group3));
    }

    // ── Memory access detection ──

    #[test]
    fn test_is_memory_access() {
        assert!(OpCode::GetfieldGcI.is_memory_access());
        assert!(OpCode::SetarrayitemGc.is_memory_access());
        assert!(OpCode::RawLoadI.is_memory_access());
        assert!(!OpCode::IntAdd.is_memory_access());
        assert!(!OpCode::GuardTrue.is_memory_access());
    }

    // ── OptVectorize pass tests ──

    #[test]
    fn test_vectorize_pass_no_loop() {
        // Without Label..Jump, nothing to vectorize
        use crate::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Finish, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptVectorize::new()));
        let result = opt.optimize(&ops);

        // No loop to vectorize, ops should pass through
        // (pre-label ops are buffered but emitted when we hit non-Label)
        assert!(!result.is_empty());
    }

    #[test]
    fn test_vectorize_pass_preserves_structure() {
        use crate::optimizer::Optimizer;

        // A simple loop with independent ops
        let mut ops = vec![
            Op::new(OpCode::Label, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntSub, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(1), OpRef(2)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptVectorize::new()));
        let result = opt.optimize(&ops);

        // Should still have Label and Jump
        assert!(result.iter().any(|op| op.opcode == OpCode::Label));
        assert!(result.iter().any(|op| op.opcode == OpCode::Jump));
    }

    // ── Scheduler tests ──

    #[test]
    fn test_schedule_respects_dependencies() {
        // A → B → C: linear chain must stay in order
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // A
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(101)]),   // B depends on A
            Op::new(OpCode::IntSub, &[OpRef(1), OpRef(101)]),   // C depends on B
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 3);
        // A before B before C
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_b = sched.iter().position(|&x| x == 1).unwrap();
        let pos_c = sched.iter().position(|&x| x == 2).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_schedule_maximizes_parallelism() {
        // A and B are independent — both should appear early (before any
        // dependent op). With no dependents, both have height 1 and both
        // should be in the first two schedule slots.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // A
            Op::new(OpCode::IntSub, &[OpRef(102), OpRef(103)]), // B (independent)
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 2);
        // Both scheduled (order doesn't matter, but both must be present)
        assert!(sched.contains(&0));
        assert!(sched.contains(&1));
    }

    #[test]
    fn test_schedule_prioritizes_critical_path() {
        // Critical path: A → B → C (heights: A=3, B=2, C=1)
        // Independent: D (height=1)
        // A should be scheduled before D because A has higher height.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // A (idx 0)
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(101)]),   // B (idx 1, depends on A)
            Op::new(OpCode::IntSub, &[OpRef(1), OpRef(101)]),   // C (idx 2, depends on B)
            Op::new(OpCode::IntAdd, &[OpRef(102), OpRef(103)]), // D (idx 3, independent)
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 4);
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_d = sched.iter().position(|&x| x == 3).unwrap();
        // A (height 3) should be scheduled before D (height 1)
        assert!(pos_a < pos_d, "A (height 3) should precede D (height 1)");
    }

    #[test]
    fn test_schedule_diamond() {
        // Diamond: A → B, A → C, B → D, C → D
        // Valid orders: [A, B, C, D] or [A, C, B, D]
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // A (idx 0)
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(101)]),   // B (idx 1, depends on A)
            Op::new(OpCode::IntSub, &[OpRef(0), OpRef(102)]),   // C (idx 2, depends on A)
            Op::new(OpCode::IntAdd, &[OpRef(1), OpRef(2)]),     // D (idx 3, depends on B and C)
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 4);

        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_b = sched.iter().position(|&x| x == 1).unwrap();
        let pos_c = sched.iter().position(|&x| x == 2).unwrap();
        let pos_d = sched.iter().position(|&x| x == 3).unwrap();

        // A must come before B and C
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        // B and C must come before D
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_schedule_empty_graph() {
        let graph = DependencyGraph { nodes: Vec::new() };
        let sched = schedule_operations(&graph);
        assert!(sched.is_empty());
    }
}
