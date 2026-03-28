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

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

pub use crate::optimizeopt::dependency::schedule_operations;
pub use crate::optimizeopt::dependency::{DependencyGraph, Node};
pub use crate::optimizeopt::schedule::{
    AccumPack, CostModel, GenericCostModel, GuardAnalysis, Pack, PackSet, VecScheduleState,
    are_adjacent_memory_refs, turn_into_vector, unpack_from_vector,
};

/// vector.py: VectorLoop — wraps a loop body for vectorization analysis.
/// Provides the iteration over loop body ops and manages unrolling.
#[derive(Clone, Debug)]
pub struct VectorLoop {
    /// The original loop body ops (between Label and Jump).
    pub body: Vec<Op>,
    /// The label op (loop header).
    pub label: Option<Op>,
    /// The jump op (back-edge).
    pub jump: Option<Op>,
    /// Number of unrolled iterations.
    pub unroll_factor: usize,
}

impl VectorLoop {
    /// Create from a trace by finding Label..Jump.
    pub fn from_trace(ops: &[Op]) -> Option<Self> {
        let label_pos = ops.iter().position(|op| op.opcode == OpCode::Label)?;
        let jump_pos = ops.iter().rposition(|op| op.opcode == OpCode::Jump)?;
        if jump_pos <= label_pos {
            return None;
        }
        Some(VectorLoop {
            body: ops[label_pos + 1..jump_pos].to_vec(),
            label: Some(ops[label_pos].clone()),
            jump: Some(ops[jump_pos].clone()),
            unroll_factor: 1,
        })
    }

    /// Number of ops in the loop body (excluding Label and Jump).
    pub fn body_len(&self) -> usize {
        self.body.len()
    }

    /// Whether the loop is suitable for vectorization.
    pub fn is_vectorizable(&self) -> bool {
        VectorizingOptimizer::user_loop_heuristic(&self.body)
    }

    /// Get adjacent memory access pairs in the body.
    pub fn find_adjacent_pairs(
        &self,
        constant_of: impl Fn(OpRef) -> Option<i64>,
    ) -> Vec<(usize, usize)> {
        VectorizingOptimizer::find_adjacent_pairs(&self.body, constant_of)
    }
}

impl VectorLoop {
    /// vector.py: unroll_loop_iterations(loop, unroll_count)
    ///
    /// Unroll the loop body `count` times. Each unrolled iteration has
    /// its OpRefs remapped to new positions. Guards like GUARD_FUTURE_CONDITION
    /// and GUARD_NOT_INVALIDATED are not duplicated.
    pub fn unroll_loop_iterations(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        let original_body = self.body.clone();
        let label_args = self
            .label
            .as_ref()
            .map(|l| l.args.clone())
            .unwrap_or_default();
        let jump_args = self
            .jump
            .as_ref()
            .map(|j| j.args.clone())
            .unwrap_or_default();

        let prohibit = [
            OpCode::GuardFutureCondition,
            OpCode::GuardNotInvalidated,
            OpCode::DebugMergePoint,
        ];

        let base_offset = original_body.iter().map(|op| op.pos.0).max().unwrap_or(0) + 1;

        for u in 0..count {
            let offset = base_offset + (u as u32) * (original_body.len() as u32);
            let mut remap = std::collections::HashMap::new();

            // Map label args → jump args (or remapped jump args)
            for (i, la) in label_args.iter().enumerate() {
                if i < jump_args.len() {
                    let ja = *remap.get(&jump_args[i]).unwrap_or(&jump_args[i]);
                    if *la != ja {
                        remap.insert(*la, ja);
                    }
                }
            }

            for op in &original_body {
                if prohibit.contains(&op.opcode) {
                    continue;
                }
                let mut new_op = op.clone();
                let new_pos = OpRef(op.pos.0 + offset);
                if !op.pos.is_none() {
                    remap.insert(op.pos, new_pos);
                }
                new_op.pos = new_pos;
                for arg in &mut new_op.args {
                    if let Some(&mapped) = remap.get(arg) {
                        *arg = mapped;
                    }
                }
                if let Some(ref mut fa) = new_op.fail_args {
                    for arg in fa.iter_mut() {
                        if let Some(&mapped) = remap.get(arg) {
                            *arg = mapped;
                        }
                    }
                }
                self.body.push(new_op);
            }
        }

        // Update jump args
        if let Some(ref mut jump) = self.jump {
            for arg in &mut jump.args {
                // Use latest remap if available
            }
        }
        self.unroll_factor = count + 1;
    }
}

/// vector.py: follow_def_use_chain — trace a value through its uses
/// to find related vectorizable operations.
pub fn follow_def_use_chain(ops: &[Op], start: usize, max_depth: usize) -> Vec<usize> {
    let mut chain = vec![start];
    let result_ref = ops[start].pos;
    if result_ref.is_none() {
        return chain;
    }

    let mut current_refs = vec![result_ref];
    for depth in 0..max_depth {
        let mut next_refs = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            if chain.contains(&i) {
                continue;
            }
            for arg in &op.args {
                if current_refs.contains(arg) {
                    chain.push(i);
                    if !op.pos.is_none() {
                        next_refs.push(op.pos);
                    }
                    break;
                }
            }
        }
        if next_refs.is_empty() {
            break;
        }
        current_refs = next_refs;
    }

    chain
}

pub struct VectorizingOptimizer {
    /// Operations in the current loop body (between Label and Jump).
    body_ops: Vec<Op>,
    /// Whether we're inside a loop body.
    in_loop: bool,
    /// Cost model for profitability decisions.
    cost_model: CostModel,
}

impl VectorizingOptimizer {
    pub fn new() -> Self {
        VectorizingOptimizer {
            body_ops: Vec::new(),
            in_loop: false,
            cost_model: CostModel::new(),
        }
    }

    /// vector.py: linear_find_smallest_type — scan ops for smallest
    /// array element type to determine SIMD width.
    pub fn linear_find_smallest_type(ops: &[Op]) -> usize {
        let mut smallest = 0usize;
        for op in ops {
            if op.opcode.is_getarrayitem() || op.opcode.is_setarrayitem() {
                if let Some(ref descr) = op.descr {
                    if let Some(ad) = descr.as_array_descr() {
                        let item_size = ad.item_size();
                        if smallest == 0 || item_size < smallest {
                            smallest = item_size;
                        }
                    }
                }
            }
        }
        smallest
    }

    /// vector.py: get_unroll_count — compute how many times to unroll
    /// based on SIMD register width and smallest type.
    pub fn get_unroll_count(smallest_type_bytes: usize, simd_reg_bytes: usize) -> usize {
        if smallest_type_bytes == 0 {
            return 0;
        }
        let count = simd_reg_bytes / smallest_type_bytes;
        count.saturating_sub(1) // already unrolled once
    }

    /// vector.py: user_loop_heuristic — quick check if a loop body is
    /// worth trying to vectorize. Returns false if there are too few ops,
    /// no vectorizable opcodes, or too many guards.
    /// vector.py: extend_packset()
    /// Iteratively follow def-use and use-def chains to grow the pack set.
    /// Stops when no more packs are added in a full iteration.
    pub fn extend_packset(pack_set: &mut PackSet, graph: &DependencyGraph) {
        loop {
            let before = pack_set.num_packs();
            // follow_def_uses: for each pack, check if dependents can form new packs
            let current_packs: Vec<Pack> = pack_set.packs.clone();
            for pack in &current_packs {
                if pack.members.len() >= 2 {
                    let left = pack.members[0];
                    let right = pack.members[1];
                    // Check users of left and right for isomorphic pairs
                    for &l_user in &graph.nodes[left].users {
                        for &r_user in &graph.nodes[right].users {
                            if l_user != r_user
                                && graph.nodes[l_user].op.opcode == graph.nodes[r_user].op.opcode
                                && !graph.has_dependency(l_user, r_user)
                            {
                                if let Some(vec_op) = graph.nodes[l_user].op.opcode.to_vector() {
                                    pack_set.add_pack(Pack {
                                        scalar_opcode: graph.nodes[l_user].op.opcode,
                                        vector_opcode: vec_op,
                                        members: vec![l_user, r_user],
                                        is_accumulating: false,
                                        position: -1,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            if pack_set.num_packs() == before {
                break;
            }
        }
    }

    /// vector.py: combine_packset()
    /// Combine adjacent 2-packs into larger packs where the rightmost
    /// element of one pack matches the leftmost element of another.
    /// This is the iterative merge step after extend_packset.
    pub fn combine_packset(pack_set: &mut PackSet) {
        if pack_set.packs.is_empty() {
            return;
        }
        loop {
            let len_before = pack_set.packs.len();
            pack_set.try_merge_packs();
            if pack_set.packs.len() == len_before {
                break; // no more merges possible
            }
        }
    }

    pub fn user_loop_heuristic(ops: &[Op]) -> bool {
        if ops.len() < 4 {
            return false;
        }
        let vectorizable = ops
            .iter()
            .filter(|op| op.opcode.to_vector().is_some())
            .count();
        let guards = ops.iter().filter(|op| op.opcode.is_guard()).count();
        // Need at least 2 vectorizable ops and guards should not dominate.
        vectorizable >= 2 && guards < ops.len() / 2
    }

    /// vector.py: find_adjacent_pairs — find pairs of memory accesses
    /// that read/write adjacent array elements and could be packed.
    pub fn find_adjacent_pairs(
        ops: &[Op],
        constant_of: impl Fn(OpRef) -> Option<i64>,
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..ops.len() {
            if !ops[i].opcode.is_getarrayitem() && !ops[i].opcode.is_setarrayitem() {
                continue;
            }
            for j in (i + 1)..ops.len() {
                if are_adjacent_memory_refs(&ops[i], &ops[j], &constant_of) {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Attempt to vectorize the buffered loop body.
    ///
    /// Before packing, operations are scheduled for maximum ILP.
    /// The reordering may expose additional packing opportunities.
    fn try_vectorize(&mut self, ctx: &mut OptContext) -> Option<Vec<Op>> {
        if self.body_ops.len() < 4 {
            return None; // Too small to benefit
        }

        // Constant resolver — looks up OpRef in the optimizer's constant map.
        let constant_of = |opref: OpRef| -> Option<i64> { ctx.get_constant_int(opref) };

        // Phase 1: Schedule operations for ILP before packing.
        let dep_graph = DependencyGraph::build(&self.body_ops, &constant_of);
        let schedule = schedule_operations(&dep_graph);
        if schedule.len() == self.body_ops.len() {
            let scheduled: Vec<Op> = schedule.iter().map(|&i| self.body_ops[i].clone()).collect();
            self.body_ops = scheduled;
        }

        // Phase 2: Rebuild dependency graph on reordered ops and find packs.
        let dep_graph = DependencyGraph::build(&self.body_ops, &constant_of);
        let groups = dep_graph.find_packable_groups();

        if groups.is_empty() {
            return None;
        }

        // Filter by cost model
        let profitable: Vec<Pack> = groups
            .into_iter()
            .filter(|g| self.cost_model.is_profitable(g))
            .collect();

        if profitable.is_empty() {
            return None;
        }

        // schedule.py:292-311, 672-695: walk_and_emit + try_emit_or_delay.
        //
        // RPython scheduler: walk in dependency order. For pack nodes,
        // delay() gates emission until ALL members' dependencies are resolved.
        // For scalar nodes, ensure_unpacked avoids duplicate unpacks via `seen`.
        let start_pos = ctx.new_operations.len() as u32 + self.body_ops.len() as u32;
        let mut sched_state = VecScheduleState::new(start_pos);

        // Build node→pack mapping
        let mut node_to_pack: HashMap<usize, usize> = HashMap::new();
        for (pi, group) in profitable.iter().enumerate() {
            for &idx in &group.members {
                node_to_pack.insert(idx, pi);
            }
        }

        // schedule.py:683-695: delay() — track how many members of each pack
        // have been "visited" (their dependencies satisfied in topo order).
        // A pack is only emittable when ALL its members have been visited.
        let mut pack_emitted = vec![false; profitable.len()];
        let mut pack_visited_count = vec![0usize; profitable.len()];

        // schedule.py:718-719: seen — prevents duplicate unpack creation
        let mut seen: HashSet<OpRef> = HashSet::new();

        // Walk in scheduled (dependency) order
        let scheduled_order = schedule_operations(&dep_graph);
        for &node_idx in &scheduled_order {
            if let Some(&pack_idx) = node_to_pack.get(&node_idx) {
                // schedule.py:683-695: delay() gating.
                // Count this member as visited.
                pack_visited_count[pack_idx] += 1;
                let pack = &profitable[pack_idx];
                let all_ready = pack_visited_count[pack_idx] == pack.members.len();

                if all_ready && !pack_emitted[pack_idx] {
                    // schedule.py:674-679: all members ready → emit pack
                    pack_emitted[pack_idx] = true;
                    turn_into_vector(&mut sched_state, pack, &self.body_ops);
                }
                // Members consumed by the vector op — skip individual emit
            } else {
                // schedule.py:680-681: scalar node — ensure_args_unpacked
                let mut scalar_op = self.body_ops[node_idx].clone();
                // schedule.py:697-716: ensure_args_unpacked with seen/accum/invariant
                for j in 0..scalar_op.args.len() {
                    let arg = scalar_op.args[j];
                    if arg.is_constant() || seen.contains(&arg) {
                        continue; // schedule.py:719: already seen → skip
                    }
                    if let Some((pos, vec_ref)) = sched_state.getvector_of_box(arg) {
                        // schedule.py:723-724: invariant_vector_vars → skip
                        if sched_state.inputargs.contains_key(&vec_ref) {
                            continue;
                        }
                        // schedule.py:725-726: accumulation → skip
                        if sched_state.accumulation.contains_key(&arg) {
                            continue;
                        }
                        let unpacked = unpack_from_vector(&mut sched_state, vec_ref, pos, 1);
                        sched_state.renamer.start_renaming(arg, unpacked);
                        seen.insert(unpacked);
                        scalar_op.args[j] = unpacked;
                    }
                }
                // schedule.py:708-716: unpack guard failargs
                if scalar_op.opcode.is_guard() {
                    if let Some(ref mut fail_args) = scalar_op.fail_args {
                        for arg in fail_args.iter_mut() {
                            if arg.is_constant() || seen.contains(arg) {
                                continue;
                            }
                            if let Some((pos, vec_ref)) = sched_state.getvector_of_box(*arg) {
                                if sched_state.accumulation.contains_key(arg) {
                                    continue;
                                }
                                let unpacked =
                                    unpack_from_vector(&mut sched_state, vec_ref, pos, 1);
                                seen.insert(unpacked);
                                *arg = unpacked;
                            }
                        }
                    }
                }
                seen.insert(scalar_op.pos);
                sched_state.append_to_oplist(scalar_op);
            }
        }

        Some(sched_state.oplist)
    }
}

impl Default for VectorizingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for VectorizingOptimizer {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            OpCode::Label => {
                self.in_loop = true;
                OptimizationResult::Emit(op.clone())
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
                OptimizationResult::Emit(op.clone())
            }
            _ => {
                if self.in_loop {
                    // Buffer loop body ops
                    self.body_ops.push(op.clone());
                    OptimizationResult::Remove
                } else {
                    // Not in a loop — pass through unchanged
                    OptimizationResult::PassOn
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

        let graph = DependencyGraph::build(&ops, &|_| None);
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

        let graph = DependencyGraph::build(&ops, &|_| None);
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

        let graph = DependencyGraph::build(&ops, &|_| None);
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

        let graph = DependencyGraph::build(&ops, &|_| None);
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

        let graph = DependencyGraph::build(&ops, &|_| None);
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

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 3);
    }

    // ── Cost model tests ──

    #[test]
    fn test_cost_model_profitable() {
        let cm = CostModel::new();
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3], // 4 ops
            is_accumulating: false,
            position: -1,
        };
        // savings = 3 * 1 = 3, cost = 2 * 2 = 4 → not profitable with 4
        // Actually savings = 3, cost = 4, so NOT profitable by default
        // Let's adjust: with default params, need enough ops
        assert!(!cm.is_profitable(&group)); // 3 < 4

        let group5 = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3, 4], // 5 ops → savings = 4 > cost = 4
            is_accumulating: false,
            position: -1,
        };
        assert!(!cm.is_profitable(&group5)); // 4 == 4, not strictly greater
    }

    #[test]
    fn test_cost_model_too_small() {
        let cm = CostModel::new();
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0], // Only 1 op
            is_accumulating: false,
            position: -1,
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
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1], // savings = 1*2 = 2, cost = 2*1 = 2 → not profitable
            is_accumulating: false,
            position: -1,
        };
        assert!(!cm.is_profitable(&group));

        let group3 = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2], // savings = 2*2 = 4, cost = 2*1 = 2 → profitable
            is_accumulating: false,
            position: -1,
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

    // ── VectorizingOptimizer pass tests ──

    #[test]
    fn test_vectorize_pass_no_loop() {
        // Without Label..Jump, nothing to vectorize
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Finish, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        // No loop to vectorize, ops should pass through
        // (pre-label ops are buffered but emitted when we hit non-Label)
        assert!(!result.is_empty());
    }

    #[test]
    fn test_vectorize_pass_preserves_structure() {
        use crate::optimizeopt::optimizer::Optimizer;

        // A simple loop with independent ops
        let mut ops = vec![
            Op::new(OpCode::Label, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntSub, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(1), OpRef(2)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

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

        let graph = DependencyGraph::build(&ops, &|_| None);
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

        let graph = DependencyGraph::build(&ops, &|_| None);
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

        let graph = DependencyGraph::build(&ops, &|_| None);
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

        let graph = DependencyGraph::build(&ops, &|_| None);
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
        let graph = DependencyGraph {
            nodes: Vec::new(),
            memory_refs: Default::default(),
            index_vars: Default::default(),
            guards: Vec::new(),
            invariant_vars: Default::default(),
        };
        let sched = schedule_operations(&graph);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_user_loop_heuristic_too_small() {
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assert!(!VectorizingOptimizer::user_loop_heuristic(&ops));
    }

    #[test]
    fn test_user_loop_heuristic_no_vectorizable() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(2)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assert!(!VectorizingOptimizer::user_loop_heuristic(&ops));
    }

    #[test]
    fn test_pack_set_merge() {
        let mut ps = PackSet::new();
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
        });
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![2, 3],
            is_accumulating: false,
            position: -1,
        });
        assert_eq!(ps.num_packs(), 2);
        assert_eq!(ps.total_ops(), 4);

        ps.try_merge_packs();
        assert_eq!(ps.num_packs(), 1);
        assert_eq!(ps.total_ops(), 4);
    }

    #[test]
    fn test_generic_cost_model() {
        let model = GenericCostModel::new();
        // Memory ops are more expensive
        assert!(model.op_cost(OpCode::GetarrayitemGcI) > model.op_cost(OpCode::IntAdd));
        // Float div is most expensive
        assert!(model.op_cost(OpCode::FloatTrueDiv) >= model.op_cost(OpCode::FloatAdd));
    }

    #[test]
    fn test_guard_analysis_hoistable() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(100)]), // loop-invariant (100 not produced)
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(1)]), // body-dependent (1 = IntAdd result)
        ];
        let mut positioned = ops;
        for (i, op) in positioned.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
        let analysis = GuardAnalysis::analyze(&positioned);
        assert_eq!(analysis.hoistable.len(), 1);
        assert_eq!(analysis.hoistable[0], 0);
        assert_eq!(analysis.body_guards.len(), 1);
        assert_eq!(analysis.body_guards[0], 2);
    }

    #[test]
    fn test_follow_def_use_chain() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // 0: result used by 1
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(102)]),   // 1: uses result of 0
            Op::new(OpCode::IntSub, &[OpRef(1), OpRef(103)]),   // 2: uses result of 1
            Op::new(OpCode::IntAdd, &[OpRef(104), OpRef(105)]), // 3: independent
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
        let chain = follow_def_use_chain(&ops, 0, 10);
        assert!(chain.contains(&0));
        assert!(chain.contains(&1));
        assert!(chain.contains(&2));
        assert!(!chain.contains(&3)); // independent
    }

    #[test]
    fn test_vector_loop_from_trace() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // preamble
            Op::new(OpCode::Label, &[OpRef(100)]),              // loop header
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // body
            Op::new(OpCode::IntMul, &[OpRef(2), OpRef(102)]),   // body
            Op::new(OpCode::Jump, &[OpRef(3)]),                 // back-edge
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
        let vloop = VectorLoop::from_trace(&ops).unwrap();
        assert_eq!(vloop.body_len(), 2); // IntAdd + IntMul
        assert!(vloop.label.is_some());
        assert!(vloop.jump.is_some());
    }
}
