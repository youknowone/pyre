//! Instruction scheduling and pack management for vectorization.
//!
//! Mirrors RPython's `schedule.py` and `costmodel.py`: pack groups,
//! pack sets, accumulation tracking, guard analysis, and cost models.

use std::collections::HashMap;

use majit_ir::{Op, OpCode, OpRef};

use crate::optimizeopt::dependency::DependencyGraph;

/// schedule.py:781+: A pack is a set of n isomorphic operations that can
/// execute as a single SIMD instruction.
#[derive(Clone, Debug)]
pub struct Pack {
    /// The scalar opcode of the group members.
    pub scalar_opcode: OpCode,
    /// The vector opcode to replace them with.
    pub vector_opcode: OpCode,
    /// Indices into the DepGraph nodes.
    pub members: Vec<usize>,
    /// schedule.py:811: whether this pack tracks an accumulation (reduction).
    pub is_accumulating: bool,
    /// schedule.py:989: accumulation argument position (-1 = none).
    pub position: i32,
}

/// vector.py: PackSet — manages packs and supports merging
/// 2-packs into 4-packs (or larger) when possible.
#[derive(Clone, Debug, Default)]
pub struct PackSet {
    /// All packs found so far.
    pub packs: Vec<Pack>,
}

impl PackSet {
    pub fn new() -> Self {
        PackSet { packs: Vec::new() }
    }

    /// Add a pack to the set.
    pub fn add_pack(&mut self, pack: Pack) {
        self.packs.push(pack);
    }

    /// vector.py:460-494: combine_packset — merge packs where
    /// pack1.rightmost == pack2.leftmost (schedule.py:931-942).
    /// Only merges packs with matching edge, NOT just same opcode.
    pub fn try_merge_packs(&mut self) {
        loop {
            let len_before = self.packs.len();
            let mut i = 0;
            while i < self.packs.len() {
                let mut j = 0;
                while j < self.packs.len() {
                    if i == j {
                        j += 1;
                        continue;
                    }
                    if i < self.packs.len() && j < self.packs.len() {
                        // schedule.py:931-942: rightmost_match_leftmost
                        let rightmost = *self.packs[i].members.last().unwrap_or(&usize::MAX);
                        let leftmost = *self.packs[j].members.first().unwrap_or(&usize::MAX);
                        // schedule.py:937-941: accumulating pack constraints
                        let accum_ok = if self.packs[i].is_accumulating {
                            self.packs[j].is_accumulating
                                && self.packs[i].position == self.packs[j].position
                        } else {
                            true
                        };
                        if rightmost == leftmost
                            && self.packs[i].scalar_opcode == self.packs[j].scalar_opcode
                            && accum_ok
                        {
                            // vector.py:753+: combine — merge j into i, skip overlap
                            let mut merged_members = self.packs[i].members.clone();
                            merged_members.extend_from_slice(&self.packs[j].members[1..]);
                            self.packs[i].members = merged_members;
                            self.packs.remove(j);
                            if j < i {
                                i -= 1;
                            }
                            continue; // re-check from j
                        }
                    }
                    j += 1;
                }
                i += 1;
            }
            if self.packs.len() == len_before {
                break;
            }
        }
    }

    /// vector.py: extend_packset()
    ///
    /// Follow dependency chains to find more candidates to put into pairs.
    /// For each existing pack, check if the users (def→use) or producers
    /// (use→def) of the packed ops can also form isomorphic pairs.
    pub fn extend_packset(&mut self, graph: &DependencyGraph) {
        loop {
            let count_before = self.packs.len();
            let num_packs = self.packs.len();
            for pi in 0..num_packs {
                if self.packs[pi].members.len() < 2 {
                    continue;
                }
                let left = self.packs[pi].members[0];
                let right = self.packs[pi].members[1];
                // follow_def_uses: users of left/right that are isomorphic
                for &uleft in &graph.nodes[left].users {
                    for &uright in &graph.nodes[right].users {
                        if uleft < uright
                            && graph.nodes[uleft].op.opcode == graph.nodes[uright].op.opcode
                            && !self.already_packed(uleft)
                            && !self.already_packed(uright)
                        {
                            let sc = graph.nodes[uleft].op.opcode;
                            self.packs.push(Pack {
                                scalar_opcode: sc,
                                vector_opcode: sc.to_vector().unwrap_or(sc),
                                members: vec![uleft, uright],
                                is_accumulating: false,
                                position: -1,
                            });
                        }
                    }
                }
                // follow_use_defs: deps of left/right that are isomorphic
                for &dleft in &graph.nodes[left].deps {
                    for &dright in &graph.nodes[right].deps {
                        if dleft < dright
                            && graph.nodes[dleft].op.opcode == graph.nodes[dright].op.opcode
                            && !self.already_packed(dleft)
                            && !self.already_packed(dright)
                        {
                            let sc = graph.nodes[dleft].op.opcode;
                            self.packs.push(Pack {
                                scalar_opcode: sc,
                                vector_opcode: sc.to_vector().unwrap_or(sc),
                                members: vec![dleft, dright],
                                is_accumulating: false,
                                position: -1,
                            });
                        }
                    }
                }
            }
            if self.packs.len() == count_before {
                break;
            }
        }
    }

    /// Check if an op index is already in some pack.
    fn already_packed(&self, idx: usize) -> bool {
        self.packs.iter().any(|p| p.members.contains(&idx))
    }

    /// vector.py: combine_packset()
    ///
    /// Combine packs that share edges: if pack1.rightmost == pack2.leftmost,
    /// merge them into a longer pack. Iterates until stable.
    pub fn combine_packset(&mut self) {
        loop {
            let len_before = self.packs.len();
            let mut i = 0;
            while i < self.packs.len() {
                let mut j = 0;
                while j < self.packs.len() {
                    if i == j {
                        j += 1;
                        continue;
                    }
                    if i < self.packs.len() && j < self.packs.len() {
                        let right_of_i = *self.packs[i].members.last().unwrap_or(&usize::MAX);
                        let left_of_j = *self.packs[j].members.first().unwrap_or(&usize::MAX);
                        if right_of_i == left_of_j
                            && self.packs[i].scalar_opcode == self.packs[j].scalar_opcode
                        {
                            // Merge j into i
                            let mut merged_members = self.packs[i].members.clone();
                            merged_members.extend_from_slice(&self.packs[j].members[1..]);
                            self.packs[i].members = merged_members;
                            self.packs.remove(j);
                            if j < i {
                                i -= 1;
                            }
                            continue; // re-check from j
                        }
                    }
                    j += 1;
                }
                i += 1;
            }
            if self.packs.len() == len_before {
                break;
            }
        }
    }

    /// Number of packs.
    pub fn num_packs(&self) -> usize {
        self.packs.len()
    }

    /// Total number of ops across all packs.
    pub fn total_ops(&self) -> usize {
        self.packs.iter().map(|p| p.members.len()).sum()
    }
}

/// vector.py: Adjacent memory reference detection.
/// Checks if two memory operations access adjacent array elements.
pub fn are_adjacent_memory_refs(
    op_a: &majit_ir::Op,
    op_b: &majit_ir::Op,
    constant_of: impl Fn(OpRef) -> Option<i64>,
) -> bool {
    // Both must be the same opcode (e.g., GETARRAYITEM_GC_I)
    if op_a.opcode != op_b.opcode {
        return false;
    }
    // Both must access the same array (arg0)
    if op_a.num_args() < 2 || op_b.num_args() < 2 {
        return false;
    }
    if op_a.arg(0) != op_b.arg(0) {
        return false;
    }
    // Indices must differ by exactly 1
    if let (Some(idx_a), Some(idx_b)) = (constant_of(op_a.arg(1)), constant_of(op_b.arg(1))) {
        return (idx_b - idx_a).abs() == 1;
    }
    false
}

/// vector.py: Accumulation pack — tracks reduction operations
/// (e.g., sum += array[i]) that can be vectorized with horizontal
/// reduction instructions.
#[derive(Clone, Debug)]
pub struct AccumPack {
    /// The scalar opcode of the accumulation (e.g., IntAdd, FloatAdd).
    pub scalar_opcode: OpCode,
    /// The initial accumulator value OpRef.
    pub init_value: OpRef,
    /// Indices of the accumulation operations in the loop body.
    pub members: Vec<usize>,
    /// Whether this is a float accumulation.
    pub is_float: bool,
}

/// vector.py: Guard analysis result — determines which guards can be
/// moved to the loop header (hoisted) to expose more vectorization.
#[derive(Clone, Debug)]
pub struct GuardAnalysis {
    /// Guards that can be hoisted to the loop header.
    pub hoistable: Vec<usize>,
    /// Guards that must remain in the loop body.
    pub body_guards: Vec<usize>,
}

impl GuardAnalysis {
    /// Analyze guards in a loop body for hoistability.
    /// vector.py: analyze_guards()
    /// A guard is hoistable if its arguments are loop-invariant
    /// (not produced by any op in the loop body).
    pub fn analyze(ops: &[Op]) -> Self {
        let mut body_results: std::collections::HashSet<OpRef> = std::collections::HashSet::new();
        for op in ops {
            if !op.pos.is_none() {
                body_results.insert(op.pos);
            }
        }

        let mut hoistable = Vec::new();
        let mut body_guards = Vec::new();

        for (i, op) in ops.iter().enumerate() {
            if !op.opcode.is_guard() {
                continue;
            }
            let all_invariant = op.args.iter().all(|arg| !body_results.contains(arg));
            if all_invariant {
                hoistable.push(i);
            } else {
                body_guards.push(i);
            }
        }

        GuardAnalysis {
            hoistable,
            body_guards,
        }
    }
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
/// costmodel.py: GenericCostModel — per-opcode cost estimation.
/// Maps opcodes to their estimated cost in abstract units.
pub struct GenericCostModel {
    /// Per-opcode cost overrides: opcode → cost.
    per_opcode_cost: std::collections::HashMap<OpCode, i32>,
    /// Default cost for opcodes not in the override map.
    default_cost: i32,
}

impl GenericCostModel {
    pub fn new() -> Self {
        let mut costs = std::collections::HashMap::new();
        // costmodel.py: memory ops are more expensive than ALU ops
        costs.insert(OpCode::GetarrayitemGcI, 3);
        costs.insert(OpCode::GetarrayitemGcR, 3);
        costs.insert(OpCode::GetarrayitemGcF, 3);
        costs.insert(OpCode::SetarrayitemGc, 3);
        costs.insert(OpCode::GetfieldGcI, 2);
        costs.insert(OpCode::GetfieldGcR, 2);
        costs.insert(OpCode::SetfieldGc, 2);
        // Float ops are more expensive
        costs.insert(OpCode::FloatAdd, 2);
        costs.insert(OpCode::FloatSub, 2);
        costs.insert(OpCode::FloatMul, 2);
        costs.insert(OpCode::FloatTrueDiv, 4);
        GenericCostModel {
            per_opcode_cost: costs,
            default_cost: 1,
        }
    }

    /// Get the cost of a single operation.
    pub fn op_cost(&self, opcode: OpCode) -> i32 {
        self.per_opcode_cost
            .get(&opcode)
            .copied()
            .unwrap_or(self.default_cost)
    }

    /// Estimate total savings from vectorizing a pack group.
    pub fn estimate_savings(&self, group: &Pack) -> i32 {
        let n = group.members.len() as i32;
        let per_op = self.op_cost(group.scalar_opcode);
        // Savings = (n-1) ops eliminated * per-op cost
        // Cost = pack + unpack overhead
        let savings = (n - 1) * per_op;
        let overhead = 2 * 2; // 2 pack/unpack ops at cost 2 each
        savings - overhead
    }
}

impl Default for GenericCostModel {
    fn default() -> Self {
        Self::new()
    }
}

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

    /// schedule.py:325 (costmodel.py): record savings from vectorizing a pack.
    pub fn record_pack_savings(&mut self, _pack: &Pack, _numops: usize) {
        // costmodel.py: GenericCostModel.record_pack_savings()
        // Accumulates savings for profitability decision.
        // In majit, is_profitable() does the calculation directly.
    }

    /// Estimate whether vectorizing a group is profitable.
    pub fn is_profitable(&self, group: &Pack) -> bool {
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

// ── schedule.py:584-779: VecScheduleState ─────────────────────

/// schedule.py:584-779: State for vector-aware instruction scheduling.
/// Tracks which scalar ops have been mapped to vector ops, handles
/// pack/unpack/expand operations, and manages the output op list.
pub struct VecScheduleState {
    /// Map from scalar OpRef → (index_in_vector, vector OpRef).
    pub box_to_vbox: HashMap<OpRef, (usize, OpRef)>,
    /// Output operations (vector + remaining scalar).
    pub oplist: Vec<Op>,
    /// Renamer for SSA fixup during vectorization.
    pub renamer: super::renamer::Renamer,
    /// Cost model for profitability analysis.
    pub costmodel: CostModel,
    /// schedule.py:587-588: expanded_map — tracks expanded scalars.
    pub expanded_map: HashMap<OpRef, Vec<(OpRef, i32)>>,
    /// schedule.py:591: inputargs of the loop label.
    pub inputargs: HashMap<OpRef, ()>,
    /// schedule.py:595: accumulation info.
    pub accumulation: HashMap<OpRef, usize>,
    /// Next OpRef counter for newly created vector ops.
    next_pos: u32,
    /// Type registry: OpRef → is_float for vector ops.
    /// Registered at creation time (before append_to_oplist),
    /// so is_float_vector works even before the op is in oplist.
    vec_type_registry: HashMap<OpRef, bool>,
}

impl VecScheduleState {
    pub fn new(start_pos: u32) -> Self {
        VecScheduleState {
            box_to_vbox: HashMap::new(),
            oplist: Vec::new(),
            renamer: super::renamer::Renamer::new(),
            costmodel: CostModel::new(),
            expanded_map: HashMap::new(),
            inputargs: HashMap::new(),
            accumulation: HashMap::new(),
            next_pos: start_pos,
            vec_type_registry: HashMap::new(),
        }
    }

    /// Allocate a fresh OpRef for a newly created vector op.
    pub fn alloc_op_pos(&mut self) -> OpRef {
        let pos = OpRef(self.next_pos);
        self.next_pos += 1;
        pos
    }

    /// Register a vector op's datatype at creation time.
    pub fn register_vec_type(&mut self, opref: OpRef, is_float: bool) {
        self.vec_type_registry.insert(opref, is_float);
    }

    /// Check if an OpRef refers to a float-type vector op.
    /// Uses the type registry (populated at op creation, before append_to_oplist).
    pub fn is_float_vector(&self, opref: OpRef) -> bool {
        self.vec_type_registry.get(&opref).copied().unwrap_or(false)
    }

    /// schedule.py:625-630: setvector_of_box — record that scalar_op
    /// is at index `idx` in the vector `vecop`.
    pub fn setvector_of_box(&mut self, scalar_op: OpRef, idx: usize, vecop: OpRef) {
        self.box_to_vbox.insert(scalar_op, (idx, vecop));
    }

    /// schedule.py:632-638: getvector_of_box — look up which vector
    /// op contains the scalar op.
    pub fn getvector_of_box(&self, scalar_op: OpRef) -> Option<(usize, OpRef)> {
        self.box_to_vbox.get(&scalar_op).copied()
    }

    /// schedule.py:640-650: append to output.
    pub fn append_to_oplist(&mut self, op: Op) {
        self.oplist.push(op);
    }
}

// ── schedule.py:317-400: turn_into_vector and helpers ─────────────────────

/// schedule.py:317-320: failnbail_transformation
pub struct NotAVectorizeableLoop;
pub struct NotAProfitableLoop;

/// schedule.py:462-474: check_if_pack_supported — validate pack constraints.
pub fn check_if_pack_supported(pack: &Pack, ops: &[Op]) -> Result<(), NotAProfitableLoop> {
    let first_op = &ops[pack.members[0]];
    // schedule.py:471-474: INT_MUL with bytesize 8 or 1 is not profitable
    if first_op.opcode == OpCode::IntMul {
        if let Some(ref vi) = first_op.vecinfo {
            let insize = vi.getbytesize();
            if insize == 8 || insize == 1 {
                return Err(NotAProfitableLoop);
            }
        }
    }
    Ok(())
}

/// schedule.py:476-486: unpack_from_vector — extract a scalar from a vector box.
/// Creates a VecUnpack op with the correct type (I or F) based on the
/// vector box's datatype. Mirrors OpHelpers.create_vec_unpack(var.type, ...).
pub fn unpack_from_vector(
    state: &mut VecScheduleState,
    vec_ref: OpRef,
    index: usize,
    count: usize,
) -> OpRef {
    assert!(count > 0);
    let index_const = OpRef(OpRef::CONST_BASE + index as u32);
    let count_const = OpRef(OpRef::CONST_BASE + count as u32);
    // schedule.py:482: create_vec_unpack(arg.type, ...) — pick opcode by type
    let unpack_opcode = if state.is_float_vector(vec_ref) {
        OpCode::VecUnpackF
    } else {
        OpCode::VecUnpackI
    };
    let mut unpack_op = Op::new(unpack_opcode, &[vec_ref, index_const, count_const]);
    unpack_op.pos = state.alloc_op_pos();
    let result = unpack_op.pos;
    state.append_to_oplist(unpack_op);
    result
}

/// schedule.py:388-400: prepare_fail_arguments — process guard failargs
/// for vectorized guard ops, unpacking vector boxes to scalar.
pub fn prepare_fail_arguments(
    state: &mut VecScheduleState,
    pack: &Pack,
    ops: &[Op],
    vecop: &mut Op,
) {
    let first_op = &ops[pack.members[0]];
    if !first_op.opcode.is_guard() {
        return;
    }
    if let Some(ref fail_args) = first_op.fail_args {
        let mut new_fail_args = fail_args.clone();
        for arg in new_fail_args.iter_mut() {
            // schedule.py:393-394: look up if arg is in a vector box
            let (pos, newarg) = state.getvector_of_box(*arg).unwrap_or((0, *arg));
            if newarg != *arg {
                // schedule.py:396-397: vector box → unpack at position 0
                let unpacked = unpack_from_vector(state, newarg, 0, 1);
                *arg = unpacked;
            }
        }
        vecop.fail_args = Some(new_fail_args);
    }
}

/// schedule.py:352-386: prepare_arguments — transform scalar args to vector args.
/// Handles reuse, crop, scatter, position, and expand cases.
/// Requires CPU vector extension restrictions; currently a no-op since majit
/// does not yet have a vector extension backend.
/// When oprestrict is None (no restriction), RPython returns immediately (line 364-365).
pub fn prepare_arguments(
    _state: &mut VecScheduleState,
    _pack: &Pack,
    _args: &mut Vec<OpRef>,
    _ops: &[Op],
) {
    // schedule.py:364-365: if not oprestrict: return
    // majit does not yet have CPU vector extension restrictions,
    // so this is equivalent to oprestrict=None → early return.
}

/// schedule.py:322-350: Turn a pack of scalar ops into a single vector op.
pub fn turn_into_vector(state: &mut VecScheduleState, pack: &Pack, ops: &[Op]) {
    if pack.members.is_empty() {
        return;
    }
    // schedule.py:324: check_if_pack_supported
    if check_if_pack_supported(pack, ops).is_err() {
        return;
    }
    let count = pack.members.len();
    let first_op = &ops[pack.members[0]];

    // schedule.py:325: costmodel.record_pack_savings
    state.costmodel.record_pack_savings(pack, count);

    let Some(vec_opcode) = first_op.opcode.to_vector() else {
        return; // not vectorizable
    };

    // schedule.py:335-336: build args list + prepare_arguments
    let mut args = first_op.args.to_vec();
    prepare_arguments(state, pack, &mut args, ops);

    // schedule.py:337-338: create VecOperation
    let mut vecop = Op::new(vec_opcode, &args);
    vecop.pos = state.alloc_op_pos();
    vecop.descr = first_op.descr.clone();
    let datatype = if first_op.opcode.result_type() == majit_ir::Type::Float {
        'f'
    } else {
        'i'
    };
    let mut vinfo = majit_ir::VectorizationInfo::new();
    vinfo.count = count as i16;
    vinfo.setinfo(datatype, -1, datatype == 'i');
    vecop.vecinfo = Some(Box::new(vinfo));

    let vecop_pos = vecop.pos;
    // Register type before append_to_oplist so is_float_vector works
    // in prepare_fail_arguments (called before append).
    state.register_vec_type(vecop_pos, datatype == 'f');
    // schedule.py:340-346: map scalar ops to vector positions
    for (i, &member_idx) in pack.members.iter().enumerate() {
        let op = &ops[member_idx];
        if op.opcode.result_type() == majit_ir::Type::Void {
            continue; // schedule.py:342-343: skip void ops
        }
        let scalar_pos = op.pos;
        if !scalar_pos.is_none() {
            state.setvector_of_box(scalar_pos, i, vecop_pos);
            // schedule.py:345-346: only rename for accumulating packs
            if pack.is_accumulating && !op.opcode.is_guard() {
                state.renamer.start_renaming(scalar_pos, vecop_pos);
            }
        }
    }

    // schedule.py:347-348: handle guard failargs
    if first_op.opcode.is_guard() {
        prepare_fail_arguments(state, pack, ops, &mut vecop);
    }

    state.append_to_oplist(vecop);
    assert!(count >= 1); // schedule.py:350
}
