//! Dependency graph for vectorization.
//!
//! Mirrors RPython's `dependency.py`: builds a DAG of data dependencies
//! between operations in a loop body. Used by the vector optimizer to
//! identify independent operations that can be packed into SIMD instructions.

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::optimizeopt::schedule::Pack;
use majit_ir::{Op, OpCode, OpRef};

// ── dependency.py:15-50: LOAD/MODIFY_COMPLEX_OBJ tables ─────────

/// dependency.py:30-48: LOAD_COMPLEX_OBJ — returns (complex_obj_arg_idx, index_arg_idx).
/// index_arg_idx == -1 means no index argument (field access, not array).
fn load_complex_obj_args(opcode: OpCode) -> (usize, i32) {
    match opcode {
        // Array loads: (array, index)
        OpCode::GetarrayitemGcI
        | OpCode::GetarrayitemGcF
        | OpCode::GetarrayitemGcR
        | OpCode::GetarrayitemRawI
        | OpCode::GetarrayitemRawF
        | OpCode::RawLoadI
        | OpCode::RawLoadF => (0, 1),
        // Interior field: (obj, index)
        OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcF | OpCode::GetinteriorfieldGcR => {
            (0, 1)
        }
        // Field loads: (obj, no index)
        OpCode::GetfieldGcI
        | OpCode::GetfieldGcR
        | OpCode::GetfieldGcF
        | OpCode::GetfieldRawI
        | OpCode::GetfieldRawR
        | OpCode::GetfieldRawF => (0, -1),
        _ => (0, -1),
    }
}

/// dependency.py:15-26: MODIFY_COMPLEX_OBJ — returns (complex_obj_arg_idx, cell_arg_idx).
/// cell_arg_idx == -1 means no cell argument (field store, not array).
fn modify_complex_obj_args(opcode: OpCode) -> Option<(usize, i32)> {
    match opcode {
        // Array stores: (array, index)
        OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw | OpCode::RawStore => Some((0, 1)),
        // Interior field stores: (obj, no cell)
        OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => Some((0, -1)),
        // Field stores: (obj, no cell)
        OpCode::SetfieldGc | OpCode::SetfieldRaw => Some((0, -1)),
        // Other
        OpCode::ZeroArray => Some((0, -1)),
        OpCode::Strsetitem | OpCode::Unicodesetitem => Some((0, -1)),
        _ => None,
    }
}

/// dependency.py:213-241: side_effect_arguments — determine which args are
/// destroyed (modified) by the operation. Returns Vec<(arg, argcell, destroyed)>.
/// `arg_type_of` resolves an OpRef to its result type for the float check.
fn side_effect_arguments(
    op: &Op,
    arg_type_of: &dyn Fn(OpRef) -> majit_ir::Type,
) -> Vec<(OpRef, Option<OpRef>, bool)> {
    let mut result = Vec::new();
    if op.opcode.is_complex_modify() {
        // dependency.py:218-230: known complex modification patterns
        if let Some((obj_idx, cell_idx)) = modify_complex_obj_args(op.opcode) {
            if obj_idx < op.args.len() {
                if cell_idx >= 0 && (cell_idx as usize) < op.args.len() {
                    result.push((op.args[obj_idx], Some(op.args[cell_idx as usize]), true));
                    for j in (cell_idx as usize + 1)..op.args.len() {
                        result.push((op.args[j], None, false));
                    }
                } else {
                    result.push((op.args[obj_idx], None, true));
                    for j in (obj_idx + 1)..op.args.len() {
                        result.push((op.args[j], None, false));
                    }
                }
            }
        }
    } else {
        // dependency.py:232-240: generic side effect
        for arg in &op.args {
            // dependency.py:237: arg.is_constant() or arg.type == 'f' → not destroyed
            if arg.is_constant() || arg_type_of(*arg) == majit_ir::Type::Float {
                result.push((*arg, None, false));
            } else {
                result.push((*arg, None, true));
            }
        }
    }
    result
}

/// dependency.py:131-300: A node in the dependency graph.
/// Each node wraps one operation and maintains forward/backward dependency edges.
#[derive(Clone, Debug)]
pub struct Node {
    /// Index in the ops list (dependency.py:134: opidx).
    pub idx: usize,
    /// The operation (dependency.py:133: op).
    pub op: Op,
    /// dependency.py:135: adjacent_list — forward dependency edges (this → target).
    pub adjacent_list: Vec<Dependency>,
    /// dependency.py:136: adjacent_list_back — backward dependency edges (source → this).
    pub adjacent_list_back: Vec<Dependency>,
    /// dependency.py:137: memory_ref — MemoryRef for array access ops.
    pub memory_ref: Option<MemoryRef>,
    /// dependency.py:138: pack — which Pack this node belongs to.
    pub pack: Option<usize>,
    /// dependency.py:139: pack_position
    pub pack_position: i32,
    /// dependency.py:140: emitted — whether this node has been scheduled.
    pub emitted: bool,
    /// dependency.py:141: schedule_position
    pub schedule_position: i32,
    /// dependency.py:142: priority — scheduling priority.
    pub priority: i32,
    /// Compat: indices of operations this one depends on.
    pub deps: Vec<usize>,
    /// Compat: indices of operations that depend on this one.
    pub users: Vec<usize>,
}

impl Node {
    pub fn new(op: Op, opidx: usize) -> Self {
        Node {
            idx: opidx,
            op,
            adjacent_list: Vec::new(),
            adjacent_list_back: Vec::new(),
            memory_ref: None,
            pack: None,
            pack_position: -1,
            emitted: false,
            schedule_position: -1,
            priority: 0,
            deps: Vec::new(),
            users: Vec::new(),
        }
    }

    /// dependency.py:161: setpriority
    pub fn setpriority(&mut self, value: i32) {
        self.priority = value;
    }

    /// dependency.py:243: provides_count
    pub fn provides_count(&self) -> usize {
        self.adjacent_list.len()
    }

    /// dependency.py:249: depends_count
    pub fn depends_count(&self) -> usize {
        self.adjacent_list_back.len()
    }

    /// dependency.py:268: is_after
    pub fn is_after(&self, other_idx: usize) -> bool {
        self.idx > other_idx
    }

    /// dependency.py:271: is_before
    pub fn is_before(&self, other_idx: usize) -> bool {
        self.idx < other_idx
    }

    /// dependency.py:167: is_pure
    pub fn is_pure(&self) -> bool {
        self.op.opcode.is_always_pure()
    }

    /// dependency.py:201-205: exits_early
    pub fn exits_early(&self) -> bool {
        if self.op.opcode.is_guard() {
            // In RPython, descr.exits_early(). We check for GUARD_FUTURE_CONDITION etc.
            matches!(self.op.opcode, OpCode::GuardFutureCondition)
        } else {
            false
        }
    }

    /// dependency.py:207-208: loads_from_complex_object
    pub fn loads_from_complex_object(&self) -> bool {
        self.op.opcode.is_complex_load()
    }

    /// dependency.py:210-211: modifies_complex_object
    pub fn modifies_complex_object(&self) -> bool {
        self.op.opcode.is_complex_modify()
    }
}

/// dependency.py:537: DependencyGraph — dependency graph for a loop body.
#[derive(Clone, Debug)]
pub struct DependencyGraph {
    pub nodes: Vec<Node>,
    /// dependency.py:567: memory_refs — node index → MemoryRef
    pub memory_refs: HashMap<usize, MemoryRef>,
    /// dependency.py:569: index_vars — OpRef → IndexVar
    pub index_vars: HashMap<OpRef, IndexVar>,
    /// dependency.py:571: guards — guard node indices
    pub guards: Vec<usize>,
    /// dependency.py:565: invariant_vars — loop-invariant variables
    pub invariant_vars: HashMap<OpRef, ()>,
}

impl DependencyGraph {
    /// dependency.py:556-572: Build a dependency graph from loop operations.
    /// Uses DefTracker and IntegralForwardModification for precise analysis.
    pub fn build(ops: &[Op], constant_of: &dyn Fn(OpRef) -> Option<i64>) -> Self {
        let mut nodes: Vec<Node> = ops
            .iter()
            .enumerate()
            .map(|(idx, op)| Node::new(op.clone(), idx))
            .collect();

        let mut graph = DependencyGraph {
            nodes,
            memory_refs: HashMap::new(),
            index_vars: HashMap::new(),
            guards: Vec::new(),
            invariant_vars: HashMap::new(),
        };

        graph.build_dependencies(ops, constant_of);
        graph
    }

    /// dependency.py:596-644: build_dependencies — construct def-use chains
    /// with DefTracker and IntegralForwardModification.
    fn build_dependencies(&mut self, ops: &[Op], constant_of: &dyn Fn(OpRef) -> Option<i64>) {
        let mut tracker = DefTracker::new(self);
        let mut intformod = IntegralForwardModification::new(constant_of);

        for i in 0..self.nodes.len() {
            let op = &self.nodes[i].op.clone();

            // dependency.py:613-616: set priority for pure/guard ops
            if op.opcode.is_always_pure() {
                self.nodes[i].setpriority(1);
            }
            if op.opcode.is_guard() {
                self.nodes[i].setpriority(2);
            }

            // dependency.py:620: inspect for index variables and memory refs
            intformod.inspect_operation(op, i);
            if let Some(mref) = intformod.memory_refs.get(&i) {
                self.nodes[i].memory_ref = Some(mref.clone());
                self.memory_refs.insert(i, mref.clone());
            }

            // dependency.py:622-624: define result variable
            if op.opcode.result_type() != majit_ir::Type::Void {
                tracker.define(op.pos, i);
            }

            // dependency.py:626-644: build edges based on op type
            if op.opcode.is_always_pure() || op.opcode.is_final() {
                // dependency.py:628-629: pure/final — depend on all args
                let args: Vec<OpRef> = op.args.to_vec();
                for arg in &args {
                    Self::depends_on_arg_static(&tracker, *arg, i, &mut self.nodes);
                }
            } else if op.opcode.is_guard() {
                // dependency.py:630-642: guard dependencies
                if !self.nodes[i].exits_early() {
                    // dependency.py:635-640: guard ordering + non-pure deps
                    if !self.guards.is_empty() {
                        let last_guard = *self.guards.last().unwrap();
                        Self::add_edge(&mut self.nodes, last_guard, i, None, true);
                    }
                    for &np_idx in &tracker.non_pure.clone() {
                        Self::add_edge(&mut self.nodes, np_idx, i, None, true);
                    }
                    tracker.non_pure.clear();
                }
                self.guards.push(i);
                // dependency.py:642: build_guard_dependencies
                self.build_guard_dependencies(i, &mut tracker, ops);
            } else {
                // dependency.py:644: non-pure (memory side effects)
                self.build_non_pure_dependencies(i, &mut tracker, ops);
            }
        }

        // Copy index_vars from intformod
        self.index_vars = intformod.index_vars;
    }

    /// dependency.py:708-735: build_guard_dependencies
    fn build_guard_dependencies(&mut self, guard_idx: usize, tracker: &mut DefTracker, ops: &[Op]) {
        let op = self.nodes[guard_idx].op.clone();
        // dependency.py:710-712: ignore invalidated & future condition & early exit guards
        if matches!(
            op.opcode,
            OpCode::GuardFutureCondition | OpCode::GuardAlwaysFails | OpCode::GuardNotInvalidated
        ) {
            return;
        }
        // dependency.py:714-715: true dependencies on args
        for arg in &op.args.to_vec() {
            Self::depends_on_arg_static(tracker, *arg, guard_idx, &mut self.nodes);
        }
        // dependency.py:717: guard_argument_protection
        self.guard_argument_protection(guard_idx, tracker);
        // dependency.py:719-721: descr.exits_early() check
        if self.nodes[guard_idx].exits_early() {
            return;
        }
        // dependency.py:723-735: fail_args dependencies — iterate ALL redefinitions
        if let Some(ref fail_args) = op.fail_args {
            let fa = fail_args.to_vec();
            for arg in &fa {
                if arg.is_none() {
                    continue;
                }
                if !tracker.is_defined(*arg) {
                    continue;
                }
                // dependency.py:730-733: for at in tracker.redefinitions(arg)
                let redefs = tracker.redefinitions(*arg);
                for at_idx in redefs {
                    if self.nodes[at_idx].is_before(guard_idx) {
                        Self::add_edge(&mut self.nodes, at_idx, guard_idx, Some(*arg), true);
                    }
                }
            }
        }
    }

    /// dependency.py:646-698: guard_argument_protection
    fn guard_argument_protection(&mut self, guard_idx: usize, tracker: &mut DefTracker) {
        let op = self.nodes[guard_idx].op.clone();
        // dependency.py:657-664: redefine non-constant, non-int, non-float args (pointers)
        for arg in &op.args.to_vec() {
            if arg.is_constant() || arg.is_none() {
                continue;
            }
            // dependency.py:658: arg.type not in ('i','f')
            // Look up the defining op's result type to determine arg type.
            let arg_type = tracker
                .definition(*arg)
                .map(|def_idx| self.nodes[def_idx].op.opcode.result_type())
                .unwrap_or(majit_ir::Type::Ref); // unknown → assume ref (conservative)
            if arg_type != majit_ir::Type::Int && arg_type != majit_ir::Type::Float {
                tracker.define(*arg, guard_idx);
            }
        }
        // dependency.py:665-698: special guard priorities
        match op.opcode {
            OpCode::GuardNotForced2 => {
                self.nodes[guard_idx].setpriority(-10);
            }
            OpCode::GuardOverflow | OpCode::GuardNoOverflow => {
                self.nodes[guard_idx].setpriority(100);
                // Find preceding overflow operation
                let mut j = guard_idx;
                while j > 0 {
                    j -= 1;
                    if self.nodes[j].op.opcode.is_ovf() {
                        Self::add_edge(&mut self.nodes, j, guard_idx, None, false);
                        break;
                    }
                }
            }
            OpCode::GuardNoException | OpCode::GuardException | OpCode::GuardNotForced => {
                self.nodes[guard_idx].setpriority(100);
                // Find preceding can-raise operation
                let mut j = guard_idx;
                while j > 0 {
                    j -= 1;
                    if self.nodes[j].op.opcode.can_raise() || self.nodes[j].op.opcode.is_guard() {
                        Self::add_edge(&mut self.nodes, j, guard_idx, None, false);
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    /// dependency.py:737-784: build_non_pure_dependencies
    fn build_non_pure_dependencies(
        &mut self,
        node_idx: usize,
        tracker: &mut DefTracker,
        _ops: &[Op],
    ) {
        let op = self.nodes[node_idx].op.clone();

        if self.nodes[node_idx].loads_from_complex_object() {
            // dependency.py:742-751: LOAD_COMPLEX_OBJ dispatch
            // (opnum, complex_obj_arg_idx, index_arg_idx)
            let (cobj_idx, index_idx) = load_complex_obj_args(op.opcode);
            if cobj_idx < op.args.len() {
                let cobj = op.args[cobj_idx];
                if index_idx >= 0 && (index_idx as usize) < op.args.len() {
                    // dependency.py:747-748: argcell-aware depends_on
                    let index_var = op.args[index_idx as usize];
                    Self::depends_on_arg_static(tracker, cobj, node_idx, &mut self.nodes);
                    Self::depends_on_arg_static(tracker, index_var, node_idx, &mut self.nodes);
                } else {
                    // dependency.py:750: no index arg
                    Self::depends_on_arg_static(tracker, cobj, node_idx, &mut self.nodes);
                }
            }
        } else {
            // dependency.py:752-777: side_effect_arguments processing
            let nodes_ref = &self.nodes;
            let arg_type_of = |opref: OpRef| -> majit_ir::Type {
                // Look up the defining op's result type
                nodes_ref
                    .iter()
                    .find(|n| n.op.pos == opref)
                    .map(|n| n.op.opcode.result_type())
                    .unwrap_or(majit_ir::Type::Int)
            };
            let side_effects = side_effect_arguments(&op, &arg_type_of);
            for (arg, argcell, destroyed) in &side_effects {
                if let Some(cell) = argcell {
                    // dependency.py:754-757: exact cell tracking
                    Self::depends_on_arg_static(tracker, *arg, node_idx, &mut self.nodes);
                    Self::depends_on_arg_static(tracker, *cell, node_idx, &mut self.nodes);
                } else if *destroyed {
                    // dependency.py:759-772: WAR/WAW dependencies
                    if let Some(def_idx) = tracker.definition(*arg) {
                        // dependency.py:767-769: war edges from def's users
                        let provides: Vec<usize> = self.nodes[def_idx]
                            .adjacent_list
                            .iter()
                            .map(|d| d.to_idx)
                            .collect();
                        for to in provides {
                            if to != node_idx {
                                Self::add_edge(&mut self.nodes, to, node_idx, *argcell, false);
                            }
                        }
                        // dependency.py:770: def_node.edge_to(node)
                        Self::add_edge(&mut self.nodes, def_idx, node_idx, *argcell, false);
                    }
                } else {
                    // dependency.py:774-775: normal use
                    Self::depends_on_arg_static(tracker, *arg, node_idx, &mut self.nodes);
                }
                if *destroyed {
                    // dependency.py:776-777: redefine
                    tracker.define(*arg, node_idx);
                }
            }

            // dependency.py:780-782: non-pure must follow last guard
            if !self.guards.is_empty() {
                let last_guard = *self.guards.last().unwrap();
                Self::add_edge(&mut self.nodes, last_guard, node_idx, None, false);
            }
            // dependency.py:784: track as non-pure
            tracker.add_non_pure(node_idx);
        }
    }

    /// Helper: add a dependency edge between two nodes (dependency.py:170-195 Node.edge_to).
    fn add_edge(
        nodes: &mut Vec<Node>,
        from_idx: usize,
        to_idx: usize,
        arg: Option<OpRef>,
        failarg: bool,
    ) {
        if from_idx == to_idx {
            return;
        }
        // Check if edge already exists
        let existing = nodes[from_idx]
            .adjacent_list
            .iter()
            .position(|d| d.to_idx == to_idx);
        if let Some(pos) = existing {
            // dependency.py:186-194: update existing edge
            if let Some(a) = arg {
                if !nodes[from_idx].adjacent_list[pos].because_of(a) {
                    nodes[from_idx].adjacent_list[pos].args.push((from_idx, a));
                }
            }
            if !(nodes[from_idx].adjacent_list[pos].failarg && failarg) {
                nodes[from_idx].adjacent_list[pos].failarg = false;
            }
        } else {
            // dependency.py:176-180: create new edge + backward edge
            let dep = Dependency::new(from_idx, to_idx, arg);
            nodes[from_idx].adjacent_list.push(dep);
            let dep_back = Dependency::new(to_idx, from_idx, arg);
            nodes[to_idx].adjacent_list_back.push(dep_back);
            // Compat: update deps/users
            if !nodes[to_idx].deps.contains(&from_idx) {
                nodes[to_idx].deps.push(from_idx);
                nodes[from_idx].users.push(to_idx);
            }
        }
    }

    /// Helper: depends_on_arg using DefTracker (works with &mut nodes borrow).
    fn depends_on_arg_static(
        tracker: &DefTracker,
        arg: OpRef,
        to_idx: usize,
        nodes: &mut Vec<Node>,
    ) {
        if let Some(at_idx) = tracker.definition(arg) {
            if at_idx != to_idx {
                // Inline add_edge logic to avoid double borrow issues
                let existing = nodes[at_idx]
                    .adjacent_list
                    .iter()
                    .position(|d| d.to_idx == to_idx);
                if let Some(pos) = existing {
                    if !nodes[at_idx].adjacent_list[pos].because_of(arg) {
                        nodes[at_idx].adjacent_list[pos].args.push((at_idx, arg));
                    }
                } else {
                    let dep = Dependency::new(at_idx, to_idx, Some(arg));
                    nodes[at_idx].adjacent_list.push(dep);
                    let dep_back = Dependency::new(to_idx, at_idx, Some(arg));
                    nodes[to_idx].adjacent_list_back.push(dep_back);
                    if !nodes[to_idx].deps.contains(&at_idx) {
                        nodes[to_idx].deps.push(at_idx);
                        nodes[at_idx].users.push(to_idx);
                    }
                }
            }
        }
    }

    /// Find groups of independent, isomorphic operations that can be packed.
    ///
    /// Two ops are "isomorphic" if they have the same opcode and their
    /// args come from independent sources (no data dependency between them).
    pub fn find_packable_groups(&self) -> Vec<Pack> {
        let mut groups: Vec<Pack> = Vec::new();
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
                groups.push(Pack {
                    scalar_opcode: *opcode,
                    vector_opcode: vec_opcode,
                    members: group_indices,
                    is_accumulating: false,
                    position: -1,
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

// ── dependency.py:981-1138: IndexVar ──────────────────────────

/// dependency.py:981-1093: Linear combination of an index variable.
/// Represents `var * (coefficient_mul / coefficient_div) + constant`.
#[derive(Clone, Debug)]
pub struct IndexVar {
    /// The base SSA variable.
    pub var: OpRef,
    /// Multiplicative coefficient (numerator).
    pub coefficient_mul: i64,
    /// Divisive coefficient (denominator).
    pub coefficient_div: i64,
    /// Additive constant.
    pub constant: i64,
}

impl IndexVar {
    pub fn new(var: OpRef) -> Self {
        IndexVar {
            var,
            coefficient_mul: 1,
            coefficient_div: 1,
            constant: 0,
        }
    }

    /// dependency.py:1042-1044
    pub fn same_variable(&self, other: &IndexVar) -> bool {
        self.var == other.var
    }

    /// dependency.py:1046-1058
    pub fn same_mulfactor(&self, other: &IndexVar) -> bool {
        if self.coefficient_mul == other.coefficient_mul
            && self.coefficient_div == other.coefficient_div
        {
            return true;
        }
        let selfmod = self.coefficient_mul % self.coefficient_div;
        let othermod = other.coefficient_mul % other.coefficient_div;
        if selfmod == 0 && othermod == 0 {
            let selfdiv = self.coefficient_mul / self.coefficient_div;
            let otherdiv = other.coefficient_mul / other.coefficient_div;
            return selfdiv == otherdiv;
        }
        false
    }

    /// dependency.py:1060-1063
    pub fn constant_diff(&self, other: &IndexVar) -> i64 {
        self.constant - other.constant
    }

    /// dependency.py:1030-1033
    pub fn is_identity(&self) -> bool {
        self.coefficient_mul == 1 && self.coefficient_div == 1 && self.constant == 0
    }

    /// dependency.py:1035-1040
    pub fn clone_var(&self) -> Self {
        IndexVar {
            var: self.var,
            coefficient_mul: self.coefficient_mul,
            coefficient_div: self.coefficient_div,
            constant: self.constant,
        }
    }
}

// ── dependency.py:1140-1220: MemoryRef ────────────────────────

/// dependency.py:1140-1220: A memory reference to an array object.
/// Tracks the array pointer, descriptor, and index variable (linear
/// combination) for adjacent-memory analysis.
#[derive(Clone, Debug)]
pub struct MemoryRef {
    /// The array pointer (op.getarg(0))
    pub array: OpRef,
    /// The array descriptor
    pub descr: majit_ir::DescrRef,
    /// The index as a linear combination
    pub index_var: IndexVar,
    /// Whether this is a raw (byte-level) access
    pub raw_access: bool,
}

impl MemoryRef {
    pub fn new(array: OpRef, descr: majit_ir::DescrRef, index_var: IndexVar) -> Self {
        MemoryRef {
            array,
            descr,
            index_var,
            raw_access: false,
        }
    }

    /// dependency.py:1158-1167: symmetric adjacency check
    pub fn is_adjacent_to(&self, other: &MemoryRef) -> bool {
        if !self.same_array(other) {
            return false;
        }
        if !self.index_var.same_variable(&other.index_var) {
            return false;
        }
        if !self.index_var.same_mulfactor(&other.index_var) {
            return false;
        }
        let stride = self.stride();
        self.index_var
            .constant_diff(&other.index_var)
            .abs()
            .saturating_sub(stride)
            == 0
    }

    /// dependency.py:1169-1178: asymmetric adjacency (self is after other)
    pub fn is_adjacent_after(&self, other: &MemoryRef) -> bool {
        if !self.same_array(other) {
            return false;
        }
        if !self.index_var.same_variable(&other.index_var) {
            return false;
        }
        if !self.index_var.same_mulfactor(&other.index_var) {
            return false;
        }
        let stride = self.stride();
        other.index_var.constant_diff(&self.index_var) == stride
    }

    /// dependency.py:1180-1194: alias check
    pub fn alias(&self, other: &MemoryRef) -> bool {
        if !self.same_array(other) {
            return false;
        }
        if !self.index_var.same_variable(&other.index_var) {
            return true;
        }
        if !self.index_var.same_mulfactor(&other.index_var) {
            return true;
        }
        self.index_var.constant_diff(&other.index_var).abs() < self.stride()
    }

    /// dependency.py:1196-1197: same_array — array identity + descriptor equality.
    /// RPython uses `self.descr == other.descr` (value equality).
    /// In majit, Descr is a trait object; we compare by index() for value equality,
    /// falling back to Arc::ptr_eq for descriptors without assigned indices.
    pub fn same_array(&self, other: &MemoryRef) -> bool {
        if self.array != other.array {
            return false;
        }
        let si = self.descr.index();
        let oi = other.descr.index();
        if si != u32::MAX && oi != u32::MAX {
            si == oi
        } else {
            std::sync::Arc::ptr_eq(&self.descr, &other.descr)
        }
    }

    /// dependency.py:1213-1217: stride in elements (1) or bytes (for raw)
    pub fn stride(&self) -> i64 {
        if !self.raw_access {
            1
        } else {
            self.descr
                .as_array_descr()
                .map(|ad| ad.item_size() as i64)
                .unwrap_or(8)
        }
    }
}

// ── dependency.py:412-471: Dependency (rich edge) ─────────────

/// dependency.py:412-471: A dependency edge in the graph.
/// Carries which args caused the dependency and whether it's a failarg dep.
#[derive(Clone, Debug)]
pub struct Dependency {
    /// Index of the source node.
    pub at_idx: usize,
    /// Index of the target node.
    pub to_idx: usize,
    /// (source_node_idx, arg OpRef) pairs that caused this dependency.
    pub args: Vec<(usize, OpRef)>,
    /// Whether this is a failarg dependency.
    pub failarg: bool,
}

impl Dependency {
    pub fn new(at_idx: usize, to_idx: usize, arg: Option<OpRef>) -> Self {
        let mut d = Dependency {
            at_idx,
            to_idx,
            args: Vec::new(),
            failarg: false,
        };
        if let Some(a) = arg {
            d.args.push((at_idx, a));
        }
        d
    }

    /// dependency.py:423-427: because_of
    pub fn because_of(&self, var: OpRef) -> bool {
        self.args.iter().any(|(_, a)| *a == var)
    }
}

// ── dependency.py:473-535: DefTracker ─────────────────────────

/// dependency.py:473-535: Tracks definitions of OpRefs during
/// dependency graph construction. Maps each OpRef to the node(s)
/// that define it, enabling def-use chain queries.
pub struct DefTracker {
    /// OpRef → list of (defining node index, optional memory ref cell)
    pub defs: HashMap<OpRef, Vec<(usize, Option<usize>)>>,
    /// Nodes with side effects (non-pure).
    pub non_pure: Vec<usize>,
}

impl DefTracker {
    pub fn new(_graph: &DependencyGraph) -> Self {
        DefTracker {
            defs: HashMap::new(),
            non_pure: Vec::new(),
        }
    }

    /// dependency.py:479-480
    pub fn add_non_pure(&mut self, node_idx: usize) {
        self.non_pure.push(node_idx);
    }

    /// dependency.py:482-488: define — register that node_idx defines arg.
    pub fn define(&mut self, arg: OpRef, node_idx: usize) {
        // dependency.py:483-484: skip constants.
        if arg.is_constant() {
            return;
        }
        self.defs
            .entry(arg)
            .or_insert_with(Vec::new)
            .push((node_idx, None));
    }

    /// dependency.py:490-492: redefinitions — yield all nodes defining arg.
    pub fn redefinitions(&self, arg: OpRef) -> Vec<usize> {
        self.defs
            .get(&arg)
            .map(|chain| chain.iter().map(|(idx, _)| *idx).collect())
            .unwrap_or_default()
    }

    /// dependency.py:494-495
    pub fn is_defined(&self, arg: OpRef) -> bool {
        self.defs.contains_key(&arg)
    }

    /// dependency.py:497-523: definition — find the defining node for arg.
    pub fn definition(&self, arg: OpRef) -> Option<usize> {
        if arg.is_constant() {
            return None;
        }
        let chain = self.defs.get(&arg)?;
        if chain.is_empty() {
            return None;
        }
        Some(chain.last()?.0)
    }

    /// dependency.py:525-534: depends_on_arg — add edge from definition to `to_idx`.
    pub fn depends_on_arg(&self, arg: OpRef, to_idx: usize, graph: &mut Vec<Vec<usize>>) {
        if let Some(at_idx) = self.definition(arg) {
            if at_idx != to_idx && !graph[at_idx].contains(&to_idx) {
                graph[at_idx].push(to_idx);
            }
        }
    }
}

// ── dependency.py:877-978: IntegralForwardModification ────────

/// dependency.py:877-978: Calculates integral modifications on integer
/// boxes. Propagates INT_ADD/INT_SUB/INT_MUL through IndexVar linear
/// combinations, and recognizes array access patterns for MemoryRef.
pub struct IntegralForwardModification<'a> {
    /// OpRef → IndexVar mapping
    pub index_vars: HashMap<OpRef, IndexVar>,
    /// Node index → MemoryRef mapping
    pub memory_refs: HashMap<usize, MemoryRef>,
    /// Callback to resolve constant OpRef → i64 value.
    /// dependency.py:885-888: is_const_integral + box.getint()
    constant_of: &'a dyn Fn(OpRef) -> Option<i64>,
}

impl<'a> IntegralForwardModification<'a> {
    pub fn new(constant_of: &'a dyn Fn(OpRef) -> Option<i64>) -> Self {
        IntegralForwardModification {
            index_vars: HashMap::new(),
            memory_refs: HashMap::new(),
            constant_of,
        }
    }

    fn is_const(opref: OpRef) -> bool {
        opref.is_constant()
    }

    fn const_val(&self, opref: OpRef) -> Option<i64> {
        (self.constant_of)(opref)
    }

    fn get_or_create(&mut self, arg: OpRef) -> IndexVar {
        self.index_vars
            .get(&arg)
            .cloned()
            .unwrap_or_else(|| IndexVar::new(arg))
    }

    /// dependency.py:896-920: operation_INT_ADD / operation_INT_SUB.
    fn inspect_additive(&mut self, op: &Op, is_sub: bool) {
        let result = op.pos;
        let a0 = op.args[0];
        let a1 = op.args[1];
        if Self::is_const(a0) && Self::is_const(a1) {
            let mut idx = IndexVar::new(result);
            let v0 = self.const_val(a0).unwrap_or(0);
            let v1 = self.const_val(a1).unwrap_or(0);
            idx.constant = if is_sub { v0 - v1 } else { v0 + v1 };
            self.index_vars.insert(result, idx);
        } else if Self::is_const(a0) {
            let mut idx = self.get_or_create(a1);
            idx = idx.clone_var();
            if let Some(v) = self.const_val(a0) {
                if is_sub {
                    idx.constant -= v;
                } else {
                    idx.constant += v;
                }
            }
            self.index_vars.insert(result, idx);
        } else if Self::is_const(a1) {
            let mut idx = self.get_or_create(a0);
            idx = idx.clone_var();
            if let Some(v) = self.const_val(a1) {
                if is_sub {
                    idx.constant -= v;
                } else {
                    idx.constant += v;
                }
            }
            self.index_vars.insert(result, idx);
        } else {
            // Both non-const: track the variable.
            let idx = self.get_or_create(a0);
            self.index_vars.insert(result, idx);
        }
    }

    /// dependency.py:922-948: operation_INT_MUL.
    fn inspect_multiplicative(&mut self, op: &Op) {
        let result = op.pos;
        let a0 = op.args[0];
        let a1 = op.args[1];
        if Self::is_const(a0) && Self::is_const(a1) {
            let mut idx = IndexVar::new(result);
            let v0 = self.const_val(a0).unwrap_or(0);
            let v1 = self.const_val(a1).unwrap_or(0);
            idx.constant = v0 * v1;
            self.index_vars.insert(result, idx);
        } else if Self::is_const(a0) {
            let mut idx = self.get_or_create(a1);
            idx = idx.clone_var();
            if let Some(v) = self.const_val(a0) {
                idx.coefficient_mul *= v;
                idx.constant *= v;
            }
            self.index_vars.insert(result, idx);
        } else if Self::is_const(a1) {
            let mut idx = self.get_or_create(a0);
            idx = idx.clone_var();
            if let Some(v) = self.const_val(a1) {
                idx.coefficient_mul *= v;
                idx.constant *= v;
            }
            self.index_vars.insert(result, idx);
        }
    }

    /// dependency.py:950-975: inspect array access ops.
    /// Only creates MemoryRef for primitive array accesses (dependency.py:954).
    fn inspect_array_access(&mut self, op: &Op, node_idx: usize, raw_access: bool) {
        if op.args.len() < 2 {
            return;
        }
        let array = op.args[0];
        let index = op.args[1];
        let idx_var = self.get_or_create(index);
        if let Some(ref descr) = op.descr {
            // dependency.py:954: descr.is_array_of_primitives()
            let is_prim = descr
                .as_array_descr()
                .map(|ad| ad.is_array_of_primitives())
                .unwrap_or(false);
            if !is_prim {
                return;
            }
            let mref = MemoryRef {
                array,
                descr: descr.clone(),
                index_var: idx_var,
                raw_access,
            };
            self.memory_refs.insert(node_idx, mref);
        }
    }

    /// dependency.py:977: inspect_operation dispatcher (integral_dispatch_opt)
    pub fn inspect_operation(&mut self, op: &Op, node_idx: usize) {
        match op.opcode {
            OpCode::IntAdd => self.inspect_additive(op, false),
            OpCode::IntSub => self.inspect_additive(op, true),
            OpCode::IntMul => self.inspect_multiplicative(op),
            // Array access ops
            OpCode::RawLoadI | OpCode::RawLoadF | OpCode::RawStore => {
                self.inspect_array_access(op, node_idx, true);
            }
            OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawF
            | OpCode::SetarrayitemRaw
            | OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcF
            | OpCode::SetarrayitemGc => {
                self.inspect_array_access(op, node_idx, false);
            }
            _ => {}
        }
    }
}
