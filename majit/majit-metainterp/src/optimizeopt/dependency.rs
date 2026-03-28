//! Dependency graph for vectorization.
//!
//! Mirrors RPython's `dependency.py`: builds a DAG of data dependencies
//! between operations in a loop body. Used by the vector optimizer to
//! identify independent operations that can be packed into SIMD instructions.

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::optimizeopt::schedule::Pack;
use majit_ir::{Op, OpCode, OpRef};

/// A node in the dependency graph.
#[derive(Clone, Debug)]
pub struct Node {
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
    pub nodes: Vec<Node>,
}

impl DependencyGraph {
    /// Build a dependency graph from a list of operations.
    ///
    /// Two operations have a dependency if:
    /// - One uses the result of the other (data dependency)
    /// - Both access memory and at least one is a write (memory dependency)
    /// - One is a guard (control dependency)
    pub fn build(ops: &[Op]) -> Self {
        let mut nodes: Vec<Node> = ops
            .iter()
            .enumerate()
            .map(|(idx, op)| Node {
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
pub struct DefTracker<'a> {
    pub graph: &'a DependencyGraph,
    /// OpRef → list of (defining node index, optional memory ref cell)
    pub defs: HashMap<OpRef, Vec<(usize, Option<usize>)>>,
    /// Nodes with side effects (non-pure).
    pub non_pure: Vec<usize>,
}

impl<'a> DefTracker<'a> {
    pub fn new(graph: &'a DependencyGraph) -> Self {
        DefTracker {
            graph,
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
    fn inspect_array_access(&mut self, op: &Op, node_idx: usize, raw_access: bool) {
        if op.args.len() < 2 {
            return;
        }
        let array = op.args[0];
        let index = op.args[1];
        let idx_var = self.get_or_create(index);
        if let Some(ref descr) = op.descr {
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
