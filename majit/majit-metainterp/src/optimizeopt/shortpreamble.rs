/// Short preamble: minimal operations to replay when entering a peeled loop
/// from a bridge rather than from the preamble.
///
/// Translated from rpython/jit/metainterp/optimizeopt/shortpreamble.py.
///
/// After loop peeling, the optimizer processes the peeled iteration (preamble)
/// and discovers facts about loop-carried values (constants, types, bounds).
/// The loop body is then optimized assuming those facts hold.
///
/// When a bridge later jumps to the loop header (Label), it doesn't have
/// the preamble's context. The "short preamble" is a minimal set of operations
/// — typically guards — that re-establish the facts the loop body depends on.
///
/// Without the short preamble, bridges would need to either:
/// 1. Re-execute the entire preamble (wasteful), or
/// 2. Be conservative and lose optimizations (slow)
///
/// # Structure
///
/// ```text
/// [preamble]               ← full first iteration, optimizer learns facts
///   Label(...)             ← loop header, short preamble stored here
///   [short preamble ops]   ← replayed when a bridge enters here
/// [optimized body]         ← relies on facts from preamble
///   Jump(...)              ← back-edge
/// ```
///
/// # Integration
///
/// The `ShortPreambleBuilder` is used during optimization. When the optimizer
/// processes the preamble and finds guards/operations that establish facts
/// the body depends on, it records them. At the Label, the builder finalizes
/// into a `ShortPreamble` that is stored alongside the compiled loop.
use std::collections::{HashMap, HashSet};

use majit_ir::{Op, OpCode, OpRef};

use crate::optimizeopt::virtualstate::VirtualState;

/// A recorded preamble operation that bridges must replay.
///
/// Each entry captures an operation from the preamble that was either:
/// - A guard that the body assumes always holds
/// - A pure operation whose result the body uses as a known value
/// - A type/class check that enables downstream specialization
#[derive(Clone, Debug)]
pub struct ShortPreambleOp {
    /// The operation to replay (with args referencing label arg indices).
    pub op: Op,
    /// Which label arg indices this op's arguments map to.
    /// Maps from op arg position to label arg index.
    pub arg_mapping: Vec<(usize, usize)>,
    /// Which label arg indices this op's fail args map to.
    /// Maps from fail_arg position to label arg index.
    pub fail_arg_mapping: Vec<(usize, usize)>,
}

/// The complete short preamble for a peeled loop.
///
/// Stored alongside the Label's target token. When a bridge targets
/// this label, the short preamble ops are prepended to establish
/// the optimization context the loop body expects.
#[derive(Clone, Debug)]
pub struct ShortPreamble {
    /// Operations to prepend when entering the loop from a bridge.
    /// These are guards and setup ops with args referencing label arg indices.
    pub ops: Vec<ShortPreambleOp>,
    /// Input args of the short preamble Label.
    /// RPython stores the full short preamble as [Label(short_inputargs), ...].
    pub inputargs: Vec<OpRef>,
    /// Extra loop-header values carried by the short preamble Jump.
    /// RPython appends `sb.used_boxes` to the loop label and jumps with
    /// `args + extra`, where `extra` is the remapped version of these boxes.
    pub used_boxes: Vec<OpRef>,
    /// Preamble producer results used by the short preamble's own trailing JUMP.
    /// RPython keeps this separate from `used_boxes`: the loop contract carries
    /// body boxes, while the short preamble JUMP reuses the corresponding
    /// preamble-produced values.
    pub jump_args: Vec<OpRef>,
    /// The exported virtual state at the loop header (from the preamble's exit).
    /// Used to check bridge compatibility and generate additional guards.
    pub exported_state: Option<VirtualState>,
}

impl ShortPreamble {
    /// Create an empty short preamble (no extra operations needed).
    pub fn empty() -> Self {
        ShortPreamble {
            ops: Vec::new(),
            inputargs: Vec::new(),
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
        }
    }

    /// Whether this short preamble has any operations to replay.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Number of operations in the short preamble.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Generate the operations to prepend when a bridge enters the loop.
    ///
    /// `bridge_args` are the OpRefs that the bridge provides as values
    /// for each label arg. The short preamble ops are instantiated with
    /// these concrete references.
    pub fn instantiate(&self, bridge_args: &[OpRef]) -> Vec<Op> {
        let mut result: Vec<Op> = Vec::with_capacity(self.ops.len());

        for entry in &self.ops {
            let mut op = entry.op.clone();

            // Remap arguments: replace label arg indices with bridge's concrete refs
            for (arg_pos, label_idx) in &entry.arg_mapping {
                if let Some(bridge_ref) = bridge_args.get(*label_idx) {
                    if *arg_pos < op.args.len() {
                        op.args[*arg_pos] = *bridge_ref;
                    }
                }
            }

            if let Some(ref mut fail_args) = op.fail_args {
                for (fail_arg_pos, label_idx) in &entry.fail_arg_mapping {
                    if let Some(bridge_ref) = bridge_args.get(*label_idx) {
                        if *fail_arg_pos < fail_args.len() {
                            fail_args[*fail_arg_pos] = *bridge_ref;
                        }
                    }
                }
            }

            if op.opcode.is_guard_overflow()
                && !matches!(result.last(), Some(prev) if prev.opcode.is_ovf())
            {
                continue;
            }

            result.push(op);
        }

        result
    }
}

impl ShortPreamble {
    /// shortpreamble.py: apply to bridge — prepend instantiated short preamble
    /// ops to a bridge trace, creating a complete trace that the optimizer
    /// can process with full preamble context.
    pub fn apply_to_bridge(&self, bridge_args: &[OpRef], bridge_ops: &[Op]) -> Vec<Op> {
        let mut result = self.instantiate(bridge_args);
        result.extend_from_slice(bridge_ops);
        result
    }

    /// Count guards in the short preamble.
    pub fn num_guards(&self) -> usize {
        self.ops.iter().filter(|e| e.op.opcode.is_guard()).count()
    }

    /// Count pure ops in the short preamble.
    pub fn num_pure_ops(&self) -> usize {
        self.ops
            .iter()
            .filter(|e| e.op.opcode.is_always_pure())
            .count()
    }
}

/// Collector that extracts short preamble operations from an already-built
/// preamble trace.
///
/// This is intentionally separate from RPython's `ShortPreambleBuilder`.
/// The RPython builder consumes exported short boxes while building phase 2;
/// this collector just turns a preamble section into a `ShortPreamble`.
pub struct CollectedShortPreambleBuilder {
    /// Raw ops collected during the preamble phase (before Label).
    raw_ops: Vec<Op>,
    /// Map from preamble OpRef to label arg index (set when Label is found).
    preamble_to_label_arg: HashMap<OpRef, usize>,
    /// Whether the builder is still collecting (before Label).
    active: bool,
}

impl CollectedShortPreambleBuilder {
    pub fn new() -> Self {
        CollectedShortPreambleBuilder {
            raw_ops: Vec::new(),
            preamble_to_label_arg: HashMap::new(),
            active: true,
        }
    }

    /// Set up the mapping from preamble OpRefs to label arg indices.
    ///
    /// Called when the Label is encountered. `label_args` are the OpRefs
    /// that the Label carries (= the loop-carried values from the preamble).
    pub fn set_label_args(&mut self, label_args: &[OpRef]) {
        self.preamble_to_label_arg.clear();
        for (i, opref) in label_args.iter().enumerate() {
            self.preamble_to_label_arg.insert(*opref, i);
        }
        self.active = false; // Switch from preamble to body phase
    }

    /// Record a guard from the preamble that the body depends on.
    ///
    /// The guard's arguments should reference preamble OpRefs that
    /// are carried across the Label as label args.
    pub fn add_preamble_guard(&mut self, op: &Op) {
        if !self.active {
            return; // Only collect during preamble phase
        }

        // Only record guard operations
        if !op.opcode.is_guard() {
            return;
        }

        self.raw_ops.push(op.clone());
    }

    /// Record any preamble operation (guard or pure) that establishes
    /// a fact the body depends on.
    pub fn add_preamble_op(&mut self, op: &Op) {
        if !self.active {
            return;
        }

        self.raw_ops.push(op.clone());
    }

    /// Finalize the builder into a ShortPreamble.
    ///
    /// Called after the Label has been processed and the mapping is set.
    /// Computes arg mappings using the preamble-to-label-arg map.
    pub fn build(self, exported_state: Option<VirtualState>) -> ShortPreamble {
        let entries = self
            .raw_ops
            .into_iter()
            .map(|op| {
                let mut arg_mapping = Vec::new();
                for (arg_pos, arg_ref) in op.args.iter().enumerate() {
                    if let Some(&label_idx) = self.preamble_to_label_arg.get(arg_ref) {
                        arg_mapping.push((arg_pos, label_idx));
                    }
                }
                let mut fail_arg_mapping = Vec::new();
                if let Some(fail_args) = &op.fail_args {
                    for (fail_arg_pos, fail_arg_ref) in fail_args.iter().enumerate() {
                        if let Some(&label_idx) = self.preamble_to_label_arg.get(fail_arg_ref) {
                            fail_arg_mapping.push((fail_arg_pos, label_idx));
                        }
                    }
                }
                ShortPreambleOp {
                    op,
                    arg_mapping,
                    fail_arg_mapping,
                }
            })
            .collect();

        // Reconstruct label_args order from preamble_to_label_arg mapping
        let mut inputargs_by_idx: Vec<(usize, OpRef)> = self
            .preamble_to_label_arg
            .iter()
            .map(|(&opref, &idx)| (idx, opref))
            .collect();
        inputargs_by_idx.sort_by_key(|(idx, _)| *idx);
        let inputargs: Vec<OpRef> = inputargs_by_idx.into_iter().map(|(_, r)| r).collect();

        ShortPreamble {
            ops: entries,
            inputargs,
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state,
        }
    }
}

impl Default for CollectedShortPreambleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Classification of preamble operations.
///
/// shortpreamble.py: PreambleOp, HeapOp, PureOp, LoopInvariantOp, GuardOp
/// Each type determines how the operation is replayed when a bridge enters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreambleOpKind {
    /// shortpreamble.py: PreambleOp — base class for all preamble operations.
    /// A generic preamble operation (guard or other).
    Guard,
    /// shortpreamble.py: ShortInputArg — renamed inputarg for a label slot.
    InputArg,
    /// shortpreamble.py: HeapOp — a heap read (GETFIELD_GC, GETARRAYITEM_GC)
    /// that was cached during the preamble. On bridge entry, the field/array
    /// must be re-read to populate the cache.
    Heap,
    /// shortpreamble.py: PureOp — a pure operation whose result was used
    /// in the loop body. On bridge entry, the pure op is re-computed.
    Pure,
    /// shortpreamble.py: LoopInvariantOp — a CALL_LOOPINVARIANT that was
    /// cached for the loop iteration. On bridge entry, re-execute the call.
    LoopInvariant,
}

/// Extended preamble operation with classification.
///
/// shortpreamble.py: used by ShortBoxes and ExtendedShortPreambleBuilder
/// to track which operations need replay and how.
#[derive(Clone, Debug)]
pub struct PreambleOp {
    /// The operation to replay.
    pub op: Op,
    /// Classification of this operation.
    pub kind: PreambleOpKind,
    /// Index of the argument in the label (None if not a label arg).
    pub label_arg_idx: Option<usize>,
    /// RPython shortpreamble.py: whether this producer was assigned an
    /// invented SameAs name because another producer won the original slot.
    pub invented_name: bool,
    /// Original result box this invented name aliases, if any.
    pub same_as_source: Option<OpRef>,
}

impl PreambleOp {
    /// shortpreamble.py: add_op_to_short(sb) — per-kind logic.
    ///
    /// For HeapOp: reconstruct the getfield/getarrayitem with remapped args.
    /// For PureOp: reconstruct the pure op (promoting to CALL_PURE if call).
    /// For LoopInvariantOp: reconstruct as CALL_LOOPINVARIANT.
    pub fn add_op_to_short(&self, sb: &mut ShortBoxes) -> Option<ProducedShortOp> {
        let preamble_op = match &self.kind {
            PreambleOpKind::InputArg | PreambleOpKind::Guard => self.op.clone(),
            PreambleOpKind::Heap => {
                let args = self
                    .op
                    .args
                    .iter()
                    .map(|&arg| sb.produce_arg(arg))
                    .collect::<Option<Vec<_>>>()?;
                let mut op = self.op.clone();
                op.args = args.into_iter().collect();
                op
            }
            PreambleOpKind::Pure => {
                let args = self
                    .op
                    .args
                    .iter()
                    .map(|&arg| sb.produce_arg(arg))
                    .collect::<Option<Vec<_>>>()?;
                let mut op = self.op.clone();
                op.args = args.into_iter().collect();
                if op.opcode.is_call() {
                    op.opcode = match op.opcode {
                        OpCode::CallI => OpCode::CallPureI,
                        OpCode::CallR => OpCode::CallPureR,
                        OpCode::CallF => OpCode::CallPureF,
                        OpCode::CallN => OpCode::CallPureN,
                        other => other,
                    };
                }
                op
            }
            PreambleOpKind::LoopInvariant => {
                let args = self
                    .op
                    .args
                    .iter()
                    .map(|&arg| sb.produce_arg(arg))
                    .collect::<Option<Vec<_>>>()?;
                let mut op = self.op.clone();
                op.args = args.into_iter().collect();
                op.opcode = match op.opcode {
                    OpCode::CallI => OpCode::CallLoopinvariantI,
                    OpCode::CallR => OpCode::CallLoopinvariantR,
                    OpCode::CallF => OpCode::CallLoopinvariantF,
                    OpCode::CallN => OpCode::CallLoopinvariantN,
                    other => other,
                };
                op
            }
        };
        Some(ProducedShortOp {
            kind: self.kind.clone(),
            preamble_op,
            invented_name: self.invented_name,
            same_as_source: self.same_as_source,
        })
    }
}

/// shortpreamble.py: ShortBoxes — tracks which values from the preamble
/// are "boxed" into the short preamble. Maps label arg indices to
/// the operations that produce them.
#[derive(Clone, Debug, Default)]
pub struct ShortBoxes {
    /// Mapping from exported label arg box to its position.
    pub label_arg_positions: HashMap<OpRef, usize>,
    /// shortpreamble.py: potential_ops
    potential_ops: HashMap<OpRef, PotentialShortOp>,
    /// Ordered insertion list for potential ops, matching shortpreamble.py's
    /// OrderedDict iteration contract.
    potential_order: Vec<OpRef>,
    /// shortpreamble.py: produced_short_boxes
    produced_short_boxes: HashMap<OpRef, ProducedShortOp>,
    /// Production order for exported short ops.
    produced_order: Vec<OpRef>,
    /// shortpreamble.py: const_short_boxes
    const_short_boxes: Vec<PreambleOp>,
    /// RPython shortpreamble.py: Const boxes are directly admissible in
    /// produce_arg(). majit models constants as OpRef entries in OptContext,
    /// so we track which OpRefs correspond to constants here.
    known_constants: HashSet<OpRef>,
    /// shortpreamble.py: short_inputargs
    short_inputargs: Vec<OpRef>,
    /// shortpreamble.py: boxes_in_production
    boxes_in_production: HashSet<OpRef>,
    /// Fresh synthetic names for invented short-box aliases.
    next_synthetic_pos: u32,
    /// The number of label args.
    pub num_label_args: usize,
}

#[derive(Clone, Debug)]
enum PotentialShortOp {
    Preamble(PreambleOp),
    Compound(CompoundOp),
}

impl PotentialShortOp {
    fn add_op_to_short(&self, sb: &mut ShortBoxes) -> Option<ProducedShortOp> {
        match self {
            PotentialShortOp::Preamble(op) => op.add_op_to_short(sb),
            PotentialShortOp::Compound(compound) => {
                let mut produced = compound.flatten(sb, Vec::new());
                if produced.is_empty() {
                    None
                } else {
                    let index = ShortBoxes::pick_produced_op_index(&produced, true);
                    let chosen = produced[index].clone();
                    for (i, mut alt) in produced.into_iter().enumerate() {
                        if i == index {
                            continue;
                        }
                        let alias = OpRef(sb.next_synthetic_pos);
                        sb.next_synthetic_pos += 1;
                        alt.preamble_op.pos = alias;
                        alt.invented_name = true;
                        alt.same_as_source = Some(compound.res);
                        sb.produced_short_boxes.insert(alias, alt.clone());
                        sb.produced_order.push(alias);
                    }
                    Some(chosen)
                }
            }
        }
    }
}

impl ShortBoxes {
    pub fn new(num_label_args: usize) -> Self {
        ShortBoxes {
            label_arg_positions: HashMap::new(),
            potential_ops: HashMap::new(),
            potential_order: Vec::new(),
            produced_short_boxes: HashMap::new(),
            produced_order: Vec::new(),
            const_short_boxes: Vec::new(),
            known_constants: HashSet::new(),
            short_inputargs: Vec::new(),
            boxes_in_production: HashSet::new(),
            next_synthetic_pos: 0,
            num_label_args,
        }
    }

    pub fn with_label_args(label_args: &[OpRef]) -> Self {
        let mut boxes = Self::new(label_args.len());
        for (idx, &arg) in label_args.iter().enumerate() {
            boxes.label_arg_positions.insert(arg, idx);
            boxes.short_inputargs.push(arg);
            boxes.next_synthetic_pos = boxes.next_synthetic_pos.max(arg.0.saturating_add(1));
        }
        boxes
    }

    pub fn lookup_label_arg(&self, opref: OpRef) -> Option<usize> {
        self.label_arg_positions.get(&opref).copied()
    }

    /// RPython parity: check if opref is reachable in the short preamble.
    pub fn is_reachable(&self, opref: OpRef) -> bool {
        self.label_arg_positions.contains_key(&opref)
            || self.known_constants.contains(&opref)
            || self.potential_ops.contains_key(&opref)
    }

    pub fn note_known_constant(&mut self, opref: OpRef) {
        self.known_constants.insert(opref);
        self.next_synthetic_pos = self.next_synthetic_pos.max(opref.0.saturating_add(1));
    }

    pub fn note_known_constants_from_ctx(&mut self, ctx: &crate::optimizeopt::OptContext) {
        for (idx, value) in ctx.constants.iter().enumerate() {
            if value.is_some() {
                self.note_known_constant(OpRef(idx as u32));
            }
        }
    }

    fn add_op(&mut self, result: OpRef, pop: PotentialShortOp) {
        if !self.potential_ops.contains_key(&result) {
            self.potential_order.push(result);
        }
        self.next_synthetic_pos = self.next_synthetic_pos.max(result.0.saturating_add(1));
        self.potential_ops.insert(result, pop);
    }

    /// Add a pure operation as a short-box candidate.
    /// shortpreamble.py: sb.add_pure_op(op)
    pub fn add_pure_op(&mut self, op: Op) {
        let result = op.pos;
        self.add_potential_op(self.lookup_label_arg(result), op, PreambleOpKind::Pure);
    }

    /// Add a heap read as a short-box candidate.
    /// shortpreamble.py: sb.add_heap_op(op, getfield_op)
    pub fn add_heap_op(&mut self, op: Op) {
        let result = op.pos;
        self.add_potential_op(self.lookup_label_arg(result), op, PreambleOpKind::Heap);
    }

    /// Add a loop-invariant call as a short-box candidate.
    pub fn add_loopinvariant_op(&mut self, op: Op) {
        let result = op.pos;
        self.add_potential_op(
            self.lookup_label_arg(result),
            op,
            PreambleOpKind::LoopInvariant,
        );
    }

    pub(crate) fn add_short_input_arg(&mut self, arg: OpRef) {
        let label_arg_idx = self.lookup_label_arg(arg);
        let op = match majit_ir::Type::Int {
            _ => {
                let mut same_as = Op::new(OpCode::SameAsI, &[arg]);
                same_as.pos = arg;
                same_as
            }
        };
        if !self.potential_order.contains(&arg) {
            self.potential_order.push(arg);
        }
        self.potential_ops.insert(
            arg,
            PotentialShortOp::Preamble(PreambleOp {
                op,
                kind: PreambleOpKind::InputArg,
                label_arg_idx,
                invented_name: false,
                same_as_source: None,
            }),
        );
    }

    fn produce_arg(&mut self, opref: OpRef) -> Option<OpRef> {
        if let Some(existing) = self.produced_short_boxes.get(&opref) {
            return Some(existing.preamble_op.pos);
        }
        if self.boxes_in_production.contains(&opref) {
            return None;
        }
        if self.known_constants.contains(&opref) {
            return Some(opref);
        }
        if self.potential_ops.contains_key(&opref) {
            return self
                .materialize_one(opref)
                .map(|produced| produced.preamble_op.pos);
        }
        // Label args are always available as inputs (RPython: isinstance(op, InputArgIntOp))
        if self.label_arg_positions.contains_key(&opref) {
            return Some(opref);
        }
        None
    }

    fn pick_produced_op_index(candidates: &[ProducedShortOp], pick_other: bool) -> usize {
        let mut index: Option<usize> = None;
        for (i, item) in candidates.iter().enumerate() {
            let prefer = !matches!(item.kind, PreambleOpKind::Heap)
                && (pick_other || item.kind == PreambleOpKind::InputArg);
            if prefer {
                if index.is_some() && pick_other {
                    return Self::pick_produced_op_index(candidates, false);
                }
                index = Some(i);
            }
        }
        index.unwrap_or(0)
    }

    fn materialize_one(&mut self, result: OpRef) -> Option<ProducedShortOp> {
        if let Some(existing) = self.produced_short_boxes.get(&result) {
            return Some(existing.clone());
        }
        if self.boxes_in_production.contains(&result) {
            return None;
        }
        let candidate = self.potential_ops.get(&result)?.clone();
        self.boxes_in_production.insert(result);
        let produced = candidate.add_op_to_short(self)?;
        self.produced_short_boxes.insert(result, produced.clone());
        self.produced_order.push(result);
        self.boxes_in_production.remove(&result);
        Some(produced)
    }

    /// shortpreamble.py: produced_short_boxes after add_op_to_short().
    pub fn produced_ops(&mut self) -> Vec<(OpRef, ProducedShortOp)> {
        let keys = self.potential_order.clone();
        for key in keys {
            let _ = self.materialize_one(key);
        }
        self.produced_order
            .iter()
            .filter_map(|key| {
                self.produced_short_boxes
                    .get(key)
                    .cloned()
                    .map(|produced| (*key, produced))
            })
            .collect()
    }

    /// shortpreamble.py: create_short_inputargs(label_args)
    /// Build the input args for the short preamble from label args.
    /// Returns OpRefs for each label arg that has a producer, or the
    /// original label arg if no producer exists.
    pub fn create_short_inputargs(&self, label_args: &[OpRef]) -> Vec<OpRef> {
        if self.short_inputargs.is_empty() {
            label_args.to_vec()
        } else {
            self.short_inputargs.clone()
        }
    }

    /// shortpreamble.py: add_potential_op(op, pop)
    /// Add a produced operation to the short boxes at the given position.
    pub fn add_potential_op(&mut self, label_arg_idx: Option<usize>, op: Op, kind: PreambleOpKind) {
        let result = op.pos;
        let pop = PotentialShortOp::Preamble(PreambleOp {
            op,
            kind,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
        let next = match self.potential_ops.remove(&result) {
            Some(prev) => PotentialShortOp::Compound(CompoundOp {
                res: result,
                one: Box::new(pop),
                two: Box::new(prev),
            }),
            None => pop,
        };
        self.add_op(result, next);
    }
}

/// shortpreamble.py: create_short_boxes(optimizer, inputargs, label_args)
///
/// Build the short boxes mapping: for each label arg, determine
/// which preamble operation produces it. This is used to build
/// the short preamble that bridges need to replay.
pub fn create_short_boxes(
    short_boxes: &mut ShortBoxes,
    label_args: &[OpRef],
    _optimizer_ops: &[Op],
) -> Vec<ProducedShortOp> {
    for &arg in label_args {
        short_boxes.add_short_input_arg(arg);
    }
    short_boxes
        .produced_ops()
        .into_iter()
        .map(|(_, op)| op)
        .collect()
}

/// Collector-side extended builder for extracting categorized preamble ops from
/// a peeled trace.
///
/// This is intentionally separate from RPython's active
/// `ExtendedShortPreambleBuilder`, which operates while building the short
/// preamble for phase 2 / bridge entry.
pub struct CollectedExtendedShortPreambleBuilder {
    /// Guards from the preamble.
    guards: Vec<PreambleOp>,
    /// Heap reads from the preamble.
    heap_ops: Vec<PreambleOp>,
    /// Pure operations from the preamble.
    pure_ops: Vec<PreambleOp>,
    /// Loop-invariant calls from the preamble.
    loopinvariant_ops: Vec<PreambleOp>,
    /// Map from preamble OpRef to label arg index.
    preamble_to_label_arg: HashMap<OpRef, usize>,
}

impl CollectedExtendedShortPreambleBuilder {
    pub fn new() -> Self {
        CollectedExtendedShortPreambleBuilder {
            guards: Vec::new(),
            heap_ops: Vec::new(),
            pure_ops: Vec::new(),
            loopinvariant_ops: Vec::new(),
            preamble_to_label_arg: HashMap::new(),
        }
    }

    /// Set the label args mapping.
    pub fn set_label_args(&mut self, label_args: &[OpRef]) {
        self.preamble_to_label_arg.clear();
        for (i, opref) in label_args.iter().enumerate() {
            self.preamble_to_label_arg.insert(*opref, i);
        }
    }

    /// Add a guard operation.
    pub fn add_guard(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.guards.push(PreambleOp {
            op,
            kind: PreambleOpKind::Guard,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a pure operation.
    pub fn add_pure_op(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.pure_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::Pure,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a heap read.
    pub fn add_heap_op(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.heap_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::Heap,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a loop-invariant call.
    pub fn add_loopinvariant_op(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.loopinvariant_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::LoopInvariant,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Total number of recorded preamble operations.
    pub fn num_ops(&self) -> usize {
        self.guards.len() + self.heap_ops.len() + self.pure_ops.len() + self.loopinvariant_ops.len()
    }

    /// Build into a ShortPreamble, emitting operations in order:
    /// guards first, then heap reads, then pure ops, then loop-invariant.
    pub fn build(self, exported_state: Option<VirtualState>) -> ShortPreamble {
        let all_ops: Vec<PreambleOp> = self
            .guards
            .into_iter()
            .chain(self.heap_ops)
            .chain(self.pure_ops)
            .chain(self.loopinvariant_ops)
            .collect();

        let entries = all_ops
            .into_iter()
            .map(|preamble_op| {
                let mut arg_mapping = Vec::new();
                for (arg_pos, arg_ref) in preamble_op.op.args.iter().enumerate() {
                    if let Some(&label_idx) = self.preamble_to_label_arg.get(arg_ref) {
                        arg_mapping.push((arg_pos, label_idx));
                    }
                }
                let mut fail_arg_mapping = Vec::new();
                if let Some(fail_args) = &preamble_op.op.fail_args {
                    for (fail_arg_pos, fail_arg_ref) in fail_args.iter().enumerate() {
                        if let Some(&label_idx) = self.preamble_to_label_arg.get(fail_arg_ref) {
                            fail_arg_mapping.push((fail_arg_pos, label_idx));
                        }
                    }
                }
                ShortPreambleOp {
                    op: preamble_op.op,
                    arg_mapping,
                    fail_arg_mapping,
                }
            })
            .collect();

        ShortPreamble {
            ops: entries,
            inputargs: Vec::new(),
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state,
        }
    }
}

impl Default for CollectedExtendedShortPreambleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// shortpreamble.py: CompoundOp — a short op that is composed of
/// two sub-operations (e.g., getfield followed by getarrayitem).
#[derive(Clone, Debug)]
pub struct CompoundOp {
    /// The result OpRef of the compound operation.
    pub res: OpRef,
    /// First sub-operation.
    one: Box<PotentialShortOp>,
    /// Second sub-operation (depends on the result of `one`).
    two: Box<PotentialShortOp>,
}

impl CompoundOp {
    /// shortpreamble.py: CompoundOp.flatten(sb, l)
    ///
    /// Recursively flatten a tree of CompoundOps into a list of
    /// ProducedShortOps in dependency order (children first).
    pub fn flatten(
        &self,
        sb: &mut ShortBoxes,
        mut produced: Vec<ProducedShortOp>,
    ) -> Vec<ProducedShortOp> {
        match self.one.as_ref() {
            PotentialShortOp::Compound(compound) => {
                produced = compound.flatten(sb, produced);
            }
            PotentialShortOp::Preamble(op) => {
                if let Some(pop) = op.add_op_to_short(sb) {
                    produced.push(pop);
                }
            }
        }
        match self.two.as_ref() {
            PotentialShortOp::Compound(compound) => compound.flatten(sb, produced),
            PotentialShortOp::Preamble(op) => {
                if let Some(pop) = op.add_op_to_short(sb) {
                    produced.push(pop);
                }
                produced
            }
        }
    }
}

/// shortpreamble.py: ShortInputArg — a short op that represents
/// a label input argument (no actual operation needed, just maps
/// a preamble value to a label arg position).
#[derive(Clone, Debug)]
pub struct ShortInputArg {
    /// The result OpRef.
    pub res: OpRef,
    /// The preamble operation that produces this value.
    pub preamble_op: Op,
}

impl ShortInputArg {
    /// shortpreamble.py: ShortInputArg.add_op_to_short(sb)
    ///
    /// Returns a ProducedShortOp wrapping the preamble_op.
    /// For input args, the preamble_op is just forwarded.
    pub fn add_op_to_short(&self) -> ProducedShortOp {
        ProducedShortOp {
            kind: PreambleOpKind::Pure,
            preamble_op: self.preamble_op.clone(),
            invented_name: false,
            same_as_source: None,
        }
    }

    /// shortpreamble.py: ShortInputArg.produce_op(opt, ...)
    ///
    /// For input args, produce_op is a no-op — the value is
    /// already available as a label argument.
    pub fn produce_op(&self) {
        // No-op: the value is directly available from label args
    }
}

/// shortpreamble.py: ProducedShortOp — wraps a short op with its
/// preamble counterpart for emission during bridge compilation.
#[derive(Clone, Debug)]
pub struct ProducedShortOp {
    /// The short op classification.
    pub kind: PreambleOpKind,
    /// The preamble operation to replay.
    pub preamble_op: Op,
    /// Whether this short op uses an invented SameAs result.
    pub invented_name: bool,
    /// Original result this invented name aliases.
    pub same_as_source: Option<OpRef>,
}

#[derive(Clone, Debug, Default)]
struct AbstractShortPreambleBuilderState {
    short: Vec<Op>,
    short_results: HashSet<OpRef>,
    label_args: Vec<OpRef>,
    used_boxes: Vec<OpRef>,
    short_preamble_jump: Vec<Op>,
    extra_same_as: Vec<Op>,
    short_inputargs: Vec<OpRef>,
}

impl AbstractShortPreambleBuilderState {
    fn record_preamble_use(&mut self, result: OpRef, produced: &ProducedShortOp) {
        let current_result = produced.preamble_op.pos;
        // RPython shortpreamble.py records the box after replacement
        // (op.get_box_replacement()), i.e. the replayed/local result, not the
        // exported source identity used as a lookup key.
        let used_box = current_result;
        if produced.invented_name {
            let source = produced.same_as_source.unwrap_or(result);
            let mut op = Op::new(
                OpCode::same_as_for_type(produced.preamble_op.result_type()),
                &[source],
            );
            op.pos = current_result;
            self.extra_same_as.push(op);
        }
        self.used_boxes.push(used_box);
        self.short_preamble_jump.push(produced.preamble_op.clone());
    }

    /// Internal: append preamble_op to short (with ovf guard).
    /// Used by add_op_to_short (recursive export-time path).
    fn append_to_short(&mut self, result: OpRef, produced: &ProducedShortOp) -> Op {
        let canonical_result = produced.preamble_op.pos;
        if self.short_results.contains(&canonical_result) {
            return produced.preamble_op.clone();
        }
        let preamble_op = produced.preamble_op.clone();
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        preamble_op
    }

    /// shortpreamble.py:382-407: use_box(box, preamble_op, optimizer)
    /// Non-recursive: iterates preamble_op's args (adding non-input deps
    /// + guards to short), then appends preamble_op + result guards.
    /// Called by force_op_from_preamble (unroll.py:32).
    ///
    /// `arg_guards`: guards collected from PtrInfo::make_guards for each arg
    /// `result_guards`: guards from PtrInfo::make_guards for the result
    /// These are pre-collected by the caller (force_op_from_preamble) which
    /// has access to PtrInfo via OptContext.
    fn use_box(
        &mut self,
        produced: &ProducedShortOp,
        already_in_short: &HashSet<OpRef>,
        all_produced: &HashMap<OpRef, ProducedShortOp>,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) -> Op {
        let canonical_result = produced.preamble_op.pos;
        if self.short_results.contains(&canonical_result)
            || already_in_short.contains(&canonical_result)
        {
            return produced.preamble_op.clone();
        }
        // shortpreamble.py:383-396: iterate preamble_op args
        for &arg in &produced.preamble_op.args {
            if self.short_results.contains(&arg)
                || already_in_short.contains(&arg)
                || self.short_inputargs.contains(&arg)
            {
                continue;
            }
            // shortpreamble.py:393: self.short.append(arg)
            if let Some(dep) = all_produced.get(&arg) {
                let dep_canonical = dep.preamble_op.pos;
                if !self.short_results.contains(&dep_canonical)
                    && !already_in_short.contains(&dep_canonical)
                {
                    self.short_results.insert(dep_canonical);
                    self.short.push(dep.preamble_op.clone());
                    if dep.preamble_op.opcode.is_ovf() {
                        self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
                    }
                }
            }
        }
        // shortpreamble.py:389,396: info.make_guards(arg, self.short, optimizer)
        self.short.extend_from_slice(arg_guards);
        // shortpreamble.py:398: self.short.append(preamble_op)
        let preamble_op = produced.preamble_op.clone();
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        // shortpreamble.py:405-406: info.make_guards(preamble_op, self.short, optimizer)
        self.short.extend_from_slice(result_guards);
        preamble_op
    }
}

fn build_short_preamble_struct_from_ops(
    short_inputargs: &[OpRef],
    ops: &[Op],
    used_boxes: &[OpRef],
    jump_args: &[OpRef],
) -> ShortPreamble {
    let short_inputarg_positions: HashMap<OpRef, usize> = short_inputargs
        .iter()
        .enumerate()
        .map(|(idx, &arg)| (arg, idx))
        .collect();
    let entries = ops
        .iter()
        .cloned()
        .map(|op| {
            let arg_mapping = op
                .args
                .iter()
                .enumerate()
                .filter_map(|(arg_pos, arg_ref)| {
                    short_inputarg_positions
                        .get(arg_ref)
                        .copied()
                        .map(|label_idx| (arg_pos, label_idx))
                })
                .collect();
            let fail_arg_mapping = op
                .fail_args
                .as_ref()
                .map(|fail_args| {
                    fail_args
                        .iter()
                        .enumerate()
                        .filter_map(|(fail_arg_pos, fail_arg_ref)| {
                            short_inputarg_positions
                                .get(fail_arg_ref)
                                .copied()
                                .map(|label_idx| (fail_arg_pos, label_idx))
                        })
                        .collect()
                })
                .unwrap_or_default();
            ShortPreambleOp {
                op,
                arg_mapping,
                fail_arg_mapping,
            }
        })
        .collect();
    ShortPreamble {
        ops: entries,
        inputargs: short_inputargs.to_vec(),
        used_boxes: used_boxes.to_vec(),
        jump_args: jump_args.to_vec(),
        exported_state: None,
    }
}

/// shortpreamble.py: ShortPreambleBuilder
///
/// Builds the replayable short preamble from exported short boxes, while also
/// collecting `used_boxes`, `short_preamble_jump`, and `extra_same_as`.
#[derive(Clone, Debug)]
pub struct ShortPreambleBuilder {
    state: AbstractShortPreambleBuilderState,
    produced_short_boxes: HashMap<OpRef, ProducedShortOp>,
}

impl ShortPreambleBuilder {
    pub fn new(
        label_args: &[OpRef],
        short_boxes: &[(OpRef, ProducedShortOp)],
        short_inputargs: &[OpRef],
    ) -> Self {
        let produced_short_boxes = short_boxes.iter().cloned().collect();
        ShortPreambleBuilder {
            state: AbstractShortPreambleBuilderState {
                label_args: label_args.to_vec(),
                short_inputargs: if short_inputargs.is_empty() {
                    label_args.to_vec()
                } else {
                    short_inputargs.to_vec()
                },
                ..AbstractShortPreambleBuilderState::default()
            },
            produced_short_boxes,
        }
    }

    fn use_box_recursive(&mut self, result: OpRef, visiting: &mut HashSet<OpRef>) -> Option<Op> {
        let produced = self.produced_short_boxes.get(&result)?.clone();
        let canonical_result = produced.preamble_op.pos;
        if self.state.short_results.contains(&canonical_result) {
            return Some(produced.preamble_op);
        }
        if !visiting.insert(result) {
            return None;
        }
        for &arg in &produced.preamble_op.args {
            if self.produced_short_boxes.contains_key(&arg) {
                let _ = self.use_box_recursive(arg, visiting);
            }
        }
        visiting.remove(&result);
        Some(self.state.append_to_short(result, &produced))
    }

    /// shortpreamble.py:310: add_op_to_short — recursive, used during
    /// export-time create_short_boxes to resolve transitive dependencies.
    pub fn add_op_to_short(&mut self, result: OpRef) -> Option<Op> {
        self.use_box_recursive(result, &mut HashSet::new())
    }

    /// shortpreamble.py:382-407: use_box(box, preamble_op, optimizer)
    /// Non-recursive. Called by force_op_from_preamble (unroll.py:32).
    pub fn use_box(
        &mut self,
        result: OpRef,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) -> Option<Op> {
        let produced = self.produced_short_boxes.get(&result)?.clone();
        Some(self.state.use_box(
            &produced,
            &HashSet::new(),
            &self.produced_short_boxes,
            arg_guards,
            result_guards,
        ))
    }

    pub fn produced_short_op(&self, result: OpRef) -> Option<ProducedShortOp> {
        self.produced_short_boxes.get(&result).cloned()
    }

    pub fn add_tracked_preamble_op(&mut self, result: OpRef, produced: &ProducedShortOp) {
        self.state.record_preamble_use(result, produced);
    }

    pub fn add_preamble_op(&mut self, result: OpRef) -> bool {
        let Some(produced) = self.produced_short_boxes.get(&result).cloned() else {
            return false;
        };
        self.add_tracked_preamble_op(result, &produced);
        true
    }

    pub fn build_short_preamble(&self) -> Vec<Op> {
        let mut result = Vec::with_capacity(self.state.short.len() + 2);
        result.push(Op::new(OpCode::Label, &self.state.short_inputargs));
        result.extend(self.state.short.iter().cloned());
        let jump_args: Vec<OpRef> = self
            .state
            .short_preamble_jump
            .iter()
            .map(|op| op.pos)
            .collect();
        result.push(Op::new(OpCode::Jump, &jump_args));
        result
    }

    pub fn build_short_preamble_struct(&self) -> ShortPreamble {
        let jump_args: Vec<OpRef> = self
            .state
            .short_preamble_jump
            .iter()
            .map(|op| op.pos)
            .collect();
        build_short_preamble_struct_from_ops(
            &self.state.short_inputargs,
            &self.state.short,
            &self.state.used_boxes,
            &jump_args,
        )
    }

    pub fn used_boxes(&self) -> &[OpRef] {
        &self.state.used_boxes
    }

    pub fn extra_same_as(&self) -> &[Op] {
        &self.state.extra_same_as
    }

    pub fn short_inputargs(&self) -> &[OpRef] {
        &self.state.short_inputargs
    }
}

/// shortpreamble.py: ExtendedShortPreambleBuilder
///
/// Keeps the existing short preamble stable while allowing inline replay to
/// discover additional required producers and append them to the loop label /
/// jump contract.
#[derive(Clone, Debug)]
pub struct ExtendedShortPreambleBuilder {
    produced_short_boxes: HashMap<OpRef, ProducedShortOp>,
    short_inputargs: Vec<OpRef>,
    base_short_ops: Vec<Op>,
    base_results: HashSet<OpRef>,
    base_extra_same_as: Vec<Op>,
    extra_state: AbstractShortPreambleBuilderState,
    label_args: Vec<OpRef>,
    used_boxes: Vec<OpRef>,
    short_jump_args: Vec<OpRef>,
    pub target_token: u64,
}

impl ExtendedShortPreambleBuilder {
    pub fn new(target_token: u64, sb: &ShortPreambleBuilder) -> Self {
        ExtendedShortPreambleBuilder {
            produced_short_boxes: sb.produced_short_boxes.clone(),
            short_inputargs: sb.short_inputargs().to_vec(),
            base_short_ops: Vec::new(),
            base_results: HashSet::new(),
            base_extra_same_as: sb.extra_same_as().to_vec(),
            extra_state: AbstractShortPreambleBuilderState {
                label_args: Vec::new(),
                short_inputargs: sb.short_inputargs().to_vec(),
                extra_same_as: sb.extra_same_as().to_vec(),
                ..AbstractShortPreambleBuilderState::default()
            },
            label_args: Vec::new(),
            used_boxes: Vec::new(),
            short_jump_args: Vec::new(),
            target_token,
        }
    }

    pub fn setup(&mut self, short_preamble: &ShortPreamble, label_args: &[OpRef]) {
        self.base_short_ops = short_preamble
            .ops
            .iter()
            .map(|entry| entry.op.clone())
            .collect();
        self.base_results = self.base_short_ops.iter().map(|op| op.pos).collect();
        self.extra_state.short.clear();
        self.extra_state.used_boxes.clear();
        self.extra_state.short_preamble_jump.clear();
        self.extra_state.extra_same_as = self.base_extra_same_as.clone();
        self.extra_state.label_args = label_args.to_vec();
        self.extra_state.short_inputargs = self.short_inputargs.clone();
        self.label_args = label_args.to_vec();
        self.used_boxes = short_preamble.used_boxes.clone();
        self.short_jump_args = short_preamble.jump_args.clone();
    }

    fn use_box_recursive(&mut self, result: OpRef, visiting: &mut HashSet<OpRef>) -> Option<Op> {
        let produced = self.produced_short_boxes.get(&result)?.clone();
        let canonical_result = produced.preamble_op.pos;
        if self.extra_state.short_results.contains(&canonical_result)
            || self.base_results.contains(&canonical_result)
        {
            return Some(produced.preamble_op);
        }
        if !visiting.insert(result) {
            return None;
        }
        for &arg in &produced.preamble_op.args {
            if self.produced_short_boxes.contains_key(&arg) {
                let _ = self.use_box_recursive(arg, visiting);
            }
        }
        visiting.remove(&result);
        Some(self.extra_state.append_to_short(result, &produced))
    }

    pub fn add_tracked_preamble_op(&mut self, result: OpRef, produced: &ProducedShortOp) {
        self.label_args.push(result);
        self.used_boxes.push(result);
        self.short_jump_args.push(produced.preamble_op.pos);
        self.extra_state.record_preamble_use(result, &produced);
    }

    pub fn add_preamble_op(&mut self, result: OpRef) -> bool {
        let Some(produced) = self.produced_short_boxes.get(&result).cloned() else {
            return false;
        };
        self.add_tracked_preamble_op(result, &produced);
        true
    }

    /// shortpreamble.py:310: add_op_to_short — recursive, export-time.
    pub fn add_op_to_short(&mut self, result: OpRef) -> Option<Op> {
        self.use_box_recursive(result, &mut HashSet::new())
    }

    /// shortpreamble.py:382-407: use_box(box, preamble_op, optimizer)
    /// Non-recursive. Called by force_op_from_preamble (unroll.py:32).
    pub fn use_box(
        &mut self,
        result: OpRef,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) -> Option<Op> {
        let produced = self.produced_short_boxes.get(&result)?.clone();
        Some(self.extra_state.use_box(
            &produced,
            &self.base_results,
            &self.produced_short_boxes,
            arg_guards,
            result_guards,
        ))
    }

    pub fn produced_short_op(&self, result: OpRef) -> Option<ProducedShortOp> {
        self.produced_short_boxes.get(&result).cloned()
    }

    pub fn short_inputargs(&self) -> &[OpRef] {
        &self.short_inputargs
    }

    pub fn build_short_preamble_struct(&self) -> ShortPreamble {
        let mut ops = self.base_short_ops.clone();
        ops.extend(self.extra_state.short.iter().cloned());
        let inputargs = if self.label_args.is_empty() {
            &self.short_inputargs
        } else {
            &self.label_args
        };
        build_short_preamble_struct_from_ops(
            inputargs,
            &ops,
            &self.used_boxes,
            &self.short_jump_args,
        )
    }

    pub fn extra_same_as(&self) -> &[Op] {
        &self.extra_state.extra_same_as
    }

    pub fn label_args(&self) -> &[OpRef] {
        &self.label_args
    }

    pub fn jump_args(&self) -> &[OpRef] {
        &self.short_jump_args
    }

    pub fn short_ops_len(&self) -> usize {
        self.base_short_ops.len() + self.extra_state.short.len()
    }

    pub fn short_op(&self, index: usize) -> Option<&Op> {
        if index < self.base_short_ops.len() {
            self.base_short_ops.get(index)
        } else {
            self.extra_state
                .short
                .get(index.saturating_sub(self.base_short_ops.len()))
        }
    }
}

/// shortpreamble.py: build short preamble from optimizer state.
/// Called after preamble optimization is complete.
/// Collects guards + pure ops from the optimized preamble and
/// maps them to label arg indices.
pub fn build_from_preamble_and_label(
    preamble_ops: &[Op],
    label_args: &[OpRef],
    exported_state: Option<VirtualState>,
) -> ShortPreamble {
    let mut builder = CollectedShortPreambleBuilder::new();
    let mut included_ovf_positions = HashSet::new();
    // Record all preamble ops
    for (idx, op) in preamble_ops.iter().enumerate() {
        if op.opcode.is_guard() {
            if op.opcode.is_guard_overflow()
                && idx > 0
                && preamble_ops[idx - 1].opcode.is_ovf()
                && included_ovf_positions.insert(preamble_ops[idx - 1].pos)
            {
                builder.add_preamble_op(&preamble_ops[idx - 1]);
            }
            builder.add_preamble_guard(op);
        } else if op.opcode.is_always_pure() {
            builder.add_preamble_op(op);
        }
    }
    // Set label args to create the mapping
    builder.set_label_args(label_args);
    builder.build(exported_state)
}

/// Extract guards AND pure ops from a peeled trace's preamble section.
///
/// Given a peeled trace (output of OptUnroll), identifies the preamble
/// section (before the Label) and collects all guard + pure operations
/// as short preamble entries.
///
/// This is a simpler alternative to integrating the builder with the
/// optimizer — it works on already-peeled traces.
pub fn extract_short_preamble(peeled_ops: &[Op]) -> ShortPreamble {
    // Find the Label position
    let label_pos = peeled_ops.iter().position(|op| op.opcode == OpCode::Label);

    let label_pos = match label_pos {
        Some(pos) => pos,
        None => return ShortPreamble::empty(), // No label = no peeling happened
    };

    let label_args = &peeled_ops[label_pos].args;

    // Build preamble-to-label-arg mapping
    let mut preamble_to_label: HashMap<OpRef, usize> = HashMap::new();
    for (i, arg) in label_args.iter().enumerate() {
        preamble_to_label.insert(*arg, i);
    }

    // shortpreamble.py: Collect guards AND pure operations from the preamble.
    // Guards must be replayed so the body's assumptions hold.
    // Pure ops whose results are used as label args must also be replayed
    // (e.g., GETFIELD from preamble that feeds into loop body).
    let mut entries = Vec::new();
    let mut included_positions = HashSet::new();
    for (idx, op) in peeled_ops[..label_pos].iter().enumerate() {
        let mut included_overflow_producer = false;
        if op.opcode.is_guard_overflow() && idx > 0 {
            let ovf_op = &peeled_ops[idx - 1];
            if ovf_op.opcode.is_ovf() && included_positions.insert(ovf_op.pos) {
                let ovf_arg_mapping: Vec<(usize, usize)> = ovf_op
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(pos, arg)| preamble_to_label.get(arg).map(|&idx| (pos, idx)))
                    .collect();
                let ovf_fail_arg_mapping: Vec<(usize, usize)> = ovf_op
                    .fail_args
                    .as_ref()
                    .into_iter()
                    .flat_map(|fail_args| fail_args.iter().enumerate())
                    .filter_map(|(pos, arg)| preamble_to_label.get(arg).map(|&idx| (pos, idx)))
                    .collect();
                if !ovf_arg_mapping.is_empty() || !ovf_fail_arg_mapping.is_empty() {
                    entries.push(ShortPreambleOp {
                        op: ovf_op.clone(),
                        arg_mapping: ovf_arg_mapping,
                        fail_arg_mapping: ovf_fail_arg_mapping,
                    });
                    included_overflow_producer = true;
                } else {
                    included_positions.remove(&ovf_op.pos);
                }
            }
        }
        if op.opcode.is_guard_overflow() && !included_overflow_producer {
            continue;
        }
        let include = op.opcode.is_guard() || op.opcode.is_always_pure();
        if !include {
            continue;
        }

        let arg_mapping: Vec<(usize, usize)> = op
            .args
            .iter()
            .enumerate()
            .filter_map(|(pos, arg)| preamble_to_label.get(arg).map(|&idx| (pos, idx)))
            .collect();
        let fail_arg_mapping: Vec<(usize, usize)> = op
            .fail_args
            .as_ref()
            .into_iter()
            .flat_map(|fail_args| fail_args.iter().enumerate())
            .filter_map(|(pos, arg)| preamble_to_label.get(arg).map(|&idx| (pos, idx)))
            .collect();

        // Only include ops that reference label args
        if (!arg_mapping.is_empty() || !fail_arg_mapping.is_empty())
            && included_positions.insert(op.pos)
        {
            entries.push(ShortPreambleOp {
                op: op.clone(),
                arg_mapping,
                fail_arg_mapping,
            });
        }
    }

    ShortPreamble {
        ops: entries,
        inputargs: Vec::new(),
        used_boxes: Vec::new(),
        jump_args: Vec::new(),
        exported_state: None,
    }
}

pub fn build_short_preamble_from_exported_boxes(
    label_args: &[OpRef],
    short_inputargs: &[OpRef],
    exported_short_boxes: &[PreambleOp],
) -> ShortPreamble {
    let inputarg_map: HashMap<OpRef, OpRef> = label_args
        .iter()
        .copied()
        .zip(short_inputargs.iter().copied())
        .collect();
    let produced: Vec<(OpRef, ProducedShortOp)> = exported_short_boxes
        .iter()
        .filter(|entry| !entry.op.opcode.is_guard_overflow())
        .map(|entry| {
            let mut preamble_op = entry.op.clone();
            for arg in &mut preamble_op.args {
                if let Some(&renamed) = inputarg_map.get(arg) {
                    *arg = renamed;
                }
            }
            if let Some(fail_args) = preamble_op.fail_args.as_mut() {
                for arg in fail_args {
                    if let Some(&renamed) = inputarg_map.get(arg) {
                        *arg = renamed;
                    }
                }
            }
            (
                preamble_op.pos,
                ProducedShortOp {
                    kind: entry.kind.clone(),
                    preamble_op,
                    invented_name: entry.invented_name,
                    same_as_source: entry.same_as_source,
                },
            )
        })
        .collect();
    let mut builder = ShortPreambleBuilder::new(label_args, &produced, short_inputargs);
    for (result, _) in &produced {
        let _ = builder.add_op_to_short(*result);
        let _ = builder.add_preamble_op(*result);
    }
    builder.build_short_preamble_struct()
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

    #[test]
    fn test_empty_short_preamble() {
        let sp = ShortPreamble::empty();
        assert!(sp.is_empty());
        assert_eq!(sp.len(), 0);
    }

    #[test]
    fn test_extract_from_peeled_trace() {
        // Simulate a peeled trace:
        // 0: guard_true(v100)        ← preamble guard on loop-carried value
        // 1: int_add(v100, v101)     ← preamble computation
        // 2: Label(v100, v101)       ← loop header
        // 3: guard_true(v100)        ← body guard (same)
        // 4: int_add(v100, v101)     ← body computation
        // 5: Jump(v4, v101)          ← back-edge
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Label, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(4), OpRef(101)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // Should have captured the preamble guard AND the pure IntAdd
        assert_eq!(sp.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::GuardTrue);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntAdd);

        // The guard's arg v100 maps to label arg index 0
        assert_eq!(sp.ops[0].arg_mapping.len(), 1);
        assert_eq!(sp.ops[0].arg_mapping[0], (0, 0)); // arg position 0 → label arg 0
    }

    #[test]
    fn test_extract_no_label() {
        // No label = no peeling happened
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Finish, &[OpRef(0)]),
        ];

        let sp = extract_short_preamble(&ops);
        assert!(sp.is_empty());
    }

    #[test]
    fn test_extract_overflow_guard_includes_preceding_ovf_op() {
        let mut ops = vec![
            Op::new(OpCode::IntMulOvf, &[OpRef(100), OpRef(100)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Label, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[OpRef(100)]),
        ];
        assign_positions(&mut ops, 0);
        ops[1].fail_args = Some(vec![OpRef(100)].into());

        let sp = extract_short_preamble(&ops);

        assert_eq!(sp.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntMulOvf);
        assert_eq!(sp.ops[1].op.opcode, OpCode::GuardNoOverflow);
    }

    #[test]
    fn test_extract_overflow_guard_without_replayable_ovf_is_skipped() {
        let mut ops = vec![
            Op::new(OpCode::IntMulOvf, &[OpRef(200), OpRef(200)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Label, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[OpRef(100)]),
        ];
        assign_positions(&mut ops, 0);
        ops[1].fail_args = Some(vec![OpRef(100)].into());

        let sp = extract_short_preamble(&ops);

        assert!(sp.is_empty());
    }

    #[test]
    fn test_extract_skips_non_label_guards() {
        // Guards that don't reference label args should not be included
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]), // refs temporary, not label arg
            Op::new(OpCode::Label, &[OpRef(100)]),   // only v100 is a label arg
            Op::new(OpCode::Jump, &[OpRef(100)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // The guard refs v0 (the IntAdd result), which is NOT in the label args.
        // But IntAdd refs v100 which IS a label arg → IntAdd IS extracted.
        // The guard on v0 is NOT extracted (v0 is not a label arg).
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_instantiate_short_preamble() {
        // Create a short preamble with a guard_class on label arg 0
        let guard_op = Op::new(OpCode::GuardClass, &[OpRef(100), OpRef(200)]);
        let sp = ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: guard_op,
                arg_mapping: vec![(0, 0)], // arg 0 maps to label arg 0
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: vec![OpRef(100), OpRef(101)],
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
        };

        // Instantiate for a bridge that provides v500 as label arg 0
        let bridge_args = &[OpRef(500), OpRef(501)];
        let instantiated = sp.instantiate(bridge_args);

        assert_eq!(instantiated.len(), 1);
        assert_eq!(instantiated[0].opcode, OpCode::GuardClass);
        assert_eq!(instantiated[0].args[0], OpRef(500)); // remapped from label arg 0
        assert_eq!(instantiated[0].args[1], OpRef(200)); // not a label arg, unchanged
    }

    #[test]
    fn test_instantiate_multiple_mappings() {
        // Guard with two args, both mapping to label args
        let guard_op = Op::new(OpCode::GuardValue, &[OpRef(100), OpRef(101)]);
        let sp = ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: guard_op,
                arg_mapping: vec![(0, 0), (1, 1)],
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: vec![OpRef(100), OpRef(101)],
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
        };

        let bridge_args = &[OpRef(300), OpRef(301)];
        let instantiated = sp.instantiate(bridge_args);

        assert_eq!(instantiated[0].args[0], OpRef(300));
        assert_eq!(instantiated[0].args[1], OpRef(301));
    }

    #[test]
    fn test_builder_collects_guards() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // Simulate preamble processing
        let guard1 = Op::new(OpCode::GuardTrue, &[OpRef(100)]);
        let guard2 = Op::new(OpCode::GuardClass, &[OpRef(101), OpRef(200)]);
        let non_guard = Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]);

        builder.add_preamble_guard(&guard1);
        builder.add_preamble_guard(&guard2);
        builder.add_preamble_guard(&non_guard); // should be ignored (not a guard)

        // Set label args (preamble phase ends)
        builder.set_label_args(&[OpRef(100), OpRef(101)]);

        // After label, no more collection
        let guard3 = Op::new(OpCode::GuardTrue, &[OpRef(100)]);
        builder.add_preamble_guard(&guard3); // should be ignored

        let sp = builder.build(None);
        assert_eq!(sp.len(), 2); // Only the two guards from the preamble
    }

    #[test]
    fn test_builder_maps_args_to_label_indices() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // Preamble has guard on v100 and v101
        let guard = Op::new(OpCode::GuardValue, &[OpRef(100), OpRef(200)]);
        builder.add_preamble_guard(&guard);

        // Label carries v100 as arg 0 and v101 as arg 1
        builder.set_label_args(&[OpRef(100), OpRef(101)]);

        let sp = builder.build(None);
        assert_eq!(sp.ops[0].arg_mapping.len(), 1); // v100 → label arg 0
        assert_eq!(sp.ops[0].arg_mapping[0], (0, 0));
        // v200 is not a label arg, so it's not in the mapping
    }

    #[test]
    fn test_builder_add_preamble_op_any_type() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // add_preamble_op accepts any op type (not just guards)
        let pure_op = Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]);
        builder.add_preamble_op(&pure_op);

        builder.set_label_args(&[OpRef(100), OpRef(101)]);

        let sp = builder.build(None);
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_extract_multiple_guards() {
        // Multiple guards in the preamble
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(101)]),
            Op::new(OpCode::GuardClass, &[OpRef(100), OpRef(200)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Label, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(100), OpRef(101)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // All three guards + the pure IntAdd reference label args
        assert_eq!(sp.len(), 4);
        assert_eq!(sp.ops[0].op.opcode, OpCode::GuardTrue);
        assert_eq!(sp.ops[1].op.opcode, OpCode::GuardNonnull);
        assert_eq!(sp.ops[2].op.opcode, OpCode::GuardClass);
        assert_eq!(sp.ops[3].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_roundtrip_extract_and_instantiate() {
        // Full round-trip: peel → extract short preamble → instantiate for bridge
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::GuardClass, &[OpRef(101), OpRef(200)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Label, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(4), OpRef(101)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // Instantiate for bridge with new values
        let bridge_args = &[OpRef(500), OpRef(501)];
        let instantiated = sp.instantiate(bridge_args);

        // 2 guards + 1 pure IntAdd
        assert_eq!(instantiated.len(), 3);

        // Guard_true now checks bridge's v500 (was v100 → label arg 0)
        assert_eq!(instantiated[0].opcode, OpCode::GuardTrue);
        assert_eq!(instantiated[0].args[0], OpRef(500));

        // Guard_class now checks bridge's v501 against constant v200
        assert_eq!(instantiated[1].opcode, OpCode::GuardClass);
        assert_eq!(instantiated[1].args[0], OpRef(501)); // remapped
        assert_eq!(instantiated[1].args[1], OpRef(200)); // constant, unchanged

        // IntAdd with remapped args
        assert_eq!(instantiated[2].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_apply_to_bridge() {
        let sp = ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
                arg_mapping: vec![(0, 0)],
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: vec![OpRef(0)],
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
        };

        let bridge_ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(500), OpRef(501)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];

        let result = sp.apply_to_bridge(&[OpRef(500)], &bridge_ops);
        // Short preamble guard + bridge ops
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GuardNonnull);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_instantiate_skips_bare_overflow_guard() {
        let sp = ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: Op::new(OpCode::GuardNoOverflow, &[]),
                arg_mapping: Vec::new(),
                fail_arg_mapping: Vec::new(),
            }],
            inputargs: vec![OpRef(100)],
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
        };

        let instantiated = sp.instantiate(&[OpRef(500)]);
        assert!(instantiated.is_empty());
    }

    #[test]
    fn test_instantiate_remaps_fail_args_by_position() {
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(10)]);
        guard.fail_args = Some(vec![OpRef(10), OpRef(20)].into());
        let sp = ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: guard,
                arg_mapping: vec![(0, 0)],
                fail_arg_mapping: vec![(0, 0), (1, 1)],
            }],
            inputargs: vec![OpRef(10), OpRef(20)],
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
        };

        let instantiated = sp.instantiate(&[OpRef(100), OpRef(200)]);
        assert_eq!(instantiated[0].args[0], OpRef(100));
        let fail_args = instantiated[0].fail_args.as_ref().unwrap();
        assert_eq!(fail_args.as_slice(), &[OpRef(100), OpRef(200)]);
    }

    #[test]
    fn test_num_guards_and_pure_ops() {
        let sp = ShortPreamble {
            ops: vec![
                ShortPreambleOp {
                    op: Op::new(OpCode::GuardTrue, &[OpRef(0)]),
                    arg_mapping: vec![(0, 0)],
                    fail_arg_mapping: Vec::new(),
                },
                ShortPreambleOp {
                    op: Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
                    arg_mapping: vec![(0, 0), (1, 1)],
                    fail_arg_mapping: Vec::new(),
                },
                ShortPreambleOp {
                    op: Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
                    arg_mapping: vec![(0, 0)],
                    fail_arg_mapping: Vec::new(),
                },
            ],
            inputargs: vec![OpRef(0), OpRef(1)],
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
        };

        assert_eq!(sp.num_guards(), 2);
        assert_eq!(sp.num_pure_ops(), 1);
    }

    #[test]
    fn test_build_short_preamble_from_exported_boxes_uses_exported_order() {
        let exported = vec![
            PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
                    op.pos = OpRef(7);
                    op
                },
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
            PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntSub, &[OpRef(7), OpRef(1)]);
                    op.pos = OpRef(8);
                    op
                },
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
        ];

        let sp = build_short_preamble_from_exported_boxes(
            &[OpRef(0), OpRef(1)],
            &[OpRef(10), OpRef(11)],
            &exported,
        );
        assert_eq!(sp.ops.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntSub);
        assert_eq!(sp.ops[1].arg_mapping, vec![(1, 1)]);
        assert_eq!(sp.inputargs, vec![OpRef(10), OpRef(11)]);
    }

    #[test]
    fn test_build_short_preamble_from_exported_boxes_skips_standalone_overflow_guards() {
        let label_args = vec![OpRef(10), OpRef(11)];
        let short_inputargs = vec![OpRef(100), OpRef(101)];

        let mut ovf = Op::new(OpCode::IntAddOvf, &[OpRef(10), OpRef(11)]);
        ovf.pos = OpRef(20);
        let guard = Op::new(OpCode::GuardNoOverflow, &[]);

        let exported = vec![
            PreambleOp {
                op: ovf,
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
            PreambleOp {
                op: guard,
                kind: PreambleOpKind::Guard,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
        ];

        let sp = build_short_preamble_from_exported_boxes(&label_args, &short_inputargs, &exported);
        let opcodes: Vec<OpCode> = sp.ops.iter().map(|entry| entry.op.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::IntAddOvf, OpCode::GuardNoOverflow]);
    }

    #[test]
    fn test_rpython_short_preamble_builder_add_op_to_short_recurses_dependencies() {
        let produced = vec![
            (
                OpRef(7),
                ProducedShortOp {
                    kind: PreambleOpKind::Pure,
                    preamble_op: {
                        let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
                        op.pos = OpRef(7);
                        op
                    },
                    invented_name: false,
                    same_as_source: None,
                },
            ),
            (
                OpRef(8),
                ProducedShortOp {
                    kind: PreambleOpKind::Pure,
                    preamble_op: {
                        let mut op = Op::new(OpCode::IntMul, &[OpRef(7), OpRef(1)]);
                        op.pos = OpRef(8);
                        op
                    },
                    invented_name: false,
                    same_as_source: None,
                },
            ),
        ];
        let mut builder =
            ShortPreambleBuilder::new(&[OpRef(0), OpRef(1)], &produced, &[OpRef(0), OpRef(1)]);

        let used = builder.add_op_to_short(OpRef(8)).unwrap();
        assert!(builder.add_preamble_op(OpRef(7)));
        assert!(builder.add_preamble_op(OpRef(8)));
        assert_eq!(used.opcode, OpCode::IntMul);
        let short = builder.build_short_preamble();
        assert_eq!(short[1].opcode, OpCode::IntAdd);
        assert_eq!(short[2].opcode, OpCode::IntMul);
        assert_eq!(builder.used_boxes(), &[OpRef(7), OpRef(8)]);
    }

    #[test]
    fn test_build_short_preamble_struct_preserves_inputargs_and_used_boxes() {
        let produced = vec![(
            OpRef(7),
            ProducedShortOp {
                kind: PreambleOpKind::Pure,
                preamble_op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
                    op.pos = OpRef(7);
                    op
                },
                invented_name: false,
                same_as_source: None,
            },
        )];
        let mut builder =
            ShortPreambleBuilder::new(&[OpRef(0), OpRef(1)], &produced, &[OpRef(0), OpRef(1)]);

        let _ = builder.add_op_to_short(OpRef(7));
        assert!(builder.add_preamble_op(OpRef(7)));
        let sp = builder.build_short_preamble_struct();

        assert_eq!(sp.inputargs, vec![OpRef(0), OpRef(1)]);
        assert_eq!(sp.used_boxes, vec![OpRef(7)]);
    }

    #[test]
    fn test_build_from_preamble_and_label() {
        let mut preamble = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
        ];
        assign_positions(&mut preamble, 0);

        let label_args = &[OpRef(100), OpRef(101)];
        let sp = build_from_preamble_and_label(&preamble, label_args, None);

        // Guard + pure IntAdd
        assert_eq!(sp.len(), 2);
    }

    #[test]
    fn test_extended_builder() {
        let mut builder = CollectedExtendedShortPreambleBuilder::new();
        builder.set_label_args(&[OpRef(100), OpRef(101)]);
        builder.add_guard(Op::new(OpCode::GuardTrue, &[OpRef(100)]));
        builder.add_pure_op(Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]));
        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(100)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, true),
        );
        heap.pos = OpRef(102);
        builder.add_heap_op(heap);
        builder.add_loopinvariant_op(Op::new(OpCode::CallI, &[OpRef(100)]));
        assert_eq!(builder.num_ops(), 4);
    }

    #[test]
    fn test_short_boxes() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(10), OpRef(11), OpRef(12)]);
        assert_eq!(sb.num_label_args, 3);
        let mut pure = Op::new(OpCode::IntAdd, &[OpRef(10), OpRef(11)]);
        pure.pos = OpRef(20);
        sb.add_pure_op(pure);
        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(10)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, true),
        );
        heap.pos = OpRef(21);
        sb.add_heap_op(heap);
        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 2);
    }

    #[test]
    fn test_short_boxes_reject_unknown_nonconstant_dependency() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(10)]);
        let mut pure = Op::new(OpCode::IntAdd, &[OpRef(10), OpRef(999)]);
        pure.pos = OpRef(20);
        sb.add_pure_op(pure);

        let produced = sb.produced_ops();
        // The label arg OpRef(10) itself is produced (as ShortInputArg),
        // but the pure op depending on unknown OpRef(999) is rejected.
        assert!(
            !produced.iter().any(|(r, _)| *r == OpRef(20)),
            "pure op with unknown dependency should be rejected"
        );
    }

    #[test]
    fn test_short_boxes_accept_known_constant_dependency() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(10)]);
        sb.note_known_constant(OpRef(999));
        let mut pure = Op::new(OpCode::IntAdd, &[OpRef(10), OpRef(999)]);
        pure.pos = OpRef(20);
        sb.add_pure_op(pure);

        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 1);
        let pure = produced
            .iter()
            .find(|(result, _)| *result == OpRef(20))
            .expect("missing produced pure op");
        assert_eq!(pure.1.preamble_op.args.as_slice(), &[OpRef(10), OpRef(999)]);
    }

    #[test]
    fn test_short_boxes_compound_prefers_non_heap_and_emits_invented_alias() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(10), OpRef(30), OpRef(31)]);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, true),
        );
        heap.pos = OpRef(10);
        sb.add_potential_op(Some(0), heap, PreambleOpKind::Heap);

        let mut pure = Op::new(OpCode::IntAdd, &[OpRef(30), OpRef(31)]);
        pure.pos = OpRef(10);
        sb.add_potential_op(Some(0), pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 2);

        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef(10))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::Pure);
        assert!(!chosen.1.invented_name);

        let alias = produced
            .iter()
            .find(|(result, _)| *result != OpRef(10))
            .unwrap();
        assert_eq!(alias.1.kind, PreambleOpKind::Heap);
        assert!(alias.1.invented_name);
        assert_eq!(alias.1.same_as_source, Some(OpRef(10)));
    }

    #[test]
    fn test_short_boxes_nested_compound_emits_multiple_invented_aliases() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(20), OpRef(30), OpRef(31)]);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, true),
        );
        heap.pos = OpRef(20);
        sb.add_potential_op(Some(0), heap, PreambleOpKind::Heap);

        let mut loopinv = Op::new(OpCode::CallI, &[OpRef(30)]);
        loopinv.pos = OpRef(20);
        sb.add_potential_op(Some(0), loopinv, PreambleOpKind::LoopInvariant);

        let mut pure = Op::new(OpCode::IntAdd, &[OpRef(30), OpRef(31)]);
        pure.pos = OpRef(20);
        sb.add_potential_op(Some(0), pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 3);

        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef(20))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::Pure);
        assert!(!chosen.1.invented_name);

        let aliases: Vec<_> = produced
            .iter()
            .filter(|(result, _)| *result != OpRef(20))
            .collect();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.iter().all(|(_, produced)| produced.invented_name));
        assert!(
            aliases
                .iter()
                .all(|(_, produced)| produced.same_as_source == Some(OpRef(20)))
        );
    }

    #[test]
    fn test_rpython_create_short_boxes_prefers_short_inputarg_over_heap_result() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(10), OpRef(30)]);
        sb.add_short_input_arg(OpRef(10));

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, true),
        );
        heap.pos = OpRef(10);
        sb.add_heap_op(heap);

        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 2);

        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef(10))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::InputArg);
        assert!(!chosen.1.invented_name);

        let alias = produced
            .iter()
            .find(|(result, _)| *result != OpRef(10))
            .unwrap();
        assert_eq!(alias.1.kind, PreambleOpKind::Heap);
        assert!(alias.1.invented_name);
        assert_eq!(alias.1.same_as_source, Some(OpRef(10)));
    }

    #[test]
    fn test_rpython_short_preamble_builder_add_op_to_short_builds_label_short_and_jump() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(10), OpRef(30), OpRef(31)]);

        let mut ovf = Op::new(OpCode::IntAddOvf, &[OpRef(30), OpRef(31)]);
        ovf.pos = OpRef(10);
        sb.add_potential_op(Some(0), ovf, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        let mut builder = ShortPreambleBuilder::new(&[OpRef(10)], &produced, &[OpRef(10)]);
        let used = builder.add_op_to_short(OpRef(10)).unwrap();
        assert!(builder.add_preamble_op(OpRef(10)));
        assert_eq!(used.opcode, OpCode::IntAddOvf);
        assert_eq!(builder.used_boxes(), &[OpRef(10)]);

        let short = builder.build_short_preamble();
        assert_eq!(short.len(), 4);
        assert_eq!(short[0].opcode, OpCode::Label);
        assert_eq!(short[1].opcode, OpCode::IntAddOvf);
        assert_eq!(short[2].opcode, OpCode::GuardNoOverflow);
        assert_eq!(short[3].opcode, OpCode::Jump);
        assert_eq!(short[3].args.as_slice(), &[OpRef(10)]);
    }

    #[test]
    fn test_rpython_short_preamble_builder_carries_used_box_in_struct() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(10), OpRef(30), OpRef(31)]);

        let mut pure = Op::new(OpCode::IntAdd, &[OpRef(30), OpRef(31)]);
        pure.pos = OpRef(10);
        sb.add_potential_op(Some(0), pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        let mut builder = ShortPreambleBuilder::new(&[OpRef(10)], &produced, &[OpRef(10)]);
        assert!(builder.add_preamble_op(OpRef(10)));

        let sp = builder.build_short_preamble_struct();
        assert_eq!(sp.used_boxes, vec![OpRef(10)]);
        assert_eq!(sp.jump_args, vec![OpRef(10)]);
    }

    #[test]
    fn test_rpython_short_preamble_builder_tracks_extra_same_as() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(20), OpRef(30), OpRef(31)]);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, true),
        );
        heap.pos = OpRef(20);
        sb.add_potential_op(Some(0), heap, PreambleOpKind::Heap);

        let mut pure = Op::new(OpCode::IntAdd, &[OpRef(30), OpRef(31)]);
        pure.pos = OpRef(20);
        sb.add_potential_op(Some(0), pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        let alias_result = produced
            .iter()
            .find(|(result, pop)| *result != OpRef(20) && pop.invented_name)
            .map(|(result, _)| *result)
            .unwrap();

        let mut builder = ShortPreambleBuilder::new(&[OpRef(20)], &produced, &[OpRef(20)]);
        assert!(builder.add_preamble_op(alias_result));
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos, alias_result);
        assert_eq!(extra[0].args.as_slice(), &[OpRef(20)]);
    }

    #[test]
    fn test_rpython_extended_builder_appends_label_and_jump_for_alias() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef(30), OpRef(40), OpRef(41)]);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef(40)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, true),
        );
        heap.pos = OpRef(30);
        sb.add_potential_op(Some(0), heap, PreambleOpKind::Heap);

        let mut pure = Op::new(OpCode::IntAdd, &[OpRef(40), OpRef(41)]);
        pure.pos = OpRef(30);
        sb.add_potential_op(Some(0), pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        let alias_result = produced
            .iter()
            .find(|(result, pop)| *result != OpRef(30) && pop.invented_name)
            .map(|(result, _)| *result)
            .unwrap();

        let builder = ShortPreambleBuilder::new(&[OpRef(30)], &produced, &[OpRef(30)]);
        let mut ext = ExtendedShortPreambleBuilder::new(7, &builder);
        ext.setup(
            &ShortPreamble {
                ops: Vec::new(),
                inputargs: vec![OpRef(30)],
                used_boxes: vec![OpRef(30)],
                jump_args: vec![OpRef(30)],
                exported_state: None,
            },
            &[OpRef(30)],
        );

        assert!(ext.add_preamble_op(alias_result));
        assert_eq!(ext.label_args(), &[OpRef(30), alias_result]);
        assert_eq!(ext.jump_args().len(), 2);
        assert_eq!(ext.extra_same_as().len(), 1);
        assert_eq!(ext.extra_same_as()[0].opcode, OpCode::SameAsI);
        assert_eq!(ext.extra_same_as()[0].pos, alias_result);
    }
}
