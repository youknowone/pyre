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
use std::collections::HashMap;

use majit_ir::{Op, OpCode, OpRef};

use crate::virtualstate::VirtualState;

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
    /// The exported virtual state at the loop header (from the preamble's exit).
    /// Used to check bridge compatibility and generate additional guards.
    pub exported_state: Option<VirtualState>,
}

impl ShortPreamble {
    /// Create an empty short preamble (no extra operations needed).
    pub fn empty() -> Self {
        ShortPreamble {
            ops: Vec::new(),
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
        let mut result = Vec::with_capacity(self.ops.len());

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

            // Also remap fail_args
            if let Some(ref mut fail_args) = op.fail_args {
                for fa in fail_args.iter_mut() {
                    // fail_args that reference label args get remapped
                    for (_, label_idx) in &entry.arg_mapping {
                        if let Some(bridge_ref) = bridge_args.get(*label_idx) {
                            if fa.0 < bridge_args.len() as u32 {
                                // Check if this fail_arg matches a label arg position
                                if fa.0 == *label_idx as u32 {
                                    *fa = *bridge_ref;
                                }
                            }
                        }
                    }
                }
            }

            result.push(op);
        }

        result
    }
}

/// Builder that collects short preamble operations during preamble optimization.
///
/// Used by the optimizer when processing a peeled trace. As the optimizer
/// processes the preamble section (before the Label), it records operations
/// that establish facts the loop body depends on.
pub struct ShortPreambleBuilder {
    /// Raw ops collected during the preamble phase (before Label).
    raw_ops: Vec<Op>,
    /// Map from preamble OpRef to label arg index (set when Label is found).
    preamble_to_label_arg: HashMap<OpRef, usize>,
    /// Whether the builder is still collecting (before Label).
    active: bool,
}

impl ShortPreambleBuilder {
    pub fn new() -> Self {
        ShortPreambleBuilder {
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
                ShortPreambleOp { op, arg_mapping }
            })
            .collect();

        ShortPreamble {
            ops: entries,
            exported_state,
        }
    }
}

impl Default for ShortPreambleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Classification of preamble operations.
///
/// shortpreamble.py: PreambleOp, HeapOp, PureOp, LoopInvariantOp, GuardOp
/// Each type determines how the operation is replayed when a bridge enters.
#[derive(Clone, Debug)]
pub enum PreambleOpKind {
    /// shortpreamble.py: PreambleOp — base class for all preamble operations.
    /// A generic preamble operation (guard or other).
    Guard,
    /// shortpreamble.py: HeapOp — a heap read (GETFIELD_GC, GETARRAYITEM_GC)
    /// that was cached during the preamble. On bridge entry, the field/array
    /// must be re-read to populate the cache.
    Heap {
        /// The field/array descriptor index.
        descr_idx: u32,
    },
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
}

/// shortpreamble.py: ShortBoxes — tracks which values from the preamble
/// are "boxed" into the short preamble. Maps label arg indices to
/// the operations that produce them.
#[derive(Clone, Debug, Default)]
pub struct ShortBoxes {
    /// Operations that produce values for each label arg.
    /// Index = label arg index, value = the PreambleOp producing it.
    pub producers: Vec<Option<PreambleOp>>,
    /// The number of label args.
    pub num_label_args: usize,
}

impl ShortBoxes {
    pub fn new(num_label_args: usize) -> Self {
        ShortBoxes {
            producers: vec![None; num_label_args],
            num_label_args,
        }
    }

    /// Record that `label_arg_idx` is produced by `op` with the given kind.
    pub fn add_op(&mut self, label_arg_idx: usize, op: Op, kind: PreambleOpKind) {
        if label_arg_idx < self.producers.len() {
            self.producers[label_arg_idx] = Some(PreambleOp {
                op,
                kind,
                label_arg_idx: Some(label_arg_idx),
            });
        }
    }

    /// Add a pure operation that produces a label arg value.
    /// shortpreamble.py: sb.add_pure_op(op)
    pub fn add_pure_op(&mut self, label_arg_idx: usize, op: Op) {
        self.add_op(label_arg_idx, op, PreambleOpKind::Pure);
    }

    /// Add a heap read that produces a label arg value.
    /// shortpreamble.py: sb.add_heap_op(op, descr)
    pub fn add_heap_op(&mut self, label_arg_idx: usize, op: Op, descr_idx: u32) {
        self.add_op(label_arg_idx, op, PreambleOpKind::Heap { descr_idx });
    }

    /// Add a loop-invariant call that produces a label arg value.
    pub fn add_loopinvariant_op(&mut self, label_arg_idx: usize, op: Op) {
        self.add_op(label_arg_idx, op, PreambleOpKind::LoopInvariant);
    }

    /// Get all operations that have producers (non-None).
    pub fn non_empty_ops(&self) -> impl Iterator<Item = (usize, &PreambleOp)> {
        self.producers
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().map(|op| (i, op)))
    }
}

/// shortpreamble.py: ExtendedShortPreambleBuilder — extended builder
/// that classifies operations into Guard/Heap/Pure/LoopInvariant types
/// for more precise short preamble generation.
pub struct ExtendedShortPreambleBuilder {
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

impl ExtendedShortPreambleBuilder {
    pub fn new() -> Self {
        ExtendedShortPreambleBuilder {
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
        });
    }

    /// Add a pure operation.
    pub fn add_pure_op(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.pure_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::Pure,
            label_arg_idx,
        });
    }

    /// Add a heap read.
    pub fn add_heap_op(&mut self, op: Op, descr_idx: u32) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.heap_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::Heap { descr_idx },
            label_arg_idx,
        });
    }

    /// Add a loop-invariant call.
    pub fn add_loopinvariant_op(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.loopinvariant_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::LoopInvariant,
            label_arg_idx,
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
                ShortPreambleOp {
                    op: preamble_op.op,
                    arg_mapping,
                }
            })
            .collect();

        ShortPreamble {
            ops: entries,
            exported_state,
        }
    }
}

impl Default for ExtendedShortPreambleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract guards from a peeled trace's preamble section.
///
/// Given a peeled trace (output of OptUnroll), identifies the preamble
/// section (before the Label) and collects all guard operations as
/// short preamble entries.
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

    // Collect guards from the preamble (before Label)
    let mut entries = Vec::new();
    for op in &peeled_ops[..label_pos] {
        if !op.opcode.is_guard() {
            continue;
        }

        let mut arg_mapping = Vec::new();
        for (arg_pos, arg_ref) in op.args.iter().enumerate() {
            if let Some(&label_idx) = preamble_to_label.get(arg_ref) {
                arg_mapping.push((arg_pos, label_idx));
            }
        }

        // Only include guards that reference label args
        // (guards on constants or temporaries don't need replay)
        if !arg_mapping.is_empty() {
            entries.push(ShortPreambleOp {
                op: op.clone(),
                arg_mapping,
            });
        }
    }

    ShortPreamble {
        ops: entries,
        exported_state: None,
    }
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

        // Should have captured the preamble guard
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::GuardTrue);

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

        // The guard refs v0 (the IntAdd result), which is NOT in the label args
        assert!(sp.is_empty());
    }

    #[test]
    fn test_instantiate_short_preamble() {
        // Create a short preamble with a guard_class on label arg 0
        let guard_op = Op::new(OpCode::GuardClass, &[OpRef(100), OpRef(200)]);
        let sp = ShortPreamble {
            ops: vec![ShortPreambleOp {
                op: guard_op,
                arg_mapping: vec![(0, 0)], // arg 0 maps to label arg 0
            }],
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
            }],
            exported_state: None,
        };

        let bridge_args = &[OpRef(300), OpRef(301)];
        let instantiated = sp.instantiate(bridge_args);

        assert_eq!(instantiated[0].args[0], OpRef(300));
        assert_eq!(instantiated[0].args[1], OpRef(301));
    }

    #[test]
    fn test_builder_collects_guards() {
        let mut builder = ShortPreambleBuilder::new();

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
        let mut builder = ShortPreambleBuilder::new();

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
        let mut builder = ShortPreambleBuilder::new();

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

        // All three guards reference label args
        assert_eq!(sp.len(), 3);
        assert_eq!(sp.ops[0].op.opcode, OpCode::GuardTrue);
        assert_eq!(sp.ops[1].op.opcode, OpCode::GuardNonnull);
        assert_eq!(sp.ops[2].op.opcode, OpCode::GuardClass);
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

        assert_eq!(instantiated.len(), 2);

        // Guard_true now checks bridge's v500 (was v100 → label arg 0)
        assert_eq!(instantiated[0].opcode, OpCode::GuardTrue);
        assert_eq!(instantiated[0].args[0], OpRef(500));

        // Guard_class now checks bridge's v501 against constant v200
        assert_eq!(instantiated[1].opcode, OpCode::GuardClass);
        assert_eq!(instantiated[1].args[0], OpRef(501)); // remapped
        assert_eq!(instantiated[1].args[1], OpRef(200)); // constant, unchanged
    }
}
