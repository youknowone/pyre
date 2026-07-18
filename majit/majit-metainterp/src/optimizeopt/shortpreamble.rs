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
use indexmap::{IndexMap, IndexSet};
use majit_ir::operand::Operand;
use majit_ir::{GcRef, Op, OpCode, OpRef};

use crate::optimizeopt::virtualstate::VirtualState;

pub type EmptyInfo = crate::optimizeopt::info::EmptyInfo;

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
    /// Stored as flat [`OpRef`] positions; an inline `OpRef::ConstPtr`
    /// ref is GC-walked directly by `walk_const_ptr_refs_mut`.
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
    /// Constant snapshot retained only for legacy test fixtures.
    /// Production short preamble ops embed inline `Const*` OpRefs
    /// directly, matching RPython where `_map_args` passes Const boxes
    /// through unchanged. This map is therefore empty on production paths.
    pub constants: majit_ir::ConstMap<majit_ir::Const>,
    /// RPython parity: PtrInfo for each inputarg, from Phase 1 export.
    /// shortpreamble.py:414-425: preamble_op.set_forwarded(info)
    /// Used by inline_short_preamble to propagate PtrInfo to jump_args
    /// so guards added by use_box are eliminated as redundant.
    pub inputarg_infos: Vec<Option<crate::optimizeopt::info::PtrInfo>>,
    /// RPython parity: Phase 1 inputargs preserved across Extended rebuilds.
    /// In RPython, short ops reference renamed inputargs (new Box objects)
    /// that are stable across compilations. In majit, ops keep original
    /// preamble OpRefs. When the Extended builder rebuilds with different
    /// inputargs (Phase 2 label_args), ops may reference Phase 1 OpRefs
    /// that aren't in the new inputargs. This field stores the original
    /// Phase 1 inputargs so inline_short_preamble can map them to jump_args.
    pub phase1_inputargs: Option<Vec<OpRef>>,
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
            constants: majit_ir::ConstMap::new(),
            phase1_inputargs: None,
            inputarg_infos: Vec::new(),
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

    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn visit_oprefs(refs: &mut [OpRef], visitor: &mut dyn FnMut(&mut GcRef)) {
            for r in refs {
                if let OpRef::ConstPtr(gcref) = r {
                    visitor(gcref);
                }
            }
        }

        for entry in &mut self.ops {
            entry.op.walk_const_ptr_refs_mut(visitor);
        }
        visit_oprefs(&mut self.inputargs, visitor);
        visit_oprefs(&mut self.used_boxes, visitor);
        visit_oprefs(&mut self.jump_args, visitor);
        if let Some(phase1_inputargs) = self.phase1_inputargs.as_mut() {
            visit_oprefs(phase1_inputargs, visitor);
        }
        if let Some(exported_state) = self.exported_state.as_mut() {
            exported_state.walk_const_ptr_refs_mut(visitor);
        }
        for (_, konst) in self.constants.iter_mut() {
            match konst {
                majit_ir::Const::Ref(gcref) => visitor(gcref),
                _ => {}
            }
        }
        for info in self.inputarg_infos.iter_mut().flatten() {
            info.walk_const_ptr_refs_mut(visitor);
        }
    }
}

impl ShortPreamble {
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
    /// shortpreamble.py:414 `ShortPreambleBuilder.label_args` — the
    /// OpRefs that the Label carries. Index in this list IS the label
    /// arg index. Lookup uses linear scan (label_args is small —
    /// bounded by loop-carried value count).
    label_args: Vec<OpRef>,
    /// Whether the builder is still collecting (before Label).
    active: bool,
}

impl CollectedShortPreambleBuilder {
    pub fn new() -> Self {
        CollectedShortPreambleBuilder {
            raw_ops: Vec::new(),
            label_args: Vec::new(),
            active: true,
        }
    }

    /// Set up the mapping from preamble OpRefs to label arg indices.
    ///
    /// Called when the Label is encountered. `label_args` are the OpRefs
    /// that the Label carries (= the loop-carried values from the preamble).
    pub fn set_label_args(&mut self, label_args: &[OpRef]) {
        self.label_args.clear();
        self.label_args.extend_from_slice(label_args);
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
        let Self {
            raw_ops,
            label_args,
            ..
        } = self;
        let entries = raw_ops
            .into_iter()
            .map(|op| {
                let mut arg_mapping = Vec::new();
                for (arg_pos, arg_ref) in op.getarglist().iter().enumerate() {
                    if let Some(label_idx) =
                        label_args.iter().position(|a| *a == arg_ref.to_opref())
                    {
                        arg_mapping.push((arg_pos, label_idx));
                    }
                }
                let mut fail_arg_mapping = Vec::new();
                if let Some(fail_args) = op.getfailargs() {
                    for (fail_arg_pos, fail_arg_ref) in fail_args.iter().enumerate() {
                        if let Some(label_idx) = label_args
                            .iter()
                            .position(|a| *a == fail_arg_ref.to_opref())
                        {
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

        ShortPreamble {
            ops: entries,
            inputargs: label_args,
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state,
            constants: majit_ir::ConstMap::new(),
            phase1_inputargs: None,
            inputarg_infos: Vec::new(),
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
    ///
    /// Carried as an [`majit_ir::OpRc`] so the exported entry, the derived
    /// `ProducedShortOp.preamble_op`, and replay-arg operands can share one
    /// object across the export/import boundary (upstream `preamble_op` is
    /// one ResOperation object, shortpreamble.py:283-296).
    pub op: majit_ir::OpRc,
    /// `short_op.res` — the result operand this entry produces. Carried as a
    /// producer-bound / const [`Operand`] so the canonical operand travels
    /// with the struct to the export boundary (`ProducedShortOp.res`),
    /// `Rc::ptr_eq`-stable on the producer.
    pub res: majit_ir::operand::Operand,
    /// Classification of this operation.
    pub kind: PreambleOpKind,
    /// Index of the argument in the label (None if not a label arg).
    pub label_arg_idx: Option<usize>,
    /// RPython shortpreamble.py: whether this producer was assigned an
    /// invented SameAs name because another producer won the original slot.
    pub invented_name: bool,
    /// Original result box this invented name aliases, if any.
    /// Carried as an [`Operand`] so the canonical (producer-bound) operand
    /// travels with the struct instead of being re-minted positionally at
    /// each use site.
    pub same_as_source: Option<majit_ir::operand::Operand>,
}

impl PreambleOp {
    /// shortpreamble.py: add_op_to_short(sb) — per-kind logic.
    ///
    /// For HeapOp: reconstruct the getfield/getarrayitem with remapped args.
    /// For PureOp: reconstruct the pure op (promoting to CALL_PURE if call).
    /// For LoopInvariantOp: reconstruct as CALL_LOOPINVARIANT.
    pub fn add_op_to_short(
        &self,
        sb: &mut ShortBoxes,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> Option<ProducedShortOp> {
        let preamble_op = match &self.kind {
            PreambleOpKind::InputArg | PreambleOpKind::Guard => (*self.op).clone(),
            PreambleOpKind::Heap => {
                // shortpreamble.py:91-102 HeapOp.add_op_to_short:
                //   preamble_arg = sb.produce_arg(sop.getarg(0))
                //   if rop.is_getfield(sop.opnum):
                //       preamble_op = ResOperation(sop.getopnum(), [preamble_arg], descr=sop.getdescr())
                //   else:
                //       preamble_op = ResOperation(sop.getopnum(), [preamble_arg, sop.getarg(1)], descr=sop.getdescr())
                let preamble_arg = sb.produce_arg(ctx, self.op.arg(0).to_opref())?;
                let args: smallvec::SmallVec<[majit_ir::operand::Operand; 3]> =
                    if self.op.opcode.is_getfield() {
                        smallvec::smallvec![preamble_arg]
                    } else {
                        smallvec::smallvec![preamble_arg, self.op.arg(1)]
                    };
                self.op.copy_and_change(self.op.opcode, Some(&args), None)
            }
            PreambleOpKind::Pure => {
                // shortpreamble.py:128-140 PureOp.add_op_to_short:
                //   arglist = [sb.produce_arg(arg) for arg in op.getarglist()]
                //   if rop.is_call(op.opnum):
                //       opnum = OpHelpers.call_pure_for_descr(op.getdescr())
                //   else:
                //       opnum = op.getopnum()
                //   return ProducedShortOp(self, op.copy_and_change(opnum, args=arglist))
                let args = self
                    .op
                    .getarglist()
                    .iter()
                    .map(|arg| sb.produce_arg(ctx, arg.to_opref()))
                    .collect::<Option<smallvec::SmallVec<[majit_ir::operand::Operand; 3]>>>()?;
                let opnum = if self.op.opcode.is_call() {
                    match self.op.opcode {
                        OpCode::CallI => OpCode::CallPureI,
                        OpCode::CallR => OpCode::CallPureR,
                        OpCode::CallF => OpCode::CallPureF,
                        OpCode::CallN => OpCode::CallPureN,
                        other => other,
                    }
                } else {
                    self.op.opcode
                };
                self.op.copy_and_change(opnum, Some(&args), None)
            }
            PreambleOpKind::LoopInvariant => {
                // shortpreamble.py:160-170 LoopInvariantOp.add_op_to_short:
                //   arglist = [sb.produce_arg(arg) for arg in op.getarglist()]
                //   opnum = OpHelpers.call_loopinvariant_for_descr(op.getdescr())
                //   return ProducedShortOp(self, op.copy_and_change(opnum, args=arglist))
                let args = self
                    .op
                    .getarglist()
                    .iter()
                    .map(|arg| sb.produce_arg(ctx, arg.to_opref()))
                    .collect::<Option<smallvec::SmallVec<[majit_ir::operand::Operand; 3]>>>()?;
                let opnum = match self.op.opcode {
                    OpCode::CallI => OpCode::CallLoopinvariantI,
                    OpCode::CallR => OpCode::CallLoopinvariantR,
                    OpCode::CallF => OpCode::CallLoopinvariantF,
                    OpCode::CallN => OpCode::CallLoopinvariantN,
                    other => other,
                };
                self.op.copy_and_change(opnum, Some(&args), None)
            }
        };
        Some(ProducedShortOp {
            kind: self.kind.clone(),
            // shortpreamble.py:120/85/170 `ProducedShortOp(self, ...)` —
            // short_op.res is the original result box; resolve canonical.
            res: ctx.materialize_operand_at(self.op.pos.get()),
            preamble_op: std::rc::Rc::new(preamble_op),
            invented_name: self.invented_name,
            same_as_source: self.same_as_source.clone(),
            label_arg_idx: self.label_arg_idx,
        })
    }
}

/// shortpreamble.py: ShortBoxes — tracks which values from the preamble
/// are "boxed" into the short preamble. Maps label arg indices to
/// the operations that produce them.
#[derive(Clone, Debug, Default)]
pub struct ShortBoxes {
    /// shortpreamble.py:249 self.potential_ops = OrderedDict()
    /// Keyed by the producer's result Box, compared by object identity
    /// (shortpreamble.py:259/290) — every insert/lookup resolves its
    /// position through `ctx.materialize_operand_at`, which memoizes one box
    /// per producer, so the same position yields the same object. Const
    /// results never key this map (they route to `const_short_boxes`).
    potential_ops: IndexMap<majit_ir::operand::Operand, PotentialShortOp>,
    /// Mirrors `to_opref()` for every key ever inserted into `potential_ops`
    /// (which is insert-only; same-key overwrites keep the same opref), so
    /// membership equals the old linear scan `any(k.to_opref() == opref)`.
    /// Parity anchor: shortpreamble.py:290 `op in self.potential_ops` is
    /// dict membership.
    potential_op_oprefs: IndexSet<OpRef>,
    /// shortpreamble.py:250 self.produced_short_boxes = {}
    /// (insertion order preserved by IndexMap for deterministic export.)
    /// Keyed by the result Box (`shortop.res`), compared by object
    /// identity (shortpreamble.py:317/338) — lookups resolve their
    /// position through `ctx.materialize_operand_at`, which memoizes one
    /// box per producer, so the same position yields the same object.
    produced_short_boxes: IndexMap<majit_ir::operand::Operand, ProducedShortOp>,
    /// shortpreamble.py: const_short_boxes
    const_short_boxes: Vec<PreambleOp>,
    /// RPython shortpreamble.py: Const boxes are directly admissible in
    /// produce_arg(). majit models constants as OpRef entries in OptContext,
    /// so we track which OpRefs correspond to constants here.
    known_constants: IndexSet<OpRef>,
    /// shortpreamble.py: short_inputargs
    ///
    /// shortpreamble.py:256 `renamed = OpHelpers.inputarg_from_tp(box.type)`
    /// mints a fresh producer-less InputArg box per label slot; the stored
    /// positions ARE the short preamble's Label args (shortpreamble.py:443).
    /// Each entry is a fresh InputArg position allocated from
    /// the op-position counter (`alloc_op_position_typed`), so its identity
    /// is DISTINCT from the original label arg (`shortpreamble.py:257`
    /// `ShortInputArg(box, renamed)` — `box` and `renamed` are two objects).
    /// `short_inputargs[i]` corresponds to `label_args[i]` positionally.
    short_inputargs: Vec<OpRef>,
    /// Strong-`Rc` rooting pool for the renamed `InputArg` objects behind
    /// `short_inputargs[i]`. Index-aligned 1:1 with `short_inputargs`. Keeps
    /// each renamed `AbstractInputArg` alive so producer-shaped consumers can
    /// bind that position to `Operand::InputArg` / a bound operand view — the
    /// analog of `TraceIterator.inputargs` rooting the outer-trace inputargs
    /// (opencoder.py:250-273).
    short_inputarg_refs: Vec<majit_ir::InputArgRc>,
    /// shortpreamble.py:256 `box = label_args[i]` — the ORIGINAL label-arg
    /// references, kept so `potential_ops`/lookups resolve a label arg by
    /// its own opref (`shortpreamble.py:259 potential_ops[box]`). pyre needs
    /// this explicitly because OpRef positions are not object identities:
    /// once `short_inputargs[i]` is a distinct renamed box, it no longer
    /// carries the original label-arg identity, so the original must be
    /// stored separately. `label_args[i]` pairs with `short_inputargs[i]`.
    label_args: Vec<OpRef>,
    /// shortpreamble.py: boxes_in_production — cycle-detection set
    /// for `materialize_one` recursion, keyed by the result Box
    /// (shortpreamble.py:314 `self.boxes_in_production[shortop.res]`).
    /// Active set is bounded by recursion depth (linear scan suffices).
    boxes_in_production: IndexSet<majit_ir::operand::Operand>,
    /// The number of label args.
    pub num_label_args: usize,
}

#[derive(Clone, Debug)]
enum PotentialShortOp {
    Preamble(PreambleOp),
    Compound(CompoundOp),
}

impl PotentialShortOp {
    fn add_op_to_short(
        &self,
        sb: &mut ShortBoxes,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> Option<ProducedShortOp> {
        match self {
            PotentialShortOp::Preamble(op) => op.add_op_to_short(sb, ctx),
            PotentialShortOp::Compound(compound) => {
                let produced = compound.flatten(sb, ctx, Vec::new());
                if produced.is_empty() {
                    None
                } else {
                    let index = ShortBoxes::_pick_op_index(&produced, true);
                    let chosen = produced[index].clone();
                    // shortpreamble.py:326-330 — alias side:
                    //   opnum = OpHelpers.same_as_for_type(shortop.res.type)
                    //   new_name = ResOperation(opnum, [shortop.res])
                    //   lst[i].short_op.res = new_name
                    // The alias's type matches `compound.res.type`. PyPy
                    // Box.type is fixed at construction (resoperation.py:
                    // 719/727/739) — an untyped compound.res would be an
                    // upstream-invariant violation.
                    for (i, mut alt) in produced.into_iter().enumerate() {
                        if i == index {
                            continue;
                        }
                        let tp = compound.res.ty().unwrap_or_else(|| {
                            panic!(
                                "compound short-preamble alias source {:?} has no \
                                 variant tag; same_as_for_type requires Int/Ref/Float \
                                 (shortpreamble.py:326-330)",
                                compound.res
                            )
                        });
                        let alias = ctx.alloc_op_position_typed(tp);
                        alt.preamble_op.pos.set(alias);
                        // shortpreamble.py:329 `lst[i].short_op.res =
                        // new_name` — the alias entry's res becomes the
                        // freshly invented name.
                        alt.res = ctx.materialize_operand_at(alias);
                        alt.invented_name = true;
                        // shortpreamble.py:330 `lst[i].short_op.res = new_name`:
                        // the alias result is now the freshly invented SameAs box,
                        // not a label arg. `label_arg_idx` was set by
                        // `lookup_label_arg` from the ORIGINAL result position, so it
                        // must be cleared to stay consistent with the updated `res`
                        // (upstream keys slot identity off `short_op.res`, which the
                        // line above just rebound). Otherwise the import slot-lookup
                        // (unroll.rs:4171) would map this invented alias onto the
                        // loop-carried `short_args[slot]` instead of minting a fresh
                        // result, collapsing the extra `same_as` identity.
                        alt.label_arg_idx = None;
                        // shortpreamble.py:328 `ResOperation(opnum, [shortop.res])`
                        // — the alias source is the Box itself; resolve to
                        // the canonical (possibly producer-bound) box.
                        alt.same_as_source = Some(ctx.materialize_operand_at(compound.res));
                        // shortpreamble.py:333 `self.produced_short_boxes[
                        // new_name] = lst[i]` — keyed by the alias box.
                        sb.produced_short_boxes.insert(alt.res.clone(), alt.clone());
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
            potential_ops: IndexMap::new(),
            potential_op_oprefs: IndexSet::new(),
            produced_short_boxes: IndexMap::new(),
            const_short_boxes: Vec::new(),
            known_constants: IndexSet::new(),
            short_inputargs: Vec::new(),
            short_inputarg_refs: Vec::new(),
            label_args: Vec::new(),
            boxes_in_production: IndexSet::new(),
            num_label_args,
        }
    }

    /// shortpreamble.py:254-259 — record the original label args. The
    /// matching `short_inputargs` renamed boxes are minted lazily by
    /// `add_short_input_arg` (one per `box`, in `label_args` order), so
    /// only the originals are stored here.
    pub fn with_label_args(label_args: &[OpRef]) -> Self {
        let mut boxes = Self::new(label_args.len());
        boxes.label_args = label_args.to_vec();
        boxes
    }

    /// shortpreamble.py:259 `self.potential_ops[box]` — resolve a label
    /// arg to its slot by the ORIGINAL box opref (not the renamed
    /// `short_inputargs` box, which is a distinct identity). `potential_ops`
    /// is box-keyed, so a box duplicated across `label_args + virtuals` (a
    /// virtual field coinciding with a label arg) OVERWRITES to the LAST
    /// occurrence's `ShortInputArg`; `rposition` mirrors that overwrite,
    /// returning the live slot — consistent with `add_short_input_arg`
    /// stamping `label_arg_idx = live_slot` (the last slot). For a
    /// duplicate-free box `rposition == position`.
    pub fn lookup_label_arg(&self, opref: OpRef) -> Option<usize> {
        self.label_args.iter().rposition(|&a| a == opref)
    }

    /// RPython parity: check if opref is reachable in the short preamble.
    pub fn is_reachable(&self, opref: OpRef) -> bool {
        self.label_args.iter().any(|&a| a == opref)
            || opref.is_constant()
            || self.potential_op_oprefs.contains(&opref)
    }

    pub fn note_known_constant(&mut self, opref: OpRef) {
        // shortpreamble.py: known_constants is a set of Const Box objects.
        // The CompoundOp alias counter is no longer ShortBoxes-internal
        // (slice β routes minting through `ctx.alloc_op_position_typed`),
        // so there is no `next_synthetic_pos` to bump past constant raws.
        self.known_constants.insert(opref);
    }

    fn add_op(&mut self, key: majit_ir::operand::Operand, pop: PotentialShortOp) {
        self.potential_op_oprefs.insert(key.to_opref());
        self.potential_ops.insert(key, pop);
    }

    /// Add a pure operation as a short-box candidate.
    /// shortpreamble.py: sb.add_pure_op(op)
    pub fn add_pure_op(&mut self, ctx: &mut crate::optimizeopt::OptContext, op: Op) {
        let result = op.pos.get();
        self.add_potential_op(ctx, self.lookup_label_arg(result), op, PreambleOpKind::Pure);
    }

    /// shortpreamble.py:369-374 add_heap_op(op, getfield_op)
    ///
    /// `op.pos` is the box the GETFIELD/GETARRAYITEM produces. If that box is
    /// a constant, route to `const_short_boxes` (RPython:
    /// `if isinstance(op, Const): self.const_short_boxes.append(HeapOp(op, getfield_op))`).
    /// Otherwise it joins `potential_ops` as a heap candidate.
    pub fn add_heap_op(&mut self, ctx: &mut crate::optimizeopt::OptContext, op: Op) {
        let result = op.pos.get();
        if result.is_constant() || self.known_constants.contains(&result) {
            // shortpreamble.py:371-373: const_short_boxes.append(HeapOp(...))
            let label_arg_idx = self.lookup_label_arg(result);
            self.const_short_boxes.push(PreambleOp {
                res: ctx.materialize_operand_at(op.pos.get()),
                op: std::rc::Rc::new(op),
                kind: PreambleOpKind::Heap,
                label_arg_idx,
                invented_name: false,
                same_as_source: None,
            });
            return;
        }
        self.add_potential_op(ctx, self.lookup_label_arg(result), op, PreambleOpKind::Heap);
    }

    /// Add a loop-invariant call as a short-box candidate.
    pub fn add_loopinvariant_op(&mut self, ctx: &mut crate::optimizeopt::OptContext, op: Op) {
        let result = op.pos.get();
        self.add_potential_op(
            ctx,
            self.lookup_label_arg(result),
            op,
            PreambleOpKind::LoopInvariant,
        );
    }

    pub(crate) fn add_short_input_arg(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        arg: OpRef,
        arg_type: majit_ir::Type,
    ) {
        // shortpreamble.py:255-259 parity: ShortInputArg's BoxType is the
        // intrinsic `box.type` (BoxInt → same_as_i, ref box → same_as_r,
        // BoxFloat → same_as_f). A `Void` reaching here is a parity
        // violation: RPython has no Void value Box / InputArgVoid class.
        if arg_type == majit_ir::Type::Void {
            panic!(
                "short preamble inputarg {arg:?} resolved to Type::Void; \
                 ShortInputArg requires an int/ref/float value box \
                 (shortpreamble.py:255-259)"
            );
        }
        // shortpreamble.py:258 `self.short_inputargs.append(renamed)`: one
        // renamed box per call, in caller order. The production caller
        // (optimizer.rs preview loop) iterates `label_args + virtuals`
        // (unroll.py:479) in order, appending one renamed box per slot, so
        // this call's slot is `short_inputargs.len()`, captured before the
        // push below.
        let live_slot = self.short_inputargs.len();
        // shortpreamble.py:256 `box = label_args[i]` — register the original
        // so `is_reachable` / `produce_arg` membership resolve it.
        // `with_label_args` pre-seeds `label_args`; the standalone
        // `create_short_boxes` path (empty `label_args`) appends here. A box
        // duplicated across `label_args + virtuals` (a virtual field that
        // coincides with a label arg) is registered once.
        if self.lookup_label_arg(arg).is_none() {
            self.label_args.push(arg);
        }
        // shortpreamble.py:257 `renamed = OpHelpers.inputarg_from_tp(box.type)`
        // — a FRESH InputArg distinct from the original `box`. Its position
        // comes from the op-position counter so it is unique and accounted
        // for by `opref_high_water`; `raw()` shares the op/inputarg integer
        // space, so a fresh op position is a fresh inputarg position too.
        // Mint the `InputArg` object itself and BIND the box to it
        // (`from_bound_inputarg`), the orthodox shape of `inputarg_from_tp`'s
        // fresh `AbstractInputArg`; the strong `Rc` is rooted in
        // `short_inputarg_refs[i]` so the box stays bound (the previous
        // `new_inputarg` left `inputarg_handle = None`, minting a position-only
        // box that shed to `Operand::Box`). `to_opref` is byte-identical
        // (reads only `(type, position)`).
        let pos = ctx.alloc_op_position_typed(arg_type).raw();
        let renamed_ia = majit_ir::InputArg::from_type_rc(arg_type, pos);
        let renamed = renamed_ia.opref();
        // shortpreamble.py:259 `self.potential_ops[box] = ShortInputArg(...)`
        // is a plain dict assignment, so for a box duplicated across the
        // combined list it OVERWRITES: the LAST slot's `ShortInputArg`
        // survives and `produce_arg` returns `short_inputargs[LAST]`, leaving
        // the FIRST slot's renamed box a dead Label arg — never produced,
        // never given info (shortpreamble.py:414-417 sets info only on
        // produced boxes). pyre mirrors that by stamping `label_arg_idx =
        // live_slot` (this call's slot) on the `potential_ops` entry below;
        // the later duplicate call overwrites with its later slot, so
        // `produce_arg` returns the same LAST-slot renamed box. In the
        // duplicate-free case `live_slot == lookup_label_arg(arg)`, so the
        // rename is unchanged.
        self.short_inputargs.push(renamed);
        self.short_inputarg_refs.push(renamed_ia);
        // shortpreamble.py:257 `ShortInputArg(box, renamed)` — `res` is the
        // original `box`; the SAME_AS replay arg is that box. Exported
        // entries carry original positions and are renamed to the matching
        // `short_inputargs` slot at import (`produced_short_boxes_from_exported_boxes`).
        // The operand's identity is the canonical `_forwarded` host Rc
        // (registered on first materialization), so every later
        // `materialize_operand_at(arg)` lookup returns the same key (ptr_eq).
        let arg_res = ctx.materialize_operand_at(arg);
        let mut same_as = Op::new(OpCode::same_as_for_type(arg_type), &[arg_res.clone()]);
        same_as.pos.set(arg);
        // shortpreamble.py:259 `self.potential_ops[box] = ShortInputArg(...)`
        // — keyed by the label-arg Box itself; `arg_res` is its canonical
        // (producer-bound) operand, shared with `res`.
        self.potential_op_oprefs.insert(arg_res.to_opref());
        self.potential_ops.insert(
            arg_res.clone(),
            PotentialShortOp::Preamble(PreambleOp {
                res: arg_res,
                op: std::rc::Rc::new(same_as),
                kind: PreambleOpKind::InputArg,
                label_arg_idx: Some(live_slot),
                invented_name: false,
                same_as_source: None,
            }),
        );
    }

    /// shortpreamble.py:285/294 `return ...preamble_op`: for a ShortInputArg
    /// the produced `preamble_op` IS the renamed inputarg box
    /// (shortpreamble.py:257 `ShortInputArg(box, renamed)`). pyre stores the
    /// renamed boxes in `short_inputargs`, paired with each label/virtual slot
    /// by `label_arg_idx` (`lookup_label_arg`), so produce_arg returns that
    /// renamed box — embedding the rename into the exported short op args at
    /// export time, the same substitution the import rename pass performed
    /// (`produced_short_boxes_from_exported_boxes`, `short_inputargs[position
    /// of arg in label_args]`).
    fn renamed_short_inputarg(&self, label_arg_idx: Option<usize>) -> majit_ir::operand::Operand {
        let idx = label_arg_idx
            .expect("InputArg short box missing label_arg_idx (set by add_short_input_arg)");
        majit_ir::operand::Operand::from_bound_inputarg(&self.short_inputarg_refs[idx])
    }

    fn produce_arg(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        opref: OpRef,
    ) -> Option<majit_ir::operand::Operand> {
        // shortpreamble.py:284 `if op in self.produced_short_boxes` — the
        // dict membership is Box identity; resolve the position to its
        // canonical box once for both identity-keyed checks. Const args
        // never key either set (they route to the Const arm below).
        if !opref.is_constant() {
            let okey = ctx.materialize_operand_at(opref);
            if let Some(existing) = self.produced_short_boxes.get(&okey) {
                // shortpreamble.py:285 `return ...preamble_op` — the
                // dependency's replay op object itself, so preamble-op
                // args carry the dep replay handle.
                //
                // ShortInputArg: upstream `preamble_op` IS the renamed
                // inputarg box (shortpreamble.py:257 `ShortInputArg(box,
                // renamed)`), so produce_arg returns the renamed
                // short_inputargs box for this slot — the export-time
                // rename. (`existing.res` is the ORIGINAL box, kept only
                // as the info-lookup key, shortpreamble.py:417.)
                if existing.kind == PreambleOpKind::InputArg {
                    let label_arg_idx = existing.label_arg_idx;
                    return Some(self.renamed_short_inputarg(label_arg_idx));
                }
                return Some(majit_ir::operand::Operand::from_bound_op(
                    &existing.preamble_op,
                ));
            }
            if self.boxes_in_production.contains(&okey) {
                return None;
            }
        }
        // shortpreamble.py:288 isinstance(op, Const) → return op.
        if opref.is_constant() {
            return Some(majit_ir::operand::Operand::from_opref(opref));
        }
        // pyre tracks iteration-known constants (body-typed OpRefs proven
        // constant for this pass) in `known_constants`; those are this
        // stage's `Const` boxes, mirroring `use_box`/`insert_dep_recursive`.
        if self.known_constants.contains(&opref) {
            return Some(ctx.materialize_operand_at(opref));
        }
        if self.potential_op_oprefs.contains(&opref) {
            // shortpreamble.py:291-294 `r = self.add_op_to_short(...);
            // return r.preamble_op`. ShortInputArg: renamed short_inputargs
            // box, see the produced arm above.
            let produced = self.materialize_one(ctx, opref)?;
            if produced.kind == PreambleOpKind::InputArg {
                return Some(self.renamed_short_inputarg(produced.label_arg_idx));
            }
            return Some(majit_ir::operand::Operand::from_bound_op(
                &produced.preamble_op,
            ));
        }
        // shortpreamble.py:295-296 `else: return None`. Every label arg is
        // registered as a ShortInputArg in `potential_ops`
        // (`add_short_input_arg`, create_short_boxes:255-259), so a label arg
        // is reached through the produced/potential arms above; an opref that
        // is none of those is not produce-able.
        None
    }

    /// shortpreamble.py:298-309 _pick_op_index
    ///
    /// Pick which compound-op alternative becomes the canonical short box.
    /// Heap ops are last-resort; the first non-Heap candidate wins on the
    /// initial `pick_other=True` pass. If two non-Heap candidates are seen,
    /// recurse with `pick_other=False`, which restricts the second pass to
    /// `ShortInputArg` candidates only (the highest-priority class).
    /// Falls back to index 0 when no candidate qualifies.
    fn _pick_op_index(lst: &[ProducedShortOp], pick_other: bool) -> usize {
        let mut index: Option<usize> = None;
        for (i, item) in lst.iter().enumerate() {
            let prefer = !matches!(item.kind, PreambleOpKind::Heap)
                && (pick_other || item.kind == PreambleOpKind::InputArg);
            if prefer {
                if index.is_some() {
                    debug_assert!(pick_other, "second-tier ambiguity");
                    return Self::_pick_op_index(lst, false);
                }
                index = Some(i);
            }
        }
        index.unwrap_or(0)
    }

    fn materialize_one(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        result: OpRef,
    ) -> Option<ProducedShortOp> {
        // shortpreamble.py:311-339 add_op_to_short — guard, cycle set,
        // and final insert all key on `shortop.res` Box identity.
        let okey = ctx.materialize_operand_at(result);
        if let Some(existing) = self.produced_short_boxes.get(&okey) {
            return Some(existing.clone());
        }
        if self.boxes_in_production.contains(&okey) {
            return None;
        }
        let candidate = self.potential_ops.get(&okey)?.clone();
        self.boxes_in_production.insert(okey.clone());
        let produced = candidate.add_op_to_short(self, ctx);
        self.boxes_in_production.swap_remove(&okey);
        let produced = produced?;
        self.produced_short_boxes.insert(okey, produced.clone());
        Some(produced)
    }

    /// shortpreamble.py: produced_short_boxes after add_op_to_short().
    pub fn produced_ops(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> Vec<(OpRef, ProducedShortOp)> {
        let keys: Vec<OpRef> = self
            .potential_ops
            .iter()
            .map(|(k, _)| k.to_opref())
            .collect();
        for key in keys {
            let _ = self.materialize_one(ctx, key);
        }
        self.produced_short_boxes
            .iter()
            .map(|(k, v)| (k.to_opref(), v.clone()))
            .collect()
    }

    /// shortpreamble.py:246-281 ShortBoxes.create_short_boxes
    ///
    /// Materialize all `potential_ops` (already populated by
    /// `produce_potential_short_preamble_ops` calls on each pass) into
    /// `produced_short_boxes`, then walk `const_short_boxes`: for each
    /// constant heap read, try to produce its struct argument; if that
    /// succeeds, emit a `getfield` short op whose first arg is the
    /// produced preamble box. Constant heap reads do not allocate fresh
    /// names — they reuse the original constant box.
    ///
    /// `label_args` / `label_arg_types` populate `short_inputargs` via
    /// `add_short_input_arg` (RPython:
    /// `self.potential_ops[box] = ShortInputArg(box, renamed)`).
    pub fn create_short_boxes(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        label_args: &[OpRef],
        label_arg_types: &[majit_ir::Type],
    ) -> Vec<ProducedShortOp> {
        // shortpreamble.py:255-259: register every label arg as a
        // ShortInputArg potential op.
        for (i, &arg) in label_args.iter().enumerate() {
            // shortpreamble.py:256 reads label_args[i].type intrinsically;
            // pyre's parallel array must match label_args length.
            let arg_type = label_arg_types.get(i).copied().unwrap_or_else(|| {
                panic!(
                    "missing label_arg_types[{}] (label_args.len()={}): \
                     create_short_boxes needs the parallel type list in \
                     lockstep with label_args",
                    i,
                    label_args.len()
                )
            });
            self.add_short_input_arg(ctx, arg, arg_type);
        }

        // shortpreamble.py:261: optimizer.produce_potential_short_preamble_ops(self)
        // — caller must have invoked this on each pass before calling
        // create_short_boxes (majit threads passes externally).

        // shortpreamble.py:263-267: short_boxes = []; for shortop in potential_ops.values(): add_op_to_short
        let mut short_boxes: Vec<ProducedShortOp> = self
            .produced_ops(ctx)
            .into_iter()
            .map(|(_, op)| op)
            .collect();

        // shortpreamble.py:272-280: walk const_short_boxes and try to
        // produce a struct preamble arg, then emit the getfield op.
        let const_pending: Vec<PreambleOp> = std::mem::take(&mut self.const_short_boxes);
        for short_op in const_pending {
            let getfield_op = &short_op.op;
            if getfield_op.num_args() == 0 {
                continue;
            }
            let struct_arg = getfield_op.arg(0);
            let Some(preamble_arg) = self.produce_arg(ctx, struct_arg.to_opref()) else {
                continue;
            };
            // shortpreamble.py:277-278: copy_and_change(opnum, [preamble_arg] + args[1:])
            let mut new_args = vec![preamble_arg];
            new_args.extend_from_slice(&getfield_op.getarglist()[1..]);
            let mut new_op = Op::with_descr(
                getfield_op.opcode,
                &new_args,
                getfield_op
                    .getdescr()
                    .unwrap_or_else(|| panic!("const_short_boxes heap op without descr")),
            );
            new_op.pos.set(getfield_op.pos.get());
            // shortpreamble.py:279: ProducedShortOp(short_op, preamble_op)
            short_boxes.push(ProducedShortOp {
                res: ctx.materialize_operand_at(getfield_op.pos.get()),
                preamble_op: std::rc::Rc::new(new_op),
                kind: PreambleOpKind::Heap,
                invented_name: false,
                same_as_source: None,
                label_arg_idx: None,
            });
        }
        short_boxes
    }

    /// shortpreamble.py:343-344 `create_short_inputargs(label_args)`:
    ///
    /// ```python
    /// def create_short_inputargs(self, label_args):
    ///     return self.short_inputargs
    ///     # ... rest of function is dead code after this early return
    /// ```
    ///
    /// Unconditionally returns `self.short_inputargs`. The pyre fallback
    /// to `label_args.to_vec()` on empty was a TODO — empty
    /// short_inputargs is the legitimate empty-loop case in upstream.
    pub fn create_short_inputargs(&self, _label_args: &[OpRef]) -> Vec<OpRef> {
        self.short_inputargs.clone()
    }

    /// The `InputArg` rooting pool paired 1:1 with `create_short_inputargs`'s
    /// boxes. Cloned (`Rc::clone` per handle) in lock-step so the carried
    /// strong `Rc`s keep each `short_inputargs[i]` box bound downstream.
    pub fn create_short_inputarg_refs(&self) -> Vec<majit_ir::InputArgRc> {
        debug_assert_eq!(
            self.short_inputarg_refs.len(),
            self.short_inputargs.len(),
            "short_inputarg_refs must stay index-aligned with short_inputargs"
        );
        self.short_inputarg_refs.clone()
    }

    /// shortpreamble.py: add_potential_op(op, pop)
    /// Add a produced operation to the short boxes at the given position.
    pub fn add_potential_op(
        &mut self,
        ctx: &mut crate::optimizeopt::OptContext,
        label_arg_idx: Option<usize>,
        op: Op,
        kind: PreambleOpKind,
    ) {
        let result = op.pos.get();
        // shortpreamble.py:290 `self.potential_ops[op]` — keyed by the
        // producer's result Box; resolve the position to its canonical
        // operand. Its identity is the canonical `_forwarded` host Rc
        // (registered on first materialization), so the insert key here and
        // the lookup keys in `materialize_one`/`produce_arg` are ptr_eq.
        let key = ctx.materialize_operand_at(result);
        let pop = PotentialShortOp::Preamble(PreambleOp {
            res: key.clone(),
            op: std::rc::Rc::new(op),
            kind,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
        let next = match self.potential_ops.get(&key) {
            Some(prev) => PotentialShortOp::Compound(CompoundOp {
                res: result,
                one: Box::new(pop),
                two: Box::new(prev.clone()),
            }),
            None => pop,
        };
        self.add_op(key, next);
    }
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
    /// shortpreamble.py:414 `ShortPreambleBuilder.label_args` — the
    /// OpRefs that the Label carries. Index in this list IS the label
    /// arg index. Lookup uses linear scan (label_args is small).
    label_args: Vec<OpRef>,
}

impl CollectedExtendedShortPreambleBuilder {
    pub fn new() -> Self {
        CollectedExtendedShortPreambleBuilder {
            guards: Vec::new(),
            heap_ops: Vec::new(),
            pure_ops: Vec::new(),
            loopinvariant_ops: Vec::new(),
            label_args: Vec::new(),
        }
    }

    /// Set the label args mapping.
    pub fn set_label_args(&mut self, label_args: &[OpRef]) {
        self.label_args.clear();
        self.label_args.extend_from_slice(label_args);
    }

    fn lookup_label_arg(&self, opref: OpRef) -> Option<usize> {
        self.label_args.iter().position(|&a| a == opref)
    }

    /// Add a guard operation.
    pub fn add_guard(&mut self, op: Op) {
        let label_arg_idx = self.lookup_label_arg(op.pos.get());
        self.guards.push(PreambleOp {
            res: majit_ir::operand::Operand::bound_from_opref(op.pos.get()),
            op: std::rc::Rc::new(op),
            kind: PreambleOpKind::Guard,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a pure operation.
    pub fn add_pure_op(&mut self, op: Op) {
        let label_arg_idx = self.lookup_label_arg(op.pos.get());
        self.pure_ops.push(PreambleOp {
            res: majit_ir::operand::Operand::bound_from_opref(op.pos.get()),
            op: std::rc::Rc::new(op),
            kind: PreambleOpKind::Pure,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a heap read.
    pub fn add_heap_op(&mut self, op: Op) {
        let label_arg_idx = self.lookup_label_arg(op.pos.get());
        self.heap_ops.push(PreambleOp {
            res: majit_ir::operand::Operand::bound_from_opref(op.pos.get()),
            op: std::rc::Rc::new(op),
            kind: PreambleOpKind::Heap,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a loop-invariant call.
    pub fn add_loopinvariant_op(&mut self, op: Op) {
        let label_arg_idx = self.lookup_label_arg(op.pos.get());
        self.loopinvariant_ops.push(PreambleOp {
            res: majit_ir::operand::Operand::bound_from_opref(op.pos.get()),
            op: std::rc::Rc::new(op),
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
        let Self {
            guards,
            heap_ops,
            pure_ops,
            loopinvariant_ops,
            label_args,
        } = self;
        let all_ops: Vec<PreambleOp> = guards
            .into_iter()
            .chain(heap_ops)
            .chain(pure_ops)
            .chain(loopinvariant_ops)
            .collect();

        let entries = all_ops
            .into_iter()
            .map(|preamble_op| {
                let mut arg_mapping = Vec::new();
                for (arg_pos, arg_ref) in preamble_op.op.getarglist().iter().enumerate() {
                    if let Some(label_idx) =
                        label_args.iter().position(|a| *a == arg_ref.to_opref())
                    {
                        arg_mapping.push((arg_pos, label_idx));
                    }
                }
                let mut fail_arg_mapping = Vec::new();
                if let Some(fail_args) = preamble_op.op.getfailargs() {
                    for (fail_arg_pos, fail_arg_ref) in fail_args.iter().enumerate() {
                        if let Some(label_idx) = label_args
                            .iter()
                            .position(|a| *a == fail_arg_ref.to_opref())
                        {
                            fail_arg_mapping.push((fail_arg_pos, label_idx));
                        }
                    }
                }
                ShortPreambleOp {
                    op: (*preamble_op.op).clone(),
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
            constants: majit_ir::ConstMap::new(),
            phase1_inputargs: None,
            inputarg_infos: Vec::new(),
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
        ctx: &mut crate::optimizeopt::OptContext,
        mut produced: Vec<ProducedShortOp>,
    ) -> Vec<ProducedShortOp> {
        match self.one.as_ref() {
            PotentialShortOp::Compound(compound) => {
                produced = compound.flatten(sb, ctx, produced);
            }
            PotentialShortOp::Preamble(op) => {
                if let Some(pop) = op.add_op_to_short(sb, ctx) {
                    produced.push(pop);
                }
            }
        }
        match self.two.as_ref() {
            PotentialShortOp::Compound(compound) => compound.flatten(sb, ctx, produced),
            PotentialShortOp::Preamble(op) => {
                if let Some(pop) = op.add_op_to_short(sb, ctx) {
                    produced.push(pop);
                }
                produced
            }
        }
    }
}

/// shortpreamble.py: ProducedShortOp — wraps a short op with its
/// preamble counterpart for emission during bridge compilation.
///
/// The upstream `ShortInputArg` role (shortpreamble.py:223-240) is
/// played by `PreambleOp { kind: InputArg }` entries; the separate
/// caller-less struct mirror was removed.
#[derive(Clone, Debug)]
pub struct ProducedShortOp {
    /// The short op classification.
    pub kind: PreambleOpKind,
    /// `short_op.res` (shortpreamble.py:58/110/151/224) — the result operand
    /// this short op produces. `add_preamble_op` reads it back as the
    /// `PreambleOp.op` operand (upstream `produce_op` passes `self.res`).
    /// Always a producer-bound / const operand (`materialize_operand_at` on
    /// the preview, `res.bound_op()`-rooted exported entries per #173), never
    /// position-only — so the same operand reaches the `expand_info`
    /// boundary, `Rc::ptr_eq`-stable on the producer.
    pub res: majit_ir::operand::Operand,
    /// The preamble operation to replay.
    pub preamble_op: majit_ir::OpRc,
    /// Whether this short op uses an invented SameAs result.
    pub invented_name: bool,
    /// Original result this invented name aliases.
    /// Carried as an [`Operand`] so the canonical producer-bound operand
    /// travels with the struct; see [`PreambleOp::same_as_source`].
    pub same_as_source: Option<majit_ir::operand::Operand>,
    /// Slot of this short box's result within the original
    /// `label_args + virtuals`, i.e. `lookup_label_arg(canonical_result)`
    /// carried over from [`PreambleOp::label_arg_idx`] across the export
    /// boundary. `None` for a result that is not a label/virtual slot. Lets
    /// the importer resolve the body-visible result slot directly instead of
    /// matching the result against a parallel originals array.
    pub label_arg_idx: Option<usize>,
}

/// Phase B B.1: helper used by `ProducedShortOp::produce_op` to seed a
/// fresh constant-pool slot in the importing trace for a Const arg seen
/// in the imported short op. Mirrors the inline `imported_const_opref`
/// closure inside the legacy `import_short_preamble_ops` (unroll.rs:3510).
fn imported_const_opref(
    imported_constants: &mut indexmap::IndexMap<OpRef, OpRef>,
    source: OpRef,
    value: &majit_ir::Value,
) -> OpRef {
    if let Some(&opref) = imported_constants.get(&source) {
        return opref;
    }
    // history.py:227/268/314 Const{Int,Float,Ptr}.value inline — fresh
    // imported short-preamble constant lands inline in `op.args` rather
    // than indexing the legacy pool. Slice 7b op-graph walker covers
    // ConstPtr slots across minor collection.
    let opref = match value {
        majit_ir::Value::Int(v) => OpRef::const_int(*v),
        majit_ir::Value::Float(v) => OpRef::const_float(*v),
        majit_ir::Value::Ref(v) => OpRef::const_ptr(*v),
        majit_ir::Value::Void => panic!("imported_const_opref: ConstVoid is not a value type"),
    };
    // ConstInt/Float/Ptr value rides inline on `opref` (history.py:227/
    // 268/314); no `seed_constant` step (its const arm is a no-op).
    imported_constants.insert(source, opref);
    opref
}

/// shortpreamble.py:283 `ShortBoxes.produce_arg` — classify an arg in
/// `produced.preamble_op.args` as Slot/Const/Produced.  RPython has a
/// single classification path; majit shares this function between
/// `ProducedShortOp::produce_op` (shortpreamble.rs) and
/// `OptContext::initialize_imported_short_preamble_builder_from_short_boxes`
/// (mod.rs) so the two consume sites cannot drift.
///
/// - Slot: `arg ∈ short_inputargs` (positional) → Phase 2 OpRef from `short_args`
/// - Const: `arg ∈ short_box_const_values` (producer-snapshotted) or `arg`
///   has known consumer-side constant value → seed fresh consumer-side slot
/// - Produced: `arg ∈ produced_results` (a previously imported producer's source)
pub(crate) fn classify_short_arg(
    ctx: &mut crate::optimizeopt::OptContext,
    arg: OpRef,
    short_inputargs: &[OpRef],
    short_args: &[OpRef],
    produced_results: &indexmap::IndexMap<OpRef, OpRef>,
    imported_constants: &mut indexmap::IndexMap<OpRef, OpRef>,
    short_box_const_values: &indexmap::IndexMap<OpRef, majit_ir::Value>,
) -> Option<crate::optimizeopt::ImportedShortPureArg> {
    if let Some(slot) = short_inputargs.iter().position(|i| *i == arg) {
        return short_args
            .get(slot)
            .copied()
            .map(crate::optimizeopt::ImportedShortPureArg::OpRef);
    }
    // Const lookup priority: producer snapshot first (handles bridges and
    // unit-test consumer ctxs without pre-seeded const pool), then consumer
    // ctx (production: pre-seeded at optimizer.rs:1927).
    if let Some(value) = short_box_const_values.get(&arg).cloned().or_else(|| {
        ctx.get_box_replacement_operand_opt(arg)
            .and_then(|cb| cb.const_value())
    }) {
        let const_opref = imported_const_opref(imported_constants, arg, &value);
        return Some(crate::optimizeopt::ImportedShortPureArg::Const(
            value,
            const_opref,
        ));
    }
    produced_results
        .get(&arg)
        .copied()
        .map(crate::optimizeopt::ImportedShortPureArg::OpRef)
}

impl ProducedShortOp {
    /// shortpreamble.py:212-214 ProducedShortOp.produce_op + per-kind dispatch:
    /// - PureOp.produce_op (shortpreamble.py:112-126)
    /// - HeapOp.produce_op (shortpreamble.py:62-85, getfield + getarrayitem)
    /// - LoopInvariantOp.produce_op (shortpreamble.py:152-159)
    /// - ShortInputArg.produce_op (shortpreamble.py:233-234, no-op)
    ///
    /// Mutates `ctx` to register the imported preamble op into the
    /// optimizer's per-kind side-table (`imported_short_pure_ops` for Pure,
    /// `set_preamble_field` / `set_preamble_item` on PtrInfo for Heap,
    /// `imported_loop_invariant_results` for LoopInvariant), mirroring the
    /// legacy `import_short_preamble_ops` per-arm body. Records the resolved
    /// result OpRef in `produced_results` keyed by `self.preamble_op.pos`
    /// so successor entries can reference it.
    ///
    /// Phase B B.1 (parallel implementation, dead code): the legacy
    /// enum-dispatch path in `import_short_preamble_ops` (unroll.rs:3504-3925)
    /// remains the active caller. B.2 wires this method as the sole produce-op
    /// driver. B.3+ retire the legacy enum.
    ///
    /// Returns `None` when:
    /// - args cannot be fully classified, mirroring legacy
    ///   `collect_exported_short_ops` skip;
    /// - the kind is non-emit (Heap with non-getfield/getarrayitem opcode).
    pub fn produce_op(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &indexmap::IndexMap<OpRef, OpRef>,
        produced_results: &mut indexmap::IndexMap<OpRef, OpRef>,
        imported_constants: &mut indexmap::IndexMap<OpRef, OpRef>,
        short_box_const_values: &indexmap::IndexMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let result = match self.kind {
            PreambleOpKind::Pure => self.produce_pure(
                ctx,
                short_inputargs,
                short_args,
                result_map,
                produced_results,
                imported_constants,
                short_box_const_values,
            )?,
            PreambleOpKind::Heap => match self.preamble_op.opcode {
                OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => self
                    .produce_heap_field(
                        ctx,
                        exported_infos,
                        short_inputargs,
                        short_args,
                        result_map,
                        produced_results,
                        imported_constants,
                        short_box_const_values,
                    )?,
                OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcF => self
                    .produce_heap_array_item(
                        ctx,
                        exported_infos,
                        short_inputargs,
                        short_args,
                        result_map,
                        produced_results,
                        imported_constants,
                        short_box_const_values,
                    )?,
                _ => return None,
            },
            PreambleOpKind::LoopInvariant => self.produce_loop_invariant(
                ctx,
                short_inputargs,
                short_args,
                result_map,
                produced_results,
                imported_constants,
                short_box_const_values,
            )?,
            // shortpreamble.py:233-234 ShortInputArg.produce_op asserts
            // `not invented_name` and otherwise does nothing; the source pos
            // is already in `short_inputargs`, no Phase 2 OpRef to record.
            PreambleOpKind::InputArg => {
                debug_assert!(
                    !self.invented_name,
                    "shortpreamble.py:234: ShortInputArg cannot have invented_name"
                );
                return None;
            }
            PreambleOpKind::Guard => return None,
        };
        produced_results.insert(self.preamble_op.pos.get(), result);
        Some(result)
    }

    /// shortpreamble.py:112-126 PureOp.produce_op
    fn produce_pure(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &indexmap::IndexMap<OpRef, OpRef>,
        produced_results: &mut indexmap::IndexMap<OpRef, OpRef>,
        imported_constants: &mut indexmap::IndexMap<OpRef, OpRef>,
        short_box_const_values: &indexmap::IndexMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos.get();
        // Result OpRef was fixed before ShortPreambleBuilder construction,
        // matching RPython's object identity being available before
        // `ProducedShortOp.produce_op` runs.
        let result_opref = *result_map.get(&source)?;
        let opcode = self.preamble_op.opcode;
        // shortpreamble.py:116-120 line-by-line:
        //
        //   if invented_name:
        //       op = self.orig_op.copy_and_change(self.orig_op.getopnum())
        //       op.set_forwarded(self.res)
        //   else:
        //       op = self.res
        //
        // PyPy's `self.res` for the invented arm is `new_name` — the alt
        // SAME_AS Box created by `add_op_to_short` at shortpreamble.py:327.
        // It is a fresh body-Box that gets emitted into the trace via
        // `add_preamble_op` and lives at its own (alt-specific) position.
        // The transient `op = orig_op.copy(...)` is the PreambleOp wrapper:
        // `op.set_forwarded(self.res)` routes body refs from the wrapper
        // straight to the alt's body slot.
        //
        // pyre's flat-OpRef analog allocates the wrapper at `source` (the
        // synthetic alias from the compound-dedup pass at
        // shortpreamble.rs:478-491) and the alt body slot at `result_opref`
        // (= `result_map[source]`, the body-visible OpRef this alt's
        // imported Pure / SAME_AS will live at). Forward `source ->
        // result_opref` so body refs after `force_op_from_preamble`
        // resolve to the alt's own body identity, NOT to the canonical
        // (`same_as_source`) — collapsing both alt and canonical onto the
        // canonical's body ref erases PyPy's invented-name Box identity
        // and corrupts the SAME_AS in `extra_same_as`.
        // Non-invented re-uses self.res directly without forwarding.
        if self.invented_name {
            let op_source = ctx
                .get_box_replacement_operand_opt(source)
                .unwrap_or_else(|| ctx.materialize_operand_at(source));
            let op_result = ctx
                .get_box_replacement_operand_opt(result_opref)
                .unwrap_or_else(|| ctx.materialize_operand_at(result_opref));
            ctx.make_equal_to(&op_source, &op_result);
        }
        // `result_opref` is a typed synthetic alias minted by
        // `add_op_to_short` via `ctx.alloc_op_position_typed(arg_type)`
        // — its variant carries `Box.type` from the chosen producer
        // (history.py:802 `record_same_as` parity). Downstream type
        // lookups read it directly via `OpRef::ty()` or the producing
        // SAME_AS body op's `op.type_` once it lands in
        // `new_operations`.
        let args = self
            .preamble_op
            .getarglist()
            .iter()
            .map(|arg| {
                classify_short_arg(
                    ctx,
                    arg.to_opref(),
                    short_inputargs,
                    short_args,
                    produced_results,
                    imported_constants,
                    short_box_const_values,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        // shortpreamble.py:121-126: PureOp.produce_op routes through
        // `optpure.pure(...)` (or `extra_call_pure` for calls). majit's
        // single-table `imported_short_pure_ops` covers both because
        // `pure.rs` consults it for both arms during `optimize_pure_op` and
        // `optimize_call_pure_*`.
        // shortpreamble.py:121 `PreambleOp(op, preamble_op, ...)` — the
        // replay op is the SAME object ShortPreambleBuilder.__init__
        // seeded (one ResOperation per short box, threaded end to end).
        // The builder was installed by
        // `initialize_imported_short_preamble_builder_from_short_boxes`
        // just before this produce_op loop; reuse its replay Rc instead
        // of rebuilding an equal-content op. Fallback keeps the local
        // construction for harnesses that call produce_op standalone.
        let builder_pop = ctx
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(&self.res));
        let imported = match builder_pop {
            Some(p) => {
                debug_assert_eq!(
                    p.preamble_op.pos.get(),
                    if self.invented_name {
                        result_opref
                    } else {
                        source
                    },
                    "builder replay pos diverged from produce_pure replay rule"
                );
                crate::optimizeopt::ImportedShortPureOp {
                    opcode,
                    descr: self.preamble_op.getdescr(),
                    args,
                    result: result_opref,
                    pop: crate::optimizeopt::info::PreambleOp {
                        op: ctx.materialize_operand_at(source),
                        invented_name: self.invented_name,
                        preamble_op: p.preamble_op.clone(),
                        same_as_source: self.same_as_source.clone(),
                    },
                }
            }
            None => crate::optimizeopt::ImportedShortPureOp::new(
                ctx,
                opcode,
                self.preamble_op.getdescr(),
                args,
                result_opref,
                source,
                self.invented_name,
                self.same_as_source.clone(),
            ),
        };
        ctx.imported_short_pure_ops.push(imported);
        // shortpreamble.py:432-440 add_preamble_op + 437-438 extra_same_as:
        // RPython collects the SameAs op into `short_preamble_producer.extra_same_as`
        // lazily at use-box time (force_op_from_preamble path).  majit's
        // `used_imported_short_aliases()` derives the alias list directly
        // from `imported_short_preamble_builder.extra_same_as()` at the same
        // phase boundary, so an eager `imported_short_aliases.push` here
        // would be a TODO dual write.
        //
        // Path B parity (B.6.7): Heap/Array/LoopInvariant produce_* return
        // `source` so successor short-op dependency args resolve through
        // `produced_results` to Phase 1 source-namespace (matching body
        // refs after `force_op_from_preamble_op` returns `preamble_source`).
        // produce_pure follows the same convention for consistency — the
        // body-visible `result_opref` is still registered (Box.type, the
        // ImportedShortPureOp `result` field) but is no longer the value
        // that successor entries see via produced_results.
        let _ = result_opref;
        Some(source)
    }

    /// shortpreamble.py:62-79 HeapOp.produce_op (getfield case)
    fn produce_heap_field(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &indexmap::IndexMap<OpRef, OpRef>,
        produced_results: &indexmap::IndexMap<OpRef, OpRef>,
        imported_constants: &mut indexmap::IndexMap<OpRef, OpRef>,
        short_box_const_values: &indexmap::IndexMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos.get();
        let result_type = self.preamble_op.result_type();
        let descr = self.preamble_op.getdescr()?;
        // Object arg classification — Slot or Const only (RPython
        // shortpreamble.py:91-95 add_op_to_short uses `produce_arg`,
        // which admits Slot/Const).  We accept Produced too for completeness.
        let object_arg = self.preamble_op.arg(0);
        let obj_class = classify_short_arg(
            ctx,
            object_arg.to_opref(),
            short_inputargs,
            short_args,
            produced_results,
            imported_constants,
            short_box_const_values,
        )?;
        let obj = match obj_class {
            crate::optimizeopt::ImportedShortPureArg::OpRef(r) => r,
            crate::optimizeopt::ImportedShortPureArg::Const(_, r) => r,
        };
        // Cat-2.2 alignment: forward `source -> result_opref` so body refs
        // after `force_op_from_preamble_op` resolve to the body-visible
        // OpRef without relying on the use-before-def assembly adaptation.
        // PyPy `HeapOp.produce_op` stores `PreambleOp(op=self.res,
        // preamble_op, invented_name)` in field cache; self.res is the
        // canonical body Box (Box identity makes it body-visible). pyre's
        // flat-OpRef analog uses `result_opref = result_map[source]` as the
        // body-visible slot.
        let result_opref = *result_map.get(&source)?;
        let _ = result_type;
        let descr_idx = descr.index();
        let obj_resolved = ctx.get_replacement_opref(obj);
        // shortpreamble.py:66-68: if g.getarg(0) in exported_infos:
        //     setinfo_from_preamble(g.getarg(0), exported_infos[...])
        // Pass the Rc handle (unroll.py:61 identity preservation).
        if let Some(crate::optimizeopt::info::OpInfo::Ptr(rc)) = exported_infos.get(&object_arg) {
            ctx.setinfo_from_preamble(obj_resolved, rc, Some(exported_infos));
        }
        let mut getfield_op = Op::new(
            OpCode::getfield_for_type(result_type),
            &[ctx.materialize_operand_at(obj_resolved)],
        );
        getfield_op.setdescr(descr.clone());
        // Cat-2.2 dual-slot rule (mod.rs:1817 replay_pos): replay.pos =
        // result_opref because `make_equal_to(source, result_opref)` installed
        // below clobbers source's slot. Seed info at result_opref's slot
        // (set_preamble_forwarded_info via replay_pos) so
        // `take_preamble_forwarded_opinfo(preamble_op.preamble_op.pos)`
        // reads it back. PyPy `preamble_op.set_forwarded(info)` lives on a
        // distinct Op object, so this slot juggling is the pyre adaptation
        // of that Box-identity invariant.
        getfield_op.pos.set(result_opref);
        // shortpreamble.py:75 `PreambleOp(self.res, preamble_op, ...)` —
        // the stored replay is the builder's object (one ResOperation per
        // short box); see produce_pure for the threading rationale.
        let replay_rc = ctx
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(&self.res))
            .map(|p| {
                debug_assert_eq!(
                    p.preamble_op.pos.get(),
                    result_opref,
                    "builder replay pos diverged from produce_heap_field rule"
                );
                p.preamble_op
            })
            .unwrap_or_else(|| std::rc::Rc::new(getfield_op.clone()));
        let pop = crate::optimizeopt::info::PreambleOp {
            // PreambleOp.op carries the Box itself (shortpreamble.py:12).
            op: ctx.materialize_operand_at(source),
            invented_name: self.invented_name,
            preamble_op: replay_rc,
            same_as_source: self.same_as_source.clone(),
        };
        let parent_descr = getfield_op
            .with_field_descr(|fd| fd.get_parent_descr())
            .flatten();
        if let Some(info) = ctx.get_const_info_mut(obj_resolved, parent_descr) {
            info.set_preamble_field(descr_idx, pop.clone());
        }
        // shortpreamble.py:72-74: ensure_ptr_info_arg0 + setfield(pop)
        let pop_for_field = pop.clone();
        ctx.with_ensured_ptr_info_arg0(&getfield_op, |mut struct_info| {
            if let Some(mut info) = struct_info.as_mut() {
                debug_assert!(
                    !info.is_virtual(),
                    "shortpreamble.py:74: imported heap field on virtual"
                );
                info.set_preamble_field(descr_idx, pop_for_field);
            }
        });
        // Cat-2.2 alignment: forward `source -> result_opref` after the
        // PtrInfo / const-info side tables have been seeded (so the seeds
        // see the unforwarded source key consistent with `pop.op = source`).
        let op_source = ctx
            .get_box_replacement_operand_opt(source)
            .unwrap_or_else(|| ctx.materialize_operand_at(source));
        let op_result = ctx
            .get_box_replacement_operand_opt(result_opref)
            .unwrap_or_else(|| ctx.materialize_operand_at(result_opref));
        ctx.make_equal_to(&op_source, &op_result);
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }

    /// shortpreamble.py:80-85 HeapOp.produce_op (getarrayitem case)
    fn produce_heap_array_item(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &indexmap::IndexMap<OpRef, OpRef>,
        produced_results: &indexmap::IndexMap<OpRef, OpRef>,
        imported_constants: &mut indexmap::IndexMap<OpRef, OpRef>,
        short_box_const_values: &indexmap::IndexMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos.get();
        let result_type = self.preamble_op.result_type();
        let descr = self.preamble_op.getdescr()?;
        let object_arg = self.preamble_op.arg(0);
        let obj_class = classify_short_arg(
            ctx,
            object_arg.to_opref(),
            short_inputargs,
            short_args,
            produced_results,
            imported_constants,
            short_box_const_values,
        )?;
        let obj = match obj_class {
            crate::optimizeopt::ImportedShortPureArg::OpRef(r) => r,
            crate::optimizeopt::ImportedShortPureArg::Const(_, r) => r,
        };
        // shortpreamble.py:81 `g.getarg(1).getint()`: read the integer
        // VALUE of the index Const, not the OpRef raw bits.
        // `OpRef::raw()` returns the trace-namespace tagged u32 — it is
        // NOT the constant integer.  Resolve via classify_short_arg
        // which checks the producer snapshot (`short_box_const_values`)
        // first, then the consumer ctx const pool.
        let index_arg = self.preamble_op.arg(1);
        let index = match classify_short_arg(
            ctx,
            index_arg.to_opref(),
            short_inputargs,
            short_args,
            produced_results,
            imported_constants,
            short_box_const_values,
        )? {
            crate::optimizeopt::ImportedShortPureArg::Const(majit_ir::Value::Int(v), _) => v,
            _ => return None,
        };
        // Cat-2.2 alignment: symmetric to produce_heap_field. Forward
        // `source -> result_opref` so body refs after
        // `force_op_from_preamble_op` resolve to the body-visible OpRef
        // without relying on the use-before-def assembly adaptation.
        let result_opref = *result_map.get(&source)?;
        let _ = result_type;
        let obj_resolved = ctx.get_replacement_opref(obj);
        // shortpreamble.py:68-71 applies to both getfield and
        // getarrayitem: if the base object has exported info, import it
        // before ensuring heap/array PtrInfo.
        // Pass the Rc handle (unroll.py:61 identity preservation).
        if let Some(crate::optimizeopt::info::OpInfo::Ptr(rc)) = exported_infos.get(&object_arg) {
            ctx.setinfo_from_preamble(obj_resolved, rc, Some(exported_infos));
        }
        let index_const = ctx.make_constant_int(index);
        let mut getarrayitem_op = Op::new(
            OpCode::getarrayitem_for_type(result_type),
            &[
                ctx.materialize_operand_at(obj_resolved),
                ctx.materialize_operand_at(index_const),
            ],
        );
        getarrayitem_op.setdescr(descr.clone());
        // Cat-2.2 dual-slot rule (mod.rs:1817 replay_pos): replay.pos =
        // result_opref. See produce_heap_field for the Box-identity-vs-flat-OpRef
        // adaptation rationale.
        getarrayitem_op.pos.set(result_opref);
        // shortpreamble.py:84 `PreambleOp(self.res, preamble_op, ...)` —
        // stored replay is the builder's object; see produce_pure.
        let replay_rc = ctx
            .imported_short_preamble_builder
            .as_ref()
            .and_then(|b| b.produced_short_op(&self.res))
            .map(|p| {
                debug_assert_eq!(
                    p.preamble_op.pos.get(),
                    result_opref,
                    "builder replay pos diverged from produce_heap_array_item rule"
                );
                p.preamble_op
            })
            .unwrap_or_else(|| std::rc::Rc::new(getarrayitem_op.clone()));
        let pop = crate::optimizeopt::info::PreambleOp {
            // PreambleOp.op carries the Box itself (shortpreamble.py:12).
            op: ctx.materialize_operand_at(source),
            invented_name: self.invented_name,
            preamble_op: replay_rc,
            same_as_source: self.same_as_source.clone(),
        };
        let obj_box = ctx.get_box_replacement_operand_opt(obj_resolved);
        if obj_resolved.is_constant()
            || obj_box
                .as_ref()
                .and_then(|b| ctx.get_constant_box(b))
                .is_some()
        {
            if let Some(info) = obj_box
                .as_ref()
                .and_then(|b| ctx.get_const_info_array_mut_box(b, descr.clone()))
            {
                info.set_preamble_item(index as usize, pop.clone());
            }
        } else {
            let pop_for_array = pop.clone();
            ctx.with_ensured_ptr_info_arg0(&getarrayitem_op, |mut array_info| {
                if let Some(mut info) = array_info.as_mut() {
                    if let crate::optimizeopt::info::PtrInfo::Array(array_info) = &mut *info {
                        let _ = array_info.lenbound.make_gt_const(index);
                        let idx = index as usize;
                        if idx >= array_info.items.len() {
                            array_info.items.resize(
                                idx + 1,
                                crate::optimizeopt::info::FieldEntry::Value(Operand::None),
                            );
                        }
                        array_info.items[idx] =
                            crate::optimizeopt::info::FieldEntry::Preamble(pop_for_array);
                    }
                }
            });
        }
        // Cat-2.2 alignment: forward `source -> result_opref` after the
        // const-info / ArrayPtrInfo side tables have been seeded.
        let op_source = ctx
            .get_box_replacement_operand_opt(source)
            .unwrap_or_else(|| ctx.materialize_operand_at(source));
        let op_result = ctx
            .get_box_replacement_operand_opt(result_opref)
            .unwrap_or_else(|| ctx.materialize_operand_at(result_opref));
        ctx.make_equal_to(&op_source, &op_result);
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }

    /// shortpreamble.py:152-159 LoopInvariantOp.produce_op
    fn produce_loop_invariant(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &indexmap::IndexMap<OpRef, OpRef>,
        produced_results: &indexmap::IndexMap<OpRef, OpRef>,
        imported_constants: &mut indexmap::IndexMap<OpRef, OpRef>,
        short_box_const_values: &indexmap::IndexMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos.get();
        let result_type = self.preamble_op.result_type();
        // shortpreamble.py:156-158 reads `self.res.getarg(0).getint()`
        // from the original Const box. In majit the const may only be
        // available through the producer-side snapshot, so classify it
        // through the same path as Pure/Heap args.
        let func_arg = classify_short_arg(
            ctx,
            self.preamble_op.arg(0).to_opref(),
            short_inputargs,
            short_args,
            produced_results,
            imported_constants,
            short_box_const_values,
        )?;
        let func_ptr = match func_arg {
            crate::optimizeopt::ImportedShortPureArg::Const(majit_ir::Value::Int(v), _) => v,
            _ => return None,
        };
        // Cat-2.2 alignment probe: forward `source -> result_opref` so body
        // refs after `force_op_from_preamble_op` resolve to the body-visible
        // OpRef without relying on the use-before-def assembly adaptation.
        // PyPy `LoopInvariantOp.produce_op` stores `PreambleOp(op=self.res,
        // preamble_op, invented_name)` in `loop_invariant_results`; self.res
        // is the canonical body Box. pyre's flat-OpRef analog uses
        // `result_opref = result_map[source]` as the body-visible slot and
        // installs the `source -> result_opref` forwarding here so the
        // consumer at `rewrite.rs:2809` resolves source to result_opref via
        // get_box_replacement uniformly.
        let result_opref = *result_map.get(&source)?;
        let _ = result_type;
        let op_source = ctx
            .get_box_replacement_operand_opt(source)
            .unwrap_or_else(|| ctx.materialize_operand_at(source));
        let op_result = ctx
            .get_box_replacement_operand_opt(result_opref)
            .unwrap_or_else(|| ctx.materialize_operand_at(result_opref));
        ctx.make_equal_to(&op_source, &op_result);
        // `rewrite.py:31` `self.opt.loop_invariant_results[key] = old_op` —
        // dict-as-map semantics; pyre's Vec-backed parity overwrites the
        // entry when `func_ptr` already exists (PyPy dict behavior),
        // otherwise appends.
        if let Some(entry) = ctx
            .imported_loop_invariant_results
            .iter_mut()
            .find(|(k, _)| *k == func_ptr)
        {
            entry.1 = source;
        } else {
            ctx.imported_loop_invariant_results.push((func_ptr, source));
        }
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }
}

#[derive(Clone, Debug, Default)]
struct AbstractShortPreambleBuilderState {
    short: Vec<majit_ir::OpRc>,
    short_results: IndexSet<OpRef>,
    used_boxes: Vec<OpRef>,
    short_preamble_jump: Vec<majit_ir::OpRc>,
    extra_same_as: Vec<Op>,
    /// shortpreamble.py:430 `self.short_inputargs = short_inputargs` —
    /// the renamed InputArg positions; reused verbatim as the Label args
    /// (shortpreamble.py:443 `ResOperation(rop.LABEL, self.short_inputargs[:])`).
    short_inputargs: Vec<OpRef>,
    /// Known constant OpRefs. In RPython, isinstance(box, Const) is a type
    /// check. In majit, constant OpRefs must be explicitly tracked.
    known_constants: IndexSet<OpRef>,
    /// B.6.4 canonical dedup for `record_imported_preamble_use`.
    /// `produced_short_boxes` is a dual-key map (source key + result_opref
    /// key both pointing at the same `ProducedShortOp`), so the source vs.
    /// body-visible distinction is not enough — RPython's Box identity
    /// makes one Box equal one slot regardless of how it is reached. The
    /// canonical key is `replay_op.pos` (a stable proxy for `self.res`):
    /// dedup here prevents two different lookup keys from pushing the
    /// same RPython Box twice into `used_boxes` /
    /// `short_preamble_jump` / `extra_same_as`.
    recorded_canonical_results: IndexSet<OpRef>,
}

impl AbstractShortPreambleBuilderState {
    /// shortpreamble.py:432-440: add_preamble_op
    ///
    /// RPython (4 lines):
    ///   op = preamble_op.op.get_box_replacement()
    ///   if preamble_op.invented_name:
    ///       self.extra_same_as.append(op)
    ///   self.used_boxes.append(op)
    ///   self.short_preamble_jump.append(preamble_op.preamble_op)
    ///
    /// `same_as_source`: the original OpRef this invented name aliases.
    fn record_imported_preamble_use(
        &mut self,
        op: majit_ir::operand::Operand,
        replay_op: &majit_ir::OpRc,
        needs_alias: bool,
        same_as_source: Option<majit_ir::operand::Operand>,
    ) {
        if !self.recorded_canonical_results.insert(replay_op.pos.get()) {
            return;
        }
        if needs_alias {
            // shortpreamble.py:436-437: extra_same_as carries the resolved
            // box itself; a producer-bound box sheds to a live operand.
            //
            // Measured dead fallback: every `invented_name = true` writer
            // carries `Some(same_as_source)` — the sole production writer
            // is the compound-alias path (`alt.invented_name = true` +
            // `alt.same_as_source = Some(...)`), `add_preamble_op_from_pop`
            // passes the pop's threaded `same_as_source`, and the
            // export/re-import paths copy both fields as a pair. Fallback
            // kept as a release safety net.
            debug_assert!(
                same_as_source.is_some(),
                "needs_alias without same_as_source at {:?}",
                replay_op.pos.get()
            );
            let source = same_as_source.unwrap_or_else(|| op.clone());
            let mut same_as = Op::new(
                OpCode::same_as_for_type(replay_op.result_type()),
                &[source.clone()],
            );
            same_as.pos.set(op.to_opref());
            self.extra_same_as.push(same_as);
        }
        self.used_boxes.push(op.to_opref());
        self.short_preamble_jump.push(replay_op.clone());
    }

    fn record_preamble_use(
        &mut self,
        result: majit_ir::operand::Operand,
        produced: &ProducedShortOp,
    ) {
        self.record_imported_preamble_use(
            result,
            &produced.preamble_op,
            produced.invented_name,
            produced.same_as_source.clone(),
        );
    }

    /// Internal: append preamble_op to short (with ovf guard).
    /// Used by add_op_to_short (recursive export-time path).
    fn append_to_short(&mut self, _result: OpRef, produced: &ProducedShortOp) -> majit_ir::OpRc {
        let canonical_result = produced.preamble_op.pos.get();
        if self.short_results.contains(&canonical_result) {
            return produced.preamble_op.clone();
        }
        let preamble_op = produced.preamble_op.clone();
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short
                .push(std::rc::Rc::new(Op::new(OpCode::GuardNoOverflow, &[])));
        }
        preamble_op
    }

    /// shortpreamble.py:382-407: use_box(box, preamble_op, optimizer)
    /// Non-recursive: iterates preamble_op's args (adding non-input deps
    /// + guards to short), then appends preamble_op + result guards.
    /// Called by force_op_from_preamble (unroll.py:32).
    ///
    /// Dependency args carry the dep's replay op object (produce_arg
    /// object-carry); a non-input, non-const arg whose bound op still
    /// holds the builder's `set_forwarded` marker IS a short-box replay
    /// op — append it and consume the marker (upstream
    /// `arg.set_forwarded(None)`, shortpreamble.py:391-396).
    fn use_box(
        &mut self,
        preamble_op: &majit_ir::OpRc,
        already_in_short: &IndexSet<OpRef>,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) -> Op {
        let canonical_result = preamble_op.pos.get();
        if self.short_results.contains(&canonical_result)
            || already_in_short.contains(&canonical_result)
        {
            return (**preamble_op).clone();
        }
        // shortpreamble.py:383-396: iterate preamble_op args
        for arg in preamble_op.getarglist().iter() {
            let arg_opref = arg.to_opref();
            if self.short_results.contains(&arg_opref)
                || already_in_short.contains(&arg_opref)
                || self.short_inputargs.iter().any(|a| *a == arg_opref)
                || self.known_constants.contains(&arg_opref)
            {
                continue;
            }
            // shortpreamble.py:390-396: `arg.get_forwarded() is None` →
            // pass; otherwise append the arg (the dep replay op itself)
            // and consume the marker.
            let Some(dep) = arg.bound_op() else { continue };
            if matches!(
                &*dep.forwarded.borrow(),
                majit_ir::forwarding::Forwarded::None
            ) {
                continue;
            }
            *dep.forwarded.borrow_mut() = majit_ir::forwarding::Forwarded::None;
            let dep_canonical = dep.pos.get();
            if !self.short_results.contains(&dep_canonical)
                && !already_in_short.contains(&dep_canonical)
            {
                self.short_results.insert(dep_canonical);
                self.short.push(dep.clone());
                if dep.opcode.is_ovf() {
                    self.short
                        .push(std::rc::Rc::new(Op::new(OpCode::GuardNoOverflow, &[])));
                }
            }
        }
        // shortpreamble.py:389,396: info.make_guards(arg, self.short, optimizer)
        self.short
            .extend(arg_guards.iter().cloned().map(std::rc::Rc::new));
        // shortpreamble.py:398: self.short.append(preamble_op)
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short
                .push(std::rc::Rc::new(Op::new(OpCode::GuardNoOverflow, &[])));
        }
        // shortpreamble.py:401-402: `info = preamble_op.get_forwarded();
        // preamble_op.set_forwarded(None)` — consume the own marker so a
        // later consumer's arg walk doesn't re-append this op.
        *preamble_op.forwarded.borrow_mut() = majit_ir::forwarding::Forwarded::None;
        // shortpreamble.py:405-406: info.make_guards(preamble_op, self.short, optimizer)
        self.short
            .extend(result_guards.iter().cloned().map(std::rc::Rc::new));
        (**preamble_op).clone()
    }
}

fn build_short_preamble_struct_from_ops(
    short_inputargs: &[OpRef],
    ops: &[majit_ir::OpRc],
    used_boxes: &[OpRef],
    jump_args: &[OpRef],
) -> ShortPreamble {
    let inputarg_idx =
        |arg: &OpRef| -> Option<usize> { short_inputargs.iter().position(|a| a == arg) };
    let entries = ops
        .iter()
        .map(|op| (**op).clone())
        .map(|op| {
            let arg_mapping = op
                .getarglist()
                .iter()
                .enumerate()
                .filter_map(|(arg_pos, arg_ref)| {
                    inputarg_idx(&arg_ref.to_opref()).map(|label_idx| (arg_pos, label_idx))
                })
                .collect();
            let fail_arg_mapping = op
                .getfailargs()
                .map(|fail_args| {
                    fail_args
                        .iter()
                        .enumerate()
                        .filter_map(|(fail_arg_pos, fail_arg_ref)| {
                            inputarg_idx(&fail_arg_ref.to_opref())
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
    // history.py:227/268/314 — `Const{Int,Float,Ptr}.value` rides inline on
    // the OpRef, so short-preamble ops embed the constant value directly in
    // `op.args` and need no parallel side table. The captured `constants`
    // map is empty along every production path; readers (`is_constant()`
    // short-circuits at every callsite) treat it as a guaranteed-empty
    // compatibility fallback retained only for legacy fixtures.
    //
    // RPython parity: `shortpreamble.py` keeps no `loop_constants` side
    // table — `arg` IS the `Const` box, so `_map_args(mapping, args)`
    // (`unroll.py:364`) passes it through unchanged.
    let constants: majit_ir::ConstMap<majit_ir::Const> = majit_ir::ConstMap::new();
    ShortPreamble {
        ops: entries,
        inputargs: short_inputargs.to_vec(),
        used_boxes: used_boxes.to_vec(),
        jump_args: jump_args.to_vec(),
        exported_state: None,
        constants,
        phase1_inputargs: None,
        inputarg_infos: Vec::new(),
    }
}

/// shortpreamble.py: ShortPreambleBuilder
///
/// Builds the replayable short preamble from exported short boxes, while also
/// collecting `used_boxes`, `short_preamble_jump`, and `extra_same_as`.
#[derive(Clone, Debug)]
pub struct ShortPreambleBuilder {
    state: AbstractShortPreambleBuilderState,
    /// shortpreamble.py:250 `self.produced_short_boxes = {}` — keyed by the
    /// short-box res Box identity (`shortop.res`), looked up by box everywhere
    /// (produce_arg/use_box/add_op_to_short). #146/S8 re-keyed this from the
    /// flat-OpRef position (which needed a dual source/result_opref key for
    /// invented names) to the single carried res box: `self.res` at the
    /// cross-peel produce loop, `materialize_operand_at(pos)` at the single-op
    /// re-export. The carried box is invariant to the replay-position aliasing
    /// the dual key compensated for, so the two entries collapse to one. The
    /// PYRE_S8B_HARNESS census measured this lookup agreeing with the former
    /// position key on every live firing across the bench corpus.
    produced_short_boxes: IndexMap<majit_ir::operand::Operand, ProducedShortOp>,
}

impl ShortPreambleBuilder {
    pub fn new(
        label_args: &[OpRef],
        short_boxes: &[(majit_ir::operand::Operand, ProducedShortOp)],
        short_inputargs: &[OpRef],
    ) -> Self {
        let mut produced_short_boxes = IndexMap::new();
        for (k, v) in short_boxes {
            // shortpreamble.py:414-425: __init__ plants
            // `preamble_op.set_forwarded(info)` on every replay op. The
            // exported infos themselves are seeded through ctx position
            // slots (`set_preamble_forwarded_info`), so the on-object
            // marker is the `empty_info` sentinel: its presence tells
            // `use_box` "this operand is a short-box replay op", and
            // consuming it (`set_forwarded(None)`) is the dedup.
            // `OpInfo::EmptyInfo` never appears in exported infos
            // (mod.rs guard), so the marker is unambiguous; Op::clone
            // resets `forwarded`, so built ShortPreamble copies never
            // carry it.
            *v.preamble_op.forwarded.borrow_mut() = majit_ir::forwarding::Forwarded::Info(
                crate::optimizeopt::info::OpInfo::EmptyInfo(crate::optimizeopt::info::EmptyInfo),
            );
            // Const res boxes are ptr-unstable (minted fresh per resolution),
            // so they can never be a stable box-identity key; export already
            // filters const short boxes (optimizer.rs:2942) so this is inert
            // for the live path, and the single-op re-export passes the
            // memoized `materialize_operand_at(pos)` box.
            if k.to_opref().is_constant() {
                continue;
            }
            produced_short_boxes.insert(k.clone(), v.clone());
        }
        // shortpreamble.py:430 `self.short_inputargs = short_inputargs` —
        // store the caller's renamed positions themselves. The empty
        // fallback (test paths without an exported preview list) mints
        // position refs from the label args once.
        let short_inputargs = if short_inputargs.is_empty() {
            label_args.to_vec()
        } else {
            short_inputargs.to_vec()
        };
        ShortPreambleBuilder {
            state: AbstractShortPreambleBuilderState {
                short_inputargs,
                ..AbstractShortPreambleBuilderState::default()
            },
            produced_short_boxes,
        }
    }

    pub fn note_known_constant(&mut self, opref: OpRef) {
        self.state.known_constants.insert(opref);
    }

    fn use_box_recursive(
        &mut self,
        result: &majit_ir::operand::Operand,
        visiting: &mut IndexSet<majit_ir::operand::Operand>,
    ) -> Option<majit_ir::OpRc> {
        let produced = self.produced_short_boxes.get(result)?.clone();
        let canonical_result = produced.preamble_op.pos.get();
        if self.state.short_results.contains(&canonical_result) {
            return Some(produced.preamble_op);
        }
        if !visiting.insert(result.clone()) {
            return None;
        }
        for arg in produced.preamble_op.getarglist().iter() {
            // shortpreamble.py:288 isinstance(arg, Const) → skip
            if arg.to_opref().is_constant() {
                continue;
            }
            // shortpreamble.py:284-285 `if op in self.produced_short_boxes`:
            // the dependency check is by the arg Box identity.
            if self.produced_short_boxes.get(arg).is_some() {
                let _ = self.use_box_recursive(arg, visiting);
            }
        }
        visiting.swap_remove(result);
        Some(self.state.append_to_short(result.to_opref(), &produced))
    }

    /// shortpreamble.py:310: add_op_to_short — recursive, used during
    /// export-time create_short_boxes to resolve transitive dependencies.
    pub fn add_op_to_short(&mut self, result: &majit_ir::operand::Operand) -> Option<Op> {
        self.use_box_recursive(result, &mut IndexSet::new())
            .map(|op| (*op).clone())
    }

    /// shortpreamble.py:382-407: use_box(box, preamble_op, optimizer)
    /// Non-recursive. Called by force_op_from_preamble (unroll.py:32).
    ///
    /// RPython passes `preamble_op.preamble_op` directly — the replay op
    /// IS the carried object, so there is no entry-selection lookup. The
    /// pop's replay Rc is the builder's own object (threaded by the
    /// produce_op family), verified by the debug probe below.
    pub fn use_box(
        &mut self,
        source: OpRef,
        preamble_op: &majit_ir::OpRc,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) {
        #[cfg(debug_assertions)]
        if let Some((_, produced)) = self
            .produced_short_boxes
            .iter()
            .find(|(_, p)| p.preamble_op.pos.get() == source)
        {
            debug_assert!(
                std::rc::Rc::ptr_eq(&produced.preamble_op, preamble_op),
                "use_box pop replay diverged from builder entry at {source:?}"
            );
        }
        #[cfg(not(debug_assertions))]
        let _ = source;
        self.state
            .use_box(preamble_op, &IndexSet::new(), arg_guards, result_guards);
    }

    /// shortpreamble.py:284-285 `op in self.produced_short_boxes`.
    ///
    /// This is the SOLE corpus-live lookup of any `produced_short_boxes` map
    /// (the ExtendedShortPreambleBuilder lookups never execute; this builder's
    /// `use_box_recursive`/`add_preamble_op` lookups never execute either —
    /// measured over the full bench corpus). The live callers reuse the
    /// builder's replay Rc during `produce_pure`/`produce_heap_field`/
    /// `produce_heap_array_item` re-export (pure.rs/shortpreamble.rs, via
    /// `OptContext.imported_short_preamble_builder`).
    ///
    /// #146/S8: keyed by the short-box res Box identity (`shortop.res`), the
    /// carried box the produce loop holds as `self.res`. This replaced the
    /// flat-OpRef position key, whose dual source/result_opref entries
    /// compensated for invented-name replay-position aliasing; the carried box
    /// is invariant to that aliasing so the entries collapse to one. The prior
    /// "blocked" verdict (75 agree / 114 diverge) measured the WRONG box:
    /// `get_box_replacement_box(source)`, a Phase-2 re-resolution that mints a
    /// fresh non-`ptr_eq` box. The carried Phase-1 res is shared across the
    /// peel boundary (an `Rc::clone` of the exported short-box res), so the
    /// box-identity lookup hits — PYRE_S8B_HARNESS measured 82/82 agreement
    /// with the former position key on every live firing across the corpus.
    pub fn produced_short_op(&self, res: &majit_ir::operand::Operand) -> Option<ProducedShortOp> {
        self.produced_short_boxes.get(res).cloned()
    }

    /// shortpreamble.py:432-440: add_preamble_op(preamble_op)
    /// Called from optimizer.force_box when popping from potential_extra_ops.
    ///
    /// RPython unconditionally uses the carried `preamble_op` with no
    /// produced_short_boxes lookup:
    ///   op = preamble_op.op.get_box_replacement()
    ///   if preamble_op.invented_name: self.extra_same_as.append(op)
    ///   self.used_boxes.append(op)
    ///   self.short_preamble_jump.append(preamble_op.preamble_op)
    ///
    /// This is that unconditional pattern (#149/S8f collapse): the prior
    /// produced_short_boxes map lookup is gone. `field_entry::PreambleOp` now
    /// carries the ORIGINAL box an invented-name CompoundOp alternate aliases
    /// (threaded from `ProducedShortOp.same_as_source` at the produce_* export
    /// sites), so the carried pop reproduces the former map entry's record:
    /// `op` resolves to `resolved_op`, and `invented_name` / `same_as_source`
    /// match — the SameAs emits `same_as(original)` rather than the old
    /// `same_as(op)` self-alias (op being the invented SameAs name).
    pub fn add_preamble_op_from_pop(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
        resolved_op: majit_ir::operand::Operand,
    ) {
        // shortpreamble.py:432-440: unconditional add_preamble_op. The carried
        // pop reproduces the builder map entry's record — `op` resolves to the
        // same `resolved_op`, and `invented_name` / `same_as_source` are
        // threaded through `field_entry::PreambleOp` — so the
        // produced_short_boxes lookup is no longer consulted here (#149/S8f).
        let replay_op = &preamble_op.preamble_op;
        // shortpreamble.py:435 `op = preamble_op.op.get_box_replacement()`:
        // for non-invented entries the resolved operand IS the preamble-defined
        // res, so the `used_boxes` label slot always has a producer at
        // label fall-through. pyre's Cat-2.2 heap import resolves through
        // `make_equal_to(source, result)` to a fresh body-visible OpRef with
        // NO preamble producer — the flat-OpRef analogue of an invented
        // name. Emit the same defining alias the invented arm uses
        // (`same_as(resolved) = carried preamble operand`) whenever the resolved
        // slot differs from the carried operand, so the extended label slot is
        // defined in the preamble exactly as upstream's box identity
        // guarantees.
        let (needs_alias, alias_source) = if preamble_op.invented_name {
            (true, preamble_op.same_as_source.clone())
        } else if resolved_op.to_opref() != preamble_op.op.to_opref() {
            (true, Some(preamble_op.op.clone()))
        } else {
            (false, None)
        };
        self.state
            .record_imported_preamble_use(resolved_op, replay_op, needs_alias, alias_source);
    }

    pub fn add_preamble_op(&mut self, result: &majit_ir::operand::Operand) -> bool {
        let Some(produced) = self.produced_short_boxes.get(result).cloned() else {
            return false;
        };
        // shortpreamble.py:435 `op = preamble_op.op.get_box_replacement()`
        // — the stored res box, not a fresh equal-positioned mint.
        let res = produced.res.clone();
        self.state.record_preamble_use(res, &produced);
        true
    }

    pub fn build_short_preamble(&self) -> Vec<Op> {
        let mut result = Vec::with_capacity(self.state.short.len() + 2);
        // shortpreamble.py:443 `ResOperation(rop.LABEL,
        // self.short_inputargs[:])` — the Label args are the stored
        // renamed-inputarg boxes themselves, not fresh mints.
        let short_inputargs_operand: Vec<majit_ir::operand::Operand> = self
            .state
            .short_inputargs
            .iter()
            .copied()
            .map(majit_ir::operand::Operand::bound_from_opref)
            .collect();
        result.push(Op::new(OpCode::Label, &short_inputargs_operand));
        result.extend(self.state.short.iter().map(|op| (**op).clone()));
        let jump_args_operand: Vec<majit_ir::operand::Operand> = self
            .state
            .short_preamble_jump
            .iter()
            .map(majit_ir::operand::Operand::from_bound_op)
            .collect();
        result.push(Op::new(OpCode::Jump, &jump_args_operand));
        result
    }

    pub fn build_short_preamble_struct(&self) -> ShortPreamble {
        let jump_args: Vec<OpRef> = self
            .state
            .short_preamble_jump
            .iter()
            .map(|op| op.pos.get())
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

    pub fn short_preamble_jump(&self) -> &[majit_ir::OpRc] {
        &self.state.short_preamble_jump
    }

    pub fn extra_same_as(&self) -> &[Op] {
        &self.state.extra_same_as
    }

    pub fn short_inputargs(&self) -> &[OpRef] {
        &self.state.short_inputargs
    }
}

/// shortpreamble.py:448-482: ExtendedShortPreambleBuilder
///
/// RPython parity: single `short` list with JUMP sentinel at end.
/// `use_box()` pops JUMP, appends deps/guards/op, re-appends JUMP.
#[derive(Clone, Debug)]
pub struct ExtendedShortPreambleBuilder {
    /// Stays `OpRef`-keyed. `setup()` populates this map (and the GC walk +
    /// constructor clone read it), but EVERY key-lookup of it
    /// (insert_dep_recursive / use_box_recursive / use_box /
    /// add_preamble_op_from_pop / add_preamble_op / produced_short_op) is dead
    /// over the full bench corpus — measured. A #146/S8 operand re-key here is
    /// therefore unverifiable (the gate cannot exercise the silent-miss
    /// surface), like the deferred vectorizer maps.
    produced_short_boxes: IndexMap<OpRef, ProducedShortOp>,
    short_inputargs: Vec<OpRef>,
    /// shortpreamble.py:460: self.short = short — single ops list (base + JUMP sentinel)
    short: Vec<Op>,
    /// Tracks which OpRefs are already in `short` (for dedup).
    short_results: IndexSet<OpRef>,
    /// Constants tracked for RPython isinstance(arg, Const) checks.
    known_constants: IndexSet<OpRef>,
    extra_same_as: Vec<Op>,
    short_preamble_jump: Vec<majit_ir::OpRc>,
    base_extra_same_as: Vec<Op>,
    label_args: Vec<OpRef>,
    used_boxes: Vec<OpRef>,
    short_jump_args: Vec<OpRef>,
    pub target_token: u64,
    /// RPython parity: remap Phase 1 preamble OpRefs → current inputargs.
    /// Values are the current-namespace boxes, bound to their producers at
    /// `setup()` insertion (the mapping values in unroll.py:396 are the
    /// jump-arg Box objects themselves), so the remap `setarg` writes
    /// produce live-tracking bound operands instead of frozen positions.
    phase1_to_inputarg: indexmap::IndexMap<OpRef, majit_ir::operand::Operand>,
    /// B.6.4 canonical dedup keyed by `produced.preamble_op.pos`. Mirrors
    /// `AbstractShortPreambleBuilderState.recorded_canonical_results` —
    /// `produced_short_boxes` carries dual entries (source-key plus
    /// result_opref-key) for the same RPython Box, so per-key dedup
    /// (`label_args` etc.) cannot catch a second add via the alternate
    /// key. RPython's Box identity collapses both paths to one entry.
    recorded_canonical_results: IndexSet<OpRef>,
}

impl ExtendedShortPreambleBuilder {
    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        fn visit_oprefs(refs: &mut [OpRef], visitor: &mut dyn FnMut(&mut GcRef)) {
            for r in refs {
                if let OpRef::ConstPtr(gcref) = r {
                    visitor(gcref);
                }
            }
        }

        fn visit_opref_set(set: &mut IndexSet<OpRef>, visitor: &mut dyn FnMut(&mut GcRef)) {
            let refs: Vec<OpRef> = set.iter().copied().collect();
            set.clear();
            for mut r in refs {
                if let OpRef::ConstPtr(gcref) = &mut r {
                    visitor(gcref);
                }
                set.insert(r);
            }
        }

        fn visit_produced(produced: &mut ProducedShortOp, visitor: &mut dyn FnMut(&mut GcRef)) {
            produced.preamble_op.walk_const_ptr_refs_mut(visitor);
            if let Some(source) = produced.same_as_source.as_ref() {
                source.walk_const_ptr_refs(visitor);
            }
        }

        for (_, produced) in self.produced_short_boxes.iter_mut() {
            visit_produced(produced, visitor);
        }
        visit_oprefs(&mut self.short_inputargs, visitor);
        for op in &mut self.short {
            op.walk_const_ptr_refs_mut(visitor);
        }
        visit_opref_set(&mut self.known_constants, visitor);
        for op in &mut self.extra_same_as {
            op.walk_const_ptr_refs_mut(visitor);
        }
        for op in &mut self.short_preamble_jump {
            op.walk_const_ptr_refs_mut(visitor);
        }
        for op in &mut self.base_extra_same_as {
            op.walk_const_ptr_refs_mut(visitor);
        }
        visit_oprefs(&mut self.label_args, visitor);
        visit_oprefs(&mut self.used_boxes, visitor);
        visit_oprefs(&mut self.short_jump_args, visitor);
        // phase1_to_inputarg keys are Phase 1 preamble OpRefs (op result
        // positions, never Const); only the bound values carry const GcRefs.
        for (_, target) in self.phase1_to_inputarg.iter() {
            target.walk_const_ptr_refs(visitor);
        }
        // recorded_canonical_results is keyed by `preamble_op.pos` (op result
        // positions, never Const) — nothing to walk.
    }

    pub fn new(target_token: u64, sb: &ShortPreambleBuilder) -> Self {
        ExtendedShortPreambleBuilder {
            // The live builder now keys `produced_short_boxes` by the short-box
            // res Box (#146/S8); this builder keys by `preamble_op.pos` (the
            // assert in `ensure_dep_from_produced`), so re-key on copy.
            produced_short_boxes: {
                let mut m = indexmap::IndexMap::new();
                for (_, p) in sb.produced_short_boxes.iter() {
                    m.insert(p.preamble_op.pos.get(), p.clone());
                }
                m
            },
            short_inputargs: sb.short_inputargs().to_vec(),
            short: Vec::new(),
            short_results: IndexSet::new(),
            known_constants: IndexSet::new(),
            extra_same_as: sb.extra_same_as().to_vec(),
            short_preamble_jump: Vec::new(),
            base_extra_same_as: sb.extra_same_as().to_vec(),
            label_args: Vec::new(),
            used_boxes: Vec::new(),
            short_jump_args: Vec::new(),
            target_token,
            phase1_to_inputarg: indexmap::IndexMap::new(),
            recorded_canonical_results: IndexSet::new(),
        }
    }

    /// shortpreamble.py:458-461: setup(jump_args, short, label_args)
    ///
    /// RPython parity: builds single `short` list from base ops + JUMP sentinel.
    /// For each base op, ensures missing deps from produced_short_boxes are
    /// inserted before the consumer (RPython use_box arg-handling parity).
    ///
    /// Returns `true` on success; `false` if any op references an
    /// unresolvable Phase 1 OpRef. RPython equivalent: `produce_arg`
    /// returning None propagates up to `add_op_to_short` returning None,
    /// and the entire short_op is dropped (shortpreamble.py:283-296,
    /// 311-341). Pyre's structural counterpart is to bail out of `setup`
    /// so the caller (`jump_to_existing_trace`) can fall back to
    /// `jump_to_preamble` instead of attempting to inline a broken short
    /// preamble.
    pub fn setup(
        &mut self,
        short_preamble: &ShortPreamble,
        label_args: &[OpRef],
        ctx: &mut crate::optimizeopt::OptContext,
    ) -> bool {
        // Build Phase 1 → current inputarg remap from arg_mapping. The
        // values bind to their current-namespace producers here, where the
        // ctx is available — the remap reads below are `&self`.
        self.phase1_to_inputarg.clear();
        for entry in &short_preamble.ops {
            for &(arg_pos, label_idx) in &entry.arg_mapping {
                if let Some(phase1_ref) = entry.op.getarglist().get(arg_pos) {
                    let phase1_ref = phase1_ref.to_opref();
                    if let Some(&current_inputarg) = label_args.get(label_idx) {
                        if phase1_ref != current_inputarg {
                            self.phase1_to_inputarg
                                .insert(phase1_ref, ctx.materialize_operand_at(current_inputarg));
                        }
                    }
                }
            }
        }
        // RPython parity: DO NOT mutate produced_short_boxes in-place.
        // RPython's setup() only sets self.short/jump_args/label_args;
        // the preamble producer (self) may be reused across multiple
        // setup() calls with different label_args. In-place remap would
        // corrupt the original Phase 1 args for subsequent calls.
        // Instead, remap on-the-fly when reading from produced_short_boxes.

        // Build single short list with inline dep resolution.
        let inputargs_set: IndexSet<OpRef> = label_args.iter().copied().collect();
        let constants_set: IndexSet<u32> = short_preamble.constants.keys().copied().collect();
        self.short.clear();
        self.short_results.clear();
        for entry in &short_preamble.ops {
            let mut op = entry.op.clone();
            // optimizer.py:651-652 setarg loop parity.
            for i in 0..op.num_args() {
                let arg = op.arg(i);
                if let Some(remapped) = self.phase1_to_inputarg.get(&arg.to_opref()) {
                    op.setarg(i, remapped.clone());
                }
            }
            // RPython use_box arg loop: insert missing deps before this op.
            // Recursive: deps of deps are also inserted (transitive closure).
            for arg in op.getarglist().iter() {
                let arg = arg.to_opref();
                if !self.insert_dep_recursive(arg, &inputargs_set, &constants_set) {
                    if crate::optimizeopt::majit_log_enabled() {
                        eprintln!(
                            "[jit] short_preamble setup: dropping inline (unresolved arg {:?} in op pos={:?} opcode={:?})",
                            arg,
                            op.pos.get(),
                            op.opcode
                        );
                    }
                    self.short.clear();
                    self.short_results.clear();
                    self.label_args = label_args.to_vec();
                    return false;
                }
            }
            self.short_results.insert(op.pos.get());
            self.short.push(op);
        }
        // JUMP sentinel at end (RPython: short[-1] is always JUMP)
        let mut jump_args = Vec::with_capacity(short_preamble.jump_args.len());
        let mut jump_args_operand = Vec::with_capacity(short_preamble.jump_args.len());
        for arg in &short_preamble.jump_args {
            if let Some(target) = self.phase1_to_inputarg.get(arg) {
                jump_args.push(target.to_opref());
                jump_args_operand.push(target.clone());
            } else {
                // Unmapped Phase 1 jump arg (no rename): keep the STATIC short
                // preamble position in `short_jump_args`. That position is
                // exactly the replay key that `inline_short_preamble` registers
                // in `mapping` (`mapping[sp_op.pos] = new_ref`, unroll.rs:3963)
                // or a seeded short-inputarg — mirroring RPython where
                // `short_jump_args = short[-1].getarglist()` references the same
                // Box identities as the short-op results (unroll.py:374).
                // Applying `get_box_replacement` here would diverge that
                // identity: under the tagged-int flip a loop-carried raw-Int
                // jump arg (e.g. IntOp(36)) box-forwards to its Phase-2
                // SameAsI alias (IntOp(247)), which is neither a replay key nor
                // a seeded inputarg, so `_map_args` (unroll.rs force loop) then
                // reads an unmapped key and panics. Forwarding is applied later,
                // AFTER the mapping lookup, via `get_replacement_opref` — the
                // orthodox `get_box_replacement(mapping[box])` order. Keep the
                // resolved operand only for the emitted JUMP sentinel's producer
                // binding.
                let operand = ctx
                    .get_box_replacement_operand_opt(*arg)
                    .unwrap_or_else(|| ctx.materialize_operand_at(*arg));
                jump_args.push(*arg);
                jump_args_operand.push(operand);
            }
        }
        self.short.push(Op::new(OpCode::Jump, &jump_args_operand));
        // Reset state
        self.extra_same_as = self.base_extra_same_as.clone();
        self.short_preamble_jump.clear();
        self.label_args = label_args.to_vec();
        self.used_boxes = short_preamble.used_boxes.clone();
        self.short_jump_args = jump_args;
        true
    }

    /// Recursively insert a dep (and its transitive deps) into self.short.
    /// Used by setup() to ensure all args of base ops are satisfied.
    /// Applies phase1_to_inputarg remap when reading from produced_short_boxes.
    ///
    /// Returns `true` if `arg` was resolved (or was already known); `false`
    /// if it was an unresolvable Phase 1 reference. Mirrors RPython
    /// `produce_arg → None` semantics (shortpreamble.py:283-296): a missing
    /// dep means the entire short preamble cannot be inlined cleanly, so
    /// the caller should bail out instead of leaving dangling args that
    /// later trip `inline_short_preamble`'s `_map_args` (unroll.py:404).
    fn insert_dep_recursive(
        &mut self,
        arg: OpRef,
        inputargs_set: &IndexSet<OpRef>,
        constants_set: &IndexSet<u32>,
    ) -> bool {
        // history.py:227/268/314 inline-Const variants short-circuit
        // before `arg.raw()` (which panics on inline) — covered by
        // `arg.is_constant()` below.
        if arg.is_constant() || arg.is_none() {
            return true;
        }
        if self.short_results.contains(&arg)
            || inputargs_set.contains(&arg)
            || self.known_constants.contains(&arg)
            || constants_set.contains(&arg.raw())
        {
            return true;
        }
        // shortpreamble.py:284-285 — `op in self.produced_short_boxes`,
        // keyed by Box identity. Every entry is keyed by its own
        // `preamble_op.pos` (produced_short_boxes_from_exported_boxes), so the
        // map key equals the producer position a dep arg references — the
        // direct lookup is exact, matching RPython's single Box-identity
        // lookup. The former `preamble_op.pos → key` reverse index was a
        // vestige of an earlier export that keyed by something other than the
        // replay pos; it can never fire now (key == pos by construction).
        let dep = self.produced_short_boxes.get(&arg).cloned();
        let Some(dep) = dep else {
            // Tripwire: a reverse `preamble_op.pos == arg` entry must not exist
            // when the direct lookup misses — that would mean some insert path
            // broke the `key == preamble_op.pos` invariant the deletion relies on.
            debug_assert!(
                !self
                    .produced_short_boxes
                    .iter()
                    .any(|(_, prod)| prod.preamble_op.pos.get() == arg),
                "produced_short_boxes key != preamble_op.pos at {arg:?}: \
                 direct lookup missed but a pos-keyed entry exists"
            );
            return false;
        };
        let dep_pos = dep.preamble_op.pos.get();
        if self.short_results.contains(&dep_pos) {
            return true;
        }
        // Remap dep args on-the-fly (don't mutate produced_short_boxes)
        let mut dep_op = (*dep.preamble_op).clone();
        // optimizer.py:651-652 setarg loop parity.
        for i in 0..dep_op.num_args() {
            let a = dep_op.arg(i);
            if let Some(remapped) = self.phase1_to_inputarg.get(&a.to_opref()) {
                dep_op.setarg(i, remapped.clone());
            }
        }
        // Recurse into dep's own args first (transitive). If any sub-dep
        // can't be resolved, bail out — the dep cannot be safely emitted.
        let dep_op_args = dep_op.getarglist_copy();
        for dep_arg in dep_op_args.iter() {
            let dep_arg = dep_arg.to_opref();
            if !self.insert_dep_recursive(dep_arg, inputargs_set, constants_set) {
                return false;
            }
        }
        self.short_results.insert(dep_pos);
        self.short.push(dep_op);
        if dep.preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        true
    }

    fn use_box_recursive(&mut self, result: OpRef, visiting: &mut IndexSet<OpRef>) -> Option<Op> {
        let produced = self.produced_short_boxes.get(&result)?.clone();
        let canonical_result = produced.preamble_op.pos.get();
        if self.short_results.contains(&canonical_result) {
            return Some((*produced.preamble_op).clone());
        }
        if !visiting.insert(result) {
            return None;
        }
        for arg in produced.preamble_op.getarglist().iter() {
            let arg = arg.to_opref();
            // shortpreamble.py:288 isinstance(arg, Const) → skip
            if arg.is_constant() {
                continue;
            }
            if self.produced_short_boxes.iter().any(|(k, _)| *k == arg) {
                let _ = self.use_box_recursive(arg, visiting);
            }
        }
        visiting.swap_remove(&result);
        // Append to self.short directly
        let preamble_op = (*produced.preamble_op).clone();
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        Some(preamble_op)
    }

    /// shortpreamble.py:465-476: add_preamble_op(preamble_op)
    ///
    /// RPython unconditionally appends to label_args and jump_args:
    ///   op = preamble_op.op.get_box_replacement()
    ///   self.label_args.append(op)
    ///   self.jump_args.append(preamble_op.preamble_op)
    /// shortpreamble.py:465-476: add_preamble_op(preamble_op)
    ///
    /// Extended version: label_args/jump_args instead of used_boxes.
    ///   op = preamble_op.op.get_box_replacement()
    ///   if preamble_op.invented_name:
    ///       self.extra_same_as.append(op)
    ///   self.label_args.append(op)
    ///   self.jump_args.append(preamble_op.preamble_op)
    pub fn add_preamble_op_from_pop(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
        resolved_op: majit_ir::operand::Operand,
    ) {
        let resolved_key = resolved_op.to_opref();
        let lookup_key = if self
            .produced_short_boxes
            .iter()
            .any(|(k, _)| *k == resolved_key)
        {
            resolved_key
        } else {
            preamble_op.op.to_opref()
        };
        if let Some(produced) = self.produced_short_boxes.get(&lookup_key).cloned() {
            self.add_tracked_preamble_op(resolved_op, &produced);
        } else {
            // shortpreamble.py:465-476: same pattern via replay_op.
            let replay_op = &preamble_op.preamble_op;
            if !self.recorded_canonical_results.insert(replay_op.pos.get()) {
                return;
            }
            let op = resolved_key;
            // shortpreamble.py:436-437 plus the Cat-2.2 fresh-slot case: a
            // non-invented pop whose resolved slot differs from the carried
            // preamble box has no preamble producer for the label slot (see
            // ShortPreambleBuilder::add_preamble_op_from_pop); emit the same
            // defining alias with the carried box as source.
            let alias_source = if preamble_op.invented_name {
                // shortpreamble.py:436-437: alias the carried original
                // (same_as_source), matching `add_tracked_preamble_op`; the
                // resolved operand is the release fallback when none was
                // threaded.
                debug_assert!(
                    preamble_op.same_as_source.is_some(),
                    "invented_name without same_as_source at {:?}",
                    replay_op.pos.get()
                );
                Some(
                    preamble_op
                        .same_as_source
                        .clone()
                        .unwrap_or_else(|| resolved_op.clone()),
                )
            } else if resolved_key != preamble_op.op.to_opref() {
                Some(preamble_op.op.clone())
            } else {
                None
            };
            if let Some(source) = alias_source {
                let mut same_as =
                    Op::new(OpCode::same_as_for_type(replay_op.result_type()), &[source]);
                same_as.pos.set(op);
                self.extra_same_as.push(same_as);
            }
            self.label_args.push(resolved_key);
            // The flattened struct only needs the replay position; the replay
            // op itself is rooted by short_preamble_jump.
            self.short_jump_args.push(replay_op.pos.get());
            self.short_preamble_jump.push(replay_op.clone());
        }
    }

    /// shortpreamble.py:471-477: add_preamble_op (internal)
    pub fn add_tracked_preamble_op(
        &mut self,
        result: majit_ir::operand::Operand,
        produced: &ProducedShortOp,
    ) {
        let current_result = produced.preamble_op.pos.get();
        if !self.recorded_canonical_results.insert(current_result) {
            return;
        }
        if produced.invented_name {
            // shortpreamble.py:436-437: the resolved box itself is the
            // SameAs source; a producer-bound box sheds to a live operand.
            let source = produced
                .same_as_source
                .clone()
                .unwrap_or_else(|| result.clone());
            let mut op = Op::new(
                OpCode::same_as_for_type(produced.preamble_op.result_type()),
                &[source.clone()],
            );
            op.pos.set(current_result);
            self.extra_same_as.push(op);
        }
        self.label_args.push(result.to_opref());
        self.used_boxes.push(produced.preamble_op.pos.get());
        self.short_jump_args.push(produced.preamble_op.pos.get());
        self.short_preamble_jump.push(produced.preamble_op.clone());
    }

    pub fn add_preamble_op(&mut self, result: OpRef) -> bool {
        let Some(produced) = self.produced_short_boxes.get(&result).cloned() else {
            return false;
        };
        // shortpreamble.py:466 `op = preamble_op.op.get_box_replacement()`
        // — the stored res box, not a fresh equal-positioned mint.
        let res = produced.res.clone();
        self.add_tracked_preamble_op(res, &produced);
        true
    }

    /// shortpreamble.py:310: add_op_to_short — recursive, export-time.
    pub fn add_op_to_short(&mut self, result: OpRef) -> Option<Op> {
        self.use_box_recursive(result, &mut IndexSet::new())
    }

    /// Remap a preamble op's args using phase1_to_inputarg (on-the-fly, no mutation).
    fn remap_op(&self, op: &Op) -> Op {
        if self.phase1_to_inputarg.is_empty() {
            return op.clone();
        }
        let mut remapped = op.clone();
        // optimizer.py:651-652 setarg loop parity.
        for i in 0..remapped.num_args() {
            let arg = remapped.arg(i);
            if let Some(r) = self.phase1_to_inputarg.get(&arg.to_opref()) {
                remapped.setarg(i, r.clone());
            }
        }
        remapped
    }

    /// shortpreamble.py:478-481: use_box — pop JUMP, add deps, re-append JUMP.
    /// Called by force_op_from_preamble (unroll.py:32).
    ///
    /// RPython passes `preamble_op.preamble_op` directly — the pop's
    /// replay Rc is the carried object (threaded by the produce_op
    /// family); the debug probe checks it against the builder entry.
    pub fn use_box(
        &mut self,
        source: OpRef,
        preamble_op: &majit_ir::OpRc,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) {
        #[cfg(debug_assertions)]
        if let Some(produced) = self.produced_short_boxes.get(&source) {
            debug_assert!(
                std::rc::Rc::ptr_eq(&produced.preamble_op, preamble_op),
                "ext use_box pop replay diverged from builder entry at {source:?}"
            );
        }
        #[cfg(not(debug_assertions))]
        let _ = source;
        let preamble_op = self.remap_op(preamble_op);
        let canonical = preamble_op.pos.get();
        // shortpreamble.py:479: jump_op = self.short.pop()
        let jump_op = self.short.pop();
        // shortpreamble.py:480: AbstractShortPreambleBuilder.use_box(...)
        if !self.short_results.contains(&canonical) {
            // Add deps for each arg. produced_short_boxes is keyed by each
            // entry's own `preamble_op.pos`, so a dep arg (a producer position)
            // hits its entry directly — RPython single Box-identity lookup. The
            // former `preamble_op.pos → key` reverse index can never fire
            // (key == pos by construction) and was removed.
            for arg in preamble_op.getarglist().iter() {
                let arg = arg.to_opref();
                if self.short_results.contains(&arg)
                    || self.short_inputargs.iter().any(|a| *a == arg)
                    || self.known_constants.contains(&arg)
                {
                    continue;
                }
                let dep = self.produced_short_boxes.get(&arg);
                if let Some(dep) = dep {
                    let dep_pos = dep.preamble_op.pos.get();
                    if !self.short_results.contains(&dep_pos) {
                        self.short_results.insert(dep_pos);
                        self.short.push(self.remap_op(&dep.preamble_op));
                        if dep.preamble_op.opcode.is_ovf() {
                            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
                        }
                    }
                }
            }
            self.short.extend_from_slice(arg_guards);
            self.short_results.insert(canonical);
            self.short.push(preamble_op.clone());
            if preamble_op.opcode.is_ovf() {
                self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
            }
            self.short.extend_from_slice(result_guards);
        }
        // shortpreamble.py:481: self.short.append(jump_op)
        if let Some(jump) = jump_op {
            self.short.push(jump);
        }
    }

    pub fn produced_short_op(&self, result: OpRef) -> Option<ProducedShortOp> {
        self.produced_short_boxes.get(&result).cloned()
    }

    pub fn short_inputargs(&self) -> &[OpRef] {
        &self.short_inputargs
    }

    pub fn build_short_preamble_struct(&self) -> ShortPreamble {
        // short[..len-1] excludes the JUMP sentinel
        let ops: Vec<majit_ir::OpRc> = self.short[..self.short_ops_len()]
            .iter()
            .map(|op| std::rc::Rc::new(op.clone()))
            .collect();
        build_short_preamble_struct_from_ops(
            &self.short_inputargs,
            &ops,
            &self.used_boxes,
            &self.short_jump_args,
        )
    }

    pub fn extra_same_as(&self) -> &[Op] {
        &self.extra_same_as
    }

    pub fn label_args(&self) -> &[OpRef] {
        &self.label_args
    }

    pub fn jump_args(&self) -> &[OpRef] {
        &self.short_jump_args
    }

    /// short ops length excluding JUMP sentinel.
    pub fn short_ops_len(&self) -> usize {
        self.short.len().saturating_sub(1)
    }

    pub fn short_op(&self, index: usize) -> Option<&Op> {
        if index < self.short_ops_len() {
            self.short.get(index)
        } else {
            None
        }
    }
}

/// shortpreamble.py: build short preamble from optimizer state.
/// Called after preamble optimization is complete.
/// Collects guards + pure ops from the optimized preamble and
/// maps them to label arg indices.
#[cfg(test)]
fn build_from_preamble_and_label(
    preamble_ops: &[Op],
    label_args: &[OpRef],
    exported_state: Option<VirtualState>,
) -> ShortPreamble {
    let mut builder = CollectedShortPreambleBuilder::new();
    let mut included_ovf_positions = IndexSet::new();
    // Record all preamble ops
    for (idx, op) in preamble_ops.iter().enumerate() {
        if op.opcode.is_guard() {
            if op.opcode.is_guard_overflow()
                && idx > 0
                && preamble_ops[idx - 1].opcode.is_ovf()
                && included_ovf_positions.insert(preamble_ops[idx - 1].pos.get())
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
pub(crate) fn extract_short_preamble(peeled_ops: &[Op]) -> ShortPreamble {
    // Find the Label position
    let label_pos = peeled_ops.iter().position(|op| op.opcode == OpCode::Label);

    let label_pos = match label_pos {
        Some(pos) => pos,
        None => return ShortPreamble::empty(), // No label = no peeling happened
    };

    let label_args = peeled_ops[label_pos].getarglist_copy();
    let label_arg_idx =
        |arg: &OpRef| -> Option<usize> { label_args.iter().position(|a| a.to_opref() == *arg) };

    // shortpreamble.py: Collect guards AND pure operations from the preamble.
    // Guards must be replayed so the body's assumptions hold.
    // Pure ops whose results are used as label args must also be replayed
    // (e.g., GETFIELD from preamble that feeds into loop body).
    let mut entries = Vec::new();
    let mut included_positions = IndexSet::new();
    for (idx, op) in peeled_ops[..label_pos].iter().enumerate() {
        let mut included_overflow_producer = false;
        if op.opcode.is_guard_overflow() && idx > 0 {
            let ovf_op = &peeled_ops[idx - 1];
            if ovf_op.opcode.is_ovf() && included_positions.insert(ovf_op.pos.get()) {
                let ovf_arg_mapping: Vec<(usize, usize)> = ovf_op
                    .getarglist()
                    .iter()
                    .enumerate()
                    .filter_map(|(pos, arg)| label_arg_idx(&arg.to_opref()).map(|idx| (pos, idx)))
                    .collect();
                let ovf_fail_arg_mapping: Vec<(usize, usize)> = ovf_op
                    .getfailargs()
                    .into_iter()
                    .flat_map(|fail_args| fail_args.into_iter().enumerate())
                    .filter_map(|(pos, arg)| label_arg_idx(&arg.to_opref()).map(|idx| (pos, idx)))
                    .collect();
                if !ovf_arg_mapping.is_empty() || !ovf_fail_arg_mapping.is_empty() {
                    entries.push(ShortPreambleOp {
                        op: ovf_op.clone(),
                        arg_mapping: ovf_arg_mapping,
                        fail_arg_mapping: ovf_fail_arg_mapping,
                    });
                    included_overflow_producer = true;
                } else {
                    included_positions.swap_remove(&ovf_op.pos.get());
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
            .getarglist()
            .iter()
            .enumerate()
            .filter_map(|(pos, arg)| label_arg_idx(&arg.to_opref()).map(|idx| (pos, idx)))
            .collect();
        let fail_arg_mapping: Vec<(usize, usize)> = op
            .getfailargs()
            .into_iter()
            .flat_map(|fail_args| fail_args.into_iter().enumerate())
            .filter_map(|(pos, arg)| label_arg_idx(&arg.to_opref()).map(|idx| (pos, idx)))
            .collect();

        // Only include ops that reference label args
        if (!arg_mapping.is_empty() || !fail_arg_mapping.is_empty())
            && included_positions.insert(op.pos.get())
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
        constants: majit_ir::ConstMap::new(),
        phase1_inputargs: None,
        inputarg_infos: Vec::new(),
    }
}

/// `unroll.py:497 ExportedState.short_boxes` shape: per-OpRef
/// `ProducedShortOp` records derived from `ctx.exported_short_boxes`.
///
/// The label-arg → short-inputarg rename happens at EXPORT time inside
/// `produce_arg` (shortpreamble.py:285/294): every short op the import
/// path emits (Pure / Heap / LoopInvariant) already carries the renamed
/// `short_inputargs` box in its args. The only exported entries that still
/// reference an original label box are the `ShortInputArg` `SameAs*`
/// stand-ins, and those are never emitted (`produce_op` returns None for
/// `InputArg`, shortpreamble.py:233-234) nor read by any import consumer:
/// `result_map` and the `produce_op` loop skip `InputArg`, the builder
/// init skips entries with no `result_map` slot, and `use_box_recursive`'s
/// arg recursion is rename-invariant for the `SameAs` self-reference. So
/// no import-time arg rewrite is needed — the produced view is a plain
/// filter + transform of `exported_short_boxes`.
///
/// OVF guards are filtered out: the guard entry depends on the
/// preceding `Int*Ovf` op and is re-emitted by the builder through
/// `append_to_short`'s `is_ovf` branch, so the standalone guard must
/// not appear in the produced map.
pub(crate) fn produced_short_boxes_from_exported_boxes(
    exported_short_boxes: &[PreambleOp],
) -> Vec<(OpRef, ProducedShortOp)> {
    exported_short_boxes
        .iter()
        .filter(|entry| !entry.op.opcode.is_guard_overflow())
        .map(|entry| {
            // Fresh clone per entry so the builder's on-object forwarded
            // marker (`ShortPreambleBuilder::new`) mutates an isolated Rc.
            let preamble_op = (*entry.op).clone();
            (
                preamble_op.pos.get(),
                ProducedShortOp {
                    kind: entry.kind.clone(),
                    // shortpreamble.py:58/110 short_op.res — the exported
                    // entry carries the Phase-1 res operand across the
                    // boundary; it already holds the live producer / const
                    // operand (#173 roots the producer, never position-only).
                    res: entry.res.clone(),
                    preamble_op: std::rc::Rc::new(preamble_op),
                    invented_name: entry.invented_name,
                    same_as_source: entry.same_as_source.clone(),
                    label_arg_idx: entry.label_arg_idx,
                },
            )
        })
        .collect()
}

/// `unroll.py:497-504` build path: drive `ShortPreambleBuilder` from
/// pre-derived `produced_short_boxes` (RPython `ExportedState.short_boxes`).
///
/// Phase B B1 first slice: separated from
/// `build_short_preamble_from_exported_boxes` so callers that already
/// hold the `Vec<(OpRef, ProducedShortOp)>` (e.g. `ExportedState.produced_short_boxes`
/// at `unroll.rs:2349`) can invoke the builder directly without
/// re-running the rename + filter pass.
pub(crate) fn build_short_preamble_from_produced_boxes(
    label_args: &[OpRef],
    short_inputargs: &[OpRef],
    produced: &[(OpRef, ProducedShortOp)],
) -> ShortPreamble {
    // #146/S8: the builder map keys by the entry res Box (`p.res`, the carried
    // Phase-1 short-box result). The input slice is keyed by `preamble_op.pos`;
    // re-key to res for the builder map and the use_box driving loop (const res
    // is ptr-unstable and never a key — export filters it).
    //
    // Correctness here rests on the OUTER loop below: it calls add_op_to_short
    // on EVERY entry in input order, which is dependency order (RPython
    // create_short_boxes appends deps before consumers). `use_box_recursive`'s
    // dep pre-append is a best-effort ordering aid whose box-identity dep
    // lookup may miss (an op's dep arg need not Rc::ptr_eq the dependency's res
    // box), but a miss only drops the redundant recursive append — the outer
    // loop still emits every entry in valid def-before-use order. This path is
    // a corpus-dead fallback (rebuilt preamble is Some in the measured corpus).
    let entries: Vec<(majit_ir::operand::Operand, ProducedShortOp)> = produced
        .iter()
        .filter(|(_, p)| !p.res.to_opref().is_constant())
        .map(|(_, p)| (p.res.clone(), p.clone()))
        .collect();
    let mut builder = ShortPreambleBuilder::new(label_args, &entries, short_inputargs);
    // history.py:227/268/314 — inline-Const variants carry the value on
    // the OpRef; `is_constant()` returns true intrinsically and the
    // `is_reachable` / `add_heap_op` checks short-circuit before
    // consulting `known_constants`. There are no constant pool entries
    // to seed into `known_constants` and the `loop_constants`
    // parameter has been retired.
    for (res, _) in &entries {
        let _ = builder.add_op_to_short(res);
        let _ = builder.add_preamble_op(res);
    }
    builder.build_short_preamble_struct()
}

#[cfg(test)]
fn build_short_preamble_from_exported_boxes(
    label_args: &[OpRef],
    short_inputargs: &[OpRef],
    exported_short_boxes: &[PreambleOp],
) -> ShortPreamble {
    let produced = produced_short_boxes_from_exported_boxes(exported_short_boxes);
    build_short_preamble_from_produced_boxes(label_args, short_inputargs, &produced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::rooted_resop_operand;
    use majit_ir::operand::Operand;
    use majit_ir::{Op, OpCode, OpRc, OpRef, Type};

    /// oparser-faithful op-arg minter: a rooted resop producer shed to its
    /// `Operand::Op` face. Op-arg slices take `Operand`.
    fn rop(ty: Type, pos: u32) -> Operand {
        rooted_resop_operand(ty, pos)
    }

    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos
                .set(OpRef::op_typed(base + i as u32, op.result_type()));
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
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]),
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Label, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]),
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Jump, &[rop(Type::Int, 4), rop(Type::Int, 101)]),
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
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Finish, &[rop(Type::Int, 0)]),
        ];

        let sp = extract_short_preamble(&ops);
        assert!(sp.is_empty());
    }

    #[test]
    fn test_extract_overflow_guard_includes_preceding_ovf_op() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMulOvf,
                &[rop(Type::Int, 100), rop(Type::Int, 100)],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Label, &[rop(Type::Int, 100)]),
            Op::new(OpCode::Jump, &[rop(Type::Int, 100)]),
        ];
        assign_positions(&mut ops, 0);
        ops[1].setfailargs(vec![rooted_resop_operand(Type::Int, 100)].into());
        let sp = extract_short_preamble(&ops);

        assert_eq!(sp.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntMulOvf);
        assert_eq!(sp.ops[1].op.opcode, OpCode::GuardNoOverflow);
    }

    #[test]
    fn test_extract_overflow_guard_without_replayable_ovf_is_skipped() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMulOvf,
                &[rop(Type::Int, 200), rop(Type::Int, 200)],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Label, &[rop(Type::Int, 100)]),
            Op::new(OpCode::Jump, &[rop(Type::Int, 100)]),
        ];
        assign_positions(&mut ops, 0);
        ops[1].setfailargs(vec![rooted_resop_operand(Type::Int, 100)].into());
        let sp = extract_short_preamble(&ops);

        assert!(sp.is_empty());
    }

    #[test]
    fn test_extract_skips_non_label_guards() {
        // Guards that don't reference label args should not be included
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 0)]), // refs temporary, not label arg
            Op::new(OpCode::Label, &[rop(Type::Int, 100)]),   // only v100 is a label arg
            Op::new(OpCode::Jump, &[rop(Type::Int, 100)]),
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
    fn test_builder_collects_guards() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // Simulate preamble processing
        let guard1 = Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]);
        let guard2 = Op::new(
            OpCode::GuardClass,
            &[rop(Type::Int, 101), rop(Type::Int, 200)],
        );
        let non_guard = Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]);

        builder.add_preamble_guard(&guard1);
        builder.add_preamble_guard(&guard2);
        builder.add_preamble_guard(&non_guard); // should be ignored (not a guard)

        // Set label args (preamble phase ends)
        builder.set_label_args(&[OpRef::int_op(100), OpRef::int_op(101)]);

        // After label, no more collection
        let guard3 = Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]);
        builder.add_preamble_guard(&guard3); // should be ignored

        let sp = builder.build(None);
        assert_eq!(sp.len(), 2); // Only the two guards from the preamble
    }

    #[test]
    fn test_builder_maps_args_to_label_indices() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // Preamble has guard on v100 and v101
        let guard = Op::new(
            OpCode::GuardValue,
            &[rop(Type::Int, 100), rop(Type::Int, 200)],
        );
        builder.add_preamble_guard(&guard);

        // Label carries v100 as arg 0 and v101 as arg 1
        builder.set_label_args(&[OpRef::int_op(100), OpRef::int_op(101)]);

        let sp = builder.build(None);
        assert_eq!(sp.ops[0].arg_mapping.len(), 1); // v100 → label arg 0
        assert_eq!(sp.ops[0].arg_mapping[0], (0, 0));
        // v200 is not a label arg, so it's not in the mapping
    }

    #[test]
    fn test_builder_add_preamble_op_any_type() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // add_preamble_op accepts any op type (not just guards)
        let pure_op = Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]);
        builder.add_preamble_op(&pure_op);

        builder.set_label_args(&[OpRef::int_op(100), OpRef::int_op(101)]);

        let sp = builder.build(None);
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_extract_multiple_guards() {
        // Multiple guards in the preamble
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]),
            Op::new(OpCode::GuardNonnull, &[rop(Type::Int, 101)]),
            Op::new(
                OpCode::GuardClass,
                &[rop(Type::Int, 100), rop(Type::Int, 200)],
            ),
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Label, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
            Op::new(OpCode::Jump, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
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
    fn test_build_short_preamble_from_exported_boxes_uses_exported_order() {
        // shortpreamble.py:285/294: the label-arg → short-inputarg rename
        // happens at EXPORT time (produce_arg), so the exported short ops
        // already carry the renamed `short_inputargs` boxes (10/11) in place
        // of the original label args (0/1). The op-result positions (7/8) are
        // not inputargs and are unchanged.
        let exported = vec![
            PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[rop(Type::Int, 10), rop(Type::Int, 11)]);
                    op.pos.set(OpRef::int_op(7));
                    std::rc::Rc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 7),
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
            PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntSub, &[rop(Type::Int, 7), rop(Type::Int, 11)]);
                    op.pos.set(OpRef::int_op(8));
                    std::rc::Rc::new(op)
                },
                res: rooted_resop_operand(Type::Int, 8),
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
        ];

        let sp = build_short_preamble_from_exported_boxes(
            &[OpRef::int_op(0), OpRef::int_op(1)],
            &[OpRef::int_op(10), OpRef::int_op(11)],
            &exported,
        );
        assert_eq!(sp.ops.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntSub);
        assert_eq!(sp.ops[1].arg_mapping, vec![(1, 1)]);
        assert_eq!(sp.inputargs, vec![OpRef::int_op(10), OpRef::int_op(11)]);
    }

    #[test]
    fn test_build_short_preamble_from_exported_boxes_skips_standalone_overflow_guards() {
        let label_args = vec![OpRef::int_op(10), OpRef::int_op(11)];
        let short_inputargs = vec![OpRef::int_op(100), OpRef::int_op(101)];

        let mut ovf = Op::new(OpCode::IntAddOvf, &[rop(Type::Int, 10), rop(Type::Int, 11)]);
        ovf.pos.set(OpRef::int_op(20));
        let guard = Op::new(OpCode::GuardNoOverflow, &[]);

        let exported = vec![
            PreambleOp {
                op: std::rc::Rc::new(ovf),
                res: rooted_resop_operand(Type::Int, 20),
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
            PreambleOp {
                op: std::rc::Rc::new(guard),
                res: majit_ir::operand::Operand::None,
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
        // #146/S8: the builder map keys by the short-box res Box identity, and a
        // dependency is found when an op's arg Box Rc::ptr_eq's the dep's res
        // (RPython box-identity `if arg in produced_short_boxes`). Express the
        // B-depends-on-A edge by binding B's IntMul arg(0) to A's producer op
        // via `from_bound_op` — the memoized result box is ptr_eq to A's `res`,
        // so the dependency lookup hits and `Op::new` round-trips the same box
        // through `getarglist()` (vs a position-only `from_opref`, which would
        // mint a non-`ptr_eq` arg box).
        let in0 = rooted_resop_operand(Type::Int, 0);
        let in1 = rooted_resop_operand(Type::Int, 1);
        let producer7 = {
            let mut op = Op::new(OpCode::IntAdd, &[in0.clone(), in1.clone()]);
            op.pos.set(OpRef::int_op(7));
            std::rc::Rc::new(op)
        };
        let res7 = Operand::from_bound_op(&producer7);
        let producer8 = {
            let mut op = Op::new(OpCode::IntMul, &[res7.clone(), in1.clone()]);
            op.pos.set(OpRef::int_op(8));
            std::rc::Rc::new(op)
        };
        let res8 = Operand::from_bound_op(&producer8);
        let produced = vec![
            (
                res7.clone(),
                ProducedShortOp {
                    kind: PreambleOpKind::Pure,
                    res: res7.clone(),
                    preamble_op: producer7,
                    invented_name: false,
                    same_as_source: None,
                    label_arg_idx: None,
                },
            ),
            (
                res8.clone(),
                ProducedShortOp {
                    kind: PreambleOpKind::Pure,
                    res: res8.clone(),
                    preamble_op: producer8,
                    invented_name: false,
                    same_as_source: None,
                    label_arg_idx: None,
                },
            ),
        ];
        let mut builder = ShortPreambleBuilder::new(
            &[OpRef::int_op(0), OpRef::int_op(1)],
            &produced,
            &[OpRef::int_op(0), OpRef::int_op(1)],
        );

        let used = builder.add_op_to_short(&res8).unwrap();
        assert!(builder.add_preamble_op(&res7));
        assert!(builder.add_preamble_op(&res8));
        assert_eq!(used.opcode, OpCode::IntMul);
        let short = builder.build_short_preamble();
        assert_eq!(short[1].opcode, OpCode::IntAdd);
        assert_eq!(short[2].opcode, OpCode::IntMul);
        assert_eq!(builder.used_boxes(), &[OpRef::int_op(7), OpRef::int_op(8)]);
    }

    #[test]
    fn test_build_from_preamble_and_label() {
        let mut preamble = vec![
            Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]),
            Op::new(OpCode::IntAdd, &[rop(Type::Int, 100), rop(Type::Int, 101)]),
        ];
        assign_positions(&mut preamble, 0);

        let label_args = &[OpRef::int_op(100), OpRef::int_op(101)];
        let sp = build_from_preamble_and_label(&preamble, label_args, None);

        // Guard + pure IntAdd
        assert_eq!(sp.len(), 2);
    }

    #[test]
    fn test_extended_builder() {
        let mut builder = CollectedExtendedShortPreambleBuilder::new();
        builder.set_label_args(&[OpRef::int_op(100), OpRef::int_op(101)]);
        builder.add_guard(Op::new(OpCode::GuardTrue, &[rop(Type::Int, 100)]));
        builder.add_pure_op(Op::new(
            OpCode::IntAdd,
            &[rop(Type::Int, 100), rop(Type::Int, 101)],
        ));
        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rop(Type::Int, 100)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos.set(OpRef::int_op(102));
        builder.add_heap_op(heap);
        builder.add_loopinvariant_op(Op::new(OpCode::CallI, &[rop(Type::Int, 100)]));
        assert_eq!(builder.num_ops(), 4);
    }

    #[test]
    fn test_short_boxes() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb =
            ShortBoxes::with_label_args(&[OpRef::int_op(10), OpRef::int_op(11), OpRef::int_op(12)]);
        assert_eq!(sb.num_label_args, 3);
        // Production seeds a renamed ShortInputArg for every label arg
        // (optimizer.rs preview loop); the pure/heap deps resolve through them.
        for arg in [OpRef::int_op(10), OpRef::int_op(11), OpRef::int_op(12)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }
        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 10), rop(Type::Int, 11)]);
        pure.pos.set(OpRef::int_op(20));
        sb.add_pure_op(&mut __ctx, pure);
        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rop(Type::Int, 10)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos.set(OpRef::int_op(21));
        sb.add_heap_op(&mut __ctx, heap);
        let produced = sb.produced_ops(&mut __ctx);
        // 3 ShortInputArgs (one per label arg) + the pure and heap ops.
        let non_input: Vec<_> = produced
            .iter()
            .filter(|(_, p)| p.kind != PreambleOpKind::InputArg)
            .collect();
        assert_eq!(non_input.len(), 2);
    }

    #[test]
    fn test_short_boxes_reject_unknown_nonconstant_dependency() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(10)]);
        sb.add_short_input_arg(&mut __ctx, OpRef::int_op(10), majit_ir::Type::Int);
        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 10), rop(Type::Int, 999)]);
        pure.pos.set(OpRef::int_op(20));
        sb.add_pure_op(&mut __ctx, pure);

        let produced = sb.produced_ops(&mut __ctx);
        // The label arg OpRef::int_op(10) itself is produced (as ShortInputArg),
        // but the pure op depending on unknown OpRef::int_op(999) is rejected.
        assert!(
            !produced.iter().any(|(r, _)| *r == OpRef::int_op(20)),
            "pure op with unknown dependency should be rejected"
        );
    }

    #[test]
    fn test_short_boxes_accept_known_constant_dependency() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(10)]);
        sb.add_short_input_arg(&mut __ctx, OpRef::int_op(10), majit_ir::Type::Int);
        sb.note_known_constant(OpRef::int_op(999));
        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 10), rop(Type::Int, 999)]);
        pure.pos.set(OpRef::int_op(20));
        sb.add_pure_op(&mut __ctx, pure);

        let produced = sb.produced_ops(&mut __ctx);
        let renamed10 = sb.create_short_inputargs(&[OpRef::int_op(10)])[0];
        // 1 ShortInputArg (label arg 10) + the accepted pure op.
        let non_input: Vec<_> = produced
            .iter()
            .filter(|(_, p)| p.kind != PreambleOpKind::InputArg)
            .collect();
        assert_eq!(non_input.len(), 1);
        let pure = produced
            .iter()
            .find(|(result, _)| *result == OpRef::int_op(20))
            .expect("missing produced pure op");
        // The label-arg dep is renamed to its short_inputargs box; the known
        // constant 999 passes through unchanged.
        assert_eq!(
            pure.1
                .preamble_op
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![renamed10, OpRef::int_op(999)]
        );
    }

    #[test]
    fn test_short_boxes_compound_prefers_non_heap_and_emits_invented_alias() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        // Only the op DEPENDENCIES (30, 31) are label args; the compound result
        // (pos 10) is an op result, not a label arg.
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30), OpRef::int_op(31)]);
        for arg in [OpRef::int_op(30), OpRef::int_op(31)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rop(Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos.set(OpRef::int_op(10));
        sb.add_potential_op(&mut __ctx, None, heap, PreambleOpKind::Heap);

        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 30), rop(Type::Int, 31)]);
        pure.pos.set(OpRef::int_op(10));
        sb.add_potential_op(&mut __ctx, None, pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops(&mut __ctx);
        // 2 ShortInputArgs (30, 31) + the compound@10 (Pure chosen + Heap alias).
        let non_input_count = produced
            .iter()
            .filter(|(_, p)| p.kind != PreambleOpKind::InputArg)
            .count();
        assert_eq!(non_input_count, 2);

        let chosen = produced
            .iter()
            .find(|(result, p)| *result == OpRef::int_op(10) && p.kind != PreambleOpKind::InputArg)
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::Pure);
        assert!(!chosen.1.invented_name);

        let alias = produced
            .iter()
            .find(|(result, p)| *result != OpRef::int_op(10) && p.kind != PreambleOpKind::InputArg)
            .unwrap();
        assert_eq!(alias.1.kind, PreambleOpKind::Heap);
        assert!(alias.1.invented_name);
        assert_eq!(
            alias.1.same_as_source.as_ref().map(|b| b.to_opref()),
            Some(OpRef::int_op(10))
        );
    }

    #[test]
    fn test_short_boxes_nested_compound_emits_multiple_invented_aliases() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        // Only the op DEPENDENCIES (30, 31) are label args; the compound result
        // (pos 20) is an op result, not a label arg.
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30), OpRef::int_op(31)]);
        for arg in [OpRef::int_op(30), OpRef::int_op(31)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rop(Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos.set(OpRef::int_op(20));
        sb.add_potential_op(&mut __ctx, None, heap, PreambleOpKind::Heap);

        let mut loopinv = Op::new(OpCode::CallI, &[rop(Type::Int, 30)]);
        loopinv.pos.set(OpRef::int_op(20));
        sb.add_potential_op(&mut __ctx, None, loopinv, PreambleOpKind::LoopInvariant);

        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 30), rop(Type::Int, 31)]);
        pure.pos.set(OpRef::int_op(20));
        sb.add_potential_op(&mut __ctx, None, pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops(&mut __ctx);
        // 2 ShortInputArgs (30, 31) + the compound@20 (Pure chosen + 2 aliases).
        let non_input_count = produced
            .iter()
            .filter(|(_, p)| p.kind != PreambleOpKind::InputArg)
            .count();
        assert_eq!(non_input_count, 3);

        let chosen = produced
            .iter()
            .find(|(result, p)| *result == OpRef::int_op(20) && p.kind != PreambleOpKind::InputArg)
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::Pure);
        assert!(!chosen.1.invented_name);

        let aliases: Vec<_> = produced
            .iter()
            .filter(|(result, p)| {
                *result != OpRef::int_op(20) && p.kind != PreambleOpKind::InputArg
            })
            .collect();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.iter().all(|(_, produced)| produced.invented_name));
        assert!(aliases.iter().all(|(_, produced)| {
            produced.same_as_source.as_ref().map(|b| b.to_opref()) == Some(OpRef::int_op(20))
        }));
    }

    #[test]
    fn test_rpython_create_short_boxes_prefers_short_inputarg_over_heap_result() {
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(10), OpRef::int_op(30)]);
        // pos 10 is both a label arg and a heap result (the case under test);
        // seed both label args (the heap dep 30 must be a ShortInputArg too).
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(10), majit_ir::Type::Int);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(30), majit_ir::Type::Int);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rop(Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos.set(OpRef::int_op(10));
        sb.add_heap_op(&mut ctx, heap);

        let produced = sb.produced_ops(&mut ctx);

        // The compound at pos 10 prefers the ShortInputArg; the Heap becomes an
        // invented alias. (Label arg 30 is also produced as a ShortInputArg.)
        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef::int_op(10))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::InputArg);
        assert!(!chosen.1.invented_name);

        let alias = produced
            .iter()
            .find(|(_, p)| p.kind == PreambleOpKind::Heap)
            .unwrap();
        assert!(alias.1.invented_name);
        assert_eq!(
            alias.1.same_as_source.as_ref().map(|b| b.to_opref()),
            Some(OpRef::int_op(10))
        );
    }

    #[test]
    fn test_compound_pure_loser_to_short_inputarg_clears_label_arg_idx() {
        // shortpreamble.py:326-333 — when a Pure alternative loses the compound
        // tie to the ShortInputArg, its result is rebound to a fresh SameAs box
        // (`lst[i].short_op.res = new_name`), so the invented alias is no longer
        // a label arg. pyre's `label_arg_idx` (the position proxy for "res is
        // label arg N") must be cleared in lockstep; otherwise the import
        // slot-lookup at unroll.rs (the path-2 Pure|LoopInvariant arm) would map
        // the invented alias onto the loop-carried `short_args[slot]`, collapsing
        // the distinct same_as identity into a wrong-result miscompile. (The
        // sibling Heap-loser test above does NOT exercise this: path-2 ignores
        // label_arg_idx for the Heap kind.)
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        let mut sb =
            ShortBoxes::with_label_args(&[OpRef::int_op(10), OpRef::int_op(30), OpRef::int_op(31)]);
        // pos 10 is both a label arg (slot 0) and a pure result (the case under
        // test); seed all three label args (the pure deps 30/31 are
        // ShortInputargs too).
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(10), majit_ir::Type::Int);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(30), majit_ir::Type::Int);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(31), majit_ir::Type::Int);

        // A pure op whose result coincides with label arg 10, depending on the
        // other two label args (avoids a self-referential in-production cycle).
        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 30), rop(Type::Int, 31)]);
        pure.pos.set(OpRef::int_op(10));
        sb.add_pure_op(&mut ctx, pure);

        let produced = sb.produced_ops(&mut ctx);

        // The compound at pos 10 prefers the ShortInputArg; the Pure becomes an
        // invented alias.
        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef::int_op(10))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::InputArg);
        assert!(!chosen.1.invented_name);
        // The winning ShortInputArg keeps its label slot (box 10 = slot 0).
        assert_eq!(chosen.1.label_arg_idx, Some(0));

        let alias = produced
            .iter()
            .find(|(_, p)| p.kind == PreambleOpKind::Pure)
            .unwrap();
        assert!(alias.1.invented_name);
        assert_eq!(
            alias.1.same_as_source.as_ref().map(|b| b.to_opref()),
            Some(OpRef::int_op(10))
        );
        // THE REGRESSION LOCK: the invented alias must NOT carry the original
        // label slot. Before the fix this was Some(0) and the import collapsed
        // the alias onto the loop-carried input.
        assert_eq!(alias.1.label_arg_idx, None);
    }

    #[test]
    fn test_short_inputargs_have_distinct_renamed_identity() {
        // shortpreamble.py:256-258 `renamed = OpHelpers.inputarg_from_tp(box.type)`
        // — each short inputarg is a FRESH InputArg distinct from the original
        // label arg `box`. pyre's old position-identity model reused the label
        // arg's OpRef; this test locks the distinct-renamed-box parity.
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        let label_args = [OpRef::int_op(10), OpRef::ref_op(11)];
        let mut sb = ShortBoxes::with_label_args(&label_args);
        sb.add_short_input_arg(&mut ctx, OpRef::int_op(10), majit_ir::Type::Int);
        sb.add_short_input_arg(&mut ctx, OpRef::ref_op(11), majit_ir::Type::Ref);

        let si = sb.create_short_inputargs(&label_args);
        assert_eq!(si.len(), 2);

        // (A) DISTINCT identity: the renamed box is not the original label arg.
        assert_ne!(si[0], OpRef::int_op(10));
        assert_ne!(si[1], OpRef::ref_op(11));

        // (B) the renamed boxes are InputArg-kind of the matching type
        // (inputarg_from_tp(box.type)).
        assert!(matches!(si[0], OpRef::InputArgInt(_)));
        assert!(matches!(si[1], OpRef::InputArgRef(_)));
        assert_eq!(si[0].ty(), Some(majit_ir::Type::Int));
        assert_eq!(si[1].ty(), Some(majit_ir::Type::Ref));

        // (C) lookups still resolve a label arg by its ORIGINAL opref
        // (shortpreamble.py:259 potential_ops[box]).
        assert_eq!(sb.lookup_label_arg(OpRef::int_op(10)), Some(0));
        assert_eq!(sb.lookup_label_arg(OpRef::ref_op(11)), Some(1));
        assert!(sb.is_reachable(OpRef::int_op(10)));
    }

    #[test]
    fn test_duplicate_slot_keeps_last_renamed_inputarg() {
        // shortpreamble.py:255-259 — when a box appears at TWO slots of the
        // combined `label_args + virtuals` (a virtual field coinciding with a
        // label arg; empirically RefOp(50)/RefOp(174) in synth/tuple_unpacking),
        // the `potential_ops[box] = ShortInputArg(box, renamed)` dict assignment
        // OVERWRITES, so the LAST slot's renamed inputarg is the live one and
        // `produce_arg` returns `short_inputargs[LAST]`; the FIRST slot's renamed
        // box is the dead Label arg. pyre stamps `label_arg_idx = live_slot` (the
        // current call's slot) so the later (overwriting) call records the LAST
        // slot, matching upstream. Before the fix `lookup_label_arg`'s
        // first-occurrence made produce_arg return short_inputargs[FIRST].
        let mut ctx = crate::optimizeopt::OptContext::new(256);
        // The SAME box at both slots: the production caller (optimizer.rs preview
        // loop) iterates label_args+virtuals, so a coinciding box reaches
        // add_short_input_arg once per slot.
        let label_args = [OpRef::ref_op(50), OpRef::ref_op(50)];
        let mut sb = ShortBoxes::with_label_args(&label_args);
        sb.add_short_input_arg(&mut ctx, OpRef::ref_op(50), majit_ir::Type::Ref);
        sb.add_short_input_arg(&mut ctx, OpRef::ref_op(50), majit_ir::Type::Ref);

        // Two FRESH distinct renamed boxes, one per slot (shortpreamble.py:258).
        let si = sb.create_short_inputargs(&label_args);
        assert_eq!(si.len(), 2);
        assert_ne!(si[0], si[1]);

        // The duplicate collapses to ONE produced InputArg entry, keyed at the
        // LAST slot (label_arg_idx == Some(1)), not the first (Some(0)).
        let produced = sb.produced_ops(&mut ctx);
        let inputarg_entries: Vec<_> = produced
            .iter()
            .filter(|(r, p)| *r == OpRef::ref_op(50) && p.kind == PreambleOpKind::InputArg)
            .collect();
        assert_eq!(inputarg_entries.len(), 1);
        // THE REGRESSION LOCK: the overwriting (later) call must record the LAST
        // slot, matching upstream's dict-overwrite. Some(0) before the fix.
        assert_eq!(inputarg_entries[0].1.label_arg_idx, Some(1));

        // produce_arg returns the LAST slot's renamed box.
        let produced_arg = sb.produce_arg(&mut ctx, OpRef::ref_op(50)).unwrap();
        assert_eq!(produced_arg.to_opref(), si[1]);
        assert_ne!(produced_arg.to_opref(), si[0]);

        // `lookup_label_arg` mirrors `potential_ops[box]`'s last-wins overwrite,
        // resolving the duplicated box to the LAST/live slot. This keeps the
        // non-InputArg export lookup (optimizer.rs `lookup_label_arg(
        // canonical_result)`) consistent with the InputArg entry's `live_slot`
        // above. `rposition`; was `Some(0)` (first) before the fix.
        assert_eq!(sb.lookup_label_arg(OpRef::ref_op(50)), Some(1));
    }

    #[test]
    fn test_rpython_short_preamble_builder_add_op_to_short_builds_label_short_and_jump() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        // The ovf result (pos 10) is an op result; its deps (30, 31) are the
        // label args and become renamed ShortInputargs that the Label carries.
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30), OpRef::int_op(31)]);
        for arg in [OpRef::int_op(30), OpRef::int_op(31)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }

        let mut ovf = Op::new(OpCode::IntAddOvf, &[rop(Type::Int, 30), rop(Type::Int, 31)]);
        ovf.pos.set(OpRef::int_op(10));
        sb.add_potential_op(&mut __ctx, None, ovf, PreambleOpKind::Pure);

        let produced = sb.produced_ops(&mut __ctx);
        let short_inputargs = sb.create_short_inputargs(&[OpRef::int_op(30), OpRef::int_op(31)]);
        let label_arg_oprefs: Vec<OpRef> = short_inputargs.clone();
        // #146/S8: the builder map keys by the entry res Box; re-key the
        // produced_ops list (keyed by `preamble_op.pos`) to res for new() and
        // look up by the res box of the int_op(10) entry.
        let entries: Vec<(majit_ir::operand::Operand, ProducedShortOp)> = produced
            .iter()
            .map(|(_, p)| (p.res.clone(), p.clone()))
            .collect();
        let res10 = produced
            .iter()
            .find(|(r, _)| *r == OpRef::int_op(10))
            .unwrap()
            .1
            .res
            .clone();
        let mut builder = ShortPreambleBuilder::new(&label_arg_oprefs, &entries, &short_inputargs);
        let used = builder.add_op_to_short(&res10).unwrap();
        assert!(builder.add_preamble_op(&res10));
        assert_eq!(used.opcode, OpCode::IntAddOvf);
        assert_eq!(builder.used_boxes(), &[OpRef::int_op(10)]);

        let short = builder.build_short_preamble();
        assert_eq!(short.len(), 4);
        assert_eq!(short[0].opcode, OpCode::Label);
        assert_eq!(short[1].opcode, OpCode::IntAddOvf);
        assert_eq!(short[2].opcode, OpCode::GuardNoOverflow);
        assert_eq!(short[3].opcode, OpCode::Jump);
        assert_eq!(
            short[3]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(10)]
        );
    }

    #[test]
    fn test_rpython_short_preamble_builder_tracks_extra_same_as() {
        let mut __ctx = crate::optimizeopt::OptContext::new(256);
        // The compound result (pos 20) is an op result; its deps (30, 31) are
        // the label args / renamed ShortInputargs.
        let mut sb = ShortBoxes::with_label_args(&[OpRef::int_op(30), OpRef::int_op(31)]);
        for arg in [OpRef::int_op(30), OpRef::int_op(31)] {
            sb.add_short_input_arg(&mut __ctx, arg, majit_ir::Type::Int);
        }

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[rop(Type::Int, 30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos.set(OpRef::int_op(20));
        sb.add_potential_op(&mut __ctx, None, heap, PreambleOpKind::Heap);

        let mut pure = Op::new(OpCode::IntAdd, &[rop(Type::Int, 30), rop(Type::Int, 31)]);
        pure.pos.set(OpRef::int_op(20));
        sb.add_potential_op(&mut __ctx, None, pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops(&mut __ctx);
        let (alias_result, alias_res) = produced
            .iter()
            .find(|(result, pop)| *result != OpRef::int_op(20) && pop.invented_name)
            .map(|(result, pop)| (*result, pop.res.clone()))
            .unwrap();

        // #146/S8: re-key the produced_ops list to res for new() + look up the
        // invented-name alias entry by its res box.
        let entries: Vec<(majit_ir::operand::Operand, ProducedShortOp)> = produced
            .iter()
            .map(|(_, p)| (p.res.clone(), p.clone()))
            .collect();
        let mut builder =
            ShortPreambleBuilder::new(&[OpRef::int_op(20)], &entries, &[OpRef::int_op(20)]);
        assert!(builder.add_preamble_op(&alias_res));
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos.get(), alias_result);
        assert_eq!(
            extra[0]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(20)]
        );
    }

    #[test]
    fn test_short_preamble_builder_fallback_keeps_invented_name_alias_identity() {
        let mut builder = ShortPreambleBuilder::new(&[OpRef::int_op(7)], &[], &[OpRef::int_op(7)]);
        let mut replay_op = Op::new(OpCode::GetfieldGcI, &[rop(Type::Int, 30)]);
        replay_op.pos.set(OpRef::int_op(14));
        let pop = crate::optimizeopt::info::PreambleOp {
            op: rooted_resop_operand(Type::Int, 14),
            invented_name: true,
            preamble_op: std::rc::Rc::new(replay_op),
            // Imported invented-name pop carries the original it aliases;
            // the else arm reads it to emit `same_as(source)`.
            same_as_source: Some(rooted_resop_operand(Type::Int, 14)),
        };

        builder.add_preamble_op_from_pop(&pop, rooted_resop_operand(Type::Int, 41));

        assert_eq!(builder.used_boxes(), &[OpRef::int_op(41)]);
        assert_eq!(builder.short_preamble_jump().len(), 1);
        assert_eq!(
            builder.short_preamble_jump()[0].pos.get(),
            OpRef::int_op(14)
        );
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos.get(), OpRef::int_op(41));
        assert_eq!(
            extra[0]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(14)]
        );
    }

    #[test]
    fn test_extended_short_preamble_builder_fallback_keeps_invented_name_alias_identity() {
        let sb = ShortPreambleBuilder::new(&[OpRef::int_op(7)], &[], &[OpRef::int_op(7)]);
        let mut builder = ExtendedShortPreambleBuilder::new(0, &sb);
        let mut replay_op = Op::new(OpCode::GetfieldGcI, &[rop(Type::Int, 30)]);
        replay_op.pos.set(OpRef::int_op(14));
        let pop = crate::optimizeopt::info::PreambleOp {
            op: rooted_resop_operand(Type::Int, 14),
            invented_name: true,
            preamble_op: std::rc::Rc::new(replay_op),
            // Imported invented-name pop carries the original it aliases;
            // the else arm reads it to emit `same_as(source)`.
            same_as_source: Some(rooted_resop_operand(Type::Int, 14)),
        };

        builder.add_preamble_op_from_pop(&pop, rooted_resop_operand(Type::Int, 41));

        assert_eq!(builder.label_args(), &[OpRef::int_op(41)]);
        assert_eq!(builder.jump_args(), &[OpRef::int_op(14)]);
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos.get(), OpRef::int_op(41));
        assert_eq!(
            extra[0]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(14)]
        );
    }
}
