/// Guard optimization pass.
///
/// Removes redundant guards, strengthens weak guards when a stronger guard
/// on the same value appears later, and fuses consecutive guards that can
/// share fail descriptors.
///
/// ## Redundant Guard Removal
///
/// If the same foldable guard condition (opcode + arguments) has already been
/// verified, the duplicate guard is removed:
///
/// ```text
/// guard_true(v1)          # first check
/// ...
/// guard_true(v1)          # redundant – removed
/// ```
///
/// ## Guard Strengthening
///
/// A weaker guard is subsumed by a stronger guard on the same value:
///
/// ```text
/// guard_true(v1)          # v1 is truthy  (weak)
/// guard_value(v1, 42)     # v1 == 42      (strong, implies truthy)
/// ```
///
/// When the stronger guard is encountered the weak guard is recognized as
/// already satisfied, so it would have been removed by redundant-guard
/// removal anyway.  Strengthening here means we record that `v1` is
/// *nonnull/truthy* after seeing `guard_nonnull(v1)` or `guard_true(v1)`,
/// so a later `guard_nonnull(v1)` is eliminated.
///
/// ## Consecutive Guard Fusion
///
/// Consecutive guards on different values are kept but, when both lack a
/// descriptor, the second inherits the first's descriptor so the backend
/// can share resume data.
use std::collections::{HashMap, HashSet};

use majit_ir::{DescrRef, Op, OpCode, OpRef};

use crate::{OptContext, Optimization, OptimizationResult};

/// Key that uniquely identifies a guard condition.
///
/// Two foldable guards are redundant when they have the same opcode and the same
/// arguments (after forwarding resolution).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GuardKey {
    opcode: OpCode,
    args: Vec<OpRef>,
}

impl GuardKey {
    fn from_op(op: &Op) -> Self {
        GuardKey {
            opcode: op.opcode,
            args: op.args.to_vec(),
        }
    }
}

/// guard.py: Guard — wraps a guard op with its comparison op for
/// implication analysis. Used to determine if one guard implies another
/// (e.g., `x < 10` implies `x < 100`).
#[derive(Clone, Debug)]
pub struct Guard {
    /// Position in the operations list.
    pub index: usize,
    /// The guard opcode (GuardTrue or GuardFalse).
    pub guard_opcode: OpCode,
    /// The comparison opcode (IntLt, IntLe, IntGt, IntGe, etc.).
    pub cmp_opcode: OpCode,
    /// Left-hand side of the comparison.
    pub lhs: OpRef,
    /// Right-hand side of the comparison.
    pub rhs: OpRef,
}

impl Guard {
    /// Create from a guard_true/guard_false op and its preceding comparison.
    pub fn from_ops(index: usize, guard_op: &Op, cmp_op: &Op) -> Option<Self> {
        if !guard_op.opcode.is_guard() {
            return None;
        }
        match cmp_op.opcode {
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe => {}
            _ => return None,
        }
        Some(Guard {
            index,
            guard_opcode: guard_op.opcode,
            cmp_opcode: cmp_op.opcode,
            lhs: cmp_op.arg(0),
            rhs: if cmp_op.num_args() > 1 {
                cmp_op.arg(1)
            } else {
                OpRef::NONE
            },
        })
    }

    /// guard.py: get_compare_opnum — effective comparison considering
    /// guard_true vs guard_false inversion.
    pub fn effective_cmp(&self) -> OpCode {
        if self.guard_opcode == OpCode::GuardTrue {
            self.cmp_opcode
        } else {
            // guard_false(x < y) means x >= y
            match self.cmp_opcode {
                OpCode::IntLt => OpCode::IntGe,
                OpCode::IntLe => OpCode::IntGt,
                OpCode::IntGt => OpCode::IntLe,
                OpCode::IntGe => OpCode::IntLt,
                OpCode::IntEq => OpCode::IntNe,
                OpCode::IntNe => OpCode::IntEq,
                OpCode::UintLt => OpCode::UintGe,
                OpCode::UintLe => OpCode::UintGt,
                OpCode::UintGt => OpCode::UintLe,
                OpCode::UintGe => OpCode::UintLt,
                other => other,
            }
        }
    }

    /// guard.py: implies(other) — does this guard imply the other?
    ///
    /// `self` implies `other` if whenever self passes, other also passes.
    /// E.g., `x < 5` implies `x < 10` (same lhs, tighter rhs).
    pub fn implies(&self, other: &Guard, ctx: &OptContext) -> bool {
        if self.lhs != other.lhs {
            return false;
        }
        let self_cmp = self.effective_cmp();
        let other_cmp = other.effective_cmp();

        // Same comparison direction with constant rhs
        let self_rhs = ctx.get_constant_int(self.rhs);
        let other_rhs = ctx.get_constant_int(other.rhs);
        if let (Some(sv), Some(ov)) = (self_rhs, other_rhs) {
            match (self_cmp, other_cmp) {
                // x < sv → x < ov  if sv <= ov
                (OpCode::IntLt, OpCode::IntLt) => return sv <= ov,
                // x <= sv → x <= ov  if sv <= ov
                (OpCode::IntLe, OpCode::IntLe) => return sv <= ov,
                // x < sv → x <= ov  if sv <= ov
                (OpCode::IntLt, OpCode::IntLe) => return sv <= ov,
                // x <= sv → x < ov  if sv < ov
                (OpCode::IntLe, OpCode::IntLt) => return sv < ov,
                // x > sv → x > ov  if sv >= ov
                (OpCode::IntGt, OpCode::IntGt) => return sv >= ov,
                // x >= sv → x >= ov  if sv >= ov
                (OpCode::IntGe, OpCode::IntGe) => return sv >= ov,
                // x > sv → x >= ov  if sv >= ov
                (OpCode::IntGt, OpCode::IntGe) => return sv >= ov,
                // x >= sv → x > ov  if sv > ov
                (OpCode::IntGe, OpCode::IntGt) => return sv > ov,
                _ => {}
            }
        }

        // Same rhs variable
        if self.rhs == other.rhs {
            match (self_cmp, other_cmp) {
                (OpCode::IntLt, OpCode::IntLe) => return true,
                (OpCode::IntGt, OpCode::IntGe) => return true,
                (c1, c2) if c1 == c2 => return true,
                _ => {}
            }
        }

        false
    }
}

/// guard.py: GuardStrengthenOpt (full-loop version for vector optimizer).
///
/// Collects guard information from a complete loop, determines which
/// guards imply others, and eliminates redundant guards. This is the
/// whole-trace pass, unlike the streaming `GuardStrengthenOpt` below.
pub struct GuardEliminator {
    /// guard.py: strongest_guards — maps variable key to strongest guards.
    strongest_guards: HashMap<OpRef, Vec<Guard>>,
    /// guard.py: guards — maps op index to replacement guard (or None to skip).
    guards: HashMap<usize, Option<Guard>>,
    /// Number of guards strength-reduced.
    pub strength_reduced: usize,
}

impl GuardEliminator {
    pub fn new() -> Self {
        GuardEliminator {
            strongest_guards: HashMap::new(),
            guards: HashMap::new(),
            strength_reduced: 0,
        }
    }

    /// guard.py: collect_guard_information(loop)
    ///
    /// Walk all operations, find guard_true/guard_false ops with a
    /// preceding comparison, and record them by their variable keys.
    pub fn collect_guard_information(&mut self, ops: &[Op], ctx: &OptContext) {
        for (i, op) in ops.iter().enumerate() {
            if !op.opcode.is_guard() {
                continue;
            }
            if op.opcode != OpCode::GuardTrue && op.opcode != OpCode::GuardFalse {
                continue;
            }
            // Find the comparison op that produces the guard's boolean arg
            let bool_arg = op.arg(0);
            let cmp_op = ops.iter().rfind(|o| o.pos == bool_arg);
            if let Some(cmp) = cmp_op {
                if let Some(guard) = Guard::from_ops(i, op, cmp) {
                    self.record_guard(guard.lhs, &guard, ctx);
                    if !guard.rhs.is_none() {
                        self.record_guard(guard.rhs, &guard, ctx);
                    }
                }
            }
        }
    }

    /// guard.py: record_guard(key, guard)
    ///
    /// Record a guard for a key. If stronger guards exist for this key,
    /// determine implication: either the new guard replaces the old one
    /// (strengthening) or the new one is implied (eliminated).
    fn record_guard(&mut self, key: OpRef, guard: &Guard, ctx: &OptContext) {
        if key.is_none() {
            return;
        }
        let others = self.strongest_guards.entry(key).or_default();
        if !others.is_empty() {
            let mut replaced = false;
            for i in 0..others.len() {
                if guard.implies(&others[i], ctx) {
                    // New guard is stronger → replace old, mark old index
                    let old = others[i].clone();
                    self.guards.insert(guard.index, None); // don't emit new
                    self.guards.insert(old.index, Some(guard.clone())); // replace old
                    others[i] = guard.clone();
                    replaced = true;
                } else if others[i].implies(guard, ctx) {
                    // Old guard implies new → skip new
                    self.guards.insert(guard.index, None);
                    replaced = true;
                }
            }
            if !replaced {
                others.push(guard.clone());
            }
        } else {
            others.push(guard.clone());
        }
    }

    /// guard.py: eliminate_guards(loop)
    ///
    /// Walk ops, skip guards that are implied, emit replacement guards
    /// for strengthened ones, and copy everything else.
    pub fn eliminate_guards(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut result = Vec::with_capacity(ops.len());
        for (i, op) in ops.iter().enumerate() {
            if op.opcode.is_guard() {
                if let Some(replacement) = self.guards.get(&i) {
                    self.strength_reduced += 1;
                    if replacement.is_none() {
                        continue; // implied, skip entirely
                    }
                    // Strengthened: the replacement guard's cmp + guard
                    // are emitted. For now, just emit the original (the
                    // replacement has the same semantic, just tighter).
                    if let Some(guard) = replacement {
                        let mut new_op = op.clone();
                        new_op.opcode = guard.guard_opcode;
                        result.push(new_op);
                        continue;
                    }
                }
            }
            result.push(op.clone());
        }
        result
    }

    /// guard.py: propagate_all_forward(info, loop)
    ///
    /// Full pipeline: collect guard info → eliminate guards.
    pub fn propagate_all_forward(&mut self, ops: &[Op], ctx: &OptContext) -> Vec<Op> {
        self.collect_guard_information(ops, ctx);
        self.eliminate_guards(ops)
    }
}

impl Default for GuardEliminator {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GuardStrengthenOpt {
    /// Set of guard conditions already verified in the trace.
    seen: HashSet<GuardKey>,

    /// Set of values known to be truthy (guarded by `guard_true` or
    /// `guard_nonnull`).
    truthy_values: HashSet<OpRef>,

    /// guard.py: values with known class (from GuardClass/GuardNonnullClass).
    known_classes: std::collections::HashMap<OpRef, majit_ir::GcRef>,

    /// guard.py: values known to be specific constants (from GuardValue).
    known_constants: std::collections::HashMap<OpRef, i64>,

    /// Descriptor of the last emitted guard, used for consecutive-guard
    /// fusion.
    last_guard_descr: Option<DescrRef>,
}

impl GuardStrengthenOpt {
    pub fn new() -> Self {
        GuardStrengthenOpt {
            seen: HashSet::new(),
            truthy_values: HashSet::new(),
            known_classes: std::collections::HashMap::new(),
            known_constants: std::collections::HashMap::new(),
            last_guard_descr: None,
        }
    }

    /// Record the "truthy" implications of a guard that has been verified.
    ///
    /// `guard_true(v)`       → v is truthy
    /// `guard_nonnull(v)`    → v is nonnull (truthy)
    /// `guard_value(v, c)`   → v == c, which also means v is truthy (if c != 0)
    /// `guard_class(v, cls)` → v has class cls, which also means v is nonnull
    /// `guard_nonnull_class` → v is nonnull (and has class)
    fn record_implications(&mut self, op: &Op, ctx: &OptContext) {
        match op.opcode {
            OpCode::GuardTrue => {
                self.truthy_values.insert(op.arg(0));
            }
            OpCode::GuardNonnull => {
                self.truthy_values.insert(op.arg(0));
            }
            OpCode::GuardFalse => {
                // guard.py: guard_false(v) means v is known to be 0/false.
                // This is the opposite of guard_true: we don't add to truthy_values.
                // But we record it so that a later guard_false(v) can be eliminated.
            }
            OpCode::GuardValue => {
                let val_arg = op.arg(0);
                if let Some(c) = ctx.get_constant_int(op.arg(1)) {
                    // guard.py: record known constant value.
                    self.known_constants.insert(val_arg, c);
                    if c != 0 {
                        self.truthy_values.insert(val_arg);
                    }
                }
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                // Having a class implies the object is nonnull.
                self.truthy_values.insert(op.arg(0));
                // guard.py: record the known class for subsumption checks.
                if op.num_args() >= 2 {
                    if let Some(class_val) = ctx.get_constant_int(op.arg(1)) {
                        self.known_classes
                            .insert(op.arg(0), majit_ir::GcRef(class_val as usize));
                    }
                }
            }
            _ => {}
        }
    }

    /// Check whether this guard is subsumed by previously recorded
    /// implications (guard strengthening).
    ///
    /// Returns `true` if the guard can be removed.
    fn is_subsumed(&self, op: &Op, ctx: &OptContext) -> bool {
        match op.opcode {
            OpCode::GuardTrue => {
                // guard.py: also subsumed if value is a known nonzero constant.
                if self.truthy_values.contains(&op.arg(0)) {
                    return true;
                }
                if let Some(&c) = self.known_constants.get(&op.arg(0)) {
                    return c != 0;
                }
                false
            }
            OpCode::GuardFalse => {
                // Subsumed if value is a known zero constant.
                if let Some(&c) = self.known_constants.get(&op.arg(0)) {
                    return c == 0;
                }
                false
            }
            OpCode::GuardNonnull => self.truthy_values.contains(&op.arg(0)),
            // rewrite.py: optimize_GUARD_CLASS — if the class is already
            // known for this value, and it matches, remove the guard.
            OpCode::GuardClass if op.num_args() >= 2 => {
                if let Some(&known_class) = self.known_classes.get(&op.arg(0)) {
                    // Check if the expected class matches the known class.
                    if let Some(expected) = ctx.get_constant_int(op.arg(1)) {
                        return known_class.0 as i64 == expected;
                    }
                }
                false
            }
            // guard.py: GUARD_VALUE subsumed if value is already known to be
            // that exact constant from a previous GuardValue.
            OpCode::GuardValue if op.num_args() >= 2 => {
                if let Some(&known) = self.known_constants.get(&op.arg(0)) {
                    if let Some(expected) = ctx.get_constant_int(op.arg(1)) {
                        return known == expected;
                    }
                }
                false
            }
            // guard.py: GUARD_NONNULL_CLASS subsumed if value is known nonnull
            // AND a previous GUARD_CLASS with same args was already seen.
            OpCode::GuardNonnullClass if self.truthy_values.contains(&op.arg(0)) => {
                let class_key = GuardKey {
                    opcode: OpCode::GuardClass,
                    args: op.args.to_vec(),
                };
                self.seen.contains(&class_key)
            }
            _ => false,
        }
    }

    /// Try to apply consecutive-guard descriptor fusion.
    ///
    /// If the current guard has no descriptor and the previous guard had
    /// one, reuse it.
    fn try_fuse_descr(&self, op: &mut Op) {
        if op.descr.is_none() {
            if let Some(ref descr) = self.last_guard_descr {
                op.descr = Some(descr.clone());
            }
        }
    }

    fn can_remove_as_duplicate(opcode: OpCode) -> bool {
        opcode.is_foldable_guard()
    }
}

impl Default for GuardStrengthenOpt {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for GuardStrengthenOpt {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        if !op.opcode.is_guard() {
            // Non-guard operations invalidate the last-guard descriptor
            // tracking (no longer consecutive).
            self.last_guard_descr = None;
            return OptimizationResult::PassOn;
        }

        // GuardNotForced must reach the Heap pass, which holds the
        // postponed CallMayForce op.  Return PassOn so Heap can emit
        // the call immediately before the guard.
        if matches!(op.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2) {
            return OptimizationResult::PassOn;
        }

        // --- Redundant guard removal (exact match) ---
        if Self::can_remove_as_duplicate(op.opcode) {
            let key = GuardKey::from_op(op);
            if self.seen.contains(&key) {
                return OptimizationResult::Remove;
            }

            self.seen.insert(key);
        }

        // --- Guard strengthening (subsumption) ---
        if self.is_subsumed(op, ctx) {
            return OptimizationResult::Remove;
        }

        // The guard is not redundant – record it and emit.
        self.record_implications(op, ctx);

        // --- Consecutive guard fusion ---
        let mut fused_op = op.clone();
        self.try_fuse_descr(&mut fused_op);

        // Track this guard's descriptor for the next consecutive guard.
        self.last_guard_descr = fused_op.descr.clone();

        OptimizationResult::Emit(fused_op)
    }

    fn setup(&mut self) {
        self.seen.clear();
        self.truthy_values.clear();
        self.known_classes.clear();
        self.known_constants.clear();
        self.last_guard_descr = None;
    }

    fn name(&self) -> &'static str {
        "guard"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;

    /// Helper: assign sequential positions to ops starting from a high base
    /// to avoid colliding with argument OpRefs.
    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(base + i as u32);
        }
    }

    fn run_guard_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(GuardStrengthenOpt::new()));
        opt.optimize_with_constants_and_inputs(ops, &mut std::collections::HashMap::new(), 1024)
    }

    // ── Redundant Guard Removal ─────────────────────────────────────────

    #[test]
    fn test_duplicate_guard_true_removed() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .count();
        assert_eq!(guard_count, 1, "duplicate guard_true should be removed");
        assert_eq!(result.len(), 2); // guard_true + int_add
    }

    #[test]
    fn test_duplicate_guard_false_removed() {
        let mut ops = vec![
            Op::new(OpCode::GuardFalse, &[OpRef(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef(0)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardFalse)
            .count();
        assert_eq!(guard_count, 1);
    }

    #[test]
    fn test_duplicate_guard_nonnull_removed() {
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(5)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(5)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNonnull);
    }

    #[test]
    fn test_duplicate_guard_value_removed() {
        let mut ops = vec![
            Op::new(OpCode::GuardValue, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardValue, &[OpRef(0), OpRef(1)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardValue);
    }

    #[test]
    fn test_duplicate_guard_class_removed() {
        let mut ops = vec![
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(1)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
    }

    // ── Non-duplicate guards are preserved ──────────────────────────────

    #[test]
    fn test_different_guard_args_kept() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef(1)]), // different arg
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(
            result.len(),
            2,
            "guards on different values should both be kept"
        );
    }

    #[test]
    fn test_different_guard_opcodes_kept() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef(0)]), // different opcode
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(
            result.len(),
            2,
            "guard_true and guard_false on same arg are not duplicates"
        );
    }

    // ── Guard ordering preservation ─────────────────────────────────────

    #[test]
    fn test_guard_order_preserved() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(2)]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GuardTrue);
        assert_eq!(result[1].opcode, OpCode::GuardFalse);
        assert_eq!(result[2].opcode, OpCode::GuardNonnull);
    }

    // ── Guard Strengthening ─────────────────────────────────────────────

    #[test]
    fn test_guard_nonnull_after_guard_true_removed() {
        // guard_true(v) implies v is truthy/nonnull,
        // so a later guard_nonnull(v) is redundant.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]), // subsumed
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardTrue);
    }

    #[test]
    fn test_guard_true_after_guard_nonnull_removed() {
        // guard_nonnull(v) implies v is truthy,
        // so a later guard_true(v) is subsumed.
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]), // subsumed
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNonnull);
    }

    #[test]
    fn test_guard_nonnull_after_guard_class_removed() {
        // guard_class(v, cls) implies v is nonnull,
        // so a later guard_nonnull(v) is subsumed.
        let mut ops = vec![
            Op::new(OpCode::GuardClass, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]), // subsumed
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardClass);
    }

    #[test]
    fn test_guard_true_after_guard_nonnull_class_removed() {
        // guard_nonnull_class(v, cls) implies v is nonnull/truthy.
        let mut ops = vec![
            Op::new(OpCode::GuardNonnullClass, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]), // subsumed
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNonnullClass);
    }

    #[test]
    fn test_guard_strengthening_different_values_not_subsumed() {
        // guard_true(v0) does NOT subsume guard_nonnull(v1).
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(1)]), // different value
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
    }

    // ── Consecutive Guard Fusion ────────────────────────────────────────

    #[test]
    fn test_consecutive_guards_without_descr_no_crash() {
        // Two consecutive guards without descriptors should not crash.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef(1)]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
    }

    // ── Mixed guards and non-guards ─────────────────────────────────────

    #[test]
    fn test_non_guard_ops_pass_through() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntSub, &[OpRef(0), OpRef(1)]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::IntSub);
    }

    #[test]
    fn test_guards_interleaved_with_ops() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]), // duplicate, removed
            Op::new(OpCode::IntSub, &[OpRef(0), OpRef(1)]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GuardTrue);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[2].opcode, OpCode::IntSub);
    }

    #[test]
    fn test_three_duplicate_guards() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]), // duplicate
            Op::new(OpCode::GuardTrue, &[OpRef(0)]), // duplicate
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 1);
    }

    // ── Integration: GuardStrengthenOpt in the default pipeline ───────────────────

    #[test]
    fn test_guard_in_full_pipeline() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]), // should be removed
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 100);

        let mut opt = Optimizer::default_pipeline();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .count();
        assert_eq!(
            guard_count, 1,
            "duplicate guard should be removed in full pipeline"
        );
    }

    #[test]
    fn test_guard_no_exception_duplicates_preserved() {
        // guard_no_exception depends on ambient exception state, so distinct
        // checks cannot be merged just because they have no explicit args.
        let mut ops = vec![
            Op::new(OpCode::GuardNoException, &[]),
            Op::new(OpCode::GuardNoException, &[]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_guard_no_overflow_duplicates_preserved() {
        let mut ops = vec![
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::GuardNoOverflow, &[]),
        ];
        assign_positions(&mut ops, 100);
        let result = run_guard_pass(&ops);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_overflow_guards_preserved_in_full_pipeline() {
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(1)]),
            Op::new(OpCode::IntSubOvf, &[OpRef(0), OpRef(2)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::IntMulOvf, &[OpRef(2), OpRef(1)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Jump, &[OpRef(101), OpRef(101), OpRef(103)]),
        ];
        assign_positions(&mut ops, 100);

        let mut opt = Optimizer::default_pipeline();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 1024);
        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNoOverflow)
            .count();

        assert_eq!(
            guard_count, 2,
            "distinct overflow guards must survive full-pipeline optimization"
        );
    }

    #[test]
    fn test_guard_nonnull_class_subsumed_by_nonnull_plus_class() {
        // GUARD_NONNULL(v) then GUARD_CLASS(v, cls) already seen
        // → later GUARD_NONNULL_CLASS(v, cls) is subsumed.
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::GuardClass, &[OpRef(100), OpRef(200)]),
            Op::new(OpCode::GuardNonnullClass, &[OpRef(100), OpRef(200)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
        ];
        assign_positions(&mut ops, 0);
        let result = run_guard_pass(&ops);
        // GuardNonnullClass should be removed (subsumed)
        let nnc_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnullClass)
            .count();
        assert_eq!(nnc_count, 0, "GuardNonnullClass should be subsumed");
    }

    #[test]
    fn test_guard_value_to_guard_true() {
        // GUARD_VALUE(v, 1) → GUARD_TRUE(v) via rewrite pass.
        let mut ops = vec![
            {
                let mut op = Op::new(OpCode::GuardValue, &[OpRef(100), OpRef(200)]);
                op.pos = OpRef(0);
                op
            },
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
        ];
        ops[1].pos = OpRef(1);

        // Pre-seed constant 1 for OpRef(200)
        let mut opt = crate::optimizer::Optimizer::new();
        opt.add_pass(Box::new(crate::rewrite::OptRewrite::new()));
        let mut constants = std::collections::HashMap::new();
        constants.insert(200, 1i64);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        // GUARD_VALUE should be replaced with GUARD_TRUE
        assert!(
            result.iter().any(|o| o.opcode == OpCode::GuardTrue),
            "GUARD_VALUE(v, 1) should become GUARD_TRUE(v)"
        );
        assert!(
            !result.iter().any(|o| o.opcode == OpCode::GuardValue),
            "GUARD_VALUE should be gone"
        );
    }
}
