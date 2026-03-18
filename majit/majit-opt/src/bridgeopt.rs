/// Bridge optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/bridgeopt.py.
///
/// When compiling a bridge (alternative path from a failed guard), we know
/// certain facts about values at the guard failure point — e.g., class is
/// known from the guard that failed, values are bounded from loop analysis.
///
/// This pass imports that knowledge so guards in the bridge body can be
/// removed or simplified.
use std::collections::HashMap;

use majit_ir::{GcRef, Op, OpCode, OpRef, Value};

use crate::{OptContext, Optimization, OptimizationResult};

/// Known facts about values at bridge entry.
#[derive(Clone, Debug)]
pub struct BridgeKnowledge {
    /// Values known to be specific constants.
    pub known_constants: HashMap<OpRef, i64>,
    /// Values known to be non-null.
    pub known_nonnull: Vec<OpRef>,
    /// Values with known class (from GuardClass).
    pub known_classes: HashMap<OpRef, GcRef>,
    /// Integer bounds: (opref, lower, upper).
    pub known_bounds: HashMap<OpRef, (i64, i64)>,
}

impl BridgeKnowledge {
    pub fn new() -> Self {
        BridgeKnowledge {
            known_constants: HashMap::new(),
            known_nonnull: Vec::new(),
            known_classes: HashMap::new(),
            known_bounds: HashMap::new(),
        }
    }
}

impl Default for BridgeKnowledge {
    fn default() -> Self {
        Self::new()
    }
}

impl BridgeKnowledge {
    /// bridgeopt.py: serialize_optimizer_knowledge(numb_state, liveboxes, ...)
    /// Serialize knowledge into a compact byte representation for embedding
    /// in resume data. This allows bridges to inherit optimization knowledge
    /// from the loop that spawned them.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Format: [num_constants, (opref, value)*, num_nonnull, opref*,
        //          num_classes, (opref, class_ptr)*, num_bounds, (opref, lo, hi)*]
        buf.extend_from_slice(&(self.known_constants.len() as u32).to_le_bytes());
        for (&opref, &value) in &self.known_constants {
            buf.extend_from_slice(&opref.0.to_le_bytes());
            buf.extend_from_slice(&value.to_le_bytes());
        }
        buf.extend_from_slice(&(self.known_nonnull.len() as u32).to_le_bytes());
        for opref in &self.known_nonnull {
            buf.extend_from_slice(&opref.0.to_le_bytes());
        }
        buf.extend_from_slice(&(self.known_classes.len() as u32).to_le_bytes());
        for (&opref, &class) in &self.known_classes {
            buf.extend_from_slice(&opref.0.to_le_bytes());
            buf.extend_from_slice(&(class.0 as u64).to_le_bytes());
        }
        buf.extend_from_slice(&(self.known_bounds.len() as u32).to_le_bytes());
        for (&opref, &(lo, hi)) in &self.known_bounds {
            buf.extend_from_slice(&opref.0.to_le_bytes());
            buf.extend_from_slice(&lo.to_le_bytes());
            buf.extend_from_slice(&hi.to_le_bytes());
        }
        buf
    }

    /// bridgeopt.py: deserialize_optimizer_knowledge(numb_state, liveboxes, ...)
    /// Reconstruct knowledge from serialized bytes.
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        let mut pos = 0;
        let mut k = BridgeKnowledge::new();

        if buf.len() < 4 { return None; }
        let n_const = u32::from_le_bytes(buf[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        for _ in 0..n_const {
            if pos + 12 > buf.len() { return None; }
            let opref = OpRef(u32::from_le_bytes(buf[pos..pos+4].try_into().ok()?));
            pos += 4;
            let value = i64::from_le_bytes(buf[pos..pos+8].try_into().ok()?);
            pos += 8;
            k.known_constants.insert(opref, value);
        }

        if pos + 4 > buf.len() { return None; }
        let n_nonnull = u32::from_le_bytes(buf[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        for _ in 0..n_nonnull {
            if pos + 4 > buf.len() { return None; }
            let opref = OpRef(u32::from_le_bytes(buf[pos..pos+4].try_into().ok()?));
            pos += 4;
            k.known_nonnull.push(opref);
        }

        if pos + 4 > buf.len() { return None; }
        let n_class = u32::from_le_bytes(buf[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        for _ in 0..n_class {
            if pos + 12 > buf.len() { return None; }
            let opref = OpRef(u32::from_le_bytes(buf[pos..pos+4].try_into().ok()?));
            pos += 4;
            let class_val = u64::from_le_bytes(buf[pos..pos+8].try_into().ok()?);
            pos += 8;
            k.known_classes.insert(opref, GcRef(class_val as usize));
        }

        if pos + 4 > buf.len() { return None; }
        let n_bounds = u32::from_le_bytes(buf[pos..pos+4].try_into().ok()?) as usize;
        pos += 4;
        for _ in 0..n_bounds {
            if pos + 20 > buf.len() { return None; }
            let opref = OpRef(u32::from_le_bytes(buf[pos..pos+4].try_into().ok()?));
            pos += 4;
            let lo = i64::from_le_bytes(buf[pos..pos+8].try_into().ok()?);
            pos += 8;
            let hi = i64::from_le_bytes(buf[pos..pos+8].try_into().ok()?);
            pos += 8;
            k.known_bounds.insert(opref, (lo, hi));
        }

        Some(k)
    }

    /// Whether this knowledge has any useful facts.
    pub fn is_empty(&self) -> bool {
        self.known_constants.is_empty()
            && self.known_nonnull.is_empty()
            && self.known_classes.is_empty()
            && self.known_bounds.is_empty()
    }
}

/// The bridge optimization pass.
pub struct OptBridgeOpt {
    knowledge: BridgeKnowledge,
    /// Maps OpRef to the Op that produced it, for looking up comparison ops
    /// when processing GuardTrue/GuardFalse.
    produced_by: HashMap<OpRef, Op>,
}

impl OptBridgeOpt {
    pub fn new() -> Self {
        OptBridgeOpt {
            knowledge: BridgeKnowledge::new(),
            produced_by: HashMap::new(),
        }
    }

    /// Create a bridge optimizer with pre-existing knowledge.
    pub fn with_knowledge(knowledge: BridgeKnowledge) -> Self {
        OptBridgeOpt {
            knowledge,
            produced_by: HashMap::new(),
        }
    }

    /// Get the effective bounds for an operand: either from known_bounds or
    /// from a known constant (where lower == upper == constant value).
    fn get_bounds(&self, opref: OpRef, ctx: &OptContext) -> Option<(i64, i64)> {
        if let Some(&bounds) = self.knowledge.known_bounds.get(&opref) {
            return Some(bounds);
        }
        if let Some(val) = ctx.get_constant_int(opref) {
            return Some((val, val));
        }
        None
    }

    /// Check whether a comparison guard can be eliminated based on known bounds.
    ///
    /// For GuardTrue, the condition must be provably always true.
    /// For GuardFalse, the negation of the condition must be provably always true.
    fn can_eliminate_comparison_guard(
        &self,
        cond_op: &Op,
        guard_true: bool,
        ctx: &OptContext,
    ) -> bool {
        let cmp_opcode = cond_op.opcode;
        let a = ctx.get_replacement(cond_op.arg(0));
        let b = ctx.get_replacement(cond_op.arg(1));

        let a_bounds = self.get_bounds(a, ctx);
        let b_bounds = self.get_bounds(b, ctx);

        let (a_lo, a_hi) = match a_bounds {
            Some(bounds) => bounds,
            None => return false,
        };
        let (b_lo, b_hi) = match b_bounds {
            Some(bounds) => bounds,
            None => return false,
        };

        if guard_true {
            // Guard asserts the comparison is true.
            match cmp_opcode {
                OpCode::IntLt => a_hi < b_lo,
                OpCode::IntLe => a_hi <= b_lo,
                OpCode::IntGt => a_lo > b_hi,
                OpCode::IntGe => a_lo >= b_hi,
                _ => false,
            }
        } else {
            // GuardFalse asserts the comparison is false.
            // IntLt false means a >= b, so check a_lo >= b_hi.
            match cmp_opcode {
                OpCode::IntLt => a_lo >= b_hi,
                OpCode::IntLe => a_lo > b_hi,
                OpCode::IntGt => a_hi <= b_lo,
                OpCode::IntGe => a_hi < b_lo,
                _ => false,
            }
        }
    }

    /// Pre-populate the optimization context with known facts.
    pub fn apply_knowledge(&self, ctx: &mut OptContext) {
        for (&opref, &value) in &self.knowledge.known_constants {
            ctx.make_constant(opref, Value::Int(value));
        }
    }
}

impl Default for OptBridgeOpt {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptBridgeOpt {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Track comparison operations so we can look them up from guards.
        match op.opcode {
            OpCode::IntLt | OpCode::IntLe | OpCode::IntGt | OpCode::IntGe => {
                self.produced_by.insert(op.pos, op.clone());
            }
            _ => {}
        }

        match op.opcode {
            // If the guard value is already known as a constant,
            // the guard is redundant.
            OpCode::GuardValue => {
                let obj = ctx.get_replacement(op.arg(0));
                let expected = ctx.get_replacement(op.arg(1));
                if let (Some(a), Some(b)) =
                    (ctx.get_constant_int(obj), ctx.get_constant_int(expected))
                {
                    if a == b {
                        return OptimizationResult::Remove;
                    }
                }
                OptimizationResult::PassOn
            }

            // If we know the value is non-null from bridge knowledge,
            // the guard is redundant.
            OpCode::GuardNonnull => {
                let obj = ctx.get_replacement(op.arg(0));
                if self.knowledge.known_nonnull.contains(&obj) {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }

            // If we know the class from bridge knowledge, class guard is redundant.
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                let obj = ctx.get_replacement(op.arg(0));
                if self.knowledge.known_classes.contains_key(&obj) {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }

            // Bounds-based guard elimination: if the comparison operands
            // have known bounds that prove the guard always succeeds,
            // the guard is redundant.
            OpCode::GuardTrue | OpCode::GuardFalse => {
                let cond_ref = ctx.get_replacement(op.arg(0));
                if let Some(cond_op) = self.produced_by.get(&cond_ref) {
                    let guard_true = op.opcode == OpCode::GuardTrue;
                    if self.can_eliminate_comparison_guard(cond_op, guard_true, ctx) {
                        return OptimizationResult::Remove;
                    }
                }
                OptimizationResult::PassOn
            }

            _ => OptimizationResult::PassOn,
        }
    }

    fn setup(&mut self) {}

    fn name(&self) -> &'static str {
        "bridgeopt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;

    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
    }

    #[test]
    fn test_bridgeopt_removes_guard_nonnull_with_knowledge() {
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_nonnull.push(OpRef(100));

        let mut ops = vec![Op::new(OpCode::GuardNonnull, &[OpRef(100)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert!(
            result.is_empty(),
            "guard_nonnull should be removed when known nonnull"
        );
    }

    #[test]
    fn test_bridgeopt_keeps_guard_nonnull_without_knowledge() {
        let mut ops = vec![Op::new(OpCode::GuardNonnull, &[OpRef(100)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::new()));
        let result = opt.optimize(&ops);

        assert_eq!(
            result.len(),
            1,
            "guard_nonnull should be kept without knowledge"
        );
    }

    #[test]
    fn test_bridgeopt_removes_guard_class_with_knowledge() {
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_classes.insert(OpRef(100), GcRef::NULL);

        let mut ops = vec![Op::new(OpCode::GuardClass, &[OpRef(100), OpRef(200)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert!(
            result.is_empty(),
            "guard_class should be removed when class known"
        );
    }

    // --- Bounds-based guard elimination tests ---

    #[test]
    fn test_bounds_guard_elimination_int_lt() {
        // known_bounds for OpRef(100) = (0, 50), constant 100
        // v0 = IntLt(OpRef(100), const_200), GuardTrue(v0)
        // Since 50 < 100, the guard is always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef(100), (0, 50));

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef(100), OpRef(200)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let mut constants = HashMap::new();
        constants.insert(200, 100);
        let result = opt.optimize_with_constants(&ops, &mut constants);

        // IntLt remains, GuardTrue removed
        assert_eq!(result.len(), 1, "GuardTrue should be removed");
        assert_eq!(result[0].opcode, OpCode::IntLt);
    }

    #[test]
    fn test_bounds_guard_not_eliminated_int_lt() {
        // known_bounds for OpRef(100) = (0, 200), constant 100
        // Since 200 >= 100, cannot prove IntLt always true → keep guard.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef(100), (0, 200));

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef(100), OpRef(200)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let mut constants = HashMap::new();
        constants.insert(200, 100);
        let result = opt.optimize_with_constants(&ops, &mut constants);

        assert_eq!(result.len(), 2, "GuardTrue should NOT be removed");
        assert_eq!(result[0].opcode, OpCode::IntLt);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
    }

    #[test]
    fn test_bounds_guard_elimination_int_ge() {
        // known_bounds for OpRef(100) = (50, 200)
        // known_bounds for OpRef(101) = (10, 50)
        // IntGe(100, 101): a.lower(50) >= b.upper(50) → always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef(100), (50, 200));
        knowledge.known_bounds.insert(OpRef(101), (10, 50));

        let mut ops = vec![
            Op::new(OpCode::IntGe, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1, "GuardTrue should be removed");
        assert_eq!(result[0].opcode, OpCode::IntGe);
    }

    #[test]
    fn test_bounds_guard_elimination_int_le() {
        // known_bounds for OpRef(100) = (0, 30)
        // known_bounds for OpRef(101) = (30, 100)
        // IntLe(100, 101): a.upper(30) <= b.lower(30) → always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef(100), (0, 30));
        knowledge.known_bounds.insert(OpRef(101), (30, 100));

        let mut ops = vec![
            Op::new(OpCode::IntLe, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1, "GuardTrue should be removed");
        assert_eq!(result[0].opcode, OpCode::IntLe);
    }

    #[test]
    fn test_bounds_guard_elimination_int_gt() {
        // known_bounds for OpRef(100) = (80, 200)
        // known_bounds for OpRef(101) = (10, 50)
        // IntGt(100, 101): a.lower(80) > b.upper(50) → always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef(100), (80, 200));
        knowledge.known_bounds.insert(OpRef(101), (10, 50));

        let mut ops = vec![
            Op::new(OpCode::IntGt, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1, "GuardTrue should be removed");
        assert_eq!(result[0].opcode, OpCode::IntGt);
    }

    #[test]
    fn test_bounds_guard_false_elimination() {
        // GuardFalse(IntLt(a, b)) means a >= b.
        // known_bounds for OpRef(100) = (80, 200), OpRef(101) = (10, 50)
        // IntLt false means a >= b: a.lower(80) >= b.upper(50) → always false → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef(100), (80, 200));
        knowledge.known_bounds.insert(OpRef(101), (10, 50));

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardFalse, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1, "GuardFalse should be removed");
        assert_eq!(result[0].opcode, OpCode::IntLt);
    }

    #[test]
    fn test_bounds_guard_false_not_eliminated() {
        // GuardFalse(IntLt(a, b)) means a >= b.
        // known_bounds for OpRef(100) = (0, 200), OpRef(101) = (10, 50)
        // a.lower(0) < b.upper(50), so cannot prove a >= b → keep guard.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef(100), (0, 200));
        knowledge.known_bounds.insert(OpRef(101), (10, 50));

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardFalse, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 2, "GuardFalse should NOT be removed");
    }

    #[test]
    fn test_bounds_guard_with_one_constant_operand() {
        // OpRef(100) has known bounds (0, 50), OpRef(200) is a constant 100.
        // IntLt(100, 200): a.upper(50) < b_const(100) → always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef(100), (0, 50));
        knowledge.known_constants.insert(OpRef(200), 100);

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef(100), OpRef(200)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let bridge = OptBridgeOpt::with_knowledge(knowledge);
        let mut ctx = OptContext::new(ops.len());
        bridge.apply_knowledge(&mut ctx);

        let mut pass = OptBridgeOpt::with_knowledge(bridge.knowledge.clone());
        // Process IntLt first so produced_by is populated.
        let r1 = pass.propagate_forward(&ops[0], &mut ctx);
        assert!(matches!(r1, OptimizationResult::PassOn));
        // Now process GuardTrue.
        let r2 = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(
            matches!(r2, OptimizationResult::Remove),
            "GuardTrue should be removed with one constant operand"
        );
    }

    #[test]
    fn test_bounds_guard_no_bounds_info() {
        // No bounds info for either operand → guard kept.
        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::new()));
        let result = opt.optimize(&ops);

        assert_eq!(
            result.len(),
            2,
            "Guard should be kept when no bounds info available"
        );
    }

    #[test]
    fn test_bridgeopt_guard_value_with_known_constant() {
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_constants.insert(OpRef(100), 42);

        let mut ops = vec![Op::new(OpCode::GuardValue, &[OpRef(100), OpRef(200)])];
        assign_positions(&mut ops);

        let bridge = OptBridgeOpt::with_knowledge(knowledge);
        // Pre-populate constants
        let mut ctx = OptContext::new(ops.len());
        bridge.apply_knowledge(&mut ctx);
        ctx.make_constant(OpRef(200), Value::Int(42));

        // Manual optimization to test with pre-populated constants
        let mut pass = OptBridgeOpt::new();
        let result = pass.propagate_forward(&ops[0], &mut ctx);
        match result {
            OptimizationResult::Remove => {} // expected
            other => panic!(
                "expected Remove, got {:?}",
                match other {
                    OptimizationResult::PassOn => "PassOn",
                    OptimizationResult::Emit(_) => "Emit",
                    OptimizationResult::Replace(_) => "Replace",
                    _ => "other",
                }
            ),
        }
    }
}
