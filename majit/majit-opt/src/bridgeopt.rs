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

use crate::info::PtrInfo;
use crate::{OptContext, OptimizationPass, PassResult};

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

/// The bridge optimization pass.
pub struct OptBridgeOpt {
    knowledge: BridgeKnowledge,
}

impl OptBridgeOpt {
    pub fn new() -> Self {
        OptBridgeOpt {
            knowledge: BridgeKnowledge::new(),
        }
    }

    /// Create a bridge optimizer with pre-existing knowledge.
    pub fn with_knowledge(knowledge: BridgeKnowledge) -> Self {
        OptBridgeOpt { knowledge }
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

impl OptimizationPass for OptBridgeOpt {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        match op.opcode {
            // If the guard value is already known as a constant,
            // the guard is redundant.
            OpCode::GuardValue => {
                let obj = ctx.get_replacement(op.arg(0));
                let expected = ctx.get_replacement(op.arg(1));
                if let (Some(a), Some(b)) = (ctx.get_constant_int(obj), ctx.get_constant_int(expected)) {
                    if a == b {
                        return PassResult::Remove;
                    }
                }
                PassResult::PassOn
            }

            // If we know the value is non-null from bridge knowledge,
            // the guard is redundant.
            OpCode::GuardNonnull => {
                let obj = ctx.get_replacement(op.arg(0));
                if self.knowledge.known_nonnull.contains(&obj) {
                    return PassResult::Remove;
                }
                PassResult::PassOn
            }

            // If we know the class from bridge knowledge, class guard is redundant.
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                let obj = ctx.get_replacement(op.arg(0));
                if self.knowledge.known_classes.contains_key(&obj) {
                    return PassResult::Remove;
                }
                PassResult::PassOn
            }

            _ => PassResult::PassOn,
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

        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert!(result.is_empty(), "guard_nonnull should be removed when known nonnull");
    }

    #[test]
    fn test_bridgeopt_keeps_guard_nonnull_without_knowledge() {
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1, "guard_nonnull should be kept without knowledge");
    }

    #[test]
    fn test_bridgeopt_removes_guard_class_with_knowledge() {
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_classes.insert(OpRef(100), GcRef::NULL);

        let mut ops = vec![
            Op::new(OpCode::GuardClass, &[OpRef(100), OpRef(200)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize(&ops);

        assert!(result.is_empty(), "guard_class should be removed when class known");
    }

    #[test]
    fn test_bridgeopt_guard_value_with_known_constant() {
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_constants.insert(OpRef(100), 42);

        let mut ops = vec![
            Op::new(OpCode::GuardValue, &[OpRef(100), OpRef(200)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        let bridge = OptBridgeOpt::with_knowledge(knowledge);
        // Pre-populate constants
        let mut ctx = OptContext::new(ops.len());
        bridge.apply_knowledge(&mut ctx);
        ctx.make_constant(OpRef(200), Value::Int(42));

        // Manual optimization to test with pre-populated constants
        let mut pass = OptBridgeOpt::new();
        let result = pass.propagate_forward(&ops[0], &mut ctx);
        match result {
            PassResult::Remove => {} // expected
            other => panic!("expected Remove, got {:?}", match other {
                PassResult::PassOn => "PassOn",
                PassResult::Emit(_) => "Emit",
                PassResult::Replace(_) => "Replace",
                _ => "other",
            }),
        }
    }
}
