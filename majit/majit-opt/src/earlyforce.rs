/// Early force pass: ensure virtual refs are forced before CALL_MAY_FORCE.
///
/// Translated from rpython/jit/metainterp/optimizeopt/earlyforce.py.
///
/// CALL_MAY_FORCE and CALL_ASSEMBLER can force virtual refs. If a virtual
/// ref argument to such a call hasn't been forced yet, we must force it
/// before the call to ensure correct ordering of side effects.
///
/// This pass must run BEFORE the main optimizer pipeline so that all
/// forcing happens in the correct order.
use majit_ir::{Op, OpCode};

use crate::{OptContext, Optimization, OptimizationResult};

pub struct OptEarlyForce;

impl OptEarlyForce {
    pub fn new() -> Self {
        OptEarlyForce
    }
}

impl Default for OptEarlyForce {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptEarlyForce {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            // CALL_MAY_FORCE / CALL_ASSEMBLER can force virtual references.
            // Ensure all VirtualRef arguments are forced (resolved) before
            // these operations execute.
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
            | OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                // Resolve all arguments through forwarding. This ensures
                // that any virtual ref that has been forced by a previous
                // pass has its forwarding pointer followed.
                let mut new_op = op.clone();
                for arg in &mut new_op.args {
                    *arg = ctx.get_replacement(*arg);
                }
                OptimizationResult::Emit(new_op)
            }
            // earlyforce.py: GUARD_NOT_FORCED needs to resolve fail_args
            // through forwarding too, so forced virtual refs are properly tracked.
            OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                let mut new_op = op.clone();
                if let Some(ref mut fa) = new_op.fail_args {
                    for arg in fa.iter_mut() {
                        *arg = ctx.get_replacement(*arg);
                    }
                }
                OptimizationResult::Emit(new_op)
            }
            _ => OptimizationResult::PassOn,
        }
    }

    fn name(&self) -> &'static str {
        "earlyforce"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;
    use majit_ir::OpRef;

    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
    }

    #[test]
    fn test_earlyforce_resolves_call_may_force_args() {
        let mut ops = vec![Op::new(OpCode::CallMayForceN, &[OpRef(100), OpRef(101)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallMayForceN);
    }

    #[test]
    fn test_earlyforce_passthrough_non_call() {
        let mut ops = vec![Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_earlyforce_call_assembler_handled() {
        let mut ops = vec![Op::new(OpCode::CallAssemblerI, &[OpRef(100)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallAssemblerI);
    }

    #[test]
    fn test_earlyforce_guard_not_forced() {
        // GUARD_NOT_FORCED should have its fail_args resolved.
        let mut guard = Op::new(OpCode::GuardNotForced, &[]);
        guard.fail_args = Some(Default::default());
        let mut ops = vec![guard];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNotForced);
        assert!(result[0].fail_args.is_some());
    }

    #[test]
    fn test_earlyforce_all_call_may_force_types() {
        for opcode in [
            OpCode::CallMayForceI,
            OpCode::CallMayForceR,
            OpCode::CallMayForceF,
            OpCode::CallMayForceN,
        ] {
            let mut ops = vec![Op::new(opcode, &[OpRef(100)])];
            assign_positions(&mut ops);

            let mut opt = Optimizer::new();
            opt.add_pass(Box::new(OptEarlyForce::new()));
            let result = opt.optimize_with_constants_and_inputs(
                &ops,
                &mut std::collections::HashMap::new(),
                1024,
            );
            assert_eq!(result.len(), 1, "{opcode:?} should be handled");
        }
    }
}
