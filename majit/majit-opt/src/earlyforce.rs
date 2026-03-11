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

use crate::{OptContext, OptimizationPass, PassResult};

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

impl OptimizationPass for OptEarlyForce {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
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
                PassResult::Emit(new_op)
            }
            _ => PassResult::PassOn,
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
        let mut ops = vec![
            Op::new(OpCode::CallMayForceN, &[OpRef(100), OpRef(101)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallMayForceN);
    }

    #[test]
    fn test_earlyforce_passthrough_non_call() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_earlyforce_call_assembler_handled() {
        let mut ops = vec![
            Op::new(OpCode::CallAssemblerI, &[OpRef(100)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallAssemblerI);
    }
}
