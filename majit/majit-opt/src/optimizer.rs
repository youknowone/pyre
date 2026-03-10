/// Main optimization driver.
///
/// Translated from rpython/jit/metainterp/optimizeopt/optimizer.py.
/// Chains multiple optimization passes and drives operations through them.

use majit_ir::Op;
use crate::{OptContext, OptimizationPass, PassResult};

/// The optimizer: chains passes and runs them over a trace.
pub struct Optimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl Optimizer {
    pub fn new() -> Self {
        Optimizer { passes: Vec::new() }
    }

    /// Add an optimization pass to the chain.
    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    /// Run all optimization passes over a list of operations.
    ///
    /// Returns the optimized operation list.
    pub fn optimize(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut ctx = OptContext::new(ops.len());

        // Setup all passes
        for pass in &mut self.passes {
            pass.setup();
        }

        // Process each operation through the pass chain
        for op in ops {
            self.propagate_one(op, &mut ctx);
        }

        // Flush all passes
        for pass in &mut self.passes {
            pass.flush();
        }

        ctx.new_operations
    }

    /// Send one operation through the pass chain.
    fn propagate_one(&mut self, op: &Op, ctx: &mut OptContext) {
        // Resolve forwarded arguments
        let mut resolved_op = op.clone();
        for arg in &mut resolved_op.args {
            *arg = ctx.get_replacement(*arg);
        }

        let mut current_op = resolved_op;

        for pass in &mut self.passes {
            match pass.propagate_forward(&current_op, ctx) {
                PassResult::Emit(op) => {
                    ctx.emit(op);
                    return;
                }
                PassResult::Replace(op) => {
                    current_op = op;
                    // Continue to next pass with the replacement
                }
                PassResult::Remove => {
                    return;
                }
                PassResult::PassOn => {
                    // Continue to next pass with the same op
                }
            }
        }

        // If no pass handled it, emit as-is
        ctx.emit(current_op);
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{OpCode, OpRef};

    /// A trivial pass that removes INT_ADD(x, 0) -> x
    struct AddZeroElimination;

    impl OptimizationPass for AddZeroElimination {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
            if op.opcode == OpCode::IntAdd {
                // Check if second arg is constant 0
                if let Some(0) = ctx.get_constant_int(op.arg(1)) {
                    // Replace with first arg
                    ctx.replace_op(op.pos, op.arg(0));
                    return PassResult::Remove;
                }
            }
            PassResult::PassOn
        }

        fn name(&self) -> &'static str { "add_zero_elim" }
    }

    #[test]
    fn test_optimizer_passthrough() {
        let mut opt = Optimizer::new();
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
        ];
        let result = opt.optimize(&ops);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }
}
