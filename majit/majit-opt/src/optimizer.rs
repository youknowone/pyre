use crate::{OptContext, OptimizationPass, PassResult};
/// Main optimization driver.
///
/// Translated from rpython/jit/metainterp/optimizeopt/optimizer.py.
/// Chains multiple optimization passes and drives operations through them.
use crate::{
    guard::OptGuard, heap::OptHeap, intbounds::OptIntBounds, pure::OptPure,
    rewrite::OptRewrite, simplify::OptSimplify, virtualize::OptVirtualize, vstring::OptString,
};
use majit_ir::Op;

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
        self.optimize_with_constants(ops, &mut std::collections::HashMap::new())
    }

    /// Run all optimization passes, with known constants pre-populated.
    ///
    /// `constants` maps OpRef indices to their integer values. This allows the
    /// optimizer to constant-fold and eliminate guards on known-constant values
    /// (e.g., constants from the trace's constant pool).
    ///
    /// After optimization, newly-discovered constants (from constant folding)
    /// are written back into the map so the backend can resolve them.
    pub fn optimize_with_constants(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
    ) -> Vec<Op> {
        self.optimize_with_constants_and_inputs(ops, constants, 0)
    }

    /// Like `optimize_with_constants`, but also takes `num_inputs` so that
    /// ops emitted by the optimizer (e.g. from force_virtual) get pos values
    /// that don't collide with input argument variable indices.
    pub fn optimize_with_constants_and_inputs(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
    ) -> Vec<Op> {
        use majit_ir::{OpRef, Value};
        let mut ctx = OptContext::with_num_inputs(ops.len(), num_inputs);

        // Pre-populate known constants so passes can see them.
        for (&idx, &val) in constants.iter() {
            ctx.make_constant(OpRef(idx), Value::Int(val));
        }

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

        // Export newly-discovered constants back to the caller's map.
        for (idx, val) in ctx.constants.iter().enumerate() {
            if let Some(Value::Int(v)) = val {
                constants.entry(idx as u32).or_insert(*v);
            }
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
        if let Some(ref mut fa) = resolved_op.fail_args {
            for arg in fa.iter_mut() {
                *arg = ctx.get_replacement(*arg);
            }
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

impl Optimizer {
    /// Create an optimizer with the standard pass pipeline.
    /// Order: IntBounds -> Rewrite -> Virtualize -> String -> Pure -> Guard -> Simplify -> Heap
    pub fn default_pipeline() -> Self {
        let mut opt = Self::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.add_pass(Box::new(OptVirtualize::new()));
        opt.add_pass(Box::new(OptString::new()));
        opt.add_pass(Box::new(OptPure::new()));
        opt.add_pass(Box::new(OptGuard::new()));
        opt.add_pass(Box::new(OptSimplify::new()));
        opt.add_pass(Box::new(OptHeap::new()));
        opt
    }

    /// Number of passes in this optimizer.
    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::default_pipeline()
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

        fn name(&self) -> &'static str {
            "add_zero_elim"
        }
    }

    #[test]
    fn test_optimizer_passthrough() {
        let mut opt = Optimizer::new();
        let ops = vec![Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)])];
        let result = opt.optimize(&ops);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_default_pipeline_has_8_passes() {
        let opt = Optimizer::default_pipeline();
        assert_eq!(opt.num_passes(), 8);
    }

    #[test]
    fn test_default_pipeline_processes_trace() {
        let mut opt = Optimizer::default_pipeline();
        // A simple trace: two INT_ADD with identical args. The Pure pass (CSE)
        // should eliminate the duplicate.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
        let result = opt.optimize(&ops);
        // The duplicate INT_ADD should be eliminated by CSE (OptPure).
        let add_count = result.iter().filter(|o| o.opcode == OpCode::IntAdd).count();
        assert_eq!(add_count, 1, "CSE should eliminate duplicate INT_ADD");
        // Jump should still be present.
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }
}
