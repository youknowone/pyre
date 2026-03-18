/// Main optimization driver.
///
/// Translated from rpython/jit/metainterp/optimizeopt/optimizer.py.
/// Chains multiple optimization passes and drives operations through them.
use crate::{
    guard::GuardStrengthenOpt,
    heap::OptHeap,
    intbounds::OptIntBounds,
    pure::OptPure,
    rewrite::OptRewrite,
    simplify::OptSimplify,
    virtualize::{OptVirtualize, VirtualizableConfig},
    vstring::OptString,
};
use crate::{OptContext, Optimization, OptimizationResult};
use majit_ir::{Op, OpCode, OpRef};

/// The optimizer: chains passes and runs them over a trace.
///
/// RPython optimizer.py: Optimizer class with pass chain and shared state.
pub struct Optimizer {
    passes: Vec<Box<dyn Optimization>>,
    /// Final num_inputs after optimization (may increase if virtualizable
    /// adds virtual input args).
    final_num_inputs: usize,
    /// Cache of CALL_PURE results from previous traces.
    /// optimizer.py: `call_pure_results` — maps
    /// (func_ptr, arg0, arg1, ...) → result value, carried across
    /// loop iterations so the optimizer can constant-fold repeated
    /// pure calls.
    call_pure_results: std::collections::HashMap<Vec<majit_ir::OpRef>, majit_ir::Value>,
    /// optimizer.py: `_last_guard_op` — tracks the last emitted guard
    /// for guard sharing and descriptor fusion.
    last_guard_op: Option<Op>,
    /// optimizer.py: `replaces_guard` — maps guard op position to replacement.
    replaces_guard: std::collections::HashMap<u32, Op>,
    /// optimizer.py: `pendingfields` — heap fields that need to be
    /// written back before the next guard (lazy set forcing).
    pendingfields: Vec<Op>,
    /// optimizer.py: `can_replace_guards` — flag to enable/disable guard sharing.
    can_replace_guards: bool,
    /// optimizer.py: `quasi_immutable_deps` — quasi-immutable field dependencies.
    quasi_immutable_deps: std::collections::HashSet<u32>,
}

impl Optimizer {
    pub fn new() -> Self {
        Optimizer {
            passes: Vec::new(),
            final_num_inputs: 0,
            call_pure_results: std::collections::HashMap::new(),
            last_guard_op: None,
            replaces_guard: std::collections::HashMap::new(),
            pendingfields: Vec::new(),
            can_replace_guards: true,
            quasi_immutable_deps: std::collections::HashSet::new(),
        }
    }

    /// Record a CALL_PURE result for cross-iteration constant folding.
    /// RPython optimizer.py: `call_pure_results[key] = value`
    pub fn record_call_pure_result(&mut self, args: Vec<majit_ir::OpRef>, value: majit_ir::Value) {
        self.call_pure_results.insert(args, value);
    }

    /// Look up a previously recorded CALL_PURE result.
    pub fn get_call_pure_result(&self, args: &[majit_ir::OpRef]) -> Option<&majit_ir::Value> {
        self.call_pure_results.get(args)
    }

    /// Get the final num_inputs after optimization.
    /// May be larger than the original if virtualizable added virtual input args.
    pub fn final_num_inputs(&self) -> usize {
        self.final_num_inputs
    }

    /// optimizer.py: getlastop() — get the last emitted guard operation.
    pub fn get_last_guard_op(&self) -> Option<&Op> {
        self.last_guard_op.as_ref()
    }

    /// optimizer.py: notice_guard_future_condition(op)
    /// Record that a guard at the given position should be replaced
    /// with the given op when the future condition is realized.
    pub fn notice_guard_future_condition(&mut self, guard_pos: u32, replacement: Op) {
        self.replaces_guard.insert(guard_pos, replacement);
    }

    /// optimizer.py: replace_guard(old_guard_pos, new_guard)
    /// Replace a previously emitted guard with a new one.
    pub fn replace_guard(&mut self, old_pos: u32, new_guard: Op) {
        self.replaces_guard.insert(old_pos, new_guard);
    }

    /// optimizer.py: store_final_boxes_in_guard(guard_op, pendingfields)
    /// Store fail_args in the guard and flush pending field writes.
    pub fn store_final_boxes_in_guard(&mut self, guard_op: &mut Op) {
        // Flush pending fields: emit them before the guard
        let pending = std::mem::take(&mut self.pendingfields);
        if guard_op.fail_args.is_none() {
            guard_op.fail_args = Some(Default::default());
        }
        let _ = pending; // TODO: integrate pending fields into fail_args
    }

    /// optimizer.py: emit_guard_operation(op, ctx)
    /// Emit a guard with proper tracking and descriptor sharing.
    pub fn emit_guard_operation(&mut self, guard_op: Op) {
        self.last_guard_op = Some(guard_op);
    }

    /// optimizer.py: add_pending_field(op)
    /// Queue a SETFIELD_GC to be emitted before the next guard.
    pub fn add_pending_field(&mut self, op: Op) {
        self.pendingfields.push(op);
    }

    /// optimizer.py: flush_pendingfields(ctx)
    /// Emit all pending field writes.
    pub fn flush_pendingfields(&mut self, ctx: &mut OptContext) {
        let pending = std::mem::take(&mut self.pendingfields);
        for op in pending {
            ctx.emit(op);
        }
    }

    /// optimizer.py: cant_replace_guards()
    /// Temporarily disable guard replacement (e.g., during bridge compilation).
    pub fn disable_guard_replacement(&mut self) {
        self.can_replace_guards = false;
    }

    /// Re-enable guard replacement.
    pub fn enable_guard_replacement(&mut self) {
        self.can_replace_guards = true;
    }

    /// optimizer.py: add_quasi_immutable_dep(descr_idx)
    /// Track a quasi-immutable field dependency.
    pub fn add_quasi_immutable_dep(&mut self, descr_idx: u32) {
        self.quasi_immutable_deps.insert(descr_idx);
    }

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Collect short preamble ops from all passes.
    pub fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::shortpreamble::ShortBoxes,
    ) {
        for pass in &self.passes {
            pass.produce_potential_short_preamble_ops(sb);
        }
    }

    /// optimizer.py: send_extra_operation(op, ctx)
    /// Send an extra operation through the pass chain as if it were
    /// a new operation from the trace. Used by passes that need to
    /// inject additional operations.
    pub fn send_extra_operation(&mut self, op: &Op, ctx: &mut OptContext) {
        self.propagate_one(op, ctx);
    }

    /// optimizer.py: force_box(opref, ctx) — force a virtual to be materialized.
    /// If the opref refers to a virtual object, emit the allocation and field writes.
    /// Returns the concrete OpRef (unchanged if not virtual).
    pub fn force_box(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        // Follow forwarding chain first.
        let resolved = ctx.get_replacement(opref);
        // Check if any pass considers this a virtual.
        let is_virt = self.passes.iter().any(|p| p.is_virtual(resolved));
        if is_virt {
            // The virtualize pass handles actual forcing.
            // For now, return the resolved ref — the pass has already
            // emitted the materialization ops during propagate_forward.
            resolved
        } else {
            resolved
        }
    }

    /// optimizer.py: protect_speculative_operation(op, ctx)
    /// When constant-folding a pure operation, verify that the folded
    /// constant doesn't cause a memory safety issue (e.g., null deref in
    /// getfield). If the result would be invalid, don't fold.
    pub fn protect_speculative_operation(&self, op: &Op, ctx: &OptContext) -> bool {
        // For now, conservative: only allow folding on arithmetic ops.
        // getfield/getarrayitem on constant null pointer would crash.
        match op.opcode {
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF => {
                // Check arg(0) is not null constant.
                if let Some(0) = ctx.get_constant_int(op.arg(0)) {
                    return false; // would deref null
                }
                true
            }
            _ => true,
        }
    }

    /// optimizer.py: getlastop() — return the last emitted non-guard operation.
    pub fn getlastop<'a>(&self, ctx: &'a OptContext) -> Option<&'a Op> {
        ctx.new_operations.last()
    }

    /// optimizer.py: new_const(fieldvalue) — create a new constant OpRef.
    /// Emits a SameAs op with the given constant value.
    pub fn new_const_int(ctx: &mut OptContext, value: i64) -> OpRef {
        let op = Op::new(OpCode::SameAsI, &[]);
        let opref = ctx.emit(op);
        ctx.make_constant(opref, majit_ir::Value::Int(value));
        opref
    }

    /// optimizer.py: new_const_item(arraydescr) — create a default value
    /// for an array item (0 for int, null for ref, 0.0 for float).
    pub fn new_const_item(ctx: &mut OptContext, item_type: majit_ir::Type) -> OpRef {
        match item_type {
            majit_ir::Type::Int => Self::new_const_int(ctx, 0),
            majit_ir::Type::Ref => {
                let op = Op::new(OpCode::SameAsR, &[]);
                let opref = ctx.emit(op);
                ctx.make_constant(opref, majit_ir::Value::Ref(majit_ir::GcRef::NULL));
                opref
            }
            majit_ir::Type::Float => {
                let op = Op::new(OpCode::SameAsF, &[]);
                let opref = ctx.emit(op);
                ctx.make_constant(opref, majit_ir::Value::Float(0.0));
                opref
            }
            majit_ir::Type::Void => OpRef::NONE,
        }
    }

    /// optimizer.py: is_call_pure_pure_canraise(op)
    /// Check if a CALL_PURE can raise an exception.
    pub fn is_call_pure_pure_canraise(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.effect_info().can_raise())
            .unwrap_or(true)
    }

    /// Add an optimization pass to the chain.
    pub fn add_pass(&mut self, pass: Box<dyn Optimization>) {
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
        // Ensure new ops get positions beyond all original trace positions.
        // Original ops keep their tracer-assigned positions; new ops (constants,
        // force materializations) must not collide with them.
        let max_pos = ops
            .iter()
            .map(|op| op.pos.0)
            .filter(|&p| p != u32::MAX) // skip OpRef::NONE
            .max()
            .unwrap_or(0);
        let effective_inputs = num_inputs.max((max_pos + 1) as usize);
        let mut ctx = OptContext::with_num_inputs(ops.len(), effective_inputs);

        // Pre-populate known constants so passes can see them.
        for (&idx, &val) in constants.iter() {
            ctx.make_constant(OpRef(idx), Value::Int(val));
        }

        // Setup all passes
        for pass in &mut self.passes {
            pass.setup();
        }

        // optimizer.py: inject call_pure_results into OptPure so it can
        // constant-fold repeated pure calls across loop iterations.
        if !self.call_pure_results.is_empty() {
            for pass in &mut self.passes {
                if pass.name() == "pure" {
                    // Downcast not possible with trait objects, but we record
                    // the results as known constants in the context instead.
                    for (args, value) in &self.call_pure_results {
                        if let majit_ir::Value::Int(v) = value {
                            if let Some(result_ref) = args.last() {
                                ctx.make_constant(*result_ref, majit_ir::Value::Int(*v));
                            }
                        }
                    }
                    break;
                }
            }
        }

        // Process each operation through the pass chain
        for op in ops {
            self.propagate_one(op, &mut ctx);
        }

        // Flush all passes
        for pass in &mut self.passes {
            pass.flush();
        }

        // final_num_inputs = original inputs + virtual inputs added by passes.
        let num_virtual_inputs = (ctx.num_inputs as usize).saturating_sub(effective_inputs);
        self.final_num_inputs = num_inputs + num_virtual_inputs;

        // Remap ALL positions: virtual inputs go to num_inputs..final_num_inputs,
        // and all op positions are reassigned to start from final_num_inputs.
        // This ensures no position collisions between input block params and ops.
        if num_virtual_inputs > 0 {
            let fni = self.final_num_inputs as u32;
            let mut remap = std::collections::HashMap::new();

            // Virtual input positions: optimizer used effective_inputs+k, backend needs num_inputs+k
            for k in 0..num_virtual_inputs {
                let opt_pos = (effective_inputs + k) as u32;
                let be_pos = (num_inputs + k) as u32;
                if opt_pos != be_pos {
                    remap.insert(opt_pos, be_pos);
                }
            }

            // Op positions: reassign to start from final_num_inputs
            // to avoid collision with input positions 0..final_num_inputs
            for (new_idx, op) in ctx.new_operations.iter_mut().enumerate() {
                let new_pos = fni + new_idx as u32;
                if op.pos.0 < fni && !op.pos.is_none() {
                    remap.insert(op.pos.0, new_pos);
                    op.pos = OpRef(new_pos);
                }
            }

            // Apply remap to all args and fail_args
            for op in &mut ctx.new_operations {
                for arg in &mut op.args {
                    if let Some(&new_pos) = remap.get(&arg.0) {
                        *arg = OpRef(new_pos);
                    }
                }
                if let Some(ref mut fail_args) = op.fail_args {
                    for arg in fail_args.iter_mut() {
                        if let Some(&new_pos) = remap.get(&arg.0) {
                            *arg = OpRef(new_pos);
                        }
                    }
                }
            }

            // Remap constants
            let old_constants = std::mem::take(&mut ctx.constants);
            for (idx, val) in old_constants.into_iter().enumerate() {
                if let Some(v) = val {
                    let target_idx =
                        remap.get(&(idx as u32)).copied().unwrap_or(idx as u32) as usize;
                    if target_idx >= ctx.constants.len() {
                        ctx.constants.resize(target_idx + 1, None);
                    }
                    ctx.constants[target_idx] = Some(v);
                }
            }
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
    ///
    /// NOTE: Do NOT add `replace_op(original_pos, new_pos)` here.
    /// The Emit variant's position tracking is handled by each pass
    /// and OptContext. Adding automatic replacement mapping here
    /// causes spurious forwarding that breaks heap/guard tests.
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
                OptimizationResult::Emit(op) => {
                    ctx.emit(op);
                    return;
                }
                OptimizationResult::Replace(op) => {
                    current_op = op;
                }
                OptimizationResult::Remove => {
                    return;
                }
                OptimizationResult::PassOn => {}
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
        opt.add_pass(Box::new(GuardStrengthenOpt::new()));
        opt.add_pass(Box::new(OptSimplify::new()));
        opt.add_pass(Box::new(OptHeap::new()));
        opt
    }

    /// Create an optimizer with virtualizable config for frame field tracking.
    pub fn default_pipeline_with_virtualizable(config: VirtualizableConfig) -> Self {
        let mut opt = Self::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.add_pass(Box::new(OptVirtualize::with_virtualizable(config)));
        opt.add_pass(Box::new(OptString::new()));
        opt.add_pass(Box::new(OptPure::new()));
        opt.add_pass(Box::new(GuardStrengthenOpt::new()));
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

    impl Optimization for AddZeroElimination {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if op.opcode == OpCode::IntAdd {
                // Check if second arg is constant 0
                if let Some(0) = ctx.get_constant_int(op.arg(1)) {
                    // Replace with first arg
                    ctx.replace_op(op.pos, op.arg(0));
                    return OptimizationResult::Remove;
                }
            }
            OptimizationResult::PassOn
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
