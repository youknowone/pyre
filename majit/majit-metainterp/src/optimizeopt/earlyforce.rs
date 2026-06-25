/// earlyforce.py: OptEarlyForce — force virtual args before escaping.
///
/// RPython earlyforce.py forces ALL arguments of most operations to ensure
/// virtual objects are materialized before they can escape. Exempt ops:
/// SETFIELD_GC, SETARRAYITEM_GC, SETARRAYITEM_RAW, QUASIIMMUT_FIELD,
/// SAME_AS_*, raw_free calls.
///
/// earlyforce.py:32: self.optimizer.optearlyforce = self
/// The pass registers itself so force_at_the_end_of_preamble can route
/// forced operations starting from earlyforce.next (= heap).
use majit_ir::operand::Operand;
use majit_ir::{Op, OpCode};

use crate::optimizeopt::info::PtrInfoExt;
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

pub struct OptEarlyForce;

impl OptEarlyForce {
    pub fn new() -> Self {
        OptEarlyForce
    }

    /// earlyforce.py:7-11: is_raw_free check.
    /// Raw free calls should not force their arguments.
    fn is_raw_free(op: &Op) -> bool {
        if !op.opcode.is_call() {
            return false;
        }
        let __descr_arc_descr = op.getdescr();
        if let Some(ref descr) = __descr_arc_descr.as_ref() {
            if let Some(cd) = descr.as_call_descr() {
                let ei = cd.get_extra_info();
                return ei.oopspecindex == majit_ir::OopSpecIndex::RawFree;
            }
        }
        false
    }

    /// earlyforce.py:15-29: should we force args for this op?
    /// RPython exempt set: SETFIELD_GC, SETARRAYITEM_GC, SETARRAYITEM_RAW,
    /// QUASIIMMUT_FIELD, SAME_AS_I/R/F, and raw_free. Note that
    /// SETFIELD_RAW is NOT exempt in RPython.
    pub(crate) fn should_force_args(op: &Op) -> bool {
        !matches!(
            op.opcode,
            OpCode::SetfieldGc
                | OpCode::SetarrayitemGc
                | OpCode::SetarrayitemRaw
                | OpCode::QuasiimmutField
                | OpCode::SameAsI
                | OpCode::SameAsR
                | OpCode::SameAsF
        ) && !Self::is_raw_free(op)
    }
}

impl Default for OptEarlyForce {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptEarlyForce {
    /// earlyforce.py:15-29: propagate_forward.
    /// Force all virtual args of non-exempt operations, then emit.
    fn propagate_forward(
        &mut self,
        op: &Op,
        _op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        if Self::should_force_args(op) {
            // earlyforce.py:28: self.optimizer.force_box(arg, self)
            // In Rust, we can't call Optimizer.force_box (borrow conflict).
            // Instead, force directly through PtrInfo.force_box_impl,
            // which uses ctx.current_pass_idx (== earlyforce_idx) for
            // emit_extra routing. This matches RPython's optforce=self.
            for i in 0..op.num_args() {
                let arg_box = ctx.resolve_operand_box_opt(&op.arg(i));
                let arg = arg_box
                    .as_ref()
                    .map(|b| b.to_opref())
                    .unwrap_or_else(|| op.arg(i).to_opref());
                // optimizer.py:354-362: force_box pops the arg's
                // potential_extra_op and hands it to the short-preamble
                // producer BEFORE (and independent of) the virtual check, so a
                // non-virtual short-box arg is added to the preamble too.
                // `take_potential_extra_op` is a no-op when not unrolling or
                // when no extra op is queued for this arg.
                if let Some(tracked) = ctx.take_potential_extra_op(arg) {
                    // shortpreamble.py:434: the resolved Box is handed
                    // to the builder; fall back to the operand's own box.
                    let arg_b = arg_box.clone().unwrap_or_else(|| op.arg(i).to_boxref());
                    if let Some(builder) = ctx.active_short_preamble_producer_mut() {
                        builder.add_preamble_op_from_pop(&tracked, arg_b);
                    } else if let Some(builder) = ctx.imported_short_preamble_builder.as_mut() {
                        builder.add_preamble_op_from_pop(&tracked, arg_b);
                    }
                }
                // optimizer.py:363-366: if the arg carries a virtual PtrInfo,
                // force it into the trace.
                let arg_is_virtual = arg_box
                    .as_ref()
                    .map_or(false, |b| ctx.is_virtual(&Operand::from_boxref(b)));
                if arg_is_virtual {
                    // `arg_box` is the box-native resolution of `op.arg(i)`
                    // (resolve_box_box_opt), already a chain terminal, and
                    // `arg_is_virtual` is only set when it is `Some` and virtual,
                    // so re-walking its OpRef would return the same info-host.
                    let arg_box = arg_box.expect("arg_is_virtual implies a resolved box");
                    let mut info = ctx.take_ptr_info(&Operand::from_boxref(&arg_box)).unwrap();
                    let _forced = info.force_box(arg_box, ctx);
                }
            }
        }
        // earlyforce.py:29: return self.emit(op)
        OptimizationResult::PassOn
    }

    fn name(&self) -> &'static str {
        "earlyforce"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::test_support::rooted_inputarg_operand;
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::OpRef;
    use majit_ir::Type;

    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            let pos = i as u32;
            op.pos.set(OpRef::op_typed(pos, op.result_type()));
        }
    }

    #[test]
    fn test_earlyforce_resolves_call_may_force_args() {
        let mut ops = vec![Op::new(
            OpCode::CallMayForceN,
            &[
                rooted_inputarg_operand(Type::Ref, 100),
                rooted_inputarg_operand(Type::Ref, 101),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecMap::new(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallMayForceN);
    }

    #[test]
    fn test_earlyforce_passthrough_non_call() {
        let mut ops = vec![Op::new(
            OpCode::IntAdd,
            &[
                rooted_inputarg_operand(Type::Int, 100),
                rooted_inputarg_operand(Type::Int, 101),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecMap::new(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_earlyforce_call_assembler_handled() {
        let mut ops = vec![Op::new(
            OpCode::CallAssemblerI,
            &[rooted_inputarg_operand(Type::Ref, 100)],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecMap::new(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallAssemblerI);
    }

    #[test]
    fn test_earlyforce_guard_not_forced() {
        // GUARD_NOT_FORCED should have its fail_args resolved.
        let mut guard = Op::new(OpCode::GuardNotForced, &[]);
        guard.setfailargs(Default::default());
        let mut ops = vec![guard];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecMap::new(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNotForced);
        assert!(result[0].has_failargs());
    }

    #[test]
    fn test_earlyforce_all_call_may_force_types() {
        for opcode in [
            OpCode::CallMayForceI,
            OpCode::CallMayForceR,
            OpCode::CallMayForceF,
            OpCode::CallMayForceN,
        ] {
            let mut ops = vec![Op::new(opcode, &[rooted_inputarg_operand(Type::Ref, 100)])];
            assign_positions(&mut ops);

            let mut opt = Optimizer::new();
            opt.add_pass(Box::new(OptEarlyForce::new()));
            let result =
                opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecMap::new(), 1024);
            assert_eq!(result.len(), 1, "{opcode:?} should be handled");
        }
    }

    #[test]
    fn test_earlyforce_exempt_setfield() {
        // SETFIELD_GC should NOT force args (earlyforce.py:18)
        let mut ops = vec![Op::new(
            OpCode::SetfieldGc,
            &[
                rooted_inputarg_operand(Type::Ref, 100),
                rooted_inputarg_operand(Type::Int, 101),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptEarlyForce::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecMap::new(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
    }
}
