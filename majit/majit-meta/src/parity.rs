use std::collections::HashMap;

use majit_ir::{Op, OpRef, Type};
use majit_trace::trace::Trace;

/// A small, stable parity case format for comparing majit traces against
/// RPython-derived expectations.
///
/// The key rule is that proc-macro expansion details do not matter. Only the
/// normalized runtime seam matters: the recorded IR operations, their argument
/// order, and any explicit guard fail-args.
#[derive(Clone, Debug)]
pub struct TraceParityCase<'a> {
    pub name: &'a str,
    pub rpython_reference: &'a str,
    pub expected_lines: &'a [&'a str],
}

fn render_arg(arg: OpRef, constants: &HashMap<u32, i64>) -> String {
    if let Some(value) = constants.get(&arg.0) {
        value.to_string()
    } else {
        format!("v{}", arg.0)
    }
}

fn render_op(op: &Op, constants: &HashMap<u32, i64>) -> String {
    let args = op
        .args
        .iter()
        .map(|&arg| render_arg(arg, constants))
        .collect::<Vec<_>>()
        .join(", ");

    let mut line = if op.opcode.is_guard() || op.opcode.result_type() == Type::Void {
        format!("{:?}({args})", op.opcode)
    } else {
        format!("v{} = {:?}({args})", op.pos.0, op.opcode)
    };

    if let Some(fail_args) = &op.fail_args {
        let fail_args = fail_args
            .iter()
            .map(|&arg| render_arg(arg, constants))
            .collect::<Vec<_>>()
            .join(", ");
        line.push_str(&format!(" [fail_args={fail_args}]"));
    }

    line
}

/// Normalize a completed trace into stable line strings for parity checks.
pub fn normalize_trace(trace: &Trace, constants: &HashMap<u32, i64>) -> Vec<String> {
    trace
        .ops
        .iter()
        .map(|op| render_op(op, constants))
        .collect()
}

/// Normalize an op slice into the same stable line format used by
/// [`normalize_trace`].
pub fn normalize_ops(ops: &[Op], constants: &HashMap<u32, i64>) -> Vec<String> {
    ops.iter().map(|op| render_op(op, constants)).collect()
}

/// Assert that a trace matches a normalized parity case.
pub fn assert_trace_parity(
    trace: &Trace,
    constants: &HashMap<u32, i64>,
    case: &TraceParityCase<'_>,
) {
    let actual = normalize_trace(trace, constants);
    let expected = case
        .expected_lines
        .iter()
        .map(|line| (*line).to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        actual, expected,
        "trace parity mismatch for {} (RPython reference: {})",
        case.name, case.rpython_reference
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_fail_descr, SymbolicStack, TraceAction, TraceCtx};
    use majit_ir::{OpCode, Type};
    use majit_trace::recorder::TraceRecorder;

    fn finish_trace_ctx(mut ctx: TraceCtx, finish_args: &[OpRef]) -> (Trace, HashMap<u32, i64>) {
        ctx.recorder
            .finish(finish_args, make_fail_descr(finish_args.len()));
        let trace = ctx.recorder.get_trace();
        let constants = ctx.constants.into_inner();
        (trace, constants)
    }

    #[test]
    fn trace_ctx_binop_matches_parity_case() {
        let mut recorder = TraceRecorder::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let i1 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let mut stack = SymbolicStack::new();
        stack.push(i0);
        stack.push(i1);

        ctx.trace_binop(&mut stack, OpCode::IntAdd);
        let result = stack.pop().unwrap();
        let (trace, constants) = finish_trace_ctx(ctx, &[result]);

        let case = TraceParityCase {
            name: "trace_ctx_binop_int_add",
            rpython_reference: "rpython/jit/metainterp/pyjitpl.py: opimpl_int_add",
            expected_lines: &["v2 = IntAdd(v0, v1)", "Finish(v2)"],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn trace_ctx_constants_match_parity_case() {
        let mut recorder = TraceRecorder::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let one = ctx.const_int(1);
        let two = ctx.const_int(2);
        let sum = ctx.record_op(OpCode::IntAdd, &[i0, one]);
        let prod = ctx.record_op(OpCode::IntMul, &[sum, two]);
        let (trace, constants) = finish_trace_ctx(ctx, &[prod]);

        let case = TraceParityCase {
            name: "trace_ctx_constant_arith",
            rpython_reference:
                "rpython/jit/metainterp/pyjitpl.py: special_int_add / opimpl_int_mul",
            expected_lines: &["v1 = IntAdd(v0, 1)", "v2 = IntMul(v1, 2)", "Finish(v2)"],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn guard_fail_args_are_part_of_normalized_trace() {
        let mut recorder = TraceRecorder::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let zero = ctx.const_int(0);
        let cond = ctx.record_op(OpCode::IntGt, &[i0, zero]);
        ctx.record_guard_with_fail_args(OpCode::GuardTrue, &[cond], 1, &[i0]);
        let (trace, constants) = finish_trace_ctx(ctx, &[i0]);

        let case = TraceParityCase {
            name: "guard_fail_args_surface_in_seam",
            rpython_reference: "rpython/jit/metainterp/pyjitpl.py: implement_guard_value",
            expected_lines: &[
                "v1 = IntGt(v0, 0)",
                "GuardTrue(v1) [fail_args=v0]",
                "Finish(v0)",
            ],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn unary_bitwise_and_comparison_ops_normalize_stably() {
        let mut recorder = TraceRecorder::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let i1 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let neg = ctx.record_op(OpCode::IntNeg, &[i0]);
        let anded = ctx.record_op(OpCode::IntAnd, &[neg, i1]);
        let seven = ctx.const_int(7);
        let zero = ctx.const_int(0);
        let xored = ctx.record_op(OpCode::IntXor, &[anded, seven]);
        let cond = ctx.record_op(OpCode::IntGe, &[xored, zero]);
        ctx.record_guard_with_fail_args(OpCode::GuardTrue, &[cond], 2, &[i0, i1]);
        let (trace, constants) = finish_trace_ctx(ctx, &[xored]);

        let case = TraceParityCase {
            name: "unary_bitwise_and_comparison_ops",
            rpython_reference:
                "rpython/jit/metainterp/pyjitpl.py: opimpl_int_and / opimpl_int_xor / opimpl_int_ge",
            expected_lines: &[
                "v2 = IntNeg(v0)",
                "v3 = IntAnd(v2, v1)",
                "v4 = IntXor(v3, 7)",
                "v5 = IntGe(v4, 0)",
                "GuardTrue(v5) [fail_args=v0, v1]",
                "Finish(v4)",
            ],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn div_and_mod_ops_match_expected_trace_shape() {
        let mut recorder = TraceRecorder::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let three = ctx.const_int(3);
        let div = ctx.record_op(OpCode::IntFloorDiv, &[i0, three]);
        let rem = ctx.record_op(OpCode::IntMod, &[i0, three]);
        let sum = ctx.record_op(OpCode::IntAdd, &[div, rem]);
        let (trace, constants) = finish_trace_ctx(ctx, &[sum]);

        let case = TraceParityCase {
            name: "div_and_mod_ops",
            rpython_reference:
                "rpython/jit/metainterp/pyjitpl.py: int_c_div / int_c_mod tracing path",
            expected_lines: &[
                "v1 = IntFloorDiv(v0, 3)",
                "v2 = IntMod(v0, 3)",
                "v3 = IntAdd(v1, v2)",
                "Finish(v3)",
            ],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn shift_ops_match_expected_trace_shape() {
        let mut recorder = TraceRecorder::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let i1 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let one = ctx.const_int(1);
        let two = ctx.const_int(2);
        let lhs = ctx.record_op(OpCode::IntLshift, &[i0, one]);
        let rhs = ctx.record_op(OpCode::IntRshift, &[i1, two]);
        let merged = ctx.record_op(OpCode::IntOr, &[lhs, rhs]);
        let (trace, constants) = finish_trace_ctx(ctx, &[merged]);

        let case = TraceParityCase {
            name: "shift_ops",
            rpython_reference:
                "rpython/jit/metainterp/pyjitpl.py: opimpl_int_lshift / opimpl_int_rshift",
            expected_lines: &[
                "v2 = IntLshift(v0, 1)",
                "v3 = IntRshift(v1, 2)",
                "v4 = IntOr(v2, v3)",
                "Finish(v4)",
            ],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn branch_guard_taken_on_false_matches_guard_false_trace() {
        let mut recorder = TraceRecorder::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let mut stack = SymbolicStack::new();
        stack.push(i0);

        let action = ctx.trace_branch_guard(&mut stack, true, false, 0, false);
        assert!(matches!(action, TraceAction::Continue));
        let (trace, constants) = finish_trace_ctx(ctx, &[]);

        let case = TraceParityCase {
            name: "branch_guard_taken_on_false",
            rpython_reference: "rpython/jit/metainterp/pyjitpl.py: generate_guard(rop.GUARD_FALSE)",
            expected_lines: &["GuardFalse(v0)", "Finish()"],
        };
        assert_trace_parity(&trace, &constants, &case);
    }

    #[test]
    fn branch_guard_taken_on_true_can_close_loop() {
        let mut recorder = TraceRecorder::new();
        let i0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let mut stack = SymbolicStack::new();
        stack.push(i0);

        let action = ctx.trace_branch_guard(&mut stack, true, true, 0, true);
        assert!(matches!(action, TraceAction::CloseLoop));
        let (trace, constants) = finish_trace_ctx(ctx, &[]);

        let case = TraceParityCase {
            name: "branch_guard_taken_on_true_close_loop",
            rpython_reference: "rpython/jit/metainterp/pyjitpl.py: generate_guard(rop.GUARD_TRUE)",
            expected_lines: &["GuardTrue(v0)", "Finish()"],
        };
        assert_trace_parity(&trace, &constants, &case);
    }
}
