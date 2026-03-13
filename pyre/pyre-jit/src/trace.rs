//! Bytecode-level trace recording for the JIT.
//!
//! `trace_bytecode` is the closure passed to `JitDriver::merge_point()`.
//! It mirrors the interpreter's dispatch loop symbolically, recording
//! IR operations into the `TraceCtx` while maintaining `PyreSym` as the
//! mapping from interpreter values to IR `OpRef`s.

use majit_ir::{DescrRef, OpCode, OpRef, Type};
use majit_meta::{TraceAction, TraceCtx};
use pyre_bytecode::CodeObject;
use pyre_bytecode::bytecode::{BinaryOperator, ComparisonOperator, Instruction};
use pyre_object::PyObjectRef;
use pyre_object::intobject::INT_INTVAL_OFFSET;
use pyre_object::pyobject::{INT_TYPE, OB_TYPE_OFFSET, PyType};
use pyre_objspace::py_is_true;

use crate::descr::make_field_descr;
use crate::helpers;
use crate::state::PyreSym;

/// Field descriptor for `PyObject.ob_type` (offset 0, 8 bytes, Ref).
fn ob_type_descr() -> DescrRef {
    make_field_descr(OB_TYPE_OFFSET, 8, Type::Ref, false)
}

/// Field descriptor for `W_IntObject.intval` (offset 8, 8 bytes, signed Int).
fn intval_descr() -> DescrRef {
    make_field_descr(INT_INTVAL_OFFSET, 8, Type::Int, true)
}

/// Record IR operations for one bytecode instruction during tracing.
///
/// Called from `driver.merge_point()` at every iteration while the JIT
/// is actively recording a trace. Returns `TraceAction` to tell the
/// driver whether to continue, close the loop, or abort.
///
/// # Arguments
/// * `ctx` — the IR recording context
/// * `sym` — symbolic state mapping interpreter values → OpRefs
/// * `code` — the code object being traced
/// * `pc` — instruction index about to be executed
/// * `concrete_stack` — snapshot of the concrete value stack (for branch direction)
pub fn trace_bytecode(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    code: &CodeObject,
    pc: usize,
    concrete_stack: &[PyObjectRef],
) -> TraceAction {
    if pc >= code.instructions.len() {
        return TraceAction::Abort;
    }

    let code_unit = code.instructions[pc];
    let (instruction, op_arg) = sym.arg_state.get(code_unit);
    let num_live = sym.locals.len() + sym.ns_values.len() + sym.stack.len();

    // Precompute type constants and field descriptors
    let int_type_ptr = &INT_TYPE as *const PyType as usize as i64;
    let ob_type_fd = ob_type_descr();
    let intval_fd = intval_descr();

    match instruction {
        Instruction::ExtendedArg
        | Instruction::Resume { .. }
        | Instruction::Nop
        | Instruction::Cache
        | Instruction::NotTaken => {
            return TraceAction::Continue;
        }

        // ── Constants & variables ─────────────────────────────────
        Instruction::LoadConst { consti } => {
            let const_idx = consti.get(op_arg);
            match &code.constants[const_idx] {
                pyre_bytecode::bytecode::ConstantData::Integer { value } => {
                    use num_traits::ToPrimitive;
                    let v = value.to_i64().unwrap_or(0);
                    let const_val = ctx.const_int(v);
                    let result = ctx.call_ref(helpers::jit_w_int_new as *const (), &[const_val]);
                    sym.stack.push(result);
                }
                pyre_bytecode::bytecode::ConstantData::Boolean { value } => {
                    let v: i64 = if *value { 1 } else { 0 };
                    let const_val = ctx.const_int(v);
                    let result =
                        ctx.call_ref(helpers::jit_w_bool_from as *const (), &[const_val]);
                    sym.stack.push(result);
                }
                pyre_bytecode::bytecode::ConstantData::None => {
                    let result = ctx.call_ref(pyre_object::w_none as *const (), &[]);
                    sym.stack.push(result);
                }
                _ => return TraceAction::Abort,
            }
        }

        Instruction::LoadSmallInt { i } => {
            let v = i.get(op_arg) as i64;
            let const_val = ctx.const_int(v);
            let result = ctx.call_ref(helpers::jit_w_int_new as *const (), &[const_val]);
            sym.stack.push(result);
        }

        Instruction::LoadFast { var_num } => {
            let idx = var_num.get(op_arg).as_usize();
            if idx < sym.locals.len() {
                sym.stack.push(sym.locals[idx]);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::StoreFast { var_num } => {
            let idx = var_num.get(op_arg).as_usize();
            let Some(val) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            if idx < sym.locals.len() {
                sym.locals[idx] = val;
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::LoadName { namei } | Instruction::LoadGlobal { namei } => {
            let idx = namei.get(op_arg) as usize;
            let name: &str = code.names[idx].as_ref();
            if let Some(ns_idx) = sym.ns_keys.iter().position(|k| k == name) {
                sym.stack.push(sym.ns_values[ns_idx]);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::StoreName { namei } | Instruction::StoreGlobal { namei } => {
            let idx = namei.get(op_arg) as usize;
            let name: &str = code.names[idx].as_ref();
            let Some(val) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            if let Some(ns_idx) = sym.ns_keys.iter().position(|k| k == name) {
                sym.ns_values[ns_idx] = val;
            } else {
                return TraceAction::Abort;
            }
        }

        // ── Stack manipulation ────────────────────────────────────
        Instruction::PopTop => {
            sym.stack.pop();
        }

        Instruction::PushNull => {
            // PushNull is for Call protocol; push a placeholder OpRef
            let null_const = ctx.const_int(0);
            sym.stack.push(null_const);
        }

        Instruction::Copy { i } => {
            let depth = i.get(op_arg) as usize;
            if depth > 0 && sym.stack.len() >= depth {
                let val = sym.stack[sym.stack.len() - depth];
                sym.stack.push(val);
            } else {
                return TraceAction::Abort;
            }
        }

        // ── Binary operations ─────────────────────────────────────
        Instruction::BinaryOp { op } => {
            let op = op.get(op_arg);
            let Some(b) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            let Some(a) = sym.stack.pop() else {
                return TraceAction::Abort;
            };

            // Guard both operands are int
            guard_int(ctx, a, &ob_type_fd, int_type_ptr, num_live);
            guard_int(ctx, b, &ob_type_fd, int_type_ptr, num_live);

            // Extract intval fields
            let va = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[a], intval_fd.clone());
            let vb = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[b], intval_fd.clone());

            // Perform the integer operation
            let result_val = match op {
                BinaryOperator::Add | BinaryOperator::InplaceAdd => {
                    ctx.record_op(OpCode::IntAdd, &[va, vb])
                }
                BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => {
                    ctx.record_op(OpCode::IntSub, &[va, vb])
                }
                BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => {
                    ctx.record_op(OpCode::IntMul, &[va, vb])
                }
                BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => {
                    ctx.record_op(OpCode::IntFloorDiv, &[va, vb])
                }
                BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => {
                    ctx.record_op(OpCode::IntMod, &[va, vb])
                }
                _ => return TraceAction::Abort,
            };

            // Allocate new int object with the result
            let result = ctx.call_ref(helpers::jit_w_int_new as *const (), &[result_val]);
            sym.stack.push(result);
        }

        // ── Comparison operations ─────────────────────────────────
        Instruction::CompareOp { opname } => {
            let op = opname.get(op_arg);
            let Some(b) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            let Some(a) = sym.stack.pop() else {
                return TraceAction::Abort;
            };

            // Guard both operands are int
            guard_int(ctx, a, &ob_type_fd, int_type_ptr, num_live);
            guard_int(ctx, b, &ob_type_fd, int_type_ptr, num_live);

            // Extract intval fields
            let va = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[a], intval_fd.clone());
            let vb = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[b], intval_fd.clone());

            // Perform comparison → i64 result (0 or 1)
            let cmp_result = match op {
                ComparisonOperator::Less => ctx.record_op(OpCode::IntLt, &[va, vb]),
                ComparisonOperator::LessOrEqual => ctx.record_op(OpCode::IntLe, &[va, vb]),
                ComparisonOperator::Greater => ctx.record_op(OpCode::IntGt, &[va, vb]),
                ComparisonOperator::GreaterOrEqual => ctx.record_op(OpCode::IntGe, &[va, vb]),
                ComparisonOperator::Equal => ctx.record_op(OpCode::IntEq, &[va, vb]),
                ComparisonOperator::NotEqual => ctx.record_op(OpCode::IntNe, &[va, vb]),
            };

            // Create bool object from comparison result
            let result = ctx.call_ref(helpers::jit_w_bool_from as *const (), &[cmp_result]);
            sym.stack.push(result);
        }

        // ── Unary operations ──────────────────────────────────────
        Instruction::UnaryNegative => {
            let Some(a) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            guard_int(ctx, a, &ob_type_fd, int_type_ptr, num_live);
            let va = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[a], intval_fd.clone());
            let zero = ctx.const_int(0);
            let neg = ctx.record_op(OpCode::IntSub, &[zero, va]);
            let result = ctx.call_ref(helpers::jit_w_int_new as *const (), &[neg]);
            sym.stack.push(result);
        }

        Instruction::UnaryNot => {
            let Some(a) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            let truth = ctx.call_int(helpers::jit_py_is_true as *const (), &[a]);
            let one = ctx.const_int(1);
            let negated = ctx.record_op(OpCode::IntSub, &[one, truth]);
            let result = ctx.call_ref(helpers::jit_w_bool_from as *const (), &[negated]);
            sym.stack.push(result);
        }

        // ── Control flow ──────────────────────────────────────────
        Instruction::JumpForward { .. } => {
            // Forward jump → continue tracing at the target
        }

        Instruction::JumpBackward { .. } => {
            // Backward jump → close the loop
            return TraceAction::CloseLoop;
        }

        Instruction::PopJumpIfFalse { .. } => {
            let Some(top_sym) = sym.stack.pop() else {
                return TraceAction::Abort;
            };

            // Determine concrete branch direction from the stack snapshot
            let concrete_val = concrete_stack[sym.stack.len()];
            let is_true = py_is_true(concrete_val);

            // Emit truth test
            let truth = ctx.call_int(helpers::jit_py_is_true as *const (), &[top_sym]);

            if is_true {
                // Branch NOT taken → guard that condition is true (stay in loop)
                ctx.record_guard(OpCode::GuardTrue, &[truth], num_live);
            } else {
                // Branch taken (forward jump) → guard false and continue tracing
                ctx.record_guard(OpCode::GuardFalse, &[truth], num_live);
            }
        }

        Instruction::PopJumpIfTrue { .. } => {
            let Some(top_sym) = sym.stack.pop() else {
                return TraceAction::Abort;
            };

            let concrete_val = concrete_stack[sym.stack.len()];
            let is_true = py_is_true(concrete_val);

            let truth = ctx.call_int(helpers::jit_py_is_true as *const (), &[top_sym]);

            if is_true {
                // Branch taken (forward jump) → guard true and continue tracing
                ctx.record_guard(OpCode::GuardTrue, &[truth], num_live);
            } else {
                // Branch NOT taken → guard false (stay in loop)
                ctx.record_guard(OpCode::GuardFalse, &[truth], num_live);
            }
        }

        // ── Return ────────────────────────────────────────────────
        Instruction::ReturnValue => {
            // Can't trace across returns in Phase 1
            return TraceAction::Abort;
        }

        // ── Not yet traceable ─────────────────────────────────────
        _ => {
            return TraceAction::Abort;
        }
    }

    if ctx.is_too_long() {
        return TraceAction::Abort;
    }

    TraceAction::Continue
}

/// Emit a type guard asserting that `obj` has `ob_type == &INT_TYPE`.
fn guard_int(
    ctx: &mut TraceCtx,
    obj: OpRef,
    ob_type_fd: &DescrRef,
    int_type_ptr: i64,
    num_live: usize,
) {
    let actual_type = ctx.record_op_with_descr(OpCode::GetfieldGcR, &[obj], ob_type_fd.clone());
    let expected = ctx.const_int(int_type_ptr);
    ctx.record_guard(OpCode::GuardClass, &[actual_type, expected], num_live);
}
