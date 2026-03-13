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
use pyre_object::floatobject::FLOAT_FLOATVAL_OFFSET;
use pyre_object::intobject::INT_INTVAL_OFFSET;
use pyre_object::pyobject::{
    DICT_TYPE, FLOAT_TYPE, INT_TYPE, LIST_TYPE, OB_TYPE_OFFSET, PyType, STR_TYPE, TUPLE_TYPE,
    is_float, is_int,
};
use pyre_object::rangeobject::{RANGE_ITER_TYPE, w_range_iter_next};
use pyre_object::strobject::is_str;
use pyre_objspace::py_is_true;
use pyre_runtime::builtinfunc::is_builtin_func;
use pyre_runtime::funcobject::{FUNC_CODE_OFFSET, FUNCTION_TYPE, is_func, w_func_get_code_ptr};

use crate::descr::{
    make_field_descr, range_iter_current_descr, range_iter_step_descr, range_iter_stop_descr,
};
use crate::frame_layout::PYFRAME_LOCALS_OFFSET;
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

/// Field descriptor for `W_FloatObject.floatval` (offset 8, 8 bytes, Float).
fn floatval_descr() -> DescrRef {
    make_field_descr(FLOAT_FLOATVAL_OFFSET, 8, Type::Float, false)
}

fn frame_locals_array(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.vable_getfield_ref(frame, PYFRAME_LOCALS_OFFSET)
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
    let num_live = 1 + sym.locals.len() + sym.ns_values.len() + sym.stack.len();

    // Precompute type constants and field descriptors
    let int_type_ptr = &INT_TYPE as *const PyType as usize as i64;
    let float_type_ptr = &FLOAT_TYPE as *const PyType as usize as i64;
    let str_type_ptr = &STR_TYPE as *const PyType as usize as i64;
    let _list_type_ptr = &LIST_TYPE as *const PyType as usize as i64;
    let _tuple_type_ptr = &TUPLE_TYPE as *const PyType as usize as i64;
    let _dict_type_ptr = &DICT_TYPE as *const PyType as usize as i64;
    let func_type_ptr = &FUNCTION_TYPE as *const PyType as usize as i64;
    let ob_type_fd = ob_type_descr();
    let intval_fd = intval_descr();
    let func_code_fd = func_code_descr();
    let floatval_fd = floatval_descr();

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
                pyre_bytecode::bytecode::ConstantData::Float { value } => {
                    let bits = value.to_bits() as i64;
                    let const_val = ctx.const_int(bits);
                    let result = ctx.call_ref(helpers::jit_w_float_new as *const (), &[const_val]);
                    sym.stack.push(result);
                }
                pyre_bytecode::bytecode::ConstantData::Boolean { value } => {
                    let v: i64 = if *value { 1 } else { 0 };
                    let const_val = ctx.const_int(v);
                    let result = ctx.call_ref(helpers::jit_w_bool_from as *const (), &[const_val]);
                    sym.stack.push(result);
                }
                pyre_bytecode::bytecode::ConstantData::Str { value } => {
                    // Strings are opaque constants in the JIT. Allocate the str
                    // object at trace time and embed its pointer as a constant.
                    let str_obj =
                        pyre_object::w_str_new(value.as_str().expect("non-UTF-8 string constant"));
                    let ptr_const = ctx.const_int(str_obj as i64);
                    sym.stack.push(ptr_const);
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

        Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
            let idx = var_num.get(op_arg).as_usize();
            if idx < sym.locals.len() {
                let idx_const = ctx.const_int(idx as i64);
                let locals_array = frame_locals_array(ctx, sym.frame);
                let value = ctx.vable_getarrayitem_ref(locals_array, idx_const);
                sym.locals[idx] = value;
                sym.stack.push(value);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::LoadFastBorrowLoadFastBorrow { var_nums } => {
            let pair = var_nums.get(op_arg);
            let idx1 = (pair >> 4) as usize;
            let idx2 = (pair & 15) as usize;
            if idx1 < sym.locals.len() && idx2 < sym.locals.len() {
                let locals_array = frame_locals_array(ctx, sym.frame);
                let idx1_const = ctx.const_int(idx1 as i64);
                let idx2_const = ctx.const_int(idx2 as i64);
                let v1 = ctx.vable_getarrayitem_ref(locals_array, idx1_const);
                let v2 = ctx.vable_getarrayitem_ref(locals_array, idx2_const);
                sym.locals[idx1] = v1;
                sym.locals[idx2] = v2;
                sym.stack.push(v1);
                sym.stack.push(v2);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::LoadFastCheck { var_num } => {
            let idx = var_num.get(op_arg).as_usize();
            if idx < sym.locals.len() {
                let idx_const = ctx.const_int(idx as i64);
                let locals_array = frame_locals_array(ctx, sym.frame);
                let value = ctx.vable_getarrayitem_ref(locals_array, idx_const);
                sym.locals[idx] = value;
                sym.stack.push(value);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::LoadFastLoadFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            let idx1 = (pair >> 4) as usize;
            let idx2 = (pair & 15) as usize;
            if idx1 < sym.locals.len() && idx2 < sym.locals.len() {
                let locals_array = frame_locals_array(ctx, sym.frame);
                let idx1_const = ctx.const_int(idx1 as i64);
                let idx2_const = ctx.const_int(idx2 as i64);
                let v1 = ctx.vable_getarrayitem_ref(locals_array, idx1_const);
                let v2 = ctx.vable_getarrayitem_ref(locals_array, idx2_const);
                sym.locals[idx1] = v1;
                sym.locals[idx2] = v2;
                sym.stack.push(v1);
                sym.stack.push(v2);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::StoreFastLoadFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            let store_idx = pair.store_idx() as usize;
            let load_idx = pair.load_idx() as usize;
            let Some(val) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            if store_idx < sym.locals.len() && load_idx < sym.locals.len() {
                let locals_array = frame_locals_array(ctx, sym.frame);
                let store_idx_const = ctx.const_int(store_idx as i64);
                ctx.vable_setarrayitem(locals_array, store_idx_const, val);
                sym.locals[store_idx] = val;
                let load_idx_const = ctx.const_int(load_idx as i64);
                let loaded = ctx.vable_getarrayitem_ref(locals_array, load_idx_const);
                sym.locals[load_idx] = loaded;
                sym.stack.push(loaded);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::StoreFastStoreFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            let idx1 = (pair >> 4) as usize;
            let idx2 = (pair & 15) as usize;
            let Some(v1) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            let Some(v2) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            if idx1 < sym.locals.len() && idx2 < sym.locals.len() {
                let locals_array = frame_locals_array(ctx, sym.frame);
                let idx1_const = ctx.const_int(idx1 as i64);
                let idx2_const = ctx.const_int(idx2 as i64);
                ctx.vable_setarrayitem(locals_array, idx1_const, v1);
                ctx.vable_setarrayitem(locals_array, idx2_const, v2);
                sym.locals[idx1] = v1;
                sym.locals[idx2] = v2;
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
                let locals_array = frame_locals_array(ctx, sym.frame);
                let idx_const = ctx.const_int(idx as i64);
                ctx.vable_setarrayitem(locals_array, idx_const, val);
                sym.locals[idx] = val;
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::LoadName { namei } => {
            let idx = namei.get(op_arg) as usize;
            let name: &str = code.names[idx].as_ref();
            if let Some(ns_idx) = sym.ns_keys.iter().position(|k| k == name) {
                sym.stack.push(sym.ns_values[ns_idx]);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::LoadGlobal { namei } => {
            let raw = namei.get(op_arg) as usize;
            let name_idx = raw >> 1;
            let push_null = (raw & 1) != 0;
            let name: &str = code.names[name_idx].as_ref();
            if let Some(ns_idx) = sym.ns_keys.iter().position(|k| k == name) {
                sym.stack.push(sym.ns_values[ns_idx]);
            } else {
                return TraceAction::Abort;
            }
            if push_null {
                let null_const = ctx.const_int(0);
                sym.stack.push(null_const);
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

            // Subscr (a[b]) is handled as a residual call
            if matches!(op, BinaryOperator::Subscr) {
                // Dispatch through the generic getitem helper
                let result = ctx.call_ref(helpers::jit_getitem as *const (), &[a, b]);
                sym.stack.push(result);
                // Skip the rest of the BinaryOp handling
                if ctx.is_too_long() {
                    return TraceAction::Abort;
                }
                return TraceAction::Continue;
            }

            // Determine concrete types to decide which guard path to emit.
            // The concrete stack positions after popping two values:
            //   a was at concrete_stack[sym.stack.len()]
            //   b was at concrete_stack[sym.stack.len() + 1]
            let concrete_a = concrete_stack[sym.stack.len()];
            let concrete_b = concrete_stack[sym.stack.len() + 1];
            let a_is_float = unsafe { is_float(concrete_a) };
            let b_is_float = unsafe { is_float(concrete_b) };

            // TrueDivide always produces float
            let is_truediv = matches!(
                op,
                BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide
            );

            if a_is_float || b_is_float || is_truediv {
                // Float path: guard both operands as their observed types,
                // extract values, convert ints to floats, perform float op
                let va = extract_as_float(
                    ctx,
                    a,
                    a_is_float,
                    &ob_type_fd,
                    int_type_ptr,
                    float_type_ptr,
                    &intval_fd,
                    &floatval_fd,
                    num_live,
                );
                let vb = extract_as_float(
                    ctx,
                    b,
                    b_is_float,
                    &ob_type_fd,
                    int_type_ptr,
                    float_type_ptr,
                    &intval_fd,
                    &floatval_fd,
                    num_live,
                );

                let result_val = match op {
                    BinaryOperator::Add | BinaryOperator::InplaceAdd => {
                        ctx.record_op(OpCode::FloatAdd, &[va, vb])
                    }
                    BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => {
                        ctx.record_op(OpCode::FloatSub, &[va, vb])
                    }
                    BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => {
                        ctx.record_op(OpCode::FloatMul, &[va, vb])
                    }
                    BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => {
                        ctx.record_op(OpCode::FloatTrueDiv, &[va, vb])
                    }
                    BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => {
                        ctx.record_op(OpCode::FloatFloorDiv, &[va, vb])
                    }
                    BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => {
                        ctx.record_op(OpCode::FloatMod, &[va, vb])
                    }
                    _ => return TraceAction::Abort,
                };

                // Convert float result to i64 bit pattern and box as W_FloatObject
                let bits = ctx.record_op(OpCode::ConvertFloatBytesToLonglong, &[result_val]);
                let result = ctx.call_ref(helpers::jit_w_float_new as *const (), &[bits]);
                sym.stack.push(result);
            } else if unsafe { is_str(concrete_a) } {
                // Str path: all ops go through residual calls
                guard_str(ctx, a, &ob_type_fd, str_type_ptr, num_live);
                match op {
                    BinaryOperator::Add | BinaryOperator::InplaceAdd => {
                        guard_str(ctx, b, &ob_type_fd, str_type_ptr, num_live);
                        let result = ctx.call_ref(helpers::jit_str_concat as *const (), &[a, b]);
                        sym.stack.push(result);
                    }
                    BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => {
                        guard_int(ctx, b, &ob_type_fd, int_type_ptr, num_live);
                        let vb =
                            ctx.record_op_with_descr(OpCode::GetfieldGcI, &[b], intval_fd.clone());
                        let result = ctx.call_ref(helpers::jit_str_repeat as *const (), &[a, vb]);
                        sym.stack.push(result);
                    }
                    _ => return TraceAction::Abort,
                }
            } else if unsafe { is_int(concrete_a) && is_int(concrete_b) } {
                // Int path — both operands are W_IntObject
                guard_int(ctx, a, &ob_type_fd, int_type_ptr, num_live);
                guard_int(ctx, b, &ob_type_fd, int_type_ptr, num_live);

                let va = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[a], intval_fd.clone());
                let vb = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[b], intval_fd.clone());

                let result_val = match op {
                    BinaryOperator::Add | BinaryOperator::InplaceAdd => {
                        ctx.int_add_ovf(va, vb, num_live)
                    }
                    BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => {
                        ctx.int_sub_ovf(va, vb, num_live)
                    }
                    BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => {
                        ctx.int_mul_ovf(va, vb, num_live)
                    }
                    BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => {
                        ctx.record_op(OpCode::IntFloorDiv, &[va, vb])
                    }
                    BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => {
                        ctx.record_op(OpCode::IntMod, &[va, vb])
                    }
                    _ => return TraceAction::Abort,
                };

                let result = ctx.call_ref(helpers::jit_w_int_new as *const (), &[result_val]);
                sym.stack.push(result);
            } else {
                // Unsupported type combination (e.g. W_LongObject) — abort trace
                return TraceAction::Abort;
            }
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

            let concrete_a = concrete_stack[sym.stack.len()];
            let concrete_b = concrete_stack[sym.stack.len() + 1];
            let a_is_float = unsafe { is_float(concrete_a) };
            let b_is_float = unsafe { is_float(concrete_b) };

            if a_is_float || b_is_float {
                // Float comparison path
                let va = extract_as_float(
                    ctx,
                    a,
                    a_is_float,
                    &ob_type_fd,
                    int_type_ptr,
                    float_type_ptr,
                    &intval_fd,
                    &floatval_fd,
                    num_live,
                );
                let vb = extract_as_float(
                    ctx,
                    b,
                    b_is_float,
                    &ob_type_fd,
                    int_type_ptr,
                    float_type_ptr,
                    &intval_fd,
                    &floatval_fd,
                    num_live,
                );

                let cmp_result = match op {
                    ComparisonOperator::Less => ctx.record_op(OpCode::FloatLt, &[va, vb]),
                    ComparisonOperator::LessOrEqual => ctx.record_op(OpCode::FloatLe, &[va, vb]),
                    ComparisonOperator::Greater => ctx.record_op(OpCode::FloatGt, &[va, vb]),
                    ComparisonOperator::GreaterOrEqual => ctx.record_op(OpCode::FloatGe, &[va, vb]),
                    ComparisonOperator::Equal => ctx.record_op(OpCode::FloatEq, &[va, vb]),
                    ComparisonOperator::NotEqual => ctx.record_op(OpCode::FloatNe, &[va, vb]),
                };

                let result = ctx.call_ref(helpers::jit_w_bool_from as *const (), &[cmp_result]);
                sym.stack.push(result);
            } else if unsafe { is_str(concrete_a) && is_str(concrete_b) } {
                // Str comparison path: residual call to jit_str_compare
                guard_str(ctx, a, &ob_type_fd, str_type_ptr, num_live);
                guard_str(ctx, b, &ob_type_fd, str_type_ptr, num_live);

                // jit_str_compare returns -1/0/1
                let cmp_ord = ctx.call_int(helpers::jit_str_compare as *const (), &[a, b]);

                // Convert ordering to boolean based on comparison operator
                let zero = ctx.const_int(0);
                let cmp_result = match op {
                    ComparisonOperator::Less => ctx.record_op(OpCode::IntLt, &[cmp_ord, zero]),
                    ComparisonOperator::LessOrEqual => {
                        ctx.record_op(OpCode::IntLe, &[cmp_ord, zero])
                    }
                    ComparisonOperator::Greater => ctx.record_op(OpCode::IntGt, &[cmp_ord, zero]),
                    ComparisonOperator::GreaterOrEqual => {
                        ctx.record_op(OpCode::IntGe, &[cmp_ord, zero])
                    }
                    ComparisonOperator::Equal => ctx.record_op(OpCode::IntEq, &[cmp_ord, zero]),
                    ComparisonOperator::NotEqual => ctx.record_op(OpCode::IntNe, &[cmp_ord, zero]),
                };

                let result = ctx.call_ref(helpers::jit_w_bool_from as *const (), &[cmp_result]);
                sym.stack.push(result);
            } else if unsafe { is_int(concrete_a) && is_int(concrete_b) } {
                // Int comparison path
                guard_int(ctx, a, &ob_type_fd, int_type_ptr, num_live);
                guard_int(ctx, b, &ob_type_fd, int_type_ptr, num_live);

                let va = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[a], intval_fd.clone());
                let vb = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[b], intval_fd.clone());

                let cmp_result = match op {
                    ComparisonOperator::Less => ctx.record_op(OpCode::IntLt, &[va, vb]),
                    ComparisonOperator::LessOrEqual => ctx.record_op(OpCode::IntLe, &[va, vb]),
                    ComparisonOperator::Greater => ctx.record_op(OpCode::IntGt, &[va, vb]),
                    ComparisonOperator::GreaterOrEqual => ctx.record_op(OpCode::IntGe, &[va, vb]),
                    ComparisonOperator::Equal => ctx.record_op(OpCode::IntEq, &[va, vb]),
                    ComparisonOperator::NotEqual => ctx.record_op(OpCode::IntNe, &[va, vb]),
                };

                let result = ctx.call_ref(helpers::jit_w_bool_from as *const (), &[cmp_result]);
                sym.stack.push(result);
            } else {
                // Unsupported type combination — abort trace
                return TraceAction::Abort;
            }
        }

        // ── Unary operations ──────────────────────────────────────
        Instruction::UnaryNegative => {
            let Some(a) = sym.stack.pop() else {
                return TraceAction::Abort;
            };

            let concrete_a = concrete_stack[sym.stack.len()];
            if unsafe { is_float(concrete_a) } {
                guard_float(ctx, a, &ob_type_fd, float_type_ptr, num_live);
                let va = ctx.record_op_with_descr(OpCode::GetfieldGcF, &[a], floatval_fd.clone());
                let neg = ctx.record_op(OpCode::FloatNeg, &[va]);
                let bits = ctx.record_op(OpCode::ConvertFloatBytesToLonglong, &[neg]);
                let result = ctx.call_ref(helpers::jit_w_float_new as *const (), &[bits]);
                sym.stack.push(result);
            } else if unsafe { is_int(concrete_a) } {
                guard_int(ctx, a, &ob_type_fd, int_type_ptr, num_live);
                let va = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[a], intval_fd.clone());
                let zero = ctx.const_int(0);
                let neg = ctx.int_sub_ovf(zero, va, num_live);
                let result = ctx.call_ref(helpers::jit_w_int_new as *const (), &[neg]);
                sym.stack.push(result);
            } else {
                return TraceAction::Abort;
            }
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

        // ── Iteration ────────────────────────────────────────────
        Instruction::GetIter => {
            // The iterator (range_iter) is already on the symbolic stack.
            // Guard that it is indeed a range iterator.
            if let Some(&iter_sym) = sym.stack.last() {
                let range_iter_type_ptr = &RANGE_ITER_TYPE as *const PyType as usize as i64;
                guard_range_iter(ctx, iter_sym, &ob_type_fd, range_iter_type_ptr, num_live);
            } else {
                return TraceAction::Abort;
            }
        }

        Instruction::ForIter { .. } => {
            // The iterator is on top of the symbolic stack (not popped).
            let Some(&iter_sym) = sym.stack.last() else {
                return TraceAction::Abort;
            };

            // Read current and stop fields from the range iterator
            let current = ctx.record_op_with_descr(
                OpCode::GetfieldGcI,
                &[iter_sym],
                range_iter_current_descr(),
            );
            let stop =
                ctx.record_op_with_descr(OpCode::GetfieldGcI, &[iter_sym], range_iter_stop_descr());

            // Determine concrete branch direction: did the loop continue?
            let concrete_iter = concrete_stack[sym.stack.len() - 1];
            let loop_continues = unsafe { w_range_iter_next(concrete_iter).is_some() };

            if loop_continues {
                // Loop continues: guard current < stop (for positive step)
                let not_exhausted = ctx.record_op(OpCode::IntLt, &[current, stop]);
                ctx.record_guard(OpCode::GuardTrue, &[not_exhausted], num_live);

                // Read step and compute next_val = current + step
                let step = ctx.record_op_with_descr(
                    OpCode::GetfieldGcI,
                    &[iter_sym],
                    range_iter_step_descr(),
                );
                let next_val = ctx.record_op(OpCode::IntAdd, &[current, step]);

                // Update iterator's current field
                ctx.record_op_with_descr(
                    OpCode::SetfieldGc,
                    &[iter_sym, next_val],
                    range_iter_current_descr(),
                );

                // Box the current value as a Python int and push to symbolic stack
                let boxed = ctx.call_ref(helpers::jit_w_int_new as *const (), &[current]);
                sym.stack.push(boxed);
            } else {
                // Loop exhausted: guard current >= stop
                let exhausted = ctx.record_op(OpCode::IntGe, &[current, stop]);
                ctx.record_guard(OpCode::GuardTrue, &[exhausted], num_live);

                // Pop the iterator from symbolic stack
                sym.stack.pop();
            }
        }

        Instruction::EndFor => {
            // End-of-for marker, no-op in trace.
        }

        Instruction::PopIter => {
            // Pop the iterator from the symbolic stack.
            sym.stack.pop();
        }

        // ── Return ────────────────────────────────────────────────
        Instruction::ReturnValue => {
            // Can't trace across returns in Phase 1
            return TraceAction::Abort;
        }

        // ── Container construction ───────────────────────────────
        Instruction::BuildList { count } => {
            let size = count.get(op_arg) as usize;
            if sym.stack.len() < size {
                return TraceAction::Abort;
            }
            let start = sym.stack.len() - size;
            let items: Vec<OpRef> = sym.stack.drain(start..).collect();
            let list = ctx.call_ref(helpers::jit_build_list_0 as *const (), &[]);
            for item in items {
                let _ = ctx.call_int(helpers::jit_list_append as *const (), &[list, item]);
            }
            sym.stack.push(list);
        }

        Instruction::BuildTuple { count } => {
            let size = count.get(op_arg) as usize;
            if sym.stack.len() < size {
                return TraceAction::Abort;
            }
            let start = sym.stack.len() - size;
            let items: Vec<OpRef> = sym.stack.drain(start..).collect();
            // Phase 1: build tuples as lists in JIT (both opaque)
            let list = ctx.call_ref(helpers::jit_build_list_0 as *const (), &[]);
            for item in items {
                let _ = ctx.call_int(helpers::jit_list_append as *const (), &[list, item]);
            }
            sym.stack.push(list);
        }

        Instruction::BuildMap { count } => {
            let size = count.get(op_arg) as usize;
            let pairs_needed = size * 2;
            if sym.stack.len() < pairs_needed {
                return TraceAction::Abort;
            }
            sym.stack.truncate(sym.stack.len() - pairs_needed);
            return TraceAction::Abort;
        }

        Instruction::StoreSubscr => {
            let Some(key) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            let Some(obj) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            let Some(value) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            let _ = ctx.call_int(helpers::jit_setitem as *const (), &[obj, key, value]);
        }

        Instruction::ListAppend { i } => {
            let Some(value) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            let depth = i.get(op_arg) as usize;
            if sym.stack.len() < depth {
                return TraceAction::Abort;
            }
            let list = sym.stack[sym.stack.len() - depth];
            ctx.call_void(helpers::jit_list_append as *const (), &[list, value]);
        }

        Instruction::UnpackSequence { count } => {
            let size = count.get(op_arg) as usize;
            let Some(seq) = sym.stack.pop() else {
                return TraceAction::Abort;
            };
            for idx in (0..size).rev() {
                let idx_const = ctx.const_int(idx as i64);
                let item = ctx.call_ref(helpers::jit_tuple_getitem as *const (), &[seq, idx_const]);
                sym.stack.push(item);
            }
        }

        // ── Function creation ─────────────────────────────────────
        Instruction::MakeFunction => {
            // Functions are typically defined outside hot loops.
            // Abort tracing when we encounter one.
            return TraceAction::Abort;
        }

        // ── Function calls (residual call pattern) ───────────────
        Instruction::Call { argc } => {
            let nargs = argc.get(op_arg) as usize;
            if sym.stack.len() < nargs + 2 {
                return TraceAction::Abort;
            }
            // Pop arguments
            let start = sym.stack.len() - nargs;
            let args: Vec<OpRef> = sym.stack.drain(start..).collect();
            // Pop null_or_self and callable
            let _null_or_self = sym.stack.pop().unwrap();
            let callable = sym.stack.pop().unwrap();

            // Determine the callable type from the concrete stack
            let concrete_callable = concrete_stack[sym.stack.len()];

            if unsafe { is_builtin_func(concrete_callable) } {
                // Builtin functions: emit a residual call
                let result = ctx.call_may_force_ref(
                    helpers::jit_call_builtin as *const (),
                    &std::iter::once(callable)
                        .chain(args.iter().copied())
                        .collect::<Vec<_>>(),
                );
                ctx.guard_not_forced(num_live);
                sym.stack.push(result);
            } else if unsafe { is_func(concrete_callable) } {
                // User-defined function: guard type, promote code_ptr, residual call
                guard_func(ctx, callable, &ob_type_fd, func_type_ptr, num_live);

                // Read the code_ptr field from the function object
                let code_ref = ctx.record_op_with_descr(
                    OpCode::GetfieldGcR,
                    &[callable],
                    func_code_fd.clone(),
                );

                // Promote code_ptr to specialize on the exact function
                let concrete_code_ptr = unsafe { w_func_get_code_ptr(concrete_callable) };
                ctx.promote_ref(code_ref, concrete_code_ptr as i64, num_live);

                // Build argv for the helper call: [callable, argc, arg0, arg1, ...]
                let argc_const = ctx.const_int(nargs as i64);
                let mut call_args = vec![callable, argc_const];
                call_args.extend_from_slice(&args);

                let result =
                    ctx.call_may_force_ref(helpers::jit_call_function as *const (), &call_args);
                ctx.guard_not_forced(num_live);
                sym.stack.push(result);
            } else {
                return TraceAction::Abort;
            }
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

/// Emit a type guard asserting that `obj` has `ob_type == &FLOAT_TYPE`.
fn guard_float(
    ctx: &mut TraceCtx,
    obj: OpRef,
    ob_type_fd: &DescrRef,
    float_type_ptr: i64,
    num_live: usize,
) {
    let actual_type = ctx.record_op_with_descr(OpCode::GetfieldGcR, &[obj], ob_type_fd.clone());
    let expected = ctx.const_int(float_type_ptr);
    ctx.record_guard(OpCode::GuardClass, &[actual_type, expected], num_live);
}

/// Emit a type guard asserting that `obj` has `ob_type == &STR_TYPE`.
fn guard_str(
    ctx: &mut TraceCtx,
    obj: OpRef,
    ob_type_fd: &DescrRef,
    str_type_ptr: i64,
    num_live: usize,
) {
    let actual_type = ctx.record_op_with_descr(OpCode::GetfieldGcR, &[obj], ob_type_fd.clone());
    let expected = ctx.const_int(str_type_ptr);
    ctx.record_guard(OpCode::GuardClass, &[actual_type, expected], num_live);
}

/// Guard an operand and extract its value as a float OpRef.
///
/// If the operand is a float, guard as float and read floatval.
/// If the operand is an int, guard as int, read intval, and cast to float.
fn extract_as_float(
    ctx: &mut TraceCtx,
    obj: OpRef,
    obj_is_float: bool,
    ob_type_fd: &DescrRef,
    int_type_ptr: i64,
    float_type_ptr: i64,
    intval_fd: &DescrRef,
    floatval_fd: &DescrRef,
    num_live: usize,
) -> OpRef {
    if obj_is_float {
        guard_float(ctx, obj, ob_type_fd, float_type_ptr, num_live);
        ctx.record_op_with_descr(OpCode::GetfieldGcF, &[obj], floatval_fd.clone())
    } else {
        guard_int(ctx, obj, ob_type_fd, int_type_ptr, num_live);
        let int_val = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[obj], intval_fd.clone());
        ctx.record_op(OpCode::CastIntToFloat, &[int_val])
    }
}

/// Emit a type guard asserting that `obj` has `ob_type == &RANGE_ITER_TYPE`.
fn guard_range_iter(
    ctx: &mut TraceCtx,
    obj: OpRef,
    ob_type_fd: &DescrRef,
    range_iter_type_ptr: i64,
    num_live: usize,
) {
    let actual_type = ctx.record_op_with_descr(OpCode::GetfieldGcR, &[obj], ob_type_fd.clone());
    let expected = ctx.const_int(range_iter_type_ptr);
    ctx.record_guard(OpCode::GuardClass, &[actual_type, expected], num_live);
}

/// Emit a type guard asserting that `obj` has `ob_type == &FUNCTION_TYPE`.
fn guard_func(
    ctx: &mut TraceCtx,
    obj: OpRef,
    ob_type_fd: &DescrRef,
    func_type_ptr: i64,
    num_live: usize,
) {
    let actual_type = ctx.record_op_with_descr(OpCode::GetfieldGcR, &[obj], ob_type_fd.clone());
    let expected = ctx.const_int(func_type_ptr);
    ctx.record_guard(OpCode::GuardClass, &[actual_type, expected], num_live);
}

/// Field descriptor for `W_FunctionObject.code_ptr` (offset 8, 8 bytes, Ref).
fn func_code_descr() -> DescrRef {
    make_field_descr(FUNC_CODE_OFFSET, 8, Type::Ref, false)
}
