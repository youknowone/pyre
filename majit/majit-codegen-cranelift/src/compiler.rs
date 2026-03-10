/// Cranelift-based JIT code generation backend.
///
/// Translates majit IR traces into native code via Cranelift, then
/// executes them as ordinary function pointers.

use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use cranelift_codegen::ir::Value as CValue;

use majit_codegen::{AsmInfo, BackendError, DeadFrame, LoopToken};
use majit_ir::{FailDescr, GcRef, InputArg, Op, OpCode, OpRef, Type, Value};

use crate::guard::{CraneliftFailDescr, FrameData};

// ---------------------------------------------------------------------------
// Helpers (free functions to avoid borrow conflicts)
// ---------------------------------------------------------------------------

fn var(idx: u32) -> Variable {
    Variable::from_u32(idx)
}

fn resolve_opref(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    opref: OpRef,
) -> CValue {
    if let Some(&c) = constants.get(&opref.0) {
        return builder.ins().iconst(cl_types::I64, c);
    }
    builder.use_var(var(opref.0))
}

fn resolve_binop(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    op: &Op,
) -> (CValue, CValue) {
    let a = resolve_opref(builder, constants, op.arg(0));
    let b = resolve_opref(builder, constants, op.arg(1));
    (a, b)
}

fn emit_icmp(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    cc: IntCC,
    op: &Op,
    vi: u32,
) {
    let (a, b) = resolve_binop(builder, constants, op);
    let cmp = builder.ins().icmp(cc, a, b);
    let r = builder.ins().uextend(cl_types::I64, cmp);
    builder.def_var(var(vi), r);
}

fn op_var_index(op: &Op, op_idx: usize, num_inputs: usize) -> usize {
    if op.pos.is_none() {
        num_inputs + op_idx
    } else {
        op.pos.0 as usize
    }
}

// ---------------------------------------------------------------------------
// Compiled loop data
// ---------------------------------------------------------------------------

struct CompiledLoop {
    _func_id: FuncId,
    code_ptr: *const u8,
    code_size: usize,
    fail_descrs: Vec<Arc<CraneliftFailDescr>>,
    num_inputs: usize,
    max_output_slots: usize,
}

unsafe impl Send for CompiledLoop {}

struct GuardInfo {
    fail_index: u32,
    fail_arg_refs: Vec<OpRef>,
}

// ---------------------------------------------------------------------------
// CraneliftBackend
// ---------------------------------------------------------------------------

pub struct CraneliftBackend {
    module: JITModule,
    func_ctx: FunctionBuilderContext,
    constants: HashMap<u32, i64>,
    func_counter: u32,
}

impl CraneliftBackend {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();

        let isa_builder = cranelift_native::builder().expect("host ISA not supported");
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(jit_builder);
        let func_ctx = FunctionBuilderContext::new();

        CraneliftBackend {
            module,
            func_ctx,
            constants: HashMap::new(),
            func_counter: 0,
        }
    }

    /// Register constants available during the next `compile_loop` call.
    pub fn set_constants(&mut self, constants: HashMap<u32, i64>) {
        self.constants = constants;
    }

    fn do_compile(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
    ) -> Result<CompiledLoop, BackendError> {
        let ptr_type = self.module.target_config().pointer_type();
        let call_conv = self.module.target_config().default_call_conv;

        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(ptr_type));
        sig.params.push(AbiParam::new(ptr_type));
        sig.returns.push(AbiParam::new(cl_types::I64));

        let func_name = format!("trace_{}", self.func_counter);
        self.func_counter += 1;

        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| BackendError::CompilationFailed(e.to_string()))?;

        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
            sig,
        );

        // Pre-scan
        let mut fail_descrs: Vec<Arc<CraneliftFailDescr>> = Vec::new();
        let mut guard_infos: Vec<GuardInfo> = Vec::new();
        let mut max_output_slots: usize = 0;
        collect_guards(ops, inputargs, &mut fail_descrs, &mut guard_infos, &mut max_output_slots);

        let num_inputs = inputargs.len();

        // Take constants out of self to avoid borrow conflicts with func_ctx
        let constants = std::mem::take(&mut self.constants);

        let mut builder = FunctionBuilder::new(&mut func, &mut self.func_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let inputs_ptr = builder.block_params(entry_block)[0];
        let outputs_ptr = builder.block_params(entry_block)[1];

        // Declare variables
        for i in 0..num_inputs {
            builder.declare_var(var(i as u32), cl_types::I64);
        }
        for (op_idx, op) in ops.iter().enumerate() {
            if op.result_type() != Type::Void {
                let vi = op_var_index(op, op_idx, num_inputs);
                builder.declare_var(var(vi as u32), cl_types::I64);
            }
        }

        // Load inputs
        for i in 0..num_inputs {
            let offset = (i as i32) * 8;
            let addr = builder.ins().iadd_imm(inputs_ptr, offset as i64);
            let val = builder.ins().load(cl_types::I64, MemFlags::trusted(), addr, 0);
            builder.def_var(var(i as u32), val);
        }

        // Find LABEL
        let label_idx = ops.iter().position(|op| op.opcode == OpCode::Label);

        // Loop header block
        let loop_block = builder.create_block();
        for _ in 0..num_inputs {
            builder.append_block_param(loop_block, cl_types::I64);
        }

        // Jump entry -> loop
        {
            let vals: Vec<CValue> = (0..num_inputs)
                .map(|i| builder.use_var(var(i as u32)))
                .collect();
            builder.ins().jump(loop_block, &vals);
        }

        builder.switch_to_block(loop_block);
        for i in 0..num_inputs {
            let param = builder.block_params(loop_block)[i];
            builder.def_var(var(i as u32), param);
        }

        // Emit body
        let body_start = label_idx.map_or(0, |i| i + 1);
        let mut guard_idx: usize = 0;

        for op_idx in body_start..ops.len() {
            let op = &ops[op_idx];
            let vi = op_var_index(op, op_idx, num_inputs) as u32;

            match op.opcode {
                OpCode::IntAdd => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().iadd(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntSub => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().isub(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntMul => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().imul(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntAnd => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().band(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntOr => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().bor(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntXor => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().bxor(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntLshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().ishl(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntRshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().sshr(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::UintRshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().ushr(a, b);
                    builder.def_var(var(vi), r);
                }

                OpCode::IntNeg => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let r = builder.ins().ineg(a);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntInvert => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let r = builder.ins().bnot(a);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntIsZero => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::Equal, a, zero);
                    let r = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntIsTrue => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::NotEqual, a, zero);
                    let r = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntForceGeZero => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, zero);
                    let r = builder.ins().select(cmp, zero, a);
                    builder.def_var(var(vi), r);
                }

                OpCode::IntLt  => emit_icmp(&mut builder, &constants, IntCC::SignedLessThan, op, vi),
                OpCode::IntLe  => emit_icmp(&mut builder, &constants, IntCC::SignedLessThanOrEqual, op, vi),
                OpCode::IntEq  => emit_icmp(&mut builder, &constants, IntCC::Equal, op, vi),
                OpCode::IntNe  => emit_icmp(&mut builder, &constants, IntCC::NotEqual, op, vi),
                OpCode::IntGt  => emit_icmp(&mut builder, &constants, IntCC::SignedGreaterThan, op, vi),
                OpCode::IntGe  => emit_icmp(&mut builder, &constants, IntCC::SignedGreaterThanOrEqual, op, vi),
                OpCode::UintLt => emit_icmp(&mut builder, &constants, IntCC::UnsignedLessThan, op, vi),
                OpCode::UintLe => emit_icmp(&mut builder, &constants, IntCC::UnsignedLessThanOrEqual, op, vi),
                OpCode::UintGt => emit_icmp(&mut builder, &constants, IntCC::UnsignedGreaterThan, op, vi),
                OpCode::UintGe => emit_icmp(&mut builder, &constants, IntCC::UnsignedGreaterThanOrEqual, op, vi),

                OpCode::SameAsI => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), a);
                }

                OpCode::GuardTrue | OpCode::GuardFalse => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);

                    if op.opcode == OpCode::GuardTrue {
                        builder.ins().brif(is_zero, exit_block, &[], cont_block, &[]);
                    } else {
                        builder.ins().brif(is_zero, cont_block, &[], exit_block, &[]);
                    }

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);

                    for (slot, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
                        let val = resolve_opref(&mut builder, &constants, arg_ref);
                        let offset = (slot as i32) * 8;
                        let addr = builder.ins().iadd_imm(outputs_ptr, offset as i64);
                        builder.ins().store(MemFlags::trusted(), val, addr, 0);
                    }
                    let idx_val = builder.ins().iconst(cl_types::I64, info.fail_index as i64);
                    builder.ins().return_(&[idx_val]);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::Jump => {
                    let vals: Vec<CValue> = op
                        .args
                        .iter()
                        .map(|&r| resolve_opref(&mut builder, &constants, r))
                        .collect();
                    builder.ins().jump(loop_block, &vals);
                }

                OpCode::Finish => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    for (slot, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
                        let val = resolve_opref(&mut builder, &constants, arg_ref);
                        let offset = (slot as i32) * 8;
                        let addr = builder.ins().iadd_imm(outputs_ptr, offset as i64);
                        builder.ins().store(MemFlags::trusted(), val, addr, 0);
                    }
                    let idx_val = builder.ins().iconst(cl_types::I64, info.fail_index as i64);
                    builder.ins().return_(&[idx_val]);
                }

                OpCode::Label => {}

                OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let fr = match op.opcode {
                        OpCode::FloatAdd => builder.ins().fadd(fa, fb),
                        OpCode::FloatSub => builder.ins().fsub(fa, fb),
                        OpCode::FloatMul => builder.ins().fmul(fa, fb),
                        OpCode::FloatTrueDiv => builder.ins().fdiv(fa, fb),
                        _ => unreachable!(),
                    };
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatNeg => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fr = builder.ins().fneg(fa);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatAbs => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fr = builder.ins().fabs(fa);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }

                OpCode::CastFloatToInt => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let r = builder.ins().fcvt_to_sint(cl_types::I64, fa);
                    builder.def_var(var(vi), r);
                }
                OpCode::CastIntToFloat => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fr = builder.ins().fcvt_from_sint(cl_types::F64, a);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }

                other => {
                    if other.result_type() != Type::Void {
                        return Err(BackendError::Unsupported(format!(
                            "opcode {:?} not yet implemented",
                            other
                        )));
                    }
                }
            }
        }

        builder.seal_block(loop_block);
        builder.finalize();

        // Compile
        let mut ctx = Context::for_function(func);
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| BackendError::CompilationFailed(e.to_string()))?;
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().unwrap();

        let code_ptr = self.module.get_finalized_function(func_id);

        Ok(CompiledLoop {
            _func_id: func_id,
            code_ptr,
            code_size: 0,
            fail_descrs,
            num_inputs: inputargs.len(),
            max_output_slots,
        })
    }
}

fn collect_guards(
    ops: &[Op],
    inputargs: &[InputArg],
    fail_descrs: &mut Vec<Arc<CraneliftFailDescr>>,
    guard_infos: &mut Vec<GuardInfo>,
    max_output_slots: &mut usize,
) {
    let num_inputs = inputargs.len();

    for op in ops {
        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;

        if !is_guard && !is_finish {
            continue;
        }

        let fail_index = fail_descrs.len() as u32;

        let (fail_arg_refs, fail_arg_types) = if is_finish {
            let refs: Vec<OpRef> = op.args.iter().copied().collect();
            let types: Vec<Type> = refs.iter().map(|_| Type::Int).collect();
            (refs, types)
        } else {
            let refs: Vec<OpRef> = (0..num_inputs as u32).map(OpRef).collect();
            let types: Vec<Type> = inputargs.iter().map(|ia| ia.tp).collect();
            (refs, types)
        };

        let n = fail_arg_refs.len();
        if n > *max_output_slots {
            *max_output_slots = n;
        }

        let descr = Arc::new(CraneliftFailDescr {
            fail_index,
            fail_arg_types,
        });
        fail_descrs.push(descr);
        guard_infos.push(GuardInfo { fail_index, fail_arg_refs });
    }
}

// ---------------------------------------------------------------------------
// Backend trait implementation
// ---------------------------------------------------------------------------

impl majit_codegen::Backend for CraneliftBackend {
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut LoopToken,
    ) -> Result<AsmInfo, BackendError> {
        token.inputarg_types = inputargs.iter().map(|ia| ia.tp).collect();
        let compiled = self.do_compile(inputargs, ops)?;
        let info = AsmInfo {
            code_addr: compiled.code_ptr as usize,
            code_size: compiled.code_size,
        };
        token.compiled = Some(Box::new(compiled));
        Ok(info)
    }

    fn compile_bridge(
        &mut self,
        _fail_descr: &dyn FailDescr,
        _inputargs: &[InputArg],
        _ops: &[Op],
        _original_token: &LoopToken,
    ) -> Result<AsmInfo, BackendError> {
        Err(BackendError::Unsupported("compile_bridge not yet implemented".into()))
    }

    fn execute_token(&self, token: &LoopToken, args: &[Value]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledLoop>()
            .expect("compiled data is not CompiledLoop");

        let mut inputs: Vec<i64> = Vec::with_capacity(compiled.num_inputs);
        for arg in args {
            inputs.push(match arg {
                Value::Int(v) => *v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.0 as i64,
                Value::Void => 0,
            });
        }

        let mut outputs = vec![0i64; compiled.max_output_slots.max(1)];

        let func: unsafe extern "C" fn(*const i64, *mut i64) -> i64 =
            unsafe { std::mem::transmute(compiled.code_ptr) };
        let fail_index = unsafe { func(inputs.as_ptr(), outputs.as_mut_ptr()) } as u32;

        let fail_descr = &compiled.fail_descrs[fail_index as usize];
        let mut values = Vec::with_capacity(fail_descr.fail_arg_types.len());
        for (i, tp) in fail_descr.fail_arg_types.iter().enumerate() {
            let raw = outputs[i];
            values.push(match tp {
                Type::Int => Value::Int(raw),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Void => Value::Void,
            });
        }

        DeadFrame {
            data: Box::new(FrameData {
                values,
                fail_descr: fail_descr.clone(),
            }),
        }
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        frame.data.downcast_ref::<FrameData>().expect("FrameData expected").fail_descr.as_ref()
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        frame.data.downcast_ref::<FrameData>().expect("FrameData expected").get_int(index)
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        frame.data.downcast_ref::<FrameData>().expect("FrameData expected").get_float(index)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        frame.data.downcast_ref::<FrameData>().expect("FrameData expected").get_ref(index)
    }

    fn invalidate_loop(&self, _token: &LoopToken) {
        // TODO: patch compiled code for invalidation
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use majit_codegen::Backend;
    use std::collections::HashMap;

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut o = Op::new(opcode, args);
        o.pos = OpRef(pos);
        o
    }

    #[test]
    fn test_count_to_million() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::IntLt, &[OpRef(1), OpRef(101)], 2),
            mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 1_000_000i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(0);
        let info = backend.compile_loop(&inputargs, &ops, &mut token).unwrap();
        assert!(info.code_addr != 0);

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 999_999);
    }

    #[test]
    fn test_simple_add_finish() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(1);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(40), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_sub() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(2);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(58)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntMul, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(3);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_bitwise_ops() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAnd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntOr, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::IntXor, &[OpRef(0), OpRef(1)], 4),
            mk_op(OpCode::Finish, &[OpRef(2), OpRef(3), OpRef(4)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(4);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0xFF00), Value::Int(0x0FF0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0x0F00);
        assert_eq!(backend.get_int_value(&frame, 1), 0xFFF0);
        assert_eq!(backend.get_int_value(&frame, 2), 0xF0F0);
    }

    #[test]
    fn test_shift_ops() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntLshift, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntRshift, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::UintRshift, &[OpRef(0), OpRef(1)], 4),
            mk_op(OpCode::Finish, &[OpRef(2), OpRef(3), OpRef(4)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(5);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(-16), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), -64);
        assert_eq!(backend.get_int_value(&frame, 1), -4);
        let expected_ushr = ((-16i64 as u64) >> 2) as i64;
        assert_eq!(backend.get_int_value(&frame, 2), expected_ushr);
    }

    #[test]
    fn test_comparisons() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntLe, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::IntEq, &[OpRef(0), OpRef(1)], 4),
            mk_op(OpCode::IntNe, &[OpRef(0), OpRef(1)], 5),
            mk_op(OpCode::IntGt, &[OpRef(0), OpRef(1)], 6),
            mk_op(OpCode::IntGe, &[OpRef(0), OpRef(1)], 7),
            mk_op(OpCode::Finish, &[OpRef(2), OpRef(3), OpRef(4), OpRef(5), OpRef(6), OpRef(7)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(6);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1); // LT
        assert_eq!(backend.get_int_value(&frame, 1), 1); // LE
        assert_eq!(backend.get_int_value(&frame, 2), 0); // EQ
        assert_eq!(backend.get_int_value(&frame, 3), 1); // NE
        assert_eq!(backend.get_int_value(&frame, 4), 0); // GT
        assert_eq!(backend.get_int_value(&frame, 5), 0); // GE
    }

    #[test]
    fn test_guard_false() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntEq, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::GuardFalse, &[OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(101)], 2),
            mk_op(OpCode::Jump, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0i64);
        constants.insert(101, 1i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(7);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(10)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    #[test]
    fn test_fail_descr() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(8);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_sum_loop() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(1), OpRef(0)], 2),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(100)], 3),
            mk_op(OpCode::IntGt, &[OpRef(3), OpRef(101)], 4),
            mk_op(OpCode::GuardTrue, &[OpRef(4)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(3), OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 0i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(9);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Start: counter=100, acc=0
        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(0)]);
        // Guard fails when counter=1, so saved i0=1, i1=5049
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 5049);
    }

    #[test]
    fn test_multi_output_finish() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntMul, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::Finish, &[OpRef(0), OpRef(1), OpRef(2), OpRef(3)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(10);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 3);
        assert_eq!(backend.get_int_value(&frame, 1), 7);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(backend.get_int_value(&frame, 3), 21);
    }
}
