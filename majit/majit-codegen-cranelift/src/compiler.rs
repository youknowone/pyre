/// Cranelift-based JIT code generation backend.
///
/// Translates majit IR traces into native code via Cranelift, then
/// executes them as ordinary function pointers.
use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use cranelift_codegen::ir::Value as CValue;

use majit_codegen::{AsmInfo, BackendError, DeadFrame, LoopToken};
use majit_ir::{CallDescr, FailDescr, GcRef, InputArg, Op, OpCode, OpRef, Type, Value};

use crate::guard::{BridgeData, CraneliftFailDescr, FrameData};

// ---------------------------------------------------------------------------
// Helpers (free functions to avoid borrow conflicts)
// ---------------------------------------------------------------------------

fn var(idx: u32) -> Variable {
    Variable::from_u32(idx)
}

/// Map a majit Type to the corresponding Cranelift IR type for call signatures.
fn cranelift_type_for(tp: &Type) -> cranelift_codegen::ir::Type {
    match tp {
        Type::Int | Type::Ref => cl_types::I64,
        Type::Float => cl_types::F64,
        Type::Void => cl_types::I64,
    }
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

fn emit_fcmp(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    cc: FloatCC,
    op: &Op,
    vi: u32,
) {
    let (a, b) = resolve_binop(builder, constants, op);
    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
    let cmp = builder.ins().fcmp(cc, fa, fb);
    let r = builder.ins().uextend(cl_types::I64, cmp);
    builder.def_var(var(vi), r);
}

/// Map a field size (in bytes) to the corresponding Cranelift type.
fn cl_type_for_size(size: usize) -> cranelift_codegen::ir::Type {
    match size {
        1 => cl_types::I8,
        2 => cl_types::I16,
        4 => cl_types::I32,
        8 => cl_types::I64,
        _ => cl_types::I64,
    }
}

fn op_var_index(op: &Op, op_idx: usize, num_inputs: usize) -> usize {
    if op.pos.is_none() {
        num_inputs + op_idx
    } else {
        op.pos.0 as usize
    }
}

/// Emit an indirect call through a function pointer.
///
/// `op.args[0]` is the function address (as an integer/pointer).
/// `op.args[1..]` are the call arguments.
/// `call_descr` provides `arg_types()` and `result_type()`.
///
/// Float arguments are bitcast from I64 before the call, and float
/// results are bitcast back to I64 for variable storage.
fn emit_indirect_call(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    op: &Op,
    call_descr: &dyn CallDescr,
    call_conv: cranelift_codegen::isa::CallConv,
    ptr_type: cranelift_codegen::ir::Type,
) -> Option<CValue> {
    let mut sig = Signature::new(call_conv);
    let arg_types = call_descr.arg_types();
    for at in arg_types {
        sig.params.push(AbiParam::new(cranelift_type_for(at)));
    }
    let result_type = call_descr.result_type();
    if result_type != Type::Void {
        sig.returns
            .push(AbiParam::new(cranelift_type_for(&result_type)));
    }
    let sig_ref = builder.import_signature(sig);

    let func_ptr_raw = resolve_opref(builder, constants, op.arg(0));
    let func_ptr = if ptr_type != cl_types::I64 {
        builder.ins().ireduce(ptr_type, func_ptr_raw)
    } else {
        func_ptr_raw
    };

    let mut args: Vec<CValue> = Vec::with_capacity(op.args.len() - 1);
    for (i, &arg_ref) in op.args[1..].iter().enumerate() {
        let raw = resolve_opref(builder, constants, arg_ref);
        if i < arg_types.len() && arg_types[i] == Type::Float {
            args.push(builder.ins().bitcast(cl_types::F64, MemFlags::new(), raw));
        } else {
            args.push(raw);
        }
    }

    let call = builder.ins().call_indirect(sig_ref, func_ptr, &args);

    if result_type != Type::Void {
        let result = builder.inst_results(call)[0];
        let stored = if result_type == Type::Float {
            builder
                .ins()
                .bitcast(cl_types::I64, MemFlags::new(), result)
        } else {
            result
        };
        Some(stored)
    } else {
        None
    }
}

/// Emit a guard side-exit: store fail args to outputs_ptr and return fail_index.
fn emit_guard_exit(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    outputs_ptr: CValue,
    info: &GuardInfo,
) {
    for (slot, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
        let val = resolve_opref(builder, constants, arg_ref);
        let offset = (slot as i32) * 8;
        let addr = builder.ins().iadd_imm(outputs_ptr, offset as i64);
        builder.ins().store(MemFlags::trusted(), val, addr, 0);
    }
    let idx_val = builder
        .ins()
        .iconst(cl_types::I64, info.fail_index as i64);
    builder.ins().return_(&[idx_val]);
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

    /// Decode raw i64 output slots into typed `Value`s.
    fn decode_frame_values(outputs: &[i64], types: &[Type]) -> Vec<Value> {
        let mut values = Vec::with_capacity(types.len());
        for (i, tp) in types.iter().enumerate() {
            let raw = outputs[i];
            values.push(match tp {
                Type::Int => Value::Int(raw),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Void => Value::Void,
            });
        }
        values
    }

    /// Execute a compiled bridge, returning the DeadFrame from the bridge's
    /// exit point (either a Finish or a further guard failure).
    ///
    /// If the bridge itself hits a guard that has another bridge attached,
    /// this chains through until a final exit is reached.
    fn execute_bridge(
        bridge: &BridgeData,
        parent_outputs: &[i64],
        parent_types: &[Type],
    ) -> DeadFrame {
        // The bridge's inputs are the parent guard's fail args.
        let num_bridge_inputs = bridge.num_inputs.min(parent_types.len());
        let bridge_inputs = &parent_outputs[..num_bridge_inputs];

        let mut outputs = vec![0i64; bridge.max_output_slots.max(1)];

        let func: unsafe extern "C" fn(*const i64, *mut i64) -> i64 =
            unsafe { std::mem::transmute(bridge.code_ptr) };
        let fail_index = unsafe { func(bridge_inputs.as_ptr(), outputs.as_mut_ptr()) } as u32;

        let fail_descr = &bridge.fail_descrs[fail_index as usize];
        fail_descr.increment_fail_count();

        // Check for chained bridges.
        let bridge_guard = fail_descr.bridge.lock().unwrap();
        if let Some(ref next_bridge) = *bridge_guard {
            return Self::execute_bridge(next_bridge, &outputs, &fail_descr.fail_arg_types);
        }
        drop(bridge_guard);

        let values = Self::decode_frame_values(&outputs, &fail_descr.fail_arg_types);
        DeadFrame {
            data: Box::new(FrameData {
                values,
                fail_descr: fail_descr.clone(),
            }),
        }
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
        collect_guards(
            ops,
            inputargs,
            &mut fail_descrs,
            &mut guard_infos,
            &mut max_output_slots,
        );

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

        // Declare variables for inputs
        for i in 0..num_inputs {
            builder.declare_var(var(i as u32), cl_types::I64);
        }
        // Declare variables for op results
        for (op_idx, op) in ops.iter().enumerate() {
            if op.result_type() != Type::Void {
                let vi = op_var_index(op, op_idx, num_inputs);
                builder.declare_var(var(vi as u32), cl_types::I64);
            }
        }

        // Load inputs from input buffer
        for i in 0..num_inputs {
            let offset = (i as i32) * 8;
            let addr = builder.ins().iadd_imm(inputs_ptr, offset as i64);
            let val = builder
                .ins()
                .load(cl_types::I64, MemFlags::trusted(), addr, 0);
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
        let mut last_ovf_flag: Option<CValue> = None;

        for op_idx in body_start..ops.len() {
            let op = &ops[op_idx];
            let vi = op_var_index(op, op_idx, num_inputs) as u32;

            match op.opcode {
                // ── Integer arithmetic ──
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
                OpCode::IntFloorDiv => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().sdiv(a, b);
                    builder.def_var(var(vi), r);
                }

                // ── Overflow arithmetic ──
                // Compute the result normally, then detect signed overflow
                // using the bit-manipulation formula:
                //   ovf = ((a ^ result) & (b ^ result)) >> 63   [for add]
                //   ovf = ((a ^ result) & ((a ^ b))) >> 63      [for sub]
                // For mul we use a widening approach: sign-extend to i128
                // then check if the result fits in i64.
                OpCode::IntAddOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().iadd(a, b);
                    builder.def_var(var(vi), r);
                    // ovf = ((a ^ r) & (b ^ r)) >> 63
                    let axr = builder.ins().bxor(a, r);
                    let bxr = builder.ins().bxor(b, r);
                    let both = builder.ins().band(axr, bxr);
                    let ovf = builder.ins().sshr_imm(both, 63);
                    last_ovf_flag = Some(ovf);
                }
                OpCode::IntSubOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().isub(a, b);
                    builder.def_var(var(vi), r);
                    // ovf = ((a ^ r) & (a ^ b)) >> 63
                    let axr = builder.ins().bxor(a, r);
                    let axb = builder.ins().bxor(a, b);
                    let both = builder.ins().band(axr, axb);
                    let ovf = builder.ins().sshr_imm(both, 63);
                    last_ovf_flag = Some(ovf);
                }
                OpCode::IntMulOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().imul(a, b);
                    builder.def_var(var(vi), r);
                    // Check overflow: if b != 0 && r / b != a, then overflow.
                    // We need to guard against sdiv trap when b == 0.
                    // Use a conditional: if b == 0, ovf = 0; else ovf = (r/b != a).
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let b_is_zero = builder.ins().icmp(IntCC::Equal, b, zero);

                    let no_div_block = builder.create_block();
                    let div_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, cl_types::I64);

                    builder
                        .ins()
                        .brif(b_is_zero, no_div_block, &[], div_block, &[]);

                    // b == 0 path: no overflow (result is just 0 * a = 0)
                    builder.switch_to_block(no_div_block);
                    builder.seal_block(no_div_block);
                    let no_ovf = builder.ins().iconst(cl_types::I64, 0);
                    builder.ins().jump(merge_block, &[no_ovf]);

                    // b != 0 path: check r / b != a
                    builder.switch_to_block(div_block);
                    builder.seal_block(div_block);
                    let div = builder.ins().sdiv(r, b);
                    let div_ne_a = builder.ins().icmp(IntCC::NotEqual, div, a);
                    let ovf_ext = builder.ins().uextend(cl_types::I64, div_ne_a);
                    builder.ins().jump(merge_block, &[ovf_ext]);

                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let ovf = builder.block_params(merge_block)[0];
                    last_ovf_flag = Some(ovf);
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

                // ── Unary integer ──
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

                // ── Integer comparisons ──
                OpCode::IntLt => emit_icmp(&mut builder, &constants, IntCC::SignedLessThan, op, vi),
                OpCode::IntLe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::SignedLessThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::IntEq => emit_icmp(&mut builder, &constants, IntCC::Equal, op, vi),
                OpCode::IntNe => emit_icmp(&mut builder, &constants, IntCC::NotEqual, op, vi),
                OpCode::IntGt => {
                    emit_icmp(&mut builder, &constants, IntCC::SignedGreaterThan, op, vi)
                }
                OpCode::IntGe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::SignedGreaterThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::UintLt => {
                    emit_icmp(&mut builder, &constants, IntCC::UnsignedLessThan, op, vi)
                }
                OpCode::UintLe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::UnsignedLessThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::UintGt => {
                    emit_icmp(&mut builder, &constants, IntCC::UnsignedGreaterThan, op, vi)
                }
                OpCode::UintGe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::UnsignedGreaterThanOrEqual,
                    op,
                    vi,
                ),

                // ── Pointer comparisons ──
                OpCode::PtrEq | OpCode::InstancePtrEq => {
                    emit_icmp(&mut builder, &constants, IntCC::Equal, op, vi)
                }
                OpCode::PtrNe | OpCode::InstancePtrNe => {
                    emit_icmp(&mut builder, &constants, IntCC::NotEqual, op, vi)
                }

                // ── Float comparisons ──
                OpCode::FloatLt => {
                    emit_fcmp(&mut builder, &constants, FloatCC::LessThan, op, vi)
                }
                OpCode::FloatLe => {
                    emit_fcmp(&mut builder, &constants, FloatCC::LessThanOrEqual, op, vi)
                }
                OpCode::FloatEq => {
                    emit_fcmp(&mut builder, &constants, FloatCC::Equal, op, vi)
                }
                OpCode::FloatNe => {
                    emit_fcmp(&mut builder, &constants, FloatCC::NotEqual, op, vi)
                }
                OpCode::FloatGt => {
                    emit_fcmp(&mut builder, &constants, FloatCC::GreaterThan, op, vi)
                }
                OpCode::FloatGe => {
                    emit_fcmp(&mut builder, &constants, FloatCC::GreaterThanOrEqual, op, vi)
                }

                // ── Identity / cast ──
                OpCode::SameAsI
                | OpCode::SameAsR
                | OpCode::SameAsF
                | OpCode::CastPtrToInt
                | OpCode::CastIntToPtr => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), a);
                }

                // ── Guards ──
                OpCode::GuardTrue
                | OpCode::GuardFalse
                | OpCode::GuardNonnull
                | OpCode::GuardIsnull => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);

                    let exit_on_zero =
                        matches!(op.opcode, OpCode::GuardTrue | OpCode::GuardNonnull);
                    if exit_on_zero {
                        builder
                            .ins()
                            .brif(is_zero, exit_block, &[], cont_block, &[]);
                    } else {
                        builder
                            .ins()
                            .brif(is_zero, cont_block, &[], exit_block, &[]);
                    }

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardValue => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let neq = builder.ins().icmp(IntCC::NotEqual, a, b);
                    builder
                        .ins()
                        .brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardClass | OpCode::GuardNonnullClass => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let neq = builder.ins().icmp(IntCC::NotEqual, a, b);
                    builder
                        .ins()
                        .brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // Guards with no arguments (0-arity): simplified stubs
                OpCode::GuardNoException => {
                    // When exception support is fully wired, this would
                    // load from an exception pointer and side-exit if non-null.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let _ = info;
                }

                OpCode::GuardException => {
                    // Produces a Ref result (the exception object).
                    // Simplified: produce null and don't side-exit.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let _ = info;
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(var(vi), zero);
                }

                OpCode::GuardNoOverflow => {
                    // Side-exit if overflow DID occur (ovf != 0).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let ovf = last_ovf_flag
                        .take()
                        .expect("GuardNoOverflow without preceding overflow op");
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, ovf, zero);
                    // If ovf == 0 (no overflow), continue; otherwise side-exit.
                    builder
                        .ins()
                        .brif(is_zero, cont_block, &[], exit_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }
                OpCode::GuardOverflow => {
                    // Side-exit if overflow did NOT occur (ovf == 0).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let ovf = last_ovf_flag
                        .take()
                        .expect("GuardOverflow without preceding overflow op");
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, ovf, zero);
                    // If ovf == 0 (no overflow), side-exit; otherwise continue.
                    builder
                        .ins()
                        .brif(is_zero, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let _ = info;
                }

                OpCode::GuardNotInvalidated
                | OpCode::GuardFutureCondition
                | OpCode::GuardAlwaysFails => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let _ = info;
                }

                OpCode::GuardGcType | OpCode::GuardIsObject | OpCode::GuardSubclass => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let _ = info;
                }

                // ── Exception operations ──
                OpCode::SaveException => {
                    // Returns current exception (Ref). Simplified: null.
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(var(vi), zero);
                }
                OpCode::SaveExcClass => {
                    // Returns current exception class (Int). Simplified: 0.
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(var(vi), zero);
                }
                OpCode::RestoreException => {
                    // args: [exception, exc_class]. No-op for now.
                }
                OpCode::CheckMemoryError => {
                    // arg(0) is the allocation result. Pass through.
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), a);
                }

                // ── Call operations ──
                //
                // Regular calls, pure calls, may-force calls, and loop-invariant
                // calls all compile the same way: an indirect call through a
                // function pointer. The optimizer treats them differently, but
                // at the backend level the semantics are identical.
                OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN
                | OpCode::CallReleaseGilI
                | OpCode::CallReleaseGilF
                | OpCode::CallReleaseGilN => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("call op must have a descriptor");
                    let call_descr = descr
                        .as_call_descr()
                        .expect("call op descriptor must be a CallDescr");

                    if let Some(result) = emit_indirect_call(
                        &mut builder,
                        &constants,
                        op,
                        call_descr,
                        call_conv,
                        ptr_type,
                    ) {
                        builder.def_var(var(vi), result);
                    }
                }

                // ── Conditional call (void result) ──
                // args[0] = condition, args[1] = func_ptr, args[2..] = call args
                // If condition != 0, perform the call.
                OpCode::CondCallN => {
                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let call_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);
                    builder
                        .ins()
                        .brif(is_zero, cont_block, &[], call_block, &[]);

                    builder.switch_to_block(call_block);
                    builder.seal_block(call_block);

                    if let Some(descr) = op.descr.as_ref() {
                        if let Some(call_descr) = descr.as_call_descr() {
                            let mut cond_sig = Signature::new(call_conv);
                            let at = call_descr.arg_types();
                            for t in at {
                                cond_sig.params.push(AbiParam::new(cranelift_type_for(t)));
                            }
                            let cond_sig_ref = builder.import_signature(cond_sig);

                            let fptr_raw =
                                resolve_opref(&mut builder, &constants, op.arg(1));
                            let fptr = if ptr_type != cl_types::I64 {
                                builder.ins().ireduce(ptr_type, fptr_raw)
                            } else {
                                fptr_raw
                            };

                            let mut cargs: Vec<CValue> = Vec::new();
                            for (i, &arg_ref) in op.args[2..].iter().enumerate() {
                                let raw =
                                    resolve_opref(&mut builder, &constants, arg_ref);
                                if i < at.len() && at[i] == Type::Float {
                                    cargs.push(builder.ins().bitcast(
                                        cl_types::F64,
                                        MemFlags::new(),
                                        raw,
                                    ));
                                } else {
                                    cargs.push(raw);
                                }
                            }
                            builder.ins().call_indirect(cond_sig_ref, fptr, &cargs);
                        }
                    }

                    builder.ins().jump(cont_block, &[]);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Conditional call with value result ──
                // args[0] = condition, args[1] = func_ptr, args[2..] = call args
                // If condition != 0: result = call(func_ptr, args...)
                // Else: result = condition (0)
                OpCode::CondCallValueI | OpCode::CondCallValueR => {
                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let call_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder.append_block_param(cont_block, cl_types::I64);

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);
                    builder
                        .ins()
                        .brif(is_zero, cont_block, &[cond], call_block, &[]);

                    builder.switch_to_block(call_block);
                    builder.seal_block(call_block);

                    let mut call_result = cond; // fallback
                    if let Some(descr) = op.descr.as_ref() {
                        if let Some(call_descr) = descr.as_call_descr() {
                            let mut cv_sig = Signature::new(call_conv);
                            let at = call_descr.arg_types();
                            for t in at {
                                cv_sig
                                    .params
                                    .push(AbiParam::new(cranelift_type_for(t)));
                            }
                            let rt = call_descr.result_type();
                            if rt != Type::Void {
                                cv_sig
                                    .returns
                                    .push(AbiParam::new(cranelift_type_for(&rt)));
                            }
                            let cv_sig_ref = builder.import_signature(cv_sig);

                            let fptr_raw =
                                resolve_opref(&mut builder, &constants, op.arg(1));
                            let fptr = if ptr_type != cl_types::I64 {
                                builder.ins().ireduce(ptr_type, fptr_raw)
                            } else {
                                fptr_raw
                            };

                            let mut cargs: Vec<CValue> = Vec::new();
                            for (i, &arg_ref) in op.args[2..].iter().enumerate() {
                                let raw =
                                    resolve_opref(&mut builder, &constants, arg_ref);
                                if i < at.len() && at[i] == Type::Float {
                                    cargs.push(builder.ins().bitcast(
                                        cl_types::F64,
                                        MemFlags::new(),
                                        raw,
                                    ));
                                } else {
                                    cargs.push(raw);
                                }
                            }
                            let c =
                                builder.ins().call_indirect(cv_sig_ref, fptr, &cargs);
                            if rt != Type::Void {
                                call_result = builder.inst_results(c)[0];
                                if rt == Type::Float {
                                    call_result = builder.ins().bitcast(
                                        cl_types::I64,
                                        MemFlags::new(),
                                        call_result,
                                    );
                                }
                            }
                        }
                    }

                    builder.ins().jump(cont_block, &[call_result]);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);

                    let phi = builder.block_params(cont_block)[0];
                    builder.def_var(var(vi), phi);
                }

                // ── GC allocation calls ──
                // Stub implementations. When a real GC runtime is wired in,
                // these will call the nursery allocator.
                OpCode::CallMallocNursery => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let _ = a;
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(var(vi), zero);
                }
                OpCode::CallMallocNurseryVarsize => {
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(var(vi), zero);
                }
                OpCode::CallMallocNurseryVarsizeFrame => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let _ = a;
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(var(vi), zero);
                }

                // ── GC write barriers ──
                OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                    // Simplified: no-op. When GC support is added, this would
                    // test a flag in the object header and call the barrier
                    // function if needed.
                }

                // ── Field access (getfield) ──
                // All getfield variants load from base + offset.
                // The loaded value is sign/zero-extended from its field_size to I64,
                // or bitcast from F64 for float fields.
                OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
                | OpCode::GetfieldGcPureI
                | OpCode::GetfieldGcPureR
                | OpCode::GetfieldGcPureF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("getfield op must have a descriptor");
                    let fd = descr
                        .as_field_descr()
                        .expect("getfield descriptor must be a FieldDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let addr = builder.ins().iadd_imm(base, fd.offset() as i64);

                    if fd.field_type() == Type::Float {
                        let fval = builder
                            .ins()
                            .load(cl_types::F64, MemFlags::trusted(), addr, 0);
                        let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fval);
                        builder.def_var(var(vi), r);
                    } else {
                        let mem_ty = cl_type_for_size(fd.field_size());
                        let raw = builder
                            .ins()
                            .load(mem_ty, MemFlags::trusted(), addr, 0);
                        let r = if mem_ty == cl_types::I64 {
                            raw
                        } else if fd.is_field_signed() {
                            builder.ins().sextend(cl_types::I64, raw)
                        } else {
                            builder.ins().uextend(cl_types::I64, raw)
                        };
                        builder.def_var(var(vi), r);
                    }
                }

                // ── Field access (setfield) ──
                // args[0] = base, args[1] = value
                OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("setfield op must have a descriptor");
                    let fd = descr
                        .as_field_descr()
                        .expect("setfield descriptor must be a FieldDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let val = resolve_opref(&mut builder, &constants, op.arg(1));
                    let addr = builder.ins().iadd_imm(base, fd.offset() as i64);

                    if fd.field_type() == Type::Float {
                        let fval = builder.ins().bitcast(cl_types::F64, MemFlags::new(), val);
                        builder.ins().store(MemFlags::trusted(), fval, addr, 0);
                    } else {
                        let mem_ty = cl_type_for_size(fd.field_size());
                        let store_val = if mem_ty == cl_types::I64 {
                            val
                        } else {
                            builder.ins().ireduce(mem_ty, val)
                        };
                        builder
                            .ins()
                            .store(MemFlags::trusted(), store_val, addr, 0);
                    }
                }

                // ── Array access (getarrayitem) ──
                // args[0] = base, args[1] = index
                // address = base + base_size + index * item_size
                OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemRawF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("getarrayitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("getarrayitem descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let index = resolve_opref(&mut builder, &constants, op.arg(1));
                    let item_sz = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                    let offset = builder.ins().imul(index, item_sz);
                    let with_base = builder.ins().iadd_imm(offset, ad.base_size() as i64);
                    let addr = builder.ins().iadd(base, with_base);

                    if ad.item_type() == Type::Float {
                        let fval = builder
                            .ins()
                            .load(cl_types::F64, MemFlags::trusted(), addr, 0);
                        let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fval);
                        builder.def_var(var(vi), r);
                    } else {
                        let mem_ty = cl_type_for_size(ad.item_size());
                        let raw = builder
                            .ins()
                            .load(mem_ty, MemFlags::trusted(), addr, 0);
                        let r = if mem_ty == cl_types::I64 {
                            raw
                        } else {
                            // Array items are unsigned by default (like GcRef pointers),
                            // but integer items are signed.
                            if ad.item_type() == Type::Int {
                                builder.ins().sextend(cl_types::I64, raw)
                            } else {
                                builder.ins().uextend(cl_types::I64, raw)
                            }
                        };
                        builder.def_var(var(vi), r);
                    }
                }

                // ── Array access (setarrayitem) ──
                // args[0] = base, args[1] = index, args[2] = value
                OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("setarrayitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("setarrayitem descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let index = resolve_opref(&mut builder, &constants, op.arg(1));
                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    let item_sz = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                    let offset = builder.ins().imul(index, item_sz);
                    let with_base = builder.ins().iadd_imm(offset, ad.base_size() as i64);
                    let addr = builder.ins().iadd(base, with_base);

                    if ad.item_type() == Type::Float {
                        let fval = builder.ins().bitcast(cl_types::F64, MemFlags::new(), val);
                        builder.ins().store(MemFlags::trusted(), fval, addr, 0);
                    } else {
                        let mem_ty = cl_type_for_size(ad.item_size());
                        let store_val = if mem_ty == cl_types::I64 {
                            val
                        } else {
                            builder.ins().ireduce(mem_ty, val)
                        };
                        builder
                            .ins()
                            .store(MemFlags::trusted(), store_val, addr, 0);
                    }
                }

                // ── Array/string length ──
                // These load the length field from the object header using
                // the array descriptor's len_descr.
                OpCode::ArraylenGc => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("arraylen op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("arraylen descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    if let Some(ld) = ad.len_descr() {
                        let addr = builder.ins().iadd_imm(base, ld.offset() as i64);
                        let mem_ty = cl_type_for_size(ld.field_size());
                        let raw = builder
                            .ins()
                            .load(mem_ty, MemFlags::trusted(), addr, 0);
                        let r = if mem_ty == cl_types::I64 {
                            raw
                        } else if ld.is_field_signed() {
                            builder.ins().sextend(cl_types::I64, raw)
                        } else {
                            builder.ins().uextend(cl_types::I64, raw)
                        };
                        builder.def_var(var(vi), r);
                    } else {
                        // No len_descr: return 0 as a fallback.
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var(vi), zero);
                    }
                }

                OpCode::Strlen | OpCode::Unicodelen => {
                    // These use the array descriptor attached to the op.
                    // The length is at len_descr().offset() from the base pointer.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("strlen/unicodelen op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("strlen/unicodelen descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    if let Some(ld) = ad.len_descr() {
                        let addr = builder.ins().iadd_imm(base, ld.offset() as i64);
                        let mem_ty = cl_type_for_size(ld.field_size());
                        let raw = builder
                            .ins()
                            .load(mem_ty, MemFlags::trusted(), addr, 0);
                        let r = if mem_ty == cl_types::I64 {
                            raw
                        } else if ld.is_field_signed() {
                            builder.ins().sextend(cl_types::I64, raw)
                        } else {
                            builder.ins().uextend(cl_types::I64, raw)
                        };
                        builder.def_var(var(vi), r);
                    } else {
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var(vi), zero);
                    }
                }

                // ── String/unicode item access ──
                // Strgetitem/Unicodegetitem: args[0] = base, args[1] = index
                // Treated as array item access using the descriptor's base_size/item_size.
                OpCode::Strgetitem | OpCode::Unicodegetitem => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("str/unicodegetitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("str/unicodegetitem descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let index = resolve_opref(&mut builder, &constants, op.arg(1));
                    let item_sz = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                    let offset = builder.ins().imul(index, item_sz);
                    let with_base = builder.ins().iadd_imm(offset, ad.base_size() as i64);
                    let addr = builder.ins().iadd(base, with_base);

                    let mem_ty = cl_type_for_size(ad.item_size());
                    let raw = builder
                        .ins()
                        .load(mem_ty, MemFlags::trusted(), addr, 0);
                    let r = if mem_ty == cl_types::I64 {
                        raw
                    } else {
                        builder.ins().uextend(cl_types::I64, raw)
                    };
                    builder.def_var(var(vi), r);
                }

                // Strsetitem/Unicodesetitem: args[0] = base, args[1] = index, args[2] = value
                OpCode::Strsetitem | OpCode::Unicodesetitem => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("str/unicodesetitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("str/unicodesetitem descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let index = resolve_opref(&mut builder, &constants, op.arg(1));
                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    let item_sz = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                    let offset = builder.ins().imul(index, item_sz);
                    let with_base = builder.ins().iadd_imm(offset, ad.base_size() as i64);
                    let addr = builder.ins().iadd(base, with_base);

                    let mem_ty = cl_type_for_size(ad.item_size());
                    let store_val = if mem_ty == cl_types::I64 {
                        val
                    } else {
                        builder.ins().ireduce(mem_ty, val)
                    };
                    builder
                        .ins()
                        .store(MemFlags::trusted(), store_val, addr, 0);
                }

                // ── Nursery pointer increment ──
                // args[0] = base ptr, args[1] = byte offset
                OpCode::NurseryPtrIncrement => {
                    let (base, offset) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().iadd(base, offset);
                    builder.def_var(var(vi), r);
                }

                // ── Control flow ──
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
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);
                }

                OpCode::Label => {}

                // ── Float arithmetic ──
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

                // ── Casts ──
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
                OpCode::ConvertFloatBytesToLonglong | OpCode::ConvertLonglongBytesToFloat => {
                    // Both are identity in our I64 storage scheme.
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), a);
                }

                // ── Debug / no-op operations ──
                OpCode::DebugMergePoint
                | OpCode::EnterPortalFrame
                | OpCode::LeavePortalFrame
                | OpCode::JitDebug
                | OpCode::Keepalive
                | OpCode::ForceSpill
                | OpCode::VirtualRefFinish
                | OpCode::RecordExactClass
                | OpCode::RecordExactValueR
                | OpCode::RecordExactValueI
                | OpCode::RecordKnownResult
                | OpCode::QuasiimmutField
                | OpCode::AssertNotNone
                | OpCode::IncrementDebugCounter => {
                    // No-op markers or optimizer hints.
                }

                // ── ForceToken ──
                OpCode::ForceToken => {
                    // Returns a pointer to the current "virtual frame".
                    // Simplified: return 0.
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(var(vi), zero);
                }

                other => {
                    if other.result_type() != Type::Void {
                        return Err(BackendError::Unsupported(format!(
                            "opcode {:?} not yet implemented",
                            other
                        )));
                    }
                    // Void result opcodes we don't explicitly handle: silently skip.
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
        } else if let Some(ref fa) = op.fail_args {
            let refs: Vec<OpRef> = fa.iter().copied().collect();
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

        let descr = Arc::new(CraneliftFailDescr::new(fail_index, fail_arg_types));
        fail_descrs.push(descr);
        guard_infos.push(GuardInfo {
            fail_index,
            fail_arg_refs,
        });
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
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &LoopToken,
    ) -> Result<AsmInfo, BackendError> {
        // Compile the bridge trace as a standalone function using the same
        // code generation path as compile_loop.
        let compiled = self.do_compile(inputargs, ops)?;
        let info = AsmInfo {
            code_addr: compiled.code_ptr as usize,
            code_size: compiled.code_size,
        };

        // Attach the bridge to the original guard's fail descriptor so that
        // execute_token can dispatch to it on subsequent guard failures.
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>())
            .ok_or_else(|| {
                BackendError::CompilationFailed(
                    "original token has no compiled loop".to_string(),
                )
            })?;

        let fi = fail_descr.fail_index() as usize;
        if fi < original_compiled.fail_descrs.len() {
            let target_descr = &original_compiled.fail_descrs[fi];
            target_descr.attach_bridge(BridgeData {
                code_ptr: compiled.code_ptr,
                fail_descrs: compiled.fail_descrs,
                num_inputs: compiled.num_inputs,
                max_output_slots: compiled.max_output_slots,
            });
        }

        Ok(info)
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

        // Increment guard failure count.
        fail_descr.increment_fail_count();

        // If a bridge is attached to this guard, execute it.
        // The bridge receives the guard's fail args as its inputs.
        let bridge_guard = fail_descr.bridge.lock().unwrap();
        if let Some(ref bridge) = *bridge_guard {
            return Self::execute_bridge(bridge, &outputs, &fail_descr.fail_arg_types);
        }
        drop(bridge_guard);

        let values = Self::decode_frame_values(&outputs, &fail_descr.fail_arg_types);

        DeadFrame {
            data: Box::new(FrameData {
                values,
                fail_descr: fail_descr.clone(),
            }),
        }
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        frame
            .data
            .downcast_ref::<FrameData>()
            .expect("FrameData expected")
            .fail_descr
            .as_ref()
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        frame
            .data
            .downcast_ref::<FrameData>()
            .expect("FrameData expected")
            .get_int(index)
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        frame
            .data
            .downcast_ref::<FrameData>()
            .expect("FrameData expected")
            .get_float(index)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        frame
            .data
            .downcast_ref::<FrameData>()
            .expect("FrameData expected")
            .get_ref(index)
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
    use majit_ir::descr::{Descr, EffectInfo, ExtraEffect};
    use std::collections::HashMap;

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut o = Op::new(opcode, args);
        o.pos = OpRef(pos);
        o
    }

    fn mk_op_with_descr(
        opcode: OpCode,
        args: &[OpRef],
        pos: u32,
        descr: majit_ir::DescrRef,
    ) -> Op {
        let mut o = Op::with_descr(opcode, args, descr);
        o.pos = OpRef(pos);
        o
    }

    #[derive(Debug)]
    struct TestCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl Descr for TestCallDescr {
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            8
        }
        fn effect_info(&self) -> &EffectInfo {
            &EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                oopspec_index: majit_ir::OopSpecIndex::None,
            }
        }
    }

    fn make_call_descr(arg_types: Vec<Type>, result_type: Type) -> majit_ir::DescrRef {
        Arc::new(TestCallDescr {
            arg_types,
            result_type,
        })
    }

    // ── Existing tests ──

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
    fn test_int_floor_div() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntFloorDiv, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(3);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42), Value::Int(6)]);
        assert_eq!(backend.get_int_value(&frame, 0), 7);
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
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4)],
                OpRef::NONE.0,
            ),
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
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4)],
                OpRef::NONE.0,
            ),
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
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4), OpRef(5), OpRef(6), OpRef(7)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = LoopToken::new(6);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 1);
        assert_eq!(backend.get_int_value(&frame, 2), 0);
        assert_eq!(backend.get_int_value(&frame, 3), 1);
        assert_eq!(backend.get_int_value(&frame, 4), 0);
        assert_eq!(backend.get_int_value(&frame, 5), 0);
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

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(0)]);
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
            mk_op(
                OpCode::Finish,
                &[OpRef(0), OpRef(1), OpRef(2), OpRef(3)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = LoopToken::new(10);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 3);
        assert_eq!(backend.get_int_value(&frame, 1), 7);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(backend.get_int_value(&frame, 3), 21);
    }

    // ── Call operation tests ──

    #[test]
    fn test_call_i_simple_add() {
        extern "C" fn add_two(a: i64, b: i64) -> i64 {
            a + b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallI,
                &[OpRef(100), OpRef(0), OpRef(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_two as *const () as i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(20);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(40), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_call_pure_i() {
        extern "C" fn multiply(a: i64, b: i64) -> i64 {
            a * b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallPureI,
                &[OpRef(100), OpRef(0), OpRef(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, multiply as *const () as i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(21);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_call_n_void_result() {
        static mut CALL_COUNTER: i64 = 0;

        extern "C" fn increment_counter(amount: i64) {
            unsafe {
                CALL_COUNTER += amount;
            }
        }

        unsafe {
            CALL_COUNTER = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef(100), OpRef(0)],
                OpRef::NONE.0,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, increment_counter as *const () as i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(22);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(10)]);
        assert_eq!(backend.get_int_value(&frame, 0), 10);
        assert_eq!(unsafe { CALL_COUNTER }, 10);
    }

    #[test]
    fn test_call_f_double_result() {
        extern "C" fn add_doubles(a: f64, b: f64) -> f64 {
            a + b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Float, Type::Float], Type::Float);

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallF,
                &[OpRef(100), OpRef(0), OpRef(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_doubles as *const () as i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(23);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(1.5), Value::Float(2.5)]);
        let raw = backend.get_int_value(&frame, 0);
        let result = f64::from_bits(raw as u64);
        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_call_in_loop() {
        extern "C" fn add_one(a: i64) -> i64 {
            a + 1
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallI, &[OpRef(100), OpRef(0)], 1, descr),
            mk_op(OpCode::IntLt, &[OpRef(1), OpRef(101)], 2),
            mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_one as *const () as i64);
        constants.insert(101, 100i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(24);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
    }

    // ── Debug / no-op tests ──

    #[test]
    fn test_debug_ops_are_noop() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::DebugMergePoint, &[], OpRef::NONE.0),
            mk_op(
                OpCode::EnterPortalFrame,
                &[OpRef(100), OpRef(101)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::LeavePortalFrame, &[OpRef(100)], OpRef::NONE.0),
            mk_op(OpCode::JitDebug, &[], OpRef::NONE.0),
            mk_op(OpCode::Keepalive, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0i64);
        constants.insert(101, 0i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(30);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    // ── SameAs variants ──

    #[test]
    fn test_same_as_r() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::SameAsR, &[OpRef(0)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(31);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(0x1234))]);
        assert_eq!(backend.get_int_value(&frame, 0), 0x1234);
    }

    #[test]
    fn test_same_as_f() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::SameAsF, &[OpRef(0)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(32);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(3.14)]);
        let raw = backend.get_int_value(&frame, 0);
        let result = f64::from_bits(raw as u64);
        assert!((result - 3.14).abs() < 1e-10);
    }

    // ── Compile bridge test ──

    #[test]
    fn test_compile_bridge() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(50);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 10i64);
        backend.set_constants(constants);

        let fail_descr = CraneliftFailDescr::new(0, vec![Type::Int]);

        let info = backend
            .compile_bridge(&fail_descr, &bridge_inputargs, &bridge_ops, &token)
            .unwrap();

        assert!(info.code_addr != 0);
    }

    // ── Conditional call tests ──

    #[test]
    fn test_cond_call_n_calls_when_nonzero() {
        static mut COND_CALL_RESULT: i64 = 0;

        extern "C" fn set_value(v: i64) {
            unsafe {
                COND_CALL_RESULT = v;
            }
        }

        unsafe {
            COND_CALL_RESULT = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallN,
                &[OpRef(0), OpRef(100), OpRef(1)],
                OpRef::NONE.0,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, set_value as *const () as i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(40);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1), Value::Int(99)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
        assert_eq!(unsafe { COND_CALL_RESULT }, 99);
    }

    #[test]
    fn test_cond_call_n_skips_when_zero() {
        static mut COND_CALL_RESULT2: i64 = 0;

        extern "C" fn set_value2(v: i64) {
            unsafe {
                COND_CALL_RESULT2 = v;
            }
        }

        unsafe {
            COND_CALL_RESULT2 = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallN,
                &[OpRef(0), OpRef(100), OpRef(1)],
                OpRef::NONE.0,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, set_value2 as *const () as i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(41);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(99)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
        assert_eq!(unsafe { COND_CALL_RESULT2 }, 0);
    }

    #[test]
    fn test_cond_call_value_i_nonzero() {
        extern "C" fn compute(a: i64) -> i64 {
            a * 10
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallValueI,
                &[OpRef(0), OpRef(100), OpRef(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, compute as *const () as i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(42);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 50);
    }

    #[test]
    fn test_cond_call_value_i_zero() {
        extern "C" fn compute2(a: i64) -> i64 {
            a * 10
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallValueI,
                &[OpRef(0), OpRef(100), OpRef(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, compute2 as *const () as i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(43);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    // ── Guard variant tests ──

    #[test]
    fn test_guard_nonnull() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::GuardNonnull, &[OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(44);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i0=5->i1=4, i0=4->i1=3, ..., i0=1->i1=0 (guard fails).
        // Guard saves the loop inputarg (i0), so saved value is 1.
        let frame = backend.execute_token(&token, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
    }

    #[test]
    fn test_guard_isnull_passes() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardIsnull, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(45);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    #[test]
    fn test_guard_isnull_fails() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardIsnull, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(46);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_guard_value() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardValue, &[OpRef(0), OpRef(100)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        // Test: value matches -> guard passes, reaches Finish
        let mut constants = HashMap::new();
        constants.insert(100, 42i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(47);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish
        assert_eq!(backend.get_int_value(&frame, 0), 42);

        // Test: value doesn't match -> guard fails
        let mut constants2 = HashMap::new();
        constants2.insert(100, 42i64);
        backend.set_constants(constants2);

        let mut token2 = LoopToken::new(48);
        backend
            .compile_loop(&inputargs, &ops, &mut token2)
            .unwrap();
        let frame2 = backend.execute_token(&token2, &[Value::Int(99)]);
        let descr2 = backend.get_latest_descr(&frame2);
        assert_eq!(descr2.fail_index(), 0); // guard failure
        assert_eq!(backend.get_int_value(&frame2, 0), 99);
    }

    // ── Test descriptors for field/array ops ──

    use majit_ir::descr::{ArrayDescr, FieldDescr};

    #[derive(Debug)]
    struct TestFieldDescr {
        offset: usize,
        field_size: usize,
        field_type: Type,
        signed: bool,
    }

    impl Descr for TestFieldDescr {
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }
        fn field_size(&self) -> usize {
            self.field_size
        }
        fn field_type(&self) -> Type {
            self.field_type
        }
        fn is_field_signed(&self) -> bool {
            self.signed
        }
    }

    #[derive(Debug)]
    struct TestArrayDescr {
        base_size: usize,
        item_size: usize,
        item_type: Type,
        len_descr: Option<Arc<TestFieldDescr>>,
    }

    impl Descr for TestArrayDescr {
        fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
            Some(self)
        }
    }

    impl ArrayDescr for TestArrayDescr {
        fn base_size(&self) -> usize {
            self.base_size
        }
        fn item_size(&self) -> usize {
            self.item_size
        }
        fn type_id(&self) -> u32 {
            0
        }
        fn item_type(&self) -> Type {
            self.item_type
        }
        fn len_descr(&self) -> Option<&dyn FieldDescr> {
            self.len_descr.as_ref().map(|d| d.as_ref() as &dyn FieldDescr)
        }
    }

    fn make_field_descr(
        offset: usize,
        field_size: usize,
        field_type: Type,
        signed: bool,
    ) -> majit_ir::DescrRef {
        Arc::new(TestFieldDescr {
            offset,
            field_size,
            field_type,
            signed,
        })
    }

    fn make_array_descr(
        base_size: usize,
        item_size: usize,
        item_type: Type,
        len_offset: Option<usize>,
    ) -> majit_ir::DescrRef {
        let len_descr = len_offset.map(|off| {
            Arc::new(TestFieldDescr {
                offset: off,
                field_size: 8,
                field_type: Type::Int,
                signed: true,
            })
        });
        Arc::new(TestArrayDescr {
            base_size,
            item_size,
            item_type,
            len_descr,
        })
    }

    // ── Float comparison tests ──

    #[test]
    fn test_float_comparisons() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::FloatLt, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::FloatLe, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::FloatEq, &[OpRef(0), OpRef(1)], 4),
            mk_op(OpCode::FloatNe, &[OpRef(0), OpRef(1)], 5),
            mk_op(OpCode::FloatGt, &[OpRef(0), OpRef(1)], 6),
            mk_op(OpCode::FloatGe, &[OpRef(0), OpRef(1)], 7),
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4), OpRef(5), OpRef(6), OpRef(7)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = LoopToken::new(60);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // 1.5 < 2.5
        let frame = backend.execute_token(&token, &[Value::Float(1.5), Value::Float(2.5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1); // lt
        assert_eq!(backend.get_int_value(&frame, 1), 1); // le
        assert_eq!(backend.get_int_value(&frame, 2), 0); // eq
        assert_eq!(backend.get_int_value(&frame, 3), 1); // ne
        assert_eq!(backend.get_int_value(&frame, 4), 0); // gt
        assert_eq!(backend.get_int_value(&frame, 5), 0); // ge
    }

    #[test]
    fn test_float_comparisons_equal() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::FloatEq, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::FloatNe, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::FloatLe, &[OpRef(0), OpRef(1)], 4),
            mk_op(OpCode::FloatGe, &[OpRef(0), OpRef(1)], 5),
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4), OpRef(5)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = LoopToken::new(61);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(3.14), Value::Float(3.14)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1); // eq
        assert_eq!(backend.get_int_value(&frame, 1), 0); // ne
        assert_eq!(backend.get_int_value(&frame, 2), 1); // le
        assert_eq!(backend.get_int_value(&frame, 3), 1); // ge
    }

    // ── Field access tests ──

    #[test]
    fn test_getfield_gc_i() {
        let mut backend = CraneliftBackend::new();

        // Simulate a struct: [padding(8), i64_field(8)]
        // The field is at offset 8, size 8, type Int.
        let fd = make_field_descr(8, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(70);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Allocate a fake struct on the heap
        let mut data: Vec<i64> = vec![0xDEAD, 42];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_setfield_gc() {
        let mut backend = CraneliftBackend::new();

        let fd = make_field_descr(8, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)], OpRef::NONE.0, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(71);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0];
        let ptr = data.as_mut_ptr() as usize;

        let _frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(99)]);
        assert_eq!(data[1], 99);
    }

    #[test]
    fn test_getfield_small_signed() {
        let mut backend = CraneliftBackend::new();

        // i32 field at offset 0, signed
        let fd = make_field_descr(0, 4, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(72);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Write -1i32 into the buffer
        let mut data: Vec<u8> = vec![0; 8];
        let val: i32 = -1;
        data[..4].copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), -1i64);
    }

    // ── Array access tests ──

    #[test]
    fn test_getarrayitem_gc_i() {
        let mut backend = CraneliftBackend::new();

        // Array: base_size=16 (header), item_size=8, items are i64
        let ad = make_array_descr(16, 8, Type::Int, None);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetarrayitemGcI, &[OpRef(0), OpRef(1)], 2, ad),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(73);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Layout: 16 bytes header + items
        // Total: 16 + 3*8 = 40 bytes = 5 i64s
        let mut data: Vec<i64> = vec![0xAAAA, 0xBBBB, 10, 20, 30]; // header(2), items(3)
        let ptr = data.as_mut_ptr() as usize;

        // Get item at index 1 (should be 20)
        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(1)]);
        assert_eq!(backend.get_int_value(&frame, 0), 20);
    }

    #[test]
    fn test_setarrayitem_gc() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(16, 8, Type::Int, None);

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(74);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0, 0, 0, 0]; // header(2) + items(3)
        let ptr = data.as_mut_ptr() as usize;

        // Set item at index 2 to 42
        let _frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(ptr)), Value::Int(2), Value::Int(42)],
        );
        assert_eq!(data[4], 42); // header(2) + index 2 = slot 4
    }

    #[test]
    fn test_arraylen_gc() {
        let mut backend = CraneliftBackend::new();

        // Array with length at offset 8 (second i64 in header)
        let ad = make_array_descr(16, 8, Type::Int, Some(8));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::ArraylenGc, &[OpRef(0)], 1, ad),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(75);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // header: [type_id, length=5]
        let mut data: Vec<i64> = vec![0xAAAA, 5, 10, 20, 30, 40, 50];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 5);
    }

    // ── NurseryPtrIncrement test ──

    #[test]
    fn test_nursery_ptr_increment() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::NurseryPtrIncrement, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(76);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(0x1000)), Value::Int(0x100)],
        );
        assert_eq!(backend.get_int_value(&frame, 0), 0x1100);
    }

    // ── Overflow detection tests ──

    #[test]
    fn test_int_add_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(80);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // 10 + 20 = 30 (no overflow)
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(20)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 30);
    }

    #[test]
    fn test_int_add_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(81);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MAX + 1 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_int_sub_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(82);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MIN - 1 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MIN), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_int_sub_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(83);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(58)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntMulOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(84);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntMulOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(85);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MAX * 2 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(2)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_guard_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(86);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // With overflow: guard_overflow passes (continues)
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (overflow happened, guard passed)

        // Without overflow: guard_overflow fails (side-exits)
        let mut token2 = LoopToken::new(87);
        backend
            .compile_loop(&inputargs, &ops, &mut token2)
            .unwrap();
        let frame2 = backend.execute_token(&token2, &[Value::Int(1), Value::Int(2)]);
        let descr2 = backend.get_latest_descr(&frame2);
        assert_eq!(descr2.fail_index(), 0); // guard failure (no overflow)
    }

    // ── Getfield float test ──

    #[test]
    fn test_getfield_gc_f() {
        let mut backend = CraneliftBackend::new();

        // f64 field at offset 0
        let fd = make_field_descr(0, 8, Type::Float, false);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetfieldGcF, &[OpRef(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(90);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let val: f64 = 3.14;
        let mut data = vec![0u8; 8];
        data.copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        let raw = backend.get_int_value(&frame, 0);
        let result = f64::from_bits(raw as u64);
        assert!((result - 3.14).abs() < 1e-10);
    }

    // ── Getfield ref (pure) test ──

    #[test]
    fn test_getfield_gc_pure_r() {
        let mut backend = CraneliftBackend::new();

        // Ref field at offset 8
        let fd = make_field_descr(8, 8, Type::Ref, false);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetfieldGcPureR, &[OpRef(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(91);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0x42424242];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 0x42424242);
    }

    // ── Setfield + getfield roundtrip ──

    #[test]
    fn test_setfield_getfield_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let fd = make_field_descr(0, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef(0), OpRef(1)],
                OpRef::NONE.0,
                fd.clone(),
            ),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef(0)], 2, fd),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(92);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(12345)]);
        assert_eq!(backend.get_int_value(&frame, 0), 12345);
    }

    // ── Array getitem/setitem roundtrip ──

    #[test]
    fn test_setarrayitem_getarrayitem_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(0, 8, Type::Int, None);

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
                ad.clone(),
            ),
            mk_op_with_descr(OpCode::GetarrayitemGcI, &[OpRef(0), OpRef(1)], 3, ad),
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(93);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0, 0, 0];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(ptr)), Value::Int(2), Value::Int(777)],
        );
        assert_eq!(backend.get_int_value(&frame, 0), 777);
    }
}
