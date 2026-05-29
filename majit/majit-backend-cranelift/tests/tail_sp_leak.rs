// Minimal reproducer for the cranelift x86_64 tail-call SP-drift hypothesis.
//
// Builds two functions:
//   - body(CallConv::Tail): if arg < N, return_call_indirect itself with arg+1;
//                            else return arg.
//   - wrapper(host call_conv): forwards arg to body via normal call.
//
// Calls wrapper several times in a tight loop and measures the host RSP
// between invocations.  If the body's tail-call epilogue leaks, the host
// RSP value either drifts or the OS terminates the test with a stack
// overflow.

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use target_lexicon::Triple;

fn current_sp() -> usize {
    let probe: usize = 0;
    &probe as *const usize as usize
}

// Counter probe so we can call from JIT with the host ABI without
// worrying about how the linker mangles a Rust extern "C" symbol.
static PROBE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
extern "C" fn host_probe(_jf: usize) {
    PROBE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// Chain cap, read by the body each tail-call so we can vary chain length
// at runtime without recompiling.
static CHAIN_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2000);

#[test]
fn tail_call_sp_drift_repro() {
    let mut flag_builder = settings::builder();
    flag_builder.set("preserve_frame_pointers", "true").unwrap();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_native::builder().expect("host machine is not a supported target");
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let host_call_conv = isa.default_call_conv();
    let triple: Triple = isa.triple().clone();
    println!("host triple={triple} call_conv={host_call_conv:?}");

    let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);
    let ptr_type = cl_types::I64;

    // body signature: (i64) -> i64 (Tail conv)
    let mut body_sig = Signature::new(CallConv::Tail);
    body_sig.params.push(AbiParam::new(ptr_type));
    body_sig.returns.push(AbiParam::new(ptr_type));
    let body_id = module
        .declare_function("body", Linkage::Local, &body_sig)
        .unwrap();
    // body2 has the SAME signature but a different declared frame.
    let body2_id = module
        .declare_function("body2", Linkage::Local, &body_sig)
        .unwrap();

    // wrapper signature: (i64) -> i64 (host conv)
    let mut wrapper_sig = Signature::new(host_call_conv);
    wrapper_sig.params.push(AbiParam::new(ptr_type));
    wrapper_sig.returns.push(AbiParam::new(ptr_type));
    let wrapper_id = module
        .declare_function("wrapper", Linkage::Local, &wrapper_sig)
        .unwrap();

    // Build the body.
    let mut fbcx = FunctionBuilderContext::new();
    let mut body_func =
        Function::with_name_signature(UserFuncName::user(0, body_id.as_u32()), body_sig.clone());
    {
        let mut bx = FunctionBuilder::new(&mut body_func, &mut fbcx);
        let entry = bx.create_block();
        bx.append_block_params_for_function_params(entry);
        bx.switch_to_block(entry);
        bx.seal_block(entry);
        let arg = bx.block_params(entry)[0];

        let cap_addr = bx.ins().iconst(ptr_type, &CHAIN_CAP as *const _ as i64);
        let cap = bx.ins().load(ptr_type, MemFlags::trusted(), cap_addr, 0);
        let cont = bx.ins().icmp(IntCC::SignedLessThan, arg, cap);
        let tail = bx.create_block();
        let exit = bx.create_block();
        bx.ins().brif(cont, tail, &[], exit, &[]);

        bx.switch_to_block(tail);
        bx.seal_block(tail);
        let one = bx.ins().iconst(ptr_type, 1);
        let next = bx.ins().iadd(arg, one);

        // Replicate pyre's tail-call epilogue: emit a host-ABI call to a
        // host probe AND a shadow-stack-like in-memory pop.
        let host_addr = bx.ins().iconst(ptr_type, host_probe as *const () as i64);
        let mut probe_sig = Signature::new(host_call_conv);
        probe_sig.params.push(AbiParam::new(ptr_type));
        let probe_sig_ref = bx.import_signature(probe_sig);
        bx.ins().call_indirect(probe_sig_ref, host_addr, &[next]);

        // Tail-call body2 via INDIRECT (different declared frame size).
        let body2_ref = module.declare_func_in_func(body2_id, bx.func);
        let body2_addr = bx.ins().func_addr(ptr_type, body2_ref);
        let mut tail_sig = Signature::new(CallConv::Tail);
        tail_sig.params.push(AbiParam::new(ptr_type));
        tail_sig.returns.push(AbiParam::new(ptr_type));
        let tail_sig_ref = bx.import_signature(tail_sig);
        bx.ins()
            .return_call_indirect(tail_sig_ref, body2_addr, &[next]);

        bx.switch_to_block(exit);
        bx.seal_block(exit);
        bx.ins().return_(&[arg]);
        bx.finalize();
    }
    let mut ctx = Context::for_function(body_func);
    module.define_function(body_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);

    // Build body2 — tail-calls back to body, with a different frame size
    // (larger local-slot usage to differentiate the prologue).
    let mut body2_func =
        Function::with_name_signature(UserFuncName::user(0, body2_id.as_u32()), body_sig.clone());
    {
        let mut bx = FunctionBuilder::new(&mut body2_func, &mut fbcx);
        let entry = bx.create_block();
        bx.append_block_params_for_function_params(entry);
        bx.switch_to_block(entry);
        bx.seal_block(entry);
        let arg = bx.block_params(entry)[0];

        // Force a much larger frame so body and body2 have different sizes.
        use cranelift_codegen::ir::stackslot::{StackSlotData, StackSlotKind};
        let big_slot =
            bx.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 128, 3));
        let big_addr = bx.ins().stack_addr(ptr_type, big_slot, 0);
        bx.ins().store(MemFlags::trusted(), arg, big_addr, 0);

        // Always tail-call back to body with arg+1.
        let one = bx.ins().iconst(ptr_type, 1);
        let next = bx.ins().iadd(arg, one);
        let body_ref = module.declare_func_in_func(body_id, bx.func);
        let body_addr = bx.ins().func_addr(ptr_type, body_ref);
        let mut tail_sig = Signature::new(CallConv::Tail);
        tail_sig.params.push(AbiParam::new(ptr_type));
        tail_sig.returns.push(AbiParam::new(ptr_type));
        let tail_sig_ref = bx.import_signature(tail_sig);
        bx.ins()
            .return_call_indirect(tail_sig_ref, body_addr, &[next]);
        bx.finalize();
    }
    let mut ctx = Context::for_function(body2_func);
    module.define_function(body2_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);

    // Build the wrapper.
    let mut wrapper_func = Function::with_name_signature(
        UserFuncName::user(0, wrapper_id.as_u32()),
        wrapper_sig.clone(),
    );
    {
        let mut bx = FunctionBuilder::new(&mut wrapper_func, &mut fbcx);
        let entry = bx.create_block();
        bx.append_block_params_for_function_params(entry);
        bx.switch_to_block(entry);
        bx.seal_block(entry);
        let arg = bx.block_params(entry)[0];
        let body_ref = module.declare_func_in_func(body_id, bx.func);
        let call = bx.ins().call(body_ref, &[arg]);
        let r = bx.inst_results(call)[0];
        bx.ins().return_(&[r]);
        bx.finalize();
    }
    let mut ctx = Context::for_function(wrapper_func);
    module.define_function(wrapper_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);

    module.finalize_definitions().unwrap();

    let wrapper_ptr = module.get_finalized_function(wrapper_id);
    let wrapper: extern "C" fn(usize) -> usize = unsafe { std::mem::transmute(wrapper_ptr) };

    use std::sync::atomic::Ordering::Relaxed;
    // Measure drift for short vs long chains. If cranelift's tail-call
    // leaks per take, drift scales with chain length. If drift is a
    // constant (test-harness noise), it doesn't.
    CHAIN_CAP.store(1000, Relaxed);
    let sp0 = current_sp();
    let r = wrapper(0);
    assert_eq!(r, 1000);
    let sp1 = current_sp();
    let drift_short = sp0.wrapping_sub(sp1) as i64;

    CHAIN_CAP.store(5_000_000, Relaxed);
    let sp2 = current_sp();
    let r = wrapper(0);
    assert_eq!(r, 5_000_000);
    let sp3 = current_sp();
    let drift_long = sp2.wrapping_sub(sp3) as i64;

    println!("drift 1k-chain = {drift_short} bytes, 5M-chain = {drift_long} bytes");
    let per_call = (drift_long - drift_short) as f64 / (5_000_000.0 - 1000.0);
    println!("per-call leak estimate: {per_call} bytes");
    // A genuine per-take leak makes drift scale with chain length, so the
    // 5M chain would dwarf the 1k chain.  A balanced epilogue leaves both
    // drifts as small, comparable, chain-length-independent harness noise.
    assert!(
        per_call.abs() < 1.0,
        "tail-call leaks {per_call} bytes/take (drift scales with chain length)"
    );
}
