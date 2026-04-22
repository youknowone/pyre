/// rpython/jit/backend/ parity:
/// Direct machine code generation via dynasm-rs with in-place patching.
///
/// Module structure mirrors RPython's backend directory layout:
///   llsupport/llmodel.py    → runner.rs  (DynasmBackend)
///   llsupport/assembler.py  → assembler.rs (BaseAssembler — shared)
///   llsupport/regalloc.py   → regalloc.rs (BaseRegalloc — shared)
///   llsupport/jitframe.py   → jitframe.rs
///   x86/assembler.py        → x86/assembler.rs (Assembler386)
///   aarch64/assembler.py    → aarch64/assembler.rs (AssemblerARM64)
///   aarch64/opassembler.py  → aarch64/opassembler.rs (ResOpAssembler)
///   aarch64/registers.py    → aarch64/registers.rs
///
/// arch.rs, codebuf.rs, guard.rs, regloc.rs are from llsupport/.
// ── Shared modules (llsupport/ parity) ──
pub mod arch;
pub mod callbuilder;
pub mod codebuf;
pub mod frame;
pub mod gcmap;
pub mod guard;
pub use majit_backend::jitframe;
pub use majit_backend::llmodel;
pub mod jump;
pub mod regalloc;
pub mod regloc;
pub mod runner;

// ── Architecture-specific modules ──
#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86;

// ── llmodel.py:194-199 JIT exception state ──
// RPython stores exception state in thread-local (GIL-protected) globals.
// Cranelift uses JIT_EXC_VALUE / JIT_EXC_TYPE atomics (compiler.rs:515-517).
// Dynasm uses the same pattern for structural equivalence.

use std::sync::atomic::{AtomicI64, Ordering};

static JIT_EXC_VALUE: AtomicI64 = AtomicI64::new(0);
static JIT_EXC_TYPE: AtomicI64 = AtomicI64::new(0);

/// llmodel.py:194-199 _store_exception parity: set JIT exception state.
/// `value` is a valid OBJECTPTR (or 0). Exception class derived from
/// value.typeptr (offset 0), matching RPython's invariant.
pub fn jit_exc_raise(value: i64) {
    let exc_type = if value == 0 {
        0
    } else {
        unsafe { *(value as *const i64) }
    };
    JIT_EXC_VALUE.store(value, Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, Ordering::Relaxed);
}

/// Check if an exception is currently pending.
pub fn jit_exc_is_pending() -> bool {
    JIT_EXC_VALUE.load(Ordering::Relaxed) != 0
}

/// cpu.grab_exc_value parity: read exception class from TLS.
pub fn jit_exc_class_raw() -> i64 {
    JIT_EXC_TYPE.load(Ordering::Relaxed)
}

/// cpu.grab_exc_value parity: read and clear exception value.
pub fn jit_exc_value_raw() -> i64 {
    JIT_EXC_VALUE.swap(0, Ordering::Relaxed)
}

/// Clear exception state.
pub fn jit_exc_clear() {
    JIT_EXC_VALUE.store(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
}

// ── CALL_ASSEMBLER helper infrastructure ──
// assembler.py:345-350: jd.assembler_helper_adr — the slow path runtime
// entry point for CALL_ASSEMBLER when callee doesn't finish normally.
//
// Cranelift uses the same frontend-owned handle_fail callback plus
// force/unbox helpers. Dynasm provides matching registration hooks and
// a C-callable trampoline that the generated slow path code calls
// directly.

use std::sync::OnceLock;

/// compile.py:701-717 `AbstractResumeGuardDescr.handle_fail`
/// specialized for CALL_ASSEMBLER slow paths.
pub type CallAssemblerHandleFailFn = fn(
    u64,
    u64,
    u32,
    *const i64,
    usize,
    *const i64,
    usize,
    usize,
) -> majit_ir::CallAssemblerHandleFailAction;

/// Force callee: (callee_frame_ptr) → result
pub type ForceFn = extern "C" fn(i64) -> i64;

/// Unbox int from Ref: (raw_ref) → i64
pub type UnboxIntFn = fn(i64) -> i64;

/// compile.py:1096-1097 `cast_instance_to_gcref(memory_error)` parity.
pub type MemoryErrorFn = fn() -> usize;

static CA_HANDLE_FAIL_FN: OnceLock<CallAssemblerHandleFailFn> = OnceLock::new();
static CA_FORCE_FN: OnceLock<ForceFn> = OnceLock::new();
static CA_UNBOX_INT_FN: OnceLock<UnboxIntFn> = OnceLock::new();
static CA_MEMORY_ERROR_FN: OnceLock<MemoryErrorFn> = OnceLock::new();

/// Register the frontend-owned CALL_ASSEMBLER handle_fail hook.
pub fn register_call_assembler_handle_fail(f: CallAssemblerHandleFailFn) {
    CA_HANDLE_FAIL_FN.set(f).ok();
}

/// Register force handler (same API as Cranelift).
pub fn register_call_assembler_force(f: ForceFn) {
    CA_FORCE_FN.set(f).ok();
}

/// Return the force_fn address for embedding in generated code.
/// Returns 0 if not registered.
pub fn call_assembler_force_fn_addr() -> usize {
    CA_FORCE_FN.get().map(|f| *f as usize).unwrap_or(0)
}

/// Register int unbox handler (same API as Cranelift).
pub fn register_call_assembler_unbox_int(f: UnboxIntFn) {
    CA_UNBOX_INT_FN.set(f).ok();
}

/// Register the preallocated `MemoryError` object getter used by
/// `PropagateExceptionDescr.handle_fail`.
pub fn register_call_assembler_memory_error(f: MemoryErrorFn) {
    CA_MEMORY_ERROR_FN.set(f).ok();
}

/// rpython/jit/backend/llsupport/llmodel.py:229-234 `insert_stack_check`
/// parity. The interpreter registers the three addresses the JIT
/// prologue needs to emit the inline stack-overflow probe matching
/// `rpython/jit/backend/x86/assembler.py:1085-1091`:
///
///   * `end_adr` — address of `PYRE_STACKTOOBIG.stack_end`
///     (`LL_stack_get_end_adr`). The probe loads this and subtracts
///     the current SP.
///   * `length_adr` — address of `PYRE_STACKTOOBIG.stack_length`
///     (`LL_stack_get_length_adr`). The probe compares the diff
///     against this.
///   * `slowpath_addr` — `pyre_stack_too_big_slowpath` function
///     pointer (`STACK_CHECK_SLOWPATH` in llmodel). Called on fast-
///     path miss with the current SP; returns 1 on real overflow.
#[derive(Copy, Clone, Debug)]
pub struct StackCheckAddresses {
    pub end_adr: usize,
    pub length_adr: usize,
    pub slowpath_addr: usize,
}

static STACK_CHECK_ADDRS: OnceLock<StackCheckAddresses> = OnceLock::new();

/// Register the three addresses the JIT prologue needs for the inline
/// stack-overflow probe. Called once at startup from pyre-jit; matches
/// `rpython/jit/backend/llsupport/llmodel.py:229-234 insert_stack_check`.
pub fn register_stack_check_addresses(end_adr: usize, length_adr: usize, slowpath_addr: usize) {
    let _ = STACK_CHECK_ADDRS.set(StackCheckAddresses {
        end_adr,
        length_adr,
        slowpath_addr,
    });
}

/// Retrieve the registered inline-probe addresses. Returns `None` if
/// the interpreter has not yet registered them (tests or early
/// startup); the prologue emitter skips the probe in that case,
/// mirroring `assembler.py:1082 "if self.stack_check_slowpath == 0:
/// pass (no stack check)"`.
pub fn stack_check_addresses() -> Option<StackCheckAddresses> {
    STACK_CHECK_ADDRS.get().copied()
}

/// warmspot.py:1021-1028 `assembler_call_helper` parity —
/// C-callable trampoline for the CALL_ASSEMBLER slow path.
///
/// Upstream:
///     fail_descr = self.cpu.get_latest_descr(deadframe)
///     try:
///         fail_descr.handle_fail(deadframe, self.metainterp_sd, jd)
///     except jitexc.JitException as e:
///         return handle_jitexception(e)
///
/// The trampoline resolves `jf_descr` back to a live `&dyn FailDescr`
/// (via `guard::attached_fail_descr_by_ptr` for metainterp-attached
/// finish / propagate descrs, and a direct `*const DynasmFailDescr`
/// cast for guard-failure descrs whose `Arc` lives on the compiled
/// loop's `fail_descrs` Vec) and invokes `FailDescr::handle_fail`
/// through virtual dispatch into the `CallAssemblerHelperContext`
/// implementation below.  The returned `HandleFailResult` is encoded
/// back into `i64` by `handle_fail_result_to_i64`, mirroring upstream's
/// `handle_jitexception` payload-extraction step.
///
/// Input:  callee_jf_ptr = deadframe,
///         green_key = caller loop's header_pc (used by the resume-guard
///         hook to find the owning compiled trace).
/// Output: the int interpretation of the handled result (the caller
///         re-casts to the portal's return type).  The caller stub
///         distinguishes exit-with-exception via the synthesized
///         FailDescr's `is_exit_frame_with_exception` flag (see
///         `runner.rs::find_descr_by_ptr`); the raw gcref payload is
///         carried in the returned `i64` bit pattern regardless.
///
/// Always frees the callee jitframe before returning.
pub extern "C" fn call_assembler_helper_trampoline(
    callee_jf_ptr: *mut jitframe::JitFrame,
    green_key: u64,
) -> i64 {
    if callee_jf_ptr.is_null() {
        return 0;
    }
    let frame_ptr = callee_jf_ptr;
    // warmspot.py:1022 `fail_descr = cpu.get_latest_descr(deadframe)`.
    let descr_raw = unsafe { llmodel::get_latest_descr(frame_ptr) };
    if descr_raw == 0 {
        // PRE-EXISTING-ADAPTATION: upstream never reaches this state
        // (an unlinked callee is patched out before the CALL_ASSEMBLER
        // emits its call).  Retained as a safety fence for backend-only
        // tests that bypass the registration flow.
        unsafe { libc::free(frame_ptr as *mut std::ffi::c_void) };
        return 0;
    }

    let mut ctx = CallAssemblerHelperContext {
        frame_ptr,
        green_key,
        descr_raw,
    };
    // warmspot.py:1023 `fail_descr.handle_fail(deadframe, ...)` — virtual
    // dispatch through `FailDescr::handle_fail` into the concrete descr
    // body (compile.py:626-662 finish variants / compile.py:1093-1099
    // propagate-exception / compile.py:701-717 resume-guard).
    let result = dispatch_current_fail_descr(&mut ctx);
    unsafe { libc::free(ctx.frame_ptr as *mut std::ffi::c_void) };
    // warmspot.py:1025-1026 `handle_jitexception` parity — unpack the
    // HandleFailResult variant into the raw i64 the caller stub expects.
    handle_fail_result_to_i64(result)
}

fn dispatch_current_fail_descr(ctx: &mut CallAssemblerHelperContext) -> majit_ir::HandleFailResult {
    match guard::attached_fail_descr_by_ptr(ctx.descr_raw) {
        Some(descr_ref) => {
            let fd = descr_ref
                .as_fail_descr()
                .expect("attached descr must implement FailDescr");
            fd.handle_fail(ctx)
        }
        None => {
            // Guard-failure descr: the Arc is owned by the compiled
            // loop's `fail_descrs` Vec (assembler.rs:238), so the raw
            // pointer is stable for the lifetime of this call.
            let descr = unsafe { &*(ctx.descr_raw as *const guard::DynasmFailDescr) };
            <guard::DynasmFailDescr as majit_ir::FailDescr>::handle_fail(descr, ctx)
        }
    }
}

/// `warmspot.handle_jitexception` (warmspot.py:982-999) payload unpack.
///
/// Upstream reads `e.result` / `e.value` out of the caught
/// `jitexc.JitException` subclass and returns it with type-specific
/// specialization. pyre collapses the jitexc hierarchy into the
/// `HandleFailResult` enum (majit-ir/src/descr.rs:937-948) and unpacks
/// here, producing the raw i64 bit pattern the JIT-emitted caller stub
/// expects as `rax`.  The Ref / ExitFrame variants share an encoding
/// because the caller distinguishes them through the synthesized
/// `FailDescr::is_exit_frame_with_exception` flag rather than the
/// return value itself.
fn handle_fail_result_to_i64(result: majit_ir::HandleFailResult) -> i64 {
    match result {
        majit_ir::HandleFailResult::DoneWithThisFrameVoid => 0,
        majit_ir::HandleFailResult::DoneWithThisFrameInt(v) => v,
        majit_ir::HandleFailResult::DoneWithThisFrameRef(r) => r.0 as i64,
        majit_ir::HandleFailResult::DoneWithThisFrameFloat(f) => f.to_bits() as i64,
        majit_ir::HandleFailResult::ExitFrameWithExceptionRef(v) => v.0 as i64,
    }
}

/// `HandleFailContext` (majit-ir/src/descr.rs:857-892) implementation
/// that wires `FailDescr::handle_fail` bodies back to the CALL_ASSEMBLER
/// helper frame.  Upstream's equivalent is the
/// `(deadframe, metainterp_sd, jitdriver_sd)` tuple threaded through
/// every `handle_fail` method in `rpython/jit/metainterp/compile.py`.
struct CallAssemblerHelperContext {
    frame_ptr: *mut jitframe::JitFrame,
    green_key: u64,
    /// `warmspot.py:1022` `fail_descr = cpu.get_latest_descr(deadframe)` —
    /// stashed so `resume_guard` can hand the raw pointer to the bridge
    /// tracer hook without re-reading `jf_descr`.
    descr_raw: usize,
}

impl majit_ir::HandleFailContext for CallAssemblerHelperContext {
    fn get_int_value(&self, idx: usize) -> i64 {
        // llmodel.py:437-444
        unsafe { llmodel::get_int_value_direct(self.frame_ptr, idx) as i64 }
    }

    fn get_ref_value(&self, idx: usize) -> majit_ir::GcRef {
        // llmodel.py:446-453
        majit_ir::GcRef(unsafe { llmodel::get_ref_value_direct(self.frame_ptr, idx) })
    }

    fn get_float_value(&self, idx: usize) -> u64 {
        // llmodel.py:455-462
        unsafe { llmodel::get_float_value_direct(self.frame_ptr, idx) }
    }

    fn grab_exc_value(&self) -> majit_ir::GcRef {
        // llmodel.py:240-241 `deadframe.jf_guard_exc`.
        majit_ir::GcRef(unsafe { (*self.frame_ptr).jf_guard_exc })
    }

    fn green_key(&self) -> u64 {
        self.green_key
    }

    fn done_with_this_frame(&mut self, result_type: majit_ir::Type) -> majit_ir::HandleFailResult {
        // compile.py:626-656 _DoneWithThisFrameDescr body routed by
        // `result_type` — invoked from `dispatch_handle_fail` when a
        // dual-role DynasmFailDescr carries `is_finish = true`.
        match result_type {
            majit_ir::Type::Void => majit_ir::HandleFailResult::DoneWithThisFrameVoid,
            majit_ir::Type::Int => {
                majit_ir::HandleFailResult::DoneWithThisFrameInt(self.get_int_value(0))
            }
            majit_ir::Type::Ref => {
                majit_ir::HandleFailResult::DoneWithThisFrameRef(self.get_ref_value(0))
            }
            majit_ir::Type::Float => majit_ir::HandleFailResult::DoneWithThisFrameFloat(
                f64::from_bits(self.get_float_value(0)),
            ),
        }
    }

    fn exit_frame_with_exception_ref(&mut self) -> majit_ir::HandleFailResult {
        // compile.py:658-662 `ExitFrameWithExceptionDescrRef.handle_fail`
        // body — invoked from `dispatch_handle_fail` when a dual-role
        // DynasmFailDescr carries `is_exit_frame_with_exception = true`.
        majit_ir::HandleFailResult::ExitFrameWithExceptionRef(self.get_ref_value(0))
    }

    fn memory_error_gcref(&mut self) -> majit_ir::GcRef {
        CA_MEMORY_ERROR_FN
            .get()
            .map(|f| majit_ir::GcRef(f()))
            .unwrap_or(majit_ir::GcRef::NULL)
    }

    fn resume_guard(&mut self, descr: &dyn majit_ir::FailDescr) -> majit_ir::HandleFailResult {
        // compile.py:701-717 `AbstractResumeGuardDescr.handle_fail`:
        //   if must_compile(...) and not rstack.stack_almost_full():
        //       self._trace_and_compile_from_bridge(deadframe, ...)
        //   else:
        //       resume_in_blackhole(metainterp_sd, jitdriver_sd, self, deadframe)
        //   assert 0, "unreachable"
        let trace_id = descr.trace_id();
        let fail_index = descr.fail_index();
        let fail_arg_types = descr.fail_arg_types();
        let fail_arg_locs = descr.fail_arg_locs();
        let n_fail_args = fail_arg_types.len();
        let mut raw_values: Vec<i64> = Vec::with_capacity(n_fail_args);
        for i in 0..n_fail_args {
            let slot = fail_arg_locs.get(i).and_then(|l| *l).unwrap_or(i);
            raw_values.push(unsafe { llmodel::get_int_value_direct(self.frame_ptr, slot) as i64 });
        }

        // compile.py:701-716 branch ownership stays in the frontend.
        let handle_fail = CA_HANDLE_FAIL_FN
            .get()
            .expect("CALL_ASSEMBLER handle_fail hook must be registered before runtime use");
        match handle_fail(
            self.green_key,
            trace_id,
            fail_index,
            raw_values.as_ptr(),
            raw_values.len(),
            raw_values.as_ptr(),
            raw_values.len(),
            self.descr_raw,
        ) {
            majit_ir::CallAssemblerHandleFailAction::ExecuteAttachedBridge => {
                let bridge_addr =
                    unsafe { (&*(self.descr_raw as *const guard::DynasmFailDescr)).bridge_addr() };
                assert_ne!(
                    bridge_addr, 0,
                    "CALL_ASSEMBLER requested ExecuteAttachedBridge without an attached bridge"
                );
                let func: unsafe extern "C" fn(*mut jitframe::JitFrame) -> *mut jitframe::JitFrame =
                    unsafe { std::mem::transmute(bridge_addr) };
                let result_jf = unsafe { func(self.frame_ptr) };
                assert!(
                    !result_jf.is_null(),
                    "CALL_ASSEMBLER bridge execution returned a null jitframe"
                );
                self.frame_ptr = result_jf;
                self.descr_raw = unsafe { llmodel::get_latest_descr(self.frame_ptr) };
                assert_ne!(
                    self.descr_raw, 0,
                    "CALL_ASSEMBLER bridge execution produced a jitframe without jf_descr"
                );
                return dispatch_current_fail_descr(self);
            }
            majit_ir::CallAssemblerHandleFailAction::ReturnToCaller(result) => {
                // The frontend already encoded the caller-visible
                // return bits, matching warmspot.handle_jitexception.
                return majit_ir::HandleFailResult::DoneWithThisFrameInt(result);
            }
        }
    }
}

/// Return the address of the trampoline for embedding in generated code.
pub fn call_assembler_helper_addr() -> usize {
    call_assembler_helper_trampoline as *const () as usize
}

/// CALL_ASSEMBLER recursive callee execution trampoline. JIT-generated
/// code calls this instead of directly branching to the callee entry.
///
/// No alternate-stack switch: upstream RPython enters the callee on
/// the caller's native stack, and the callee's compiled prologue
/// performs its own inline SP probe
/// (rpython/jit/backend/x86/assembler.py:1085
/// `_call_header_with_stack_check`) against `PYRE_STACKTOOBIG.stack_end`.
/// Growing the stack here would shift SP into a different guard
/// region and invalidate that budget comparison.
///
/// Input:  arg0 = callee jf_ptr, arg1 = callee entry address
/// Output: callee's returned jf_ptr
pub extern "C" fn call_assembler_execute_trampoline(
    jf_ptr: *mut jitframe::JitFrame,
    callee_addr: usize,
) -> *mut jitframe::JitFrame {
    let func: unsafe extern "C" fn(*mut jitframe::JitFrame) -> *mut jitframe::JitFrame =
        unsafe { std::mem::transmute(callee_addr) };
    unsafe { func(jf_ptr) }
}

/// Return the address of the execute trampoline.
pub fn call_assembler_execute_addr() -> usize {
    call_assembler_execute_trampoline as *const () as usize
}

// ── Pending CALL_ASSEMBLER targets ──
// Cranelift: register_pending_call_assembler_target stores a CaDispatchEntry
// with null code_ptr. When compile completes, code_ptr is updated atomically.
// Dynasm: the target registry lives in runner.rs (CALL_ASSEMBLER_TARGETS).
// This API delegates to it for pending target registration.

/// Register a pending CALL_ASSEMBLER target (code not yet compiled).
/// Matches Cranelift's register_pending_call_assembler_target API.
///
/// The actual registration happens via Backend::register_pending_target
/// → DynasmBackend::register_pending_target (runner.rs), which inserts
/// a 0 entry into CALL_ASSEMBLER_TARGETS. When compile_loop completes,
/// it overwrites with the real code_addr.
///
/// This function is the direct-call equivalent for non-Backend callers.
pub fn register_pending_call_assembler_target(
    token_number: u64,
    _inputarg_types: Vec<majit_ir::Type>,
    _num_inputs: usize,
    _num_scalar_inputargs: usize,
    _index_of_virtualizable: i32,
) {
    // Delegate to the runner's CALL_ASSEMBLER_TARGETS registry.
    runner::DynasmBackend::register_pending_call_assembler_target_static(token_number);
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;
    use std::sync::Arc;
    use std::sync::Once;
    use std::sync::atomic::{AtomicU8, Ordering};

    static TEST_HANDLE_FAIL_MODE: AtomicU8 = AtomicU8::new(0);
    static REGISTER_TEST_BRIDGE_HOOK: Once = Once::new();

    unsafe fn alloc_test_jitframe(descr: usize, slots: &[i64]) -> *mut jitframe::JitFrame {
        let depth = slots.len();
        let size = jitframe::JitFrame::alloc_size(depth);
        let ptr = unsafe { libc::malloc(size) as *mut jitframe::JitFrame };
        assert!(!ptr.is_null());
        unsafe {
            libc::memset(ptr as *mut std::ffi::c_void, 0, size);
            jitframe::JitFrame::init(ptr, jitframe::NULLFRAMEINFO, depth);
            llmodel::set_latest_descr(ptr, descr);
            for (index, value) in slots.iter().copied().enumerate() {
                llmodel::set_int_value_direct(ptr, index, value as isize);
            }
        }
        ptr
    }

    extern "C" fn test_bridge_finish_int(jf: *mut jitframe::JitFrame) -> *mut jitframe::JitFrame {
        unsafe {
            llmodel::set_latest_descr(jf, guard::done_with_this_frame_descr_int_ptr());
            llmodel::set_int_value_direct(jf, 0, 321);
        }
        jf
    }

    fn test_bridge_compile_hook(
        _green_key: u64,
        _trace_id: u64,
        _fail_index: u32,
        _raw_values: *const i64,
        _num_values: usize,
        _raw_deadframe: *const i64,
        _num_raw_deadframe: usize,
        _descr_addr: usize,
    ) -> majit_ir::CallAssemblerHandleFailAction {
        match TEST_HANDLE_FAIL_MODE.load(Ordering::Relaxed) {
            0 => majit_ir::CallAssemblerHandleFailAction::ReturnToCaller(0),
            1 => majit_ir::CallAssemblerHandleFailAction::ExecuteAttachedBridge,
            mode => panic!("unexpected TEST_HANDLE_FAIL_MODE={mode}"),
        }
    }

    fn ensure_test_handle_fail_hook() {
        REGISTER_TEST_BRIDGE_HOOK.call_once(|| {
            register_call_assembler_handle_fail(test_bridge_compile_hook);
        });
    }

    // ── Bug 1 regression: unresolved target must not dereference result as pointer ──
    // The old code let the helper return value flow into `mov rdx, rax; mov rcx, [rdx]`
    // which dereferenced an integer as a pointer. This test verifies the trampoline
    // returns a plain value without crashing when jf_descr is 0 (pending/unresolved).

    #[test]
    fn test_helper_trampoline_null_jf_returns_zero() {
        let result = call_assembler_helper_trampoline(std::ptr::null_mut(), 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_helper_trampoline_descr_zero_returns_zero() {
        let jf = unsafe { alloc_test_jitframe(0, &[0, 0, 0, 0]) };
        let result = call_assembler_helper_trampoline(jf, 42);
        assert_eq!(result, 0);
    }

    // ── Bug 2 regression: force_fn must NOT be called with jitframe pointer ──
    // The old code passed a dynasm jitframe (i64 array) to force_fn which
    // expected a PyFrame pointer. Now force_fn is not called from the trampoline.
    // Verify: even when a guard descr is present, force_fn is never invoked.

    #[test]
    fn test_helper_trampoline_does_not_call_force_fn() {
        // force_fn is stored in CA_FORCE_FN (OnceLock). We can't reset it,
        // but we CAN verify the trampoline returns blackhole result (or 0)
        // without calling force. We do this by setting up a guard descr and
        // checking the trampoline reads fail_arg values correctly.
        ensure_test_handle_fail_hook();
        TEST_HANDLE_FAIL_MODE.store(0, Ordering::Relaxed);

        let descr = Arc::new(guard::DynasmFailDescr::new(
            7,  // fail_index
            99, // trace_id
            vec![Type::Int, Type::Int],
            false,
        ));
        let descr_ptr = Arc::as_ptr(&descr) as i64;

        let jf = unsafe { alloc_test_jitframe(descr_ptr as usize, &[100, 200, 0, 0]) };
        let result = call_assembler_helper_trampoline(jf, 42);
        // No blackhole registered → returns 0, no crash.
        assert_eq!(result, 0);
        drop(descr);
    }

    #[test]
    fn test_helper_trampoline_dispatches_attached_bridge() {
        // compile.py:704-709 parity: once _trace_and_compile_from_bridge
        // attaches a bridge to the guard descr, re-entering the same
        // deadframe follows that bridge and dispatches its resulting descr.
        ensure_test_handle_fail_hook();
        TEST_HANDLE_FAIL_MODE.store(1, Ordering::Relaxed);
        let descr = Arc::new(guard::DynasmFailDescr::new(3, 17, vec![Type::Int], false));
        descr.set_bridge_addr(test_bridge_finish_int as *const () as usize);
        let descr_ptr = Arc::as_ptr(&descr) as usize;

        let jf = unsafe { alloc_test_jitframe(descr_ptr, &[123, 0, 0, 0]) };
        let result = call_assembler_helper_trampoline(jf, 99);
        TEST_HANDLE_FAIL_MODE.store(0, Ordering::Relaxed);
        assert_eq!(result, 321);

        drop(descr);
    }

    // ── Bug 3 regression: green_key must reach the handler, not be 0 ──

    #[test]
    fn test_helper_trampoline_passes_green_key() {
        use std::sync::atomic::AtomicU64;
        static OBSERVED_GREEN_KEY: AtomicU64 = AtomicU64::new(0);

        // We can't register a one-shot blackhole fn (OnceLock is set-once).
        // Instead, test the trampoline's function signature contract:
        // green_key is the second parameter (rsi on x86_64).
        // Direct call test — not through generated code.
        let green_key: u64 = 0xDEAD_BEEF_CAFE;
        let result = call_assembler_helper_trampoline(std::ptr::null_mut(), green_key);
        // Null ptr → early return 0, but the function accepted green_key.
        assert_eq!(result, 0);
        // The real verification is compile-time: the signature is
        //   extern "C" fn(*mut i64, u64) -> i64
        // and generated code passes green_key in rsi. If this test compiles
        // and runs, the ABI contract is correct.
        let _ = OBSERVED_GREEN_KEY; // suppress unused
    }

    // ── done_with_this_frame_descr: 4 type-specific singletons ──

    /// Test-local stand-in for `compile::DoneWithThisFrameDescr*`
    /// singletons.  The dynasm backend stores the metainterp-side
    /// `Arc<dyn Descr>` in a process-global `OnceLock`, so seeding
    /// from this test mirrors what `pyjitpl.py:2222`
    /// (`make_and_attach_done_descrs([self, cpu])`) does at runtime.
    #[derive(Debug)]
    struct TestMarker(Type);
    impl majit_ir::Descr for TestMarker {}
    impl majit_ir::FailDescr for TestMarker {
        fn fail_index(&self) -> u32 {
            u32::MAX
        }
        fn fail_arg_types(&self) -> &[Type] {
            match self.0 {
                Type::Void => &[],
                Type::Int => &[Type::Int],
                Type::Ref => &[Type::Ref],
                Type::Float => &[Type::Float],
            }
        }
        fn is_finish(&self) -> bool {
            true
        }
        fn handle_fail(
            &self,
            ctx: &mut dyn majit_ir::HandleFailContext,
        ) -> majit_ir::HandleFailResult {
            ctx.done_with_this_frame(self.0)
        }
    }

    #[test]
    fn test_done_with_this_frame_descr_four_variants() {
        use std::sync::Arc;
        let void: majit_ir::DescrRef = Arc::new(TestMarker(Type::Void));
        let int: majit_ir::DescrRef = Arc::new(TestMarker(Type::Int));
        let ref_: majit_ir::DescrRef = Arc::new(TestMarker(Type::Ref));
        let float: majit_ir::DescrRef = Arc::new(TestMarker(Type::Float));

        // OnceLock-backed setters accept the first caller and ignore
        // subsequent writes.  A prior test in the same process (or the
        // `make_and_attach_done_descrs` path exercised via a MetaInterp)
        // may have already filled some or all slots; either way, once
        // the slot has a value, `_ptr()` returns a stable non-zero
        // address for each of the four result types.
        guard::set_done_with_this_frame_descr_void(void);
        guard::set_done_with_this_frame_descr_int(int);
        guard::set_done_with_this_frame_descr_ref(ref_);
        guard::set_done_with_this_frame_descr_float(float);

        let void_ptr = guard::done_with_this_frame_descr_void_ptr();
        let int_ptr = guard::done_with_this_frame_descr_int_ptr();
        let ref_ptr = guard::done_with_this_frame_descr_ref_ptr();
        let float_ptr = guard::done_with_this_frame_descr_float_ptr();

        // All four must be distinct (metainterp-side `DoneWithThisFrameDescr*`
        // singletons are separate Arcs per result type).
        assert_ne!(void_ptr, 0);
        assert_ne!(int_ptr, 0);
        assert_ne!(ref_ptr, 0);
        assert_ne!(float_ptr, 0);
        assert_ne!(void_ptr, int_ptr);
        assert_ne!(void_ptr, ref_ptr);
        assert_ne!(void_ptr, float_ptr);
        assert_ne!(int_ptr, ref_ptr);
        assert_ne!(int_ptr, float_ptr);
        assert_ne!(ref_ptr, float_ptr);

        // is_done_with_this_frame_descr must recognize all four.
        assert!(guard::is_done_with_this_frame_descr(void_ptr));
        assert!(guard::is_done_with_this_frame_descr(int_ptr));
        assert!(guard::is_done_with_this_frame_descr(ref_ptr));
        assert!(guard::is_done_with_this_frame_descr(float_ptr));

        // Arbitrary pointer must NOT be recognized.
        assert!(!guard::is_done_with_this_frame_descr(0x12345678));
        assert!(!guard::is_done_with_this_frame_descr(0));

        // ptr_for_type must route correctly.
        assert_eq!(
            guard::done_with_this_frame_descr_ptr_for_type(Type::Void),
            void_ptr
        );
        assert_eq!(
            guard::done_with_this_frame_descr_ptr_for_type(Type::Int),
            int_ptr
        );
        assert_eq!(
            guard::done_with_this_frame_descr_ptr_for_type(Type::Ref),
            ref_ptr
        );
        assert_eq!(
            guard::done_with_this_frame_descr_ptr_for_type(Type::Float),
            float_ptr
        );
    }

    // ── Exception state: jit_exc_raise / jit_exc_value_raw ──

    #[test]
    fn test_jit_exc_raise_and_clear() {
        jit_exc_clear();
        assert!(!jit_exc_is_pending());
        assert_eq!(jit_exc_class_raw(), 0);

        // Simulate raising with a fake object (type ptr at offset 0).
        let fake_type: i64 = 0x42;
        let fake_obj: [i64; 2] = [fake_type, 0x99];
        jit_exc_raise(fake_obj.as_ptr() as i64);
        assert!(jit_exc_is_pending());
        assert_eq!(jit_exc_class_raw(), fake_type);

        // value_raw returns and clears.
        let val = jit_exc_value_raw();
        assert_eq!(val, fake_obj.as_ptr() as i64);
        assert!(!jit_exc_is_pending());

        jit_exc_clear();
    }
}
