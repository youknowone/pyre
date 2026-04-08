/// rpython/jit/backend/x86 + aarch64 parity:
/// Direct machine code generation via dynasm-rs with in-place patching.
///
/// Module structure mirrors RPython's backend/x86/:
///   runner.py   → runner.rs  (DynasmBackend — Backend trait impl)
///   assembler.py → assembler.rs (Assembler386 — code generation)
///   regalloc.py → regalloc.rs (RegAlloc — register allocation)
///   regloc.py  → regloc.rs  (register/location types)
///   arch.py    → arch.rs    (architecture constants)
///   callbuilder.py → callbuilder.rs (FFI call ABI)
///   codebuf.py → codebuf.rs (code buffer management)
///   jump.py    → jump.rs    (frame layout remapping)
///
/// guard.rs and frame.rs are from compile.py / jitframe.py.
pub mod arch;
pub mod assembler;
pub mod callbuilder;
pub mod codebuf;
pub mod frame;
pub mod guard;
pub mod jump;
pub mod regalloc;
pub mod regloc;
pub mod runner;

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
// Cranelift uses register_call_assembler_blackhole/bridge/force/unbox_int.
// Dynasm provides the same registration API and a C-callable trampoline
// that the generated slow path code calls directly.

use std::sync::OnceLock;

/// Blackhole resume: (green_key, trace_id, fail_index, raw_values, len,
/// typed_outputs, typed_len) → Option<result>
pub type BlackholeFn = fn(u64, u64, u32, *const i64, usize, *const i64, usize) -> Option<i64>;

/// Bridge compilation: (green_key, trace_id, fail_index, raw_values, len,
/// descr_addr) → compiled?
pub type BridgeFn = fn(u64, u64, u32, *const i64, usize, usize) -> bool;

/// Force callee: (callee_frame_ptr) → result
pub type ForceFn = extern "C" fn(i64) -> i64;

/// Unbox int from ref: (raw_ref) → int_value
pub type UnboxIntFn = fn(i64) -> i64;

static CA_BLACKHOLE_FN: OnceLock<BlackholeFn> = OnceLock::new();
static CA_BRIDGE_FN: OnceLock<BridgeFn> = OnceLock::new();
static CA_FORCE_FN: OnceLock<ForceFn> = OnceLock::new();
static CA_UNBOX_INT_FN: OnceLock<UnboxIntFn> = OnceLock::new();

/// Register blackhole resume handler (same API as Cranelift).
pub fn register_call_assembler_blackhole(f: BlackholeFn) {
    CA_BLACKHOLE_FN.set(f).ok();
}

/// Register bridge compilation handler (same API as Cranelift).
pub fn register_call_assembler_bridge(f: BridgeFn) {
    CA_BRIDGE_FN.set(f).ok();
}

/// Register force handler (same API as Cranelift).
pub fn register_call_assembler_force(f: ForceFn) {
    CA_FORCE_FN.set(f).ok();
}

/// Register int unbox handler (same API as Cranelift).
pub fn register_call_assembler_unbox_int(f: UnboxIntFn) {
    CA_UNBOX_INT_FN.set(f).ok();
}

/// assembler.py:345 assembler_helper_adr parity:
/// C-callable trampoline for CALL_ASSEMBLER slow path.
///
/// Called from generated machine code when:
/// (a) callee finished with a guard failure (jf_descr != done_descr), or
/// (b) target was pending/unresolved (jf_descr == 0).
///
/// Dispatch order matches Cranelift's call_assembler_guard_failure:
///   1. bridge (if CA_BRIDGE_FN registered and compiles successfully)
///   2. blackhole resume (if CA_BLACKHOLE_FN registered)
///   3. force (if CA_FORCE_FN registered — re-execute callee in interpreter)
///   4. return 0 (no handler available)
///
/// Input:  rdi/x0 = callee_jf_ptr (heap-allocated jitframe)
/// Output: rax/x0 = result value (0 if void/unhandled)
///
/// Always frees the callee jitframe before returning.
pub extern "C" fn call_assembler_helper_trampoline(callee_jf_ptr: *mut i64) -> i64 {
    if callee_jf_ptr.is_null() {
        return 0;
    }

    // Read jf_descr — raw pointer to DynasmFailDescr (or 0 for pending)
    let descr_raw = unsafe { *callee_jf_ptr } as usize;

    let result = if descr_raw == 0 || guard::is_done_with_this_frame_descr(descr_raw) {
        // Case (b): pending target (jf_descr == 0) or somehow a finish descr
        // leaked here. Fall through to force_fn which re-executes callee
        // in the interpreter.
        //
        // Cranelift parity: call_assembler_fast_path detects null code_ptr,
        // calls force_fn(callee_frame_ptr) with PENDING_FORCE_LOCAL0.
        if let Some(force_fn) = CA_FORCE_FN.get() {
            force_fn(callee_jf_ptr as i64)
        } else {
            0
        }
    } else {
        // Case (a): guard failure. jf_descr points to DynasmFailDescr.
        let descr = unsafe { &*(descr_raw as *const guard::DynasmFailDescr) };
        let trace_id = descr.trace_id;
        let fail_index = descr.fail_index;
        let descr_addr = descr_raw;

        // Read fail_arg values from callee frame: jf_frame[0..n] at jf_ptr[1..]
        let n_fail_args = descr.fail_arg_types.len();
        let mut raw_values: Vec<i64> = Vec::with_capacity(n_fail_args);
        for i in 0..n_fail_args {
            let slot = descr.fail_arg_locs.get(i).and_then(|l| *l).unwrap_or(i);
            raw_values.push(unsafe { *callee_jf_ptr.add(1 + slot) });
        }

        // Step 1: Try bridge compilation (compile.py:701-717 must_compile)
        if let Some(bridge_fn) = CA_BRIDGE_FN.get() {
            let _compiled = bridge_fn(
                0, // green_key (resolved by handler)
                trace_id,
                fail_index,
                raw_values.as_ptr(),
                raw_values.len(),
                descr_addr,
            );
            // If bridge compiled, it will be used on next guard failure.
            // For this invocation, fall through to blackhole.
        }

        // Step 2: Try blackhole resume (resume.py:1312 parity)
        if let Some(blackhole) = CA_BLACKHOLE_FN.get() {
            match blackhole(
                0, // green_key
                trace_id,
                fail_index,
                raw_values.as_ptr(),
                raw_values.len(),
                raw_values.as_ptr(),
                raw_values.len(),
            ) {
                Some(val) => val,
                None => {
                    // Step 3: Force fallback — re-execute callee in interpreter
                    if let Some(force_fn) = CA_FORCE_FN.get() {
                        force_fn(callee_jf_ptr as i64)
                    } else {
                        0
                    }
                }
            }
        } else if let Some(force_fn) = CA_FORCE_FN.get() {
            // No blackhole, try force directly
            force_fn(callee_jf_ptr as i64)
        } else {
            0
        }
    };

    // Free the callee jitframe
    unsafe { libc::free(callee_jf_ptr as *mut std::ffi::c_void) };
    result
}

/// Return the address of the trampoline for embedding in generated code.
pub fn call_assembler_helper_addr() -> usize {
    call_assembler_helper_trampoline as usize
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
) {
    // Delegate to the runner's CALL_ASSEMBLER_TARGETS registry.
    runner::DynasmBackend::register_pending_call_assembler_target_static(token_number);
}
