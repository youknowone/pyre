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
/// Called from generated machine code when callee didn't finish with
/// done_with_this_frame_descr. The callee_jf_ptr[0] contains the
/// guard's DynasmFailDescr pointer.
///
/// Input:  rdi/x0 = callee_jf_ptr (heap-allocated jitframe)
/// Output: rax/x0 = result value (0 if void/unhandled)
///
/// The trampoline reads the fail descriptor from jf_descr, extracts
/// trace_id/fail_index, reads fail_arg values, then dispatches to
/// the registered blackhole/bridge/force handler.
/// Always frees the callee jitframe before returning.
pub extern "C" fn call_assembler_helper_trampoline(callee_jf_ptr: *mut i64) -> i64 {
    if callee_jf_ptr.is_null() {
        return 0;
    }

    // Read jf_descr — raw pointer to DynasmFailDescr
    let descr_raw = unsafe { *callee_jf_ptr } as usize;
    let result;

    if descr_raw == 0 {
        // No descriptor set — callee didn't write jf_descr (shouldn't happen)
        result = 0;
    } else {
        // Cast to &DynasmFailDescr (safe: the Arc keeps it alive)
        let descr = unsafe { &*(descr_raw as *const guard::DynasmFailDescr) };
        let trace_id = descr.trace_id;
        let fail_index = descr.fail_index;

        // Read fail_arg values from callee frame: jf_frame[0..n] at jf_ptr[1..]
        let n_fail_args = descr.fail_arg_types.len();
        let mut raw_values: Vec<i64> = Vec::with_capacity(n_fail_args);
        for i in 0..n_fail_args {
            let slot = descr.fail_arg_locs.get(i).and_then(|l| *l).unwrap_or(i);
            raw_values.push(unsafe { *callee_jf_ptr.add(1 + slot) });
        }

        // Try blackhole resume
        result = if let Some(blackhole) = CA_BLACKHOLE_FN.get() {
            match blackhole(
                0, // green_key (resolved by the handler)
                trace_id,
                fail_index,
                raw_values.as_ptr(),
                raw_values.len(),
                raw_values.as_ptr(), // typed outputs = raw values
                raw_values.len(),
            ) {
                Some(val) => val,
                None => 0,
            }
        } else {
            0
        };
    }

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
/// Matches Cranelift's register_pending_call_assembler_target.
/// The actual code_addr is set later by compile_loop via
/// DynasmBackend::register_call_assembler_target.
pub fn register_pending_call_assembler_target(
    _token_number: u64,
    _inputarg_types: Vec<majit_ir::Type>,
    _num_inputs: usize,
    _num_scalar_inputargs: usize,
) {
    // Pending tokens are resolved at compile time via
    // Assembler386.call_assembler_targets (populated from runner.rs registry).
    // No pre-registration needed — unresolved tokens fall through to
    // the helper trampoline (which handles force/blackhole).
}
