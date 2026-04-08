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
