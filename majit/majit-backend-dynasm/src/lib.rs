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
