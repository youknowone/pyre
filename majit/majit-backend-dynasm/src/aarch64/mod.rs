/// rpython/jit/backend/aarch64/ parity: aarch64-specific backend.
///
/// RPython class hierarchy:
///   BaseAssembler (llsupport/assembler.py)
///     └── ResOpAssembler (aarch64/opassembler.py)
///           └── AssemblerARM64 (aarch64/assembler.py)
pub mod arch;
pub mod assembler;
pub mod cpu_ext;
mod opassembler;
pub mod regalloc;
pub mod registers;
