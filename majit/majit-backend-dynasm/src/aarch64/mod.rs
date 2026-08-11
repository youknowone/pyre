/// rpython/jit/backend/aarch64/ parity: aarch64-specific backend.
///
/// RPython class hierarchy:
///   BaseAssembler (llsupport/assembler.py)
///     └── ResOpAssembler (aarch64/opassembler.py)
///           └── AssemblerARM64 (aarch64/assembler.py)
pub mod arch;
#[expect(
    clippy::useless_conversion,
    clippy::unnecessary_cast,
    reason = "dynasm's AArch64 operand grammar expands register expressions through Into-based macro matchers; removing these apparent identity conversions corrupts the generated macro token stream"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the assembler constructor keeps RPython's explicit trace, register-allocation, GC, and operation inputs visible; wrapping them would obscure the audited backend construction boundary"
)]
pub mod assembler;
#[expect(
    clippy::useless_conversion,
    reason = "dynasm's AArch64 operand grammar expands register expressions through Into-based macro matchers; removing these apparent identity conversions corrupts the generated macro token stream"
)]
pub mod codebuilder;
pub mod cpu_ext;
#[expect(
    clippy::useless_conversion,
    reason = "dynasm's AArch64 operand grammar expands register expressions through Into-based macro matchers; removing these apparent identity conversions corrupts the generated macro token stream"
)]
mod opassembler;
pub mod regalloc;
pub mod registers;
