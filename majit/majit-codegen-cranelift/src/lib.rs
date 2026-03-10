/// Cranelift-based JIT code generation backend for majit.
///
/// This crate implements the `majit_codegen::Backend` trait using Cranelift
/// to translate majit IR traces into native machine code.

pub mod compiler;
pub mod guard;

pub use compiler::CraneliftBackend;
