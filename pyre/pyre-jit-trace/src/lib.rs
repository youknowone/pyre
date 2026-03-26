//! pyre-jit-trace: Trace-time JIT for pyre.
//!
//! This crate contains MIFrame (the meta-interpreter frame) and all
//! trace-time logic. It is compiled as a separate compilation unit
//! from pyre-jit's eval_loop_jit to prevent MIFrame's monomorphization
//! of `execute_opcode_step<E>` from bloating the eval loop's codegen.
