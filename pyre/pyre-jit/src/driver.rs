//! JIT driver for pyre — manages trace compilation and execution.
//!
//! This module is the Rust equivalent of RPython's jit/metainterp
//! integration layer. It connects the interpreter (pyre-interpreter) with
//! the JIT compiler (majit) through auto-generated tracing code.
//!
//! Key principle: pyre-interpreter has zero JIT dependencies.
//! All JIT logic lives here in pyre-jit.

// Re-export the auto-generated tracing functions
pub use crate::{
    trace_box_int, trace_int_binop, trace_int_binop_ovf, trace_int_compare, trace_unbox_int,
};
