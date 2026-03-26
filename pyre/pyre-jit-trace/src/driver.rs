//! JIT driver access from pyre-jit-trace.
//!
//! make_green_key is a pure function. driver_pair() is accessed through
//! callbacks since the JIT_DRIVER TLS lives in pyre-jit/eval.rs.

/// RPython green_key = (pycode, next_instr).
/// Each (code, pc) pair has independent warmup counter and compiled loop.
#[inline(always)]
pub fn make_green_key(code_ptr: *const pyre_bytecode::CodeObject, pc: usize) -> u64 {
    (code_ptr as u64).wrapping_mul(1000003) ^ (pc as u64)
}
