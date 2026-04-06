/// callbuilder.py: FFI call setup/teardown.
///
/// CallBuilder64 — System V AMD64 ABI calling convention.
/// Handles argument passing, stack alignment, shadow space (Win64).
use crate::regloc::*;

/// callbuilder.py:361 CallBuilder64 ��� x86_64 calling convention.
pub struct CallBuilder64 {
    /// Arguments to pass in registers (rdi, rsi, rdx, rcx, r8, r9).
    pub reg_args: Vec<(RegLoc, i64)>,
    /// Arguments to pass on the stack.
    pub stack_args: Vec<i64>,
}

/// System V AMD64 ABI: first 6 integer args in registers.
pub const ARG_REGS: [RegLoc; 6] = [EDI, ESI, EDX, ECX, R8, R9];
/// System V AMD64 ABI: first 8 float args in XMM registers.
pub const FLOAT_ARG_REGS: [RegLoc; 8] = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];

impl CallBuilder64 {
    pub fn new() -> Self {
        CallBuilder64 {
            reg_args: Vec::new(),
            stack_args: Vec::new(),
        }
    }
}
