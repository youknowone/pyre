//! Port of `rpython/jit/backend/x86/regalloc.py` — arch-specific
//! register configuration that the shared
//! `majit-backend-dynasm/src/regalloc.rs` (mirroring
//! `rpython/jit/backend/llsupport/regalloc.py`) reads at construction
//! time.
//!
//! Upstream splits per arch directory; pyre matches that split here.

use crate::regloc::{
    EAX, EBP, EBX, ECX, EDI, EDX, ESI, R8, R9, R10, R12, R13, R14, R15, RegLoc, XMM0, XMM1, XMM2,
    XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
};

/// x86/regalloc.py X86_64_RegisterManager.all_regs — the GPR allocation
/// pool.  Order chosen to prefer caller-save first (popped from end).
pub fn all_core_regs() -> Vec<RegLoc> {
    vec![
        ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
    ]
}

/// x86/regalloc.py: caller-save GPR list (registers spilled around
/// calls per System V AMD64 / Win64 ABI).
pub fn save_around_call_core_regs() -> Vec<RegLoc> {
    vec![EAX, ECX, EDX, ESI, EDI, R8, R9, R10]
}

/// x86/regalloc.py X86_64_XMMRegisterManager.all_regs — XMM allocation
/// pool (xmm15 reserved as scratch).
pub fn all_float_regs() -> Vec<RegLoc> {
    vec![
        XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13,
        XMM14,
    ]
}

/// `frame_reg` on x86_64 is RBP (callee-save), holding the JitFrame
/// pointer for the duration of the JIT-compiled procedure.
pub fn frame_reg() -> RegLoc {
    EBP
}

/// `call_result_gpr` — x86_64 AMD64 ABI return register.
pub fn call_result_gpr() -> RegLoc {
    EAX
}

/// `call_result_fpr` — x86_64 AMD64 ABI XMM return register.
pub fn call_result_fpr() -> RegLoc {
    XMM0
}

/// `core_reg_index` returns the position of `reg` inside the canonical
/// `all_core_regs` ordering for gcmap and jitframe slot tables.
pub fn core_reg_index(reg: RegLoc) -> Option<usize> {
    all_core_regs()
        .iter()
        .position(|candidate| *candidate == reg)
}
