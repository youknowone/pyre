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
/// On Win64, `all_regs.remove(r13)` runs at class-construction time, so
/// the pool length drops 13 → 12 and `all_reg_indexes` (which is built
/// from the post-removal `all_regs`) shifts R14/R15 down by one slot.
#[cfg(not(target_os = "windows"))]
pub const ALL_CORE_REGS: &[RegLoc] = &[
    ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
];
#[cfg(target_os = "windows")]
pub const ALL_CORE_REGS: &[RegLoc] = &[ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R14, R15];

pub fn all_core_regs() -> Vec<RegLoc> {
    ALL_CORE_REGS.to_vec()
}

/// x86/regalloc.py: caller-save GPR list (registers spilled around
/// calls per System V AMD64 / Win64 ABI).
#[cfg(not(target_os = "windows"))]
pub const SAVE_AROUND_CALL_CORE_REGS: &[RegLoc] = &[EAX, ECX, EDX, ESI, EDI, R8, R9, R10];
#[cfg(target_os = "windows")]
pub const SAVE_AROUND_CALL_CORE_REGS: &[RegLoc] = &[EAX, ECX, EDX, R8, R9, R10];

pub fn save_around_call_core_regs() -> Vec<RegLoc> {
    SAVE_AROUND_CALL_CORE_REGS.to_vec()
}

/// x86/regalloc.py X86_64_XMMRegisterManager.all_regs — XMM allocation
/// pool.  On non-Win64 xmm15 is reserved as scratch.  On Win64 PyPy
/// uses a separate `X86_64_WIN_XMMRegisterManager` (regalloc.py:128)
/// with only `[xmm0..xmm4]`, reserving xmm5 as scratch and leaving
/// xmm6..xmm15 callee-save untouched so the JIT prologue/epilogue
/// does not need to save them.  `save_around_call_regs = all_regs`
/// for XMMs across both ABIs.
#[cfg(not(target_os = "windows"))]
pub const ALL_FLOAT_REGS: &[RegLoc] = &[
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
];
#[cfg(target_os = "windows")]
pub const ALL_FLOAT_REGS: &[RegLoc] = &[XMM0, XMM1, XMM2, XMM3, XMM4];

pub fn all_float_regs() -> Vec<RegLoc> {
    ALL_FLOAT_REGS.to_vec()
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

/// `core_reg_index` returns the canonical jitframe slot for `reg`.
///
/// regalloc.py `all_reg_indexes` is built from the post-Win64-removal
/// `all_regs` list: on Win64 `all_regs.remove(r13)` runs first, so
/// the per-reg index table records R14 at slot 10 and R15 at slot 11
/// (instead of 11/12 on non-Win64).  Mirror that here by looking up
/// the position in `ALL_CORE_REGS`, which is itself Win64-aware.
///
/// `_push_all_regs_to_frame`, `_pop_all_regs_from_frame`, `get_gcmap`
/// must all consume slots through this function — never via iteration
/// position inside `all_core_regs()` — so save_regs_label, the gcmap
/// bitmap, and the post-call pop stay in agreement (they happen to be
/// equal once both sides walk the same Win64-aware list, but keeping
/// the lookup central guards against a future helper computing slots
/// off a non-canonical iteration).
pub fn core_reg_index(reg: RegLoc) -> Option<usize> {
    ALL_CORE_REGS.iter().position(|candidate| *candidate == reg)
}

/// x86/regalloc.py:1013 consider_call_malloc_nursery:
///   `spill_or_move_registers_before_call([ecx, edx])`
///   `force_allocate_reg(op, selected_reg=ecx)`        → result
///   `force_allocate_reg(tmp_box, selected_reg=edx)`   → temp
/// reghint.py:123 consider_call_malloc_nursery:
///   `longevity.fixed_register(position, ecx, op)`
///   `longevity.fixed_register(position, edx)`
pub const MALLOC_NURSERY_CLOBBER: [RegLoc; 2] = [ECX, EDX];

/// x86_64: result register after the nursery bump (ecx per
/// regalloc.py:1021).
pub const MALLOC_NURSERY_RESULT: RegLoc = ECX;
