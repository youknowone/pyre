//! Port of `rpython/jit/backend/aarch64/regalloc.py` — arch-specific
//! register configuration that the shared
//! `majit-backend-dynasm/src/regalloc.rs` (mirroring
//! `rpython/jit/backend/llsupport/regalloc.py`) reads at construction
//! time.
//!
//! Upstream splits per arch directory; pyre matches that split here.
//! The aarch64 file is intentionally small — most of the `consider_*`
//! and lifetime logic still lives in the shared base, just like
//! upstream's RPython aarch64 inherits from llsupport.

use crate::aarch64::registers;
use crate::regloc::RegLoc;

/// aarch64/registers.py:14
///   `all_regs = registers[:14] + [x19, x20] #, x21, x22]`
pub fn all_core_regs() -> Vec<RegLoc> {
    registers::ALL_REGS.to_vec()
}

/// aarch64/registers.py:43
///   `caller_resp = argument_regs + [x8, x9, x10, x11, x12, x13]`
///
/// In RPython, `save_around_call_regs` is the AAPCS64 caller-saved
/// subset (x0..x13) — the regs whose contents must be assumed
/// clobbered after a `bl`.  Mirrored verbatim here.
pub fn save_around_call_core_regs() -> Vec<RegLoc> {
    registers::CALLER_RESP.to_vec()
}

/// aarch64/registers.py: `all_vfp_regs = vfpregisters[:8]`.  Pyre's
/// VFP allocator pool stays at the upstream cap of 8 (d0..d7).
pub fn all_float_regs() -> Vec<RegLoc> {
    registers::ALL_VFP_REGS.to_vec()
}

/// aarch64/registers.py:18 `fp = x29`.  RPython's frame-pointer
/// register on AAPCS64.
pub fn frame_reg() -> RegLoc {
    registers::FP
}

/// aarch64/locations.py: `call_result_location` returns x0 for GPR
/// (AAPCS64 first return register).
pub fn call_result_gpr() -> RegLoc {
    RegLoc::new(0, false)
}

/// aarch64/locations.py: `call_result_location` returns d0 for VFP.
pub fn call_result_fpr() -> RegLoc {
    RegLoc::new(0, true)
}

/// `core_reg_index` returns the position of `reg` in the canonical
/// `all_core_regs` list — used by gcmap and jitframe slot tables.
///
/// aarch64 mapping (matches `all_core_regs`):
///   x0..x13 → 0..13, x19 → 14, x20 → 15.
pub fn core_reg_index(reg: RegLoc) -> Option<usize> {
    match reg.value {
        0..=13 => Some(reg.value as usize),
        19 => Some(14),
        20 => Some(15),
        _ => None,
    }
}

/// aarch64/regalloc.py:962 nursery-bump path clobbers `[r.x0, r.x1]`.
/// Exported as a per-arch pair so the shared regalloc base can spill
/// the clobbered set without branching on `cfg!`.
pub const MALLOC_NURSERY_CLOBBER: [RegLoc; 2] = [
    RegLoc {
        value: 0,
        is_xmm: false,
    },
    RegLoc {
        value: 1,
        is_xmm: false,
    },
];

/// aarch64: result register after the nursery bump (x0).  Identical
/// value to MALLOC_NURSERY_CLOBBER[0] but separated to document
/// intent.
pub const MALLOC_NURSERY_RESULT: RegLoc = RegLoc {
    value: 0,
    is_xmm: false,
};
