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

use crate::regloc::RegLoc;

/// aarch64/registers.py:14
///   `all_regs = registers[:14] + [x19, x20] #, x21, x22]`
///
/// pyre activates upstream's commented-out `, x21, x22` extension.
/// Empirically required: with only x19/x20 the four loop-carried
/// values in fib_loop (n, a, b, i) cannot fit in callee-save and
/// `longest_free_reg`'s caller-save fallback then selects an x-reg
/// the inner int_add helper clobbers.  Stays a strict subset of
/// AAPCS64 x19..x28 and matches `JITFRAME_FIXED_SIZE` in arch.rs +
/// `_call_header` / `_call_footer` stp/ldp pairs in
/// `aarch64/assembler.rs`.
pub fn all_core_regs() -> Vec<RegLoc> {
    vec![
        RegLoc::new(0, false),
        RegLoc::new(1, false),
        RegLoc::new(2, false),
        RegLoc::new(3, false),
        RegLoc::new(4, false),
        RegLoc::new(5, false),
        RegLoc::new(6, false),
        RegLoc::new(7, false),
        RegLoc::new(8, false),
        RegLoc::new(9, false),
        RegLoc::new(10, false),
        RegLoc::new(11, false),
        RegLoc::new(12, false),
        RegLoc::new(13, false),
        RegLoc::new(19, false),
        RegLoc::new(20, false),
        RegLoc::new(21, false),
        RegLoc::new(22, false),
    ]
}

/// aarch64/registers.py:43
///   `caller_resp = argument_regs + [x8, x9, x10, x11, x12, x13]`
///
/// In RPython, `save_around_call_regs` is the AAPCS64 caller-saved
/// subset (x0..x13) — the regs whose contents must be assumed
/// clobbered after a `bl`.  Mirrored verbatim here.
pub fn save_around_call_core_regs() -> Vec<RegLoc> {
    vec![
        RegLoc::new(0, false),
        RegLoc::new(1, false),
        RegLoc::new(2, false),
        RegLoc::new(3, false),
        RegLoc::new(4, false),
        RegLoc::new(5, false),
        RegLoc::new(6, false),
        RegLoc::new(7, false),
        RegLoc::new(8, false),
        RegLoc::new(9, false),
        RegLoc::new(10, false),
        RegLoc::new(11, false),
        RegLoc::new(12, false),
        RegLoc::new(13, false),
    ]
}

/// aarch64/registers.py: `all_vfp_regs = vfpregisters[:8]`.  Pyre's
/// VFP allocator pool stays at the upstream cap of 8 (d0..d7).
pub fn all_float_regs() -> Vec<RegLoc> {
    vec![
        RegLoc::new(0, true),
        RegLoc::new(1, true),
        RegLoc::new(2, true),
        RegLoc::new(3, true),
        RegLoc::new(4, true),
        RegLoc::new(5, true),
        RegLoc::new(6, true),
        RegLoc::new(7, true),
    ]
}

/// aarch64/registers.py:18 `fp = x29`.  RPython's frame-pointer
/// register on AAPCS64.
pub fn frame_reg() -> RegLoc {
    RegLoc::new(29, false)
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
/// pyre's aarch64 mapping (kept in sync with `all_core_regs`):
///   x0..x13 → 0..13, x19 → 14, x20 → 15, x21 → 16, x22 → 17.
pub fn core_reg_index(reg: RegLoc) -> Option<usize> {
    match reg.value {
        0..=13 => Some(reg.value as usize),
        19 => Some(14),
        20 => Some(15),
        21 => Some(16),
        22 => Some(17),
        _ => None,
    }
}

/// aarch64/registers.py:21 `ip1 = x17` — scratch register reserved for
/// large-immediate stitching (movz/movk sequences).  Available to the
/// shared regalloc base as a tmp slot via the per-arch re-export.
pub const IP1: RegLoc = RegLoc {
    value: 17,
    is_xmm: false,
};

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
