//! Port of `rpython/jit/backend/aarch64/registers.py`.
//!
//! Register lists and scratch-register assignments for the aarch64
//! backend.  Upstream keeps these definitions separate from
//! `aarch64/regalloc.py`; pyre mirrors that split here.

use crate::regloc::RegLoc;

/// registers.py:13 `all_vfp_regs = vfpregisters[:8]`
pub const ALL_VFP_REGS: [RegLoc; 8] = [
    RegLoc::new(0, true),
    RegLoc::new(1, true),
    RegLoc::new(2, true),
    RegLoc::new(3, true),
    RegLoc::new(4, true),
    RegLoc::new(5, true),
    RegLoc::new(6, true),
    RegLoc::new(7, true),
];

/// registers.py:14 `all_regs = registers[:14] + [x19, x20] #, x21, x22]`
pub const ALL_REGS: [RegLoc; 16] = [
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
];

/// registers.py:17 `fp = x29`
pub const FP: RegLoc = RegLoc::new(29, false);

/// registers.py:21 `ip1 = x17`
pub const IP1: RegLoc = RegLoc::new(17, false);

/// registers.py:26 `callee_saved_registers = [x19, x20] # , x21, x22]`
pub const CALLEE_SAVED_REGISTERS: [RegLoc; 2] = [RegLoc::new(19, false), RegLoc::new(20, false)];

/// registers.py:27 `vfp_argument_regs = caller_vfp_resp = all_vfp_regs[:8]`
pub const VFP_ARGUMENT_REGS: [RegLoc; 8] = ALL_VFP_REGS;

/// registers.py:34 `argument_regs = [x0, x1, x2, x3, x4, x5, x6, x7]`
pub const ARGUMENT_REGS: [RegLoc; 8] = [
    RegLoc::new(0, false),
    RegLoc::new(1, false),
    RegLoc::new(2, false),
    RegLoc::new(3, false),
    RegLoc::new(4, false),
    RegLoc::new(5, false),
    RegLoc::new(6, false),
    RegLoc::new(7, false),
];

/// registers.py:35 `callee_resp = [x19, x20] # ,x21, x22]`
pub const CALLEE_RESP: [RegLoc; 2] = CALLEE_SAVED_REGISTERS;

/// registers.py:36 `caller_resp = argument_regs + [x8, x9, x10, x11, x12, x13]`
pub const CALLER_RESP: [RegLoc; 14] = [
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
];
