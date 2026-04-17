//! Port of `rpython/jit/backend/aarch64/registers.py`.
//!
//! Register lists and scratch-register assignments for the aarch64
//! backend. Upstream keeps these definitions separate from
//! `aarch64/regalloc.py`; pyre mirrors that split here.

use crate::regloc::RegLoc;

pub const X0: RegLoc = RegLoc::new(0, false);
pub const X1: RegLoc = RegLoc::new(1, false);
pub const X2: RegLoc = RegLoc::new(2, false);
pub const X3: RegLoc = RegLoc::new(3, false);
pub const X4: RegLoc = RegLoc::new(4, false);
pub const X5: RegLoc = RegLoc::new(5, false);
pub const X6: RegLoc = RegLoc::new(6, false);
pub const X7: RegLoc = RegLoc::new(7, false);
pub const X8: RegLoc = RegLoc::new(8, false);
pub const X9: RegLoc = RegLoc::new(9, false);
pub const X10: RegLoc = RegLoc::new(10, false);
pub const X11: RegLoc = RegLoc::new(11, false);
pub const X12: RegLoc = RegLoc::new(12, false);
pub const X13: RegLoc = RegLoc::new(13, false);
pub const X14: RegLoc = RegLoc::new(14, false);
pub const X15: RegLoc = RegLoc::new(15, false);
pub const X16: RegLoc = RegLoc::new(16, false);
pub const X17: RegLoc = RegLoc::new(17, false);
pub const X18: RegLoc = RegLoc::new(18, false);
pub const X19: RegLoc = RegLoc::new(19, false);
pub const X20: RegLoc = RegLoc::new(20, false);
pub const X21: RegLoc = RegLoc::new(21, false);
pub const X22: RegLoc = RegLoc::new(22, false);
pub const X23: RegLoc = RegLoc::new(23, false);
pub const X24: RegLoc = RegLoc::new(24, false);
pub const X25: RegLoc = RegLoc::new(25, false);
pub const X26: RegLoc = RegLoc::new(26, false);
pub const X27: RegLoc = RegLoc::new(27, false);
pub const X28: RegLoc = RegLoc::new(28, false);
pub const X29: RegLoc = RegLoc::new(29, false);
pub const X30: RegLoc = RegLoc::new(30, false);

/// registers.py:6 `registers = [RegisterLocation(i) for i in range(31)]`
pub const REGISTERS: [RegLoc; 31] = [
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20,
    X21, X22, X23, X24, X25, X26, X27, X28, X29, X30,
];

/// registers.py:7 `sp = xzr = ZeroRegister()`
///
/// RPython uses a distinct `ZeroRegister` location type here. The Rust
/// backend only needs the architectural register number, so both aliases
/// point at x31.
pub const SP: RegLoc = RegLoc::new(31, false);
pub const XZR: RegLoc = SP;

pub const D0: RegLoc = RegLoc::new(0, true);
pub const D1: RegLoc = RegLoc::new(1, true);
pub const D2: RegLoc = RegLoc::new(2, true);
pub const D3: RegLoc = RegLoc::new(3, true);
pub const D4: RegLoc = RegLoc::new(4, true);
pub const D5: RegLoc = RegLoc::new(5, true);
pub const D6: RegLoc = RegLoc::new(6, true);
pub const D7: RegLoc = RegLoc::new(7, true);
pub const D8: RegLoc = RegLoc::new(8, true);
pub const D9: RegLoc = RegLoc::new(9, true);
pub const D10: RegLoc = RegLoc::new(10, true);
pub const D11: RegLoc = RegLoc::new(11, true);
pub const D12: RegLoc = RegLoc::new(12, true);
pub const D13: RegLoc = RegLoc::new(13, true);
pub const D14: RegLoc = RegLoc::new(14, true);
pub const D15: RegLoc = RegLoc::new(15, true);
pub const D16: RegLoc = RegLoc::new(16, true);
pub const D17: RegLoc = RegLoc::new(17, true);
pub const D18: RegLoc = RegLoc::new(18, true);
pub const D19: RegLoc = RegLoc::new(19, true);
pub const D20: RegLoc = RegLoc::new(20, true);
pub const D21: RegLoc = RegLoc::new(21, true);
pub const D22: RegLoc = RegLoc::new(22, true);
pub const D23: RegLoc = RegLoc::new(23, true);
pub const D24: RegLoc = RegLoc::new(24, true);
pub const D25: RegLoc = RegLoc::new(25, true);
pub const D26: RegLoc = RegLoc::new(26, true);
pub const D27: RegLoc = RegLoc::new(27, true);
pub const D28: RegLoc = RegLoc::new(28, true);
pub const D29: RegLoc = RegLoc::new(29, true);
pub const D30: RegLoc = RegLoc::new(30, true);
pub const D31: RegLoc = RegLoc::new(31, true);

/// registers.py:12 `vfpregisters = [VFPRegisterLocation(i) for i in range(32)]`
pub const VFPREGISTERS: [RegLoc; 32] = [
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20,
    D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31,
];

/// registers.py:13 `all_vfp_regs = vfpregisters[:8]`
pub const ALL_VFP_REGS: [RegLoc; 8] = [D0, D1, D2, D3, D4, D5, D6, D7];

/// registers.py:14 `all_regs = registers[:14] + [x19, x20] #, x21, x22]`
pub const ALL_REGS: [RegLoc; 16] = [
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X19, X20,
];

/// registers.py:16 `lr = x30`
pub const LR: RegLoc = X30;

/// registers.py:17 `fp = x29`
pub const FP: RegLoc = X29;

/// registers.py:21 `ip1 = x17`
pub const IP1: RegLoc = X17;

/// registers.py:22 `ip0 = x16`
pub const IP0: RegLoc = X16;

/// registers.py:23 `ip2 = x15`
pub const IP2: RegLoc = X15;

/// registers.py:24 `ip3 = x14`
pub const IP3: RegLoc = X14;

/// registers.py:26 `callee_saved_registers = [x19, x20] # , x21, x22]`
pub const CALLEE_SAVED_REGISTERS: [RegLoc; 2] = [X19, X20];

/// registers.py:27 `vfp_argument_regs = caller_vfp_resp = all_vfp_regs[:8]`
pub const VFP_ARGUMENT_REGS: [RegLoc; 8] = ALL_VFP_REGS;
pub const CALLER_VFP_RESP: [RegLoc; 8] = ALL_VFP_REGS;

/// registers.py:32 `vfp_ip = d15`
pub const VFP_IP: RegLoc = D15;

/// registers.py:34 `argument_regs = [x0, x1, x2, x3, x4, x5, x6, x7]`
pub const ARGUMENT_REGS: [RegLoc; 8] = [X0, X1, X2, X3, X4, X5, X6, X7];

/// registers.py:35 `callee_resp = [x19, x20] # ,x21, x22]`
pub const CALLEE_RESP: [RegLoc; 2] = CALLEE_SAVED_REGISTERS;

/// registers.py:36 `caller_resp = argument_regs + [x8, x9, x10, x11, x12, x13]`
pub const CALLER_RESP: [RegLoc; 14] = [X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13];
