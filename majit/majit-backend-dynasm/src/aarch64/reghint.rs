//! AAPCS64 counterpart of `rpython/jit/backend/x86/reghint.py`.
//!
//! **Structural parity note — latent RPython aarch64 bug.**
//!
//! Upstream `rpython/jit/backend/aarch64/` does NOT ship a
//! `reghint.py` file.  Without one, caller-save-vs-callee-save
//! information never reaches `LifetimeManager.free_reg_whole_lifetime`
//! and it returns `unfixed_reg` for any loop-invariant whose
//! lifetime spans an inner call.  `longest_free_reg`'s fallback then
//! picks a caller-save register, which the call immediately
//! clobbers — the exact symptom observed on fib_loop before pyre
//! started running a reghint pass on aarch64.
//!
//! x86 avoids this because `rpython/jit/backend/x86/reghint.py` runs
//! before the regalloc walk and marks caller-save registers as
//! `fixed_register(call_pos, reg, None)` for every call, so
//! `free_reg_whole_lifetime` rejects them for vars that span the
//! call.  The mechanism is ABI-driven (caller-save vs callee-save
//! has the same semantics on AAPCS64) and nothing in it is
//! architecturally x86-specific.
//!
//! This file is the proper port of that mechanism to aarch64 — the
//! file upstream RPython aarch64 *should* carry.  It is structurally
//! a peer of `x86/reghint.rs`, not a downstream trampoline, and is
//! invoked the same way from `RegAlloc::_prepare` (via the
//! `arch_reghint` alias).

use crate::regalloc::LifetimeManager;
use crate::regloc::RegLoc;
use majit_ir::{Op, OpCode, OpRef, Type};

/// reghint.py:13.
pub const SAVE_DEFAULT_REGS: u8 = 0;
pub const SAVE_GCREF_REGS: u8 = 1;
pub const SAVE_ALL_REGS: u8 = 2;

/// aarch64/callbuilder.py:21 `argument_regs = [x0..x7]`.
const ARGUMENTS_GPR: &[RegLoc] = &[
    RegLoc {
        value: 0,
        is_xmm: false,
    },
    RegLoc {
        value: 1,
        is_xmm: false,
    },
    RegLoc {
        value: 2,
        is_xmm: false,
    },
    RegLoc {
        value: 3,
        is_xmm: false,
    },
    RegLoc {
        value: 4,
        is_xmm: false,
    },
    RegLoc {
        value: 5,
        is_xmm: false,
    },
    RegLoc {
        value: 6,
        is_xmm: false,
    },
    RegLoc {
        value: 7,
        is_xmm: false,
    },
];

/// aarch64/registers.py `vfp_argument_regs = vfpregisters[:8]`.
const ARGUMENTS_XMM: &[RegLoc] = &[
    RegLoc {
        value: 0,
        is_xmm: true,
    },
    RegLoc {
        value: 1,
        is_xmm: true,
    },
    RegLoc {
        value: 2,
        is_xmm: true,
    },
    RegLoc {
        value: 3,
        is_xmm: true,
    },
    RegLoc {
        value: 4,
        is_xmm: true,
    },
    RegLoc {
        value: 5,
        is_xmm: true,
    },
    RegLoc {
        value: 6,
        is_xmm: true,
    },
    RegLoc {
        value: 7,
        is_xmm: true,
    },
];

/// Mirror of `X86RegisterHints`, adapted to AAPCS64.  The body matches
/// `x86/reghint.rs` line by line — the only differences are the
/// argument-register constants above.
pub struct RegisterHints {
    save_around_call_regs_gpr: Vec<RegLoc>,
    all_regs_gpr: Vec<RegLoc>,
    all_regs_xmm: Vec<RegLoc>,
}

impl RegisterHints {
    pub fn new(
        save_around_call_regs_gpr: Vec<RegLoc>,
        _save_around_call_regs_xmm: Vec<RegLoc>,
        all_regs_gpr: Vec<RegLoc>,
        all_regs_xmm: Vec<RegLoc>,
    ) -> Self {
        RegisterHints {
            save_around_call_regs_gpr,
            all_regs_gpr,
            all_regs_xmm,
        }
    }

    /// reghint.py:30 `add_hints`.
    pub fn add_hints(
        &self,
        longevity: &mut LifetimeManager,
        _inputargs: &[majit_ir::InputArg],
        operations: &[Op],
    ) {
        for (i, op) in operations.iter().enumerate() {
            let position = i as i32;
            match op.opcode {
                OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN => {
                    self.consider_real_call(longevity, op, position);
                }
                OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN => {
                    self.consider_call(longevity, op, position, true, 1);
                }
                OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                    self.consider_call(longevity, op, position, true, 2);
                }
                _ => {}
            }
        }
    }

    /// reghint.py:138 `_consider_real_call`.
    fn consider_real_call(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let Some(descr) = op.descr.as_ref() else {
            return;
        };
        let Some(calldescr) = descr.as_call_descr() else {
            return;
        };
        if calldescr.get_extra_info().has_oopspec() {
            return;
        }
        self.consider_call(longevity, op, position, false, 1);
    }

    /// reghint.py:130 `_consider_call`.
    fn consider_call(
        &self,
        longevity: &mut LifetimeManager,
        op: &Op,
        position: i32,
        guard_not_forced: bool,
        first_arg_index: usize,
    ) {
        let Some(descr) = op.descr.as_ref() else {
            return;
        };
        let Some(calldescr) = descr.as_call_descr() else {
            return;
        };
        let gc_level = compute_gc_level(calldescr, guard_not_forced);
        let args = &op.args[first_arg_index..];
        let argtypes = calldescr.arg_types();
        self.hint(longevity, position, args, argtypes, gc_level);
    }

    /// reghint.py:207 `CallHints64.hint` (AAPCS64 mapping).
    fn hint(
        &self,
        longevity: &mut LifetimeManager,
        position: i32,
        args: &[OpRef],
        argtypes: &[Type],
        save_all_regs: u8,
    ) {
        let mut hinted_xmm: Vec<RegLoc> = Vec::new();
        let mut hinted_gpr: Vec<RegLoc> = Vec::new();
        let mut hinted_args: Vec<OpRef> = Vec::new();
        let mut next_arg_gpr = 0usize;
        let mut next_arg_xmm = 0usize;

        for (i, &arg) in args.iter().enumerate() {
            let arg_type = argtypes.get(i).copied().unwrap_or(Type::Int);
            if arg_type == Type::Float {
                if next_arg_xmm < ARGUMENTS_XMM.len() {
                    let tgt = ARGUMENTS_XMM[next_arg_xmm];
                    if !arg.is_constant() && !hinted_args.contains(&arg) {
                        longevity.fixed_register(position, tgt, Some(arg));
                        hinted_xmm.push(tgt);
                        hinted_args.push(arg);
                    }
                    next_arg_xmm += 1;
                }
            } else if next_arg_gpr < ARGUMENTS_GPR.len() {
                let tgt = ARGUMENTS_GPR[next_arg_gpr];
                if !arg.is_constant() && !hinted_args.contains(&arg) {
                    longevity.fixed_register(position, tgt, Some(arg));
                    hinted_gpr.push(tgt);
                    hinted_args.push(arg);
                }
                next_arg_gpr += 1;
            }
        }
        self.block_non_caller_save(longevity, position, save_all_regs, &hinted_gpr, &hinted_xmm);
    }

    /// reghint.py:176 `_block_non_caller_save` (AAPCS64: caller-save is
    /// `x0..x13` per aarch64/registers.py `caller_resp`, plus every
    /// VFP reg pyre uses, `d0..d7`).
    fn block_non_caller_save(
        &self,
        longevity: &mut LifetimeManager,
        position: i32,
        save_all_regs: u8,
        hinted_gpr: &[RegLoc],
        hinted_xmm: &[RegLoc],
    ) {
        let gpr_regs: &[RegLoc] = if save_all_regs == SAVE_ALL_REGS {
            &self.all_regs_gpr
        } else {
            &self.save_around_call_regs_gpr
        };
        for &reg in gpr_regs {
            if !hinted_gpr.contains(&reg) {
                longevity.fixed_register(position, reg, None);
            }
        }
        for &reg in &self.all_regs_xmm {
            if !hinted_xmm.contains(&reg) {
                longevity.fixed_register(position, reg, None);
            }
        }
    }
}

/// x86/regalloc.py:35 `compute_gc_level` — ABI-neutral logic, kept
/// per-arch for local closure.
fn compute_gc_level(calldescr: &dyn majit_ir::descr::CallDescr, guard_not_forced: bool) -> u8 {
    if guard_not_forced {
        return SAVE_ALL_REGS;
    }
    if !calldescr.get_extra_info().check_can_collect() {
        return SAVE_DEFAULT_REGS;
    }
    SAVE_GCREF_REGS
}
