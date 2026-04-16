/// Port of rpython/jit/backend/x86/reghint.py.
///
/// Register hints that tell the regalloc which registers to prefer for
/// specific opcodes. Runs as a pre-pass before the main regalloc walk.
///
/// RPython ships reghint only under x86; the aarch64 backend has no
/// counterpart. For pyre we apply the same mechanism uniformly because
/// the call-hint logic is ABI-driven (caller-save vs callee-save) and
/// aarch64 has the identical shape. Without this pass, long-lived
/// values whose lifetime spans a call can be assigned to caller-save
/// registers and be clobbered across calls (observed on fib_loop:
/// loop-invariant n ends up in x5 and is wiped by the inner call).
///
/// The x86 file also hints ABI arg registers and emits binop-specific
/// try_use_same_register hints for two-operand x86 instructions. Those
/// optimizations are gated under #[cfg(target_arch = "x86_64")] so the
/// aarch64 port only carries the call-hint mechanism.
use crate::regalloc::LifetimeManager;
use crate::regloc::RegLoc;
use majit_ir::{Op, OpCode, OpRef, Type};

/// reghint.py:13 SAVE_DEFAULT_REGS, SAVE_GCREF_REGS, SAVE_ALL_REGS.
pub const SAVE_DEFAULT_REGS: u8 = 0;
pub const SAVE_GCREF_REGS: u8 = 1;
pub const SAVE_ALL_REGS: u8 = 2;

// ── ABI argument registers (ARGUMENTS_GPR/ARGUMENTS_XMM) ─────────────

/// aarch64/callbuilder.py:21 argument_regs = [x0..x7].
#[cfg(target_arch = "aarch64")]
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

/// aarch64/registers.py: vfp_argument_regs = vfpregisters[:8].
#[cfg(target_arch = "aarch64")]
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

/// x86/callbuilder.py:83 CallBuilder64.ARGUMENTS_GPR on Linux/macOS.
#[cfg(target_arch = "x86_64")]
const ARGUMENTS_GPR: &[RegLoc] = &[
    // rdi, rsi, rdx, rcx, r8, r9 — encoded the same way regalloc.rs uses.
    RegLoc {
        value: 7,
        is_xmm: false,
    }, // rdi
    RegLoc {
        value: 6,
        is_xmm: false,
    }, // rsi
    RegLoc {
        value: 2,
        is_xmm: false,
    }, // rdx
    RegLoc {
        value: 1,
        is_xmm: false,
    }, // rcx
    RegLoc {
        value: 8,
        is_xmm: false,
    }, // r8
    RegLoc {
        value: 9,
        is_xmm: false,
    }, // r9
];

#[cfg(target_arch = "x86_64")]
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

// ── reghint.py:29 X86RegisterHints / RegisterHints ──────────────────

/// Entry point: iterate operations once and call per-opcode `consider_*`
/// hint methods.  Mirrors x86/reghint.py:30 X86RegisterHints.add_hints.
pub struct RegisterHints {
    save_around_call_regs_gpr: Vec<RegLoc>,
    save_around_call_regs_xmm: Vec<RegLoc>,
    all_regs_gpr: Vec<RegLoc>,
    all_regs_xmm: Vec<RegLoc>,
}

impl RegisterHints {
    pub fn new(
        save_around_call_regs_gpr: Vec<RegLoc>,
        save_around_call_regs_xmm: Vec<RegLoc>,
        all_regs_gpr: Vec<RegLoc>,
        all_regs_xmm: Vec<RegLoc>,
    ) -> Self {
        RegisterHints {
            save_around_call_regs_gpr,
            save_around_call_regs_xmm,
            all_regs_gpr,
            all_regs_xmm,
        }
    }

    /// reghint.py:30 add_hints — main entry point called from _prepare.
    pub fn add_hints(
        &self,
        longevity: &mut LifetimeManager,
        _inputargs: &[majit_ir::InputArg],
        operations: &[Op],
    ) {
        for (i, op) in operations.iter().enumerate() {
            let position = i as i32;
            match op.opcode {
                // reghint.py:147-150 consider_call_{i,r,f,n}
                // reghint.py:138-145 _consider_real_call — only plain calls get hinted.
                OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN => {
                    self.consider_real_call(longevity, op, position);
                }
                // reghint.py:154-157 consider_call_may_force_*
                OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN => {
                    self.consider_call(longevity, op, position, true, 1);
                }
                // reghint.py:162-164 consider_call_release_gil_*
                OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                    self.consider_call(longevity, op, position, true, 2);
                }
                // reghint.py:123-127 consider_call_malloc_nursery — x86-only reg
                // pins (ecx, edx). aarch64 malloc nursery uses a different
                // calling convention in opassembler; skip here for now.
                _ => {}
            }
        }
    }

    /// reghint.py:138 _consider_real_call — plain calls without special effect.
    fn consider_real_call(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let Some(descr) = op.descr.as_ref() else {
            return;
        };
        let Some(calldescr) = descr.as_call_descr() else {
            return;
        };
        // reghint.py:142-144 — skip oopspec helpers; they are hinted
        // by their op-specific consider_* methods (math_sqrt etc.).
        if calldescr.get_extra_info().has_oopspec() {
            return;
        }
        self.consider_call(longevity, op, position, false, 1);
    }

    /// reghint.py:130 _consider_call(op, position, guard_not_forced, first_arg_index).
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

    /// reghint.py:207 CallHints64.hint — hint args into ABI arg regs, then
    /// _block_non_caller_save for all other call-clobbered regs.
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

    /// reghint.py:176 _block_non_caller_save.
    ///
    /// Reserve every caller-save GPR (and every call-clobbered XMM) at the
    /// call position so that `free_reg_whole_lifetime` excludes them for
    /// variables whose lifetime spans the call.
    fn block_non_caller_save(
        &self,
        longevity: &mut LifetimeManager,
        position: i32,
        save_all_regs: u8,
        hinted_gpr: &[RegLoc],
        hinted_xmm: &[RegLoc],
    ) {
        // reghint.py:184-188 GPR selection.
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
        // reghint.py:192-194 XMM always uses all_regs (every XMM reg is
        // caller-save on both x86-64 SysV and aarch64 AAPCS64 in pyre).
        for &reg in &self.all_regs_xmm {
            if !hinted_xmm.contains(&reg) {
                longevity.fixed_register(position, reg, None);
            }
        }
    }
}

/// x86/regalloc.py:35 compute_gc_level.
fn compute_gc_level(calldescr: &dyn majit_ir::descr::CallDescr, guard_not_forced: bool) -> u8 {
    if guard_not_forced {
        return SAVE_ALL_REGS;
    }
    if !calldescr.get_extra_info().check_can_collect() {
        return SAVE_DEFAULT_REGS;
    }
    SAVE_GCREF_REGS
}
