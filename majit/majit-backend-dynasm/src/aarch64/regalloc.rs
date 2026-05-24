//! Port of `rpython/jit/backend/aarch64/regalloc.py` — arch-specific
//! register configuration plus the `prepare_op_int_*` family that
//! depends on AAPCS64's 3-operand encoding (`ADD/SUB/MUL Rd, Rn, Rm`).
//! Upstream splits per arch directory; pyre matches that split here.
//!
//! Methods that read/write the shared `RegAlloc` state are declared
//! as a second `impl` block on `crate::regalloc::RegAlloc<'a>`, so
//! Rust's name resolution picks up the right arch flavour at link
//! time (the `aarch64` module is `#[cfg(target_arch = "aarch64")]`
//! gated at `lib.rs:33`).

use crate::aarch64::registers;
use crate::regalloc::{RegAlloc, RegAllocOp};
use crate::regloc::{Loc, RegLoc};
use majit_ir::{OpRef, Type};

/// aarch64/regalloc.py:159 `DEFAULT_IMM_SIZE = 4096`.
const DEFAULT_IMM_SIZE: i64 = 4096;

/// aarch64/regalloc.py:169 `check_imm_box`. Accepts only `ConstInt`
/// values in the AArch64 immediate range; non-Int constants
/// (ConstFloat/ConstPtr) and box references fall through to the
/// register form. `ConstInt(slot)` is a structural promise that the
/// constant-pool entry exists (parity with `ConstInt.value` being
/// set during `__init__`); a missing entry would be a corrupted IR
/// and panics here, matching the assumption baked into
/// `arg.getint()` upstream.
fn check_imm_box(arg: OpRef, constants: &majit_ir::VecAssoc<u32, i64>) -> bool {
    if !matches!(arg, OpRef::ConstInt(_)) {
        return false;
    }
    let val = *constants
        .get(&arg.raw())
        .expect("ConstInt slot must have a constant-pool entry (parity with ConstInt.getint())");
    val >= 0 && val < DEFAULT_IMM_SIZE
}

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

/// `prepare_op_int_*` family — AArch64-side `consider_*_j2` entries.
///
/// AAPCS64's `ADD/SUB/MUL/AND/ORR/EOR/LSL/ASR/LSR Rd, Rn, Rm` accepts
/// three distinct registers, so the result is always allocated via
/// `force_allocate_reg` independently of the inputs.  RPython parity:
/// `rpython/jit/backend/aarch64/regalloc.py:341 prepare_op_int_add`
/// and `:362 prepare_op_int_mul` (the latter is reused for
/// `and/or/xor/lshift/rshift/urshift/uint_mul_high`).
impl<'a> RegAlloc<'a> {
    /// aarch64/regalloc.py:362 `prepare_op_int_mul`. 3-operand form:
    /// both operands in registers, result allocated separately.
    /// PyPy passes the full `boxes = op.getarglist()` as
    /// `forbidden_vars` for both `make_sure_var_in_reg` calls
    /// (regalloc.py:366-367), and after allocating the result also
    /// calls `possibly_free_var(op)` (regalloc.py:372) so the result
    /// register is reclaimable when the op is dead.
    pub(crate) fn consider_binop_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let boxes = [lhs, rhs];
        let lhs_loc = self.make_sure_var_in_reg(lhs, Type::Int, &boxes, None, false);
        let rhs_loc = self.make_sure_var_in_reg(rhs, Type::Int, &boxes, None, false);
        self.possibly_free_var(lhs, Type::Int);
        self.possibly_free_var(rhs, Type::Int);
        let res = self.force_allocate_reg(dst, Type::Int, &[], None, false);
        self.possibly_free_var(dst, Type::Int);
        let res_loc = Loc::Reg(res);
        self.perform(i, vec![lhs_loc, rhs_loc], Some(res_loc), output);
    }

    /// aarch64/regalloc.py:362 — `prepare_op_int_mul` is reused for
    /// symmetric binops too; no x86-style swap optimisation is needed
    /// because 3-operand encoding already lets `res` be distinct.
    pub(crate) fn consider_binop_symm_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_binop_j2(dst, lhs, rhs, i, output);
    }

    /// aarch64/regalloc.py:341 `prepare_op_int_add` → `prepare_int_ri`:
    /// allows either operand to be an immediate that fits the AArch64
    /// `add Rd, Rn, #imm12` (or shifted) encoding.  For non-immediate
    /// or out-of-range constants, fall through to the register form.
    pub(crate) fn consider_int_add_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_int_ri_j2(dst, lhs, rhs, i, output);
    }

    /// aarch64/regalloc.py:344 `prepare_op_int_sub`. `lhs` always
    /// becomes Rn; `rhs` accepts the `sub Rd, Rn, #imm12` immediate
    /// form only when it is a `ConstInt` in `[0, 4096)`
    /// (`check_imm_box`). Other constants (ConstFloat/ConstPtr) and
    /// out-of-range ints fall through to the register form.
    pub(crate) fn consider_int_sub_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let boxes = [lhs, rhs];
        let imm_rhs = check_imm_box(rhs, &self.constants);
        let lhs_loc = self.make_sure_var_in_reg(lhs, Type::Int, &boxes, None, false);
        let rhs_loc = if imm_rhs {
            self.loc(rhs, Type::Int)
        } else {
            self.make_sure_var_in_reg(rhs, Type::Int, &boxes, None, false)
        };
        self.possibly_free_var(lhs, Type::Int);
        self.possibly_free_var(rhs, Type::Int);
        let res = self.force_allocate_reg(dst, Type::Int, &[], None, false);
        self.perform(i, vec![lhs_loc, rhs_loc], Some(Loc::Reg(res)), output);
    }

    /// aarch64/regalloc.py:877 `prepare_comp_op_int_add_ovf =
    /// prepare_int_ri`. The overflow form shares the immediate-friendly
    /// preparation with `int_add` because `adds` accepts `#imm12`.
    pub(crate) fn consider_int_add_ovf_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_int_ri_j2(dst, lhs, rhs, i, output);
    }

    /// aarch64/regalloc.py:358 `prepare_comp_op_int_sub_ovf =
    /// prepare_op_int_sub`. Shares preparation with `int_sub` since
    /// `subs Rd, Rn, #imm12` accepts an immediate `rhs`.
    pub(crate) fn consider_int_sub_ovf_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_int_sub_j2(dst, lhs, rhs, i, output);
    }

    /// aarch64/regalloc.py:362 — shifts piggy-back on `prepare_op_int_mul`.
    pub(crate) fn consider_int_lshift_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_binop_j2(dst, lhs, rhs, i, output);
    }

    /// aarch64/regalloc.py:456 `prepare_unary` covers `int_neg`,
    /// `int_invert`, `int_is_true`, `int_is_zero`. `neg X(d), X(s)`
    /// and `mvn X(d), X(s)` are 3-operand: result independent of
    /// source. PyPy asserts `not isinstance(a0, Const)`
    /// (regalloc.py:458) since the optimizer is expected to have
    /// folded any const-arg unary into a constant result before this
    /// point.
    pub(crate) fn consider_unary_int_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        assert!(
            !arg.is_constant(),
            "prepare_unary expects a non-const arg; got constant OpRef {arg:?} (should have been folded earlier)"
        );
        let arg_loc = self.make_sure_var_in_reg(arg, Type::Int, &[], None, false);
        self.possibly_free_var(arg, Type::Int);
        let res = self.force_allocate_reg(dst, Type::Int, &[], None, false);
        self.perform(i, vec![arg_loc], Some(Loc::Reg(res)), output);
    }

    /// aarch64 `int_is_true` / `int_is_zero`: shares the 3-op `prepare_unary`
    /// shape from regalloc.py:456 since `cmp Xn, #0 ; cset Wd, ne` keeps
    /// the input register live while writing a fresh destination.
    pub(crate) fn consider_int_is_true_j2(
        &mut self,
        dst: OpRef,
        arg: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_unary_int_j2(dst, arg, i, output);
    }

    /// aarch64/regalloc.py:397 `prepare_op_uint_mul_high = prepare_op_int_mul`.
    /// `umulh Rd, Rn, Rm` is 3-operand; no scratch-pair constraints
    /// (unlike x86 which forces EAX/EDX).
    pub(crate) fn consider_uint_mul_high_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        self.consider_binop_j2(dst, lhs, rhs, i, output);
    }

    /// aarch64/regalloc.py:321 `prepare_int_ri`. Either operand may
    /// take the `add Rd, Rn, #imm12` immediate form when it is a
    /// `ConstInt` in `[0, 4096)` (`check_imm_box`). Non-Int constants
    /// (ConstFloat/ConstPtr) and out-of-range ints fall through to
    /// the register form.
    fn consider_int_ri_j2(
        &mut self,
        dst: OpRef,
        lhs: OpRef,
        rhs: OpRef,
        i: usize,
        output: &mut Vec<RegAllocOp>,
    ) {
        let boxes = [lhs, rhs];
        let imm_lhs = check_imm_box(lhs, &self.constants);
        let imm_rhs = check_imm_box(rhs, &self.constants);
        let (l0, l1) = if !imm_lhs && imm_rhs {
            let r0 = self.make_sure_var_in_reg(lhs, Type::Int, &boxes, None, false);
            let r1 = self.loc(rhs, Type::Int);
            (r0, r1)
        } else if imm_lhs && !imm_rhs {
            // PyPy regalloc.py:329-331 swaps so the immediate stays on
            // the Rn slot; aarch64 `add` is commutative.
            let r1 = self.loc(lhs, Type::Int);
            let r0 = self.make_sure_var_in_reg(rhs, Type::Int, &boxes, None, false);
            (r0, r1)
        } else {
            let r0 = self.make_sure_var_in_reg(lhs, Type::Int, &boxes, None, false);
            let r1 = self.make_sure_var_in_reg(rhs, Type::Int, &boxes, None, false);
            (r0, r1)
        };
        self.possibly_free_var(lhs, Type::Int);
        self.possibly_free_var(rhs, Type::Int);
        let res = self.force_allocate_reg(dst, Type::Int, &[], None, false);
        self.perform(i, vec![l0, l1], Some(Loc::Reg(res)), output);
    }
}
