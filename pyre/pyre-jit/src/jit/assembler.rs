//! Line-by-line port of `rpython/jit/codewriter/assembler.py` — the
//! `Assembler.assemble(ssarepr, jitcode, num_regs)` driver.
//!
//! Consumes an `SSARepr` (see `super::flatten`) and produces a finished
//! `JitCode`. The bytecode-emission primitives (`emit_reg`, `emit_const`,
//! label fixup) still live on `majit_metainterp::jitcode::JitCodeBuilder`
//! — this module is the high-level dispatcher that walks
//! `ssarepr.insns` and forwards each `Insn` to the correct builder call,
//! mirroring `assembler.py:34-54`.
//!
//! Status — Phase 3a: framework + a subset of the opname dispatch table.
//! The pyre bytecode walker currently emits directly into
//! `JitCodeBuilder`; the migration strategy is:
//!   1. Extend this dispatch table until every opcode pyre emits has a
//!      `(opname, argcodes) -> builder call` mapping.
//!   2. Refactor `codewriter.rs::transform_graph_to_jitcode` to build an
//!      `SSARepr` (Phase 3b) instead of driving `JitCodeBuilder` directly.
//!   3. Route both paths through `assemble()` here (Phase 3c).
//!
//! Phases 3b/3c are tracked in `B6_CODEWRITER_PIPELINE_PLAN.md`. The
//! current implementation only builds the dispatch skeleton; calling
//! `assemble` on a non-empty `SSARepr` with an unported opname panics so
//! the gap is visible rather than silent.

use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};

use super::flatten::{Insn, Kind, Operand, Register, SSARepr};

/// `assembler.py:34-54` `Assembler.assemble(self, ssarepr, jitcode=None, num_regs=None)`.
///
/// Walks `ssarepr.insns`, dispatches each `Insn` to the matching builder
/// call, fixes up labels, and returns the finished `JitCode`. The
/// `num_regs` overrides are applied before emission so the per-kind
/// register-file sizes match the caller's regalloc output.
///
/// The `assembler` argument is pre-populated by the caller with the
/// pyre-specific helper-function table (fn_ptrs) and any other
/// builder-state that doesn't come from the SSARepr. This mirrors
/// RPython where the `CodeWriter` holds a reusable `Assembler` instance
/// across multiple `assemble` calls.
pub fn assemble(
    ssarepr: &SSARepr,
    mut assembler: JitCodeBuilder,
    num_regs: Option<NumRegs>,
) -> JitCode {
    if let Some(nr) = num_regs {
        assembler.ensure_i_regs(nr.int);
        assembler.ensure_r_regs(nr.ref_);
        assembler.ensure_f_regs(nr.float);
    }
    assembler.set_name(ssarepr.name.clone());

    for insn in &ssarepr.insns {
        write_insn(&mut assembler, insn);
    }

    // Label fixup + finalisation. `JitCodeBuilder::finish` already performs
    // label patching internally, so the explicit `fix_labels` /
    // `check_result` calls from `assembler.py:45-46` are folded into
    // `finish`.
    assembler.finish()
}

/// `assembler.py:65` `count_regs = dict.fromkeys(KINDS, 0)` override.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumRegs {
    pub int: u16,
    pub ref_: u16,
    pub float: u16,
}

/// `assembler.py:140-158,215-223` `write_insn(insn)`.
///
/// Dispatches on the tuple shape. RPython's path looks up
/// `opname+argcodes` in `self.insns` and auto-assigns an opcode number;
/// pyre uses a fixed `BC_*` opcode table, so the dispatch is a match on
/// `opname + collected argcodes`.
///
/// Unknown opnames panic — the migration is incomplete until every
/// pyre bytecode walker call has a matching arm here.
fn write_insn(_assembler: &mut JitCodeBuilder, insn: &Insn) {
    match insn {
        // `assembler.py:141` `if isinstance(insn[0], Label): ... return`.
        Insn::Label(_label) => {
            // TODO(phase-3b): call assembler.mark_label(label_id) once the
            // Insn::Label carries the builder-side u16 id (or pyre's
            // flatten step maps name → id before calling here).
        }
        // `assembler.py:146-158` `if insn[0] == '-live-'`.
        Insn::Live(_args) => {
            // TODO(phase-3b): emit BC_LIVE + inline 2-byte offset, then
            // intern the liveness bitset via ASSEMBLER_STATE.
        }
        // `assembler.py` has no `---` branch; the unreachable marker is
        // consumed by `liveness.py` and stripped before the assembler
        // sees the SSARepr. Keep the arm for parity-audit clarity.
        Insn::Unreachable => {}
        // `assembler.py:159-223` regular-op dispatch.
        Insn::Op {
            opname,
            args,
            result,
        } => {
            dispatch_op(_assembler, opname, args, result.as_ref());
        }
    }
}

/// Per-opname dispatch — phase-3a stub covering only the trivial cases
/// so the framework compiles; the full mapping is populated as
/// `codewriter.rs` is migrated.
fn dispatch_op(
    _assembler: &mut JitCodeBuilder,
    opname: &str,
    _args: &[Operand],
    _result: Option<&Register>,
) {
    match opname {
        // The remaining mappings are added as each bytecode handler in
        // `codewriter.rs` is converted from "call assembler directly" to
        // "emit Insn::Op". Until the handler is migrated, its path still
        // uses `JitCodeBuilder` directly, so the dispatch table here is
        // intentionally empty.
        _ => panic!(
            "assemble(): unimplemented opname {:?} — pyre bytecode walker \
             has not yet been ported to SSARepr emission for this op",
            opname
        ),
    }
}

/// Small helper used by the dispatcher to validate that an operand is a
/// register of the expected kind — matches `assembler.emit_reg`'s
/// internal `assert reg.kind == expected`.
#[inline]
#[allow(dead_code)]
fn expect_reg(op: &Operand, expected: Kind) -> u16 {
    match op {
        Operand::Register(Register { kind, index }) if *kind == expected => *index,
        _ => panic!(
            "expect_reg: expected Register({:?}, _), got {:?}",
            expected, op
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_empty_ssarepr_produces_valid_jitcode() {
        let ssarepr = SSARepr::new("empty");
        let assembler = JitCodeBuilder::default();
        let jitcode = assemble(&ssarepr, assembler, None);
        assert_eq!(jitcode.name, "empty");
        assert!(jitcode.code.is_empty(), "empty SSARepr → empty code");
    }
}
