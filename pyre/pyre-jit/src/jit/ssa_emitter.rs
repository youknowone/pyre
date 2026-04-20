//! SSAReprEmitter — setup-side carrier for the walker.
//!
//! The walker (`codewriter.rs::transform_graph_to_jitcode`) accumulates
//! per-op work directly into a walker-local `SSARepr`; this emitter
//! survives only to host the `JitCodeBuilder` setup state
//! (register-file sizing, constant pools, fn-pointer table,
//! virtualizable setup, jitcode name, abort flag) that
//! `Assembler::assemble` needs as its starting builder. The per-op
//! methods that used to mirror `JitCodeBuilder` and push `Insn::Op`
//! values are gone — Phase 3c collapsed the dual emitter into the
//! single walker-local `SSARepr` (commit bc0d6a06c4).
//!
//! Reference: `rpython/jit/codewriter/codewriter.py:33-73`.
//!
//! `emit_portal_jit_merge_point` is the only semantic helper left —
//! it registers the portal greens/reds constants with the builder and
//! pushes the `jit_merge_point` Insn into the walker-local `SSARepr`.

use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};

use super::assembler::{Assembler, NumRegs};
use super::flatten::{Insn, Kind, ListOfKind, Operand, SSARepr};

/// Setup-side carrier. Every method is either a `JitCodeBuilder`
/// passthrough or `finish_with_positions_from` — the finalization hook
/// that hands the builder to `Assembler::assemble` alongside the
/// walker-local `SSARepr`.
pub(super) struct SSAReprEmitter {
    /// Setup-state builder — carries `fn_ptrs`, constant pools,
    /// register-file sizing, and the jitcode name. `Assembler::assemble`
    /// consumes this builder after the walker is done; the finished
    /// `JitCode` keeps these tables intact.
    builder: JitCodeBuilder,
    /// Counter for walker-side label-id allocation. `new_label()`
    /// returns the next id; walker macros format it into the TLabel
    /// name (`catch_landing_{id}`) that `Assembler::assemble` resolves
    /// against `state.label_positions`.
    next_label_id: u16,
}

impl SSAReprEmitter {
    pub fn new() -> Self {
        Self {
            builder: JitCodeBuilder::default(),
            next_label_id: 0,
        }
    }

    // ---- setup passthrough (mirrors JitCodeBuilder setup API) ----

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.builder.set_name(name);
    }

    pub fn ensure_i_regs(&mut self, count: u16) {
        self.builder.ensure_i_regs(count);
    }

    pub fn ensure_r_regs(&mut self, count: u16) {
        self.builder.ensure_r_regs(count);
    }

    fn add_const_i(&mut self, value: i64) -> u16 {
        self.builder.add_const_i(value)
    }

    fn add_const_r(&mut self, value: i64) -> u16 {
        self.builder.add_const_r(value)
    }

    pub fn add_fn_ptr(&mut self, ptr: *const ()) -> u16 {
        self.builder.add_fn_ptr(ptr)
    }

    pub fn has_abort_flag(&self) -> bool {
        self.builder.has_abort_flag()
    }

    // ---- label id allocation ----

    /// Allocate the next u16 label id. The walker formats the id into
    /// a TLabel name (`catch_landing_{id}`) that `Assembler::assemble`
    /// resolves against the matching `Insn::Label` pushed into the
    /// walker-local `SSARepr`.
    pub fn new_label(&mut self) -> u16 {
        let id = self.next_label_id;
        self.next_label_id = self
            .next_label_id
            .checked_add(1)
            .expect("label id overflow");
        id
    }

    // ---- portal jit_merge_point ----

    /// Portal-only lowering for `interp_jit.py`'s merge point arguments.
    /// Registers the constant-pool entries for `next_instr` /
    /// `is_being_profiled` / `pycode`, computes their `num_regs + idx`
    /// positions, and pushes the `jit_merge_point` Insn into the
    /// walker-local `SSARepr` with three `ListOfKind` sublists
    /// (greens_i, greens_r, reds_r) — matching
    /// `jtransform.py:rewrite_op_jit_merge_point`'s SpaceOperation shape.
    pub fn emit_portal_jit_merge_point(
        &mut self,
        ssarepr: &mut SSARepr,
        next_instr: usize,
        w_code: i64,
        frame_reg: u16,
        ec_reg: u16,
    ) {
        let next_instr_const_idx = self.add_const_i(next_instr as i64);
        let is_being_profiled_const_idx = self.add_const_i(0);
        let pycode_const_idx = self.add_const_r(w_code);
        let num_regs_i = self.builder.num_regs_i();
        let num_regs_r = self.builder.num_regs_r();
        let gi_next_instr_reg =
            u8::try_from(u32::from(num_regs_i) + u32::from(next_instr_const_idx))
                .expect("jit_merge_point next_instr arg exceeds u8 encoding");
        let gi_is_profiled_reg =
            u8::try_from(u32::from(num_regs_i) + u32::from(is_being_profiled_const_idx))
                .expect("jit_merge_point is_being_profiled arg exceeds u8 encoding");
        let gr_pycode_reg = u8::try_from(u32::from(num_regs_r) + u32::from(pycode_const_idx))
            .expect("jit_merge_point pycode arg exceeds u8 encoding");
        let frame_reg =
            u8::try_from(frame_reg).expect("jit_merge_point frame reg exceeds u8 encoding");
        let ec_reg = u8::try_from(ec_reg).expect("jit_merge_point ec reg exceeds u8 encoding");
        let greens_i: &[u8] = &[gi_next_instr_reg, gi_is_profiled_reg];
        let greens_r: &[u8] = &[gr_pycode_reg];
        let reds_r: &[u8] = &[frame_reg, ec_reg];
        ssarepr.insns.push(Insn::op(
            "jit_merge_point",
            vec![
                list_of_regs(Kind::Int, greens_i),
                list_of_regs(Kind::Ref, greens_r),
                list_of_regs(Kind::Ref, reds_r),
            ],
        ));
    }

    // ---- finalization ----

    /// Translate an insn-index position into the corresponding JitCode
    /// byte offset using the `ssarepr.insns_pos` table that
    /// `Assembler::assemble` populates (`assembler.py:41-44`).
    pub fn insn_pos_to_byte_offset(
        ssarepr: &SSARepr,
        positions: impl IntoIterator<Item = usize>,
    ) -> Vec<usize> {
        let pos_table = ssarepr
            .insns_pos
            .as_ref()
            .expect("ssarepr.insns_pos not populated — call after assemble()");
        positions
            .into_iter()
            .map(|i| {
                *pos_table.get(i).unwrap_or_else(|| {
                    panic!(
                        "insn_pos_to_byte_offset: insn index {} out of range (len {})",
                        i,
                        pos_table.len()
                    )
                })
            })
            .collect()
    }

    /// Feed the walker-local `SSARepr` into `Assembler::assemble`
    /// against the pre-populated builder, translate the walker's
    /// per-PC insn-index map into byte offsets, and return the
    /// finished `JitCode` alongside the translated positions.
    ///
    /// `num_regs` is the post-regalloc per-kind ceiling computed by
    /// `super::regalloc::allocate_registers` from `max(color)+1`
    /// (`codewriter.py:62-67`). Passing pre-regalloc builder values
    /// would over-allocate the `JitCode.num_regs_*` slots that
    /// `Assembler::emit_reg`'s 256-bound assertion (`assembler.py:73`)
    /// checks against.
    pub fn finish_with_positions_from(
        self,
        assembler: &mut Assembler,
        mut ssarepr: SSARepr,
        insn_positions: &[usize],
        num_regs: NumRegs,
    ) -> (JitCode, Vec<usize>) {
        let jitcode = assembler.assemble(&mut ssarepr, self.builder, Some(num_regs));
        let byte_positions =
            Self::insn_pos_to_byte_offset(&ssarepr, insn_positions.iter().copied());
        (jitcode, byte_positions)
    }
}

impl Default for SSAReprEmitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Materialise a `&[u8]` register list (walker-side compact form) as
/// `ListOfKind` inside an `Operand`. Matches the encoding `assembler.py`
/// expects for the `greens_i` / `greens_r` / `reds_r` arguments of
/// `jit_merge_point`.
fn list_of_regs(kind: Kind, regs: &[u8]) -> Operand {
    let content: Vec<Operand> = regs.iter().map(|&r| Operand::reg(kind, r as u16)).collect();
    Operand::ListOfKind(ListOfKind::new(kind, content))
}
