//! Line-by-line port of `rpython/jit/codewriter/assembler.py`.
//!
//! This module mirrors the high-level `Assembler` flow from RPython:
//! `setup()` → `write_insn()` → `fix_labels()` → `check_result()` →
//! `make_jitcode()`. pyre still targets `majit_metainterp::jitcode::JitCode`
//! and its fixed `BC_*` builder API, so regular-op dispatch lowers the
//! textual `SSARepr` opnames into `JitCodeBuilder` calls instead of growing
//! `self.insns` into a runtime opcode table.

use std::collections::HashMap;

use majit_codewriter::jitcode::BhDescr;
use majit_metainterp::jitcode::{JitCallArg, JitCode, JitCodeBuilder};

use super::flatten::{DescrOperand, Insn, Kind, ListOfKind, Operand, Register, SSARepr, TLabel};
use super::liveness::encode_liveness;

/// `assembler.py:65` `count_regs = dict.fromkeys(KINDS, 0)` override.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumRegs {
    pub int: u16,
    pub ref_: u16,
    pub float: u16,
}

/// `assembler.py:19-32` `class Assembler(object)`.
///
/// The persistent state kept across multiple `assemble()` calls matches the
/// RPython fields that are semantically shared across jitcodes: the deduped
/// `all_liveness` table and liveness statistics.
#[derive(Debug, Default)]
pub struct Assembler {
    /// `assembler.py:29` `self.all_liveness = []`.
    all_liveness: Vec<u8>,
    /// `assembler.py:30` `self.all_liveness_length = 0`.
    pub all_liveness_length: usize,
    /// `assembler.py:31` `self.all_liveness_positions = {}`.
    all_liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), usize>,
    /// `assembler.py:32` `self.num_liveness_ops = 0`.
    pub num_liveness_ops: usize,
}

/// Builder-local state created by `setup()` for one `assemble()` call.
///
/// RPython keeps most of these on `self`; pyre keeps them in a per-call
/// struct because the runtime `JitCodeBuilder` is consumed by `finish()`.
struct AssemblyState {
    builder: JitCodeBuilder,
    /// `assembler.py:59` `self.label_positions = {}`.
    label_positions: HashMap<String, usize>,
    /// Builder adapter for `Label/TLabel` name → builder label id.
    /// RPython stores bytecode positions directly in `label_positions`; this
    /// extra vector exists only because `JitCodeBuilder` patches jumps by
    /// symbolic label id rather than by rewriting raw bytes in `fix_labels()`.
    builder_labels: Vec<(String, u16)>,
    /// SSARepr-side switch descrs that must be attached after all labels have
    /// final positions (`assembler.py:258-263`).
    switch_descrs: Vec<(usize, Vec<(i64, TLabel)>)>,
    /// Runtime descr table for this jitcode (`assembler.py:26` `self.descrs`).
    descrs: Vec<BhDescr>,
    /// `assembler.py:26` `self._descr_dict = {}` — identity-keyed dedup so
    /// re-using the same SSARepr descr across multiple ops yields a stable
    /// `descrs` index. `DescrOperand` has no inherent identity in Rust;
    /// the key is the Vec-slot pointer captured when the `Operand` was
    /// first interned — semantically equivalent to Python's `id()` lookup.
    descr_dict: HashMap<DescrKey, usize>,
}

/// Identity key for `DescrOperand` dedup. Matches
/// `assembler.py:197-199` `if x not in self._descr_dict` which uses
/// Python object identity.
///
/// `Bh` descrs are deduped by pointer-equality against the `DescrOperand`
/// borrow (the SSARepr owns the operand, so pointers are stable during
/// assemble()). `SwitchDict` descrs must be deduped by the same rule —
/// two distinct SwitchDictDescr operands produce two `descrs` entries
/// even if their `labels` happen to match (matches RPython's identity
/// semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DescrKey(*const DescrOperand);

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn all_liveness(&self) -> &[u8] {
        &self.all_liveness
    }

    /// `assembler.py:34-54` `assemble(self, ssarepr, jitcode=None, num_regs=None)`.
    pub fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        mut builder: JitCodeBuilder,
        num_regs: Option<NumRegs>,
    ) -> JitCode {
        if let Some(nr) = num_regs {
            builder.ensure_i_regs(nr.int);
            builder.ensure_r_regs(nr.ref_);
            builder.ensure_f_regs(nr.float);
        }
        builder.set_name(ssarepr.name.clone());

        let mut state = AssemblyState {
            builder,
            label_positions: HashMap::new(),
            builder_labels: Vec::new(),
            switch_descrs: Vec::new(),
            descrs: Vec::new(),
            descr_dict: HashMap::new(),
        };

        ssarepr.insns_pos = Some(Vec::with_capacity(ssarepr.insns.len()));
        for insn in &ssarepr.insns {
            ssarepr
                .insns_pos
                .as_mut()
                .expect("insns_pos initialized")
                .push(state.builder.current_pos());
            self.write_insn(&mut state, insn);
        }

        self.fix_labels(&mut state);

        let mut jitcode = state.builder.finish();
        jitcode.descrs = state.descrs;
        self.check_result(&jitcode);
        jitcode
    }

    /// `assembler.py:140-223` `write_insn(insn)`.
    fn write_insn(&mut self, state: &mut AssemblyState, insn: &Insn) {
        match insn {
            Insn::Unreachable => {}
            Insn::Label(label) => {
                let label_id = builder_label(state, &label.name);
                state
                    .label_positions
                    .insert(label.name.clone(), state.builder.current_pos());
                state.builder.mark_label(label_id);
            }
            Insn::Live(args) => {
                self.num_liveness_ops += 1;
                let live_i = get_liveness_info(args, Kind::Int);
                let live_r = get_liveness_info(args, Kind::Ref);
                let live_f = get_liveness_info(args, Kind::Float);
                let patch_offset = state.builder.live_placeholder();
                let offset = self.encode_liveness_info(&live_i, &live_r, &live_f);
                state.builder.patch_live_offset(patch_offset, offset);
            }
            Insn::Op {
                opname,
                args,
                result,
            } => {
                // `assembler.py:197-206` registers descrs inline during op
                // encoding, guarded by `self._descr_dict`. The previous
                // pre-scan violated parity by interning every
                // `Operand::Descr` unconditionally (even on ops that don't
                // consume a descr in the emitted bytecode) and without
                // dedup. Dispatch arms that actually need a descr index
                // now call `record_descr_operand(state, descr)` themselves.
                dispatch_op(state, opname, args, result.as_ref());
            }
        }
    }

    /// `assembler.py:250-263` `fix_labels()`.
    fn fix_labels(&mut self, state: &mut AssemblyState) {
        for (descr_index, labels) in &state.switch_descrs {
            let mut dict = HashMap::new();
            for (key, label) in labels {
                let target = *state
                    .label_positions
                    .get(&label.name)
                    .unwrap_or_else(|| panic!("missing switch target label {:?}", label.name));
                dict.insert(*key, target);
            }
            state.descrs[*descr_index] = BhDescr::Switch { dict };
        }
    }

    /// `assembler.py:265-269` `check_result()`.
    fn check_result(&self, jitcode: &JitCode) {
        assert!(
            (jitcode.num_regs_i() as usize) + jitcode.constants_i.len() <= 256,
            "too many int registers/constants"
        );
        assert!(
            (jitcode.num_regs_r() as usize) + jitcode.constants_r.len() <= 256,
            "too many ref registers/constants"
        );
        assert!(
            (jitcode.num_regs_f() as usize) + jitcode.constants_f.len() <= 256,
            "too many float registers/constants"
        );
    }

    /// `assembler.py:234-248` `_encode_liveness(...)`.
    fn encode_liveness_info(&mut self, live_i: &[u8], live_r: &[u8], live_f: &[u8]) -> u16 {
        let key = (live_i.to_vec(), live_r.to_vec(), live_f.to_vec());
        let pos = if let Some(&pos) = self.all_liveness_positions.get(&key) {
            pos
        } else {
            let pos = self.all_liveness_length;
            self.all_liveness_positions.insert(key, pos);
            self.all_liveness.push(live_i.len() as u8);
            self.all_liveness.push(live_r.len() as u8);
            self.all_liveness.push(live_f.len() as u8);
            for live in [live_i, live_r, live_f] {
                let encoded = encode_liveness(live);
                self.all_liveness.extend_from_slice(&encoded);
            }
            self.all_liveness_length = self.all_liveness.len();
            pos
        };
        u16::try_from(pos).expect("all_liveness offset overflow")
    }
}

/// Test-only convenience wrapper — creates a throwaway `Assembler` per
/// call. **Do not wire this into production paths.** RPython's
/// `CodeWriter` holds a single `Assembler` whose `all_liveness`,
/// `all_liveness_positions`, and `num_liveness_ops` accumulate across
/// every JitCode compiled in the session (`codewriter.py:19-22` /
/// `pyjitpl.py:2264`). A fresh `Assembler` per call splits the
/// global liveness table and breaks that invariant.
///
/// Production callers must construct one `Assembler` up front and call
/// `assembler.assemble(...)` repeatedly on it.
#[cfg(test)]
fn assemble(ssarepr: &mut SSARepr, builder: JitCodeBuilder, num_regs: Option<NumRegs>) -> JitCode {
    let mut assembler = Assembler::new();
    assembler.assemble(ssarepr, builder, num_regs)
}

fn builder_label(state: &mut AssemblyState, name: &str) -> u16 {
    if let Some((_, label)) = state
        .builder_labels
        .iter()
        .find(|(existing, _)| existing == name)
    {
        return *label;
    }
    let label = state.builder.new_label();
    state.builder_labels.push((name.to_owned(), label));
    label
}

fn get_liveness_info(args: &[Operand], kind: Kind) -> Vec<u8> {
    let mut lives = Vec::new();
    for arg in args {
        if let Operand::Register(Register {
            kind: reg_kind,
            index,
        }) = arg
        {
            if *reg_kind == kind {
                lives.push(u8::try_from(*index).expect("liveness register index exceeds u8"));
            }
        }
    }
    lives.sort_unstable();
    lives.dedup();
    lives
}

fn dispatch_op(
    state: &mut AssemblyState,
    opname: &str,
    args: &[Operand],
    result: Option<&Register>,
) {
    match opname {
        "goto" | "jump" => {
            let label = expect_tlabel(&args[0]);
            let label_id = builder_label(state, &label.name);
            state.builder.jump(label_id);
        }
        "goto_if_not" | "branch_reg_zero" => {
            let cond = expect_reg(&args[0], Kind::Int);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.branch_reg_zero(cond, label_id);
        }
        "catch_exception" => {
            let label = expect_tlabel(&args[0]);
            let label_id = builder_label(state, &label.name);
            state.builder.catch_exception(label_id);
        }
        "jit_merge_point" => {
            let greens_i = expect_list_regs(&args[0], Kind::Int);
            let greens_r = expect_list_regs(&args[1], Kind::Ref);
            let reds_r = expect_list_regs(&args[2], Kind::Ref);
            state.builder.jit_merge_point(&greens_i, &greens_r, &reds_r);
        }
        "jump_target" => {
            // pyre-only marker: loop-header entry point that isn't a
            // jit_merge_point. RPython uses plain `Label` entries; pyre
            // emits a dedicated BC_JUMP_TARGET so the runtime dispatch
            // loop can recognise loop heads cheaply.
            state.builder.jump_target();
        }
        "ref_return" => {
            let src = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.ref_return(src);
        }
        "raise" => {
            let src = expect_reg(&args[0], Kind::Ref);
            state.builder.emit_raise(src);
        }
        "reraise" => state.builder.emit_reraise(),
        "last_exc_value" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.last_exc_value(dst);
        }
        "abort" => state.builder.abort(),
        "abort_permanent" => state.builder.abort_permanent(),
        "move_i" | "int_copy" => {
            let (dst, src) = expect_move(args, result, Kind::Int);
            state.builder.move_i(dst, src);
        }
        "move_r" | "ref_copy" => {
            let (dst, src) = expect_move(args, result, Kind::Ref);
            state.builder.move_r(dst, src);
        }
        "move_f" | "float_copy" => {
            let (dst, src) = expect_move(args, result, Kind::Float);
            state.builder.move_f(dst, src);
        }
        "load_const_i" => {
            let (dst, value) = expect_load_const_i(args, result);
            state.builder.load_const_i_value(dst, value);
        }
        "load_const_r" => {
            let (dst, value) = expect_load_const_r(args, result);
            state.builder.load_const_r_value(dst, value);
        }
        "load_const_f" => {
            let (dst, value) = expect_load_const_f(args, result);
            state.builder.load_const_f_value(dst, value);
        }
        "load_state_field" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state
                .builder
                .load_state_field(expect_small_u16(&args[0]), dst);
        }
        "store_state_field" => {
            state
                .builder
                .store_state_field(expect_small_u16(&args[0]), expect_reg(&args[1], Kind::Int));
        }
        "load_state_array" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.load_state_array(
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
                dst,
            );
        }
        "store_state_array" => {
            state.builder.store_state_array(
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
                expect_reg(&args[2], Kind::Int),
            );
        }
        "load_state_varray" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.load_state_varray(
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
                dst,
            );
        }
        "store_state_varray" => {
            state.builder.store_state_varray(
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
                expect_reg(&args[2], Kind::Int),
            );
        }
        // RPython opname parity (jtransform.py:765-927, blackhole.py:1374-1493):
        // SpaceOperation names use `*_vable_*` infix. pyre's JitCodeBuilder
        // methods retain `vable_*` prefix as a PRE-EXISTING-ADAPTATION so
        // the rename is scoped to the Insn::Op key.
        "getfield_vable_i" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state
                .builder
                .vable_getfield_int(dst, expect_small_u16(&args[0]));
        }
        "getfield_vable_r" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state
                .builder
                .vable_getfield_ref(dst, expect_small_u16(&args[0]));
        }
        "getfield_vable_f" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            state
                .builder
                .vable_getfield_float(dst, expect_small_u16(&args[0]));
        }
        "setfield_vable_i" => state
            .builder
            .vable_setfield_int(expect_small_u16(&args[0]), expect_reg(&args[1], Kind::Int)),
        "setfield_vable_r" => state
            .builder
            .vable_setfield_ref(expect_small_u16(&args[0]), expect_reg(&args[1], Kind::Ref)),
        "setfield_vable_f" => state.builder.vable_setfield_float(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Float),
        ),
        "getarrayitem_vable_i" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.vable_getarrayitem_int(
                dst,
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "getarrayitem_vable_r" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.vable_getarrayitem_ref(
                dst,
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "getarrayitem_vable_f" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            state.builder.vable_getarrayitem_float(
                dst,
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "setarrayitem_vable_i" => state.builder.vable_setarrayitem_int(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Int),
            expect_reg(&args[2], Kind::Int),
        ),
        "setarrayitem_vable_r" => state.builder.vable_setarrayitem_ref(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Int),
            expect_reg(&args[2], Kind::Ref),
        ),
        "setarrayitem_vable_f" => state.builder.vable_setarrayitem_float(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Int),
            expect_reg(&args[2], Kind::Float),
        ),
        "arraylen_vable" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state
                .builder
                .vable_arraylen(dst, expect_small_u16(&args[0]));
        }
        "hint_force_virtualizable" => state.builder.vable_force(),
        "record_binop_i" => {
            let dst = expect_result_reg(result, Kind::Int, "record_binop_i needs result");
            state.builder.record_binop_i(
                dst,
                expect_opcode(&args[0]),
                expect_reg(&args[1], Kind::Int),
                expect_reg(&args[2], Kind::Int),
            );
        }
        "record_binop_f" => {
            let dst = expect_result_reg(result, Kind::Float, "record_binop_f needs result");
            state.builder.record_binop_f(
                dst,
                expect_opcode(&args[0]),
                expect_reg(&args[1], Kind::Float),
                expect_reg(&args[2], Kind::Float),
            );
        }
        "record_unary_i" => {
            let dst = expect_result_reg(result, Kind::Int, "record_unary_i needs result");
            state.builder.record_unary_i(
                dst,
                expect_opcode(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "record_unary_f" => {
            let dst = expect_result_reg(result, Kind::Float, "record_unary_f needs result");
            state.builder.record_unary_f(
                dst,
                expect_opcode(&args[0]),
                expect_reg(&args[1], Kind::Float),
            );
        }
        "call_void" | "residual_call_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .residual_call_void_typed_args(fn_idx, &call_args);
        }
        "call_may_force_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_may_force_void_typed_args(fn_idx, &call_args);
        }
        "call_release_gil_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_release_gil_void_typed_args(fn_idx, &call_args);
        }
        "call_loopinvariant_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_loopinvariant_void_typed_args(fn_idx, &call_args);
        }
        "call_assembler_void" => {
            let target_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_assembler_void_typed_args(target_idx, &call_args);
        }
        "call_int"
        | "call_pure_int"
        | "call_may_force_int"
        | "call_release_gil_int"
        | "call_loopinvariant_int"
        | "call_assembler_int" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Int, "int call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_int" => state.builder.call_int_typed(fn_idx, &call_args, dst),
                "call_pure_int" => state.builder.call_pure_int_typed(fn_idx, &call_args, dst),
                "call_may_force_int" => state
                    .builder
                    .call_may_force_int_typed(fn_idx, &call_args, dst),
                "call_release_gil_int" => state
                    .builder
                    .call_release_gil_int_typed(fn_idx, &call_args, dst),
                "call_loopinvariant_int" => state
                    .builder
                    .call_loopinvariant_int_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_int_typed(fn_idx, &call_args, dst),
            }
        }
        "call_ref"
        | "call_pure_ref"
        | "call_may_force_ref"
        | "call_release_gil_ref"
        | "call_loopinvariant_ref"
        | "call_assembler_ref" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Ref, "ref call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_ref" => state.builder.call_ref_typed(fn_idx, &call_args, dst),
                "call_pure_ref" => state.builder.call_pure_ref_typed(fn_idx, &call_args, dst),
                "call_may_force_ref" => state
                    .builder
                    .call_may_force_ref_typed(fn_idx, &call_args, dst),
                "call_release_gil_ref" => state
                    .builder
                    .call_release_gil_ref_typed(fn_idx, &call_args, dst),
                "call_loopinvariant_ref" => state
                    .builder
                    .call_loopinvariant_ref_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_ref_typed(fn_idx, &call_args, dst),
            }
        }
        "call_float"
        | "call_pure_float"
        | "call_may_force_float"
        | "call_release_gil_float"
        | "call_loopinvariant_float"
        | "call_assembler_float" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Float, "float call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_float" => state.builder.call_float_typed(fn_idx, &call_args, dst),
                "call_pure_float" => state.builder.call_pure_float_typed(fn_idx, &call_args, dst),
                "call_may_force_float" => state
                    .builder
                    .call_may_force_float_typed(fn_idx, &call_args, dst),
                "call_release_gil_float" => state
                    .builder
                    .call_release_gil_float_typed(fn_idx, &call_args, dst),
                "call_loopinvariant_float" => state
                    .builder
                    .call_loopinvariant_float_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_float_typed(fn_idx, &call_args, dst),
            }
        }
        "conditional_call_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let cond_reg = expect_reg(&args[1], Kind::Int);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_void_typed_args(fn_idx, cond_reg, &call_args);
        }
        "conditional_call_value_int" => {
            let fn_idx = expect_small_u16(&args[0]);
            let value_reg = expect_reg(&args[1], Kind::Int);
            let dst =
                expect_result_reg(result, Kind::Int, "conditional_call_value_int needs result");
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_value_int_typed_args(fn_idx, value_reg, &call_args, dst);
        }
        "conditional_call_value_ref" => {
            let fn_idx = expect_small_u16(&args[0]);
            let value_reg = expect_reg(&args[1], Kind::Int);
            let dst =
                expect_result_reg(result, Kind::Ref, "conditional_call_value_ref needs result");
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_value_ref_typed_args(fn_idx, value_reg, &call_args, dst);
        }
        "record_known_result_int" => {
            let fn_idx = expect_small_u16(&args[0]);
            let result_reg = expect_reg(&args[1], Kind::Int);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .record_known_result_int_typed_args(fn_idx, result_reg, &call_args);
        }
        "record_known_result_ref" => {
            let fn_idx = expect_small_u16(&args[0]);
            let result_reg = expect_reg(&args[1], Kind::Ref);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .record_known_result_ref_typed_args(fn_idx, result_reg, &call_args);
        }
        other => panic!(
            "assemble(): unimplemented opname {:?} — add a builder mapping in jit/assembler.rs",
            other
        ),
    }
}

/// `assembler.py:197-206` inline descr registration. Called from
/// dispatch arms that actually consume a descr operand in the emitted
/// bytecode.
///
/// Deduplicates on `Operand::Descr` pointer identity — matches
/// RPython's `if x not in self._descr_dict` which hashes by object id.
/// Reusing the same SSARepr descr operand twice yields a stable
/// `descrs` index both times.
fn record_descr_operand(state: &mut AssemblyState, descr: &DescrOperand) -> usize {
    let key = DescrKey(descr as *const DescrOperand);
    if let Some(&idx) = state.descr_dict.get(&key) {
        return idx;
    }
    let index = state.descrs.len();
    state.descr_dict.insert(key, index);
    match descr {
        DescrOperand::Bh(bh) => state.descrs.push(bh.clone()),
        DescrOperand::SwitchDict(switch) => {
            state.descrs.push(BhDescr::Switch {
                dict: HashMap::new(),
            });
            state.switch_descrs.push((index, switch.labels.clone()));
        }
    }
    index
}

/// Helper for dispatch arms: decode an `Operand::Descr`, register it
/// inline, and return the 16-bit descr index. Used by ops that actually
/// emit a `d` argcode slot (`assembler.py:205-207`).
#[allow(dead_code)]
fn emit_descr(state: &mut AssemblyState, op: &Operand) -> u16 {
    match op {
        Operand::Descr(descr) => u16::try_from(record_descr_operand(state, descr))
            .expect("too many descrs (index > u16::MAX)"),
        _ => panic!("expected Descr operand, got {:?}", op),
    }
}

fn expect_result_or_first_reg(args: &[Operand], result: Option<&Register>, kind: Kind) -> u16 {
    match result {
        Some(reg) if reg.kind == kind => reg.index,
        Some(reg) => panic!("expected result register of kind {:?}, got {:?}", kind, reg),
        None => expect_reg(&args[0], kind),
    }
}

fn expect_result_reg(result: Option<&Register>, kind: Kind, msg: &str) -> u16 {
    match result {
        Some(reg) if reg.kind == kind => reg.index,
        Some(reg) => panic!("expected result register of kind {:?}, got {:?}", kind, reg),
        None => panic!("{}", msg),
    }
}

fn expect_reg(op: &Operand, expected: Kind) -> u16 {
    match op {
        Operand::Register(Register { kind, index }) if *kind == expected => *index,
        _ => panic!("expected Register({:?}, _), got {:?}", expected, op),
    }
}

fn expect_tlabel(op: &Operand) -> &TLabel {
    match op {
        Operand::TLabel(label) => label,
        _ => panic!("expected TLabel, got {:?}", op),
    }
}

fn expect_small_u16(op: &Operand) -> u16 {
    match op {
        Operand::ConstInt(value) => u16::try_from(*value).expect("expected u16-sized ConstInt"),
        _ => panic!("expected ConstInt(u16), got {:?}", op),
    }
}

/// `record_binop_*` / `record_unary_*` arg decoder.
///
/// The recorded op is passed as `Operand::OpCode(majit_ir::OpCode)`
/// matching `JitCodeBuilder::record_*`'s signature. RPython's
/// `codewriter/assembler.py` does not narrow this to a fixed enum; the
/// opcode is an `AbstractDescr` equivalent and the record path in the
/// metainterp consumes whatever is passed, so any valid `OpCode` must
/// round-trip through here without a hand-maintained allowlist.
fn expect_opcode(op: &Operand) -> majit_ir::OpCode {
    match op {
        Operand::OpCode(code) => *code,
        _ => panic!("expected OpCode operand, got {:?}", op),
    }
}

fn expect_list_regs(op: &Operand, expected: Kind) -> Vec<u8> {
    match op {
        Operand::ListOfKind(ListOfKind { kind, content }) if *kind == expected => content
            .iter()
            .map(|item| {
                u8::try_from(expect_reg(item, expected)).expect("register index exceeds u8")
            })
            .collect(),
        _ => panic!("expected ListOfKind({:?}), got {:?}", expected, op),
    }
}

fn expect_call_args(args: &[Operand]) -> Vec<JitCallArg> {
    args.iter().map(expect_call_arg).collect()
}

fn expect_call_arg(op: &Operand) -> JitCallArg {
    match op {
        Operand::Register(Register { kind, index }) => match kind {
            Kind::Int => JitCallArg::int(*index),
            Kind::Ref => JitCallArg::reference(*index),
            Kind::Float => JitCallArg::float(*index),
        },
        _ => panic!("expected typed call register, got {:?}", op),
    }
}

fn expect_move(args: &[Operand], result: Option<&Register>, kind: Kind) -> (u16, u16) {
    if let Some(dst) = result {
        return (dst.index, expect_reg(&args[0], kind));
    }
    (expect_reg(&args[0], kind), expect_reg(&args[1], kind))
}

fn expect_load_const_i(args: &[Operand], result: Option<&Register>) -> (u16, i64) {
    if let Some(dst) = result {
        let Operand::ConstInt(value) = &args[0] else {
            panic!("load_const_i expects ConstInt, got {:?}", args[0]);
        };
        return (dst.index, *value);
    }
    let Operand::ConstInt(value) = &args[1] else {
        panic!("load_const_i expects ConstInt, got {:?}", args[1]);
    };
    (expect_reg(&args[0], Kind::Int), *value)
}

fn expect_load_const_r(args: &[Operand], result: Option<&Register>) -> (u16, i64) {
    if let Some(dst) = result {
        let Operand::ConstRef(value) = &args[0] else {
            panic!("load_const_r expects ConstRef, got {:?}", args[0]);
        };
        return (dst.index, *value);
    }
    let Operand::ConstRef(value) = &args[1] else {
        panic!("load_const_r expects ConstRef, got {:?}", args[1]);
    };
    (expect_reg(&args[0], Kind::Ref), *value)
}

fn expect_load_const_f(args: &[Operand], result: Option<&Register>) -> (u16, i64) {
    if let Some(dst) = result {
        let Operand::ConstFloat(value) = &args[0] else {
            panic!("load_const_f expects ConstFloat, got {:?}", args[0]);
        };
        return (dst.index, *value);
    }
    let Operand::ConstFloat(value) = &args[1] else {
        panic!("load_const_f expects ConstFloat, got {:?}", args[1]);
    };
    (expect_reg(&args[0], Kind::Float), *value)
}

#[cfg(test)]
mod tests {
    use super::super::flatten::{DescrOperand, Label, SwitchDictDescr};
    use super::*;

    fn r(kind: Kind, index: u16) -> Register {
        Register::new(kind, index)
    }

    #[test]
    fn assemble_empty_ssarepr_produces_valid_jitcode() {
        let mut ssarepr = SSARepr::new("empty");
        let jitcode = assemble(&mut ssarepr, JitCodeBuilder::default(), None);
        assert_eq!(jitcode.name, "empty");
        assert!(jitcode.code.is_empty(), "empty SSARepr -> empty code");
        assert_eq!(ssarepr.insns_pos, Some(Vec::new()));
    }

    #[test]
    fn assemble_records_insn_positions_and_liveness() {
        let mut ssarepr = SSARepr::new("live");
        ssarepr
            .insns
            .push(Insn::Live(vec![Operand::Register(r(Kind::Ref, 0))]));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let mut assembler = Assembler::new();
        let jitcode = assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        assert_eq!(assembler.num_liveness_ops, 1);
        assert!(!assembler.all_liveness().is_empty());
        assert_eq!(ssarepr.insns_pos.as_ref().map(Vec::len), Some(2));
        assert!(!jitcode.code.is_empty());
    }

    #[test]
    fn assemble_patches_jumps_through_labels() {
        let mut ssarepr = SSARepr::new("jump");
        ssarepr
            .insns
            .push(Insn::op("goto", vec![Operand::TLabel(TLabel::new("L1"))]));
        ssarepr.insns.push(Insn::Label(Label::new("L1")));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        assert_eq!(jitcode.follow_jump(3), 3);
    }

    /// `assembler.py:197-206` parity: a `Descr` operand attached to an op
    /// that does not consume a descr in its emitted bytecode MUST NOT be
    /// registered in `jitcode.descrs`. Registration happens inline during
    /// per-op encoding — no pre-scan.
    #[test]
    fn assemble_does_not_register_unconsumed_descr() {
        let mut switch = SwitchDictDescr::new();
        switch.labels.push((4, TLabel::new("L2")));

        let mut ssarepr = SSARepr::new("switch");
        ssarepr.insns.push(Insn::Label(Label::new("L2")));
        // `abort_permanent` does not consume a descr in its bytecode (it
        // emits a single BC_ABORT_PERMANENT byte), so the attached
        // Descr operand is ignored by the assembler.
        ssarepr.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::Descr(DescrOperand::SwitchDict(switch))],
        ));

        let jitcode = assemble(&mut ssarepr, JitCodeBuilder::default(), None);

        assert!(
            jitcode.descrs.is_empty(),
            "descrs attached to non-consuming ops must not be registered; \
             assembler.py:197-206 registers inline at 'd' argcode emission only, \
             got descrs={:?}",
            jitcode.descrs
        );
    }

    // TODO: once a descr-consuming op (e.g. `switch`, `getfield_gc_d`) is
    // ported into `dispatch_op`, add a positive test that
    // `record_descr_operand` is invoked during its encoding and that
    // `SwitchDictDescr._labels` → `BhDescr::Switch.dict` round-trips via
    // `fix_labels()`.
}
