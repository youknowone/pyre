//! SSAReprEmitter — adapter that exposes a `JitCodeBuilder`-shaped API
//! but records per-op work as `Insn::Op` values into an `SSARepr`.
//!
//! Enables pyre's CPython-bytecode walker (`codewriter.rs::transform_graph_to_jitcode`)
//! to migrate from direct `JitCodeBuilder` emission (a pyre-only fusion
//! of `flatten_graph` + `assembler.assemble`) to the RPython pipeline
//! shape `flatten_graph → compute_liveness → assembler.assemble`
//! without rewriting every handler line-by-line at once. Each walker
//! handler that still calls `builder.xxx(...)` against a real
//! `JitCodeBuilder` keeps compiling if the variable is swapped for
//! `SSAReprEmitter`, because every production `xxx(...)` method is
//! mirrored here with the same signature — the only difference being
//! that the body pushes an `Insn::Op` onto `self.ssarepr` instead of
//! emitting bytes.
//!
//! Setup operations that don't map to an `Insn::Op` (register-file
//! sizing, constant-pool registration, function-pointer table,
//! virtualizable setup) pass through to an internal `JitCodeBuilder`
//! unchanged; `assemble_into_jitcode()` at the end feeds that builder
//! (as the `setup`-side carrier) into `Assembler::assemble` alongside
//! the accumulated `SSARepr`, which then emits every per-op bytecode.
//!
//! Reference: `rpython/jit/codewriter/codewriter.py:33-73`.

use majit_ir::OpCode;
use majit_metainterp::jitcode::{JitCallArg, JitCode, JitCodeBuilder};

use super::assembler::{Assembler, NumRegs};
use super::flatten::{Insn, Kind, Label, ListOfKind, Operand, Register, SSARepr, TLabel};

/// Walker-visible builder that is a `JitCodeBuilder` look-alike. Every
/// per-op method appends an `Insn::Op` to an internal `SSARepr` and
/// every setup call is forwarded to a real `JitCodeBuilder`.
pub(super) struct SSAReprEmitter {
    /// Setup-state builder — carries `fn_ptrs`, constant pools,
    /// register-file sizing, and the jitcode name. `Assembler::assemble`
    /// consumes this builder after the walker is done; the finished
    /// `JitCode` keeps these tables intact.
    builder: JitCodeBuilder,
    /// Accumulating SSARepr. `name` is assigned by `set_name`; `insns`
    /// is appended to by every per-op method and the label/live helpers.
    pub ssarepr: SSARepr,
    /// Bridges pyre's integer label IDs (walker-local u16 counter) to
    /// RPython-style `Label(name)` / `TLabel(name)` strings used inside
    /// `SSARepr.insns`. `new_label()` allocates the next id and pushes
    /// `format!("L{id}")` into this vector; `mark_label` / `jump` /
    /// `branch_reg_zero` / `catch_exception` resolve ids back to the
    /// stored name.
    label_names: Vec<String>,
    /// `Insn::Live` indices returned from `live_placeholder` for later
    /// `patch_live_offset` no-op bookkeeping (see method comment).
    pending_live_positions: Vec<usize>,
}

impl SSAReprEmitter {
    pub fn new() -> Self {
        Self {
            builder: JitCodeBuilder::default(),
            ssarepr: SSARepr::new(String::new()),
            label_names: Vec::new(),
            pending_live_positions: Vec::new(),
        }
    }

    // ---- setup passthrough (mirrors JitCodeBuilder setup API) ----

    pub fn set_name(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.builder.set_name(name.clone());
        self.ssarepr.name = name;
    }

    pub fn ensure_i_regs(&mut self, count: u16) {
        self.builder.ensure_i_regs(count);
    }

    pub fn ensure_r_regs(&mut self, count: u16) {
        self.builder.ensure_r_regs(count);
    }

    pub fn ensure_f_regs(&mut self, count: u16) {
        self.builder.ensure_f_regs(count);
    }

    pub fn num_regs_i(&self) -> u16 {
        self.builder.num_regs_i()
    }

    pub fn num_regs_r(&self) -> u16 {
        self.builder.num_regs_r()
    }

    pub fn num_regs_f(&self) -> u16 {
        self.builder.num_regs_f()
    }

    pub fn num_consts_i(&self) -> u16 {
        self.builder.num_consts_i()
    }

    pub fn num_consts_r(&self) -> u16 {
        self.builder.num_consts_r()
    }

    pub fn num_consts_f(&self) -> u16 {
        self.builder.num_consts_f()
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

    pub fn set_abort_flag(&mut self, v: bool) {
        self.builder.set_abort_flag(v);
    }

    // ---- label mechanics ----

    /// `assembler.py` uses string-named labels; pyre's walker allocates
    /// u16 ids and indexes into a `Vec<u16>`. Bridge: each new id gets
    /// a unique `L{id}` name stored here, exposed as `TLabel(name)` in
    /// the SSARepr.
    pub fn new_label(&mut self) -> u16 {
        let id = self.label_names.len() as u16;
        self.label_names.push(format!("L{}", id));
        id
    }

    pub fn mark_label(&mut self, id: u16) {
        let name = self.label_name(id);
        self.ssarepr.insns.push(Insn::Label(Label::new(name)));
    }

    fn label_name(&self, id: u16) -> String {
        self.label_names
            .get(id as usize)
            .cloned()
            .unwrap_or_else(|| panic!("SSAReprEmitter: unknown label id {}", id))
    }

    fn tlabel(&self, id: u16) -> Operand {
        Operand::TLabel(TLabel::new(self.label_name(id)))
    }

    // ---- liveness placeholders ----

    /// Emit an empty `Insn::Live` and return its position in
    /// `ssarepr.insns`. The pyre walker uses this as a pseudo-"byte
    /// offset" key to later call `patch_live_offset(offset, ...)`;
    /// under SSARepr semantics the live register set is filled by
    /// `compute_liveness` before `Assembler::assemble` runs.
    pub fn live_placeholder(&mut self) -> usize {
        let idx = self.ssarepr.insns.len();
        self.ssarepr.insns.push(Insn::Live(Vec::new()));
        self.pending_live_positions.push(idx);
        idx
    }

    /// pyre's walker passes in an interned all-liveness offset here.
    /// Under SSARepr semantics, the live set is represented as
    /// `Insn::Live` register operands, and `Assembler::assemble`
    /// re-interns them per-jitcode (`assembler.py:234-248`). We keep
    /// this method so walker call sites still compile; the offset
    /// value itself is dropped on the floor until the walker migrates
    /// to emitting `Insn::Live(vec![Operand::Register(...), ...])`
    /// directly.
    pub fn patch_live_offset(&mut self, _patch_offset: usize, _offset: u16) {
        // Intentional no-op — SSARepr owns the live set semantics.
    }

    /// Replace the `Insn::Live` placeholder at `insn_idx` with one
    /// whose args are the full live-register triple.
    ///
    /// pyre's walker produces liveness via `LiveVars` after finishing
    /// the bytecode walk; this method is the hook that transfers that
    /// externally-computed information back into the SSARepr so
    /// `Assembler::assemble`'s `encode_liveness_info` sees the correct
    /// set and delegates to `pyre_jit_trace::state::intern_liveness`
    /// with it. Future work (Phase 4) replaces this with
    /// `super::liveness::compute_liveness(&mut ssarepr)` which
    /// derives the same information from the SSARepr backward
    /// dataflow (`liveness.py:19-23`).
    pub fn fill_live_args(
        &mut self,
        insn_idx: usize,
        live_i: &[u16],
        live_r: &[u16],
        live_f: &[u16],
    ) {
        let mut args: Vec<Operand> = Vec::with_capacity(live_i.len() + live_r.len() + live_f.len());
        for &r in live_i {
            args.push(Operand::reg(Kind::Int, r));
        }
        for &r in live_r {
            args.push(Operand::reg(Kind::Ref, r));
        }
        for &r in live_f {
            args.push(Operand::reg(Kind::Float, r));
        }
        match self.ssarepr.insns.get_mut(insn_idx) {
            Some(Insn::Live(existing)) => *existing = args,
            Some(other) => panic!(
                "fill_live_args: insn at {} is not Insn::Live: {:?}",
                insn_idx, other
            ),
            None => panic!("fill_live_args: insn index {} out of bounds", insn_idx),
        }
    }

    /// pyre's walker uses `current_pos()` to record the byte offset of
    /// a given Python PC into `pc_map`. In the SSARepr world the
    /// "position" of the next-emitted instruction is the `insns` index;
    /// `Assembler::assemble` populates `ssarepr.insns_pos` post-hoc so
    /// the walker's pc_map is translated from insn index → byte offset
    /// in a single follow-up pass. Return the insn index for now.
    pub fn current_pos(&self) -> usize {
        self.ssarepr.insns.len()
    }

    // ---- per-op emission: every method below is an `Insn::Op` push ----

    pub fn move_r(&mut self, dst: u16, src: u16) {
        // flatten.py `'->', reg` parity: destination in the result slot,
        // not in args. Keeps `compute_liveness` `alive.discard(result)`
        // at liveness.rs:148 correct for register-copy semantics.
        self.push_op_with_result(
            "move_r",
            vec![Operand::reg(Kind::Ref, src)],
            Register::new(Kind::Ref, dst),
        );
    }

    pub fn move_i(&mut self, dst: u16, src: u16) {
        self.push_op_with_result(
            "move_i",
            vec![Operand::reg(Kind::Int, src)],
            Register::new(Kind::Int, dst),
        );
    }

    pub fn move_f(&mut self, dst: u16, src: u16) {
        self.push_op_with_result(
            "move_f",
            vec![Operand::reg(Kind::Float, src)],
            Register::new(Kind::Float, dst),
        );
    }

    pub fn load_const_i_value(&mut self, dst: u16, value: i64) {
        self.push_op_with_result(
            "load_const_i",
            vec![Operand::ConstInt(value)],
            Register::new(Kind::Int, dst),
        );
    }

    pub fn load_const_r_value(&mut self, dst: u16, value: i64) {
        self.push_op_with_result(
            "load_const_r",
            vec![Operand::ConstRef(value)],
            Register::new(Kind::Ref, dst),
        );
    }

    pub fn load_const_f_value(&mut self, dst: u16, value: i64) {
        self.push_op_with_result(
            "load_const_f",
            vec![Operand::ConstFloat(value)],
            Register::new(Kind::Float, dst),
        );
    }

    pub fn ref_return(&mut self, src: u16) {
        self.push_op("ref_return", vec![Operand::reg(Kind::Ref, src)]);
        // flatten.py:144-146 parity: terminator emits `('---',)` so the
        // backward liveness pass clears its alive set at block boundaries
        // and does not propagate through dead fall-through regions.
        self.ssarepr.insns.push(Insn::Unreachable);
    }

    pub fn abort(&mut self) {
        self.push_op("abort", Vec::new());
    }

    pub fn abort_permanent(&mut self) {
        self.push_op("abort_permanent", Vec::new());
    }

    pub fn emit_raise(&mut self, src: u16) {
        self.push_op("raise", vec![Operand::reg(Kind::Ref, src)]);
    }

    pub fn emit_reraise(&mut self) {
        self.push_op("reraise", Vec::new());
    }

    pub fn last_exc_value(&mut self, dst: u16) {
        // Destination in result slot so backward liveness sees the define.
        self.push_op_with_result("last_exc_value", Vec::new(), Register::new(Kind::Ref, dst));
    }

    pub fn jump(&mut self, label: u16) {
        self.push_op("jump", vec![self.tlabel(label)]);
        // flatten.py:111-112 parity: unconditional goto emits `('---',)`.
        self.ssarepr.insns.push(Insn::Unreachable);
    }

    /// RPython jtransform.py:1714-1718 handle_jit_marker__loop_header.
    /// Emits `SpaceOperation('loop_header', [c_index], None)` where c_index
    /// is the jitdriver index. pyre has a single jitdriver (PyPyJitDriver),
    /// so callers pass 0.
    pub fn loop_header(&mut self, jdindex: u16) {
        self.push_op("loop_header", vec![Operand::ConstInt(jdindex as i64)]);
    }

    pub fn branch_reg_zero(&mut self, cond: u16, label: u16) {
        self.push_op(
            "branch_reg_zero",
            vec![Operand::reg(Kind::Int, cond), self.tlabel(label)],
        );
    }

    pub fn catch_exception(&mut self, label: u16) {
        self.push_op("catch_exception", vec![self.tlabel(label)]);
    }

    fn jit_merge_point(&mut self, greens_i: &[u8], greens_r: &[u8], reds_r: &[u8]) {
        self.push_op(
            "jit_merge_point",
            vec![
                list_of_regs(Kind::Int, greens_i),
                list_of_regs(Kind::Ref, greens_r),
                list_of_regs(Kind::Ref, reds_r),
            ],
        );
    }

    /// Portal-only lowering for interp_jit.py's merge point arguments.
    /// Keeping the constant-pool/register arithmetic here matches the
    /// upstream shape more closely than open-coding it in the bytecode
    /// walker and preserves the u8 operand invariant at the point where
    /// the indices are formed.
    pub fn emit_portal_jit_merge_point(
        &mut self,
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
        self.jit_merge_point(
            &[gi_next_instr_reg, gi_is_profiled_reg],
            &[gr_pycode_reg],
            &[frame_reg, ec_reg],
        );
    }

    pub fn record_binop_i(&mut self, dst: u16, op: OpCode, lhs: u16, rhs: u16) {
        self.push_op_with_result(
            "record_binop_i",
            vec![
                Operand::OpCode(op),
                Operand::reg(Kind::Int, lhs),
                Operand::reg(Kind::Int, rhs),
            ],
            Register::new(Kind::Int, dst),
        );
    }

    pub fn vable_getfield_int(&mut self, dest: u16, field_idx: u16) {
        // dispatch_op "getfield_vable_i" reads dst from `result`,
        // args[0] as ConstInt(field_idx).
        self.push_op_with_result(
            "getfield_vable_i",
            vec![Operand::ConstInt(field_idx as i64)],
            Register::new(Kind::Int, dest),
        );
    }

    pub fn vable_getfield_ref(&mut self, dest: u16, field_idx: u16) {
        self.push_op_with_result(
            "getfield_vable_r",
            vec![Operand::ConstInt(field_idx as i64)],
            Register::new(Kind::Ref, dest),
        );
    }

    pub fn vable_setfield_int(&mut self, field_idx: u16, src: u16) {
        // dispatch_op "setfield_vable_i" reads field_idx as args[0],
        // src register as args[1]. No result.
        self.push_op(
            "setfield_vable_i",
            vec![
                Operand::ConstInt(field_idx as i64),
                Operand::reg(Kind::Int, src),
            ],
        );
    }

    pub fn vable_setfield_ref(&mut self, field_idx: u16, src: u16) {
        self.push_op(
            "setfield_vable_r",
            vec![
                Operand::ConstInt(field_idx as i64),
                Operand::reg(Kind::Ref, src),
            ],
        );
    }

    pub fn vable_getarrayitem_ref(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        // dispatch_op "getarrayitem_vable_r" reads dst from `result`,
        // args[0] ConstInt(array_idx), args[1] Register::Int(index_reg).
        self.push_op_with_result(
            "getarrayitem_vable_r",
            vec![
                Operand::ConstInt(array_idx as i64),
                Operand::reg(Kind::Int, index_reg),
            ],
            Register::new(Kind::Ref, dest),
        );
    }

    pub fn vable_setarrayitem_ref(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.push_op(
            "setarrayitem_vable_r",
            vec![
                Operand::ConstInt(array_idx as i64),
                Operand::reg(Kind::Int, index_reg),
                Operand::reg(Kind::Ref, src),
            ],
        );
    }

    pub fn call_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.push_call_like(
            "call_int",
            fn_ptr_idx,
            arg_regs,
            Some(Register::new(Kind::Int, dst)),
        );
    }

    pub fn call_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.push_call_like(
            "call_ref",
            fn_ptr_idx,
            arg_regs,
            Some(Register::new(Kind::Ref, dst)),
        );
    }

    pub fn call_may_force_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.push_call_like(
            "call_may_force_ref",
            fn_ptr_idx,
            arg_regs,
            Some(Register::new(Kind::Ref, dst)),
        );
    }

    pub fn call_may_force_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.push_call_like("call_may_force_void", fn_ptr_idx, arg_regs, None);
    }

    // ---- finalization ----

    /// `assembler.py:34-54` `assemble(ssarepr, jitcode, num_regs)`.
    /// Consume this emitter, run the accumulated SSARepr through
    /// `Assembler::assemble` against the pre-populated builder, and
    /// return the resulting `JitCode`.
    ///
    /// The `num_regs` override forwarded to `Assembler::assemble` is
    /// derived from whatever the caller already called
    /// `ensure_{i,r,f}_regs(...)` with — the emitter treats those calls
    /// as the walker's `num_regs` promise and re-hands them to the
    /// assembler so `emit_reg`'s bounds check (`assembler.py:73`) sees
    /// the same ceiling the walker computed.
    ///
    /// `assembler` must be a persistent `Assembler` shared across all
    /// jitcodes compiled in this process so `all_liveness` /
    /// `all_liveness_positions` accumulate correctly
    /// (`pyjitpl.py:2264`).
    pub fn finish_with(mut self, assembler: &mut Assembler) -> JitCode {
        let num_regs = NumRegs {
            int: self.builder.num_regs_i(),
            ref_: self.builder.num_regs_r(),
            float: self.builder.num_regs_f(),
        };
        assembler.assemble(&mut self.ssarepr, self.builder, Some(num_regs))
    }

    /// Same as `finish_with` but overrides the per-kind register-file
    /// ceiling. Mirrors RPython `transform_graph_to_jitcode`'s
    /// `num_regs` pass-through.
    pub fn finish_with_num_regs(mut self, assembler: &mut Assembler, num_regs: NumRegs) -> JitCode {
        assembler.assemble(&mut self.ssarepr, self.builder, Some(num_regs))
    }

    /// **Test-only** wrapper around `finish_with` that creates a
    /// throw-away `Assembler`.
    ///
    /// `codewriter.py:20-23` pins the `Assembler` to a single long-lived
    /// instance on `CodeWriter`; fresh assemblers per call split the
    /// per-kind liveness counters and diverge from upstream. Production
    /// paths must borrow the CodeWriter's `Assembler` via
    /// `crate::jit::codewriter::with_codewriter(...)` and call
    /// `finish_with` / `finish_with_positions` explicitly. This wrapper
    /// remains only for unit tests that do not exercise the
    /// translator-session accumulator invariant.
    #[cfg(test)]
    pub fn finish(self) -> JitCode {
        let mut assembler = Assembler::new();
        self.finish_with(&mut assembler)
    }

    /// Translate an insn-index position (returned from `current_pos()`
    /// or `live_placeholder()`) into the corresponding JitCode byte
    /// offset using the `ssarepr.insns_pos` table that
    /// `Assembler::assemble` populates (`assembler.py:41-44`).
    ///
    /// Only meaningful AFTER `finish*()` — but since those methods
    /// consume `self`, callers that need both the JitCode and the
    /// translation must invoke `finish_and_translate_positions` instead.
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

    /// Assemble and return the finished JitCode along with the
    /// translated byte offsets for each caller-provided insn index.
    /// Used by the walker to convert its `pc_map` (Python PC →
    /// insn index) into the final byte-offset form that runtime
    /// readers expect.
    pub fn finish_with_positions(
        mut self,
        assembler: &mut Assembler,
        insn_positions: &[usize],
    ) -> (JitCode, Vec<usize>) {
        let num_regs = NumRegs {
            int: self.builder.num_regs_i(),
            ref_: self.builder.num_regs_r(),
            float: self.builder.num_regs_f(),
        };
        let jitcode = assembler.assemble(&mut self.ssarepr, self.builder, Some(num_regs));
        let byte_positions =
            Self::insn_pos_to_byte_offset(&self.ssarepr, insn_positions.iter().copied());
        (jitcode, byte_positions)
    }

    /// **Test-only** twin of `finish_with_positions` that creates a
    /// throw-away `Assembler`. Production callers must supply the
    /// CodeWriter-owned `Assembler` — see the note on `finish()`.
    #[cfg(test)]
    pub fn finish_and_translate_positions(self, insn_positions: &[usize]) -> (JitCode, Vec<usize>) {
        let mut assembler = Assembler::new();
        self.finish_with_positions(&mut assembler, insn_positions)
    }

    // ---- internals ----

    fn push_op(&mut self, opname: &'static str, args: Vec<Operand>) {
        self.ssarepr.insns.push(Insn::op(opname, args));
    }

    fn push_op_with_result(&mut self, opname: &'static str, args: Vec<Operand>, result: Register) {
        self.ssarepr
            .insns
            .push(Insn::op_with_result(opname, args, result));
    }

    fn push_call_like(
        &mut self,
        opname: &'static str,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        result: Option<Register>,
    ) {
        let mut args: Vec<Operand> = Vec::with_capacity(1 + arg_regs.len());
        args.push(Operand::ConstInt(fn_ptr_idx as i64));
        for a in arg_regs {
            args.push(call_arg_to_operand(a));
        }
        match result {
            Some(reg) => self.push_op_with_result(opname, args, reg),
            None => self.push_op(opname, args),
        }
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

/// Translate a runtime `JitCallArg` back into the `Operand::Register`
/// shape the Assembler's `expect_call_arg` consumes.
fn call_arg_to_operand(arg: &JitCallArg) -> Operand {
    use majit_metainterp::jitcode::JitArgKind;
    let kind = match arg.kind {
        JitArgKind::Int => Kind::Int,
        JitArgKind::Ref => Kind::Ref,
        JitArgKind::Float => Kind::Float,
    };
    Operand::reg(kind, arg.reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitter_builds_simple_return_sequence() {
        // Build: move_r r0 <- r1 ; ref_return r0 ; '---'
        let mut em = SSAReprEmitter::new();
        em.set_name("trivial");
        em.ensure_r_regs(2);
        em.move_r(0, 1);
        em.ref_return(0);

        assert_eq!(em.ssarepr.insns.len(), 3);
        match &em.ssarepr.insns[0] {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "move_r");
                // After the push_op_with_result conversion, the
                // destination lives in `result`, leaving `args = [src]`.
                assert_eq!(args.len(), 1);
                let dst = result.expect("move_r must set a result register");
                assert_eq!(dst.index, 0);
            }
            other => panic!("expected Op, got {:?}", other),
        }
        match &em.ssarepr.insns[1] {
            Insn::Op { opname, args, .. } => {
                assert_eq!(opname, "ref_return");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected Op, got {:?}", other),
        }
        assert!(matches!(em.ssarepr.insns[2], Insn::Unreachable));

        let mut assembler = Assembler::new();
        let jitcode = em.finish_with(&mut assembler);
        assert_eq!(jitcode.name, "trivial");
        assert!(!jitcode.code.is_empty());
    }

    #[test]
    fn emitter_labels_bridge_u16_ids_to_tlabel_names() {
        // Build: L0 ; ref_return r0 ; '---' ; jump L0 ; '---'
        let mut em = SSAReprEmitter::new();
        em.set_name("loop");
        em.ensure_r_regs(1);
        let loop_label = em.new_label();
        em.mark_label(loop_label);
        em.ref_return(0);
        em.jump(loop_label);

        assert!(matches!(&em.ssarepr.insns[0], Insn::Label(l) if l.name == "L0"));
        // Index 1: ref_return, 2: '---', 3: jump, 4: '---'
        match &em.ssarepr.insns[3] {
            Insn::Op { opname, args, .. } => {
                assert_eq!(opname, "jump");
                match &args[0] {
                    Operand::TLabel(t) => assert_eq!(t.name, "L0"),
                    other => panic!("expected TLabel, got {:?}", other),
                }
            }
            other => panic!("expected Op, got {:?}", other),
        }
        assert!(matches!(em.ssarepr.insns[4], Insn::Unreachable));

        let mut assembler = Assembler::new();
        let _jitcode = em.finish_with(&mut assembler);
    }

    #[test]
    fn emitter_live_placeholder_returns_insn_index() {
        let mut em = SSAReprEmitter::new();
        em.set_name("live");
        em.ensure_r_regs(1);
        let a = em.live_placeholder();
        em.ref_return(0);
        let b = em.live_placeholder();
        // After the Live at 0 and ref_return + '---' (indices 1 and 2),
        // the next Live lands at index 3.
        assert_eq!(a, 0);
        assert_eq!(b, 3);
        assert_eq!(em.pending_live_positions, vec![0, 3]);
    }
}
