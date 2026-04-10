/// JitCodeBuilder — bytecode assembler for JitCode construction.
///
/// RPython codewriter.py: assembler that emits bytecodes into a JitCode object.
use std::cmp::max;

use majit_backend::JitCellToken;
use majit_ir::OpCode;

use super::{
    BC_ABORT, BC_ABORT_PERMANENT, BC_ARRAYLEN_VABLE, BC_BRANCH_REG_ZERO, BC_BRANCH_ZERO,
    BC_CALL_ASSEMBLER_FLOAT, BC_CALL_ASSEMBLER_INT, BC_CALL_ASSEMBLER_REF, BC_CALL_ASSEMBLER_VOID,
    BC_CALL_FLOAT, BC_CALL_INT, BC_CALL_LOOPINVARIANT_FLOAT, BC_CALL_LOOPINVARIANT_INT,
    BC_CALL_LOOPINVARIANT_REF, BC_CALL_LOOPINVARIANT_VOID, BC_CALL_MAY_FORCE_FLOAT,
    BC_CALL_MAY_FORCE_INT, BC_CALL_MAY_FORCE_REF, BC_CALL_MAY_FORCE_VOID, BC_CALL_PURE_FLOAT,
    BC_CALL_PURE_INT, BC_CALL_PURE_REF, BC_CALL_REF, BC_CALL_RELEASE_GIL_FLOAT,
    BC_CALL_RELEASE_GIL_INT, BC_CALL_RELEASE_GIL_REF, BC_CALL_RELEASE_GIL_VOID,
    BC_COND_CALL_VALUE_INT, BC_COND_CALL_VALUE_REF, BC_COND_CALL_VOID, BC_COPY_FROM_BOTTOM,
    BC_DUP_STACK, BC_FLOAT_GUARD_VALUE, BC_GETARRAYITEM_VABLE_F, BC_GETARRAYITEM_VABLE_I,
    BC_GETARRAYITEM_VABLE_R, BC_GETFIELD_VABLE_F, BC_GETFIELD_VABLE_I, BC_GETFIELD_VABLE_R,
    BC_HINT_FORCE_VIRTUALIZABLE, BC_INLINE_CALL, BC_INT_GUARD_VALUE, BC_JIT_MERGE_POINT, BC_JUMP,
    BC_JUMP_TARGET, BC_LOAD_CONST_F, BC_LOAD_CONST_I, BC_LOAD_CONST_R, BC_LOAD_STATE_ARRAY,
    BC_LOAD_STATE_FIELD, BC_LOAD_STATE_VARRAY, BC_MOVE_F, BC_MOVE_I, BC_MOVE_R, BC_PEEK_I,
    BC_POP_DISCARD, BC_POP_F, BC_POP_I, BC_POP_R, BC_PUSH_F, BC_PUSH_I, BC_PUSH_R, BC_PUSH_TO,
    BC_RAISE, BC_RECORD_BINOP_F, BC_RECORD_BINOP_I, BC_RECORD_KNOWN_RESULT_INT,
    BC_RECORD_KNOWN_RESULT_REF, BC_RECORD_UNARY_F, BC_RECORD_UNARY_I, BC_REF_GUARD_VALUE,
    BC_REF_RETURN, BC_REQUIRE_STACK, BC_RERAISE, BC_RESIDUAL_CALL_VOID, BC_SET_SELECTED,
    BC_SETARRAYITEM_VABLE_F, BC_SETARRAYITEM_VABLE_I, BC_SETARRAYITEM_VABLE_R, BC_SETFIELD_VABLE_F,
    BC_SETFIELD_VABLE_I, BC_SETFIELD_VABLE_R, BC_STORE_DOWN, BC_STORE_STATE_ARRAY,
    BC_STORE_STATE_FIELD, BC_STORE_STATE_VARRAY, BC_SWAP_STACK, JitArgKind, JitCallArg,
    JitCallAssemblerTarget, JitCallTarget, JitCode,
};

#[derive(Default)]
pub struct JitCodeBuilder {
    /// RPython `jitcode.py:15` `self.name = name`. Propagated to the
    /// finished `JitCode` by `finish()`; empty by default until a
    /// caller provides the source function name via `set_name`.
    name: String,
    code: Vec<u8>,
    num_regs_i: u16,
    num_regs_r: u16,
    num_regs_f: u16,
    constants_i: Vec<i64>,
    opcodes: Vec<OpCode>,
    labels: Vec<Option<usize>>,
    patches: Vec<(usize, usize)>,
    sub_jitcodes: Vec<JitCode>,
    fn_ptrs: Vec<JitCallTarget>,
    assembler_targets: Vec<JitCallAssemblerTarget>,
    has_abort: bool,
}

impl JitCodeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// RPython `jitcode.py:14-15` `__init__(name, ...)`: set the symbolic
    /// name used by `dump()` / `Display`. Callers that know the source
    /// function name should call this before `finish()`.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Whether `abort()` was called on this builder.
    ///
    /// pyre-specific: when true the caller should treat the resulting
    /// JitCode as non-executable and fall back to the interpreter.
    /// RPython has no equivalent because a translator that emits a bad
    /// bytecode crashes the build via AssertionError instead. Keeping
    /// this flag on the builder (not on `JitCode` itself) lets the
    /// metainterp-side `JitCode` stay aligned with RPython jitcode.py.
    pub fn has_abort_flag(&self) -> bool {
        self.has_abort
    }

    /// Pyre-specific: force-set the abort flag from an outer pipeline
    /// step (e.g. liveness overflow after `finish()` has already packed
    /// the JitCode). Updates the builder state but since finish() has
    /// consumed self, callers use this before finish() or on a new
    /// tracking variable.
    pub fn set_abort_flag(&mut self, v: bool) {
        self.has_abort = v;
    }

    /// Current bytecode emission position.
    pub fn current_pos(&self) -> usize {
        self.code.len()
    }

    pub fn add_const_i(&mut self, value: i64) -> u16 {
        if let Some(index) = self
            .constants_i
            .iter()
            .position(|&existing| existing == value)
        {
            return index as u16;
        }
        let index = self.constants_i.len() as u16;
        self.constants_i.push(value);
        index
    }

    pub fn load_const_i_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_i(value);
        self.load_const_i(dst, const_idx);
    }

    pub fn load_const_i(&mut self, dst: u16, const_idx: u16) {
        self.touch_reg(dst);
        self.push_u8(BC_LOAD_CONST_I);
        self.push_u16(dst);
        self.push_u16(const_idx);
    }

    pub fn pop_i(&mut self, dst: u16) {
        self.touch_reg(dst);
        self.push_u8(BC_POP_I);
        self.push_u16(dst);
    }

    pub fn peek_i(&mut self, dst: u16) {
        self.touch_reg(dst);
        self.push_u8(BC_PEEK_I);
        self.push_u16(dst);
    }

    pub fn push_i(&mut self, src: u16) {
        self.touch_reg(src);
        self.push_u8(BC_PUSH_I);
        self.push_u16(src);
    }

    pub fn pop_discard(&mut self) {
        self.push_u8(BC_POP_DISCARD);
    }

    pub fn dup_stack(&mut self) {
        self.push_u8(BC_DUP_STACK);
    }

    pub fn swap_stack(&mut self) {
        self.push_u8(BC_SWAP_STACK);
    }

    // ── State field access (register/tape machines) ──

    /// Load a scalar state field value into an int register.
    pub fn load_state_field(&mut self, field_idx: u16, dest: u16) {
        self.touch_reg(dest);
        self.push_u8(BC_LOAD_STATE_FIELD);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    /// Store an int register value into a scalar state field.
    pub fn store_state_field(&mut self, field_idx: u16, src: u16) {
        self.touch_reg(src);
        self.push_u8(BC_STORE_STATE_FIELD);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    /// Load an array state field element into an int register.
    /// The element index comes from another int register.
    pub fn load_state_array(&mut self, array_idx: u16, index_reg: u16, dest: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        self.push_u8(BC_LOAD_STATE_ARRAY);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    /// Store an int register value into an array state field element.
    /// The element index comes from another int register.
    pub fn store_state_array(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.push_u8(BC_STORE_STATE_ARRAY);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    // ── First-class virtualizable access (getfield_vable_*) ──

    pub fn vable_getfield_int(&mut self, dest: u16, field_idx: u16) {
        self.touch_reg(dest);
        self.push_u8(BC_GETFIELD_VABLE_I);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_getfield_ref(&mut self, dest: u16, field_idx: u16) {
        self.touch_ref_reg(dest);
        self.push_u8(BC_GETFIELD_VABLE_R);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_getfield_float(&mut self, dest: u16, field_idx: u16) {
        self.touch_float_reg(dest);
        self.push_u8(BC_GETFIELD_VABLE_F);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_setfield_int(&mut self, field_idx: u16, src: u16) {
        self.touch_reg(src);
        self.push_u8(BC_SETFIELD_VABLE_I);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_setfield_ref(&mut self, field_idx: u16, src: u16) {
        self.touch_ref_reg(src);
        self.push_u8(BC_SETFIELD_VABLE_R);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_setfield_float(&mut self, field_idx: u16, src: u16) {
        self.touch_float_reg(src);
        self.push_u8(BC_SETFIELD_VABLE_F);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_getarrayitem_int(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_reg(dest);
        self.touch_reg(index_reg);
        self.push_u8(BC_GETARRAYITEM_VABLE_I);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_getarrayitem_ref(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_ref_reg(dest);
        self.touch_reg(index_reg);
        self.push_u8(BC_GETARRAYITEM_VABLE_R);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_getarrayitem_float(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_float_reg(dest);
        self.touch_reg(index_reg);
        self.push_u8(BC_GETARRAYITEM_VABLE_F);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_setarrayitem_int(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.push_u8(BC_SETARRAYITEM_VABLE_I);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn vable_setarrayitem_ref(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_ref_reg(src);
        self.push_u8(BC_SETARRAYITEM_VABLE_R);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn vable_setarrayitem_float(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_float_reg(src);
        self.push_u8(BC_SETARRAYITEM_VABLE_F);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn vable_arraylen(&mut self, dest: u16, array_idx: u16) {
        self.touch_reg(dest);
        self.push_u8(BC_ARRAYLEN_VABLE);
        self.push_u16(array_idx);
        self.push_u16(dest);
    }

    pub fn vable_force(&mut self) {
        self.push_u8(BC_HINT_FORCE_VIRTUALIZABLE);
    }

    /// Load from a virtualizable state array: emit GETARRAYITEM_RAW_I.
    /// The array stays on heap; only ptr+len are tracked as inputargs.
    pub fn load_state_varray(&mut self, array_idx: u16, index_reg: u16, dest: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        self.push_u8(BC_LOAD_STATE_VARRAY);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    /// Store to a virtualizable state array: emit SETARRAYITEM_RAW.
    /// The array stays on heap; only ptr+len are tracked as inputargs.
    pub fn store_state_varray(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.push_u8(BC_STORE_STATE_VARRAY);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn copy_from_bottom(&mut self, idx_reg: u16) {
        self.touch_reg(idx_reg);
        self.push_u8(BC_COPY_FROM_BOTTOM);
        self.push_u16(idx_reg);
    }

    /// Pop top of stack and store at `index` (from bottom, 0-based).
    pub fn store_down(&mut self, idx_reg: u16) {
        self.touch_reg(idx_reg);
        self.push_u8(BC_STORE_DOWN);
        self.push_u16(idx_reg);
    }

    pub fn record_binop_i(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let opcode_idx = self.intern_opcode(opcode);
        self.touch_reg(dst);
        self.touch_reg(lhs);
        self.touch_reg(rhs);
        self.push_u8(BC_RECORD_BINOP_I);
        self.push_u16(dst);
        self.push_u16(opcode_idx);
        self.push_u16(lhs);
        self.push_u16(rhs);
    }

    pub fn record_unary_i(&mut self, dst: u16, opcode: OpCode, src: u16) {
        let opcode_idx = self.intern_opcode(opcode);
        self.touch_reg(dst);
        self.touch_reg(src);
        self.push_u8(BC_RECORD_UNARY_I);
        self.push_u16(dst);
        self.push_u16(opcode_idx);
        self.push_u16(src);
    }

    pub fn require_stack(&mut self, required: u16) {
        self.push_u8(BC_REQUIRE_STACK);
        self.push_u16(required);
    }

    pub fn branch_zero(&mut self) {
        self.push_u8(BC_BRANCH_ZERO);
    }

    pub fn new_label(&mut self) -> u16 {
        let label = self.labels.len() as u16;
        self.labels.push(None);
        label
    }

    pub fn mark_label(&mut self, label: u16) {
        let slot = self
            .labels
            .get_mut(label as usize)
            .expect("jitcode label out of bounds");
        *slot = Some(self.code.len());
    }

    pub fn branch_reg_zero(&mut self, reg: u16, label: u16) {
        self.touch_reg(reg);
        self.push_u8(BC_BRANCH_REG_ZERO);
        self.push_u16(reg);
        self.push_label_ref(label);
    }

    pub fn jump(&mut self, label: u16) {
        self.push_u8(BC_JUMP);
        self.push_label_ref(label);
    }

    pub fn jump_target(&mut self) {
        self.push_u8(BC_JUMP_TARGET);
    }

    /// blackhole.py:1066 bhimpl_jit_merge_point: portal merge point.
    pub fn jit_merge_point(&mut self) {
        self.push_u8(BC_JIT_MERGE_POINT);
    }

    pub fn abort(&mut self) {
        self.push_u8(BC_ABORT);
        self.has_abort = true;
    }

    /// RPython bhimpl_ref_return: emit return-ref opcode.
    /// The return value is in register `src`.
    pub fn ref_return(&mut self, src: u16) {
        self.push_u8(BC_REF_RETURN);
        self.push_u16(src);
    }

    pub fn abort_permanent(&mut self) {
        self.push_u8(BC_ABORT_PERMANENT);
    }

    /// blackhole.py bhimpl_raise(excvalue): raise exception from register.
    pub fn emit_raise(&mut self, src: u16) {
        self.push_u8(BC_RAISE);
        self.push_u16(src);
    }

    /// blackhole.py bhimpl_reraise(): re-raise exception_last_value.
    pub fn emit_reraise(&mut self) {
        self.push_u8(BC_RERAISE);
    }

    /// pyjitpl.py opimpl_int_guard_value: promote int register to constant.
    ///
    /// Blackhole: no-op (value passes through).
    /// Tracing: emits GUARD_VALUE to specialize the trace on this value.
    pub fn int_guard_value(&mut self, src: u16) {
        self.push_u8(BC_INT_GUARD_VALUE);
        self.push_u16(src);
    }

    /// pyjitpl.py opimpl_ref_guard_value: promote ref register to constant.
    pub fn ref_guard_value(&mut self, src: u16) {
        self.push_u8(BC_REF_GUARD_VALUE);
        self.push_u16(src);
    }

    /// pyjitpl.py opimpl_float_guard_value: promote float register to constant.
    pub fn float_guard_value(&mut self, src: u16) {
        self.push_u8(BC_FLOAT_GUARD_VALUE);
        self.push_u16(src);
    }

    pub fn inline_call(&mut self, sub_jitcode_idx: u16) {
        self.inline_call_i(sub_jitcode_idx, &[], None);
    }

    pub fn inline_call_i(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(u16, u16)],
        return_i: Option<(u16, u16)>,
    ) {
        self.inline_call_full(sub_jitcode_idx, args, return_i, None, None);
    }

    pub fn inline_call_r(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(u16, u16)],
        return_r: Option<(u16, u16)>,
    ) {
        self.inline_call_full(sub_jitcode_idx, args, None, return_r, None);
    }

    pub fn inline_call_f(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(u16, u16)],
        return_f: Option<(u16, u16)>,
    ) {
        self.inline_call_full(sub_jitcode_idx, args, None, None, return_f);
    }

    /// Inline call with typed arguments and a typed return slot.
    ///
    /// `args` maps each typed caller register to a callee register,
    /// `return_kind` selects which register file the return is routed through:
    /// 0 = Int, 1 = Ref, 2 = Float.
    pub fn inline_call_with_typed_args(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(JitArgKind, u16, u16)],
        return_slot: Option<(u16, u16)>,
        return_kind: u8,
    ) {
        let (return_i, return_r, return_f) = match return_kind {
            1 => (None, return_slot, None),
            2 => (None, None, return_slot),
            _ => (return_slot, None, None),
        };
        self.inline_call_typed(sub_jitcode_idx, args, return_i, return_r, return_f);
    }

    fn inline_call_full(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(u16, u16)],
        return_i: Option<(u16, u16)>,
        return_r: Option<(u16, u16)>,
        return_f: Option<(u16, u16)>,
    ) {
        // Default: all args are int-typed when using (u16, u16) pairs
        let typed_args: Vec<_> = args
            .iter()
            .map(|&(src, dst)| (JitArgKind::Int, src, dst))
            .collect();
        self.inline_call_typed(sub_jitcode_idx, &typed_args, return_i, return_r, return_f);
    }

    fn inline_call_typed(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(JitArgKind, u16, u16)],
        return_i: Option<(u16, u16)>,
        return_r: Option<(u16, u16)>,
        return_f: Option<(u16, u16)>,
    ) {
        for &(kind, caller_src, _) in args {
            match kind {
                JitArgKind::Int => self.touch_reg(caller_src),
                JitArgKind::Ref => self.touch_ref_reg(caller_src),
                JitArgKind::Float => self.touch_float_reg(caller_src),
            }
        }
        if let Some((_, caller_dst)) = return_i {
            self.touch_reg(caller_dst);
        }
        if let Some((_, caller_dst)) = return_r {
            self.touch_ref_reg(caller_dst);
        }
        if let Some((_, caller_dst)) = return_f {
            self.touch_float_reg(caller_dst);
        }
        self.push_u8(BC_INLINE_CALL);
        self.push_u16(sub_jitcode_idx);
        self.push_u16(args.len() as u16);
        for &(kind, caller_src, callee_dst) in args {
            self.push_u8(kind.encode());
            self.push_u16(caller_src);
            self.push_u16(callee_dst);
        }
        self.push_return_slot(return_i);
        self.push_return_slot(return_r);
        self.push_return_slot(return_f);
    }

    fn push_return_slot(&mut self, ret: Option<(u16, u16)>) {
        match ret {
            Some((callee_src, caller_dst)) => {
                self.push_u16(callee_src);
                self.push_u16(caller_dst);
            }
            None => {
                self.push_u16(u16::MAX);
                self.push_u16(u16::MAX);
            }
        }
    }

    pub fn residual_call_void(&mut self, fn_ptr_idx: u16, src_reg: u16) {
        self.residual_call_void_args(fn_ptr_idx, &[src_reg]);
    }

    pub fn residual_call_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.residual_call_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn residual_call_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(BC_RESIDUAL_CALL_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_may_force_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_may_force_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(BC_CALL_MAY_FORCE_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_release_gil_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_release_gil_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(BC_CALL_RELEASE_GIL_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_loopinvariant_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_loopinvariant_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(BC_CALL_LOOPINVARIANT_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_assembler_void_args(&mut self, target_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_void_typed_args(target_idx, &args);
    }

    pub fn call_assembler_void_typed_args(&mut self, target_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_assembler_void_like(BC_CALL_ASSEMBLER_VOID, target_idx, arg_regs);
    }

    pub fn call_may_force_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(BC_CALL_MAY_FORCE_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_int_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_int_like(BC_CALL_RELEASE_GIL_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_loopinvariant_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_int_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_int_like(BC_CALL_LOOPINVARIANT_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_assembler_int(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_int_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_int_typed(&mut self, target_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_assembler_int_like(BC_CALL_ASSEMBLER_INT, target_idx, arg_regs, dst);
    }

    pub fn call_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(BC_CALL_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(BC_CALL_PURE_INT, fn_ptr_idx, arg_regs, dst);
    }

    // ── conditional_call / record_known_result (jtransform.py:1665-1688, 292-313) ──

    /// RPython: `conditional_call_ir_v(condition, funcptr, calldescr, [i], [r])`
    /// Condition in cond_reg; if nonzero, call func with args. Result void.
    /// `typed_args` carries per-argument kind (int/ref) — RPython make_three_lists parity.
    pub fn conditional_call_void_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        cond_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.touch_reg(cond_reg);
        self.call_cond_like(BC_COND_CALL_VOID, fn_ptr_idx, cond_reg, typed_args);
    }

    /// RPython: `conditional_call_value_ir_i(value, funcptr, calldescr, [i], [r])`
    pub fn conditional_call_value_int_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        value_reg: u16,
        typed_args: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_reg(value_reg);
        self.touch_reg(dst);
        self.call_cond_value_like(
            BC_COND_CALL_VALUE_INT,
            fn_ptr_idx,
            value_reg,
            typed_args,
            dst,
        );
    }

    /// RPython: `conditional_call_value_ir_r`
    pub fn conditional_call_value_ref_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        value_reg: u16,
        typed_args: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_reg(value_reg);
        self.touch_reg(dst);
        self.call_cond_value_like(
            BC_COND_CALL_VALUE_REF,
            fn_ptr_idx,
            value_reg,
            typed_args,
            dst,
        );
    }

    /// RPython: `record_known_result_i_ir_v(result, funcptr, calldescr, [i], [r])`
    pub fn record_known_result_int_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        result_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.touch_reg(result_reg);
        self.call_cond_like(
            BC_RECORD_KNOWN_RESULT_INT,
            fn_ptr_idx,
            result_reg,
            typed_args,
        );
    }

    /// RPython: `record_known_result_r_ir_v`
    pub fn record_known_result_ref_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        result_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.touch_reg(result_reg);
        self.call_cond_like(
            BC_RECORD_KNOWN_RESULT_REF,
            fn_ptr_idx,
            result_reg,
            typed_args,
        );
    }

    fn call_cond_like(&mut self, bc: u8, fn_ptr_idx: u16, first_reg: u16, args: &[JitCallArg]) {
        self.push_u8(bc);
        self.push_u16(first_reg);
        self.push_u16(fn_ptr_idx);
        self.push_u8(args.len() as u8);
        for arg in args {
            self.push_u8(arg.kind as u8);
        }
        for arg in args {
            self.push_u16(arg.reg);
        }
    }

    fn call_cond_value_like(
        &mut self,
        bc: u8,
        fn_ptr_idx: u16,
        value_reg: u16,
        args: &[JitCallArg],
        dst: u16,
    ) {
        self.push_u8(bc);
        self.push_u16(value_reg);
        self.push_u16(fn_ptr_idx);
        self.push_u8(args.len() as u8);
        for arg in args {
            self.push_u8(arg.kind as u8);
        }
        for arg in args {
            self.push_u16(arg.reg);
        }
        self.push_u16(dst);
    }

    pub fn set_selected(&mut self, const_idx: u16) {
        self.push_u8(BC_SET_SELECTED);
        self.push_u16(const_idx);
    }

    pub fn push_to(&mut self, src_reg: u16, target_stack: u16) {
        self.touch_reg(src_reg);
        self.push_u8(BC_PUSH_TO);
        self.push_u16(src_reg);
        self.push_u16(target_stack);
    }

    pub fn move_i(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_reg(src);
        self.push_u8(BC_MOVE_I);
        self.push_u16(dst);
        self.push_u16(src);
    }

    pub fn ensure_i_regs(&mut self, count: u16) {
        self.num_regs_i = max(self.num_regs_i, count);
    }

    pub fn ensure_r_regs(&mut self, count: u16) {
        self.num_regs_r = max(self.num_regs_r, count);
    }

    pub fn ensure_f_regs(&mut self, count: u16) {
        self.num_regs_f = max(self.num_regs_f, count);
    }

    // ── Ref-typed builder methods ─────────────────────────────

    pub fn load_const_r_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_i(value);
        self.load_const_r(dst, const_idx);
    }

    pub fn load_const_r(&mut self, dst: u16, const_idx: u16) {
        self.touch_ref_reg(dst);
        self.push_u8(BC_LOAD_CONST_R);
        self.push_u16(dst);
        self.push_u16(const_idx);
    }

    pub fn pop_r(&mut self, dst: u16) {
        self.touch_ref_reg(dst);
        self.push_u8(BC_POP_R);
        self.push_u16(dst);
    }

    pub fn push_r(&mut self, src: u16) {
        self.touch_ref_reg(src);
        self.push_u8(BC_PUSH_R);
        self.push_u16(src);
    }

    pub fn move_r(&mut self, dst: u16, src: u16) {
        self.touch_ref_reg(dst);
        self.touch_ref_reg(src);
        self.push_u8(BC_MOVE_R);
        self.push_u16(dst);
        self.push_u16(src);
    }

    pub fn call_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(BC_CALL_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(BC_CALL_PURE_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_may_force_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(BC_CALL_MAY_FORCE_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_ref_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_ref_like(BC_CALL_RELEASE_GIL_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_loopinvariant_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_ref_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_ref_like(BC_CALL_LOOPINVARIANT_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_assembler_ref(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_ref_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_ref_typed(&mut self, target_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_assembler_ref_like(BC_CALL_ASSEMBLER_REF, target_idx, arg_regs, dst);
    }

    // ── Float-typed builder methods ───────────────────────────

    pub fn load_const_f_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_i(value);
        self.load_const_f(dst, const_idx);
    }

    pub fn load_const_f(&mut self, dst: u16, const_idx: u16) {
        self.touch_float_reg(dst);
        self.push_u8(BC_LOAD_CONST_F);
        self.push_u16(dst);
        self.push_u16(const_idx);
    }

    pub fn pop_f(&mut self, dst: u16) {
        self.touch_float_reg(dst);
        self.push_u8(BC_POP_F);
        self.push_u16(dst);
    }

    pub fn push_f(&mut self, src: u16) {
        self.touch_float_reg(src);
        self.push_u8(BC_PUSH_F);
        self.push_u16(src);
    }

    pub fn move_f(&mut self, dst: u16, src: u16) {
        self.touch_float_reg(dst);
        self.touch_float_reg(src);
        self.push_u8(BC_MOVE_F);
        self.push_u16(dst);
        self.push_u16(src);
    }

    pub fn call_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_float_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_float_like(BC_CALL_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_float_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_float_like(BC_CALL_PURE_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_may_force_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(BC_CALL_MAY_FORCE_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(BC_CALL_RELEASE_GIL_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_loopinvariant_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(BC_CALL_LOOPINVARIANT_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_assembler_float(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_float_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_float_typed(
        &mut self,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_assembler_float_like(BC_CALL_ASSEMBLER_FLOAT, target_idx, arg_regs, dst);
    }

    pub fn record_binop_f(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let opcode_idx = self.intern_opcode(opcode);
        self.touch_float_reg(dst);
        self.touch_float_reg(lhs);
        self.touch_float_reg(rhs);
        self.push_u8(BC_RECORD_BINOP_F);
        self.push_u16(dst);
        self.push_u16(opcode_idx);
        self.push_u16(lhs);
        self.push_u16(rhs);
    }

    pub fn record_unary_f(&mut self, dst: u16, opcode: OpCode, src: u16) {
        let opcode_idx = self.intern_opcode(opcode);
        self.touch_float_reg(dst);
        self.touch_float_reg(src);
        self.push_u8(BC_RECORD_UNARY_F);
        self.push_u16(dst);
        self.push_u16(opcode_idx);
        self.push_u16(src);
    }

    pub fn add_sub_jitcode(&mut self, jitcode: JitCode) -> u16 {
        let idx = self.sub_jitcodes.len() as u16;
        self.sub_jitcodes.push(jitcode);
        idx
    }

    pub fn add_fn_ptr(&mut self, ptr: *const ()) -> u16 {
        self.add_call_target(ptr, ptr)
    }

    pub fn add_call_target(&mut self, trace_ptr: *const (), concrete_ptr: *const ()) -> u16 {
        let target = JitCallTarget::new(trace_ptr, concrete_ptr);
        if let Some(index) = self.fn_ptrs.iter().position(|existing| *existing == target) {
            return index as u16;
        }
        let idx = self.fn_ptrs.len() as u16;
        self.fn_ptrs.push(target);
        idx
    }

    pub fn add_call_assembler_target_number(
        &mut self,
        token_number: u64,
        concrete_ptr: *const (),
    ) -> u16 {
        let target = JitCallAssemblerTarget::new(token_number, concrete_ptr);
        if let Some(index) = self
            .assembler_targets
            .iter()
            .position(|existing| *existing == target)
        {
            return index as u16;
        }
        let idx = self.assembler_targets.len() as u16;
        self.assembler_targets.push(target);
        idx
    }

    pub fn add_call_assembler_target(
        &mut self,
        target: &JitCellToken,
        concrete_ptr: *const (),
    ) -> u16 {
        self.add_call_assembler_target_number(target.number, concrete_ptr)
    }

    pub fn finish(mut self) -> JitCode {
        self.patch_labels();
        JitCode {
            name: self.name,
            code: self.code,
            c_num_regs_i: self.num_regs_i,
            c_num_regs_r: self.num_regs_r,
            c_num_regs_f: self.num_regs_f,
            constants_i: self.constants_i,
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            liveness: Vec::new(),
            liveness_info: Vec::new(),
            liveness_offsets: std::collections::HashMap::new(),
            opcodes: self.opcodes,
            sub_jitcodes: self.sub_jitcodes,
            fn_ptrs: self.fn_ptrs,
            assembler_targets: self.assembler_targets,
            exception_handlers: Vec::new(),
            jit_to_py_pc: Vec::new(),
            py_to_jit_pc: Vec::new(),
            nlocals: 0,
            stack_base: 0,
            depth_at_py_pc: Vec::new(),
            jitdriver_sd: None,
            descrs: Vec::new(),
        }
    }

    fn push_u8(&mut self, value: u8) {
        self.code.push(value);
    }

    fn push_u16(&mut self, value: u16) {
        self.code.extend_from_slice(&value.to_le_bytes());
    }

    fn call_int_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_void_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_void_like(&mut self, opcode: u8, target_idx: u16, arg_regs: &[JitCallArg]) {
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(target_idx);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn push_label_ref(&mut self, label: u16) {
        let patch_offset = self.code.len();
        self.push_u16(0);
        self.patches.push((label as usize, patch_offset));
    }

    fn touch_reg(&mut self, reg: u16) {
        self.num_regs_i = max(self.num_regs_i, reg.saturating_add(1));
    }

    fn touch_ref_reg(&mut self, reg: u16) {
        self.num_regs_r = max(self.num_regs_r, reg.saturating_add(1));
    }

    fn touch_float_reg(&mut self, reg: u16) {
        self.num_regs_f = max(self.num_regs_f, reg.saturating_add(1));
    }

    fn call_ref_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_ref_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_int_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_ref_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_ref_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_float_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_float_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_float_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_float_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.push_u8(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn touch_call_arg(&mut self, arg: JitCallArg) {
        match arg.kind {
            JitArgKind::Int => self.touch_reg(arg.reg),
            JitArgKind::Ref => self.touch_ref_reg(arg.reg),
            JitArgKind::Float => self.touch_float_reg(arg.reg),
        }
    }

    fn patch_labels(&mut self) {
        for &(label_idx, patch_offset) in &self.patches {
            let target = self.labels[label_idx].expect("jitcode label was never marked") as u16;
            let bytes = target.to_le_bytes();
            self.code[patch_offset] = bytes[0];
            self.code[patch_offset + 1] = bytes[1];
        }
    }

    fn intern_opcode(&mut self, opcode: OpCode) -> u16 {
        if let Some(index) = self.opcodes.iter().position(|&existing| existing == opcode) {
            return index as u16;
        }
        let index = self.opcodes.len() as u16;
        self.opcodes.push(opcode);
        index
    }
}
