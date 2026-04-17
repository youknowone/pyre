use std::marker::PhantomData;
use std::sync::Arc;

use majit_backend::JitCellToken;
use majit_ir::{OpCode, OpRef};

use super::{MIFrame, MIFrameStack};
use crate::jitcode::{
    BC_ABORT, BC_ABORT_PERMANENT, BC_ARRAYLEN_VABLE, BC_BRANCH_REG_ZERO, BC_CALL_ASSEMBLER_FLOAT,
    BC_CALL_ASSEMBLER_INT, BC_CALL_ASSEMBLER_REF, BC_CALL_ASSEMBLER_VOID, BC_CALL_FLOAT,
    BC_CALL_INT, BC_CALL_LOOPINVARIANT_FLOAT, BC_CALL_LOOPINVARIANT_INT, BC_CALL_LOOPINVARIANT_REF,
    BC_CALL_LOOPINVARIANT_VOID, BC_CALL_MAY_FORCE_FLOAT, BC_CALL_MAY_FORCE_INT,
    BC_CALL_MAY_FORCE_REF, BC_CALL_MAY_FORCE_VOID, BC_CALL_PURE_FLOAT, BC_CALL_PURE_INT,
    BC_CALL_PURE_REF, BC_CALL_REF, BC_CALL_RELEASE_GIL_FLOAT, BC_CALL_RELEASE_GIL_INT,
    BC_CALL_RELEASE_GIL_REF, BC_CALL_RELEASE_GIL_VOID, BC_COND_CALL_VALUE_INT,
    BC_COND_CALL_VALUE_REF, BC_COND_CALL_VOID, BC_FLOAT_GUARD_VALUE, BC_GETARRAYITEM_VABLE_F,
    BC_GETARRAYITEM_VABLE_I, BC_GETARRAYITEM_VABLE_R, BC_GETFIELD_VABLE_F, BC_GETFIELD_VABLE_I,
    BC_GETFIELD_VABLE_R, BC_HINT_FORCE_VIRTUALIZABLE, BC_INLINE_CALL, BC_INT_GUARD_VALUE,
    BC_JIT_MERGE_POINT, BC_JUMP, BC_JUMP_TARGET, BC_LOAD_CONST_F, BC_LOAD_CONST_I, BC_LOAD_CONST_R,
    BC_LOAD_STATE_ARRAY, BC_LOAD_STATE_FIELD, BC_LOAD_STATE_VARRAY, BC_MOVE_F, BC_MOVE_I,
    BC_MOVE_R, BC_RECORD_BINOP_F, BC_RECORD_BINOP_I, BC_RECORD_KNOWN_RESULT_INT,
    BC_RECORD_KNOWN_RESULT_REF, BC_RECORD_UNARY_F, BC_RECORD_UNARY_I, BC_REF_GUARD_VALUE,
    BC_RESIDUAL_CALL_VOID, BC_SETARRAYITEM_VABLE_F, BC_SETARRAYITEM_VABLE_I,
    BC_SETARRAYITEM_VABLE_R, BC_SETFIELD_VABLE_F, BC_SETFIELD_VABLE_I, BC_SETFIELD_VABLE_R,
    BC_STORE_STATE_ARRAY, BC_STORE_STATE_FIELD, BC_STORE_STATE_VARRAY, JitArgKind, JitCallArg,
    JitCallTarget, JitCode, MAX_HOST_CALL_ARITY,
};
use crate::{TraceAction, TraceCtx};

pub trait JitCodeSym {
    fn begin_portal_op(&mut self, _pc: usize) {}
    fn commit_portal_op(&mut self) {}
    fn abort_portal_op(&mut self) {}
    fn total_slots(&self) -> usize;
    fn loop_header_pc(&self) -> usize;
    /// Full interpreter-visible state to materialize on guard failure.
    ///
    /// When `None`, guards fall back to the legacy auto-generated fail args.
    fn fail_args(&self) -> Option<Vec<OpRef>>;

    /// Guard-failure state materialization that may record extra IR.
    fn fail_args_with_ctx(&mut self, _ctx: &mut TraceCtx) -> Option<Vec<OpRef>> {
        self.fail_args()
    }

    /// Types of fail_args values. When Some, used instead of default all-Int.
    fn fail_args_types(&self) -> Option<Vec<majit_ir::Type>> {
        None
    }

    // -- State field support (register/tape machines) -----
    //
    // When state_fields is configured, scalar and array fields on the
    // interpreter state are tracked as OpRefs in the Sym.

    /// Read a scalar state field's current OpRef.
    fn state_field_ref(&self, _field_idx: usize) -> Option<OpRef> {
        None
    }

    /// Update a scalar state field's OpRef.
    fn set_state_field_ref(&mut self, _field_idx: usize, _value: OpRef) {}

    /// Read a scalar state field's current concrete value.
    fn state_field_value(&self, _field_idx: usize) -> Option<i64> {
        None
    }

    /// Update a scalar state field's concrete value.
    fn set_state_field_value(&mut self, _field_idx: usize, _value: i64) {}

    /// Read an array state field element's current OpRef.
    fn state_array_ref(&self, _array_idx: usize, _elem_idx: usize) -> Option<OpRef> {
        None
    }

    /// Update an array state field element's OpRef.
    fn set_state_array_ref(&mut self, _array_idx: usize, _elem_idx: usize, _value: OpRef) {}

    /// Read an array state field element's current concrete value.
    fn state_array_value(&self, _array_idx: usize, _elem_idx: usize) -> Option<i64> {
        None
    }

    /// Update an array state field element's concrete value.
    fn set_state_array_value(&mut self, _array_idx: usize, _elem_idx: usize, _value: i64) {}

    // -- State virtualizable array support ---------------
    //
    // For state_fields with `[type; virt]`: array stays on heap,
    // accessed via GetarrayitemRawI/SetarrayitemRaw. Only the data
    // pointer and length are tracked as inputargs.

    /// Get the data pointer OpRef for a virtualizable state array.
    fn state_varray_ptr(&self, _array_idx: usize) -> Option<OpRef> {
        None
    }

    /// Get the length OpRef for a virtualizable state array.
    fn state_varray_len(&self, _array_idx: usize) -> Option<OpRef> {
        None
    }
}

pub trait JitCodeRuntime {
    fn label_at(&self, pc: usize) -> usize;
}

pub struct ClosureRuntime<FLabel> {
    label_at: FLabel,
}

impl<FLabel> ClosureRuntime<FLabel> {
    pub fn new(label_at: FLabel) -> Self {
        Self { label_at }
    }
}

impl<FLabel> JitCodeRuntime for ClosureRuntime<FLabel>
where
    FLabel: Fn(usize) -> usize,
{
    fn label_at(&self, pc: usize) -> usize {
        (self.label_at)(pc)
    }
}

/// JitCode bytecode interpreter for tracing.
///
/// Borrows `frames: &'mi mut MIFrameStack` from the owning
/// `MetaInterp<M>` so that pyre's runtime keeps a single canonical
/// framestack — matching `pyjitpl.py`'s `self.framestack` invariant
/// where MIFrame.run_one_step and the metainterp share one stack.
///
/// Trace-side helpers that have no MetaInterp handy can wrap an
/// interim stack via [`StandaloneFrameStack`] — the legacy entry
/// points (`trace_jitcode`, test fixtures) take that path.
pub struct JitCodeMachine<'mi, S, R> {
    frames: &'mi mut MIFrameStack,
    marker: PhantomData<(S, R)>,
}

/// Owns an [`MIFrameStack`] for legacy `trace_jitcode` callers that
/// do not (yet) hand a `MetaInterp::framestack` borrow into the
/// jitcode interpreter.  Drops back to RPython parity once those
/// call sites migrate.
pub struct StandaloneFrameStack {
    pub frames: MIFrameStack,
}

impl StandaloneFrameStack {
    pub fn new() -> Self {
        Self {
            frames: MIFrameStack::empty(),
        }
    }
}

impl Default for StandaloneFrameStack {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct ActiveStandardVirtualizable {
    vable_opref: OpRef,
    info: crate::virtualizable::VirtualizableInfo,
    obj_ptr: *mut u8,
}

impl<'mi, S, R> JitCodeMachine<'mi, S, R>
where
    S: JitCodeSym,
    R: JitCodeRuntime,
{
    fn active_standard_virtualizable(&self, ctx: &TraceCtx) -> Option<ActiveStandardVirtualizable> {
        let vable_opref = ctx.standard_virtualizable_box()?;
        let info = ctx.virtualizable_info()?.clone();
        let obj_ptr = self.frames.frames.iter().rev().find_map(|frame| {
            frame
                .ref_regs
                .iter()
                .zip(frame.ref_values.iter())
                .find_map(|(opref, concrete)| {
                    (*opref == Some(vable_opref))
                        .then_some(*concrete)
                        .flatten()
                        .map(|value| value as usize as *mut u8)
                })
        })?;
        Some(ActiveStandardVirtualizable {
            vable_opref,
            info,
            obj_ptr,
        })
    }

    fn prepare_standard_virtualizable_before_residual_call(
        &mut self,
        ctx: &mut TraceCtx,
    ) -> Option<ActiveStandardVirtualizable> {
        let active = self.active_standard_virtualizable(ctx)?;
        unsafe {
            active.info.tracing_before_residual_call(active.obj_ptr);
        }
        let force_token = ctx.force_token();
        ctx.vable_setfield_descr(
            active.vable_opref,
            force_token,
            active.info.token_field_descr(),
        );
        Some(active)
    }

    fn finish_standard_virtualizable_after_residual_call(
        active: Option<ActiveStandardVirtualizable>,
    ) -> bool {
        let Some(active) = active else {
            return false;
        };
        unsafe { active.info.tracing_after_residual_call(active.obj_ptr) }
    }

    fn finalize_standard_virtualizable_may_force(
        ctx: &mut TraceCtx,
        sym: &mut S,
        active: Option<ActiveStandardVirtualizable>,
    ) -> TraceAction {
        if Self::finish_standard_virtualizable_after_residual_call(active) {
            TraceAction::Abort
        } else {
            ctx.guard_not_forced(sym.total_slots());
            TraceAction::Continue
        }
    }

    fn record_state_guard(
        ctx: &mut TraceCtx,
        sym: &mut S,
        opcode: OpCode,
        args: &[OpRef],
        extra_fail_args: &[OpRef],
    ) {
        if let Some(mut fail_args) = sym.fail_args_with_ctx(ctx) {
            let mut fail_types = sym.fail_args_types();
            fail_args.extend_from_slice(extra_fail_args);
            if let Some(ref mut types) = fail_types {
                types.extend(std::iter::repeat(majit_ir::Type::Int).take(extra_fail_args.len()));
            }
            if let Some(types) = fail_types {
                ctx.record_guard_typed_with_fail_args(opcode, args, types, &fail_args);
            } else {
                ctx.record_guard_with_fail_args(opcode, args, fail_args.len(), &fail_args);
            }
        } else {
            ctx.record_guard(opcode, args, sym.total_slots());
        }
    }

    fn raw_word_array_descr() -> majit_ir::DescrRef {
        majit_ir::descr::make_array_descr(0, 8, majit_ir::Type::Int)
    }

    /// pyjitpl.py: standard virtualizable → (vable_box, fielddescr).
    /// Converts a bytecode field_idx to the cached DescrRef from VirtualizableInfo.
    fn standard_vable_field_descr(
        ctx: &TraceCtx,
        field_idx: usize,
    ) -> Option<(OpRef, majit_ir::DescrRef)> {
        let vable_opref = ctx.standard_virtualizable_box()?;
        let info = ctx.virtualizable_info()?;
        let descr = info.static_field_descrs().get(field_idx)?.clone();
        Some((vable_opref, descr))
    }

    /// pyjitpl.py: standard virtualizable → (vable_box, array_field_descr).
    /// Converts a bytecode array_idx to the cached array field DescrRef.
    fn standard_vable_array_field_descr(
        ctx: &TraceCtx,
        array_idx: usize,
    ) -> Option<(OpRef, majit_ir::DescrRef)> {
        let vable_opref = ctx.standard_virtualizable_box()?;
        let info = ctx.virtualizable_info()?;
        let descr = info.array_field_descrs().get(array_idx)?.clone();
        Some((vable_opref, descr))
    }

    /// Construct a `JitCodeMachine` over an existing framestack borrow.
    ///
    /// The caller — typically `MetaInterp::trace_jitcode_with_framestack`
    /// or a [`StandaloneFrameStack`] wrapper — pushes the root MIFrame
    /// before calling and pops it after the machine returns.
    pub fn with_framestack(
        frames: &'mi mut MIFrameStack,
        _sub_jitcodes: &[Arc<JitCode>],
        _fn_ptrs: &[JitCallTarget],
    ) -> Self {
        Self {
            frames,
            marker: PhantomData,
        }
    }

    pub fn run_to_end(&mut self, ctx: &mut TraceCtx, sym: &mut S, runtime: &R) -> TraceAction {
        let portal_pc = self.frames.current_mut().pc;
        sym.begin_portal_op(portal_pc);
        while !self.frames.is_empty() {
            // Catch panics from BigInt overflow in runtime stack operations.
            // RPython doesn't have this issue (no BigInt); we abort the trace.
            let action = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.run_one_step(ctx, sym, runtime)
            })) {
                Ok(a) => a,
                Err(payload) => {
                    if crate::majit_log_enabled() {
                        let message = if let Some(msg) = payload.downcast_ref::<&str>() {
                            *msg
                        } else if let Some(msg) = payload.downcast_ref::<String>() {
                            msg.as_str()
                        } else {
                            "<non-string panic payload>"
                        };
                        eprintln!(
                            "[jit] trace_jitcode panic while tracing pc={}: {}",
                            self.frames.current_mut().pc,
                            message
                        );
                    }
                    sym.abort_portal_op();
                    return TraceAction::AbortPermanent;
                }
            };
            if !matches!(action, TraceAction::Continue) {
                match action {
                    TraceAction::CloseLoop => sym.commit_portal_op(),
                    _ => sym.abort_portal_op(),
                }
                return action;
            }
            // pyjitpl.py:2843 blackhole_if_trace_too_long — check AFTER
            // executing the step, matching RPython's _interpret() loop:
            //   self.framestack[-1].run_one_step()
            //   self.blackhole_if_trace_too_long()
            if ctx.is_too_long() {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] trace_jitcode aborting: trace too long at portal pc={}",
                        portal_pc
                    );
                }
                sym.abort_portal_op();
                return TraceAction::Abort;
            }
        }

        // Post-loop overflow check: the jitcode ran to completion (all
        // frames empty) but may have exceeded the limit on the last step.
        if ctx.is_too_long() {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] trace_jitcode aborting: trace too long at portal pc={}",
                    portal_pc
                );
            }
            sym.abort_portal_op();
            TraceAction::Abort
        } else {
            sym.commit_portal_op();
            TraceAction::Continue
        }
    }

    pub fn run_one_step(&mut self, ctx: &mut TraceCtx, sym: &mut S, runtime: &R) -> TraceAction {
        if self.frames.is_empty() {
            return TraceAction::Continue;
        }

        let finished = {
            let frame = self.frames.current_mut();
            frame.finished()
        };
        if finished {
            let finished_frame = self.frames.pop().expect("finished frame stack was empty");
            if finished_frame.inline_frame {
                ctx.pop_inline_frame();
            }
            if let Some(parent) = self.frames.frames.last_mut() {
                if let Some((callee_src, caller_dst)) = finished_frame.return_i {
                    parent.int_regs[caller_dst] = finished_frame.int_regs[callee_src];
                    parent.int_values[caller_dst] = finished_frame.int_values[callee_src];
                }
                if let Some((callee_src, caller_dst)) = finished_frame.return_r {
                    parent.ref_regs[caller_dst] = finished_frame.ref_regs[callee_src];
                    parent.ref_values[caller_dst] = finished_frame.ref_values[callee_src];
                }
                if let Some((callee_src, caller_dst)) = finished_frame.return_f {
                    parent.float_regs[caller_dst] = finished_frame.float_regs[callee_src];
                    parent.float_values[caller_dst] = finished_frame.float_values[callee_src];
                }
            }
            return TraceAction::Continue;
        }

        let bytecode = self.frames.current_mut().next_u8();
        match bytecode {
            BC_LOAD_CONST_I => {
                let (dst, value) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let const_idx = frame.next_u16() as usize;
                    let value = *frame
                        .jitcode
                        .constants_i
                        .get(const_idx)
                        .expect("jitcode const index out of bounds");
                    (dst, value)
                };
                self.set_int_reg(dst, Some(ctx.const_int(value)), Some(value));
            }

            // -- State field access (register/tape machines) --
            BC_LOAD_STATE_FIELD => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let opref = sym
                    .state_field_ref(field_idx)
                    .expect("state field not initialized");
                let value = sym
                    .state_field_value(field_idx)
                    .expect("state field concrete value not initialized");
                self.set_int_reg(dest, Some(opref), Some(value));
            }
            BC_STORE_STATE_FIELD => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let (opref, value) = self.read_int_reg(src);
                sym.set_state_field_ref(field_idx, opref);
                sym.set_state_field_value(field_idx, value);
            }
            BC_LOAD_STATE_ARRAY => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let (_, index_concrete) = self.read_int_reg(index_reg);
                let elem_idx = index_concrete as usize;
                let opref = sym.state_array_ref(array_idx, elem_idx);
                if let Some(opref) = opref {
                    let value = sym
                        .state_array_value(array_idx, elem_idx)
                        .expect("state array concrete value not initialized");
                    self.set_int_reg(dest, Some(opref), Some(value));
                } else {
                    // Array element beyond initialized range (e.g., push expanded).
                    // Abort trace -- this path needs dynamic array support.
                    return TraceAction::Abort;
                }
            }
            BC_STORE_STATE_ARRAY => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let (_, index_concrete) = self.read_int_reg(index_reg);
                let elem_idx = index_concrete as usize;
                let (opref, value) = self.read_int_reg(src);
                sym.set_state_array_ref(array_idx, elem_idx, opref);
                sym.set_state_array_value(array_idx, elem_idx, value);
            }

            // -- First-class virtualizable access (RPython getfield_vable_*) --
            BC_GETFIELD_VABLE_I => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fielddescr)) =
                    Self::standard_vable_field_descr(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let result = ctx.vable_getfield_int(vable_opref, fielddescr);
                self.set_int_reg(dest, Some(result), Some(0));
            }
            BC_GETFIELD_VABLE_R => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fielddescr)) =
                    Self::standard_vable_field_descr(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let result = ctx.vable_getfield_ref(vable_opref, fielddescr);
                self.set_ref_reg(dest, Some(result), Some(0));
            }
            BC_GETFIELD_VABLE_F => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fielddescr)) =
                    Self::standard_vable_field_descr(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let result = ctx.vable_getfield_float(vable_opref, fielddescr);
                self.set_float_reg(dest, Some(result), Some(0));
            }
            BC_SETFIELD_VABLE_I => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fielddescr)) =
                    Self::standard_vable_field_descr(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let (value, _) = self.read_int_reg(src);
                ctx.vable_setfield(vable_opref, fielddescr, value);
            }
            BC_SETFIELD_VABLE_R => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fielddescr)) =
                    Self::standard_vable_field_descr(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let (value, _) = self.read_ref_reg(src);
                ctx.vable_setfield(vable_opref, fielddescr, value);
            }
            BC_SETFIELD_VABLE_F => {
                let field_idx = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fielddescr)) =
                    Self::standard_vable_field_descr(ctx, field_idx)
                else {
                    return TraceAction::Abort;
                };
                let (value, _) = self.read_float_reg(src);
                ctx.vable_setfield(vable_opref, fielddescr, value);
            }
            BC_GETARRAYITEM_VABLE_I => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fdescr)) =
                    Self::standard_vable_array_field_descr(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, index_value) = self.read_int_reg(index_reg);
                let result =
                    ctx.vable_getarrayitem_int_indexed(vable_opref, index, index_value, fdescr);
                self.set_int_reg(dest, Some(result), Some(0));
            }
            BC_GETARRAYITEM_VABLE_R => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fdescr)) =
                    Self::standard_vable_array_field_descr(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, index_value) = self.read_int_reg(index_reg);
                let result =
                    ctx.vable_getarrayitem_ref_indexed(vable_opref, index, index_value, fdescr);
                self.set_ref_reg(dest, Some(result), Some(0));
            }
            BC_GETARRAYITEM_VABLE_F => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fdescr)) =
                    Self::standard_vable_array_field_descr(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, index_value) = self.read_int_reg(index_reg);
                let result =
                    ctx.vable_getarrayitem_float_indexed(vable_opref, index, index_value, fdescr);
                self.set_float_reg(dest, Some(result), Some(0));
            }
            BC_SETARRAYITEM_VABLE_I => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fdescr)) =
                    Self::standard_vable_array_field_descr(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, index_value) = self.read_int_reg(index_reg);
                let (value, _) = self.read_int_reg(src);
                ctx.vable_setarrayitem_indexed(vable_opref, index, index_value, fdescr, value);
            }
            BC_SETARRAYITEM_VABLE_R => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fdescr)) =
                    Self::standard_vable_array_field_descr(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, index_value) = self.read_int_reg(index_reg);
                let (value, _) = self.read_ref_reg(src);
                ctx.vable_setarrayitem_indexed(vable_opref, index, index_value, fdescr, value);
            }
            BC_SETARRAYITEM_VABLE_F => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fdescr)) =
                    Self::standard_vable_array_field_descr(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let (index, index_value) = self.read_int_reg(index_reg);
                let (value, _) = self.read_float_reg(src);
                ctx.vable_setarrayitem_indexed(vable_opref, index, index_value, fdescr, value);
            }
            BC_ARRAYLEN_VABLE => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let Some((vable_opref, fdescr)) =
                    Self::standard_vable_array_field_descr(ctx, array_idx)
                else {
                    return TraceAction::Abort;
                };
                let result = ctx.vable_arraylen_vable(vable_opref, fdescr);
                self.set_int_reg(dest, Some(result), Some(0));
            }
            BC_HINT_FORCE_VIRTUALIZABLE => {
                let Some(vable_opref) = ctx.standard_virtualizable_box() else {
                    return TraceAction::Abort;
                };
                ctx.gen_store_back_in_vable(vable_opref);
            }

            // -- Virtualizable state array access --
            // Array stays on heap; emit raw memory load/store IR ops.
            BC_LOAD_STATE_VARRAY => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let dest = self.frames.current_mut().next_u16() as usize;
                let (index_opref, _) = self.read_int_reg(index_reg);
                let array_ptr = sym
                    .state_varray_ptr(array_idx)
                    .expect("virtualizable array not initialized");
                let result = ctx.record_op_with_descr(
                    OpCode::GetarrayitemRawI,
                    &[array_ptr, index_opref],
                    Self::raw_word_array_descr(),
                );
                self.set_int_reg(dest, Some(result), Some(0));
            }
            BC_STORE_STATE_VARRAY => {
                let array_idx = self.frames.current_mut().next_u16() as usize;
                let index_reg = self.frames.current_mut().next_u16() as usize;
                let src = self.frames.current_mut().next_u16() as usize;
                let (index_opref, _) = self.read_int_reg(index_reg);
                let (value_opref, _) = self.read_int_reg(src);
                let array_ptr = sym
                    .state_varray_ptr(array_idx)
                    .expect("virtualizable array not initialized");
                ctx.record_op_with_descr(
                    OpCode::SetarrayitemRaw,
                    &[array_ptr, index_opref, value_opref],
                    Self::raw_word_array_descr(),
                );
            }

            BC_RECORD_BINOP_I => {
                let (dst, lhs_idx, rhs_idx, opcode) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let opcode_idx = frame.next_u16() as usize;
                    let lhs_idx = frame.next_u16() as usize;
                    let rhs_idx = frame.next_u16() as usize;
                    let opcode = *frame
                        .jitcode
                        .opcodes
                        .get(opcode_idx)
                        .expect("jitcode opcode index out of bounds");
                    (dst, lhs_idx, rhs_idx, opcode)
                };
                let (lhs, lhs_value) = self.read_int_reg(lhs_idx);
                let (rhs, rhs_value) = self.read_int_reg(rhs_idx);
                if opcode.is_ovf() {
                    // Overflow-checked op: abort trace if overflow occurs.
                    match eval_binop_ovf(opcode, lhs_value, rhs_value) {
                        Some(value) => {
                            let result = ctx.record_op(opcode, &[lhs, rhs]);
                            let resume_pc = ctx.const_int(self.frames.current_mut().pc as i64);
                            Self::record_state_guard(
                                ctx,
                                sym,
                                OpCode::GuardNoOverflow,
                                &[],
                                &[resume_pc],
                            );
                            self.set_int_reg(dst, Some(result), Some(value));
                        }
                        None => return TraceAction::Abort,
                    }
                } else {
                    let value = eval_binop_i(opcode, lhs_value, rhs_value);
                    self.set_int_reg(dst, Some(ctx.record_op(opcode, &[lhs, rhs])), Some(value));
                }
            }
            BC_RECORD_UNARY_I => {
                let (dst, src_idx, opcode) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let opcode_idx = frame.next_u16() as usize;
                    let src_idx = frame.next_u16() as usize;
                    let opcode = *frame
                        .jitcode
                        .opcodes
                        .get(opcode_idx)
                        .expect("jitcode opcode index out of bounds");
                    (dst, src_idx, opcode)
                };
                let (src, src_value) = self.read_int_reg(src_idx);
                let value = eval_unary_i(opcode, src_value);
                self.set_int_reg(dst, Some(ctx.record_op(opcode, &[src])), Some(value));
            }
            BC_BRANCH_REG_ZERO => {
                let (cond_idx, target) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (cond, cond_value) = self.read_int_reg(cond_idx);
                let branch_taken = cond_value == 0;
                let opcode = if branch_taken {
                    OpCode::GuardFalse
                } else {
                    OpCode::GuardTrue
                };
                let resume_pc = ctx.const_int(self.frames.current_mut().pc as i64);
                Self::record_state_guard(ctx, sym, opcode, &[cond], &[resume_pc]);
                if branch_taken {
                    self.frames.current_mut().code_cursor = target;
                }
            }
            BC_JIT_MERGE_POINT => {
                // blackhole.py:1066 bhimpl_jit_merge_point parity.
                // Portal merge point: close the loop if at the traced header.
                let pc = self.frames.current_mut().pc;
                if runtime.label_at(pc) == sym.loop_header_pc() {
                    return TraceAction::CloseLoop;
                }
            }
            BC_JUMP_TARGET => {
                // Non-portal loop header marker (helper jitcodes only).
                let pc = self.frames.current_mut().pc;
                if runtime.label_at(pc) == sym.loop_header_pc() {
                    return TraceAction::CloseLoop;
                }
            }
            BC_JUMP => {
                let target = self.frames.current_mut().next_u16() as usize;
                self.frames.current_mut().code_cursor = target;
            }
            BC_INLINE_CALL => {
                let (sub_idx, arg_triples, return_i, return_r, return_f) = {
                    let frame = self.frames.current_mut();
                    let sub_idx = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_triples = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let caller_src = frame.next_u16() as usize;
                        let callee_dst = frame.next_u16() as usize;
                        arg_triples.push((kind, caller_src, callee_dst));
                    }
                    let decode_return_slot = |f: &mut MIFrame| {
                        let src = f.next_u16() as usize;
                        let dst = f.next_u16() as usize;
                        if src == u16::MAX as usize && dst == u16::MAX as usize {
                            None
                        } else {
                            Some((src, dst))
                        }
                    };
                    let return_i = decode_return_slot(frame);
                    let return_r = decode_return_slot(frame);
                    let return_f = decode_return_slot(frame);
                    (sub_idx, arg_triples, return_i, return_r, return_f)
                };
                let pc = self.frames.current_mut().pc;
                let sub_jitcode = self.frames.current_mut().jitcode.sub_jitcodes[sub_idx].clone();
                let mut sub_frame = MIFrame::new(sub_jitcode, pc);
                ctx.push_inline_frame(((pc as u64) << 32) | sub_idx as u64, u32::MAX);
                sub_frame.inline_frame = true;
                for (kind, caller_src, callee_dst) in arg_triples {
                    match kind {
                        JitArgKind::Int => {
                            let (value, concrete) = self.read_int_reg(caller_src);
                            sub_frame.int_regs[callee_dst] = Some(value);
                            sub_frame.int_values[callee_dst] = Some(concrete);
                        }
                        JitArgKind::Ref => {
                            let (value, concrete) = self.read_ref_reg(caller_src);
                            sub_frame.ref_regs[callee_dst] = Some(value);
                            sub_frame.ref_values[callee_dst] = Some(concrete);
                        }
                        JitArgKind::Float => {
                            let (value, concrete) = self.read_float_reg(caller_src);
                            sub_frame.float_regs[callee_dst] = Some(value);
                            sub_frame.float_values[callee_dst] = Some(concrete);
                        }
                    }
                }
                sub_frame.return_i = return_i;
                sub_frame.return_r = return_r;
                sub_frame.return_f = return_f;
                self.frames.push(sub_frame);
            }
            BC_RESIDUAL_CALL_VOID
            | BC_CALL_MAY_FORCE_VOID
            | BC_CALL_RELEASE_GIL_VOID
            | BC_CALL_LOOPINVARIANT_VOID
            | BC_CALL_ASSEMBLER_VOID => {
                let (fn_ptr_idx, arg_regs) = {
                    let frame = self.frames.current_mut();
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    (fn_ptr_idx, arg_regs)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                if bytecode == BC_CALL_ASSEMBLER_VOID {
                    let (token_number, concrete_ptr) = self
                        .frames
                        .current_mut()
                        .jitcode
                        .call_assembler_target(fn_ptr_idx);
                    let token = JitCellToken::new(token_number);
                    ctx.call_assembler_void_typed(&token, &args, &arg_types);
                    call_void_function(concrete_ptr, &concrete_args);
                } else {
                    let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                    let trace_ptr = if target.trace_ptr.is_null() {
                        target.concrete_ptr
                    } else {
                        target.trace_ptr
                    };
                    let concrete_ptr = if target.concrete_ptr.is_null() {
                        trace_ptr
                    } else {
                        target.concrete_ptr
                    };
                    let active_vable = if bytecode == BC_CALL_MAY_FORCE_VOID {
                        self.prepare_standard_virtualizable_before_residual_call(ctx)
                    } else {
                        None
                    };
                    match bytecode {
                        BC_RESIDUAL_CALL_VOID => ctx.call_void_typed(trace_ptr, &args, &arg_types),
                        BC_CALL_MAY_FORCE_VOID => {
                            ctx.call_may_force_void_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_RELEASE_GIL_VOID => {
                            ctx.call_release_gil_void_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_LOOPINVARIANT_VOID => {
                            ctx.call_loopinvariant_void_typed(trace_ptr, &args, &arg_types)
                        }
                        _ => unreachable!(),
                    }
                    call_void_function(concrete_ptr, &concrete_args);
                    if bytecode == BC_CALL_MAY_FORCE_VOID
                        && matches!(
                            Self::finalize_standard_virtualizable_may_force(ctx, sym, active_vable),
                            TraceAction::Abort
                        )
                    {
                        return TraceAction::Abort;
                    }
                }
            }
            // ── conditional_call / record_known_result (jtransform.py:1665, 292) ──
            BC_COND_CALL_VOID
            | BC_COND_CALL_VALUE_INT
            | BC_COND_CALL_VALUE_REF
            | BC_RECORD_KNOWN_RESULT_INT
            | BC_RECORD_KNOWN_RESULT_REF => {
                let (first_reg, fn_ptr_idx, arg_regs, dst) = {
                    let frame = self.frames.current_mut();
                    let first_reg = frame.next_u16();
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let num_args = frame.next_u8() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    let dst = if matches!(bytecode, BC_COND_CALL_VALUE_INT | BC_COND_CALL_VALUE_REF)
                    {
                        Some(frame.next_u16())
                    } else {
                        None
                    };
                    (first_reg, fn_ptr_idx, arg_regs, dst)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in &arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(*arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                let trace_ptr = if target.trace_ptr.is_null() {
                    target.concrete_ptr
                } else {
                    target.trace_ptr
                };
                let concrete_ptr = if target.concrete_ptr.is_null() {
                    trace_ptr
                } else {
                    target.concrete_ptr
                };
                match bytecode {
                    BC_COND_CALL_VOID => {
                        // RPython pyjitpl.py opimpl_conditional_call_ir_v:
                        //   if condition != 0: call func(args)
                        let first_val =
                            self.frames.current_mut().int_values[first_reg as usize].unwrap_or(0);
                        ctx.cond_call_void_typed(first_val, trace_ptr, &args, &arg_types);
                        if first_val != 0 {
                            call_void_function(concrete_ptr, &concrete_args);
                        }
                    }
                    BC_COND_CALL_VALUE_INT => {
                        // RPython pyjitpl.py opimpl_conditional_call_value_ir_i
                        let first_val =
                            self.frames.current_mut().int_values[first_reg as usize].unwrap_or(0);
                        let result =
                            ctx.cond_call_value_int_typed(first_val, trace_ptr, &args, &arg_types);
                        let concrete_result = if first_val == 0 {
                            call_int_function(concrete_ptr, &concrete_args)
                        } else {
                            first_val
                        };
                        if let Some(dst) = dst {
                            self.frames.current_mut().int_values[dst as usize] =
                                Some(concrete_result);
                        }
                        let _ = result;
                    }
                    BC_COND_CALL_VALUE_REF => {
                        // RPython pyjitpl.py opimpl_conditional_call_value_ir_r:
                        // value is a ref — read from ref register bank.
                        let first_val =
                            self.frames.current_mut().ref_values[first_reg as usize].unwrap_or(0);
                        let result =
                            ctx.cond_call_value_ref_typed(first_val, trace_ptr, &args, &arg_types);
                        let concrete_result = if first_val == 0 {
                            call_int_function(concrete_ptr, &concrete_args)
                        } else {
                            first_val
                        };
                        if let Some(dst) = dst {
                            self.frames.current_mut().ref_values[dst as usize] =
                                Some(concrete_result);
                        }
                        let _ = result;
                    }
                    BC_RECORD_KNOWN_RESULT_INT => {
                        // RPython pyjitpl.py opimpl_record_known_result_i:
                        let result_val =
                            self.frames.current_mut().int_values[first_reg as usize].unwrap_or(0);
                        ctx.record_known_result_typed(result_val, trace_ptr, &args, &arg_types);
                    }
                    BC_RECORD_KNOWN_RESULT_REF => {
                        // RPython pyjitpl.py opimpl_record_known_result_r:
                        let result_val =
                            self.frames.current_mut().ref_values[first_reg as usize].unwrap_or(0);
                        ctx.record_known_result_typed(result_val, trace_ptr, &args, &arg_types);
                    }
                    _ => unreachable!(),
                }
            }
            BC_MOVE_I => {
                let (dst, src) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (value, concrete) = self.read_int_reg(src);
                self.set_int_reg(dst, Some(value), Some(concrete));
            }
            BC_CALL_INT
            | BC_CALL_PURE_INT
            | BC_CALL_MAY_FORCE_INT
            | BC_CALL_RELEASE_GIL_INT
            | BC_CALL_LOOPINVARIANT_INT
            | BC_CALL_ASSEMBLER_INT => {
                let (opcode, fn_ptr_idx, dst, arg_regs) = {
                    let frame = self.frames.current_mut();
                    let opcode = bytecode;
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let dst = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    (opcode, fn_ptr_idx, dst, arg_regs)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                if opcode == BC_CALL_ASSEMBLER_INT {
                    let (token_number, concrete_ptr) = self
                        .frames
                        .current_mut()
                        .jitcode
                        .call_assembler_target(fn_ptr_idx);
                    let token = JitCellToken::new(token_number);
                    let traced = ctx.call_assembler_int_typed(&token, &args, &arg_types);
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    self.set_int_reg(dst, Some(traced), Some(concrete));
                } else {
                    let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                    let trace_ptr = if target.trace_ptr.is_null() {
                        target.concrete_ptr
                    } else {
                        target.trace_ptr
                    };
                    let concrete_ptr = if target.concrete_ptr.is_null() {
                        trace_ptr
                    } else {
                        target.concrete_ptr
                    };
                    let active_vable = if opcode == BC_CALL_MAY_FORCE_INT {
                        self.prepare_standard_virtualizable_before_residual_call(ctx)
                    } else {
                        None
                    };
                    // pyjitpl.py:1941-1948: CALL_PURE records plain CALL
                    // first, executes, then patches via record_result_of_call_pure.
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    let traced = match opcode {
                        BC_CALL_INT => ctx.call_int_typed(trace_ptr, &args, &arg_types),
                        BC_CALL_PURE_INT => {
                            let patch_pos = ctx.get_trace_position();
                            let plain_op = ctx.call_int_typed(trace_ptr, &args, &arg_types);
                            let func_ref = ctx.const_int(trace_ptr as usize as i64);
                            let mut call_args = vec![func_ref];
                            call_args.extend_from_slice(&args);
                            let concrete_values =
                                build_concrete_values(trace_ptr, &concrete_args, &arg_types);
                            ctx.record_result_of_call_pure(
                                plain_op,
                                &call_args,
                                &concrete_values,
                                crate::call_descr::make_call_descr(&arg_types, majit_ir::Type::Int),
                                patch_pos,
                                majit_ir::OpCode::CallI,
                                majit_ir::Value::Int(concrete),
                            )
                        }
                        BC_CALL_MAY_FORCE_INT => {
                            ctx.call_may_force_int_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_RELEASE_GIL_INT => {
                            ctx.call_release_gil_int_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_LOOPINVARIANT_INT => {
                            ctx.call_loopinvariant_int_typed(trace_ptr, &args, &arg_types)
                        }
                        _ => unreachable!(),
                    };
                    if opcode == BC_CALL_MAY_FORCE_INT
                        && matches!(
                            Self::finalize_standard_virtualizable_may_force(ctx, sym, active_vable),
                            TraceAction::Abort
                        )
                    {
                        return TraceAction::Abort;
                    }
                    self.set_int_reg(dst, Some(traced), Some(concrete));
                }
            }
            // -- Ref-typed bytecodes ----
            BC_LOAD_CONST_R => {
                let (dst, value) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let const_idx = frame.next_u16() as usize;
                    let value = *frame
                        .jitcode
                        .constants_r
                        .get(const_idx)
                        .expect("jitcode const index out of bounds");
                    (dst, value)
                };
                self.set_ref_reg(dst, Some(ctx.const_ref(value)), Some(value));
            }
            BC_MOVE_R => {
                let (dst, src) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (value, concrete) = self.read_ref_reg(src);
                self.set_ref_reg(dst, Some(value), Some(concrete));
            }
            BC_CALL_REF
            | BC_CALL_PURE_REF
            | BC_CALL_MAY_FORCE_REF
            | BC_CALL_RELEASE_GIL_REF
            | BC_CALL_LOOPINVARIANT_REF
            | BC_CALL_ASSEMBLER_REF => {
                let (opcode, fn_ptr_idx, dst, arg_regs) = {
                    let frame = self.frames.current_mut();
                    let opcode = bytecode;
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let dst = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    (opcode, fn_ptr_idx, dst, arg_regs)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                if opcode == BC_CALL_ASSEMBLER_REF {
                    let (token_number, concrete_ptr) = self
                        .frames
                        .current_mut()
                        .jitcode
                        .call_assembler_target(fn_ptr_idx);
                    let token = JitCellToken::new(token_number);
                    let traced = ctx.call_assembler_ref_typed(&token, &args, &arg_types);
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    self.set_ref_reg(dst, Some(traced), Some(concrete));
                } else {
                    let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                    let trace_ptr = if target.trace_ptr.is_null() {
                        target.concrete_ptr
                    } else {
                        target.trace_ptr
                    };
                    let concrete_ptr = if target.concrete_ptr.is_null() {
                        trace_ptr
                    } else {
                        target.concrete_ptr
                    };
                    let active_vable = if opcode == BC_CALL_MAY_FORCE_REF {
                        self.prepare_standard_virtualizable_before_residual_call(ctx)
                    } else {
                        None
                    };
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    let traced = match opcode {
                        BC_CALL_REF => ctx.call_ref_typed(trace_ptr, &args, &arg_types),
                        BC_CALL_PURE_REF => {
                            let patch_pos = ctx.get_trace_position();
                            let plain_op = ctx.call_ref_typed(trace_ptr, &args, &arg_types);
                            let func_ref = ctx.const_int(trace_ptr as usize as i64);
                            let mut call_args = vec![func_ref];
                            call_args.extend_from_slice(&args);
                            let concrete_values =
                                build_concrete_values(trace_ptr, &concrete_args, &arg_types);
                            ctx.record_result_of_call_pure(
                                plain_op,
                                &call_args,
                                &concrete_values,
                                crate::call_descr::make_call_descr(&arg_types, majit_ir::Type::Ref),
                                patch_pos,
                                majit_ir::OpCode::CallR,
                                majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
                            )
                        }
                        BC_CALL_MAY_FORCE_REF => {
                            ctx.call_may_force_ref_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_RELEASE_GIL_REF => {
                            ctx.call_release_gil_ref_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_LOOPINVARIANT_REF => {
                            ctx.call_loopinvariant_ref_typed(trace_ptr, &args, &arg_types)
                        }
                        _ => unreachable!(),
                    };
                    if opcode == BC_CALL_MAY_FORCE_REF
                        && matches!(
                            Self::finalize_standard_virtualizable_may_force(ctx, sym, active_vable),
                            TraceAction::Abort
                        )
                    {
                        return TraceAction::Abort;
                    }
                    self.set_ref_reg(dst, Some(traced), Some(concrete));
                }
            }
            // -- Float-typed bytecodes ---
            BC_LOAD_CONST_F => {
                let (dst, value) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let const_idx = frame.next_u16() as usize;
                    let value = *frame
                        .jitcode
                        .constants_f
                        .get(const_idx)
                        .expect("jitcode const index out of bounds");
                    (dst, value)
                };
                self.set_float_reg(dst, Some(ctx.const_float(value)), Some(value));
            }
            BC_MOVE_F => {
                let (dst, src) = {
                    let frame = self.frames.current_mut();
                    (frame.next_u16() as usize, frame.next_u16() as usize)
                };
                let (value, concrete) = self.read_float_reg(src);
                self.set_float_reg(dst, Some(value), Some(concrete));
            }
            BC_CALL_FLOAT
            | BC_CALL_PURE_FLOAT
            | BC_CALL_MAY_FORCE_FLOAT
            | BC_CALL_RELEASE_GIL_FLOAT
            | BC_CALL_LOOPINVARIANT_FLOAT
            | BC_CALL_ASSEMBLER_FLOAT => {
                let (opcode, fn_ptr_idx, dst, arg_regs) = {
                    let frame = self.frames.current_mut();
                    let opcode = bytecode;
                    let fn_ptr_idx = frame.next_u16() as usize;
                    let dst = frame.next_u16() as usize;
                    let num_args = frame.next_u16() as usize;
                    let mut arg_regs = Vec::with_capacity(num_args);
                    for _ in 0..num_args {
                        let kind = JitArgKind::decode(frame.next_u8());
                        let reg = frame.next_u16();
                        arg_regs.push(JitCallArg { kind, reg });
                    }
                    (opcode, fn_ptr_idx, dst, arg_regs)
                };
                let mut args = Vec::with_capacity(arg_regs.len());
                let mut concrete_args = Vec::with_capacity(arg_regs.len());
                let mut arg_types = Vec::with_capacity(arg_regs.len());
                for arg_spec in arg_regs {
                    let (arg, concrete, arg_type) = self.read_call_arg(arg_spec);
                    args.push(arg);
                    concrete_args.push(concrete);
                    arg_types.push(arg_type);
                }
                if opcode == BC_CALL_ASSEMBLER_FLOAT {
                    let (token_number, concrete_ptr) = self
                        .frames
                        .current_mut()
                        .jitcode
                        .call_assembler_target(fn_ptr_idx);
                    let token = JitCellToken::new(token_number);
                    let traced = ctx.call_assembler_float_typed(&token, &args, &arg_types);
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    self.set_float_reg(dst, Some(traced), Some(concrete));
                } else {
                    let target = self.frames.current_mut().jitcode.fn_ptrs[fn_ptr_idx];
                    let trace_ptr = if target.trace_ptr.is_null() {
                        target.concrete_ptr
                    } else {
                        target.trace_ptr
                    };
                    let concrete_ptr = if target.concrete_ptr.is_null() {
                        trace_ptr
                    } else {
                        target.concrete_ptr
                    };
                    let active_vable = if opcode == BC_CALL_MAY_FORCE_FLOAT {
                        self.prepare_standard_virtualizable_before_residual_call(ctx)
                    } else {
                        None
                    };
                    let concrete = call_int_function(concrete_ptr, &concrete_args);
                    let traced = match opcode {
                        BC_CALL_FLOAT => ctx.call_float_typed(trace_ptr, &args, &arg_types),
                        BC_CALL_PURE_FLOAT => {
                            let patch_pos = ctx.get_trace_position();
                            let plain_op = ctx.call_float_typed(trace_ptr, &args, &arg_types);
                            let func_ref = ctx.const_int(trace_ptr as usize as i64);
                            let mut call_args = vec![func_ref];
                            call_args.extend_from_slice(&args);
                            let concrete_values =
                                build_concrete_values(trace_ptr, &concrete_args, &arg_types);
                            ctx.record_result_of_call_pure(
                                plain_op,
                                &call_args,
                                &concrete_values,
                                crate::call_descr::make_call_descr(
                                    &arg_types,
                                    majit_ir::Type::Float,
                                ),
                                patch_pos,
                                majit_ir::OpCode::CallF,
                                majit_ir::Value::Float(f64::from_bits(concrete as u64)),
                            )
                        }
                        BC_CALL_MAY_FORCE_FLOAT => {
                            ctx.call_may_force_float_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_RELEASE_GIL_FLOAT => {
                            ctx.call_release_gil_float_typed(trace_ptr, &args, &arg_types)
                        }
                        BC_CALL_LOOPINVARIANT_FLOAT => {
                            ctx.call_loopinvariant_float_typed(trace_ptr, &args, &arg_types)
                        }
                        _ => unreachable!(),
                    };
                    if opcode == BC_CALL_MAY_FORCE_FLOAT
                        && matches!(
                            Self::finalize_standard_virtualizable_may_force(ctx, sym, active_vable),
                            TraceAction::Abort
                        )
                    {
                        return TraceAction::Abort;
                    }
                    self.set_float_reg(dst, Some(traced), Some(concrete));
                }
            }
            BC_RECORD_BINOP_F => {
                let (dst, lhs_idx, rhs_idx, opcode) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let opcode_idx = frame.next_u16() as usize;
                    let lhs_idx = frame.next_u16() as usize;
                    let rhs_idx = frame.next_u16() as usize;
                    let opcode = *frame
                        .jitcode
                        .opcodes
                        .get(opcode_idx)
                        .expect("jitcode opcode index out of bounds");
                    (dst, lhs_idx, rhs_idx, opcode)
                };
                let (lhs, lhs_value) = self.read_float_reg(lhs_idx);
                let (rhs, rhs_value) = self.read_float_reg(rhs_idx);
                let value = eval_binop_f(opcode, lhs_value, rhs_value);
                self.set_float_reg(dst, Some(ctx.record_op(opcode, &[lhs, rhs])), Some(value));
            }
            BC_RECORD_UNARY_F => {
                let (dst, src_idx, opcode) = {
                    let frame = self.frames.current_mut();
                    let dst = frame.next_u16() as usize;
                    let opcode_idx = frame.next_u16() as usize;
                    let src_idx = frame.next_u16() as usize;
                    let opcode = *frame
                        .jitcode
                        .opcodes
                        .get(opcode_idx)
                        .expect("jitcode opcode index out of bounds");
                    (dst, src_idx, opcode)
                };
                let (src, src_value) = self.read_float_reg(src_idx);
                let value = eval_unary_f(opcode, src_value);
                self.set_float_reg(dst, Some(ctx.record_op(opcode, &[src])), Some(value));
            }
            // pyjitpl.py opimpl_int_guard_value → implement_guard_value
            // Blackhole: no-op.  Tracing: emit GUARD_VALUE to promote.
            BC_INT_GUARD_VALUE => {
                let src = self.frames.current_mut().next_u16() as usize;
                let (opref, concrete) = self.read_int_reg(src);
                let promoted = ctx.promote_int(opref, concrete, 0);
                self.set_int_reg(src, Some(promoted), Some(concrete));
            }
            // pyjitpl.py opimpl_ref_guard_value → implement_guard_value
            BC_REF_GUARD_VALUE => {
                let src = self.frames.current_mut().next_u16() as usize;
                let (opref, concrete) = self.read_ref_reg(src);
                let promoted = ctx.promote_ref(opref, concrete, 0);
                self.set_ref_reg(src, Some(promoted), Some(concrete));
            }
            // pyjitpl.py:1515 opimpl_float_guard_value = _opimpl_guard_value
            BC_FLOAT_GUARD_VALUE => {
                let src = self.frames.current_mut().next_u16() as usize;
                let (opref, concrete) = self.read_float_reg(src);
                let promoted = ctx.promote_float(opref, concrete, 0);
                self.set_float_reg(src, Some(promoted), Some(concrete));
            }
            BC_ABORT => return TraceAction::Abort,
            BC_ABORT_PERMANENT => return TraceAction::AbortPermanent,
            other => panic!("unknown jitcode bytecode {other}"),
        }

        TraceAction::Continue
    }

    fn set_int_reg(&mut self, reg: usize, opref: Option<OpRef>, value: Option<i64>) {
        let frame = self.frames.current_mut();
        frame.int_regs[reg] = opref;
        frame.int_values[reg] = value;
    }

    fn read_int_reg(&mut self, reg: usize) -> (OpRef, i64) {
        let frame = self.frames.current_mut();
        (
            frame.int_regs[reg].expect("jitcode register was uninitialized"),
            frame.int_values[reg].expect("jitcode concrete register was uninitialized"),
        )
    }

    fn set_ref_reg(&mut self, reg: usize, opref: Option<OpRef>, value: Option<i64>) {
        let frame = self.frames.current_mut();
        frame.ref_regs[reg] = opref;
        frame.ref_values[reg] = value;
    }

    fn read_ref_reg(&mut self, reg: usize) -> (OpRef, i64) {
        let frame = self.frames.current_mut();
        (
            frame.ref_regs[reg].expect("jitcode ref register was uninitialized"),
            frame.ref_values[reg].expect("jitcode concrete ref register was uninitialized"),
        )
    }

    fn set_float_reg(&mut self, reg: usize, opref: Option<OpRef>, value: Option<i64>) {
        let frame = self.frames.current_mut();
        frame.float_regs[reg] = opref;
        frame.float_values[reg] = value;
    }

    fn read_float_reg(&mut self, reg: usize) -> (OpRef, i64) {
        let frame = self.frames.current_mut();
        (
            frame.float_regs[reg].expect("jitcode float register was uninitialized"),
            frame.float_values[reg].expect("jitcode concrete float register was uninitialized"),
        )
    }

    fn read_call_arg(&mut self, arg: JitCallArg) -> (OpRef, i64, majit_ir::Type) {
        match arg.kind {
            JitArgKind::Int => {
                let (opref, value) = self.read_int_reg(arg.reg as usize);
                (opref, value, majit_ir::Type::Int)
            }
            JitArgKind::Ref => {
                let (opref, value) = self.read_ref_reg(arg.reg as usize);
                (opref, value, majit_ir::Type::Ref)
            }
            JitArgKind::Float => {
                let (opref, value) = self.read_float_reg(arg.reg as usize);
                (opref, value, majit_ir::Type::Float)
            }
        }
    }
}

/// Legacy entry point used by tests and integrations that still hold
/// `JitCode` by reference and do not pass a `MetaInterp` framestack
/// borrow.  Allocates a [`StandaloneFrameStack`], pushes the root
/// frame, runs the machine, and discards the stack — preserving
/// pre-unification semantics for callers that have not yet migrated.
pub fn trace_jitcode<S, FLabel>(
    ctx: &mut TraceCtx,
    sym: &mut S,
    jitcode: &JitCode,
    pc: usize,
    label_at: FLabel,
) -> TraceAction
where
    S: JitCodeSym,
    FLabel: Fn(usize) -> usize,
{
    let runtime = ClosureRuntime::new(label_at);
    let jitcode_arc = Arc::new(jitcode.clone());
    let sub_jitcodes = jitcode_arc.sub_jitcodes.clone();
    let fn_ptrs = jitcode_arc.fn_ptrs.clone();
    let mut standalone = StandaloneFrameStack::new();
    standalone.frames.push(MIFrame::new(jitcode_arc, pc));
    let mut machine =
        JitCodeMachine::<S, _>::with_framestack(&mut standalone.frames, &sub_jitcodes, &fn_ptrs);
    machine.run_to_end(ctx, sym, &runtime)
}

pub(crate) fn eval_binop_i(opcode: OpCode, lhs: i64, rhs: i64) -> i64 {
    match opcode {
        OpCode::IntAdd => lhs.wrapping_add(rhs),
        OpCode::IntSub => lhs.wrapping_sub(rhs),
        OpCode::IntMul => lhs.wrapping_mul(rhs),
        OpCode::IntFloorDiv => {
            if rhs == 0 {
                0
            } else {
                lhs.wrapping_div(rhs)
            }
        }
        OpCode::IntMod => {
            if rhs == 0 {
                0
            } else {
                lhs.wrapping_rem(rhs)
            }
        }
        OpCode::IntAnd => lhs & rhs,
        OpCode::IntOr => lhs | rhs,
        OpCode::IntXor => lhs ^ rhs,
        OpCode::IntLshift => lhs.wrapping_shl(rhs as u32),
        OpCode::IntRshift => lhs.wrapping_shr(rhs as u32),
        OpCode::IntEq => i64::from(lhs == rhs),
        OpCode::IntNe => i64::from(lhs != rhs),
        OpCode::IntLt => i64::from(lhs < rhs),
        OpCode::IntLe => i64::from(lhs <= rhs),
        OpCode::IntGt => i64::from(lhs > rhs),
        OpCode::IntGe => i64::from(lhs >= rhs),
        other => panic!("unsupported jitcode integer binop {other:?}"),
    }
}

/// Evaluate an overflow-checked binop. Returns `None` on overflow.
pub(crate) fn eval_binop_ovf(opcode: OpCode, lhs: i64, rhs: i64) -> Option<i64> {
    match opcode {
        OpCode::IntAddOvf => lhs.checked_add(rhs),
        OpCode::IntSubOvf => lhs.checked_sub(rhs),
        OpCode::IntMulOvf => lhs.checked_mul(rhs),
        other => panic!("unsupported jitcode overflow binop {other:?}"),
    }
}

pub(crate) fn eval_unary_i(opcode: OpCode, value: i64) -> i64 {
    match opcode {
        OpCode::IntNeg => value.wrapping_neg(),
        other => panic!("unsupported jitcode integer unary op {other:?}"),
    }
}

/// Evaluate a float binary operation. Values are stored as i64 (bit-cast).
pub(crate) fn eval_binop_f(opcode: OpCode, lhs: i64, rhs: i64) -> i64 {
    let a = f64::from_bits(lhs as u64);
    let b = f64::from_bits(rhs as u64);
    let result = match opcode {
        OpCode::FloatAdd => a + b,
        OpCode::FloatSub => a - b,
        OpCode::FloatMul => a * b,
        OpCode::FloatTrueDiv => a / b,
        OpCode::FloatFloorDiv => (a / b).floor(),
        OpCode::FloatMod => a % b,
        other => panic!("unsupported jitcode float binop {other:?}"),
    };
    f64::to_bits(result) as i64
}

/// Evaluate a float unary operation.
pub(crate) fn eval_unary_f(opcode: OpCode, value: i64) -> i64 {
    let a = f64::from_bits(value as u64);
    let result = match opcode {
        OpCode::FloatNeg => -a,
        OpCode::FloatAbs => a.abs(),
        other => panic!("unsupported jitcode float unary op {other:?}"),
    };
    f64::to_bits(result) as i64
}

/// executor.py:544 constant_from_op — typed Value from raw i64 + Type.
fn typed_value_from_raw(raw: i64, tp: majit_ir::Type) -> majit_ir::Value {
    match tp {
        majit_ir::Type::Int => majit_ir::Value::Int(raw),
        majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(raw as usize)),
        majit_ir::Type::Float => majit_ir::Value::Float(f64::from_bits(raw as u64)),
        majit_ir::Type::Void => majit_ir::Value::Void,
    }
}

/// executor.py:544 parity: build typed concrete_values for call_pure_results key.
/// First element is func_ptr (always Int), rest use arg_types.
fn build_concrete_values(
    func_ptr: *const (),
    concrete_args: &[i64],
    arg_types: &[majit_ir::Type],
) -> Vec<majit_ir::Value> {
    let mut values = vec![majit_ir::Value::Int(func_ptr as usize as i64)];
    for (i, &v) in concrete_args.iter().enumerate() {
        let tp = arg_types[i];
        values.push(typed_value_from_raw(v, tp));
    }
    values
}

pub(crate) fn call_int_function(func_ptr: *const (), args: &[i64]) -> i64 {
    unsafe {
        match args {
            [] => {
                let func: extern "C" fn() -> i64 = std::mem::transmute(func_ptr);
                func()
            }
            [a0] => {
                let func: extern "C" fn(i64) -> i64 = std::mem::transmute(func_ptr);
                func(*a0)
            }
            [a0, a1] => {
                let func: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1)
            }
            [a0, a1, a2] => {
                let func: extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2)
            }
            [a0, a1, a2, a3] => {
                let func: extern "C" fn(i64, i64, i64, i64) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3)
            }
            [a0, a1, a2, a3, a4] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4)
            }
            [a0, a1, a2, a3, a4, a5] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5)
            }
            [a0, a1, a2, a3, a4, a5, a6] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12,
                )
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13,
                )
            }
            [
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
            ] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14,
                )
            }
            [
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
            ] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) -> i64 = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14,
                    *a15,
                )
            }
            _ => panic!(
                "unsupported JitCode int call arity {} (max {})",
                args.len(),
                MAX_HOST_CALL_ARITY
            ),
        }
    }
}

pub(crate) fn call_void_function(func_ptr: *const (), args: &[i64]) {
    unsafe {
        match args {
            [] => {
                let func: extern "C" fn() = std::mem::transmute(func_ptr);
                func()
            }
            [a0] => {
                let func: extern "C" fn(i64) = std::mem::transmute(func_ptr);
                func(*a0)
            }
            [a0, a1] => {
                let func: extern "C" fn(i64, i64) = std::mem::transmute(func_ptr);
                func(*a0, *a1)
            }
            [a0, a1, a2] => {
                let func: extern "C" fn(i64, i64, i64) = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2)
            }
            [a0, a1, a2, a3] => {
                let func: extern "C" fn(i64, i64, i64, i64) = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3)
            }
            [a0, a1, a2, a3, a4] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4)
            }
            [a0, a1, a2, a3, a4, a5] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5)
            }
            [a0, a1, a2, a3, a4, a5, a6] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) =
                    std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(*a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11)
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12,
                )
            }
            [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13,
                )
            }
            [
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
            ] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14,
                )
            }
            [
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
            ] => {
                let func: extern "C" fn(
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = std::mem::transmute(func_ptr);
                func(
                    *a0, *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8, *a9, *a10, *a11, *a12, *a13, *a14,
                    *a15,
                )
            }
            _ => panic!(
                "unsupported JitCode void call arity {} (max {})",
                args.len(),
                MAX_HOST_CALL_ARITY
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitcode::JitCodeBuilder;
    use crate::virtualizable::VirtualizableInfo;
    use majit_ir::Type;

    #[derive(Default)]
    struct DummySym;

    impl JitCodeSym for DummySym {
        fn total_slots(&self) -> usize {
            0
        }

        fn loop_header_pc(&self) -> usize {
            0
        }

        fn fail_args(&self) -> Option<Vec<OpRef>> {
            None
        }
    }

    fn make_test_vable_info() -> VirtualizableInfo {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "stack",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        info
    }

    #[repr(C)]
    struct ResidualVable {
        token: u64,
    }

    extern "C" fn residual_no_force(_vable: i64) {}

    extern "C" fn residual_int_no_force(_vable: i64) -> i64 {
        7
    }

    extern "C" fn residual_ref_no_force(vable: i64) -> i64 {
        vable
    }

    extern "C" fn residual_float_no_force(_vable: i64) -> i64 {
        f64::to_bits(3.5) as i64
    }

    extern "C" fn residual_force(vable: i64) {
        unsafe {
            (*(vable as usize as *mut ResidualVable)).token = 0;
        }
    }

    #[test]
    fn jitcode_vable_reads_use_standard_boxes_without_heap_ops() {
        let mut builder = JitCodeBuilder::new();
        builder.load_const_i_value(0, 0);
        builder.vable_getfield_int(1, 0);
        builder.vable_getarrayitem_int(2, 0, 0);
        builder.vable_arraylen(3, 0);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let info = make_test_vable_info();
        let field_box = ctx.const_int(111);
        let array_box = ctx.const_int(222);
        let vable_ref = ctx.const_int(999);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[field_box, array_box], &[1]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(&mut ctx, &mut sym, &jitcode, 0, |_pc| 0);
        assert!(matches!(action, TraceAction::Continue));

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 0);
    }

    #[test]
    fn jitcode_call_may_force_marks_standard_virtualizable_token_and_guards() {
        let mut obj = ResidualVable { token: 0 };
        let obj_ptr = (&mut obj as *mut ResidualVable) as usize as i64;

        let mut builder = JitCodeBuilder::new();
        builder.load_const_r_value(0, obj_ptr);
        let fn_idx = builder.add_fn_ptr(residual_no_force as *const ());
        builder.call_may_force_void_typed_args(fn_idx, &[JitCallArg::reference(0)]);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let mut info = VirtualizableInfo::new(0);
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        let vable_ref = ctx.const_ref(obj_ptr);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[], &[]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(&mut ctx, &mut sym, &jitcode, 0, |_pc| 0);
        assert!(matches!(action, TraceAction::Continue));
        assert_eq!(obj.token, 0, "tracing side must restore TOKEN_NONE");

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 4);
        assert_eq!(
            recorder.get_op_by_pos(OpRef(0)).unwrap().opcode,
            OpCode::ForceToken
        );
        let set_token = recorder.get_op_by_pos(OpRef(1)).unwrap();
        assert_eq!(set_token.opcode, OpCode::SetfieldGc);
        assert_eq!(
            set_token.descr.as_ref().map(|d| d.index()),
            Some(info.token_field_descr().index())
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(2)).unwrap().opcode,
            OpCode::CallMayForceN
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(3)).unwrap().opcode,
            OpCode::GuardNotForced
        );
    }

    #[test]
    fn jitcode_call_may_force_aborts_when_standard_virtualizable_escapes() {
        let mut obj = ResidualVable { token: 0 };
        let obj_ptr = (&mut obj as *mut ResidualVable) as usize as i64;

        let mut builder = JitCodeBuilder::new();
        builder.load_const_r_value(0, obj_ptr);
        let fn_idx = builder.add_fn_ptr(residual_force as *const ());
        builder.call_may_force_void_typed_args(fn_idx, &[JitCallArg::reference(0)]);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let mut info = VirtualizableInfo::new(0);
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        let vable_ref = ctx.const_ref(obj_ptr);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[], &[]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(&mut ctx, &mut sym, &jitcode, 0, |_pc| 0);
        assert!(matches!(action, TraceAction::Abort));
        assert_eq!(obj.token, 0, "forced residual call must clear the token");

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 3);
        assert_eq!(
            recorder.get_op_by_pos(OpRef(0)).unwrap().opcode,
            OpCode::ForceToken
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(1)).unwrap().opcode,
            OpCode::SetfieldGc
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(2)).unwrap().opcode,
            OpCode::CallMayForceN
        );
    }

    #[test]
    fn jitcode_call_may_force_int_marks_standard_virtualizable_token_and_guards() {
        let mut obj = ResidualVable { token: 0 };
        let obj_ptr = (&mut obj as *mut ResidualVable) as usize as i64;

        let mut builder = JitCodeBuilder::new();
        builder.load_const_r_value(0, obj_ptr);
        let fn_idx = builder.add_fn_ptr(residual_int_no_force as *const ());
        builder.call_may_force_int_typed(fn_idx, &[JitCallArg::reference(0)], 1);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let mut info = VirtualizableInfo::new(0);
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        let vable_ref = ctx.const_ref(obj_ptr);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[], &[]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(&mut ctx, &mut sym, &jitcode, 0, |_pc| 0);
        assert!(matches!(action, TraceAction::Continue));
        assert_eq!(obj.token, 0);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 4);
        assert_eq!(
            recorder.get_op_by_pos(OpRef(0)).unwrap().opcode,
            OpCode::ForceToken
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(1)).unwrap().opcode,
            OpCode::SetfieldGc
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(2)).unwrap().opcode,
            OpCode::CallMayForceI
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(3)).unwrap().opcode,
            OpCode::GuardNotForced
        );
    }

    #[test]
    fn jitcode_call_may_force_ref_marks_standard_virtualizable_token_and_guards() {
        let mut obj = ResidualVable { token: 0 };
        let obj_ptr = (&mut obj as *mut ResidualVable) as usize as i64;

        let mut builder = JitCodeBuilder::new();
        builder.load_const_r_value(0, obj_ptr);
        let fn_idx = builder.add_fn_ptr(residual_ref_no_force as *const ());
        builder.call_may_force_ref_typed(fn_idx, &[JitCallArg::reference(0)], 1);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let mut info = VirtualizableInfo::new(0);
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        let vable_ref = ctx.const_ref(obj_ptr);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[], &[]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(&mut ctx, &mut sym, &jitcode, 0, |_pc| 0);
        assert!(matches!(action, TraceAction::Continue));
        assert_eq!(obj.token, 0);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 4);
        assert_eq!(
            recorder.get_op_by_pos(OpRef(0)).unwrap().opcode,
            OpCode::ForceToken
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(1)).unwrap().opcode,
            OpCode::SetfieldGc
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(2)).unwrap().opcode,
            OpCode::CallMayForceR
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(3)).unwrap().opcode,
            OpCode::GuardNotForced
        );
    }

    #[test]
    fn jitcode_call_may_force_float_marks_standard_virtualizable_token_and_guards() {
        let mut obj = ResidualVable { token: 0 };
        let obj_ptr = (&mut obj as *mut ResidualVable) as usize as i64;

        let mut builder = JitCodeBuilder::new();
        builder.load_const_r_value(0, obj_ptr);
        let fn_idx = builder.add_fn_ptr(residual_float_no_force as *const ());
        builder.call_may_force_float_typed(fn_idx, &[JitCallArg::reference(0)], 1);
        let jitcode = builder.finish();

        let mut ctx = TraceCtx::for_test(0);
        let mut info = VirtualizableInfo::new(0);
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        let vable_ref = ctx.const_ref(obj_ptr);
        ctx.init_virtualizable_boxes(&info, vable_ref, &[], &[]);

        let mut sym = DummySym::default();
        let action = trace_jitcode(&mut ctx, &mut sym, &jitcode, 0, |_pc| 0);
        assert!(matches!(action, TraceAction::Continue));
        assert_eq!(obj.token, 0);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 4);
        assert_eq!(
            recorder.get_op_by_pos(OpRef(0)).unwrap().opcode,
            OpCode::ForceToken
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(1)).unwrap().opcode,
            OpCode::SetfieldGc
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(2)).unwrap().opcode,
            OpCode::CallMayForceF
        );
        assert_eq!(
            recorder.get_op_by_pos(OpRef(3)).unwrap().opcode,
            OpCode::GuardNotForced
        );
    }
}
