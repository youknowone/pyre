//! `pyjitpl.py MetaInterp.execute_and_record` and its `_record_helper` /
//! `_all_constants` companions — the single funnel every non-raising
//! operation passes through on its way into the trace.
//!
//! # Why the funnel does not execute
//!
//! Upstream runs `resvalue = executor.execute(...)` *inside* the funnel, so
//! the fold decides only whether to **record**, never whether to **compute**.
//! Every caller here has already computed: the tracer advances its own
//! register banks (`read_int_reg` → `eval_binop_i` → `set_int_reg`) before it
//! can decide anything, and `execute_nonspec_const` cannot perform a live
//! memory read for the callers that need one. So the result arrives as the
//! `resvalue` parameter and the funnel keeps only the invariant that matters:
//! it decides whether to record.
//!
//! `resvalue: None` is a state upstream cannot have, because
//! `executor.execute` always runs on a real CPU. It means "no trace-time
//! concrete available" — `TraceCtx::opimpl_arraylen_gc` receives it when
//! `arraylen_sanity_load` finds no lendescr, and the layout-missing
//! virtualizable fallbacks return it. A `None` never folds, always records,
//! and never stamps.
//!
//! # The fold value and the stamped value come from different evaluators
//!
//! Upstream has one `executor.execute` feeding both this funnel and
//! `Optimizer.constant_fold`. Here the fold runs
//! `executor::execute_nonspec_const` — the same evaluator `constant_fold`
//! consults — while the stamp uses the caller's `resvalue`, which came from
//! the tracer-side evaluators. Those two disagree in a handful of cases:
//! `eval_binop_i` masks out-of-range shifts and answers 0 for div/mod by
//! zero, `eval_binop_f` has no zero-divisor guard, and the tracer casts
//! saturate, where `execute_binary_int_const` and its siblings *decline*. A
//! decline records instead of folding, so the trace and the optimizer can
//! never disagree about what a constant is. That is narrower than upstream,
//! which would fold `float_truediv(1.0, 0.0)` to `inf`; widening it is one
//! line in `execute_binary_float_const`, not in this funnel.
//!
//! # A caller holding only a `Backend` cannot be routed
//!
//! `TraceCtx::set_cpu` installs a `majit_backend::Backend`, and that is what
//! `TraceCtx::field_sanity_load` reads a field through. The fold here needs a
//! `majit_backend::model::Cpu` — an unrelated trait, no supertrait relation,
//! its own `bh_getfield_gc_i`. So the `vable_getfield_{int,ref,float}` /
//! `vable_arraylen_vable` family cannot be routed as it stands: reaching for
//! `cpu::default_cpu()` would fold the memory read through a stand-in while
//! the stamp came from the real backend — two readers of one field, which is
//! the disagreement the D3 rule above exists to rule out.
//!
//! What makes it *only* a wiring gap is that `MetaInterp` holds both objects
//! at the three sites where it calls `ctx.set_cpu(Some(&self.backend))`. Until
//! it hands over the second one too, those arms stay on `record_op_with_descr`.
//! Operations that read no memory are exempt and say so at the call site —
//! `_nonstandard_virtualizable`'s `PTR_EQ` is the one that takes the exemption.
//!
//! # Heapcache invalidation stays outside
//!
//! Upstream's `_record_helper` calls `heapcache.invalidate_caches`. Here
//! invalidation is hand-called per arm, and `TraceCtx::record_call_with_descr`
//! does its own, so pulling it in would double-invalidate every site that
//! already calls it. Adopting sites keep their existing call, before the
//! funnel call. Skipping it on the *fold* path is separately sound: a fold
//! requires every argument constant, and `HeapCache::_escape_box` opens with
//! `if opref.is_constant() { return; }`, exactly as upstream's `_escape_box`
//! acts only on `RefFrontendOp`. That is a licence to skip invalidation when
//! folding, **not** a licence to widen the foldable opcode set.

use majit_ir::{OpCode, OpRef, Type, Value, descr::DescrRef};

use crate::trace_ctx::TraceCtx;

impl TraceCtx {
    /// `pyjitpl.py MetaInterp.execute_and_record(opnum, descr, *argboxes)`.
    ///
    /// Returns a `Const*` OpRef when the operation folded and nothing was
    /// recorded, else the recorded op's OpRef, already stamped with
    /// `resvalue`. Callers distinguish the two with `.is_constant()` — that is
    /// upstream's `isinstance(resbox, Const)`.
    ///
    /// `cpu` and `last_exc_value` are the two pieces of `MetaInterp` state
    /// the funnel needs and `TraceCtx` does not hold, so both are passed.
    /// `cpu` is upstream's `self.cpu`, which `executor.execute` takes as an
    /// explicit argument anyway; production installs a real one through
    /// `MetaInterp::set_cpu`, so reaching for `cpu::default_cpu()` here would
    /// fold memory reads against the wrong backend. `last_exc_value` is
    /// upstream's `self.last_exc_value`, which majit splits across
    /// `JitCodeMachine::last_exception_value`, `MetaInterp::last_exc_box` and
    /// the `blackhole::BH_LAST_EXC_VALUE` thread-local — a caller must name
    /// which one it means rather than have a fourth copy minted here.
    ///
    /// The `is_pure_with_descr` test is structurally first, and must stay
    /// there. `_all_constants` is **vacuously true** for a zero-argument
    /// operation, so every `New`, `NewWithVtable`, `ForceToken`,
    /// `LeavePortalFrame` and `Keepalive` would fold — deleting every
    /// allocation and every portal marker from the trace — if the predicate
    /// were dropped in favour of an arguments-only test.
    pub fn execute_and_record(
        &mut self,
        cpu: &dyn crate::cpu::Cpu,
        opnum: OpCode,
        descr: Option<DescrRef>,
        args: &[OpRef],
        resvalue: Option<Value>,
        last_exc_value: i64,
    ) -> OpRef {
        // `assert not (rop._CANRAISE_FIRST <= opnum <= rop._CANRAISE_LAST)` —
        // a can-raise operation belongs to `execute_and_record_varargs`.
        // Spelled with the `is_ovf` escape because the two ranges nest the
        // other way round here: `CANRAISE_LAST` is `IntMulOvf`, so
        // `is_ovf()` is a subset of `can_raise()`, where `resoperation.py`
        // closes the can-raise range before `_OVF_FIRST` opens and can spell
        // the assert bare.
        debug_assert!(
            !opnum.can_raise() || opnum.is_ovf(),
            "execute_and_record: {opnum:?} can raise",
        );
        // `profiler.count_ops(opnum)` — counted on both paths.
        self.profiler().count_ops(opnum, crate::counters::OPS);

        if opnum.is_pure_with_descr(descr.as_ref()) || (opnum.is_ovf() && last_exc_value == 0) {
            // `canfold = self._all_constants(*argboxes)`. `OpRef::is_constant`
            // is the variant test, so it answers the `isinstance(box, Const)`
            // question for every Const subclass at once. The extra
            // `resvalue.is_some()` conjunct is the `None` rule from the module
            // doc: a caller that could not compute the value has no register
            // to receive a constant either.
            if resvalue.is_some()
                && args.iter().all(|a| a.is_constant())
                && let Some(folded) = Self::fold_value(cpu, opnum, args, descr.as_ref())
            {
                // `executor.wrap_constant(resvalue)`. Upstream's structural
                // guarantee that no foldable opcode has result type `/n`.
                debug_assert_ne!(opnum.result_type(), Type::Void);
                return OpRef::const_inline_from_value(&folded);
            }
        }
        self.record_helper(opnum, resvalue, descr, args)
    }

    /// `pyjitpl.py MetaInterp._record_helper(opnum, resvalue, descr, *argboxes)`.
    ///
    /// Upstream returns `None` for `op.type == 'v'`; here a void operation
    /// still gets a real position from `recorder::Trace::record_op`, and
    /// `set_opref_concrete` asserts every recorded op has one, so the OpRef is
    /// returned for every result type.
    fn record_helper(
        &mut self,
        opnum: OpCode,
        resvalue: Option<Value>,
        descr: Option<DescrRef>,
        args: &[OpRef],
    ) -> OpRef {
        self.profiler()
            .count_ops(opnum, crate::counters::RECORDED_OPS);
        // `self.heapcache.invalidate_caches(...)` stays at the call sites —
        // see the module doc.
        let op = match descr {
            Some(d) => self.record_op_with_descr(opnum, args, d),
            None => self.record_op(opnum, args),
        };
        // `self.attach_debug_info(op)` is a documented no-op stub.
        if let Some(v) = resvalue {
            self.set_opref_concrete(op, v);
        }
        op
    }

    /// The constant the fold wraps, or `None` when the operation must be
    /// recorded after all.
    ///
    /// Both non-folding outcomes of `execute_nonspec_const` record.
    /// `Ok(None)` is a registered evaluator declining its operands — an
    /// out-of-range shift, a zero divisor, an OVF opcode that overflowed, a
    /// non-finite `cast_float_to_int` — and declining is a decision.
    /// `Err(NoConstExecutor)` is upstream's `NotImplementedError`, which
    /// `Optimizer::constant_fold` spells as a `panic!`. This funnel records
    /// instead: it runs inside the tracer, where a panic is caught and
    /// downgraded to a silent `TraceAction::Abort`, so an evaluator gap would
    /// cost the whole loop rather than one folded operation.
    fn fold_value(
        cpu: &dyn crate::cpu::Cpu,
        opnum: OpCode,
        args: &[OpRef],
        descr: Option<&DescrRef>,
    ) -> Option<Value> {
        let vals: Vec<Value> = args
            .iter()
            .map(|a| {
                a.inline_const_to_value()
                    .expect("_all_constants held for every argument")
            })
            .collect();
        crate::executor::execute_nonspec_const(cpu, opnum, &vals, descr, opnum.result_type())
            .ok()
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::Trace;

    fn fresh_ctx() -> TraceCtx {
        TraceCtx::new(
            Trace::new(),
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        )
    }

    /// `_all_constants` is vacuously true for a zero-argument operation, so
    /// the funnel is kept off `ForceToken`, `New`, `NewWithVtable`,
    /// `LeavePortalFrame` and `Keepalive` only by `is_pure_with_descr`
    /// answering false first. Simplifying the gate into an arguments-only
    /// test deletes every allocation and every portal marker from the trace.
    #[test]
    fn a_zero_argument_impure_op_is_recorded_not_folded() {
        let mut ctx = fresh_ctx();
        let cpu = crate::cpu::default_cpu();
        let before = ctx.num_ops();
        let op = ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::ForceToken,
            None,
            &[],
            Some(Value::Int(7)),
            0,
        );
        assert!(!op.is_constant(), "ForceToken must not fold");
        assert_eq!(ctx.num_ops(), before + 1);
    }

    #[test]
    fn an_all_constant_pure_op_folds_and_records_nothing() {
        let mut ctx = fresh_ctx();
        let cpu = crate::cpu::default_cpu();
        let before = ctx.num_ops();
        let op = ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::IntAdd,
            None,
            &[OpRef::ConstInt(2), OpRef::ConstInt(3)],
            Some(Value::Int(5)),
            0,
        );
        assert_eq!(op, OpRef::ConstInt(5));
        assert_eq!(ctx.num_ops(), before, "a fold records nothing");
    }

    #[test]
    fn a_non_constant_argument_blocks_the_fold() {
        let mut recorder = Trace::new();
        let arg = recorder.record_input_arg(majit_ir::Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let cpu = crate::cpu::default_cpu();
        let op = ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::IntAdd,
            None,
            &[arg, OpRef::ConstInt(3)],
            Some(Value::Int(5)),
            0,
        );
        assert!(!op.is_constant());
        assert_eq!(
            ctx.box_value(op),
            Some(Value::Int(5)),
            "resvalue is stamped"
        );
    }

    /// `execute_binary_int_const` declines `IntFloorDiv` by zero rather than
    /// answering the tracer's 0, so the operation is recorded and the trace
    /// and the optimizer cannot disagree about what the constant is.
    #[test]
    fn a_declining_evaluator_records_instead_of_folding() {
        let mut ctx = fresh_ctx();
        let cpu = crate::cpu::default_cpu();
        let before = ctx.num_ops();
        let op = ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::IntFloorDiv,
            None,
            &[OpRef::ConstInt(5), OpRef::ConstInt(0)],
            Some(Value::Int(0)),
            0,
        );
        assert!(!op.is_constant(), "div by zero must not fold");
        assert_eq!(ctx.num_ops(), before + 1);
    }

    /// `_OVF_FIRST <= opnum <= _OVF_LAST and not self.last_exc_value` — the
    /// ovf arm of the fold gate is closed while an exception is pending.
    #[test]
    fn a_pending_exception_closes_the_ovf_fold_gate() {
        let mut ctx = fresh_ctx();
        let cpu = crate::cpu::default_cpu();
        let args = [OpRef::ConstInt(2), OpRef::ConstInt(3)];

        let before = ctx.num_ops();
        let clear = ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::IntAddOvf,
            None,
            &args,
            Some(Value::Int(5)),
            0,
        );
        assert_eq!(clear, OpRef::ConstInt(5));
        assert_eq!(ctx.num_ops(), before);

        let pending = ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::IntAddOvf,
            None,
            &args,
            Some(Value::Int(5)),
            0xdead,
        );
        assert!(
            !pending.is_constant(),
            "a pending exception blocks the fold"
        );
        assert_eq!(ctx.num_ops(), before + 1);
    }

    /// `checked_add` in `execute_binary_int_const` means a constant pair that
    /// really overflows declines the fold, so the op and its guard survive.
    #[test]
    fn an_overflowing_constant_ovf_pair_is_recorded() {
        let mut ctx = fresh_ctx();
        let cpu = crate::cpu::default_cpu();
        let before = ctx.num_ops();
        let op = ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::IntAddOvf,
            None,
            &[OpRef::ConstInt(i64::MAX), OpRef::ConstInt(1)],
            Some(Value::Int(i64::MIN)),
            0,
        );
        assert!(!op.is_constant());
        assert_eq!(ctx.num_ops(), before + 1);
    }

    /// The C6 trap, pinned: `CANRAISE_LAST` is `IntMulOvf` here, so the OVF
    /// range sits *inside* the can-raise range, where `resoperation.py` closes
    /// the can-raise range before `_OVF_FIRST` opens. A literal port of
    /// `assert not (_CANRAISE_FIRST <= opnum <= _CANRAISE_LAST)` would reject
    /// every OVF opcode the funnel exists to fold.
    #[test]
    fn every_ovf_opcode_also_answers_can_raise() {
        for opnum in [OpCode::IntAddOvf, OpCode::IntSubOvf, OpCode::IntMulOvf] {
            assert!(opnum.is_ovf(), "{opnum:?}");
            assert!(opnum.can_raise(), "{opnum:?}");
        }
    }

    /// `resvalue: None` — no trace-time concrete available — never folds,
    /// always records, never stamps.
    #[test]
    fn a_missing_resvalue_records_without_stamping() {
        let mut ctx = fresh_ctx();
        let cpu = crate::cpu::default_cpu();
        let op = ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::IntAdd,
            None,
            &[OpRef::ConstInt(2), OpRef::ConstInt(3)],
            None,
            0,
        );
        assert!(!op.is_constant());
        assert_eq!(ctx.box_value(op), None);
    }

    /// `OPS` counts every operation the funnel is asked for and
    /// `RECORDED_OPS` only those it records, so the two differ by exactly the
    /// folds. Each is bumped from one place, and this is what says so: a
    /// second bump on either — or a fold that reached `record_helper` anyway —
    /// changes a number no committed baseline can hold, because neither count
    /// has a healthy value and a successful fold lowers `RECORDED_OPS` by
    /// design.
    #[test]
    fn the_funnel_counts_ops_once_and_recorded_ops_only_when_it_records() {
        let mut ctx = fresh_ctx();
        let cpu = crate::cpu::default_cpu();
        let counts = |ctx: &TraceCtx| {
            (
                ctx.profiler().get_counter(crate::counters::OPS),
                ctx.profiler().get_counter(crate::counters::RECORDED_OPS),
            )
        };
        assert_eq!(counts(&ctx), (Some(0), Some(0)));

        ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::IntAdd,
            None,
            &[OpRef::ConstInt(2), OpRef::ConstInt(3)],
            Some(Value::Int(5)),
            0,
        );
        assert_eq!(
            counts(&ctx),
            (Some(1), Some(0)),
            "a fold is counted, not recorded"
        );

        ctx.execute_and_record(
            cpu.as_ref(),
            OpCode::ForceToken,
            None,
            &[],
            Some(Value::Int(7)),
            0,
        );
        assert_eq!(
            counts(&ctx),
            (Some(2), Some(1)),
            "a record is counted on both"
        );
    }
}
