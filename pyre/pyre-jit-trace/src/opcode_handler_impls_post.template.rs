
impl pyre_interpreter::ControlFlowOpcodeHandler for crate::state::MIFrame {
    fn fallthrough_target(&mut self) -> usize {
        self.fallthrough_pc()
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::set_next_instr(this, ctx, target);
            Ok(())
        })
    }

    fn close_loop_args(
        &mut self,
        target: usize,
    ) -> Result<Option<Vec<Self::Value>>, pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            // pyjitpl.py:2950-3036 reached_loop_header
            let code_ptr = unsafe { (*this.sym().jitcode).code };
            let back_edge_key = crate::driver::make_green_key(code_ptr, target);
            // pyjitpl.py:2951 self.heapcache.reset()
            ctx.reset_heap_cache();
            // pyjitpl.py:2957-2965 build live_arg_boxes ONCE.
            let live_args =
                crate::state::MIFrame::close_loop_args_at(this, ctx, Some(target));
            let live_types = {
                let s = this.sym();
                let mut types = crate::virtualizable_gen::virt_live_value_types(0);
                types.extend(s.symbolic_local_types.iter().copied());
                let stack_only = s.stack_only_depth();
                types.extend(
                    s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())]
                        .iter()
                        .copied(),
                );
                types
            };

            // pyjitpl.py:2978-2983 compile_trace attempt.
            {
                let (driver, _) = crate::driver::driver_pair();
                let has_partial = driver.meta_interp().has_partial_trace();
                let bridge_origin = driver.bridge_origin();
                if !has_partial && driver.meta_interp().has_compiled_targets(back_edge_key) {
                    let outcome = driver.meta_interp_mut().compile_trace(
                        back_edge_key,
                        &live_args,
                        bridge_origin,
                    );
                    if matches!(outcome, majit_metainterp::CompileOutcome::Compiled { .. }) {
                        if majit_metainterp::majit_log_enabled() {
                            eprintln!(
                                "[jit][reached_loop_header] compile_trace success: key={} pc={} bridge={:?}",
                                back_edge_key, target, bridge_origin
                            );
                        }
                        return Ok(Some(
                            live_args
                                .into_iter()
                                .map(crate::state::FrontendOp::opref_only)
                                .collect(),
                        ));
                    }
                }
            }
            // pyjitpl.py:2994-3036 search current_merge_points
            if !ctx.has_merge_point(back_edge_key) {
                // pyjitpl.py:3034-3036 first visit, register & continue
                ctx.add_merge_point(back_edge_key, live_args, live_types, target);
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][reached_loop_header] first visit, unroll: key={} pc={}",
                        back_edge_key, target
                    );
                }
                return Ok(None);
            }
            // pyjitpl.py:3002-3030 found, compile as loop.
            Ok(Some(
                live_args
                    .into_iter()
                    .map(crate::state::FrontendOp::opref_only)
                    .collect(),
            ))
        })
    }
}

impl pyre_interpreter::BranchOpcodeHandler for crate::state::MIFrame {
    fn enter_branch_truth(
        &mut self,
        value: Self::Value,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.sym_mut().pending_branch_value = Some(value.opref);
        Ok(())
    }

    fn leave_branch_truth(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let sym = self.sym_mut();
        sym.pending_branch_value = None;
        sym.pending_branch_other_target = None;
        Ok(())
    }

    fn set_branch_other_target(&mut self, target: usize) {
        self.sym_mut().pending_branch_other_target = Some(target);
    }

    fn branch_other_target(&self) -> Option<usize> {
        self.sym().pending_branch_other_target
    }

    fn concrete_truth_as_bool(
        &mut self,
        value: Self::Value,
        _truth: Self::Truth,
    ) -> Result<bool, pyre_interpreter::PyError> {
        crate::state::MIFrame::concrete_branch_truth_for_value(
            self,
            value.opref,
            value.concrete.to_pyobj(),
        )
    }

    fn guard_truth_value(
        &mut self,
        truth: Self::Truth,
        expect_true: bool,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            let opcode = if expect_true {
                majit_ir::OpCode::GuardTrue
            } else {
                majit_ir::OpCode::GuardFalse
            };
            crate::state::MIFrame::generate_guard(this, ctx, opcode, &[truth]);
            Ok(())
        })
    }

    fn record_branch_guard(
        &mut self,
        value: Self::Value,
        truth: Self::Truth,
        concrete_truth: bool,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::record_branch_guard(
                this,
                ctx,
                value.opref,
                truth,
                concrete_truth,
            );
            Ok(())
        })
    }
}

impl pyre_interpreter::NamespaceOpcodeHandler for crate::state::MIFrame {
    fn load_name_value(
        &mut self,
        name: &str,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let ns = self.sym().concrete_namespace;
        let Some(slot) = crate::state::namespace_slot_direct(ns, name) else {
            let opref = self.trace_load_name(name)?;
            return Ok(crate::state::FrontendOp::opref_only(opref));
        };
        let concrete_cv = crate::state::namespace_value_direct(ns, slot);
        let result_concrete = concrete_cv
            .map(crate::state::ConcreteValue::from_pyobj)
            .unwrap_or(crate::state::ConcreteValue::Null);
        if let Some(concrete_value) = concrete_cv {
            if !concrete_value.is_null() {
                // celldict.py @elidable_promote + quasiimmut.py parity:
                // 1. QUASIIMMUT_FIELD(ns, slot) — collected as quasi_immutable_deps
                //    + GUARD_NOT_INVALIDATED.
                // 2. RECORD_KNOWN_RESULT(result, ns, slot) — cache trace-time
                //    lookup result (call_pure_results).
                // 3. CALL_PURE_R(ns, slot) — elidable lookup; record_result_of_call_pure
                //    folds via OptPure lookup_known_result.
                let opref = self.with_ctx(|this, ctx| {
                    let ns_const = ctx.const_ref(ns as i64);
                    let slot_const = ctx.const_int(slot as i64);
                    ctx.record_op(majit_ir::OpCode::QuasiimmutField, &[ns_const, slot_const]);
                    let result_const = ctx.const_ref(concrete_value as i64);
                    ctx.record_op(
                        majit_ir::OpCode::RecordKnownResult,
                        &[result_const, ns_const, slot_const],
                    );
                    let call_result =
                        ctx.record_op(majit_ir::OpCode::CallPureR, &[ns_const, slot_const]);
                    this.sym_mut()
                        .symbolic_namespace_slots
                        .insert(slot, call_result);
                    Ok::<_, pyre_interpreter::PyError>(call_result)
                })?;
                return Ok(crate::state::FrontendOp::new(opref, result_concrete));
            }
        }
        let opref = self.with_ctx(|this, ctx| {
            crate::state::MIFrame::load_namespace_value(this, ctx, slot)
        })?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn store_name_value(
        &mut self,
        name: &str,
        value: Self::Value,
    ) -> Result<(), pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let ns = self.sym().concrete_namespace;
        let Some(slot) = crate::state::namespace_slot_direct(ns, name) else {
            return self.trace_store_name(name, value.opref);
        };
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::store_namespace_value(this, ctx, slot, value.opref)
        })
    }

    fn null_value(&mut self) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let opref = self.trace_null_value()?;
        Ok(crate::state::FrontendOp::new(
            opref,
            crate::state::ConcreteValue::Ref(pyre_object::PY_NULL),
        ))
    }
}

impl pyre_interpreter::ArithmeticOpcodeHandler for crate::state::MIFrame {
    fn binary_value(
        &mut self,
        a_fop: Self::Value,
        b_fop: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let a = a_fop.opref;
        let b = b_fop.opref;
        let lhs_obj = a_fop.concrete.to_pyobj();
        let rhs_obj = b_fop.concrete.to_pyobj();
        // Concrete result via interpreter dispatch (baseobjspace).
        let result_concrete = crate::concrete_binary_value(op, lhs_obj, rhs_obj);
        if matches!(op, BinaryOperator::Subscr) {
            let fop = self.binary_subscr_value(a, b, lhs_obj, rhs_obj)?;
            let concrete = if result_concrete.is_null() {
                fop.concrete
            } else {
                result_concrete
            };
            return Ok(crate::state::FrontendOp::new(fop.opref, concrete));
        }
        let is_float_path = (!lhs_obj.is_null()
            && !rhs_obj.is_null()
            && unsafe { pyre_object::is_float(lhs_obj) || pyre_object::is_float(rhs_obj) })
            || self.value_type(a) == majit_ir::Type::Float
            || self.value_type(b) == majit_ir::Type::Float;
        let opref = if is_float_path {
            self.binary_float_value(a, b, op, lhs_obj, rhs_obj)?
        } else {
            self.binary_int_value(a, b, op, lhs_obj, rhs_obj)?
        };
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn compare_value(
        &mut self,
        a_fop: Self::Value,
        b_fop: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let a = a_fop.opref;
        let b = b_fop.opref;
        let lhs_obj = a_fop.concrete.to_pyobj();
        let rhs_obj = b_fop.concrete.to_pyobj();
        // Concrete result via interpreter dispatch (baseobjspace::compare).
        let result_concrete = crate::concrete_compare_value(op, lhs_obj, rhs_obj);
        let opref = self.compare_value_direct(a, b, op, lhs_obj, rhs_obj)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn unary_negative_value(
        &mut self,
        value: Self::Value,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let concrete_val = value.concrete.to_pyobj();
        let mut result_concrete = crate::state::ConcreteValue::Null;
        if !concrete_val.is_null() && unsafe { pyre_object::is_int(concrete_val) } {
            let v = unsafe { pyre_object::w_int_get_value(concrete_val) };
            result_concrete = crate::state::ConcreteValue::Int(v.wrapping_neg());
        }
        let opref =
            self.unary_int_value(value.opref, majit_ir::OpCode::IntNeg, concrete_val)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn unary_invert_value(
        &mut self,
        value: Self::Value,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let concrete_val = value.concrete.to_pyobj();
        let mut result_concrete = crate::state::ConcreteValue::Null;
        if !concrete_val.is_null() && unsafe { pyre_object::is_int(concrete_val) } {
            let v = unsafe { pyre_object::w_int_get_value(concrete_val) };
            result_concrete = crate::state::ConcreteValue::Int(!v);
        }
        let opref =
            self.unary_int_value(value.opref, majit_ir::OpCode::IntInvert, concrete_val)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }
}
