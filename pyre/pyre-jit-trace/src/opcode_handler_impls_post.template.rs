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

    #[inline(never)]
    fn close_loop_args(
        &mut self,
        target: usize,
    ) -> Result<Option<Vec<Self::Value>>, pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            // pyjitpl.py:2950-3036 reached_loop_header
            let code_ptr = unsafe { (*this.sym().jitcode).code };
            let back_edge_key = crate::driver::make_green_key(code_ptr, target);
            // pyjitpl.py:2951 self.heapcache.reset()
            ctx.heap_cache_mut().reset();
            // pyjitpl.py:2957-2965 build live_arg_boxes ONCE.
            let live_args =
                crate::state::MIFrame::close_loop_args_at(this, ctx, Some(target));
            let live_green_boxes: Vec<majit_metainterp::GreenBox> = live_args
                .iter()
                .map(|opref| {
                    let ty = ctx.get_opref_type(*opref).unwrap_or_else(|| {
                        panic!(
                            "live_arg {opref:?} has no type in OptContext; \
                             RPython Box always carries its type"
                        )
                    });
                    majit_metainterp::GreenBox::new(*opref, ty)
                })
                .collect();

            // pyjitpl.py:2978-2983 compile_trace attempt.
            {
                let (driver, _) = crate::driver::driver_pair();
                let has_partial = driver.meta_interp().partial_trace().is_some();
                let bridge_origin = driver
                    .meta_interp()
                    .bridge_info()
                    .map(|b| (b.trace_id, b.fail_index));
                let has_targets = driver.meta_interp().has_compiled_targets(back_edge_key);
                if !has_partial && has_targets {
                    let outcome = match bridge_origin {
                        // Guard-origin: existing bridge path.
                        Some(_) => Some(driver.meta_interp_mut().compile_trace(
                            back_edge_key,
                            &live_args,
                            bridge_origin,
                        )),
                        // pyjitpl.py:3003-3007 interp-origin: close the trace as
                        // an entry bridge (ResumeFromInterpDescr) jumping into the
                        // already-compiled loop at `back_edge_key`, instead of
                        // falling through to compile_loop -> has_compiled_targets
                        // -> SwitchToBlackhole(ABORT_BAD_LOOP). compile_and_attach
                        // installs an entry bridge ending in a JUMP to the target
                        // loop (compile.py:1002-1021).
                        None => {
                            // `compile_trace_entry_data` returns `Some` only for
                            // a function-entry trace (ResumeFromInterpDescr: the
                            // interpreter portal into the function). Such a trace
                            // runs the function prologue and closes with a JUMP
                            // into the already-compiled hot loop at
                            // `back_edge_key`. A trace rooted at a *loop header*
                            // returns `None` here and falls through to
                            // compile_loop, which aborts cleanly — closing one
                            // loop header as an entry bridge into another would
                            // drop the outer loop's back-edge (its iterations
                            // would no longer run in compiled code).
                            match driver.compile_trace_entry_data() {
                                Some((original_green_key, entry_meta)) => {
                                    Some(driver.meta_interp_mut().compile_trace_from_interp(
                                        back_edge_key,
                                        &live_args,
                                        original_green_key,
                                        entry_meta,
                                    ))
                                }
                                None => Some(driver.meta_interp_mut().compile_trace(
                                    back_edge_key,
                                    &live_args,
                                    None,
                                )),
                            }
                        }
                    };
                    if matches!(outcome, Some(majit_metainterp::CompileOutcome::Compiled { .. })) {
                        if majit_metainterp::majit_log_enabled() {
                            eprintln!(
                                "[jit][reached_loop_header] compile_trace success: key={} pc={} bridge={:?}",
                                back_edge_key, target, bridge_origin
                            );
                        }
                        // pyjitpl.py:3095 raise_if_successful() — successful
                        // compile_trace raises ContinueRunningNormally and
                        // unwinds; no value propagates to the caller. Pyre
                        // ports that as an early `return Ok(None)` so the
                        // tracer stops at this point rather than falling
                        // through to the loop-close path.
                        driver.note_compile_trace_success();
                        return Ok(None);
                    }
                }
            }
            // pyjitpl.py:2994-3036 search current_merge_points; the
            // shape assert at line 2996 fires for every visited merge
            // point, regardless of whether the greenkey matches —
            // structural red-bank invariant under the same jitdriver.
            if !ctx.has_merge_point_with_shape_assert(back_edge_key, live_args.len()) {
                // pyjitpl.py:3034-3036 first visit, register & continue
                ctx.add_merge_point(back_edge_key, live_green_boxes, target);
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
                    .map(crate::state::FrontendOp::void)
                    .collect(),
            ))
        })
    }
}

impl pyre_interpreter::BranchOpcodeHandler for crate::state::MIFrame {
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
        other_target: usize,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::record_branch_guard(
                this,
                ctx,
                value.opref,
                truth,
                concrete_truth,
                other_target,
            );
            Ok(())
        })
    }
}

impl pyre_interpreter::NamespaceOpcodeHandler for crate::state::MIFrame {
    fn load_name_value(
        &mut self,
        name: &str,
        _nameindex: usize,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let w_globals_obj = self.sym().concrete_namespace;
        // `celldict.py:285-322 _LOAD_GLOBAL_cached` + `:42-54
        // getdictvalue_no_unwrapping`: the JIT global fast path promotes
        // the module dict's `version?` quasi-immutable, folds the stored
        // value-or-cell for `name`, then reads `cell.w_value` LIVE.  The
        // cell path applies only to a `W_ModuleDictObject` still in
        // `ModuleDictStrategy` mode whose slot holds a raw value or an
        // `ObjectMutableCell`; plain dict globals, post-
        // `switch_to_object_strategy` module dicts, absent/null keys, and
        // `IntMutableCell` slots (which need `w_int_new` boxing of
        // `intvalue`) fall through to the layout-agnostic name-based path
        // (`_load_global` -> `space.finditem_str`).
        let cell_path = crate::state::module_dict_cell_slot_direct(w_globals_obj, name).and_then(
            |slot| {
                let stored = crate::state::module_dict_cell_value_direct(w_globals_obj, slot)?;
                if stored.is_null()
                    || unsafe { pyre_object::celldict::is_int_mutable_cell(stored) }
                {
                    return None;
                }
                Some((slot, stored))
            },
        );
        let Some((slot, stored)) = cell_path else {
            let frame = self.trace_frame();
            let globals = self.trace_globals_ptr();
            let opref = self.with_trace_ctx(|ctx| {
                crate::helpers::emit_trace_load_name_from_namespace(ctx, frame, globals, name)
            });
            // `pyopcode.py:958-967 _load_global`: concrete value at
            // trace time mirrors the extern's full fallback chain —
            // globals strategy dispatch, then
            // frame.get_builtin().getdictvalue.  `jit_load_name_from_namespace`
            // performs exactly that chain (and null-checks globals), so
            // it gives the correct concrete for module dicts, plain
            // dict globals, and builtin hits alike.
            let frame_ptr = self.concrete_frame_addr as i64;
            let name_bytes = name.as_bytes();
            let obj = crate::helpers::jit_load_name_from_namespace(
                frame_ptr,
                w_globals_obj as i64,
                name_bytes.as_ptr() as i64,
                name_bytes.len() as i64,
            ) as pyre_object::PyObjectRef;
            let result_concrete = crate::state::ConcreteValue::from_pyobj(obj);
            return Ok(crate::state::FrontendOp::new(opref, result_concrete));
        };
        // `typeobject.py:48-51 unwrap_cell`: the concrete LOAD result is
        // the unwrapped value (identity for a raw stored value).
        let is_obj_cell = unsafe { pyre_object::celldict::is_object_mutable_cell(stored) };
        let result_obj = unsafe { pyre_object::celldict::unwrap_cell(stored) };
        let result_concrete = crate::state::ConcreteValue::from_pyobj(result_obj);
        let opref = self.with_ctx(|_this, ctx| {
            let ns_const = ctx.const_ref(w_globals_obj as i64);
            let slot_const = ctx.const_int(slot as i64);
            // `quasiimmut.py` + `celldict.py:34 _immutable_fields_=["version?"]`:
            // 1. QUASIIMMUT_FIELD(ns, slot) — recorded as a version dep +
            //    GUARD_NOT_INVALIDATED; the compile-time glue registers the
            //    loop's invalidation flag on the strategy's version watchers.
            // 2. RECORD_KNOWN_RESULT(cell, helper, ns, slot) — seed the
            //    trace-time cell fold into call_pure_results.
            // 3. CALL_PURE_R(helper, ns, slot) — elidable cell lookup folds
            //    to the constant cell (or raw value) pointer.
            ctx.record_op(majit_ir::OpCode::QuasiimmutField, &[ns_const, slot_const]);
            let lookup_fn = crate::helpers::jit_namespace_cell_lookup as *const ();
            let lookup_args = [ns_const, slot_const];
            let lookup_arg_types = [majit_ir::Type::Ref, majit_ir::Type::Int];
            ctx.record_known_result_typed(
                stored as i64,
                lookup_fn,
                &lookup_args,
                &lookup_arg_types,
                majit_ir::Type::Ref,
                majit_metainterp::EffectInfoSlot::ElidableCannotRaise,
            );
            let concrete_args =
                crate::helpers::namespace_slot_lookup_values(lookup_fn, w_globals_obj, slot);
            let concrete_cell = crate::helpers::namespace_slot_lookup_result(stored);
            let cell_opref = crate::helpers::emit_trace_call_ref_typed_elidable_cannot_raise(
                ctx,
                lookup_fn,
                &lookup_args,
                &lookup_arg_types,
                &concrete_args,
                concrete_cell,
            );
            // When the fold produced an `ObjectMutableCell`, read
            // `cell.w_value` LIVE so a same-key reassign (in-place
            // `write_cell`, no version bump) is observed each iteration; a
            // raw stored value needs no indirection.
            let opref = if is_obj_cell {
                crate::state::opimpl_getfield_gc_r(
                    ctx,
                    cell_opref,
                    crate::descr::object_mutable_cell_value_descr(),
                )
            } else {
                cell_opref
            };
            Ok::<_, pyre_interpreter::PyError>(opref)
        })?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn store_name_value(
        &mut self,
        name: &str,
        _nameindex: usize,
        value: Self::Value,
    ) -> Result<(), pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        // `celldict.py:328-333 STORE_GLOBAL_cached`: under the JIT the
        // cell/slot path is bypassed entirely —
        // `self.space.setitem_str(self.get_w_globals(), varname, w_newvalue)`.
        // The emitted IR is layout-agnostic: `w_dict_setitem_str` on the
        // globals dict object runs the full ModuleDictStrategy.setitem_str
        // (write_cell + version mutated()) and mirrors the str-keyed write
        // into the paired DictStorage, so the LOAD fast path still observes
        // it.
        let globals = self.trace_globals_ptr();
        self.with_trace_ctx(|ctx| {
            crate::helpers::emit_trace_store_name_to_namespace(ctx, globals, name, value.opref);
            Ok(())
        })
    }

    fn null_value(&mut self) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let opref = self.with_trace_ctx(|ctx| ctx.const_int(0));
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
        let opref = self.unary_int_value(value.opref, majit_ir::OpCode::IntNeg, concrete_val)?;
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
        let opref = self.unary_int_value(value.opref, majit_ir::OpCode::IntInvert, concrete_val)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }
}
