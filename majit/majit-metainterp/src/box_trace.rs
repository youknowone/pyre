//! Trace-time box / unbox / binop / compare recording for boxed primitives.
//!
//! Interpreter-agnostic: each helper takes the boxed primitive's type
//! address and field descrs as parameters and emits only generic IR —
//! `GuardClass` + `GetfieldGc` to unbox, the typed op plus an overflow
//! guard, and `NewWithVtable` + `SetfieldGc` to rebox. Per-operator
//! dispatch and concrete computation stay with the interpreter; only the
//! recording sequence lives here. (`int`/`float` name the IR primitive
//! width, not any interpreter object type.)

/// Unbox a Python int object: emit GuardClass + GetfieldGc(I|PureI).
///
/// Auto-generated equivalent of PyPy's int_unbox annotation.
/// Returns the raw i64 OpRef, with heapcache integration.
fn getfield_gc_i_pureornot(
    ctx: &mut crate::TraceCtx,
    obj: majit_ir::OpRef,
    descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    use majit_ir::Value;
    // heapcache: check if this field was already read/written in this trace
    let field_index = descr.index();
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:934-945 cache-hit sanity check (int arm). The
        // line-by-line port runs `executor.execute(cpu, mi, opnum,
        // fielddescr, box)` and asserts `resvalue ==
        // upd.currfieldbox.getint()`. Pyre projects the struct
        // pointer through `concrete_of_opref(obj)`; the cached Box's
        // intrinsic value is fetched via `box_value(cached)` —
        // covering const pool, standard-virtualizable shadow, and
        // the frontend object's `value` field (RPython
        // `currfieldbox.getint()` dispatch parity).
        let expected_int = match ctx.box_value(cached) {
            Some(Value::Int(n)) => Some(n),
            _ => None,
        };
        if let Some(cached_int) = expected_int {
            if let Some(Value::Ref(struct_ref)) = ctx.box_value(obj) {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    if let Some(Value::Int(loaded)) =
                        ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
                    {
                        assert_eq!(
                            loaded, cached_int,
                            "_opimpl_getfield_gc_any_pureornot sanity \
                             check (int): loaded {loaded} != cached \
                             {cached_int} (field_index={field_index}, \
                             struct_ptr={struct_ptr:#x})"
                        );
                    }
                }
            }
        }
        // pyjitpl.py:946 profiler.count_ops(rop.GETFIELD_GC_I,
        // Counters.HEAPCACHED_OPS) — folded-away op accounting on cache hit.
        ctx.profiler()
            .count_ops(OpCode::GetfieldGcI, crate::counters::HEAPCACHED_OPS);
        return cached;
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureI
    } else {
        OpCode::GetfieldGcI
    };
    let result = ctx.record_op_with_descr(opcode, &[obj], descr.clone());
    // pyjitpl.py:948-949 — pair the recorded opref with the live int
    // payload so subsequent `box_value(result)` mirrors RPython's
    // executor-returned Box (history.py BoxInt(value=...)).
    // `box_value` exposes the same Box.value chain PyPy reads via
    // `obj.getref_base()`.
    let live_value = if let Some(Value::Ref(struct_ref)) = ctx.box_value(obj) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
        } else {
            None
        }
    } else {
        None
    };
    // RPython `Box(value)` constructor analog — stamp the recorded
    // OpRef so subsequent `box_value(result)` consumers see the
    // runtime concrete instead of the GcRef(usize::MAX) sentinel.
    if let Some(live_value) = live_value {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

pub fn trace_unbox_int(
    ctx: &mut crate::TraceCtx,
    obj: majit_ir::OpRef,
    int_type_addr: i64,
    _ob_type_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    // GUARD_CLASS(box, cls): guard takes object box directly,
    // backend loads typeptr at offset 0 (llgraph/runner.py:1245).
    // Production callers (`trace_unbox_int_with_resume_descr`,
    // `trace_unbox_float_with_resume`) already emit the GuardClass via
    // `frame.generate_guard` — by the time control reaches here,
    // `is_class_known(obj)` returns true and this branch is dead.  In
    // unit tests (`pyre/pyre-jit/src/trace_verify.rs`) the GuardClass is
    // recorded with no resume context; the test asserts only on the
    // emitted opcode shape.
    if !obj.is_constant() && !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(int_type_addr);
        ctx.record_guard_typed(OpCode::GuardClass, &[obj, type_const], Vec::new());
        ctx.heap_cache_mut().class_now_known(obj, int_type_addr);
    }
    getfield_gc_i_pureornot(ctx, obj, intval_descr)
}

/// Box a raw i64 into a Python int object.
///
/// Auto-generated equivalent of PyPy's int_box annotation.
/// The helper route preserves pyre's small-int cache instead of always
/// materializing a fresh heap object in the trace.
pub fn trace_box_int(
    ctx: &mut crate::TraceCtx,
    value: majit_ir::OpRef,
    size_descr: majit_ir::DescrRef,
    _ob_type_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
    _int_type_addr: i64,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;

    // Inline W_Int allocation so OptVirtualize can see the object shape
    // and fold later GetfieldRawI(intval) reads back to `value`.
    // RPython parity: NEW_WITH_VTABLE (not NEW) for classes with vtable.
    // jtransform.py:908-911 parity: typeptr setfield filtered in trace.
    // rewrite.py:479-484 GC rewriter emits vtable via fielddescr_vtable.
    let obj = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(obj);
    let intval_idx = intval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[obj, value], intval_descr);
    // `upd.setfield(valuebox)` parity — the cache stores the Box
    // identity (`value` OpRef); cache-hit readers fetch the
    // intrinsic value via `box_value(cached)` at hit time.
    ctx.heapcache_setfield_cached(obj, intval_idx, value);
    obj
}

/// Emit an overflow-checked binary int operation.
///
/// Auto-generated: unbox a, unbox b, emit ovf op, guard no overflow, box result.
pub fn trace_int_binop_ovf(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    int_type_addr: i64,
    ob_type_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
    size_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    let a_val = trace_unbox_int(
        ctx,
        a,
        int_type_addr,
        ob_type_descr.clone(),
        intval_descr.clone(),
    );
    let b_val = trace_unbox_int(
        ctx,
        b,
        int_type_addr,
        ob_type_descr.clone(),
        intval_descr.clone(),
    );
    let result = ctx.record_op(opcode, &[a_val, b_val]);
    // Box(value) parity: derive the concrete result from the operands'
    // stamped Box.value carriers (BoxInt(value) — wrap semantics match
    // backend execution for the no-overflow branch; overflow branch
    // exits via the GuardNoOverflow below so stamped value is unused).
    if let (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) =
        (ctx.box_value(a_val), ctx.box_value(b_val))
    {
        let folded = crate::eval_binop_i(opcode, la, rb);
        ctx.set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    // No production caller of this AST→trace helper: pyre-jit-trace
    // routes overflow guards through the retired int fast path's
    // `frame.generate_guard` instead.  The bare
    // `record_guard_typed` keeps the guard shape correct in unit tests
    // (`trace_verify.rs`) without a synthetic snapshot — convergence
    // path is convergence to register-machine jitcode.
    ctx.record_guard_typed(OpCode::GuardNoOverflow, &[], Vec::new());
    trace_box_int(
        ctx,
        result,
        size_descr,
        ob_type_descr,
        intval_descr,
        int_type_addr,
    )
}

/// Emit a non-overflow binary int operation (bitwise ops, shifts).
pub fn trace_int_binop(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    int_type_addr: i64,
    ob_type_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
    size_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    let a_val = trace_unbox_int(
        ctx,
        a,
        int_type_addr,
        ob_type_descr.clone(),
        intval_descr.clone(),
    );
    let b_val = trace_unbox_int(
        ctx,
        b,
        int_type_addr,
        ob_type_descr.clone(),
        intval_descr.clone(),
    );
    let result = ctx.record_op(opcode, &[a_val, b_val]);
    // Box(value) parity: derive the concrete result from the operands'
    // stamped Box.value carriers.
    if let (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) =
        (ctx.box_value(a_val), ctx.box_value(b_val))
    {
        let folded = crate::eval_binop_i(opcode, la, rb);
        ctx.set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    trace_box_int(
        ctx,
        result,
        size_descr,
        ob_type_descr,
        intval_descr,
        int_type_addr,
    )
}

/// Emit a comparison between two Python ints.
pub fn trace_int_compare(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    int_type_addr: i64,
    ob_type_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    let a_val = trace_unbox_int(
        ctx,
        a,
        int_type_addr,
        ob_type_descr.clone(),
        intval_descr.clone(),
    );
    let b_val = trace_unbox_int(
        ctx,
        b,
        int_type_addr,
        ob_type_descr.clone(),
        intval_descr.clone(),
    );
    let result = ctx.record_op(opcode, &[a_val, b_val]);
    // Box(value) parity: stamp the bool result from the operands' Box.value
    // carriers (BoxInt(0|1) — IntEq/IntNe/IntLt/IntLe/IntGt/IntGe all map
    // through eval_binop_i).
    if let (Some(majit_ir::Value::Int(la)), Some(majit_ir::Value::Int(rb))) =
        (ctx.box_value(a_val), ctx.box_value(b_val))
    {
        let folded = crate::eval_binop_i(opcode, la, rb);
        ctx.set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    result
}

/// Unbox a Python float object: emit GuardClass + GetfieldGc(F|PureF).
///
/// Returns the raw f64 OpRef.
fn getfield_gc_f_pureornot(
    ctx: &mut crate::TraceCtx,
    obj: majit_ir::OpRef,
    descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    use majit_ir::Value;
    let field_index = descr.index();
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:941-945 cache-hit sanity check (float arm).
        // ConstFloat.same_constant compares via longlong.extract_bits
        // (history.py:283-294); pyre's Value Eq for Float uses
        // to_bits — bit-identical, NaN==NaN, 0.0!=-0.0.  The cached
        // Box's intrinsic value is fetched via `box_value(cached)` —
        // covering const pool, standard-virtualizable shadow, and
        // the frontend object's `value` field (RPython
        // `currfieldbox.getfloat_storage()` dispatch parity).
        let expected_float = match ctx.box_value(cached) {
            Some(Value::Float(f)) => Some(f),
            _ => None,
        };
        if let Some(cached_float) = expected_float {
            if let Some(Value::Ref(struct_ref)) = ctx.box_value(obj) {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    if let Some(Value::Float(loaded)) =
                        ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Float)
                    {
                        assert_eq!(
                            loaded.to_bits(),
                            cached_float.to_bits(),
                            "_opimpl_getfield_gc_any_pureornot sanity \
                             check (float): loaded {loaded} != cached \
                             {cached_float} (field_index={field_index}, \
                             struct_ptr={struct_ptr:#x})"
                        );
                    }
                }
            }
        }
        // pyjitpl.py:946 profiler.count_ops(rop.GETFIELD_GC_I,
        // Counters.HEAPCACHED_OPS) — the opnum literal in upstream is
        // GETFIELD_GC_I regardless of type, so wire the float variant
        // to the same counter bucket.
        ctx.profiler()
            .count_ops(OpCode::GetfieldGcI, crate::counters::HEAPCACHED_OPS);
        return cached;
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureF
    } else {
        OpCode::GetfieldGcF
    };
    let result = ctx.record_op_with_descr(opcode, &[obj], descr.clone());
    // Pair the recorded opref with the live float payload — RPython's
    // executor returns a BoxFloat with both identity and value.
    let live_value = if let Some(Value::Ref(struct_ref)) = ctx.box_value(obj) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Float)
        } else {
            None
        }
    } else {
        None
    };
    if let Some(live_value) = live_value {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

pub fn trace_unbox_float(
    ctx: &mut crate::TraceCtx,
    obj: majit_ir::OpRef,
    float_type_addr: i64,
    _ob_type_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    if !obj.is_constant() && !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(float_type_addr);
        ctx.record_guard_typed(OpCode::GuardClass, &[obj, type_const], Vec::new());
        ctx.heap_cache_mut().class_now_known(obj, float_type_addr);
    }
    getfield_gc_f_pureornot(ctx, obj, floatval_descr)
}

/// Box a raw f64 into a Python float object: emit New + SetfieldGc.
pub fn trace_box_float(
    ctx: &mut crate::TraceCtx,
    value: majit_ir::OpRef,
    size_descr: majit_ir::DescrRef,
    _ob_type_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
    _float_type_addr: i64,
) -> majit_ir::OpRef {
    use majit_ir::OpCode;
    // RPython parity: NEW_WITH_VTABLE + jtransform.py typeptr filter +
    // rewrite.py GC rewriter fielddescr_vtable emission.
    let obj = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(obj);
    let floatval_idx = floatval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[obj, value], floatval_descr);
    // `upd.setfield(valuebox)` parity — the cache stores the Box
    // identity (`value` OpRef); cache-hit readers fetch the intrinsic
    // value via `box_value(cached)` at hit time.
    ctx.heapcache_setfield_cached(obj, floatval_idx, value);
    obj
}

/// Emit a binary float operation: unbox a, unbox b, emit float op, box result.
pub fn trace_float_binop(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    float_type_addr: i64,
    ob_type_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
    size_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    let a_val = trace_unbox_float(
        ctx,
        a,
        float_type_addr,
        ob_type_descr.clone(),
        floatval_descr.clone(),
    );
    let b_val = trace_unbox_float(
        ctx,
        b,
        float_type_addr,
        ob_type_descr.clone(),
        floatval_descr.clone(),
    );
    let result = ctx.record_op(opcode, &[a_val, b_val]);
    // Box(value) parity: derive the concrete result from the operands'
    // stamped Box.value carriers (BoxFloat(value)).
    if let (Some(majit_ir::Value::Float(a)), Some(majit_ir::Value::Float(b))) =
        (ctx.box_value(a_val), ctx.box_value(b_val))
    {
        let bits = crate::eval_binop_f(opcode, a.to_bits() as i64, b.to_bits() as i64);
        ctx.set_opref_concrete(result, majit_ir::Value::Float(f64::from_bits(bits as u64)));
    }
    trace_box_float(
        ctx,
        result,
        size_descr,
        ob_type_descr,
        floatval_descr,
        float_type_addr,
    )
}

/// Emit a comparison between two Python floats.
pub fn trace_float_compare(
    ctx: &mut crate::TraceCtx,
    a: majit_ir::OpRef,
    b: majit_ir::OpRef,
    opcode: majit_ir::OpCode,
    float_type_addr: i64,
    ob_type_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
) -> majit_ir::OpRef {
    let a_val = trace_unbox_float(
        ctx,
        a,
        float_type_addr,
        ob_type_descr.clone(),
        floatval_descr.clone(),
    );
    let b_val = trace_unbox_float(
        ctx,
        b,
        float_type_addr,
        ob_type_descr.clone(),
        floatval_descr.clone(),
    );
    let result = ctx.record_op(opcode, &[a_val, b_val]);
    // Box(value) parity: stamp the bool result from the operands' Box.value
    // carriers (BoxInt(0|1) — FloatLt/FloatLe/FloatEq/FloatNe/FloatGt/FloatGe
    // route through eval_float_cmp).
    if let (Some(majit_ir::Value::Float(a)), Some(majit_ir::Value::Float(b))) =
        (ctx.box_value(a_val), ctx.box_value(b_val))
    {
        let folded = crate::eval_float_cmp(opcode, a.to_bits() as i64, b.to_bits() as i64);
        ctx.set_opref_concrete(result, majit_ir::Value::Int(folded));
    }
    result
}
