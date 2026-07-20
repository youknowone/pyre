//! Heapcache-aware field and array-item recording.
//!
//! **Parity:** trace-side counterpart of `pyjitpl.py`'s field / array
//! `_opimpl_*` consulting `heapcache.py` (implementation mirror
//! `majit-metainterp/heapcache.rs`).
//!
//! getfield / setfield / getarrayitem / setarrayitem recorded through
//! the heapcache (`pyjitpl.py` `_opimpl_*field*` / `_do_*arrayitem_gc`):
//! a cache hit returns the cached OpRef without emitting IR; a miss
//! records the op and writes the result back into the cache.

use super::*;

/// `getarrayitem_gc_<i|r|f>/rid>X` handler. Operand layout `rid>X`:
/// 1B r-reg(array) + 1B i-reg(index) + 2B descr + 1B X-dst.
///
/// RPython parity: `pyjitpl.py _do_getarrayitem_gc_any`:
///
///   tobox = heapcache.getarrayitem(arraybox, indexbox, arraydescr)
///   if tobox: return tobox        # cache hit, no IR (recording-only)
///   resop = self.execute_with_descr(op, arraydescr, arraybox, indexbox)
///   heapcache.getarrayitem_now_known(arraybox, indexbox, resop, arraydescr)
///   return resop
///
/// `opcode` is one of `GetarrayitemGc{I,R,F}`; `dst_bank` selects the
/// result bank (`'i'`/`'r'`/`'f'`) the walker writes back into.  The
/// index operand is always int-classified, so it is decoded from the
/// `i` register bank.
pub(crate) fn getarrayitem_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let array = read_ref_reg(code, op, 0, ctx)?;
    let index = read_int_reg(code, op, 1, ctx)?;
    let descr = read_descr(code, op, 2, ctx)?;
    let descr_index = descr.index();

    let result = if let Some(cached) =
        ctx.trace_ctx
            .heapcache_getarrayitem(array, index, descr_index)
    {
        // pyjitpl.py `_do_getarrayitem_gc_any` cache hit:
        //   tobox = heapcache.getarrayitem(...)
        //   if tobox:
        //       profiler.count_ops(rop.GETARRAYITEM_GC_I, HEAPCACHED_OPS)
        //       return tobox
        // RPython hardcodes `GETARRAYITEM_GC_I` regardless of the
        // recorded `typ` ('i' / 'r' / 'f'); pyre matches the hardcode
        // for profiling parity.
        ctx.trace_ctx.profiler().count_ops(
            OpCode::GetarrayitemGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        cached
    } else {
        let resbox = ctx
            .trace_ctx
            .record_op_with_descr(opcode, &[array, index], descr.clone());
        // Box.value parity: `box_value` exposes the resolution chain
        // PyPy reads off `arraybox.getref_base()` / `indexbox.getint()`
        // (`rpython/jit/metainterp/executor.py`).  Any operand
        // whose Box.value is known unblocks `array_sanity_load`, not
        // just Const-pool entries (`pyjitpl.py resbox =
        // execute_with_descr(...); getarrayitem_now_known(...)`
        // parity).
        let load_type = match opcode {
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcPureI => Some(majit_ir::Type::Int),
            OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcPureR => Some(majit_ir::Type::Ref),
            OpCode::GetarrayitemGcF | OpCode::GetarrayitemGcPureF => Some(majit_ir::Type::Float),
            _ => None,
        };
        let live_value = if let (
            Some(ty),
            Some(majit_ir::Value::Ref(array_ref)),
            Some(majit_ir::Value::Int(index_value)),
        ) = (
            load_type,
            ctx.trace_ctx.box_value(array),
            ctx.trace_ctx.box_value(index),
        ) {
            let array_ptr = array_ref.0 as i64;
            if array_ptr != usize::MAX as i64 && array_ptr != 0 {
                ctx.trace_ctx
                    .array_sanity_load(array_ptr, index_value, &descr, ty)
            } else {
                None
            }
        } else {
            None
        };
        // Stamp the loaded value as Box.value of the recorded result
        // (RPython `Box(value)` constructor analog) so subsequent
        // consumers see the runtime concrete instead of the
        // GcRef(usize::MAX) sentinel.
        if let Some(live_value) = live_value {
            ctx.trace_ctx.set_opref_concrete(resbox, live_value);
        }
        ctx.trace_ctx
            .heapcache_getarrayitem_now_known(array, index, descr_index, resbox);
        resbox
    };

    let dst = code[op.pc + 5] as usize;
    // concrete_of_opref derivation: derive shadow concrete from the recorded result's
    // `concrete_of_opref` entry instead of inventing Null.  Constant
    // arraybox + constant index hits land in `constants.get_value`;
    // virtualizable hits surface via `standard_virtualizable_box`;
    // `set_opref_concrete` stamps from upstream `binop_int_record`
    // flow back here too.  Null fallback preserves the prior contract.
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = result;
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `setarrayitem_gc_<i|r|f>/ri{i,r,f}d` handler. Operand layout per
/// `bhimpl_setarrayitem_gc_{i,r,f}(cpu, array, index, newvalue,
/// arraydescr)` (`blackhole.py`):
/// 1B r-reg(array) + 1B i-reg(index) + 1B {i,r,f}-reg(newvalue) + 2B descr.
///
/// RPython parity: `pyjitpl.py _opimpl_setarrayitem_gc_any`
/// dispatches through `metainterp.execute_setarrayitem_gc(arraydescr,
/// arraybox, indexbox, itembox)` — RPython's wrapper records
/// `rop.SETARRAYITEM_GC` and updates the heapcache via
/// `setarrayitem`.
///
/// No skip-on-redundant short-circuit (matches RPython —
/// `_opimpl_setarrayitem_gc_any` has no `if cached == value: return`,
/// because `heapcache.setarrayitem` already handles aliasing
/// invalidation at the right granularity).
///
/// `value_bank` selects the newvalue register source: `'i'` /
/// `'r'` / `'f'`.
pub(crate) fn setarrayitem_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let array = read_ref_reg(code, op, 0, ctx)?;
    let index = read_int_reg(code, op, 1, ctx)?;
    // Operand layout `<r><i><v>d`: r-reg(array) + i-reg(index) + v(value)
    // + 2B descr-index.  For the `c`-coded short form
    // (`setarrayitem_gc_i/ricd`) the value byte is an inline signed
    // constant (`signedord`, `blackhole.py`) read as a `ConstInt`
    // box instead of an `i`-register slot, mirroring `setfield_gc_i/rcd`.
    let value = match value_bank {
        'i' => read_int_reg(code, op, 2, ctx)?,
        'r' => read_ref_reg(code, op, 2, ctx)?,
        'f' => read_float_reg(code, op, 2, ctx)?,
        'c' => OpRef::ConstInt(code[op.pc + 3] as i8 as i64),
        _ => unreachable!("value_bank must be 'i', 'r', 'f' or 'c'"),
    };
    let descr = read_descr(code, op, 3, ctx)?;
    let descr_index = descr.index();

    ctx.trace_ctx
        .record_op_with_descr(OpCode::SetarrayitemGc, &[array, index, value], descr);
    // `upd.setarrayitem(valuebox)` (heapcache.py) parity — the
    // cache stores the Box identity (`value` OpRef); cache-hit
    // readers fetch the intrinsic value via `box_value(cached)` at
    // hit time.
    ctx.trace_ctx
        .heapcache_setarrayitem(array, index, descr_index, value);
    walker_fill_materialized_array(ctx, array, index, value);
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Walker virtual-force fill — companion to the `NEW_ARRAY_CLEAR`
/// materialization in the `new_array_clear` handler (module-global
/// fresh-container off-by-one fix). When `array` is a still-unescaped block we
/// materialized to a concrete GC `ItemsBlock` at NEW_ARRAY_CLEAR, write the
/// concrete element into it so a later BUILD_LIST / BUILD_TUPLE residual reads
/// a complete block during the walk. If the element value (not a ref) or the
/// index has no known concrete, the block cannot be completed, so revert the
/// array to the no-concrete sentinel (`Ref(usize::MAX)`): the residual then
/// declines and the void store aborts, exactly as without materialization.
///
/// A real (already-escaped) array store — whose `array` operand is a
/// `GetfieldGcR` load of a live container's items block, not a fresh
/// allocation — is left untouched because `is_unescaped(array)` is false, so
/// the runtime trace's own SETARRAYITEM is never duplicated eagerly here.
pub(crate) fn walker_fill_materialized_array<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    // heapcache.py is_unescaped — only a fresh, not-yet-escaped allocation
    // is a materialization candidate; a loaded (escaped) array is a real store.
    if !ctx.trace_ctx.heap_cache().is_unescaped(array) {
        return;
    }
    let block = match ctx.trace_ctx.box_value(array) {
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE && r.as_usize() != 0 => {
            r.as_usize() as *mut pyre_object::object_array::ItemsBlock
        }
        // No stamped concrete → not materialized (gate off / non-ref / non-const).
        _ => return,
    };
    // Confirm it is one of our GC-managed materialization blocks.
    if !pyre_object::gc_hook::try_gc_owns_object(block as *mut u8) {
        return;
    }
    let idx = match ctx.trace_ctx.box_value(index) {
        Some(majit_ir::Value::Int(i)) if i >= 0 => i as usize,
        _ => {
            ctx.trace_ctx
                .try_set_opref_concrete(array, majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE));
            return;
        }
    };
    let cap = unsafe { pyre_object::object_array::items_block_capacity(block) };
    if idx >= cap {
        ctx.trace_ctx
            .try_set_opref_concrete(array, majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE));
        return;
    }
    let elem = match ctx.trace_ctx.box_value(value) {
        Some(majit_ir::Value::Ref(r)) if r != majit_ir::GcRef::NO_CONCRETE => {
            r.as_usize() as pyre_object::PyObjectRef
        }
        _ => {
            ctx.trace_ctx
                .try_set_opref_concrete(array, majit_ir::Value::Ref(majit_ir::GcRef::NO_CONCRETE));
            return;
        }
    };
    unsafe {
        let base = pyre_object::object_array::items_block_items_base(block);
        *base.add(idx) = elem;
    }
    // Old→young barrier: the materialization block may have been promoted to
    // old-gen while `elem` is still young (the construction-barrier gap). A
    // nursery block carries no TRACK_YOUNG_PTRS so the barrier is a no-op.
    pyre_object::gc_hook::try_gc_write_barrier(block as *mut u8);
}

/// `setfield_gc_<i|r>/<rid|rrd>` handler: read box (r-reg), valuebox
/// (i or r reg per `value_bank`), descr operand, then either skip
/// the IR emission (cache says the same value is already there) or
/// record `OpCode::SetfieldGc` and write through to the heapcache.
///
/// RPython parity: `pyjitpl.py _opimpl_setfield_gc_any`:
///
///   upd = heapcache.get_field_updater(box, fielddescr)
///   if upd.currfieldbox is valuebox:
///       return                       # cache hit, no IR
///   self.metainterp.execute_and_record(rop.SETFIELD_GC, fielddescr,
///                                       box, valuebox)
///   upd.setfield(valuebox)
///
/// **Alias-clearing writeback**: goes through
/// `HeapCache::setfield_cached` instead of `getfield_now_known`. The
/// difference is the alias-clearing semantic that RPython's
/// `FieldUpdater.setfield()` carries (heapcache.py routes to
/// `CacheEntry.do_write_with_aliasing`):
///
///   `_clear_cache_on_write(seen_alloc)` (heapcache.py) wipes
///   `cache_anything` unconditionally and additionally wipes
///   `cache_seen_allocation` when the write target itself is not
///   seen-allocated.  This conservatively kills any cached entry whose
///   source-box might alias the SETFIELD target.
///
/// `getfield_now_known` only inserts the new (obj, field, value) tuple
/// — it does NOT clear sibling entries.  Using it here meant a
/// subsequent `getfield_gc(other_obj, same_field)` could return a
/// stale value cached from before the SETFIELD.  Switching to
/// `setfield_cached` matches `do_write_with_aliasing` exactly.
///
/// `value_bank` selects the valuebox source: `'i'` reads
/// `registers_i[v]`, `'r'` reads `registers_r[v]`.
pub(crate) fn setfield_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Operand layout `<r><v>d`: 1B r-reg(box) + 1B v(value) + 2B descr-index.
    // For the `c`-coded short form (`setfield_gc_i/rcd`) the value byte is
    // an inline signed constant (`signedord`, `blackhole.py`) read as a
    // `ConstInt` box instead of an `i`-register slot; obj and descr keep the
    // `rid` byte positions.
    let obj = read_ref_reg(code, op, 0, ctx)?;
    let valuebox = match value_bank {
        'i' => read_int_reg(code, op, 1, ctx)?,
        'r' => read_ref_reg(code, op, 1, ctx)?,
        'f' => read_float_reg(code, op, 1, ctx)?,
        'c' => OpRef::ConstInt(code[op.pc + 2] as i8 as i64),
        _ => unreachable!("value_bank must be 'i', 'r', 'f' or 'c'"),
    };
    let descr = read_descr(code, op, 2, ctx)?;
    let descr_index = descr.index();

    // Cache hit: if the heapcache already records `valuebox` as the
    // current value of `(obj, descr)`, the SETFIELD_GC is redundant —
    // skip recording. RPython pyjitpl.py _opimpl_setfield_gc_any:
    //   if upd.currfieldbox is valuebox:
    //       self.metainterp.staticdata.profiler.count_ops(rop.SETFIELD_GC, Counters.HEAPCACHED_OPS)
    //       return
    let is_redundant = ctx
        .trace_ctx
        .heapcache_getfield_cached(obj, descr_index)
        .map(|b| b)
        == Some(valuebox);
    if is_redundant {
        ctx.trace_ctx.profiler().count_ops(
            OpCode::SetfieldGc,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
    } else {
        ctx.trace_ctx
            .record_op_with_descr(OpCode::SetfieldGc, &[obj, valuebox], descr);
        // Write-through with alias-clearing semantics
        // (`heapcache.py do_write_with_aliasing`).  Mirrors
        // `upd.setfield(valuebox)` (heapcache.py).
        ctx.trace_ctx
            .heapcache_setfield_cached(obj, descr_index, valuebox);
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `getfield_gc_<i|r>/rd>X` handler: read a Ref-bank source register
/// + descr operand, consult the heapcache, and either return the
/// cached field box (no IR op recorded) or record the appropriate
/// `OpCode::GetfieldGc<I|R>` op and update the cache.
///
/// RPython parity: `opimpl_getfield_gc_<i|r>` →
/// `_opimpl_getfield_gc_any_pureornot` (`pyjitpl.py`).
/// RPython has a ConstPtr+is_always_pure() fast path
/// that fires `executor.execute(cpu, metainterp, opnum, fielddescr,
/// box)` and returns `ConstInt/ConstFloat/ConstPtr(resvalue)` —
/// recording NO trace op (the value is directly substituted as a Const
/// literal). The walker's `executor.execute` counterpart is
/// `field_sanity_load`, so the fast path is implemented: a constant
/// source register through an always-pure descr folds to the loaded
/// value as a Const literal with no recorded op.
///
/// Walker behaviour mirrors `_opimpl_getfield_gc_any_pureornot`
/// uniformly: heapcache hit returns the cached box (no IR op);
/// heapcache miss records `GetfieldGc<I|R>` (non-pure variant) +
/// writes through. The optimizer's always-pure pass later folds the
/// non-pure read into `GetfieldGcPure*` based on `descr.is_always_pure()`,
/// which is `OpHelpers.getfield_pure_for_descr` (resoperation.py)
/// parity. Walker emitting Pure variants directly would be
/// a TODO since RPython's opimpl_* never emits the Pure
/// opcodes; they're an optimizer-rewrite artifact.
///
/// `dst_bank` selects the result bank: `'i'` writes `registers_i[dst]`,
/// `'r'` writes `registers_r[dst]`.
pub(crate) fn getfield_gc_via_heapcache<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    opcode: OpCode,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Operand layout `rd>X`: 1B r-reg + 2B descr-index + 1B dst.
    let obj = read_ref_reg(code, op, 0, ctx)?;
    let descr = read_descr(code, op, 1, ctx)?;
    let descr_index = descr.index();

    // ConstPtr + always-pure fast path (pyjitpl.py): a constant
    // source through an immutable descr loads the field now and
    // substitutes the value as a Const literal, recording no op.
    let const_pure_result = if obj.is_constant() && descr.is_always_pure() {
        let load_type = match opcode {
            OpCode::GetfieldGcI | OpCode::GetfieldGcPureI => Some(majit_ir::Type::Int),
            OpCode::GetfieldGcR | OpCode::GetfieldGcPureR => Some(majit_ir::Type::Ref),
            OpCode::GetfieldGcF | OpCode::GetfieldGcPureF => Some(majit_ir::Type::Float),
            _ => None,
        };
        let struct_ptr = match ctx.trace_ctx.box_value(obj) {
            Some(majit_ir::Value::Ref(struct_ref)) => {
                let p = struct_ref.0 as i64;
                (p != 0 && p != usize::MAX as i64).then_some(p)
            }
            _ => None,
        };
        match (load_type, struct_ptr) {
            (Some(ty), Some(p)) => {
                ctx.trace_ctx
                    .field_sanity_load(p, &descr, ty)
                    .map(|v| match v {
                        majit_ir::Value::Int(n) => ctx.trace_ctx.const_int(n),
                        majit_ir::Value::Ref(r) => ctx.trace_ctx.const_ref(r.0 as i64),
                        majit_ir::Value::Float(f) => ctx.trace_ctx.const_float(f.to_bits() as i64),
                        _ => unreachable!("field_sanity_load returns Int/Ref/Float only"),
                    })
            }
            _ => None,
        }
    } else {
        None
    };

    // heaptracker.py special-cases the `typeptr` field: once a GUARD_CLASS
    // has pinned an object's class, reading its typeptr yields the known
    // class constant.  Inside an inline sub-walk the receiver's concrete
    // pointer often lives only in the register shadow (not the box value), so
    // the const-pure path above misses; fold the typeptr read straight from
    // the heapcache's known class instead.  This lets inlined type predicates
    // (`is_int`/`is_bool`, which read the typeptr and compare it against a
    // type address) fold during the walk.
    let is_typeptr_field = descr
        .as_field_descr()
        .is_some_and(|fd| fd.offset() == pyre_object::pyobject::OB_TYPE_OFFSET);
    let typeptr_const = if ctx.fbw_mode.inline_subwalk && !obj.is_constant() && is_typeptr_field {
        let known = ctx.trace_ctx.heap_cache().get_known_class(obj);
        match (known, opcode) {
            (Some(cls), OpCode::GetfieldGcI | OpCode::GetfieldGcPureI) => {
                Some(ctx.trace_ctx.const_int(cls))
            }
            (Some(cls), OpCode::GetfieldGcR | OpCode::GetfieldGcPureR) => {
                Some(ctx.trace_ctx.const_ref(cls))
            }
            _ => None,
        }
    } else {
        None
    };

    let result = if let Some(folded) = typeptr_const {
        folded
    } else if let Some(constant) = const_pure_result {
        constant
    } else if let Some(cached) = ctx.trace_ctx.heapcache_getfield_cached(obj, descr_index) {
        // Cache hit (RPython _opimpl_getfield_gc_any_pureornot):
        //   if upd.currfieldbox is not None:
        //       self.metainterp.staticdata.profiler.count_ops(rop.GETFIELD_GC_I, Counters.HEAPCACHED_OPS)
        //       return upd.currfieldbox
        // RPython hardcodes `GETFIELD_GC_I` for the count regardless of
        // the actual rop variant (`_i` / `_r` / `_f`); match the
        // hardcode for profiling parity.
        ctx.trace_ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        cached
    } else {
        // Recording a `getfield_gc_*` whose FieldDescr lacks a parent_descr
        // backreference would later crash the optimizer's
        // `ensure_ptr_info_arg0` (`optimizer.py`).  Inside a sub-walk
        // abort gracefully instead, so the trace falls back to the interpreter
        // rather than carrying an op the optimizer cannot lower.  The fold
        // paths above (typeptr / const-pure / cache-hit) record nothing, so
        // they are unaffected; production sub-walks never reach this (they
        // would already panic).
        if ctx.fbw_mode.inline_subwalk
            && matches!(
                opcode,
                OpCode::GetfieldGcI
                    | OpCode::GetfieldGcR
                    | OpCode::GetfieldGcF
                    | OpCode::GetfieldGcPureI
                    | OpCode::GetfieldGcPureR
                    | OpCode::GetfieldGcPureF
            )
            && descr
                .as_field_descr()
                .is_some_and(|fd| fd.get_parent_descr().is_none())
        {
            return Err(DispatchError::FieldDescrMissingParentDescr { pc: op.pc });
        }
        // Cache miss — record op + write through.  `box_value`
        // resolves the Box.value chain PyPy reads off
        // `box.getref_base()` in `executor.do_getfield_gc_*`
        // (`executor.py`); the sanity load fires whenever the
        // struct pointer is known (Const, vable shadow, or stamped),
        // mirroring `pyjitpl.py resbox = execute_with_descr(...);
        // upd.getfield_now_known(resbox)`.
        let resbox = ctx
            .trace_ctx
            .record_op_with_descr(opcode, &[obj], descr.clone());
        let load_type = match opcode {
            OpCode::GetfieldGcI | OpCode::GetfieldGcPureI => Some(majit_ir::Type::Int),
            OpCode::GetfieldGcR | OpCode::GetfieldGcPureR => Some(majit_ir::Type::Ref),
            OpCode::GetfieldGcF | OpCode::GetfieldGcPureF => Some(majit_ir::Type::Float),
            _ => None,
        };
        let live_value = if let (Some(ty), Some(majit_ir::Value::Ref(struct_ref))) =
            (load_type, ctx.trace_ctx.box_value(obj))
        {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                ctx.trace_ctx.field_sanity_load(struct_ptr, &descr, ty)
            } else {
                None
            }
        } else {
            None
        };
        // Stamp the loaded value as the Box.value of the recorded
        // result so subsequent reads (cache hits + non-Const
        // `concrete_of_opref` consumers) see the real runtime
        // concrete instead of the GcRef(usize::MAX) sentinel.
        if let Some(live_value) = live_value {
            ctx.trace_ctx.set_opref_concrete(resbox, live_value);
        }
        ctx.trace_ctx
            .heapcache_getfield_now_known(obj, descr_index, resbox);
        resbox
    };

    let dst = code[op.pc + 4] as usize;
    // concrete_of_opref derivation: derive shadow concrete via `concrete_of_opref` so a
    // constant-folded predecessor (e.g. `binop_int_record` having
    // stamped this OpRef in OpRef concrete stamping) propagates through.  RPython
    // `Box.value` parity: `pyjitpl.py:executor.py` per-opcode LLOp
    // stamps `box.value` post-exec; pyre's `concrete_of_opref` reads
    // that channel.  Null fallback preserves the prior unknown-result
    // behaviour for cache-miss recorded ops.
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'f' => {
            let len = ctx.registers_f.len();
            let slot = ctx
                .registers_f
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "f",
                })?;
            *slot = result;
        }
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `virtualizable_gen.rs` pyre PyFrame static-field order
/// `[last_instr, pycode, valuestackdepth, debugdata, lastblock, w_globals]`.
pub(crate) const VABLE_CODE_FIELD_IDX: usize = 1;
pub(crate) const VABLE_NAMESPACE_FIELD_IDX: usize = 5;
