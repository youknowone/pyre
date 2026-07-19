//! Virtualizable field / array access via the metainterp.
//!
//! getfield / setfield / getarrayitem / setarrayitem / arraylen against
//! a virtualizable, routed through the metainterp vable bookkeeping.

use super::*;

/// `getfield_vable_<i|r|f>/rd>X` handler. Operand layout `rd>X`:
/// 1B r-reg(vable_box) + 2B descr(field) + 1B X-dst.
///
/// RPython parity: `pyjitpl.py:1167-1186 opimpl_getfield_vable_{i,r,f}`:
///
///   def opimpl_getfield_vable_i(self, box, fielddescr, pc):
///       if self._nonstandard_virtualizable(pc, box, fielddescr):
///           return self.opimpl_getfield_gc_i(box, fielddescr)
///       self.metainterp.check_synchronized_virtualizable()
///       index = self._get_virtualizable_field_index(fielddescr)
///       return self.metainterp.virtualizable_boxes[index]
///
/// The walker delegates to the orthodox `TraceCtx::vable_getfield_{int,
/// ref,float}` ports (`majit-metainterp/src/trace_ctx.rs:1715, 1801,
/// 1839`) which already implement the full
/// `_nonstandard_virtualizable` check + heapcache-aware GETFIELD_GC
/// fallback + `virtualizable_boxes[index]` cache read.  Only the OpRef
/// component of the `(OpRef, Value)` tuple is meaningful here, since
/// register banks carry only OpRefs — the concrete `Value` is tracked
/// via the per-step concrete frame snapshot.  `dst_bank` selects the result bank
/// (`'i'`/`'r'`/`'f'`) the walker writes back into, mirroring
/// `getfield_gc_via_heapcache`'s shape.
pub(crate) fn getfield_vable_via_metainterp(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let obj = read_ref_reg(code, op, 0, ctx)?;
    // #68/gap-10: inside an inlined-callee sub-walk, a scalar getfield_vable_r
    // of the callee's namespace(idx5)/pycode(idx1) must resolve to the callee's
    // compile-time `InlineCalleeConsts` whether the callee frame is unseeded
    // (strict path, handled by the `obj.is_none()` branch below) OR a seeded
    // virtual frame (multiframe path). On the multiframe path the read hits the
    // seeded frame box but the heapcache forward MISSES — the codewriter's
    // per-fn vable descr identity differs from the seeding descr — so it would
    // record a non-const `GetfieldGcR`, leaving the LOAD_GLOBAL fold's namespace
    // operand non-concrete (the residual that the nested-unjournaled abort
    // declines, e.g. nbody `advance()`'s `len`). The consts route is
    // descr-identity-independent (`const_ref`) and only intercepts the two
    // static Ref fields (`try_resolve_inline_callee_static_field` returns `None`
    // for everything else), so it is safe to consult on the seeded path too.
    if fbw_inline_nsfold_enabled() {
        if let Some(resolved) = try_resolve_inline_callee_static_field(code, op, ctx, dst_bank)? {
            return Ok(resolved);
        }
    }
    // RPython's `box` is always a live virtualizable-frame box. An
    // unseeded walker Ref register holds `OpRef::None` (`raw() ==
    // u32::MAX`); feeding it into the metainterp vable path would resize
    // the heapcache flag vector to 16 GiB. Bail to a trace abort instead.
    if obj.is_none() {
        // Path-1 (#68): an inlined callee reading a scalar field off its
        // OWN unseeded portal frame.  A resolvable static field (`w_globals`
        // namespace for a LOAD_GLOBAL, `pycode` promote-to-const) folds to
        // the callee constant; anything else is declined up-front by
        // `callee_fast_path_inlinable`, so an unresolved field here is a
        // genuine unseeded-box error.
        if let Some(resolved) = try_resolve_inline_callee_static_field(code, op, ctx, dst_bank)? {
            return Ok(resolved);
        }
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let descr = read_descr(code, op, 1, ctx)?;

    // R7 parity: RPython `opimpl_getfield_vable_{i,r,f}(box, fielddescr,
    // pc)` threads orgpc through `_nonstandard_virtualizable(pc, ...)`
    // (pyjitpl.py:1167-1186 + :1137).  Pyre's walker has the matching
    // JitCode PC in `op.pc`; pass it through so the helper signature
    // stays line-by-line equivalent even if `is_nonstandard_virtualizable`
    // currently ignores the pc at the leaf (`trace_ctx.rs let _ = pc;`).
    let pc = op.pc;
    // Concrete struct pointer for pyjitpl.py:934-945 cache-hit sanity
    // check.  The walker keeps a parallel concrete Ref-bank shadow;
    // thread the same live pointer that RPython's `box.getref_base()`
    // would expose to `executor.execute(...)`.
    let vable_struct_ptr = match read_ref_reg_concrete(code, op, 0, ctx) {
        ConcreteValue::Ref(ptr) => ptr as i64,
        ConcreteValue::Null => 0,
        ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => 0,
    };
    let guards_before = ctx.trace_ctx.num_guards();
    let (result, shadow_value) = match dst_bank {
        'i' => ctx
            .trace_ctx
            .vable_getfield_int(pc, obj, vable_struct_ptr, descr),
        'r' => ctx
            .trace_ctx
            .vable_getfield_ref(pc, obj, vable_struct_ptr, descr),
        'f' => ctx
            .trace_ctx
            .vable_getfield_float(pc, obj, vable_struct_ptr, descr),
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    };
    walker_capture_inline_nonstandard_vable_guard(ctx, op.pc, guards_before)?;
    // RPython `opimpl_getfield_vable_{i,r,f}` returns
    // `virtualizable_boxes[index]` (`pyjitpl.py:1186`) — a Box whose
    // `_resint`/`_resref`/`_resfloat` is filled at construction time.
    // `box.getint()` returns the live value without any side-lookup.
    // Pyre splits OpRef↔concrete into a side table; mirror the Box.value
    // contract by stamping the read result's concrete into
    // `opref_concrete` so `concrete_of_opref(result)` honors the same
    // contract for downstream consumers (`goto_if_not/iL`,
    // `switch/id`, `int_*` arithmetic).  The non-standard heapcache
    // path inside `vable_getfield_int` already does the same stamp;
    // the standard path returns the cached `(opref, value)` pair
    // without stamping.  `None` means no live concrete is available
    // for this slot — skip to match the heapcache path's gating.
    if let Some(shadow_value) = shadow_value {
        ctx.trace_ctx.set_opref_concrete(result, shadow_value);
    }
    let dst = code[op.pc + 4] as usize;
    // concrete_of_opref derivation: derive shadow concrete via `concrete_of_opref`.  The
    // `vable_getfield_*` helpers in `TraceCtx` already populate the
    // concrete shadow for virtualizable-resident fields via the
    // `standard_virtualizable_box()`/`virtualizable_boxes` channel,
    // and feed `set_opref_concrete` on the GETFIELD_GC fallback for
    // non-vable structs — both surface through this lookup.
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => {
            write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
        }
        'r' => {
            write_vable_field_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
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

/// `setfield_vable_<i|r|f>/r<v>d` handler. Operand layout:
/// 1B r-reg(vable_box) + 1B <v>-reg(value) + 2B descr(field).
/// No dst byte (set, not get).
///
/// RPython parity: `pyjitpl.py:1188-1199 _opimpl_setfield_vable`:
///
///   def _opimpl_setfield_vable(self, box, valuebox, fielddescr, pc):
///       if self._nonstandard_virtualizable(pc, box, fielddescr):
///           return self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
///       index = self._get_virtualizable_field_index(fielddescr)
///       self.metainterp.virtualizable_boxes[index] = valuebox
///       self.metainterp.synchronize_virtualizable()
///       # XXX only the index'th field needs to be synchronized, really
///
/// The walker delegates to `TraceCtx::vable_setfield`
/// (`majit-metainterp/src/trace_ctx.rs:1759`) which implements the
/// full `_nonstandard_virtualizable` -> SETFIELD_GC fallback +
/// `virtualizable_boxes[index] = valuebox` write + `synchronize_virtualizable`
/// mirror.  The concrete `Value` is reconstructed via
/// `TraceCtx::concrete_of_opref` (matches the trait-leg's
/// `pyjitpl/dispatch.rs:1608-1609` shape `let (value, concrete) =
/// self.read_<bank>_reg(src); ctx.vable_setfield(...)`).
///
/// `value_bank` selects the value register bank (`'i'`/`'r'`/`'f'`),
/// mirroring `setfield_gc_via_heapcache`'s parameter shape.
pub(crate) fn setfield_vable_via_metainterp(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Strict fresh-frame fold: a scalar setfield to the current inline level's
    // own (unseeded) portal frame is a virtual-field write — `valuestackdepth`
    // / `last_instr` sync on a branchless leaf that resumes at the caller
    // boundary — so it folds to a no-op, emitting no SETFIELD_GC.  The
    // `fresh_virtualizable` OptVirtualize elision (jtransform.py:990-993).
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }
    let obj = read_ref_reg(code, op, 0, ctx)?;
    // Same unseeded-register guard as `getfield_vable_via_metainterp`:
    // a `None` box would resize the heapcache flag vector to 16 GiB.
    if obj.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let value = match value_bank {
        'i' => read_int_reg(code, op, 1, ctx)?,
        'r' => read_ref_reg(code, op, 1, ctx)?,
        'f' => read_float_reg(code, op, 1, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
    };
    let descr = read_descr(code, op, 2, ctx)?;
    let concrete = ctx.trace_ctx.concrete_of_opref(value);
    // R7 parity: pyjitpl.py:1188-1199 `_opimpl_setfield_vable(box,
    // valuebox, fielddescr, pc)` threads orgpc through
    // `_nonstandard_virtualizable(pc, ...)`; walker has `op.pc` for the
    // JitCode PC, pass through.
    let guards_before = ctx.trace_ctx.num_guards();
    ctx.trace_ctx
        .vable_setfield(op.pc, obj, descr, value, concrete);
    walker_capture_inline_nonstandard_vable_guard(ctx, op.pc, guards_before)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// Resolve the `(fdescr, adescr)` virtualizable-array descr pair from a
/// `(VableArray, Array)` jitcode descr-pool pair.
///
/// RPython parity: `MIFrame.vable_array_index_pair_at`
/// (`blackhole.rs:1613-1628`) reads the two 2-byte descr-pool indices,
/// asserts they resolve to `(BhDescr::VableArray { index }, BhDescr::Array
/// { .. })`, and yields the `index`. `vable_array_descrs`
/// (`pyjitpl/dispatch.rs:845-856`) then maps that index through the
/// jitdriver's `VirtualizableInfo` to the canonical `array_field_descrs`
/// /`array_descrs` pair.  The walker must use the *vinfo* descrs (not the
/// raw jitcode-pool descrs) because `TraceCtx::vable_array_flat_index`
/// keys on `array_field_by_descr` (descr identity, `virtualizable.rs:602`).
///
/// `field_offset` / `array_offset` are byte offsets (from `op.pc + 1`) of
/// the two descr operands, matching the per-op argcode layout.
pub(crate) fn vable_array_descrs_from_jitcode(
    code: &[u8],
    op: &DecodedOp,
    field_offset: usize,
    array_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<(DescrRef, DescrRef), DispatchError> {
    let read_pool_idx = |off: usize| {
        let lo = code[op.pc + 1 + off] as usize;
        let hi = code[op.pc + 1 + off + 1] as usize;
        lo | (hi << 8)
    };
    let field_idx = read_pool_idx(field_offset);
    let array_pool_idx = read_pool_idx(array_offset);
    // RPython `MIFrame.vable_array_index_pair_at` reads `self.descrs[idx]`
    // — pyre's single per-walk pool, selected by `ctx.raw_descrs`
    // (global `ALL_DESCRS` for arm walks, per-`CodeObject` `exec.descrs`
    // for full-body walks).
    let array_field_index = match (
        ctx.raw_descrs.bh_descr_at(field_idx),
        ctx.raw_descrs.bh_descr_at(array_pool_idx),
    ) {
        (
            Some(majit_translate::jitcode::BhDescr::VableArray { index }),
            Some(majit_translate::jitcode::BhDescr::Array { .. }),
        ) => *index,
        _ => {
            if std::env::var("PYRE_FBW_INLINE_DIAG").is_ok() {
                eprintln!(
                    "[vable-arr-malformed] pc={} pool_len={:?} field_idx={field_idx} \
                     field={:?} array_idx={array_pool_idx} array={:?}",
                    op.pc,
                    ctx.raw_descrs.len(),
                    ctx.raw_descrs.bh_descr_at(field_idx),
                    ctx.raw_descrs.bh_descr_at(array_pool_idx),
                );
            }
            return Err(DispatchError::VableArrayDescrMalformed {
                pc: op.pc,
                field_idx,
                array_idx: array_pool_idx,
            });
        }
    };
    let info = ctx
        .trace_ctx
        .virtualizable_info()
        .ok_or(DispatchError::VableArrayMissingVirtualizableInfo { pc: op.pc })?;
    let fdescr = info
        .array_field_descrs()
        .get(array_field_index)
        .cloned()
        .ok_or(DispatchError::VableArrayIndexOutOfRange {
            pc: op.pc,
            index: array_field_index,
        })?;
    let adescr = info.array_descrs.get(array_field_index).cloned().ok_or(
        DispatchError::VableArrayIndexOutOfRange {
            pc: op.pc,
            index: array_field_index,
        },
    )?;
    Ok((fdescr, adescr))
}

/// `getarrayitem_vable_<i|r|f>/ridd>X` handler. Operand layout `ridd>X`:
/// 1B r-reg(vable) + 1B i-reg(index) + 2B fdescr(VableArray) + 2B
/// adescr(Array) + 1B X-dst.
///
/// RPython parity: `pyjitpl.py:1218-1234 _opimpl_getarrayitem_vable`
/// (`opimpl_getarrayitem_vable_{i,r,f}`).  Delegates to
/// `TraceCtx::vable_getarrayitem_{int,ref,float}_indexed`
/// (`trace_ctx.rs:2982/3043/3099`) which implements the
/// `_nonstandard_virtualizable` GETFIELD_GC + GETARRAYITEM_GC fallback
/// and the standard-vable `virtualizable_boxes[index]` cache read.
/// Mirrors `getfield_vable_via_metainterp`'s concrete-stamp + dst-write
/// shape; the trait counterpart is `pyjitpl/dispatch.rs:1909-1977`.
pub(crate) fn getarrayitem_vable_via_metainterp(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Strict fresh-frame fold: when this op reads the current inline level's
    // own (unseeded) portal frame, resolve it register-to-register through the
    // per-slot OpRef shadow and emit NO GC op — the `fresh_virtualizable` case
    // (jtransform.py:990-993 `is_virtualizable_getset` returns `False`).  The
    // param-store seed dominates every read of a branchless leaf's own frame,
    // so the slot is present; a genuinely-unbound read falls through to the
    // `VableBoxNotSeeded` abort below (never the 16 GiB metainterp path).
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        let index = read_int_reg(code, op, 1, ctx)?;
        if let Some(majit_ir::Value::Int(slot)) = ctx.trace_ctx.concrete_of_opref(index) {
            if let Some(result) = ctx
                .callee_shadow
                .as_ref()
                .and_then(|shadow| shadow.opref.get(&slot).copied())
            {
                let dst = code[op.pc + 7] as usize;
                let concrete = concrete_from_recorded_opref(ctx, result);
                match dst_bank {
                    'i' => write_int_reg(ctx, op.pc, dst, result, concrete)?,
                    'r' => write_ref_reg(ctx, op.pc, dst, result, concrete)?,
                    'f' => {
                        let len = ctx.registers_f.len();
                        let slot_ref = ctx.registers_f.get_mut(dst).ok_or(
                            DispatchError::RegisterOutOfRange {
                                pc: op.pc,
                                reg: dst,
                                len,
                                bank: "f",
                            },
                        )?;
                        *slot_ref = result;
                    }
                    _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
                }
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
        }
    }
    let vable = read_ref_reg(code, op, 0, ctx)?;
    // An unseeded walker Ref register holds `OpRef::None` (`raw() ==
    // u32::MAX`); feeding it into the metainterp vable path would resize
    // the heapcache flag vector to 16 GiB. Bail to a trace abort, mirroring
    // the scalar `getfield_vable_via_metainterp` guard.
    if vable.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let index = read_int_reg(code, op, 1, ctx)?;
    // pyjitpl.py:1206 `indexbox.getint()` — the array slot is chosen from
    // the concrete index. Fail loud if the walker can't resolve it.
    let index_value = match ctx.trace_ctx.concrete_of_opref(index) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::VableArrayIndexNotConcrete {
                pc: op.pc,
                value: index,
            });
        }
    };
    let (fdescr, adescr) = vable_array_descrs_from_jitcode(code, op, 2, 4, ctx)?;
    let guards_before = ctx.trace_ctx.num_guards();
    let (result, shadow_value) = match dst_bank {
        'i' => ctx.trace_ctx.vable_getarrayitem_int_indexed(
            op.pc,
            vable,
            index,
            index_value,
            fdescr,
            adescr,
        ),
        'r' => ctx.trace_ctx.vable_getarrayitem_ref_indexed(
            op.pc,
            vable,
            index,
            index_value,
            fdescr,
            adescr,
        ),
        'f' => ctx.trace_ctx.vable_getarrayitem_float_indexed(
            op.pc,
            vable,
            index,
            index_value,
            fdescr,
            adescr,
        ),
        _ => unreachable!("dst_bank must be 'i', 'r' or 'f'"),
    };
    walker_capture_inline_nonstandard_vable_guard(ctx, op.pc, guards_before)?;
    let shadow_value = shadow_value.unwrap_or(Value::Void);
    // When the read missed every concrete channel (`Void`) but we are inside
    // an inlined callee, fall back to the per-frame concrete-locals shadow —
    // a loop-carried local read after a may-force op in the loop body has no
    // heapcache entry this pass, yet its recording-time concrete is known.
    let shadow_value = if matches!(shadow_value, Value::Void) {
        ctx.callee_shadow
            .as_ref()
            .and_then(|shadow| shadow.concrete.get(&index_value).copied())
            .filter(|entry| entry.frame_reg == code[op.pc + 1] as u16)
            .map(|entry| entry.value)
            .unwrap_or(Value::Void)
    } else {
        shadow_value
    };
    // Mirror `getfield_vable_via_metainterp`: stamp the read result's
    // concrete so `concrete_of_opref(result)` honors the Box.value
    // contract for downstream consumers; `Value::Void` = no live concrete.
    //
    // An indexed read hands back the STORED box, so the stamp is a write onto a
    // box that may already carry a live concrete. A frame whose array slot is
    // still an unmaterialized hole reads NULL while the real Ref lives in the
    // guard's register file, so stamping that NULL would clobber the live value
    // (last write wins) and fold a downstream may-force residual's Ref arg to
    // NULL -> `MayForceNullRefArgUnsupported`. Skip ONLY that clobber: a NULL
    // read onto a box with no live Ref still stamps, keeping NULL a real
    // concrete for the slots that genuinely hold one.
    let clobbers_live_ref = matches!(shadow_value, Value::Ref(majit_ir::GcRef(0)))
        && matches!(
            ctx.trace_ctx.box_value(result),
            Some(Value::Ref(majit_ir::GcRef(addr))) if addr != 0
        );
    if !matches!(shadow_value, Value::Void) && !clobbers_live_ref {
        ctx.trace_ctx.set_opref_concrete(result, shadow_value);
    }
    let dst = code[op.pc + 7] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    match dst_bank {
        'i' => write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?,
        'r' => write_ref_reg(ctx, op.pc, dst, result, concrete_for_shadow)?,
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

/// `setarrayitem_vable_<i|r|f>/riXdd` handler. Operand layout `riXdd`:
/// 1B r-reg(vable) + 1B i-reg(index) + 1B X-reg(value) + 2B
/// fdescr(VableArray) + 2B adescr(Array). No dst byte.
///
/// RPython parity: `pyjitpl.py:1236-1247 _opimpl_setarrayitem_vable`.
/// Delegates to `TraceCtx::vable_setarrayitem_indexed`
/// (`trace_ctx.rs:3153`) which implements the `_nonstandard_virtualizable`
/// SETARRAYITEM_GC fallback + the standard-vable
/// `virtualizable_boxes[index] = valuebox` + `synchronize_virtualizable`.
/// Trait counterpart: `pyjitpl/dispatch.rs:1978-2052`.
pub(crate) fn setarrayitem_vable_via_metainterp(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
    value_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    // Strict fresh-frame fold (twin of `getarrayitem_vable_via_metainterp`): a
    // write to the current inline level's own (unseeded) portal frame updates
    // the per-slot OpRef + concrete shadow and emits NO SETARRAYITEM_GC.  Both
    // local stores (`index < nlocals`) and operand-stack pushes (`index >=
    // nlocals`) on the fresh frame are pure mirror writes here — the value
    // lives in an SSA register; the vable array is folded away.
    let fold_frame_reg = fbw_strict_fold_frame_reg(ctx);
    if fold_frame_reg != u16::MAX && code[op.pc + 1] as u16 == fold_frame_reg {
        let index = read_int_reg(code, op, 1, ctx)?;
        if let Some(majit_ir::Value::Int(slot)) = ctx.trace_ctx.concrete_of_opref(index) {
            let value = match value_bank {
                'i' => read_int_reg(code, op, 2, ctx)?,
                'r' => read_ref_reg(code, op, 2, ctx)?,
                'f' => read_float_reg(code, op, 2, ctx)?,
                _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
            };
            let concrete = ctx
                .trace_ctx
                .concrete_of_opref(value)
                .unwrap_or(majit_ir::Value::Void);
            if let Some(shadow) = ctx.callee_shadow.as_mut() {
                shadow.set_opref(slot, value);
                shadow.set_concrete(fold_frame_reg, slot, concrete);
            }
            return Ok((DispatchOutcome::Continue, op.next_pc));
        }
    }
    let vable = read_ref_reg(code, op, 0, ctx)?;
    // See `getarrayitem_vable_via_metainterp`: an unseeded `OpRef::None`
    // vable would resize the heapcache flag vector to 16 GiB; bail instead.
    if vable.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let index = read_int_reg(code, op, 1, ctx)?;
    let index_value = match ctx.trace_ctx.concrete_of_opref(index) {
        Some(Value::Int(v)) => v,
        _ => {
            return Err(DispatchError::VableArrayIndexNotConcrete {
                pc: op.pc,
                value: index,
            });
        }
    };
    let mut value = match value_bank {
        'i' => read_int_reg(code, op, 2, ctx)?,
        'r' => read_ref_reg(code, op, 2, ctx)?,
        'f' => read_float_reg(code, op, 2, ctx)?,
        _ => unreachable!("value_bank must be 'i', 'r' or 'f'"),
    };
    // A STORE_FAST (`index < nlocals`) pops the operand-stack TOS and writes
    // it into a local slot.  When the popped value lives in an operand-stack
    // temp that spans a nested loop, the loop-header merge-point seeds no
    // operand-stack color (`trace.rs` loop-header entry), so the codewriter's
    // value register reads back `OpRef::NONE` even though the box is live.
    // The walk-level operand-stack mirror (`vstack_boxes`, the analog of
    // `MIFrame.registers_r`) is the authoritative kept-stack source and was
    // maintained across the nested loop; recover the TOS box from it.  Writing
    // the unbound `NONE` into the vable array otherwise leaves an untyped slot
    // that fails the guard-snapshot buildability precondition
    // (`GuardSnapshotVableUntyped`).  Gate on a live mirror + a Ref store into
    // the local region so a genuine null / non-mirrored slot keeps the legacy
    // read.
    if value.is_none() && value_bank == 'r' && ctx.vstack_valid {
        let full_body_sym = ctx.fbw_mode.snapshot_sym;
        if !full_body_sym.is_null() {
            // SAFETY: pointer live for the full-body walk; read-only.
            let nlocals = unsafe { (*full_body_sym).nlocals as i64 };
            if index_value >= 0 && index_value < nlocals {
                if let Some(&tos) = ctx.vstack_boxes.last() {
                    if !tos.is_none() {
                        value = tos;
                    }
                }
            }
        }
    }
    let (fdescr, adescr) = vable_array_descrs_from_jitcode(code, op, 3, 5, ctx)?;
    let concrete = ctx
        .trace_ctx
        .concrete_of_opref(value)
        .unwrap_or(Value::Void);
    let guards_before = ctx.trace_ctx.num_guards();
    ctx.trace_ctx.vable_setarrayitem_indexed(
        op.pc,
        vable,
        index,
        index_value,
        fdescr,
        adescr,
        value,
        concrete,
    );
    // Keep the inline concrete-locals shadow current so a later read of this
    // slot (after a may-force op clears the heapcache) recovers the concrete.
    if let Some(shadow) = ctx.callee_shadow.as_mut() {
        shadow.set_concrete(code[op.pc + 1] as u16, index_value, concrete);
    }
    walker_capture_inline_nonstandard_vable_guard(ctx, op.pc, guards_before)?;
    // A Ref stored to the operand-stack region of the vable array is an
    // operand-stack push (`pyframe.pushvalue` lowers to
    // `setarrayitem_vable_r(locals_cells_stack_w, depth, w_obj)`). Retain the
    // last-write TOS candidate for ordinary single-result opcodes. Method-form
    // LOAD_ATTR is the one two-result shape that also needs its exact lower
    // slot mirrored: it writes `[method, self]`, and retaining only `self`
    // made boundary reconciliation refill the method hole from the stale
    // pre-LOAD_ATTR receiver. A later guard then resumed CALL with that
    // receiver as its callee. Scope the positional write to that opcode; the
    // general boundary model remains authoritative for other push/pop shapes.
    // Local/cell stores remain excluded by the `index >= nlocals` gate.
    if value_bank == 'r' && ctx.vstack_valid {
        let full_body_sym = ctx.fbw_mode.snapshot_sym;
        if !full_body_sym.is_null() {
            // SAFETY: pointer live for the full-body walk; read-only.
            let nlocals = unsafe { (*full_body_sym).nlocals } as i64;
            if index_value >= nlocals {
                let method_load = unsafe {
                    let jitcode = (*full_body_sym).jitcode;
                    if jitcode.is_null() {
                        false
                    } else {
                        let jitcode = &*jitcode;
                        if jitcode.payload.code_ptr.is_null() {
                            false
                        } else {
                            pyre_interpreter::decode_instruction_at(
                                &*jitcode.payload.code_ptr,
                                ctx.vstack_cur_pypc as usize,
                            )
                            .is_some_and(|(instr, op_arg)| match instr
                            {
                                pyre_interpreter::Instruction::LoadAttr { namei } => {
                                    namei.get(op_arg).is_method()
                                }
                                _ => false,
                            })
                        }
                    }
                };
                if method_load {
                    let stack_slot = (index_value - nlocals) as usize;
                    if ctx.vstack_boxes.len() <= stack_slot {
                        ctx.vstack_boxes.resize(stack_slot + 1, OpRef::NONE);
                    }
                    ctx.vstack_boxes[stack_slot] = value;
                }
                ctx.vstack_last_ref = value;
            }
        }
    }
    Ok((DispatchOutcome::Continue, op.next_pc))
}

/// `arraylen_vable/rdd>i` handler. Operand layout `rdd>i`: 1B r-reg(vable)
/// + 2B fdescr(VableArray) + 2B adescr(Array) + 1B i-dst.
///
/// RPython parity: `pyjitpl.py:1253-1263 opimpl_arraylen_vable`.
/// Delegates to `TraceCtx::vable_arraylen_vable` (`trace_ctx.rs:3195`)
/// which implements the `_nonstandard_virtualizable` GETFIELD_GC +
/// ARRAYLEN_GC fallback and the standard-vable
/// `ConstInt(get_array_length(...))` read.  Trait counterpart:
/// `pyjitpl/dispatch.rs:2053-2068`.
pub(crate) fn arraylen_vable_via_metainterp(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let vable = read_ref_reg(code, op, 0, ctx)?;
    // See `getarrayitem_vable_via_metainterp`: an unseeded `OpRef::None`
    // vable would resize the heapcache flag vector to 16 GiB; bail instead.
    if vable.is_none() {
        return Err(DispatchError::VableBoxNotSeeded { pc: op.pc });
    }
    let vable_struct_ptr = match read_ref_reg_concrete(code, op, 0, ctx) {
        ConcreteValue::Ref(ptr) => ptr as i64,
        ConcreteValue::Null
        | ConcreteValue::Int(_)
        | ConcreteValue::Float(_)
        | ConcreteValue::Bool(_) => 0,
    };
    let (fdescr, adescr) = vable_array_descrs_from_jitcode(code, op, 1, 3, ctx)?;
    let guards_before = ctx.trace_ctx.num_guards();
    let result = ctx
        .trace_ctx
        .vable_arraylen_vable(op.pc, vable, vable_struct_ptr, fdescr, adescr);
    walker_capture_inline_nonstandard_vable_guard(ctx, op.pc, guards_before)?;
    let dst = code[op.pc + 6] as usize;
    let concrete_for_shadow = concrete_from_recorded_opref(ctx, result);
    write_int_reg(ctx, op.pc, dst, result, concrete_for_shadow)?;
    Ok((DispatchOutcome::Continue, op.next_pc))
}
