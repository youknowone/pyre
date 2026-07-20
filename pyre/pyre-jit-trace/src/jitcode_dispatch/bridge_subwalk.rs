//! Bridge / sub-walk driving.
//!
//! **Parity:** trace-side counterpart of `MetaInterp`'s bridge / sub-trace
//! handling (`pyjitpl.py` + `compile.py`).
//!
//! `dispatch_via_miframe` entry plus the bridge-root / recipe-parent /
//! middle-frame / outer-frame continuation drivers that recurse the
//! walker into inlined and bridged frames.

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn dispatch_via_miframe<Sym: WalkSym>(
    trace_ctx: &mut TraceCtx,
    sym: &mut Sym,
    concrete_frame_addr: usize,
    orgpc: usize,
    session: &std::cell::RefCell<WalkSession>,
    jitcode_code: &[u8],
    position: usize,
    descr_refs: &[DescrRef],
    raw_descrs: RawDescrPool,
    is_authoritative_executor: bool,
    sub_jitcode_lookup: &SubJitCodeLookup,
    done_with_this_frame_descr_ref: DescrRef,
    done_with_this_frame_descr_int: DescrRef,
    done_with_this_frame_descr_float: DescrRef,
    done_with_this_frame_descr_void: DescrRef,
    exit_frame_with_exception_descr_ref: DescrRef,
    is_top_level: bool,
    // PyPy `pyjitpl.py:171-176 MIFrame.__init__` analog: the
    // top-level jitcode's per-bank register count.  `dispatch_via_miframe`
    // allocates fresh `Vec<OpRef>`s sized to `top_num_regs_* +
    // top_constants_*.len()` — replacing the prior TODO that
    // reused `sym.registers_r` (a Python locals/stack mirror) as the
    // MIFrame register file.  The codewriter-compiled arm jitcode
    // expects `R[0]_r = handler = MIFrame self ptr`, which the
    // `argboxes_*` parameters supply via the `setup_call` analog
    // below.
    top_num_regs_r: usize,
    top_num_regs_i: usize,
    top_num_regs_f: usize,
    // Top-level jitcode's per-bank constant pool — seeded into
    // register slots `[num_regs_*, num_regs_* + constants_*.len())`
    // per `pyjitpl.py:98-119 copy_constants`.
    top_constants_r: &[i64],
    top_constants_i: &[i64],
    top_constants_f: &[i64],
    // PyPy `pyjitpl.py:188-200 setup_call(argboxes)` analog.
    // `argboxes_*[i]` is written to `registers_*[i]` before walking.
    // Production callers supply `argboxes_r = [const_ref(miframe_ptr)]`
    // so the codewriter-compiled arm finds the MIFrame self ptr at
    // `R[0]_r`.
    argboxes_r: &[OpRef],
    argboxes_i: &[OpRef],
    argboxes_f: &[OpRef],
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sym_ptr = sym as *mut Sym;
    let entry_py_pc = EntryPyPc::Py(orgpc as u32);

    // Phase 7: this IS the full-body walk over the outer `sym.jitcode`,
    // so guard snapshots can resolve a per-guard resume coordinate from
    // `op_pc`.  Set `fbw_mode.snapshot_sym` for the walk's lifetime;
    // `walker_capture_snapshot_for_last_guard` and
    // `fbw_foriter_body_from_op_pc` read it.  This is the PRODUCTION
    // default tracer: `trace.rs` enters `full_body_walk_trace` whenever
    // `PYRE_FULL_BODY_WALK` is not explicitly `0` (the env gate defaults ON),
    // so `fbw_mode.snapshot_sym` is non-null on every default-JIT
    // run.  `PYRE_FULL_BODY_WALK=0` is the only opt-out
    // (the transitional trait leg), which leaves the pointer null.
    // Recover the portal EC red off `sym.frame` before the first opcode is
    // dispatched (thus before any guard is recorded), caching it into
    // `sym.execution_context`.  A bridge-from-guard sym whose ec color collides
    // with a real frame slot is left `OpRef::NONE` by `setup_bridge_sym`, which
    // defers recovery to `ensure_execution_context`.  The walker's
    // snapshot-capture path runs `collect_outer_active_boxes` AFTER the guard,
    // so recovering there would record the getfield after the guard that
    // references it (use-before-def).  Seed here — the trait's pre-guard
    // cache-once analog — so every guard snapshot reads a real EC OpRef.
    seed_execution_context_for_walk(sym, trace_ctx);
    seed_standing_exception_for_walk(sym, trace_ctx);

    // RPython parity: `metainterp.last_exc_value` (pyjitpl.py:1695)
    // is the standing exception OpRef. Walker's `WalkContext::last_exc_value`
    // mirrors this as `Option<OpRef>` — `None` means "no active
    // exception", matching RPython's `assert self.metainterp.last_exc_value`
    // (pyjitpl.py:1702).
    let initial_last_exc_value = if sym.last_exc_box().is_none() {
        None
    } else {
        Some(sym.last_exc_box())
    };

    // PyPy `pyjitpl.py:171-176 MIFrame.__init__` analog: allocate
    // fresh per-bank register vectors sized to `top_num_regs_* +
    // top_constants_*.len()`.  This replaces the prior TODO
    // that reused `sym.registers_r` (a Python locals/stack mirror,
    // whose `[0]` slot is Python local 0) as the MIFrame register
    // file.  The codewriter-compiled arm jitcode emits getfield
    // chains rooted at `R[0] = handler = MIFrame self ptr`; the
    // `argboxes_r` parameter supplies that handler ptr below via the
    // `setup_call` analog.
    let total_r = top_num_regs_r + top_constants_r.len();
    let total_i = top_num_regs_i + top_constants_i.len();
    let total_f = top_num_regs_f + top_constants_f.len();
    let mut top_regs_r = vec![OpRef::NONE; total_r];
    let mut top_regs_i = vec![OpRef::NONE; total_i];
    let mut top_regs_f = vec![OpRef::NONE; total_f];
    let mut top_concrete_r = vec![ConcreteValue::Null; total_r];
    let mut top_concrete_i = vec![ConcreteValue::Null; total_i];

    // PyPy `pyjitpl.py:98-119 copy_constants` analog: seed each
    // constant into the upper slot range `[num_regs_*, total_*)`.
    // `box_value` resolves these via `TraceCtx::constants` so
    // downstream getfield chains see the constant's `Value::*`.
    for (i, &v) in top_constants_i.iter().enumerate() {
        top_regs_i[top_num_regs_i + i] = trace_ctx.const_int(v);
        top_concrete_i[top_num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, &v) in top_constants_r.iter().enumerate() {
        top_regs_r[top_num_regs_r + i] = trace_ctx.const_ref(v);
        if v != 0 {
            top_concrete_r[top_num_regs_r + i] = ConcreteValue::Ref(v as pyre_object::PyObjectRef);
        }
    }
    for (i, &v) in top_constants_f.iter().enumerate() {
        top_regs_f[top_num_regs_f + i] = trace_ctx.const_float(v);
    }

    // PyPy `pyjitpl.py:188-200 setup_call(argboxes)` analog: write
    // each argbox into the leading register slot.  The concrete
    // shadow is derived from `box_value(box)` — for `ConstRef(ptr)`
    // (the common case: argbox=miframe self ptr), this is
    // `Some(Value::Ref(GcRef(ptr)))` resolved via the constant pool;
    // for non-const argboxes it consults the `opref_concrete` stamp
    // table.
    //
    // CodeRabbit Major (PR #89): reject oversized argbox lists up
    // front instead of silently truncating with a per-loop `break`.
    // The `_*_arity_mismatch` DispatchError shapes already exist for
    // the inline-call paths (`InlineCall*ArityMismatch`); reuse them
    // here so a caller/shape mismatch surfaces as a typed failure
    // rather than a partially seeded frame.
    if argboxes_r.len() > top_num_regs_r {
        return Err(DispatchError::InlineCallArityMismatch {
            pc: position,
            provided: argboxes_r.len(),
            callee_num_regs_r: top_num_regs_r,
        });
    }
    if argboxes_i.len() > top_num_regs_i {
        return Err(DispatchError::InlineCallIntArityMismatch {
            pc: position,
            provided: argboxes_i.len(),
            callee_num_regs_i: top_num_regs_i,
        });
    }
    if argboxes_f.len() > top_num_regs_f {
        return Err(DispatchError::InlineCallFloatArityMismatch {
            pc: position,
            provided: argboxes_f.len(),
            callee_num_regs_f: top_num_regs_f,
        });
    }
    for (i, &box_ref) in argboxes_r.iter().enumerate() {
        top_regs_r[i] = box_ref;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = trace_ctx.box_value(box_ref) {
            top_concrete_r[i] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }
    for (i, &box_ref) in argboxes_i.iter().enumerate() {
        top_regs_i[i] = box_ref;
        if let Some(majit_ir::Value::Int(v)) = trace_ctx.box_value(box_ref) {
            top_concrete_i[i] = ConcreteValue::Int(v);
        }
    }
    for (i, &box_ref) in argboxes_f.iter().enumerate() {
        top_regs_f[i] = box_ref;
    }

    // Seed last_exc_value_concrete from
    // sym.last_exc_value (the live PyObjectRef written by trait-side
    // `seed_raised_exception` at `trace_opcode.rs:6646`).  Null when
    // no active exception, matching `initial_last_exc_value == None`.
    let initial_last_exc_value_concrete = if sym.last_exc_value().is_null() {
        ConcreteValue::Null
    } else {
        ConcreteValue::Ref(sym.last_exc_value())
    };

    let result = {
        let mut wc = WalkContext {
            callee_shadow: None,
            inline_callee_consts: None,
            fbw_mode: FbwWalkMode {
                snapshot_sym: sym_ptr,
                current_exception_seed: (trace_ctx.is_bridge_trace
                    && !sym.last_exc_box().is_none())
                .then_some(sym.last_exc_box()),
                current_exception_seed_concrete: if trace_ctx.is_bridge_trace {
                    sym.last_exc_value()
                } else {
                    pyre_object::PY_NULL
                },
                class_of_last_exc_is_const: sym.class_of_last_exc_is_const(),
                // A guard-failure bridge resumes at the opcode boundary, so
                // its first `jit_merge_point` crossing at this python-pc is
                // the same op it is resuming INTO, not a loop crossing. The
                // merge-point arm skips exactly that first crossing. Seeded
                // only for bridge walks; a loop compile leaves it `None`.
                bridge_entry_merge_pc: match (trace_ctx.is_bridge_trace, entry_py_pc) {
                    (true, EntryPyPc::Py(pc)) => Some(pc as usize),
                    _ => None,
                },
                ..Default::default()
            },
            session,
            registers_r: &mut top_regs_r,
            registers_i: &mut top_regs_i,
            registers_f: &mut top_regs_f,
            concrete_registers_r: &mut top_concrete_r,
            concrete_registers_i: &mut top_concrete_i,
            descr_refs,
            raw_descrs,
            is_authoritative_executor,
            // `dispatch_via_miframe` is the full-body walk entry
            // (production tracer, diagnostic probe — the non-production
            // roots are excluded from concrete execution by
            // `is_authoritative_executor: false` instead).
            is_full_body_walk: true,
            trace_ctx,
            done_with_this_frame_descr_ref,
            done_with_this_frame_descr_int,
            done_with_this_frame_descr_float,
            done_with_this_frame_descr_void,
            exit_frame_with_exception_descr_ref,
            is_top_level,
            sub_jitcode_lookup,
            last_exc_value: initial_last_exc_value,
            last_exc_value_concrete: initial_last_exc_value_concrete,
            entry_py_pc,
            outer_resume_marker_jit_pc: None,
            outer_jitcode_index: 0,
            outer_active_boxes: Vec::new(),
            // This entry (test/fixture) hard-codes
            // `outer_jitcode_index = 0` and an empty `outer_active_boxes`
            // rather than seeding them from `sym.jitcode` /
            // `collect_outer_active_boxes` like the retired per-opcode
            // arm entry did.  A specialized
            // `generated_store_subscr_value` guard captured via
            // `walker_capture_snapshot_for_last_guard` would attach
            // resume data pointing at the wrong frame, so keep
            // STORE_SUBSCR specialization off on this entry.
            store_subscr_fn_addr: None,
            pending_guard_snapshot_error: None,
            vstack_boxes: Vec::new(),
            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_ceiling: u32::MAX,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
        };
        // #73: seed the walk-level operand-stack box mirror
        // at entry.  The mirror is only enabled when the outer sym owns the
        // virtualizable shadow (the production full-body loop trace) — the
        // synthetic/test entries leave it disabled (`vstack_valid = false`).
        // Seed at the FIRST-walked jitcode pc (`position`), not `entry_py_pc`,
        // so the first `step_vstack_mirror` is a no-op (no spurious
        // entry-boundary reconcile of the not-yet-executed first opcode).
        // Pure side-data: the snapshot read stays LEGACY until a later slice
        // makes it authoritative.
        seed_vstack_mirror(&mut wc, sym, position);
        let outcome = walk(jitcode_code, position, &mut wc);
        // Read final last_exc_value before wc drops so the borrow
        // checker can release sym for the writeback below.
        let final_last_exc = wc.last_exc_value;
        let final_class_of_last_exc_is_const = wc.fbw_mode.class_of_last_exc_is_const;
        drop(wc);
        // Full `sym.last_exc_*` state writeback parity.
        //
        // RPython `pyjitpl.py:1694-1696 opimpl_raise` sets THREE pieces
        // of metainterp state when a raise fires:
        //   self.metainterp.class_of_last_exc_is_const = True
        //   self.metainterp.last_exc_value = exc_value_box.getref(rclass.OBJECTPTR)
        //   self.metainterp.last_exc_box = exc_value_box
        //
        // Of these, the walker can produce:
        //   - `last_exc_box`: the symbolic OpRef. Mirrored from
        //     `wc.last_exc_value` (RPython's metainterp.last_exc_value
        //     and last_exc_box are different fields — concrete pointer
        //     vs Box — but the walker tracks only the symbolic one,
        //     which lines up with `sym.last_exc_box`).
        //   - `class_of_last_exc_is_const`: true after a raise/r or a
        //     SubRaise routed into a catch handler. RPython sets this
        //     in `opimpl_raise` (line 1694) AND `execute_ll_raised`
        //     (pyjitpl.py:2752 with `constant=...` parameter — set
        //     after GUARD_CLASS / GUARD_EXCEPTION). Walker's raise/r
        //     arm always sets `wc.last_exc_value = Some(exc)` so
        //     mirroring `Some` → const=true is RPython-orthodox.
        //
        // The walker CANNOT produce:
        //   - `sym.last_exc_value` (concrete `PyObjectRef`): RPython
        //     `exc_value_box.getref(rclass.OBJECTPTR)` reads the
        //     concrete pointer at trace-recording time. The symbolic
        //     walker has only OpRefs — concrete writeback is the
        //     production tracer's responsibility (the trait-driven
        //     `MIFrame::execute_opcode_step` path). This is a known
        //     TODO (the walker is symbolic-only,
        //     concrete state is fed by another path).
        if let Some(exc) = final_last_exc {
            sym.set_last_exc_box(exc);
            sym.set_class_of_last_exc_is_const(final_class_of_last_exc_is_const);
        }
        outcome
    };
    result
}

/// Build the paused root portal frame for a multi-frame bridge-carrier
/// sub-walk (#215 item 2 / P2 drain).  The root resumes at `root_pc` once the
/// reconstructed deepest callee returns; the callee's in-callee guards must
/// snapshot this frame on the walk framestack so a guard-failure resume
/// rebuilds both Python frames.  Mirror of [`compute_inline_caller_frame`], but
/// the root register banks come straight from the bridge-seeded `root_sym`
/// rather than a live caller [`WalkContext`] (the root walk has not started —
/// this resumes mid-flight).
pub(crate) fn compute_bridge_root_parent_frame<Sym: WalkSym>(
    root_sym: &Sym,
    trace_ctx: &mut TraceCtx,
    root_pc: usize,
) -> Option<InlineParentFrame> {
    if root_sym.jitcode().is_null() {
        return None;
    }
    let jitcode_index = unsafe { (*root_sym.jitcode()).index as u32 };
    // `root_pc` (`resume_data.frames[0].pc`) is already the post-call resume
    // point — the slot the inner frame's result lands in — so its Python
    // coordinate is a direct backtranslation (no `semantic_fallthrough_pc`).
    // This local remains needed by the outer-active-box collector; the paused
    // parent frame itself carries `ParentResumeCoord::Backxlat(root_pc)`.
    let root_py_pc = crate::state::backxlat_py_pc(jitcode_index as i32, root_pc as i32) as u32;
    // Null the not-yet-produced call-result slot before collecting the active
    // boxes (the reconstructed callee supplies it on `SubReturn`), mirroring
    // `compute_inline_caller_frame`.  Operate on a clone so `root_sym` stays a
    // shared borrow.
    //
    // `collect_outer_active_boxes` reads the Ref bank by abstract register
    // color (`_get_list_of_active_boxes`, pyjitpl.py:216-233), so it needs the
    // color-indexed `f.registers_r` (`consume_boxes`, resume.py:1055), NOT the
    // slot-indexed semantic mirror `setup_bridge_sym` left in
    // `sym.registers_r`.  The mirror leaves an operand live across the resumed
    // call (e.g. `t1` in `return fib(n-1)+fib(n-2)`) at `OpRef::NONE` under its
    // color, which resolves to a NULL const and aborts the second residual
    // call.  Prefer the persisted color decode; fall back to `registers_r` for
    // non-bridge callers (`bridge_registers_r == None`).
    let mut regs_r = root_sym
        .bridge_registers_r()
        .cloned()
        .unwrap_or_else(|| root_sym.registers_r().to_vec());
    if let Some(result_color) = unsafe { &(*root_sym.jitcode()).payload }
        .result_color_trivia_for_jitcode_pc(root_pc)
        .map(|c| c as usize)
        .filter(|&c| c != u16::MAX as usize)
    {
        if result_color < regs_r.len() {
            regs_r[result_color] = trace_ctx.const_ref(pyre_object::PY_NULL as i64);
        }
    }
    let root_word = ((root_pc as i32) != majit_ir::resumedata::NO_JITCODE_PC
        && (root_pc as i32) >= 0)
        .then_some(root_pc);
    let root_liveness_word = match root_word {
        Some(w) => w as i32,
        None => majit_ir::resumedata::NO_JITCODE_PC,
    };
    let boxes = collect_outer_active_boxes(
        root_sym,
        trace_ctx,
        root_sym.registers_i(),
        &regs_r,
        root_sym.registers_f(),
        jitcode_index,
        root_py_pc,
        None,
        // Key the query off the same carried root-frame word the snapshot and
        // decode side read from `frames[0].jitcode_pc`, so both resolve the
        // identical liveness window.
        root_liveness_word,
        root_liveness_word,
        OuterActiveBoxesEntryTwin::Trivia,
        "bridge_root_parent",
        None,
        &[],
    );
    Some(InlineParentFrame {
        jitcode_index,
        call_jitcode_pc: None,
        call_stack_overrides: Vec::new(),
        resume_coord: ParentResumeCoord::Backxlat(root_pc),
        // Parent-frame words are never branch-tagged; negative tags belong to
        // a branch guard's own top-frame word.
        resume_marker_jit_pc: root_word,
        boxes,
    })
}

/// Issue #215 item 2 (P2 drain, increment 2b): drive the reconstructed deepest
/// callee frame of a multi-frame bridge as an INLINE SUB-WALK
/// (`is_top_level = false`) rooted on the caller-visible portal `root_sym`.
///
/// The callee resumes at `entry` (its translated recipe Python pc) with
/// its registers seeded by `argboxes_r` (portal reds + in-flight operand-stack
/// temps from `setup_reconstructed_callee_frame`) and its locals carried in the
/// already-emitted frame vable.  Because the walk is a sub-walk, the callee's
/// `ref_return` surfaces `SubReturn { result }` (`pyjitpl.py:1688 finishframe`)
/// instead of the top-level `Finish` that pyre's own-portal model rejects with
/// `NonStandardVableFinishPortalUnsupported` — the original #215 item-2 wall.
///
/// The root portal is installed as `fbw_mode.snapshot_sym` and pushed onto the
/// walk framestack for the sub-walk's lifetime, so an in-callee guard
/// snapshots both the callee frame and the paused root
/// (`walker_capture_multi_frame_inline_snapshot`).
///
/// Increment 2b-i (diagnostic): returns the sub-walk outcome; the caller logs it
/// and aborts (trace discarded).  Threading `SubReturn` into the root operand
/// stack + the root top-level walk forward to a terminator is increment 2b-ii.
/// `None` signals a setup failure (terminal descrs unwired / no root frame).
pub(crate) fn call_dst_reg_for_residual_return(code: &[u8], entry: usize) -> Option<usize> {
    for op in crate::jitcode_runtime::decoded_ops(code) {
        if op.next_pc == entry {
            return (op.opname.starts_with("residual_call") && op.argcodes.ends_with(">r"))
                .then(|| code.get(entry - 1).map(|&b| b as usize))
                .flatten();
        }
    }
    None
}

pub(crate) fn recipe_parent_frame_from_recipe(
    ctx: &mut TraceCtx,
    recipe: &majit_metainterp::ReconstructRecipe,
    root_ec: *const pyre_interpreter::PyExecutionContext,
) -> Option<InlineParentFrame> {
    let pjc = crate::state::pyjitcode_for_jitcode_index(recipe.jitcode_index)?;
    if !pjc.is_populated() || pjc.code_ptr.is_null() {
        return None;
    }
    let entry =
        if crate::state::frame_pc_is_resolved_offset_at(recipe.jitcode_index, recipe.jitcode_pc) {
            recipe.jitcode_pc as usize
        } else {
            return None;
        };
    let call_jit_pc = crate::jitcode_runtime::decoded_ops(pjc.jitcode.code.as_slice())
        .find(|op| op.next_pc == entry && op.opname.starts_with("residual_call"))
        .map(|op| op.pc);
    let resume_marker_jit_pc =
        call_jit_pc.and_then(|pc| pjc.after_residual_marker_for_jitcode_pc(pc));

    // Reconstruct this paused parent frame's vable + ec (the same
    // `emit_new_pyframe_inline_with_params` the deepest-callee setup uses) so
    // the paused-frame snapshot resolves the portal reds
    // [frame, ec] (`interp_jit.py:67`) to real boxes rather than reading the
    // slot-indexed `registers_r` at the portal-red color positions.  Only
    // `pending.sym.frame` / `pending.sym.execution_context` are consumed here;
    // the `argboxes_r` register seeding is for the forward drive, not the
    // snapshot.
    let (pending, _argboxes_r) =
        crate::state::setup_reconstructed_callee_frame(ctx, recipe, root_ec, Vec::new())?;
    let frame_box = pending.sym.frame();
    let ec_box = pending.sym.execution_context();

    let (frame_reg, ec_reg) = crate::state::portal_red_regs_at(recipe.jitcode_index);
    let (frame_reg, ec_reg) = (u32::from(frame_reg), u32::from(ec_reg));
    let sentinel = u32::from(u16::MAX);
    let result_color = pjc
        .result_color_trivia_for_jitcode_pc(recipe.jitcode_pc as usize)
        .map(|c| c as usize)
        .filter(|&c| c != u16::MAX as usize);

    let banks = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
        recipe.jitcode_index,
        recipe.jitcode_pc,
    );
    let stack_only = recipe.valuestackdepth.saturating_sub(recipe.nlocals);
    let maps = crate::state::bridge_semantic_maps_from_pc(recipe.jitcode_index, recipe.jitcode_pc);
    let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
    let mut boxes = Vec::with_capacity(banks.total_len());
    for &color in &banks.int {
        boxes.push(
            recipe
                .registers_i
                .get(color as usize)
                .copied()
                .unwrap_or(OpRef::NONE),
        );
    }
    // Ref bank, in liveness-color order — mirror the trait encoder
    // `get_list_of_active_boxes` (trace_opcode.rs:1902-1957) box-for-box:
    //   * the not-yet-produced call-result color is NULL-seeded (`in_a_call`);
    //   * a force-alived portal-red SCRATCH color (no live semantic slot at this
    //     pc) routes to the reconstructed frame's `frame`/`ec` box, NOT the
    //     slot-indexed register file;
    //   * every other live color reads its semantic `locals_cells_stack_w` slot
    //     from the slot-indexed `registers_r` (the reconstruct decode).
    for &color in &banks.ref_ {
        let c = color as usize;
        if result_color == Some(c) {
            boxes.push(null_ref);
            continue;
        }
        let semantic_idx = crate::state::semantic_ref_slot_for_reg_color(
            recipe.nlocals,
            stack_only,
            &maps.pcdep_entries,
            c,
        );
        let is_portal_red_scratch = semantic_idx.is_none()
            && ((color == frame_reg && frame_reg != sentinel)
                || (color == ec_reg && ec_reg != sentinel));
        if is_portal_red_scratch {
            boxes.push(if color == frame_reg {
                frame_box
            } else {
                ec_box
            });
            continue;
        }
        let slot = semantic_idx.or_else(|| (c < recipe.valuestackdepth).then_some(c))?;
        boxes.push(recipe.registers_r.get(slot).copied().unwrap_or(OpRef::NONE));
    }
    for &color in &banks.float {
        boxes.push(
            recipe
                .registers_f
                .get(color as usize)
                .copied()
                .unwrap_or(OpRef::NONE),
        );
    }
    if boxes.iter().any(|b| b.is_none()) {
        return None;
    }

    Some(InlineParentFrame {
        jitcode_index: recipe.jitcode_index as u32,
        call_jitcode_pc: call_jit_pc,
        call_stack_overrides: Vec::new(),
        // The recipe's resolved word was `backxlat_py_pc(jitcode_index,
        // jitcode_pc)` by construction, exactly the bridge-root flavor.
        resume_coord: ParentResumeCoord::Backxlat(recipe.jitcode_pc as usize),
        resume_marker_jit_pc,
        boxes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_bridge_frame_subwalk<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<WalkSession>,
    root_sym: &Sym,
    root_pc: usize,
    callee_pjc: &std::sync::Arc<crate::PyJitCode>,
    callee_code_key: usize,
    callee_w_globals: usize,
    entry: usize,
    argboxes_r: &[OpRef],
    local_concretes: &[majit_ir::Value],
    child_result: Option<OpRef>,
    paused_parent_recipes: &[majit_metainterp::ReconstructRecipe],
) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
    use majit_metainterp::jitcode::RuntimeBhDescr;

    // Terminal descrs off MetaInterpStaticData (mirror `dispatch_perfn_frame`).
    let (done_void, done_int, done_ref, done_float, exit_exc_ref) = {
        let sd = ctx.metainterp_sd();
        match (
            sd.done_with_this_frame_descr_void.clone(),
            sd.done_with_this_frame_descr_int.clone(),
            sd.done_with_this_frame_descr_ref.clone(),
            sd.done_with_this_frame_descr_float.clone(),
            sd.exit_frame_with_exception_descr_ref.clone(),
        ) {
            (Some(v), Some(i), Some(r), Some(f), Some(e)) => (v, i, r, f, e),
            _ => return None,
        }
    };

    // Per-fn descr pool + sub-jitcode lookup off the callee body's own runtime
    // pool (mirror `dispatch_perfn_frame`).  `callee_pjc` is an `Arc` that
    // outlives the walk, so extend the descr-slice borrow to `'static` for the
    // `'static`-bodied `SubJitCodeBody` lookup.
    let perfn_descrs: &'static [RuntimeBhDescr] =
        unsafe { &*(callee_pjc.jitcode.exec.descrs.as_slice() as *const [RuntimeBhDescr]) };
    let perfn_descr_refs: Vec<majit_ir::DescrRef> = perfn_descrs
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            RuntimeBhDescr::Descr(bh) => crate::descr::make_descr_from_bh(bh),
            RuntimeBhDescr::JitCode(_)
            | RuntimeBhDescr::Call(_)
            | RuntimeBhDescr::AssemblerToken(_) => crate::descr::make_jitcode_descr(i),
        })
        .collect();
    let sub_jitcode_lookup = |idx: usize| -> Option<SubJitCodeBody> {
        perfn_descrs
            .get(idx)
            .and_then(|d| d.as_jitcode())
            .map(|jc| SubJitCodeBody {
                code: jc.code.as_slice(),
                num_regs_r: jc.num_regs_r() as usize,
                num_regs_i: jc.num_regs_i() as usize,
                num_regs_f: jc.num_regs_f() as usize,
                constants_i: jc.constants_i.as_slice(),
                constants_r: jc.constants_r.as_slice(),
                constants_f: jc.constants_f.as_slice(),
            })
    };

    // Allocate the callee register banks sized to `num_regs_* + constants_*`,
    // seed the constant pool into the upper slots and `argboxes_r` into the
    // leading slots (mirror `dispatch_via_miframe`).
    let jc = &callee_pjc.jitcode;
    let num_regs_r = jc.num_regs_r() as usize;
    let num_regs_i = jc.num_regs_i() as usize;
    let num_regs_f = jc.num_regs_f() as usize;
    let total_r = num_regs_r + jc.constants_r.len();
    let total_i = num_regs_i + jc.constants_i.len();
    let total_f = num_regs_f + jc.constants_f.len();
    let mut regs_r = vec![OpRef::NONE; total_r];
    let mut regs_i = vec![OpRef::NONE; total_i];
    let mut regs_f = vec![OpRef::NONE; total_f];
    let mut concrete_r = vec![ConcreteValue::Null; total_r];
    let mut concrete_i = vec![ConcreteValue::Null; total_i];
    for (i, &v) in jc.constants_i.iter().enumerate() {
        regs_i[num_regs_i + i] = ctx.const_int(v);
        concrete_i[num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, &v) in jc.constants_r.iter().enumerate() {
        regs_r[num_regs_r + i] = ctx.const_ref(v);
        if v != 0 {
            concrete_r[num_regs_r + i] = ConcreteValue::Ref(v as pyre_object::PyObjectRef);
        }
    }
    for (i, &v) in jc.constants_f.iter().enumerate() {
        regs_f[num_regs_f + i] = ctx.const_float(v);
    }
    if argboxes_r.len() > num_regs_r {
        return Some(Err(DispatchError::InlineCallArityMismatch {
            pc: entry,
            provided: argboxes_r.len(),
            callee_num_regs_r: num_regs_r,
        }));
    }
    for (i, &box_ref) in argboxes_r.iter().enumerate() {
        regs_r[i] = box_ref;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = ctx.box_value(box_ref) {
            concrete_r[i] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }
    if let Some(result) = child_result {
        let call_dst_reg = call_dst_reg_for_residual_return(jc.code.as_slice(), entry)?;
        if call_dst_reg >= regs_r.len() {
            return Some(Err(DispatchError::InlineCallArityMismatch {
                pc: entry,
                provided: call_dst_reg + 1,
                callee_num_regs_r: regs_r.len(),
            }));
        }
        regs_r[call_dst_reg] = result;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = ctx.box_value(result) {
            concrete_r[call_dst_reg] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }

    // Paused root portal frame for the multi-frame guard snapshot.
    let root_frame = compute_bridge_root_parent_frame(root_sym, ctx, root_pc)?;
    let outer_jitcode_index = root_frame.jitcode_index;
    let outer_active_boxes = root_frame.boxes.clone();

    let callee_code = jc.code.as_slice();
    let lookup_ref: &SubJitCodeLookup = &sub_jitcode_lookup;
    let consts = InlineCalleeConsts {
        w_globals: callee_w_globals,
        w_code: callee_code_key,
    };

    // Install the ROOT sym as the snapshot sym (NOT the callee's) so in-callee
    // guards snapshot the paused root.
    let root_sym_ptr = root_sym as *const Sym;

    let mut parent_guards = Vec::new();
    let mut parent_for_current = root_frame.clone();
    for parent_recipe in paused_parent_recipes {
        let guard_parent = parent_for_current.clone();
        parent_guards.push(InlineFrameGuard::enter(
            session,
            parent_recipe.code_ptr as usize,
            Some(guard_parent),
        ));
        parent_for_current = recipe_parent_frame_from_recipe(
            ctx,
            parent_recipe,
            root_sym.concrete_execution_context(),
        )?;
    }

    let outcome = {
        let mut sub_wc = WalkContext {
            callee_shadow: Some(Default::default()),
            inline_callee_consts: Some(consts),
            fbw_mode: FbwWalkMode {
                snapshot_sym: root_sym_ptr,
                inline_subwalk: true,
                carrier_resume: true,
                current_exception_seed: (!root_sym.last_exc_box().is_none())
                    .then_some(root_sym.last_exc_box()),
                current_exception_seed_concrete: root_sym.last_exc_value(),
                class_of_last_exc_is_const: root_sym.class_of_last_exc_is_const(),
                ..Default::default()
            },
            session,
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut regs_f,
            concrete_registers_r: &mut concrete_r,
            concrete_registers_i: &mut concrete_i,
            descr_refs: &perfn_descr_refs,
            raw_descrs: RawDescrPool::PerFn(perfn_descrs),
            // The carrier sub-walk IS the bridge-resume metainterp: after a
            // guard failure `handle_guard_failure` rebuilds the frame state and
            // drives it forward through the SAME `self.interpret()` the initial
            // trace uses (`pyjitpl.py:2937 _handle_guard_failure` →
            // `prepare_resume_from_failure` → `interpret`, cf.
            // `_compile_and_run_once:2899`).  There is no second-class executor
            // mode: the resume walk concrete-executes every residual call
            // (`do_residual_call` → `execute_varargs`, `pyjitpl.py:1995`) exactly
            // like the initial trace, which is what lets a nested self-recursive
            // call fold to a live `CALL_ASSEMBLER`.  The residual it reaches was
            // never run pre-deopt (the deopt cut the trace there), so this is its
            // first and only concrete execution.
            is_authoritative_executor: true,
            is_full_body_walk: true,
            store_subscr_fn_addr: None,
            pending_guard_snapshot_error: None,
            vstack_boxes: Vec::new(),
            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_ceiling: u32::MAX,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
            trace_ctx: ctx,
            done_with_this_frame_descr_ref: done_ref,
            done_with_this_frame_descr_int: done_int,
            done_with_this_frame_descr_float: done_float,
            done_with_this_frame_descr_void: done_void,
            exit_frame_with_exception_descr_ref: exit_exc_ref,
            is_top_level: false,
            sub_jitcode_lookup: lookup_ref,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            // The outer Python frame is the root, paused at `root_pc`.
            entry_py_pc: EntryPyPc::Jit(root_pc),
            outer_resume_marker_jit_pc: root_frame.resume_marker_jit_pc,
            outer_jitcode_index,
            outer_active_boxes,
        };
        let _inline_frame =
            InlineFrameGuard::enter(session, callee_code_key, Some(parent_for_current));
        // Nested self-recursive calls inside the resumed callee fold straight to
        // a recursive-portal CALL_ASSEMBLER (the bridge is the deopt
        // continuation, not a fresh unroll).
        // Seed the reconstructed callee's local slot concretes into its frame-
        // owned shadow.  The resume is mid-body,
        // so the locals were stored to the frame vable before the guard fired;
        // the map is empty until seeded.  A concrete local lets a callee
        // `getarrayitem_vable(frame, slot)` read fold to its value, so a nested
        // self-recursive call's int arg is known (`arg_is_int`) and the call
        // folds to `CALL_ASSEMBLER` instead of declining.
        for (slot, &v) in local_concretes.iter().enumerate() {
            sub_wc.callee_shadow.as_mut().unwrap().set_concrete(
                callee_pjc.metadata.portal_frame_reg,
                slot as i64,
                v,
            );
        }
        walk(callee_code, entry, &mut sub_wc)
    };
    drop(parent_guards);
    Some(outcome)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_bridge_carrier_subwalk<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<WalkSession>,
    root_sym: &Sym,
    root_pc: usize,
    callee_pjc: &std::sync::Arc<crate::PyJitCode>,
    callee_code_key: usize,
    callee_w_globals: usize,
    entry: usize,
    argboxes_r: &[OpRef],
    local_concretes: &[majit_ir::Value],
    paused_parent_recipes: &[majit_metainterp::ReconstructRecipe],
) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
    drive_bridge_frame_subwalk(
        ctx,
        session,
        root_sym,
        root_pc,
        callee_pjc,
        callee_code_key,
        callee_w_globals,
        entry,
        argboxes_r,
        local_concretes,
        None,
        paused_parent_recipes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_bridge_middle_frame<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<WalkSession>,
    root_sym: &Sym,
    root_pc: usize,
    middle_pjc: &std::sync::Arc<crate::PyJitCode>,
    middle_code_key: usize,
    middle_w_globals: usize,
    entry: usize,
    argboxes_r: &[OpRef],
    local_concretes: &[majit_ir::Value],
    child_result: OpRef,
) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
    drive_bridge_frame_subwalk(
        ctx,
        session,
        root_sym,
        root_pc,
        middle_pjc,
        middle_code_key,
        middle_w_globals,
        entry,
        argboxes_r,
        local_concretes,
        Some(child_result),
        &[],
    )
}

/// #41 continuous cross-frame walk: after the deepest callee sub-walk
/// ([`drive_bridge_carrier_subwalk`]) returns the callee result, continue the
/// OUTER (root portal) frame forward from its resume pc, APPENDING to the same
/// `ctx` (no cut, no fresh `run_perfn_walk` — that resets the trace and
/// discards the sub-walk's live `CALL_ASSEMBLER` continuation).
///
/// The outer resumes AFTER its residual call (`entry` = the jitcode pc of the
/// `live/` marker following the call), so the callee `result` is delivered into
/// the outer's call-dst register (`call_dst_reg`) — the physical slot the call
/// op wrote, i.e. `make_result_of_lastop` (`pyjitpl.py:258-275`), NOT a resume
/// color. The frame vable identity is seeded at `frame_reg` (the standard
/// virtualizable box) so the outer's local reads (`getarrayitem_vable`) resolve
/// off the paused root frame. `is_top_level=true`: the outer IS the portal, so
/// its `*_return` surfaces the portal `Terminate`/finish, not a `SubReturn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_outer_frame_continuation<Sym: WalkSym>(
    ctx: &mut TraceCtx,
    session: &std::cell::RefCell<WalkSession>,
    root_sym: &Sym,
    root_pjc: &std::sync::Arc<crate::PyJitCode>,
    root_code_key: usize,
    root_w_globals: usize,
    root_pc: usize,
    entry: usize,
    frame_box: OpRef,
    frame_reg: usize,
    result: OpRef,
    call_dst_reg: usize,
) -> Option<Result<(DispatchOutcome, usize), DispatchError>> {
    use majit_metainterp::jitcode::RuntimeBhDescr;

    let (done_void, done_int, done_ref, done_float, exit_exc_ref) = {
        let sd = ctx.metainterp_sd();
        match (
            sd.done_with_this_frame_descr_void.clone(),
            sd.done_with_this_frame_descr_int.clone(),
            sd.done_with_this_frame_descr_ref.clone(),
            sd.done_with_this_frame_descr_float.clone(),
            sd.exit_frame_with_exception_descr_ref.clone(),
        ) {
            (Some(v), Some(i), Some(r), Some(f), Some(e)) => (v, i, r, f, e),
            _ => return None,
        }
    };

    let perfn_descrs: &'static [RuntimeBhDescr] =
        unsafe { &*(root_pjc.jitcode.exec.descrs.as_slice() as *const [RuntimeBhDescr]) };
    let perfn_descr_refs: Vec<majit_ir::DescrRef> = perfn_descrs
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            RuntimeBhDescr::Descr(bh) => crate::descr::make_descr_from_bh(bh),
            RuntimeBhDescr::JitCode(_)
            | RuntimeBhDescr::Call(_)
            | RuntimeBhDescr::AssemblerToken(_) => crate::descr::make_jitcode_descr(i),
        })
        .collect();
    let sub_jitcode_lookup = |idx: usize| -> Option<SubJitCodeBody> {
        perfn_descrs
            .get(idx)
            .and_then(|d| d.as_jitcode())
            .map(|jc| SubJitCodeBody {
                code: jc.code.as_slice(),
                num_regs_r: jc.num_regs_r() as usize,
                num_regs_i: jc.num_regs_i() as usize,
                num_regs_f: jc.num_regs_f() as usize,
                constants_i: jc.constants_i.as_slice(),
                constants_r: jc.constants_r.as_slice(),
                constants_f: jc.constants_f.as_slice(),
            })
    };

    let jc = &root_pjc.jitcode;
    let num_regs_r = jc.num_regs_r() as usize;
    let num_regs_i = jc.num_regs_i() as usize;
    let num_regs_f = jc.num_regs_f() as usize;
    let total_r = num_regs_r + jc.constants_r.len();
    let total_i = num_regs_i + jc.constants_i.len();
    let total_f = num_regs_f + jc.constants_f.len();
    let mut regs_r = vec![OpRef::NONE; total_r];
    let mut regs_i = vec![OpRef::NONE; total_i];
    let mut regs_f = vec![OpRef::NONE; total_f];
    let mut concrete_r = vec![ConcreteValue::Null; total_r];
    let mut concrete_i = vec![ConcreteValue::Null; total_i];
    for (i, &v) in jc.constants_i.iter().enumerate() {
        regs_i[num_regs_i + i] = ctx.const_int(v);
        concrete_i[num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, &v) in jc.constants_r.iter().enumerate() {
        regs_r[num_regs_r + i] = ctx.const_ref(v);
        if v != 0 {
            concrete_r[num_regs_r + i] = ConcreteValue::Ref(v as pyre_object::PyObjectRef);
        }
    }
    for (i, &v) in jc.constants_f.iter().enumerate() {
        regs_f[num_regs_f + i] = ctx.const_float(v);
    }

    let root_frame = compute_bridge_root_parent_frame(root_sym, ctx, root_pc)?;
    let outer_jitcode_index = root_frame.jitcode_index;
    let outer_active_boxes = root_frame.boxes.clone();

    // Seed the full live outer-frame register file, matching `consume_boxes`
    // (resume.py:1054-1055 + `_callback_i/_r/_f`) which fills every live
    // register color of `framestack[-1]` from the resume numbering.  Without
    // this only `frame_reg`/`call_dst_reg` are bound: an operand live across
    // the resumed call (e.g. the first result of `return fib(n-1)+fib(n-2)`)
    // stays `OpRef::NONE`, and the second residual call aborts with
    // `ResidualCallArgUnbound`.
    //
    // `root_frame.boxes` already holds each live register's resolved box in
    // `banks.int ++ banks.ref_ ++ banks.float` liveness order
    // (`collect_outer_active_boxes`), applying the color->slot inversion the
    // Ref bank needs.  Re-query the same (deterministic) liveness banks to
    // recover each box's register color and scatter it into the color-indexed
    // walker banks.  Only live colors are touched; dead slots stay
    // `OpRef::NONE` so a later guard snapshot cannot capture a stale operand.
    {
        let root_word = ((root_pc as i32) != majit_ir::resumedata::NO_JITCODE_PC
            && (root_pc as i32) >= 0)
            .then_some(root_pc);
        // Mirror `compute_bridge_root_parent_frame` so scatter reads the banks in collection order.
        let banks = crate::state::frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
            outer_jitcode_index as i32,
            match root_word {
                Some(w) => w as i32,
                None => majit_ir::resumedata::NO_JITCODE_PC,
            },
        );
        let mut cursor = 0usize;
        for &color in &banks.int {
            let v = outer_active_boxes
                .get(cursor)
                .copied()
                .unwrap_or(OpRef::NONE);
            cursor += 1;
            let c = color as usize;
            if c < regs_i.len() && v != OpRef::NONE {
                regs_i[c] = v;
                if let Some(majit_ir::Value::Int(n)) = ctx.box_value(v) {
                    concrete_i[c] = ConcreteValue::Int(n);
                }
            }
        }
        for &color in &banks.ref_ {
            let v = outer_active_boxes
                .get(cursor)
                .copied()
                .unwrap_or(OpRef::NONE);
            cursor += 1;
            let c = color as usize;
            if c < regs_r.len() && v != OpRef::NONE {
                regs_r[c] = v;
                if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = ctx.box_value(v) {
                    if ptr != 0 && ptr != usize::MAX {
                        concrete_r[c] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
                    }
                }
            }
        }
        for &color in &banks.float {
            let v = outer_active_boxes
                .get(cursor)
                .copied()
                .unwrap_or(OpRef::NONE);
            cursor += 1;
            let c = color as usize;
            if c < regs_f.len() && v != OpRef::NONE {
                regs_f[c] = v;
            }
        }
    }

    // Seed the standard virtualizable identity (frame) so the outer's vable
    // reads hit the standard fast path, and the delivered callee result into the
    // outer's call-dst register (`make_result_of_lastop`).  These overwrite the
    // scattered snapshot values (the call-dst slot was nulled in
    // `compute_bridge_root_parent_frame` for the not-yet-produced result).
    if frame_reg < regs_r.len() {
        regs_r[frame_reg] = frame_box;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = ctx.box_value(frame_box) {
            concrete_r[frame_reg] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }
    if call_dst_reg < regs_r.len() {
        regs_r[call_dst_reg] = result;
        if let Some(majit_ir::Value::Ref(majit_ir::GcRef(ptr))) = ctx.box_value(result) {
            concrete_r[call_dst_reg] = ConcreteValue::Ref(ptr as pyre_object::PyObjectRef);
        }
    }

    let root_code = jc.code.as_slice();
    let lookup_ref: &SubJitCodeLookup = &sub_jitcode_lookup;
    let consts = InlineCalleeConsts {
        w_globals: root_w_globals,
        w_code: root_code_key,
    };

    let root_sym_ptr = root_sym as *const Sym;

    let outcome = {
        let mut outer_wc = WalkContext {
            callee_shadow: None,
            inline_callee_consts: Some(consts),
            fbw_mode: FbwWalkMode {
                snapshot_sym: root_sym_ptr,
                inline_subwalk: true,
                carrier_resume: true,
                current_exception_seed: (!root_sym.last_exc_box().is_none())
                    .then_some(root_sym.last_exc_box()),
                current_exception_seed_concrete: root_sym.last_exc_value(),
                class_of_last_exc_is_const: root_sym.class_of_last_exc_is_const(),
                ..Default::default()
            },
            session,
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            registers_f: &mut regs_f,
            concrete_registers_r: &mut concrete_r,
            concrete_registers_i: &mut concrete_i,
            descr_refs: &perfn_descr_refs,
            raw_descrs: RawDescrPool::PerFn(perfn_descrs),
            is_authoritative_executor: true,
            is_full_body_walk: true,
            store_subscr_fn_addr: None,
            pending_guard_snapshot_error: None,
            vstack_boxes: Vec::new(),
            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_ceiling: u32::MAX,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
            trace_ctx: ctx,
            done_with_this_frame_descr_ref: done_ref,
            done_with_this_frame_descr_int: done_int,
            done_with_this_frame_descr_float: done_float,
            done_with_this_frame_descr_void: done_void,
            exit_frame_with_exception_descr_ref: exit_exc_ref,
            is_top_level: true,
            sub_jitcode_lookup: lookup_ref,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: EntryPyPc::Jit(root_pc),
            outer_resume_marker_jit_pc: root_frame.resume_marker_jit_pc,
            outer_jitcode_index,
            outer_active_boxes,
        };
        let _inline_frame = InlineFrameGuard::enter(session, root_code_key, Some(root_frame));
        // The outer's own second self-recursive call folds to a live
        // recursive-portal CALL_ASSEMBLER (same as the callee's), not a fresh
        // unroll.
        walk(root_code, entry, &mut outer_wc)
    };
    Some(outcome)
}
