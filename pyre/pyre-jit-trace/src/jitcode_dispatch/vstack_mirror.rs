//! Value-stack mirror: classify / reconcile / reseed / seed / step /
//! enter-handler.
//!
//! **Parity:** pyre-specific — mirrors the interpreter value-stack shape
//! for the FBW walker; PyPy's `MIFrame` uses register banks with no stack
//! mirror.
//!
//! Tracks the interpreter value-stack shape alongside the walker so a
//! guard resume coordinate and the catch-landing depth can be
//! reconstructed. Includes the pc-map block-head helpers the mirror
//! steps through.

use super::*;

/// #73: classify `instr` for the [`VstackOpClass`] taxonomy.  Mirrors the
/// stack-effect grouping in [`crate::liveness`]'s `stack_effects`, but
/// collapsed to the three categories the operand-stack box maintenance
/// cares about.  `op_arg` is read only where the net effect depends on it
/// (LOAD_GLOBAL's NULL-sentinel low bit).
pub(crate) fn classify_vstack_opcode(
    instr: &pyre_interpreter::bytecode::Instruction,
    op_arg: pyre_interpreter::OpArg,
) -> VstackOpClass {
    use pyre_interpreter::bytecode::Instruction;
    match instr {
        // Trivia / no stack effect — neither produces a TOS box nor pops.
        // Treat as pop-only-or-side-store: truncate to the (unchanged)
        // depth, leave the surviving slots intact.
        Instruction::Nop
        | Instruction::Resume { .. }
        | Instruction::Cache
        | Instruction::NotTaken
        | Instruction::ExtendedArg => VstackOpClass::PopOnlyOrSideStore,

        // Single value lands on the new TOS = the last Ref written.
        Instruction::LoadConst { .. }
        | Instruction::LoadSmallInt { .. }
        | Instruction::LoadFast { .. }
        | Instruction::LoadFastBorrow { .. }
        | Instruction::LoadFastCheck { .. }
        | Instruction::LoadFastAndClear { .. }
        | Instruction::LoadName { .. }
        | Instruction::LoadDeref { .. }
        | Instruction::LoadLocals
        | Instruction::UnaryNegative
        | Instruction::UnaryNot
        | Instruction::UnaryInvert
        | Instruction::ToBool
        | Instruction::GetIter
        | Instruction::GetLen
        | Instruction::LoadAttr { .. }
        | Instruction::ImportFrom { .. }
        | Instruction::BinaryOp { .. }
        | Instruction::CompareOp { .. }
        | Instruction::IsOp { .. }
        | Instruction::ContainsOp { .. }
        | Instruction::Call { .. }
        | Instruction::BuildTuple { .. }
        | Instruction::BuildList { .. }
        | Instruction::BuildSet { .. }
        | Instruction::BuildMap { .. }
        | Instruction::BuildString { .. }
        // Compute opcodes that pop their operands and leave a single result on
        // the new TOS (= the last Ref written, captured via `write_ref_reg` or
        // the operand-stack push chokepoint) — same shape as the arithmetic /
        // build group above.  PUSH_NULL's sole new TOS is the pushed NULL
        // marker; FORMAT_SIMPLE/FORMAT_WITH_SPEC/CONVERT_VALUE/BINARY_SLICE/
        // IMPORT_NAME each push exactly one result.
        | Instruction::PushNull
        | Instruction::FormatSimple
        | Instruction::FormatWithSpec
        | Instruction::ConvertValue { .. }
        | Instruction::BinarySlice
        | Instruction::ImportName { .. }
        // MAKE_FUNCTION pops the code object and pushes the built function
        // (net 0, `stack_effects` `(d, d)`).  The `make_function_value`
        // residual's Ref result reaches the new TOS through the operand-stack
        // push chokepoint (`emit_pushvalue_ref!`, codewriter.rs:9260), so it is
        // the same `ResultToTos` shape as the value producers above.  Left
        // unmodeled it killed the mirror for the rest of the walk at the first
        // nested `def`, declining any later depth > 1 kept-stack branch guard.
        | Instruction::MakeFunction
        // SET_FUNCTION_ATTRIBUTE follows MAKE_FUNCTION for defaults,
        // annotations, and closures, and pushes the updated function back to
        // TOS.  The remaining ops here also leave one result on the new TOS;
        // their arg-dependent depths are already baked into
        // `pyre/pyre-jit-trace/src/liveness.rs`'s depth table.
        | Instruction::SetFunctionAttribute { .. }
        | Instruction::CallKw { .. }
        | Instruction::BuildSlice { .. }
        | Instruction::CallFunctionEx
        | Instruction::CallIntrinsic1 { .. }
        | Instruction::LoadCommonConstant { .. }
        | Instruction::LoadFromDictOrGlobals { .. }
        // #73: LOAD_FAST/STORE_FAST super-instructions.  Their net
        // result still lands on the new TOS as the LAST Ref written (the
        // second load, resp. the load following the store), so `ResultToTos`
        // models the top slot correctly.  A two-push pair
        // (`LoadFast(Borrow)LoadFast(Borrow)`, net +2) additionally leaves the
        // slot BELOW the new TOS a NONE hole; the general hole-fill in
        // `reconcile_vstack_at_boundary` recovers it from the virtualizable
        // shadow (or leaves it NONE when unsourceable — the overlay then
        // omits the slot, which resume re-materializes) WITHOUT invalidating
        // the mirror.  Net-0 `StoreFastLoadFast` overwrites the
        // consumed TOS with the loaded value (no hole).  Before this slice
        // these fell through to `Unmodeled`, killing the mirror at the first
        // super-instruction in a short-circuit / condexpr loop body.
        | Instruction::LoadFastLoadFast { .. }
        | Instruction::LoadFastBorrowLoadFastBorrow { .. }
        | Instruction::StoreFastLoadFast { .. } => VstackOpClass::ResultToTos,

        // Pop-only / side-store / control transfer: the surviving TOS box
        // is already in `vstack_boxes`, do NOT overwrite it from the last
        // ref (which targets a local/global/attr, not the new stack TOS).
        Instruction::PopTop
        | Instruction::PopIter
        | Instruction::PopExcept
        | Instruction::StoreFast { .. }
        // STORE_FAST__STORE_FAST: two consecutive local stores, pops 2 with no
        // stack result — a pure side-store, the surviving slots just truncate.
        | Instruction::StoreFastStoreFast { .. }
        | Instruction::StoreName { .. }
        | Instruction::StoreGlobal { .. }
        | Instruction::StoreDeref { .. }
        | Instruction::StoreAttr { .. }
        | Instruction::DeleteAttr { .. }
        | Instruction::StoreSubscr
        | Instruction::DeleteSubscr
        | Instruction::StoreSlice
        // LIST_APPEND / SET_ADD / MAP_ADD / LIST_EXTEND and the dict/set
        // update opcodes pop their value operand(s) and mutate the collection
        // PEEK'd in place below them — a side-store, same shape as
        // STORE_SUBSCR: the surviving TOS box stays put. MAKE_CELL stores its
        // result into the frame-local virtualizable slot, not operand TOS.
        | Instruction::ListAppend { .. }
        | Instruction::SetAdd { .. }
        | Instruction::MapAdd { .. }
        | Instruction::ListExtend { .. }
        | Instruction::DictUpdate { .. }
        | Instruction::DictMerge { .. }
        | Instruction::SetUpdate { .. }
        | Instruction::MakeCell { .. }
        | Instruction::DeleteFast { .. }
        | Instruction::DeleteName { .. }
        | Instruction::DeleteGlobal { .. }
        | Instruction::DeleteDeref { .. }
        | Instruction::PopJumpIfTrue { .. }
        | Instruction::PopJumpIfFalse { .. }
        | Instruction::PopJumpIfNone { .. }
        | Instruction::PopJumpIfNotNone { .. }
        | Instruction::JumpForward { .. }
        | Instruction::JumpBackward { .. }
        | Instruction::JumpBackwardNoInterrupt { .. }
        | Instruction::ReturnValue => VstackOpClass::PopOnlyOrSideStore,

        // LOAD_GLOBAL: the global value is the new TOS = the last Ref written.
        // When `namei & 1` the lowering also pushes a NULL sentinel BENEATH the
        // result (net +2, for the upcoming method CALL).  Exactly like the
        // two-push `LoadFast*LoadFast*` super-instructions, that leaves the slot
        // below the new TOS a NONE hole which the general hole-fill below
        // recovers from the virtualizable shadow (or leaves NONE when
        // unsourceable — the overlay then omits the slot, which resume
        // re-materializes) WITHOUT invalidating the mirror.  The NULL sentinel
        // is consumed by the CALL before any short-circuit branch guard, so it
        // is never a live kept-stack slot at a resume.  (Previously the
        // `namei & 1` arm declined to `Unmodeled`; the hole-fill makes that
        // unnecessary, and the decline killed the mirror for the rest of any
        // walk with a method-form global load — the dominant mirror=NONE gap.)
        Instruction::LoadGlobal { .. } => VstackOpClass::ResultToTos,

        // LOAD_SUPER_ATTR: the attribute (non-method form) is the sole new TOS.
        // In the method form (`op_arg & 1`) it pushes `func` then `self` (net
        // -1), so `self` is the new TOS (= last Ref written) and the `func` slot
        // beneath becomes a NONE hole the general hole-fill recovers from the
        // shadow (both pushed through `setarrayitem_vable_r`); like the
        // method-form LOAD_GLOBAL the func slot is consumed by the CALL before
        // any branch guard, so it is never a live kept-stack slot at a resume.
        Instruction::LoadSuperAttr { .. } => VstackOpClass::ResultToTos,

        // SWAP(i): exchange TOS with the box `i` positions below.  A pure
        // permutation (net depth 0); the decoded `i` drives the
        // `vstack_boxes` exchange in `reconcile_vstack_at_boundary`.
        Instruction::Swap { i } => VstackOpClass::Swap(i.get(op_arg) as usize),

        // COPY(i): duplicate the box `i` positions from the top onto the new
        // TOS (net +1).  The decoded `i` drives the duplicate-from-slot copy
        // in `reconcile_vstack_at_boundary` (sources `vstack_boxes[depth-1-i]`,
        // not `vstack_last_ref`, so `COPY i>1` is faithful).
        Instruction::Copy { i } => VstackOpClass::Copy(i.get(op_arg) as usize),

        // Exception machinery inside a handler body.  The unwinder + exc-info
        // operations rewrite the operand stack in ways a producer/pop model
        // cannot express, but every resulting slot is written through
        // `setarrayitem_vable_r`, so the virtualizable shadow is authoritative
        // — `ShadowReseed` reconciles by reseeding from it.  Inert on the
        // non-exception path: these are reached only via the unwind/catch
        // edge, where the mirror is already invalid unless
        // `vstack_enter_exception_handler` re-seeded it at handler entry.
        Instruction::PushExcInfo
        | Instruction::CheckExcMatch
        | Instruction::Reraise { .. }
        | Instruction::RaiseVarargs { .. }
        | Instruction::WithExceptStart => VstackOpClass::ShadowReseed,

        // UNPACK_SEQUENCE / UNPACK_EX: pop one sequence, push its elements
        // (net push > 1).  Every pushed element is in the virtualizable
        // shadow, so reconcile reseeds the pushed range from it.
        Instruction::UnpackSequence { .. } | Instruction::UnpackEx { .. } => {
            VstackOpClass::MultiResultFromShadow
        }

        // FOR_ITER (continue arm): peeks the iterator (kept on the stack) and
        // pushes the yielded item on the new TOS (net +1) — the same shape as
        // the value producers above, so the item is `vstack_last_ref`.  The
        // item never reaches either mirror chokepoint on its own (the
        // `for_iter_next` residual result is stamped via `set_opref_concrete`,
        // not `write_ref_reg`, and the item lands on TOS through the
        // codewriter's `pin!` slot binding, not a `setarrayitem_vable_r`
        // push), so the residual-execution path seeds `vstack_last_ref` with
        // the item OpRef explicitly (the `ForIterNext` capture site).  The
        // exhaustion arm pushes no item, but it is a non-fallthrough guard
        // exit, so the boundary's `sequential` gate suppresses this per-op
        // effect there.
        Instruction::ForIter { .. } => VstackOpClass::ResultToTos,

        // Everything else is not modeled — decline; the overlay then omits
        // the affected slots, which resume re-materializes.  TO_BOOL emits no
        // JitCode (codewriter.rs:8598
        // `Instruction::ToBool => {}`) so it never reaches this classifier;
        // the py-pc boundary mapping skips it.
        _ => VstackOpClass::Unmodeled,
    }
}

/// #73: reconcile the PREVIOUS Python opcode's stack effect into
/// [`WalkContext::vstack_boxes`] at an opcode boundary, BEFORE the new
/// opcode (`new_pypc`) is walked.  Running this before the new op means
/// that when the new op is a branch guard, `vstack_boxes` already holds
/// the correct boxes for the guard's resume depth.
///
/// `code` is the Python `CodeObject` of the outer (full-body) jitcode;
/// `new_pypc` is the Python pc the walk is about to enter; `new_depth` is
/// `depth_at_py_pc[new_pypc]` (stack-only).  The previous opcode is
/// decoded from `code` at `ctx.vstack_cur_pypc`.
///
/// On any unmodeled effect (or a structurally impossible depth) the
/// function latches `ctx.vstack_valid = false` so the `stack_sync`
/// overlay omits every operand slot, which resume re-materializes (zero
/// regression).
pub(crate) fn reconcile_vstack_at_boundary(
    ctx: &mut WalkContext<'_, '_>,
    code: &pyre_interpreter::CodeObject,
    new_pypc: u32,
    new_depth: usize,
) {
    if !ctx.vstack_valid {
        return;
    }
    let prev_pypc = ctx.vstack_cur_pypc as usize;
    let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, prev_pypc) else {
        ctx.vstack_valid = false;
        return;
    };
    let class = classify_vstack_opcode(&instr, op_arg);
    // RPython's MIFrame follows one flow-graph link at a time; the target
    // block receives its register state from that link's inputargs
    // (pyjitpl.py:2371-2387 `interpret` / `run_one_step`).  The full-body
    // walk instead traverses flattened JitCode whose layout can switch to a
    // different source block without the Python PCs forming a CFG edge.  Do
    // not replay the previous Python opcode across such a layout switch: its
    // `vstack_last_ref` belongs to the other block.  Doing so at the two arms
    // of a conditional expression overwrote the loop-carried FOR_ITER slot
    // with the selected local, and a later branch guard serialized that
    // foreign box into the virtualizable resume image.
    //
    // Enter a shadow-reseed region until the walk passes both endpoints of
    // the out-of-order transition.  This subsumes the former SWAP/COPY-only
    // detection for non-monotonic comprehension lowering and applies the
    // same block-input rule to ordinary conditional-expression arms.
    let fallthrough = crate::pyjitpl::semantic_fallthrough_pc(code, prev_pypc);
    use pyre_interpreter::Instruction;
    let has_fallthrough = !matches!(
        instr,
        Instruction::JumpForward { .. }
            | Instruction::JumpBackward { .. }
            | Instruction::JumpBackwardNoInterrupt { .. }
            | Instruction::ReturnValue
            | Instruction::Reraise { .. }
            | Instruction::RaiseVarargs { .. }
    );
    let cfg_successor = (has_fallthrough && new_pypc as usize == fallthrough)
        || crate::liveness::target_pc(code, &instr, prev_pypc, op_arg) == Some(new_pypc as usize);
    if !cfg_successor && ctx.vstack_reorder_ceiling == u32::MAX {
        ctx.vstack_reorder_ceiling = (new_pypc as usize).max(prev_pypc) as u32;
    }
    let in_reorder_region = ctx.vstack_reorder_ceiling != u32::MAX;
    let (fallthrough_depth, branch_depth) =
        crate::liveness::stack_effects(&instr, op_arg, ctx.vstack_depth);
    if std::env::var_os("PYRE_VSTACK_DIAG").is_some() {
        eprintln!(
            "[vstack-reconcile] prev_pypc={prev_pypc} new_pypc={new_pypc} \
             new_depth={new_depth} prev_depth={} class={class:?} reorder={in_reorder_region} \
             last_ref={:?} instr={instr:?}",
            ctx.vstack_depth, ctx.vstack_last_ref
        );
    }
    // A JitCode's block layout can visit source-PC floor segments out of
    // Python bytecode order.  In that case `vstack_cur_pypc` is only the
    // preceding layout segment, not the opcode whose stack effect produced
    // this boundary.  Applying its effect is unsound: a LOAD_FAST segment
    // whose expected depth is `d + 1` can otherwise overwrite a surviving
    // FOR_ITER iterator when the observed depth stayed at `d`.
    //
    // RPython reads the live MIFrame registers and never reconstructs this
    // transition from source PCs.  Preserve the slots that demonstrably
    // survive at the observed depth whenever neither real successor depth
    // matches; holes remain conservative and are handled by the shadow fill
    // below.
    let layout_only_boundary = new_depth != fallthrough_depth && new_depth != branch_depth;

    // PER-OP RECONCILE.  In the SEQUENTIAL case the previous opcode's stack
    // effect explains the depth change: a producer (`ResultToTos`) lands its
    // result box (`vstack_last_ref`) on the new TOS; a pop / side-store just
    // truncates.  This captures the kept boxes from the walk register file
    // (LOAD_FAST / LOAD_NAME / COPY results) — values that may NOT be present
    // in the virtualizable shadow (function-local LOAD_FAST temps live only
    // in the walk register bank, never written through to the portal array).
    // Inside the out-of-order permutation region the per-op replay is invalid;
    // reseed from the shadow (same shape as `ShadowReseed`).
    let effective_class = if in_reorder_region {
        VstackOpClass::ShadowReseed
    } else {
        class
    };
    match effective_class {
        // A layout-only boundary (the observed depth matches neither real
        // successor of the previous opcode) means the per-op effect cannot
        // explain this transition; preserve the surviving slots instead of
        // replaying a stale effect.  The reorder region is already served by
        // the `ShadowReseed` arm, so exclude it here.
        _ if layout_only_boundary && !in_reorder_region => {
            ctx.vstack_boxes.truncate(new_depth);
            if ctx.vstack_boxes.len() < new_depth {
                ctx.vstack_boxes.resize(new_depth, OpRef::NONE);
            }
        }
        VstackOpClass::ResultToTos => {
            ctx.vstack_boxes.truncate(new_depth);
            if ctx.vstack_boxes.len() < new_depth {
                ctx.vstack_boxes.resize(new_depth, OpRef::NONE);
            }
            if new_depth > 0 && ctx.vstack_last_ref != OpRef::NONE {
                ctx.vstack_boxes[new_depth - 1] = ctx.vstack_last_ref;
            }
        }
        VstackOpClass::PopOnlyOrSideStore => {
            ctx.vstack_boxes.truncate(new_depth);
        }
        VstackOpClass::Swap(i) => {
            // SWAP is net-depth-0 (prev_depth == new_depth).  Exchange the
            // TOS box with the box `i` positions below it, matching
            // `swap_values` (`localsplus[depth-1] <-> localsplus[depth-i]`).
            // A NONE in either slot is just permuted (the later hole-fill /
            // legacy-defer handles it); a malformed / out-of-range arg
            // declines (latch invalid).
            ctx.vstack_boxes.truncate(new_depth);
            if ctx.vstack_boxes.len() < new_depth {
                ctx.vstack_boxes.resize(new_depth, OpRef::NONE);
            }
            if new_depth >= 1 && i >= 1 && i <= new_depth {
                let top = new_depth - 1;
                let other = new_depth - i;
                ctx.vstack_boxes.swap(top, other);
            } else {
                ctx.vstack_valid = false;
            }
        }
        VstackOpClass::Copy(i) => {
            // COPY(i): duplicate the box `i` positions from the top onto the
            // new TOS (net +1).  The duplicated box is the COPIED slot
            // `vstack_boxes[new_depth-1-i]` (`opcode_copy_value` =
            // `push(peek_at(i-1))`), sourced directly rather than from
            // `vstack_last_ref` so `COPY i>1` (duplicating a deeper operand)
            // is faithful; `COPY 1` reduces to dup-of-TOS.  A missing source
            // slot or out-of-range arg declines (latch invalid).
            match new_depth.checked_sub(1 + i) {
                Some(src_idx) if i >= 1 && src_idx < ctx.vstack_boxes.len() => {
                    let src = ctx.vstack_boxes[src_idx];
                    ctx.vstack_boxes.truncate(new_depth);
                    if ctx.vstack_boxes.len() < new_depth {
                        ctx.vstack_boxes.resize(new_depth, OpRef::NONE);
                    }
                    ctx.vstack_boxes[new_depth - 1] = src;
                }
                _ => ctx.vstack_valid = false,
            }
        }
        VstackOpClass::ShadowReseed => {
            // Resize to the post-opcode depth, leaving every slot a NONE
            // hole; the shadow-backed hole-fill below sources each slot from
            // the virtualizable shadow the exception lowering just wrote.  An
            // unsourceable slot (genuine NULL exc-info / Int temp) stays NONE
            // and `mirror_covers_kept` declines for it — the conservative
            // fallback, never a corrupt box.
            ctx.vstack_boxes.clear();
            ctx.vstack_boxes.resize(new_depth, OpRef::NONE);
        }
        VstackOpClass::MultiResultFromShadow => {
            // UNPACK_* pops ONE sequence (at `prev_depth - 1`) and pushes its
            // elements upward.  Clear only the affected range
            // `[pop_point .. new_depth)` to NONE so the hole-fill below
            // sources each pushed element from the shadow (all were written
            // through `setarrayitem_vable_r`); slots BELOW the popped sequence
            // keep their mirror-tracked boxes.
            let pop_point = ctx.vstack_depth.saturating_sub(1);
            ctx.vstack_boxes.truncate(new_depth);
            if ctx.vstack_boxes.len() < new_depth {
                ctx.vstack_boxes.resize(new_depth, OpRef::NONE);
            }
            for s in pop_point..new_depth {
                ctx.vstack_boxes[s] = OpRef::NONE;
            }
        }
        VstackOpClass::Unmodeled => {
            ctx.vstack_valid = false;
        }
    }
    // The FBW walk follows jitcode control flow, not just sequential opcodes:
    // an `and`/`or` chain's short-circuit continuation jumps BACKWARD to a
    // deeper merge point, so the previous opcode did NOT produce the slots
    // below the new TOS — the per-op reconcile leaves a NONE hole there.
    // Recover those slots from the virtualizable shadow (kept current by the
    // portal `setarrayitem_vable_r` pushes for values that ARE written
    // through).  `reseed_vstack_from_shadow` rejects a NULL-const shadow slot
    // (a function-local temp the portal never wrote), so a genuinely
    // unrecoverable kept slot fails the re-seed.
    //
    // A non-reseedable hole does NOT latch `vstack_valid = false`: an
    // Int/Float-bank operand-stack temp (e.g. the `while i < N` loop
    // condition's `LoadConst N`, a transient `BINARY_OP` int result) is not
    // a Ref the Ref-only mirror can ever hold, but it is CONSUMED before the
    // all-Ref short-circuit guard region — invalidating the whole mirror
    // there made it die at the loop condition, never reaching the kept-stack
    // guard.  Instead keep the mirror TRACKING (advance position / depth)
    // with the NONE slot left in place; `stack_sync` (USE) omits any NONE
    // mirror slot, which resume re-materializes.
    if ctx.vstack_valid {
        let skip_shadow_fill = ctx.fbw_mode.inline_subwalk && fbw_callee_vstack_enabled();
        let hole = ctx
            .vstack_boxes
            .get(..new_depth)
            .map(|s| s.iter().any(|&b| b == OpRef::NONE))
            .unwrap_or(true);
        if hole && !skip_shadow_fill {
            // Best-effort fill from the shadow; leave un-fillable slots NONE.
            let _ = reseed_vstack_from_shadow(ctx, new_depth);
        }
    }
    if ctx.vstack_valid {
        ctx.vstack_cur_pypc = new_pypc;
        ctx.vstack_depth = new_depth;
        ctx.vstack_last_ref = OpRef::NONE;
    }
    // #389(b): leave the out-of-order permutation region once the walk has
    // advanced PAST the py_pc it backed off from — py order is monotonic again
    // and the per-op reconcile is valid from here.
    if ctx.vstack_reorder_ceiling != u32::MAX && new_pypc > ctx.vstack_reorder_ceiling {
        ctx.vstack_reorder_ceiling = u32::MAX;
    }
}

/// #73: re-seed `ctx.vstack_boxes[0..new_depth]` from the virtualizable
/// shadow's operand-stack slots (`virtualizable_box_at(nvs + nlocals + s)`).
/// Used when a control-flow edge makes the per-opcode reconcile model
/// inapplicable (a backward/forward jump landing at a different stack
/// level).  The portal `pyframe.pushvalue` lowers every Ref push to
/// `setarrayitem_vable_r(locals_cells_stack_w, depth, w_obj)`, so the
/// shadow holds the live operand stack at a merge point.
///
/// Returns `true` on success (every slot `0..new_depth` sourced as a
/// non-NONE box), `false` if any slot is unsourceable (caller then
/// latches `vstack_valid = false`).
pub(crate) fn reseed_vstack_from_shadow(ctx: &mut WalkContext<'_, '_>, new_depth: usize) -> bool {
    let full_body_sym = ctx.fbw_mode.snapshot_sym;
    if full_body_sym.is_null() {
        return false;
    }
    // SAFETY: pointer live for the full-body walk; read-only nlocals.
    let nlocals = unsafe { (*full_body_sym).nlocals };
    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    // Only FILL the NONE holes the per-op reconcile could not source from
    // the walk register file; keep the boxes the reconcile DID capture (the
    // shadow may carry a stale value in a non-hole slot).  A hole the shadow
    // also cannot source (NONE / NULL const-ptr — a function-local temp the
    // portal never wrote through) fails the whole re-seed so the caller
    // leaves the slot NONE; `stack_sync` then omits it (resume
    // re-materializes).
    if ctx.vstack_boxes.len() < new_depth {
        ctx.vstack_boxes.resize(new_depth, OpRef::NONE);
    }
    let mut all_present = true;
    for s in 0..new_depth {
        if ctx.vstack_boxes[s] != OpRef::NONE {
            continue;
        }
        match ctx.trace_ctx.virtualizable_box_at(nvs + nlocals + s) {
            Some(b) if b != OpRef::NONE && !opref_is_null_const_ptr(b) => {
                ctx.vstack_boxes[s] = b;
            }
            // Fill what we can; an unsourceable hole (NONE / NULL const-ptr —
            // an Int/Float-bank temp or a function-local the portal never
            // wrote) stays NONE.  `stack_sync` omits a NONE slot (resume
            // re-materializes), so an unfilled slot is never a corrupt box.
            _ => all_present = false,
        }
    }
    all_present
}

/// #73: map a jitcode pc to the Python opcode whose lowering region
/// CONTAINS it, WITHOUT the `python_pc_for_jitcode_pc` block-head marker
/// special-case.  For the operand-stack mirror we want the containing
/// opcode (where the walk physically is), not the resume block-head a
/// `-live-` marker names — the marker case returns an EARLIER py_pc and
/// makes the mirror's boundary detection oscillate.  Uses only the
/// JitCode-PC floor pivot (largest floor boundary at-or-before `jit_pc`).
pub(crate) fn vstack_containing_py_pc(metadata: &crate::PyJitCodeMetadata, jit_pc: usize) -> u32 {
    if !metadata.py_floor_by_jit_pc.is_empty() {
        return crate::pyjitcode::floor_segment_for_jitcode_pc(
            &metadata.py_floor_by_jit_pc,
            jit_pc,
        )
        .expect("drained JitCode PC floor pivot must begin at byte offset zero")
        .1;
    }
    0
}

pub(crate) fn vstack_initial_py_pc(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
    permuted_for_iter_entry: bool,
) -> u32 {
    if !permuted_for_iter_entry {
        return vstack_containing_py_pc(metadata, jit_pc);
    }
    metadata_block_head_py_pc(metadata, jit_pc)
        .unwrap_or_else(|| vstack_containing_py_pc(metadata, jit_pc))
}

pub(crate) fn metadata_block_head_py_pc(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
) -> Option<u32> {
    metadata
        .block_head_py_by_jit_pc
        .binary_search_by_key(&jit_pc, |&(off, _)| off)
        .ok()
        .map(|i| metadata.block_head_py_by_jit_pc[i].1)
}

pub(crate) fn vstack_step_py_pc(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
    current_py_pc: u32,
) -> u32 {
    if metadata_block_head_py_pc(metadata, jit_pc) == Some(current_py_pc) {
        current_py_pc
    } else {
        vstack_containing_py_pc(metadata, jit_pc)
    }
}

/// #73: step the walk-level operand-stack box mirror at
/// the top of every jitcode `step`.  Detects a Python-opcode boundary by
/// mapping the current `jit_pc` back to its containing Python opcode; when
/// that differs from `ctx.vstack_cur_pypc`, reconciles the previous
/// opcode's stack effect into `vstack_boxes` (see
/// [`reconcile_vstack_at_boundary`]).
///
/// No-op unless the outer full-body sym owns the virtualizable shadow and
/// `vstack_valid` is still set.  Reached only on the full-body walk
/// (`fbw_mode.snapshot_sym` non-null); the per-opcode arm walk leaves the
/// mirror untouched (its guards use the static outer coordinate).  Writes
/// only the `vstack_*` side-fields; never the registers / snapshot.
pub(crate) fn step_vstack_mirror(ctx: &mut WalkContext<'_, '_>, jit_pc: usize) {
    if !ctx.vstack_valid {
        return;
    }
    // On genuine callee sub-walk paths, `jit_pc` is a callee coordinate with
    // no meaning in the outer (`fbw_mode.snapshot_sym`) jitcode's py_pc→jitcode
    // tables.  `inline_subwalk` is also set for the carrier walk of root code,
    // where that premise does not hold and the mirror is simply never seeded.
    // With `PYRE_FBW_CALLEE_VSTACK` off this branch documents intent only:
    // `seed_callee_vstack_mirror` is gated by the same flag.
    let (new_pypc, code_ptr, new_depth) = if ctx.fbw_mode.inline_subwalk {
        if !fbw_callee_vstack_enabled() {
            ctx.vstack_valid = false;
            return;
        }
        let Some(frame) = ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym) else {
            ctx.vstack_valid = false;
            return;
        };
        let Some(coord) = frame.vstack_coordinate_for_jitcode_pc(jit_pc) else {
            ctx.vstack_valid = false;
            return;
        };
        coord
    } else {
        let full_body_sym = ctx.fbw_mode.snapshot_sym;
        if full_body_sym.is_null() {
            return;
        }
        // SAFETY: the pointer is live for the lifetime of the full-body walk
        // (set in `dispatch_via_miframe`); read-only access to immutable
        // layout fields (jitcode / code_ptr / metadata).
        let sym = unsafe { &*full_body_sym };
        if sym.jitcode.is_null() {
            return;
        }
        unsafe {
            let jc = &*sym.jitcode;
            if jc.payload.code_ptr.is_null() {
                return;
            }
            let py_pc = vstack_step_py_pc(&jc.payload.metadata, jit_pc, ctx.vstack_cur_pypc);
            let depth = crate::liveness::liveness_for(jc.payload.code_ptr)
                .depth_at_py_pc()
                .get(py_pc as usize)
                .copied()
                .unwrap_or(0) as usize;
            (py_pc, jc.payload.code_ptr, depth)
        }
    };
    if new_pypc == ctx.vstack_cur_pypc {
        return;
    }
    let code = unsafe { &*code_ptr };
    reconcile_vstack_at_boundary(ctx, code, new_pypc, new_depth);
}

pub(crate) fn seed_callee_vstack_mirror(ctx: &mut WalkContext<'_, '_>, frame: &ActiveResumeFrame) {
    if !fbw_callee_vstack_enabled() {
        return;
    }
    let Some((first_pypc, _code_ptr, _depth)) = frame.vstack_coordinate_for_jitcode_pc(0) else {
        return;
    };
    ctx.vstack_boxes.clear();
    ctx.vstack_depth = 0;
    ctx.vstack_cur_pypc = first_pypc;
    ctx.vstack_last_ref = OpRef::NONE;
    ctx.vstack_valid = true;
}

/// #73: seed the walk-level operand-stack box mirror
/// ([`WalkContext::vstack_boxes`]) at full-body-walk entry.  Enables the
/// mirror (`vstack_valid = true`) only when the outer `sym` owns the
/// virtualizable shadow AND the entry operand stack can be fully sourced
/// from that shadow.  Sets `vstack_cur_pypc = entry_py_pc` and
/// `vstack_depth = depth_at_py_pc[entry_py_pc]`, filling
/// `vstack_boxes[0..depth]` from the virtualizable shadow's operand-stack
/// slots (`virtualizable_box_at(nvs + nlocals + s)`) — the SAME source
/// `collect_outer_active_boxes` / `stack_sync` read.  Any unsourceable
/// slot leaves `vstack_valid = false`; the overlay then omits operand
/// slots, which resume re-materializes (zero regression).
pub(crate) fn seed_vstack_mirror(
    ctx: &mut WalkContext<'_, '_>,
    sym: &crate::state::PyreSym,
    start_pc: usize,
) {
    if sym.jitcode.is_null() || !sym.owns_virtualizable_shadow() {
        return;
    }
    // Ordinarily seed at the opcode containing the first walked jitcode op,
    // so the first step is a no-op and reconciliation starts only after an
    // opcode actually runs. A SWAP/COPY immediately preceding a FOR_ITER
    // block-head marker is different: the shadow already contains the
    // post-permutation stack at trace entry. Seed that marker at FOR_ITER and
    // let `vstack_step_py_pc` ignore the marker itself, preventing the
    // predecessor permutation from being applied to the mirror a second time.
    let (first_pypc, depth, nlocals) = unsafe {
        let jc = &*sym.jitcode;
        if jc.payload.code_ptr.is_null() {
            return;
        }
        let containing_pypc = vstack_containing_py_pc(&jc.payload.metadata, start_pc);
        let predecessor_permuted_stack = pyre_interpreter::decode_instruction_at(
            &*jc.payload.code_ptr,
            containing_pypc as usize,
        )
        .is_some_and(|(instr, op_arg)| {
            matches!(
                classify_vstack_opcode(&instr, op_arg),
                VstackOpClass::Swap(_) | VstackOpClass::Copy(_)
            )
        });
        let target_is_for_iter = metadata_block_head_py_pc(&jc.payload.metadata, start_pc)
            .and_then(|target| {
                pyre_interpreter::decode_instruction_at(&*jc.payload.code_ptr, target as usize)
            })
            .is_some_and(|(instr, _)| {
                matches!(
                    instr,
                    pyre_interpreter::bytecode::Instruction::ForIter { .. }
                )
            });
        let first_pypc = vstack_initial_py_pc(
            &jc.payload.metadata,
            start_pc,
            predecessor_permuted_stack && target_is_for_iter,
        );
        let d = crate::liveness::liveness_for(jc.payload.code_ptr)
            .depth_at_py_pc()
            .get(first_pypc as usize)
            .copied()
            .unwrap_or(0) as usize;
        (first_pypc, d, sym.nlocals)
    };
    let nvs = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    let mut boxes = Vec::with_capacity(depth);
    for s in 0..depth {
        // Read the operand-stack slot `s` from the virtualizable shadow.
        // A missing / NONE slot means the entry stack box is not
        // reconstructible here — decline the whole mirror for this walk.
        match ctx.trace_ctx.virtualizable_box_at(nvs + nlocals + s) {
            Some(b) if b != OpRef::NONE => boxes.push(b),
            _ => return,
        }
    }
    ctx.vstack_boxes = boxes;
    ctx.vstack_depth = depth;
    ctx.vstack_cur_pypc = first_pypc;
    ctx.vstack_last_ref = OpRef::NONE;
    ctx.vstack_valid = true;
}

/// #370: model the exception-unwind boundary on the operand-stack mirror.
/// When a raised exception is caught by THIS frame's handler (the SubRaise
/// catch in the dispatch loop), the unwinder truncates the operand stack to
/// the handler's setup depth and pushes the exception value.
/// [`reconcile_vstack_at_boundary`] cannot model this NON-SEQUENTIAL
/// transition — it explains a depth change via the previous opcode's normal
/// stack effect — so without this hook the mirror latches `vstack_valid =
/// false` at handler entry and every kept-stack guard inside the handler
/// declines.  Re-seed the mirror at the handler-entry coordinate instead:
/// place the caught `exc` box on the new TOS and source the surviving slots
/// below it from the virtualizable shadow (the unwind only truncates ABOVE
/// the handler depth, so those slots are unchanged at the raise point).
/// Subsequent in-handler exception opcodes reconcile via
/// [`VstackOpClass::ShadowReseed`], re-reading the shadow the lowering keeps
/// current.  A survivor slot the shadow cannot source stays a NONE hole and
/// the guard's `mirror_covers_kept` declines for it (safe fallback).
///
/// `handler_jit_pc` is the catch target (an OUTER full-body jitcode pc).
/// No-op outside the full-body walk or inside an inline sub-walk (where the
/// pc is a callee coordinate the outer metadata cannot map).
pub(crate) fn vstack_enter_exception_handler(
    ctx: &mut WalkContext<'_, '_>,
    handler_jit_pc: usize,
    exc: OpRef,
) {
    // The handler-entry operand stack is a FRESH reconstruction from the
    // authoritative virtualizable shadow plus the caught `exc` — it does NOT
    // depend on the pre-raise mirror being valid (the unwind discards the
    // operand stack above the handler depth, and the surviving slots below
    // are read from the shadow).  So REVIVE the mirror here even when the
    // pre-raise walk invalidated it (e.g. at a `LOAD_GLOBAL` NULL-sentinel on
    // the `raise` expression).  Only an inline sub-walk (callee coordinate)
    // or a missing full-body sym is unrecoverable.
    if ctx.fbw_mode.inline_subwalk {
        ctx.vstack_valid = false;
        return;
    }
    let full_body_sym = ctx.fbw_mode.snapshot_sym;
    if full_body_sym.is_null() {
        return;
    }
    // SAFETY: pointer live for the full-body walk; read-only layout fields.
    let sym = unsafe { &*full_body_sym };
    if sym.jitcode.is_null() {
        ctx.vstack_valid = false;
        return;
    }
    let (handler_py, code_ptr) = unsafe {
        let jc = &*sym.jitcode;
        if jc.payload.code_ptr.is_null() {
            ctx.vstack_valid = false;
            return;
        }
        (
            vstack_containing_py_pc(&jc.payload.metadata, handler_jit_pc),
            jc.payload.code_ptr,
        )
    };
    let handler_depth = crate::liveness::liveness_for(code_ptr)
        .depth_at_py_pc()
        .get(handler_py as usize)
        .copied()
        .unwrap_or(0) as usize;
    ctx.vstack_boxes.clear();
    ctx.vstack_boxes.resize(handler_depth, OpRef::NONE);
    // The unwinder pushes the caught exception onto the new TOS.
    if handler_depth >= 1 && exc != OpRef::NONE {
        ctx.vstack_boxes[handler_depth - 1] = exc;
    }
    ctx.vstack_cur_pypc = handler_py;
    ctx.vstack_depth = handler_depth;
    ctx.vstack_last_ref = OpRef::NONE;
    // Revive: the handler-entry state is shadow-sourced, independent of the
    // pre-raise mirror.
    ctx.vstack_valid = true;
    // Fill the surviving slots below the pushed exc from the shadow; reseed
    // skips the already-set exc slot (non-NONE).  Leaves un-sourceable slots
    // NONE (per-slot decline) rather than latching the whole mirror invalid.
    let _ = reseed_vstack_from_shadow(ctx, handler_depth);
}
