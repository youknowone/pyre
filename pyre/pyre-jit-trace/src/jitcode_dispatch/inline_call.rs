//! Inline-call tracing: resolving a user-defined callee and walking its
//! body inline instead of recording a residual call.
//!
//! **Parity:** trace-side counterpart of `pyjitpl.py`'s
//! `opimpl_inline_call_*` / `opimpl_recursive_call` and the `MetaInterp`
//! inline-callee machinery. PyPy keeps these inside `pyjitpl.py`; the split
//! into this file is pyre-local navigability, not a PyPy file boundary.
//!
//! Relocated verbatim from `jitcode_dispatch/mod.rs`. Covers callee
//! recognition and inlinability checks, active-box / call-stack
//! reconstruction, callee register-bank allocation, the sub-jitcode walk
//! driver, the self-recursive `CALL_ASSEMBLER` fold, the inline user-call /
//! binop / compareop / exception-string specializers, and the
//! `dispatch_inline_call_*` per-shape dispatchers. The `inline_call_*`
//! opname arms stay in `handle` (mod.rs) and call into these.

use super::*;

/// Path-1 (#68): resolve a scalar `getfield_vable_r` read off an inlined
/// callee's OWN (unseeded) portal frame to the callee's compile-time
/// constant.  This is the walk-time mirror of the codewriter's non-portal
/// branch (`codewriter.rs` LOAD_CONST/LOAD_GLOBAL):
/// a non-portal callee's `pycode`/`w_globals` are constants fed as
/// `ConstRef`, never read off the portal frame reg (which, when inlined,
/// aliases the caller's frame and would read the wrong field).  Only the
/// Ref-typed `pycode` (field 1) and `w_globals` (field 5) carry a
/// compile-time constant; Int frame state (`last_instr`, `valuestackdepth`)
/// does not.  Returns `None` when not an inline sub-walk, the field is not
/// resolvable, or the layout is absent — callers fall through to the
/// `VableBoxNotSeeded` error (such callees are declined up-front by
/// [`callee_fast_path_inlinable`], so reaching here unresolved is genuine).
pub(crate) fn try_resolve_inline_callee_static_field<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if dst_bank != 'r' {
        return Ok(None);
    }
    let Some(consts) = ctx.inline_callee_consts else {
        return Ok(None);
    };
    let descr = read_descr(code, op, 1, ctx)?;
    let field_idx = {
        let Some(info) = ctx.trace_ctx.virtualizable_info() else {
            return Ok(None);
        };
        match info.static_field_by_descr(&descr) {
            Some(idx) => idx,
            None => return Ok(None),
        }
    };
    let const_ptr = match field_idx {
        VABLE_NAMESPACE_FIELD_IDX => consts.w_globals,
        VABLE_CODE_FIELD_IDX => consts.w_code,
        _ => return Ok(None),
    };
    let result = ctx.trace_ctx.const_ref(const_ptr as i64);
    let dst = code[op.pc + 4] as usize;
    write_vable_field_ref_reg(
        ctx,
        op.pc,
        dst,
        result,
        ConcreteValue::Ref(const_ptr as pyre_object::PyObjectRef),
    )?;
    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

#[allow(non_snake_case)]
/// #62 call-inlining recognition probe (env-gated `PYRE_DIAG_INLINE_RECOG`,
/// ZERO behavior change — emits only diagnostics).  A user-function `CALL`
/// lowers to a `call_fn` residual whose `funcptr` is the generic call helper,
/// not the callee; the actual callable is a runtime Ref arg.  This scans the
/// Ref args' concrete values for a user Python function (`FUNCTION_TYPE`,
/// non-builtin) and reports whether its per-`CodeObject` JitCode is installed
/// (`jitcode_lookup`).  It confirms the runtime callable -> `CodeObject` ->
/// JitCode recognition seam fires before any inline sub-walk wiring lands.
pub(crate) fn diagnose_inline_recognition(arg_concretes: &[ConcreteValue], op_pc: usize) {
    let function_type_addr = &pyre_interpreter::FUNCTION_TYPE as *const _ as usize;
    // Single-letter kind tag per arg, so the call_fn arg layout (which slot
    // holds the callable vs the positional args) can be read off empirically
    // without touching CodeObject internals.
    let shape: String = arg_concretes
        .iter()
        .map(|cv| match cv {
            ConcreteValue::Int(_) => 'i',
            ConcreteValue::Float(_) => 'f',
            ConcreteValue::Bool(_) => 'b',
            ConcreteValue::Ref(_) => 'r',
            ConcreteValue::Null => '_',
        })
        .collect();
    for (i, cv) in arg_concretes.iter().enumerate() {
        let ConcreteValue::Ref(obj) = *cv else {
            continue;
        };
        if obj.is_null() {
            continue;
        }
        unsafe {
            if !pyre_interpreter::is_function(obj) {
                continue;
            }
            // Only the pure-Python `function` type is a sub-walk candidate;
            // builtins share `is_function` but have no per-fn JitCode.
            if (*obj).ob_type as *const () as usize != function_type_addr {
                continue;
            }
            let code = pyre_interpreter::function_get_code(obj);
            // Exercise the slice-(3) obtain step: build (if needed) + view the
            // callee per-fn JitCode as a SubJitCodeBody.  Reports the callee
            // register-bank shape the sub-walk will allocate.
            match crate::state::sub_jitcode_body_for_code(code) {
                Some(body) => eprintln!(
                    "[inline-recog] pc={op_pc} nargs={} shape=[{shape}] callable@{i} \
                     code={code:?} body-OK regs_r={} regs_i={} regs_f={} code_len={}",
                    arg_concretes.len(),
                    body.num_regs_r,
                    body.num_regs_i,
                    body.num_regs_f,
                    body.code.len()
                ),
                None => eprintln!(
                    "[inline-recog] pc={op_pc} nargs={} shape=[{shape}] callable@{i} \
                     code={code:?} body-NONE",
                    arg_concretes.len()
                ),
            }
        }
    }
}

/// The FBW fast-path inline convention (`try_walker_inline_user_call`) seeds
/// only the callee's positional-argument registers `r0..nparams`; the
/// callee's virtualizable frame box is left unseeded.  A callee whose body
/// reads or writes that frame through a `*_vable_*` op — emitted by the
/// codewriter when a local must survive a sub-call — generally cannot be
/// satisfied by register seeding and would abort the *whole* enclosing trace
/// with `VableBoxNotSeeded`.  The ONE exception is a scalar `getfield_vable_r`
/// reading a compile-time-constant static field (`pycode` / `w_globals`):
/// [`try_resolve_inline_callee_static_field`] folds it to the callee constant
/// (the walk-time mirror of the codewriter non-portal branch,
/// `codewriter.rs`).  Detect everything else
/// pre-flight so the call lowers to an ordinary residual call (the orthodox
/// non-inlinable path, `should_inline` = False → `do_residual_call`,
/// `pyjitpl.py`) instead of aborting.
///
/// Also decline callees that are not *straight-line leaves*.  The inline
/// convention resumes a guard inside the callee at the caller's CALL boundary
/// via the inherited single-frame snapshot — sound only when re-executing the
/// whole call on deopt reproduces the state ([`try_walker_inline_user_call`]
/// docstring).  A callee with an internal conditional branch (`goto_if_not` /
/// `switch`) emits a branch guard whose fail snapshot needs to resume *into*
/// the callee mid-body; the single-frame model then serialises a resume
/// section whose liveness shape disagrees with the encoded stream (a folded
/// branch operand is numbered `TAGINT` in a slot the outer liveness reports as
/// a ref → `resume.rs decode_ref: unexpected tag`).  Until the multi-frame
/// resume coordinate is ported (#68), only branchless leaves are inlinable;
/// a branchy callee lowers to an ordinary residual call (correct).
pub(crate) fn callee_fast_path_inlinable<Sym: WalkSym>(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
    ctx: &WalkContext<'_, '_, Sym>,
    callee_frame_reg: u16,
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            // Undecodable tail — be conservative and decline the fast path.
            return false;
        };
        if d.opname.starts_with("goto_if_not") || d.opname.starts_with("switch") {
            if std::env::var_os("PYRE_FBW_STRICT_DIAG").is_some() {
                eprintln!("[strict-reject] pc={} op={} (branch)", d.pc, d.opname);
            }
            return false;
        }
        // A vable op is inlinable on the strict straight-line path when it is
        // either a static const-field read (`pycode` / `w_globals`, resolved
        // frame-free) OR a read/write off the callee's OWN portal frame
        // register — the latter is the `fresh_virtualizable` case, folded
        // register-to-register through the per-slot OpRef shadow by the two
        // `*_vable_via_metainterp` short-circuits (no GC op emitted).  Under
        // the unconditional Portal input shape, LOAD_FAST / STORE_FAST lower to
        // `getarrayitem_vable_r` / `setarrayitem_vable_r(frame, slot)`, so a
        // branchless leaf's locals prologue must not decline the fast path.
        // `callee_frame_reg == u16::MAX` for shapeless skeletons makes
        // `inline_resolvable_seeded_frame_op` return false, so this is inert
        // outside drained per-code jitcodes.
        if d.opname.contains("vable")
            && !inline_resolvable_static_vable_read(body_code, &d, callee_descr_refs, ctx)
            && !inline_resolvable_seeded_frame_op(body_code, &d, callee_frame_reg)
        {
            if std::env::var_os("PYRE_FBW_STRICT_DIAG").is_some() {
                eprintln!(
                    "[strict-reject] pc={} op={} (non-static vable)",
                    d.pc, d.opname
                );
            }
            return false;
        }
        pc = d.next_pc;
    }
    true
}

/// True iff `d` is a scalar `getfield_vable_r` whose field is a Ref-typed
/// compile-time constant (`pycode` / `w_globals`) — the only vable op
/// [`try_resolve_inline_callee_static_field`] can satisfy without a seeded
/// frame box.  `setfield_vable`, array vable ops, and `getfield_vable_i/f`
/// (mutable Int/Float frame state) all return false → decline the inline.
pub(crate) fn inline_resolvable_static_vable_read<Sym: WalkSym>(
    body_code: &[u8],
    d: &DecodedOp,
    callee_descr_refs: &[DescrRef],
    ctx: &WalkContext<'_, '_, Sym>,
) -> bool {
    if !d.opname.starts_with("getfield_vable_r") {
        return false;
    }
    let Some(info) = ctx.trace_ctx.virtualizable_info() else {
        return false;
    };
    // `rd>r` layout: 1B reg + 2B descr-pool index + 1B dst; descr at `pc + 2`.
    if d.pc + 3 >= body_code.len() {
        return false;
    }
    let descr_index = body_code[d.pc + 2] as usize | ((body_code[d.pc + 3] as usize) << 8);
    let Some(descr) = callee_descr_refs.get(descr_index) else {
        return false;
    };
    matches!(
        info.static_field_by_descr(descr),
        Some(VABLE_CODE_FIELD_IDX) | Some(VABLE_NAMESPACE_FIELD_IDX)
    )
}

/// Relaxed variant of [`callee_fast_path_inlinable`] for the multi-frame
/// inline path (#68, `PYRE_FBW_INLINE_MULTIFRAME`): a FORWARD `goto_if_not`
/// (branch target ahead of the branch op) is now inlinable because its
/// in-callee guard resumes through a multi-frame snapshot
/// ([`walker_capture_snapshot_for_last_guard_impl`]'s parent-frame branch).
/// A BACKWARD `goto_if_not` (a loop back-edge) and any `switch` still decline:
/// a loop in the callee needs a `jit_merge_point` the inline snapshot does
/// not model, and a multi-target switch is not yet handled.
///
/// Vable reads are accepted in two cases: (a) a scalar static-field read
/// (`pycode` / `w_globals`) resolvable without a seeded frame
/// (`inline_resolvable_static_vable_read`), or (b) a frame-LOCAL read
/// (`getfield_vable_r` / `getarrayitem_vable_r`) whose base register equals the
/// seeded callee frame reg `callee_frame_reg` — the multiframe path seeds that
/// frame as a virtual object graph (`emit_new_pyframe_inline_with_params`), so
/// the optimizer folds the read to the seeded param value rather than aborting
/// `VableBoxNotSeeded`.  Every `setfield_vable_*` / `setarrayitem_vable_*`
/// (a write into a vable, which would escape the virtual frame) and any vable
/// op against a base reg OTHER than the seeded frame still decline.
pub(crate) fn callee_fast_path_inlinable_allowing_forward_branch<Sym: WalkSym>(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
    ctx: &WalkContext<'_, '_, Sym>,
    callee_frame_reg: u16,
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if d.opname.starts_with("switch") {
            return false;
        }
        if d.opname.starts_with("goto_if_not") {
            // `iL`: 2B LE label at operand offset 1 (after the 1B Int reg).
            let target = read_label(body_code, &d, 1);
            if target <= d.pc {
                return false;
            }
        }
        if d.opname.contains("vable")
            && !inline_resolvable_static_vable_read(body_code, &d, callee_descr_refs, ctx)
            && !inline_resolvable_seeded_frame_op(body_code, &d, callee_frame_reg)
        {
            if std::env::var_os("PYRE_FBW_STRICT_DIAG").is_some() {
                eprintln!(
                    "[strict-reject-mf] pc={} op={} base_reg={:?} frame_reg={callee_frame_reg} \
                     (non-static, foreign vable)",
                    d.pc,
                    d.opname,
                    body_code.get(d.pc + 1).copied()
                );
            }
            return false;
        }
        pc = d.next_pc;
    }
    true
}

/// True iff `d` is a frame-LOCAL vable op — a `getfield_vable` /
/// `getarrayitem_vable` read OR a `setfield_vable` / `setarrayitem_vable` write
/// — whose base register byte (`body_code[d.pc + 1]`, the first operand for
/// every `rX...` / `riX...` vable layout) equals the seeded callee frame
/// register.  The multiframe inline seeds this frame as a VIRTUAL `PyFrame`
/// (`emit_new_pyframe_inline_with_params`) whose locals array holds the param
/// boxes; the post-trace optimizer keeps the frame + its array virtual, folding
/// reads to the seeded/stored value (`optimize_getfield_gc` /
/// `optimize_getarrayitem_gc`) and recording writes into `vinfo.items` /
/// `vinfo.fields` WITHOUT forcing (`optimize_setarrayitem_gc` /
/// `optimize_setfield_gc`).  A store INTO the callee's own virtual frame
/// (param-init `STORE_FAST` prologue, intermediate local writes) is therefore
/// foldable too — only a vable op against a DIFFERENT base reg (a genuinely
/// foreign vable, e.g. the caller's frame the seed does not own) escapes the
/// fold and must decline.  An int/float-base vable op (the `iid`/`ird`
/// intbase set variants) is not a frame-local store and also declines.
pub(crate) fn inline_resolvable_seeded_frame_op(
    body_code: &[u8],
    d: &DecodedOp,
    callee_frame_reg: u16,
) -> bool {
    if callee_frame_reg == u16::MAX || callee_frame_reg > u8::MAX as u16 {
        return false;
    }
    // Only ref-base vable ops (`getfield_vable_r/rd>r`,
    // `getarrayitem_vable_r/ridd>r`, `setfield_vable_*/rXd`,
    // `setarrayitem_vable_*/riXdd`) carry the frame ref in operand 0.  The
    // intbase set variants (`setfield_vable_*/iXd`) take an Int base — not the
    // seeded ref frame — so reject them.
    let is_frame_vable = d.opname.starts_with("getfield_vable_r")
        || d.opname.starts_with("getarrayitem_vable_r")
        || (d.opname.starts_with("setfield_vable") && d.argcodes.starts_with('r'))
        || (d.opname.starts_with("setarrayitem_vable") && d.argcodes.starts_with('r'));
    if !is_frame_vable {
        return false;
    }
    match body_code.get(d.pc + 1).copied() {
        Some(base) => base as u16 == callee_frame_reg,
        None => false,
    }
}

pub(crate) fn method_form_callee_body_supported(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if residual_call_helper_kind_in_body(body_code, &d, callee_descr_refs)
            == Some(majit_ir::PyreHelperKind::LoadAttr)
        {
            return false;
        }
        pc = d.next_pc;
    }
    true
}

/// Whether sampling an exception string override before recording can have no
/// app-visible effect. Portal-frame vable traffic and constant/int boxing are
/// local; branches, other calls, and live-heap writes decline the sample.
pub(crate) fn exception_string_override_sample_safe(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if d.opname.starts_with("goto_if_not") || d.opname.starts_with("switch") {
            return false;
        }
        if d.opname.starts_with("residual_call") {
            let kind = residual_call_helper_kind_in_body(body_code, &d, callee_descr_refs);
            if !matches!(
                kind,
                Some(majit_ir::PyreHelperKind::LoadConst | majit_ir::PyreHelperKind::BoxInt)
            ) {
                return false;
            }
        } else if d.opname.starts_with("setfield_gc")
            || d.opname.starts_with("setarrayitem_gc")
            || d.opname.starts_with("setinteriorfield_gc")
            || d.opname.starts_with("raw_store")
            || d.opname.starts_with("cond_call")
            || d.opname.starts_with("call_assembler")
            || d.opname.starts_with("inline_call")
        {
            return false;
        }
        pc = d.next_pc;
    }
    true
}

/// The bounded builtin-dispatch route only admits a straight-line app-level
/// override. A control-flow-bearing method stays on the original residual
/// dispatch path, where the interpreter owns its frame and branch semantics.
pub(crate) fn exception_string_override_straight_line(body_code: &[u8]) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if d.opname.starts_with("goto_if_not") || d.opname.starts_with("switch") {
            return false;
        }
        pc = d.next_pc;
    }
    true
}

/// Whether an exception string-override body issues a nested Python call.  The
/// bounded string-override route inlines the override as a leaf; a nested call
/// (`CallFn` residual, or a `cond_call`/`call_assembler`/`inline_call`) forces a
/// multi-frame guard-resume snapshot the sub-walk cannot build, aborting
/// mid-recording (`LoopBearingCalleeInlineUnsupported`) and discarding the whole
/// loop.  Such a body must stay on the residual dispatch path.
pub(crate) fn exception_string_override_has_nested_call(
    body_code: &[u8],
    callee_descr_refs: &[DescrRef],
) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return true;
        };
        if d.opname.starts_with("cond_call")
            || d.opname.starts_with("call_assembler")
            || d.opname.starts_with("inline_call")
        {
            return true;
        }
        if d.opname.starts_with("residual_call")
            && residual_call_helper_kind_in_body(body_code, &d, callee_descr_refs)
                == Some(majit_ir::PyreHelperKind::CallFn)
        {
            return true;
        }
        pc = d.next_pc;
    }
    false
}

/// Active boxes for an inlined callee's OWN frame in a multi-frame snapshot
/// (#68).  The fast-path inline predicate guarantees the callee does not own a
/// virtualizable (any vable op declines the inline), so the owns_vable /
/// portal-reg / semantic-slot machinery in [`collect_outer_active_boxes`]
/// reduces to a plain per-bank `registers_{i,r,f}[live_color]` read — RPython
/// `pyjitpl.py _get_list_of_active_boxes`, banks in int → ref → float
/// order to match the `all_liveness` header layout the decoder consumes.  A
/// liveness-active register holding `OpRef::NONE` is a tracer-side invariant
/// violation (callee banks are sized to the jitcode num_regs co-published with
/// liveness), so panic loudly rather than bleed NONE into the encoder.
/// Build the inlined callee (top/innermost) snapshot frame's live box list
/// from the sub-walk register banks at the guard's carried resume coordinate.
///
/// Unlike [`collect_outer_active_boxes`], the callee sub-walk is sym-less and
/// owns no virtualizable, so none of the vable-shadow / portal-red / #124
/// kept-stack recovery applies — every live color must be present directly in
/// the sub-walk's `registers_*`.  A liveness-active color the sub-walk never
/// wrote (`OpRef::NONE` — e.g. a static-ref operand-stack slot that trace-time
/// int-specialization left only in the int bank, or a py_pc↔jit_pc round-trip
/// landing on a different liveness window) cannot be sourced, so return `Err`
/// to abort the multi-frame inline and interpret rather than
/// encode a NONE box.  `PYRE_FBW_MF_DIAG` dumps the missing color.
pub(crate) fn collect_callee_active_boxes(
    regs_i: &[OpRef],
    regs_r: &[OpRef],
    regs_f: &[OpRef],
    callee_jitcode_index: u32,
    callee_op_pc: usize,
    carried_jitcode_pc: i32,
) -> Result<Vec<OpRef>, DispatchError> {
    // Without a carried coordinate there is no `-live-` window to size this
    // frame's box section from, and the decoder would resume on one anyway;
    // decline the inline rather than encode against an empty window.
    if carried_jitcode_pc == majit_ir::resumedata::NO_JITCODE_PC {
        return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: callee_op_pc });
    }
    // The resume decoder consumes this frame's section per the liveness at the
    // carried `jitcode_pc` (`setposition` → `get_current_position_info`), not
    // a Python-pc translation. Query the same carried coordinate so the
    // encoder's box bank set agrees with the decoder's section sizes. A
    // mismatched window let a callee that int-specializes a param encode a Ref
    // where the decoder expects an int → `getvirtual_int: not a raw virtual`.
    let banks = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
        callee_jitcode_index as i32,
        carried_jitcode_pc,
    );
    let mut active = Vec::with_capacity(banks.int.len() + banks.ref_.len() + banks.float.len());
    let diag = std::env::var_os("PYRE_FBW_MF_DIAG").is_some();
    let read = |bank: &[OpRef], idx: u32, name: &str| -> Result<OpRef, DispatchError> {
        match bank.get(idx as usize).copied() {
            Some(v) if v != OpRef::NONE => Ok(v),
            other => {
                if diag {
                    eprintln!(
                        "[fbw-mf-diag] decline: callee {name} reg {idx} {} \
                         (callee_jitcode_index={callee_jitcode_index}, \
                         bank_len={}, live_i={:?} live_r={:?} live_f={:?})",
                        if other.is_none() {
                            "out-of-range"
                        } else {
                            "holds OpRef::NONE"
                        },
                        bank.len(),
                        banks.int,
                        banks.ref_,
                        banks.float,
                    );
                }
                Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: callee_op_pc })
            }
        }
    };
    for &idx in &banks.int {
        active.push(read(regs_i, idx, "int")?);
    }
    for &idx in &banks.ref_ {
        active.push(read(regs_r, idx, "ref")?);
    }
    for &idx in &banks.float {
        active.push(read(regs_f, idx, "float")?);
    }
    Ok(active)
}

/// #62: full-body-walk direct `CALL_ASSEMBLER` for a self-recursive call
/// at the inline recursion-bound boundary (dev-gated `PYRE_FBW_REC_CA`).
///
/// When the FBW inline depth for a callee reaches `FBW_MAX_INLINE_RECURSION`
/// the call would otherwise degrade to a generic may-force residual, which
/// re-enters the callee through the func-entry residency door — one
/// heavyweight frame build + entry-bridge per recursive call (the
/// `fib_recursive` ~30x slowdown).  This emits instead the direct
/// assembler->assembler jump: `CALL_ASSEMBLER_R` to the callee's own
/// loop/pending token (mirror of `_opimpl_recursive_call`
/// `recursion_exceeded -> assembler_call`, `pyjitpl.py`, and
/// `do_residual_call`'s assembler branch, `pyjitpl.py`).
///
/// First cut — the `fib` shape only: a single positional INT argument to a
/// self-recursive (`callee code == portal code`) callee whose frame is
/// `ncells == 0`, non-global-storing, and inline-buildable via
/// [`crate::helpers::emit_new_pyframe_inline_self_recursive`]
/// (Branch A of the retired trait-side callee-frame path).  Any unmet
/// precondition returns `Ok(None)` *before* recording any IR, so the call
/// falls back to the proven (slow) residual path.  No callable-identity
/// guard is emitted: matching the trait's self-recursive arm, the function
/// identity is pinned upstream by the same `LOAD_GLOBAL` machinery the
/// residual path already relies on.
///
/// Parity note: upstream `_opimpl_recursive_call` (`pyjitpl.py`)
/// counts same-greenkey portal frames on the framestack and flips to
/// `assembler_call` only at `count >= memmgr.max_unroll_recursion`,
/// inlining (`perform_call`) below the bound.  This function fires for
/// the FIRST self-recursive occurrence the inline path declines — there
/// is no unroll count.  Value-correct (the callee runs as its own
/// compiled loop either way), but recursion shallower than
/// `max_unroll_recursion` that upstream would have unrolled in-trace is
/// cut over to `CALL_ASSEMBLER` immediately here.
pub(crate) fn try_walker_call_assembler_self_recursive<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    pyre_helper: majit_ir::PyreHelperKind,
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    // ---- non-emitting eligibility checks (free to bail with Ok(None)) ----
    // Default ON since the Phase 5 flip; `PYRE_FBW_REC_CA=0` opts out.
    // Authoritative walks only: the CALL_ASSEMBLER record + walk-commit
    // bookkeeping is FBW machinery; a non-authoritative context (the
    // diagnostic probe, tests) records the plain residual instead.
    if !ctx.is_authoritative_executor
        || std::env::var_os("PYRE_FBW_REC_CA").as_deref() == Some(std::ffi::OsStr::new("0"))
    {
        return Ok(None);
    }
    // Only a genuine `call_fn` residual is a candidate — every
    // container/builtin helper carries a distinct tag.
    if pyre_helper != majit_ir::PyreHelperKind::CallFn {
        return Ok(None);
    }
    // Positional args only (`r_args = [callable, null_or_self, arg0, ..]`);
    // Ref dst only (`residual_call_r_r`, the boxed PyObject consumed by a
    // following BINARY_OP).  The only `residual_call_r_i` helper is the
    // 1-arg `truth_fn`, so an Int dst is structurally unreachable here —
    // don't accept one.  At least one positional argument is required.
    if dst_bank != 'r' || r_args.len() < 3 {
        return Ok(None);
    }
    // A self-recursive CALL_ASSEMBLER raising inside a `try` body must
    // route its GUARD_NO_EXCEPTION deopt into the handler.  The concrete
    // CALL_ASSEMBLER fold here cannot encode that resume in its snapshot;
    // decline so the body takes the generic residual path, which walks the
    // handler-bearing body and resumes the deopt into the handler.
    if jitcode_has_exception_handler(code) {
        return Ok(None);
    }
    let arg_concretes = read_ref_var_list_concrete(code, op, 1, ctx);
    let ConcreteValue::Ref(callable) = arg_concretes[0] else {
        return Ok(None);
    };
    if callable.is_null() {
        return Ok(None);
    }
    // Plain-call shape only: a non-null `null_or_self` is a method
    // receiver `bh_call_fn_impl` would prepend as arg0; an unknown
    // concrete cannot be proven plain.  Either way, decline to the
    // residual call.
    let ConcreteValue::Ref(null_or_self) = arg_concretes[1] else {
        return Ok(None);
    };
    if !null_or_self.is_null() {
        return Ok(None);
    }
    // The callable must be a plain Python function with N positional
    // parameters and no closure.  Unlike the inline path this does NOT
    // require a leaf body: the callee runs as its own compiled loop reached
    // through `CALL_ASSEMBLER`, not traced through — so a branchy
    // self-recursive body (`fib`'s `if n < 2`) is eligible here even though
    // `callee_fast_path_inlinable` declines it for inlining.
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(callable) })
    else {
        return Ok(None);
    };
    // Dense positional fill only: the call must pass exactly `nparams`
    // positional args (`r_args = [callable, null_or_self, arg0..arg{n-1}]`),
    // so the built frame's `locals[0..nparams]` come straight from the args.
    // A default/vararg/kwarg mismatch would leave a hole the frame build
    // cannot fill — decline to the residual.
    if has_closure || nparams < 1 || nparams != r_args.len() - 2 || nparams > FBW_REC_CA_MAX_PARAMS
    {
        return Ok(None);
    }
    // Every positional argument must be an exact boxed int at trace time
    // (`concrete_arg is_int`): the callee was traced against int locals whose
    // speculative low-bit guard would deopt on a non-int box.  `is_int` also
    // accepts `bool`, whose payload reads through a different accessor than the
    // int one the unbox below uses, so a `bool` argument must decline too.  A
    // non-int (or bool) argument declines to the residual call.
    for i in 0..nparams {
        let ConcreteValue::Ref(arg_obj) = arg_concretes[2 + i] else {
            return Ok(None);
        };
        if arg_obj.is_null()
            || !unsafe { pyre_object::is_int(arg_obj) }
            || unsafe { pyre_object::is_bool(arg_obj) }
        {
            return Ok(None);
        }
    }
    // Loopless self-call shape only: the operand stack below the call's own
    // operands (`r_args = [callable, null_or_self, arg0..]`) must hold no
    // loop-carried input arg.  A self-call inside a `for`/`while` body keeps
    // the loop's InputArg operands (the `FOR_ITER` iterator, an accumulator
    // reloaded for `+=`) on the caller stack under the call; the concrete
    // CALL_ASSEMBLER fold cannot carry them across the assembler call, so the
    // loop-back-edge guard resumes the loop-carried iterator as NULL and the
    // blackhole faults on the next `FOR_ITER`.  The residual path keeps those
    // operands live, so decline the loop-bearing shape to it.  The loopless
    // `fib` shape keeps only within-iteration temps (a prior call result), no
    // InputArg, and stays foldable.
    if ctx.vstack_valid {
        let kept_below = ctx.vstack_boxes.len().saturating_sub(r_args.len());
        if ctx.vstack_boxes[..kept_below]
            .iter()
            .any(|slot| slot.is_input_arg())
        {
            return Ok(None);
        }
    }
    // The outer portal sym (the only materialized frame across sub-walks)
    // via the FBW thread-local — the same read mechanism
    // `walker_capture_snapshot_for_last_guard` uses.  Null outside a
    // production full-body walk (arm/shadow/diagnostic), in which case the
    // sym.frame / sym.execution_context reds are unavailable: bail.
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    let sym = unsafe { &*sym_ptr };
    let caller_frame = sym.frame();
    // `is_self_recursive = callee code == portal code`. During
    // recording `we_are_jitted()` is false, so `function_get_code` (the
    // `w_code` already in hand) equals `getcode` — the pointer the
    // jit_merge_point green key and the portal jitcode were registered
    // under.
    let caller_code = unsafe {
        pyre_interpreter::live_code_wrapper((*sym.jitcode()).raw_code() as *const ()) as *const ()
    };
    // Self-fold requires callee code == portal code.  The full-portal cutover
    // (`PYRE_FBW_REC_MUTUAL_CUTOVER`) additionally admits a *mutual*-recursive
    // callee — one whose code is already on the inline framestack, i.e. a
    // genuine recursion cycle (`is_even` → `is_odd` → `is_even` at the unroll
    // cap).  It must NOT admit an arbitrary foreign call: folding a
    // non-recursive callee (e.g. a CALL_KW-bearing leaf) to CALL_ASSEMBLER
    // builds and enters a frame the callee's own loop was never traced against
    // and faults.  The emit below keys on `w_code` (callee-agnostic); the token
    // is resolved / synthesised per `callee_key` via `get_assembler_token`.
    if w_code as usize != caller_code as usize {
        let admit_mutual = fbw_rec_mutual_cutover_enabled()
            && ctx
                .session
                .borrow()
                .framestack
                .iter()
                .any(|f| f.w_code == w_code as usize);
        if !admit_mutual {
            return Ok(None);
        }
    }
    // A foreign (non self-recursive) non-pure residual already executed
    // concretely earlier in this walk (e.g. `events.append(n)` ahead of the
    // self-call).  Folding to CALL_ASSEMBLER terminates the walk symbolically;
    // a later value-unavailable decline then leaves it uncommittable, so the
    // interpreter replays the region and double-applies that mutation.  Decline
    // to the plain residual path, which eagerly executes and commits the call.
    if fbw_executed_body_residual() {
        return Ok(None);
    }
    // Branch A frame shape only: `ncells == 0`, non-global-storing callee.
    let raw = unsafe { pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef) }
        as *const pyre_interpreter::CodeObject;
    if raw.is_null() {
        return Ok(None);
    }
    let callee_code = unsafe { &*raw };
    if pyre_interpreter::ncells(callee_code) != 0 {
        return Ok(None);
    }
    // The callee's globals OBJECT (`function.w_func_globals_obj`) for the
    // `frame_stores_global` stamp.
    let callee_globals_obj = unsafe { pyre_interpreter::function_get_globals_obj(callable) };
    if unsafe {
        pyre_interpreter::w_code_frame_stores_global(
            w_code as pyre_object::PyObjectRef,
            callee_globals_obj,
        )
    } {
        return Ok(None);
    }
    // Resolve the callee's own loop or trace-in-progress marker with
    // `make_green_key(w_callee_code, 0)` (`pc = 0` = function entry). A
    // pending token only proves the callee is
    // being traced; emission below resolves compiled-or-tmp so the descr never
    // carries a bodyless token.
    let (driver, _) = crate::driver::driver_pair();
    let callee_key = crate::driver::make_green_key(w_code, 0);
    let has_existing = driver.get_loop_token_arc(callee_key).is_some()
        || driver.get_pending_token_arc(callee_key).is_some();
    if !has_existing && !fbw_rec_mutual_cutover_enabled() {
        if std::env::var_os("PYRE_P2_DIAG").is_some() {
            eprintln!("[p2-ca] decline pc={} reason=no-token", op.pc);
        }
        return Ok(None);
    }
    // warmstate.py / compile.py: resolve an installed
    // procedure token, or synthesize a tmp callback token while the real loop
    // is still tracing.
    let greenboxes = [
        majit_ir::Value::Int(0),
        majit_ir::Value::Int(0),
        majit_ir::Value::Ref(majit_ir::GcRef(w_code as usize)),
    ];
    let red_types = [Type::Ref, Type::Ref];
    let token =
        match driver.get_or_make_portal_assembler_token_arc(callee_key, &greenboxes, &red_types) {
            Some(token) => token,
            None => {
                if std::env::var_os("PYRE_P2_DIAG").is_some() {
                    eprintln!("[p2-ca] decline pc={} reason=synth-failed", op.pc);
                }
                return Ok(None);
            }
        };
    if std::env::var_os("PYRE_P2_DIAG").is_some() {
        eprintln!("[p2-ca] EMIT pc={} token={}", op.pc, token.number);
    }

    // ---- emission ----
    // Past this point every step records IR; `?` propagation aborts the
    // whole walk (the trace is discarded), the correct failure mode for a
    // recording error.
    let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
    let nlocals = callee_code.varnames.len();
    let max_stack = callee_code.max_stackdepth as usize;

    // Unbox each boxed int argument -> raw payload, then re-box it into the
    // callee's `locals[i]` through `wrapint` so the local carries the same
    // representation the callee was traced against.  Under `CAN_BE_TAGGED` a
    // small `int` becomes a tagged immediate (`ll_int_box`); a heap-only
    // re-box would force a `W_IntObject` and the callee's speculative low-bit
    // guard on the local would deopt on every recursion.  Mirror of
    // `trace_guarded_int_payload(args[i])`.
    let mut param_boxes: Vec<OpRef> = Vec::with_capacity(nparams);
    for i in 0..nparams {
        let raw_arg = walker_unbox_int(ctx, op.pc, r_args[2 + i], int_type_addr)?;
        param_boxes.push(crate::state::wrapint(ctx.trace_ctx, raw_arg));
    }

    // Execution-context red: recover it fresh off the materialized caller
    // portal frame via `GETFIELD_GC_R(frame, execution_context_descr)` rather
    // than trusting the seeded `sym.execution_context` OpRef.  The seeded OpRef
    // is a bridge-decode color-bank value (`setup_bridge_sym`) that is
    // concrete-correct at forward-compile but rebinds to the callee's own
    // `pycode` when this compiled self-recursive trace re-enters as a NESTED
    // bridge, building the callee frame with `ec == pycode` and faulting later
    // in `frame_builtin`.  The outer portal frame's `execution_context` field
    // is always the true ec (single ExecutionContext, boot-pinned), so reading
    // it off `caller_frame` is the nested-resume-safe source — the same
    // recovery `ensure_execution_context` (`trace_opcode.rs`) performs.
    let ec = ctx.trace_ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[caller_frame],
        crate::descr::pyframe_execution_context_descr(),
    );

    // Build the callee PyFrame inline (Branch A): a single positional
    // local, no cells, constant code / globals.
    let pycode_const = ctx.trace_ctx.const_ref(w_code as i64);
    let w_globals_obj_const = ctx.trace_ctx.const_ref(callee_globals_obj as i64);
    let callee_frame = crate::helpers::emit_new_pyframe_inline_with_params(
        ctx.trace_ctx,
        &param_boxes,
        nlocals + max_stack,
        nlocals,
        pycode_const,
        w_globals_obj_const,
        ec,
    );

    // do_residual_call step 1 (`pyjitpl.py`): FORCE_TOKEN +
    // SETFIELD_GC(vable_token) before the assembler call.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx);

    let ca_result = ctx.trace_ctx.call_assembler_red_only_ref_arc(
        token,
        &[callee_frame, ec],
        &[Type::Ref, Type::Ref],
    );
    // pyjitpl.py: KEEPALIVE on the callee virtualizable so it
    // survives until the result is consumed.
    ctx.trace_ctx.record_op(OpCode::Keepalive, &[callee_frame]);

    // pyjitpl.py `execute_and_record_varargs(CALL_MAY_FORCE_R)`:
    // the forces branch EXECUTES the call during tracing —
    // `direct_assembler_call` (pyjitpl.py) only rewrites the
    // already-recorded op into CALL_ASSEMBLER afterwards, so the result
    // box always carries the executed value. The retired call-replay leg's
    // `trace_guarded_int_payload(ca_result)` consumed the same concrete
    // result (trace_opcode.rs).
    // Without the stamp the downstream BINARY_OP on two recursive-call
    // results cannot take the int specialization and records the generic
    // dunder-dispatch residual instead — the compiled loop then runs the
    // full `lookup_where`/type-dispatch chain per call.  Reuse the
    // residual executor primitive: it brackets the active vable with the
    // TOKEN_TRACING_RESCALL protocol, suspends re-entrant trace
    // continuation across the callee's `jit_merge_point`, stamps
    // `ca_result` with the executed concrete on success, and seeds the
    // standing exception state on a raise.
    let argbox_types: Vec<Type> = vec![Type::Ref; r_args.len()];
    let allboxes = build_allboxes(funcptr, r_args, &argbox_types, call_descr.arg_types());
    let exec = {
        let _selfrec_ca_fold_guard = SelfRecCaFoldGuard::enter();
        try_execute_residual_call_via_executor(
            ctx,
            OpCode::CallMayForceR,
            &allboxes,
            call_descr,
            ca_result,
            op.pc,
        )?
    };
    // A decline leaves the CALL_ASSEMBLER recorded symbolically WITHOUT
    // running it — a side effect only the legacy replay applies, so the
    // walk-end no-replay commit must stay off for this trace (see
    // `fbw_has_unjournaled_effect`).
    let exec_raised = match exec {
        ResidualExecOutcome::Executed(result) => result.is_err(),
        ResidualExecOutcome::Declined(cause) => {
            fbw_mark_unjournaled_effect(cause);
            false
        }
    };

    // pyjitpl.py: heapcache invalidation for the escaped frame.
    ctx.trace_ctx
        .heap_cache_mut()
        .invalidate_caches_for_escaped();

    // pyjitpl.py `make_result_of_lastop`: the result lands in
    // `registers_*[reg_index]` BEFORE GUARD_NOT_FORCED (2079) and
    // `handle_possible_exception` (2082).  The writeback MUST precede the
    // two guards so their after-call resume snapshots read the recorded
    // OpRef in the dst slot the resume position points at — deferring it
    // past the guards surfaces a stale box in the fail_args for the `>X`
    // slot on a raising/forcing deopt.  Mirror of the sibling residual
    // path (jitcode_dispatch.rs) and
    // `do_residual_call_walker_emit`.  `CALL_ASSEMBLER_R` yields the boxed
    // PyObject return value, taken as-is by the Ref dst (the consuming
    // BINARY_OP unboxes); eligibility pinned `dst_bank == 'r'`.
    // Written REGARDLESS of `exec_raised`: on a raise `ca_result` is still
    // the recorded CALL_ASSEMBLER OpRef (carrying a Null concrete shadow,
    // never read on the exception path), and the after-call resume snapshots
    // must see it in the dst slot — the same unconditional non-void writeback
    // as the residual dispatcher (pyjitpl.py / 2074-2077:
    // make_result_of_lastop before handle_possible_exception for
    // get_list_of_active_boxes).
    write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, ca_result)?;

    // The CALL_ASSEMBLER fold records both post-call guards before the walk
    // reaches the next Python-opcode boundary, so advance the caller's stack
    // mirror here.  Keep any operands below the call, discard the callable,
    // NULL marker, and arguments, then put the assembler result on the new
    // TOS.  The register writeback above already carries the same result, but
    // guard snapshotting sources operand-stack slots from this mirror.
    if ctx.vstack_valid {
        let caller_jitcode = unsafe { &*sym.jitcode() };
        let caller_code = unsafe { &*caller_jitcode.payload.code_ptr };
        let call_py_pc = python_pc_for_jitcode_pc(&caller_jitcode.payload.metadata, op.pc) as usize;
        let resume_py_pc = crate::pyjitpl::semantic_fallthrough_pc(caller_code, call_py_pc) as u32;
        let resume_depth = crate::liveness::liveness_for(caller_jitcode.payload.code_ptr)
            .depth_at_py_pc()
            .get(resume_py_pc as usize)
            .copied()
            .unwrap_or(0) as usize;
        ctx.vstack_boxes.truncate(resume_depth);
        ctx.vstack_boxes.resize(resume_depth, OpRef::NONE);
        if resume_depth > 0 {
            ctx.vstack_boxes[resume_depth - 1] = ca_result;
        }
        ctx.vstack_cur_pypc = resume_py_pc;
        ctx.vstack_depth = resume_depth;
        ctx.vstack_last_ref = OpRef::NONE;
    }

    // pyjitpl.py: GUARD_NOT_FORCED + resume snapshot advanced past
    // the call (`capture_resumedata(after_residual_call=True)`).
    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    // pyjitpl.py `handle_possible_exception`.
    if exec_raised {
        // Raising branch (pyjitpl.py): `GUARD_EXCEPTION` with
        // the const class pin, then `finishframe_exception()` — the
        // remaining bytes of the arm never run.  Mirror of the residual
        // dispatcher's raising tail: surface `SubRaise` so `walk_loop`
        // emits the outer `FINISH(exc)` (or an outer inline frame's
        // handler catches it).
        walker_record_guard_exception(ctx, op.pc);
        let exc = ctx
            .last_exc_value
            .expect("exec_raised implies last_exc_value seeded by the Err branch");
        let exc_concrete = ctx.last_exc_value_concrete;
        return Ok(Some((
            DispatchOutcome::SubRaise { exc, exc_concrete },
            op.next_pc,
        )));
    }
    // GUARD_NO_EXCEPTION on the non-raising recording path.
    ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;

    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// Walker mirror of `opimpl_recursive_call_assembler`
/// (`metainterp.rs`): a multi-frame inlined callee sub-walk reached its
/// OWN loop header (surfaced as `SubLoopCalleeCallAssembler`) and a compiled
/// loop token already exists for it. The inlined prologue already populated
/// the seeded virtual callee frame's locals via `setarrayitem_vable`, so this
/// only pins the loop-entry resume position (`last_instr = target_pc - 1`) on
/// the frame, then emits `CALL_ASSEMBLER([frame, ec])` into the token —
/// forcing the virtual frame materializes the locals the compiled loop reads
/// at entry. The op sequence (vable/vref-before, CALL_ASSEMBLER + KEEPALIVE,
/// residual executor to run the call concretely and stamp `ca_result`, dst
/// writeback, GUARD_NOT_FORCED + GUARD_NO_EXCEPTION) mirrors
/// [`try_walker_call_assembler_self_recursive`]. `PYRE_FBW_LOOP_CALLEE_CA`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_walker_loop_callee_call_assembler<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
    callee_frame: OpRef,
    callee_ec: OpRef,
    nlocals: usize,
    token: std::sync::Arc<majit_backend::JitCellToken>,
    target_pc: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    debug_assert!(callee_frame != OpRef::NONE && callee_ec != OpRef::NONE);
    let _ = nlocals;

    // Pin the loop-entry resume position on the (still-virtual) callee frame:
    // override `last_instr` from -1 (the fresh-frame entry value
    // `emit_new_pyframe_inline_with_params` wrote) to `target_pc - 1`, so the
    // compiled loop's `next_instr()` lands on the merge point. The frame's
    // `valuestackdepth` was already seeded to `nlocals` at construction (empty
    // stack at the while-header), and the locals themselves are recorded as
    // virtual-frame items by the inlined prologue stores — both flow through
    // when the CALL_ASSEMBLER forces the virtual. Uses `SetfieldGc` + a real
    // `FieldDescr` (the same field-set the builder uses), so
    // `optimize_setfield_gc` records it into the virtual's `vinfo.fields`.
    let last_instr = ctx.trace_ctx.const_int(target_pc as i64 - 1);
    let last_instr_descr = crate::descr::pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_frame, last_instr],
        last_instr_descr,
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(callee_frame, last_instr_idx, last_instr);

    // do_residual_call step 1 (`pyjitpl.py`): FORCE_TOKEN +
    // SETFIELD_GC(vable_token) before the assembler call.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx);

    let ca_result = if fbw_vable_scalar_ca_enabled() {
        // `PYRE_FBW_VABLE_SCALAR_CA`: route through the vable-scalar
        // emitter so loop-carried locals become scalar CALL_ASSEMBLER args +
        // `VableExpansion` arg_overrides, letting the optimizer elide the
        // per-call frame-array build. Scaffolding only: the emitter currently
        // produces the identical red-only CA; the vable_expansion routing lands
        // in S2.
        emit_loop_callee_ca_vable_scalar(ctx, callee_frame, callee_ec, token)
    } else {
        ctx.trace_ctx.call_assembler_red_only_ref_arc(
            token,
            &[callee_frame, callee_ec],
            &[Type::Ref, Type::Ref],
        )
    };
    ctx.trace_ctx.record_op(OpCode::Keepalive, &[callee_frame]);

    // Run the call concretely to stamp `ca_result` (same rationale as the
    // self-recursive arm: the downstream consumer needs the real concrete to
    // take its int specialization). ⚠️ The inlined prologue already ran the
    // callee's pre-loop bytecode concretely during the sub-walk; the executor
    // re-runs the WHOLE call fresh, so a side-effecting pre-loop body would
    // execute twice at trace time. The corpus target (`loop_callee_return`)
    // has a side-effect-free callee; a side-effecting prologue is out of scope.
    //
    // GC-rooting of the materialized callee virtualizable frame is equivalent
    // to the GC-clean self-recursive arm (a four-lens audit found no
    // content-dependent rooting defect): the frame is built with the same
    // `pyframe_size_descr()` + `pyobject_gcarray_descr()` locals array as the
    // fib frame, is JUMP-loop-carried so its slot is in every inner
    // residual-call gcmap, and the runtime `PyFrame`/array GC type registration
    // traces frame->array->elements with no int-vs-ref branch. A historical
    // GC-stress SEGV (freed, not-forwarded receiver under nursery pressure)
    // reproduced ONLY on layout-shifting diagnostic-probe builds; on clean
    // binaries it does not reproduce across the GC-stress matrix
    // (r1/r5/r6/r2/r4 × nursery {default,1M,256K,64K,16K,4K} × dynasm+x86, all
    // clean) — a diagnostic-build layout artifact, with content-agnostic
    // rooting ruling out a ref-specific defect here. See
    // `fbw_loop_callee_ca_enabled` for the full default-ON rationale.
    let argbox_types: Vec<Type> = vec![Type::Ref; r_args.len()];
    let allboxes = build_allboxes(funcptr, r_args, &argbox_types, call_descr.arg_types());
    let exec = try_execute_residual_call_via_executor(
        ctx,
        OpCode::CallMayForceR,
        &allboxes,
        call_descr,
        ca_result,
        op.pc,
    )?;
    let exec_raised = match exec {
        ResidualExecOutcome::Executed(result) => result.is_err(),
        ResidualExecOutcome::Declined(cause) => {
            fbw_mark_unjournaled_effect(cause);
            false
        }
    };

    ctx.trace_ctx
        .heap_cache_mut()
        .invalidate_caches_for_escaped();
    write_residual_call_result_to_dst(ctx, op.pc, dst, dst_bank, ca_result)?;

    ctx.trace_ctx.record_guard(OpCode::GuardNotForced, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    if exec_raised {
        walker_record_guard_exception(ctx, op.pc);
        let exc = ctx
            .last_exc_value
            .expect("exec_raised implies last_exc_value seeded by the Err branch");
        let exc_concrete = ctx.last_exc_value_concrete;
        return Ok(Some((
            DispatchOutcome::SubRaise { exc, exc_concrete },
            op.next_pc,
        )));
    }
    ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
    walker_capture_snapshot_for_last_guard(ctx, op.pc)?;

    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// `PYRE_FBW_VABLE_SCALAR_CA` emission seam (scaffolding only).
///
/// Emits the loop-callee CALL_ASSEMBLER when the vable-scalar mode is
/// on. Today it produces the identical red-only `[callee_frame, callee_ec]` CA
/// as the default path, so flag-ON is byte-identical to flag-OFF. Wiring the
/// body to `call_assembler_with_vable_expansion` — passing the callee's
/// loop-carried locals as scalar args plus a `VableExpansion` whose
/// `arg_overrides` map each scalar to a callee jitframe slot
/// (`rewrite.py` handle_call_assembler parity) — would let the
/// optimizer elide the per-call frame-array build.
pub(crate) fn emit_loop_callee_ca_vable_scalar<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    callee_frame: OpRef,
    callee_ec: OpRef,
    token: std::sync::Arc<majit_backend::JitCellToken>,
) -> OpRef {
    ctx.trace_ctx.call_assembler_red_only_ref_arc(
        token,
        &[callee_frame, callee_ec],
        &[Type::Ref, Type::Ref],
    )
}

/// #62 slice (3c): full-body-walk inline of a recognized user-function
/// `call_fn`.  Dev-gated by `PYRE_FBW_INLINE` (default OFF — the production
/// flag-on path is unchanged until this is validated and the gate retired).
///
/// Returns:
/// * `Ok(Some((outcome, next_pc)))` — the call was inlined; caller returns it.
/// * `Ok(None)` — not eligible (gate off, not a pure-Python function, has a
///   closure, or not an exact-positional call).  This branch emits NO IR, so
///   the caller's residual-call fallback is clean.
/// * `Err(..)` — a sub-walk step hit an unsupported op AFTER emitting IR;
///   propagated as a trace abort (sound — aborts to the interpreter rather
///   than mixing inlined + residual emission).
///
/// Arg layout: `r_args = [callable@0, null_or_self@1, positional@2..]`.
/// `bh_call_fn_impl` prepends a non-null `null_or_self` as arg0, so the
/// inlined callee's positional locals are either `positional` for plain calls
/// or `[null_or_self, positional...]` for method-form calls.
/// Only exact-positional, closure-free callees are inlined.  Guards inside a
/// pure-leaf callee resume to the caller's CALL boundary via the inherited
/// single-frame snapshot (`entry_py_pc` / `outer_active_boxes`), which is
/// sound for side-effect-free leaves (re-execute the whole call on deopt).
pub(crate) fn reconstructed_all_ref_call_stack<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &WalkContext<'_, '_, Sym>,
) -> Option<Vec<pyre_object::PyObjectRef>> {
    let fresh = read_ref_var_list_concrete(code, op, 1, ctx);
    if fresh.is_empty() {
        return None;
    }
    // The encoded residual args describe only the CALL operands.  Values can
    // remain below them on the Python operand stack (notably the iterator of
    // an enclosing FOR_ITER).  RPython resumes the complete MIFrame stack, so
    // retain that prefix from the authoritative vstack mirror.
    let prefix_len = if ctx.vstack_valid {
        ctx.vstack_boxes.len().checked_sub(fresh.len())?
    } else {
        0
    };
    let mut stack = Vec::with_capacity(prefix_len + fresh.len());
    for &value in &ctx.vstack_boxes[..prefix_len] {
        match concrete_from_recorded_opref(ctx, value) {
            ConcreteValue::Ref(r) if !r.is_null() => stack.push(r),
            _ => return None,
        }
    }
    for c in fresh {
        match c {
            ConcreteValue::Ref(r) => stack.push(r),
            _ => return None,
        }
    }
    stack.first().is_some_and(|c| !c.is_null()).then_some(stack)
}

pub(crate) fn try_walker_inline_user_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    ref_operand_offset: usize,
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    pyre_helper: majit_ir::PyreHelperKind,
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    // Default ON since the Phase 5 flip; `PYRE_FBW_INLINE=0` opts out.
    // Authoritative walks only: inline sub-walks lean on FBW multi-frame
    // snapshot plumbing a non-authoritative context does not carry.
    if !ctx.is_authoritative_executor || std::env::var("PYRE_FBW_INLINE").as_deref() == Ok("0") {
        return Ok(None);
    }
    // Only a genuine Python call helper (`call_fn` / `call_fn_N`, tagged
    // `PyreHelperKind::CallFn` by the flatten lowering) is an inline
    // target.  Every container/builtin helper routed here carries a
    // different tag or `None` (`store_subscr_fn` -> StoreSubscr,
    // `normalize_raise_varargs_fn` / `set_current_exception` -> None).
    // Without this guard `d[f] = v` with a 1-arg function key `f` lowers
    // to `residual_call_r_v(store_subscr_fn, [d, f, v])`, whose ref args
    // pass the function sniff below and are mis-inlined as `f(v)`,
    // skipping the store.  Upstream never inlines a Python call at a
    // residual_call site (inlinable calls get their own inline_call
    // jitcodes); this restores that invariant for the pyre FBW
    // inline-at-residual lever.
    if pyre_helper != majit_ir::PyreHelperKind::CallFn {
        return Ok(None);
    }
    if r_args.is_empty() {
        return Ok(None);
    }
    let mut arg_concretes = read_ref_var_list_concrete(code, op, ref_operand_offset, ctx);
    if r_args.len() < 2 {
        return Ok(None);
    }
    for i in 0..2 {
        if matches!(arg_concretes.get(i), Some(ConcreteValue::Null)) {
            if let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.box_value(r_args[i]) {
                if r != majit_ir::GcRef::NO_CONCRETE && r.as_usize() != 0 {
                    arg_concretes[i] = ConcreteValue::Ref(r.as_usize() as pyre_object::PyObjectRef);
                }
            }
        }
    }
    let ConcreteValue::Ref(callable) = arg_concretes[0] else {
        return Ok(None);
    };
    if callable.is_null() {
        return Ok(None);
    }
    let ConcreteValue::Ref(null_or_self) = arg_concretes[1] else {
        return Ok(None);
    };
    let method_form = !null_or_self.is_null();
    let mut callee_args = Vec::with_capacity(r_args.len().saturating_sub(1));
    let mut callee_arg_concretes = Vec::with_capacity(arg_concretes.len().saturating_sub(1));
    if method_form {
        callee_args.push(r_args[1]);
        callee_arg_concretes.push(arg_concretes[1]);
    }
    callee_args.extend_from_slice(&r_args[2..]);
    callee_arg_concretes.extend_from_slice(&arg_concretes[2..]);
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(callable) })
    else {
        return Ok(None);
    };
    try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        callable,
        r_args[0],
        callable,
        arg_concretes,
        callee_args,
        callee_arg_concretes,
        method_form,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        false,
        false,
    )
}

/// Shared post-resolution half of the FBW inline lever. Ordinary Python calls
/// resolve their callee from the CALL operand; builtin-dispatch specializers
/// resolve an app-level descriptor first and enter here with that function as
/// the callee while independently pinning the original builtin callable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_resolved_user_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
    callable: pyre_object::PyObjectRef,
    callable_guard_op: OpRef,
    callable_guard_value: pyre_object::PyObjectRef,
    arg_concretes: Vec<ConcreteValue>,
    callee_args: Vec<OpRef>,
    callee_arg_concretes: Vec<ConcreteValue>,
    method_form: bool,
    w_code: *const (),
    nparams: usize,
    has_closure: bool,
    exception_receiver_guard: Option<ExceptionInlineReceiverGuard>,
    arg_class_guard: Option<ArgClassGuard>,
    allow_method_load_attr: bool,
    require_str_result: bool,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    // Only exact-positional, closure-free calls: every callee local [0..nparams]
    // is bound from a passed arg, none from defaults/varargs/cells.
    if has_closure || callee_args.len() != nparams {
        return Ok(None);
    }
    // Bound recursive inlining at `max_unroll_recursion`: a callee already
    // this deep on the FBW inline stack falls back to a residual call rather
    // than unrolling its (exponentially branching) call tree at trace time.
    // Mirror of `pyjitpl.py` `recursion_exceeded` →
    // `assembler_call` instead of trace-through.
    let callee_code_key = w_code as pyre_object::PyObjectRef as usize;
    if fbw_inline_recursion_count(ctx, callee_code_key) >= FBW_MAX_INLINE_RECURSION {
        return Ok(None);
    }
    let Some(body) = crate::state::sub_jitcode_body_for_code(w_code) else {
        return Ok(None);
    };
    if nparams > body.num_regs_r {
        return Ok(None);
    }
    // The callee body resolves its `d`/`j` descr operands through its OWN
    // per-fn pool, not the caller's.  Without this the sub-walk reads the
    // wrong descr at the first `getfield_vable_r` / `residual_call`
    // (`VableArrayDescrMalformed` / `ResidualCallDescrNotCallDescr`).
    let Some((callee_descr_refs, callee_perfn_descrs, callee_lookup)) =
        crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return Ok(None);
    };
    let args_all_numeric = callee_arg_concretes.iter().all(|concrete| match concrete {
        ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => true,
        ConcreteValue::Ref(obj) if !obj.is_null() => unsafe {
            pyre_object::is_int(*obj) || pyre_object::is_float(*obj)
        },
        ConcreteValue::Ref(_) | ConcreteValue::Null => false,
    });
    let args_all_builtin_integer = callee_arg_concretes.iter().all(|concrete| match concrete {
        ConcreteValue::Int(_) | ConcreteValue::Bool(_) => true,
        ConcreteValue::Ref(obj) if !obj.is_null() => unsafe { pyre_object::is_int_or_long(*obj) },
        ConcreteValue::Float(_) | ConcreteValue::Ref(_) | ConcreteValue::Null => false,
    });
    // Keep exact-integer arithmetic callees as one residual call when tracing
    // a guard-origin bridge.  Re-inlining their BinaryOp body would create a
    // second virtual frame whose operand stack is not a red bridge input; an
    // overflow path can then compile NULL vable stack slots into the bridge.
    // The primary loop still inlines the callee, and non-integer/user-
    // overridable calls continue through the ordinary inline/abort policy.
    if ctx.trace_ctx.is_bridge_trace
        && args_all_builtin_integer
        && fbw_callee_body_has_binary_op_residual(body.code, callee_descr_refs)
    {
        return Ok(None);
    }
    // An inline sub-walk inside a FOR_ITER body resumes a guard at the
    // caller's CALL boundary, so deopt re-executes the whole callee.  Replaying
    // a live-heap mutation would double it; the nested-residual decline catches
    // that only after an abort storm.  A side-effect-free callee replays
    // benignly, so admit it.  `PYRE_FBW_FORITER_INLINE=0` restores the former
    // blanket decline as a rollback escape hatch.
    if fbw_foriter_inflight_active()
        && (std::env::var("PYRE_FBW_FORITER_INLINE").as_deref() == Ok("0")
            || !fbw_callee_body_side_effect_free(
                body.code,
                args_all_numeric,
                body.num_regs_i,
                body.constants_i,
                callee_descr_refs,
            ))
    {
        return Ok(None);
    }
    if method_form
        && !allow_method_load_attr
        && !method_form_callee_body_supported(body.code, callee_descr_refs)
    {
        return Ok(None);
    }
    if std::env::var("PYRE_FBW_INLINE_DIAG").is_ok() {
        let mut pc = 0usize;
        let mut shown = 0;
        while pc < body.code.len() && shown < 8 {
            let Some(d) = crate::jitcode_runtime::decode_op_at(body.code, pc) else {
                break;
            };
            let ops: Vec<u8> = body.code[d.pc + 1..d.next_pc.min(body.code.len())].to_vec();
            eprintln!("[inline-body] pc={} {} operands={:?}", d.pc, d.key, ops);
            pc = d.next_pc;
            shown += 1;
        }
    }
    // The inlined callee body is entered at pc=0 with the fast-path
    // register convention `registers_r[0..nparams] = positional args` —
    // the same seeding `dispatch_inline_call_dr_kind` uses for `n_*`
    // inline calls and the retired `can_skip_traced_callee_frame` branch used
    // (`sym.registers_r = args.to_vec()`). This only holds for a callee
    // that reads its params straight from `r0`/`r1` (ref_copy +
    // residual_call args).  A callee that materializes a frame — any
    // `*_vable_*` op, emitted when a local must survive a sub-call —
    // reads from the unseeded frame box; inlining it would abort the
    // *whole* enclosing trace with `VableBoxNotSeeded`.
    //
    // The zero-param case lowers to an ordinary residual call (orthodox
    // non-inlinable path); a residual zero-arg call is cheap and has no
    // positional-arg inline win to recover.
    if nparams == 0 {
        return Ok(None);
    }
    // A param-bearing Python callee that is otherwise inline-eligible but
    // whose body is not a straight-line leaf (loop / branch / non-static
    // vable) cannot be served by the fast-path register seeding.  Emitting
    // the residual leaves it re-interpreted per iteration and lets its short
    // inner loops compile + deopt-storm — strictly slower than interpreting.
    // Decline the enclosing key to interpretation
    // (`FBW_DECLINED_KEYS`) instead of recording the slow residual.
    // Resolve the callee's own portal frame register up-front so both the
    // strict predicate (own-frame vable acceptance) and the multiframe gate
    // share one `ensure_jitcode_index` + `portal_red_regs_at` lookup.  A
    // portal-shaped strict straight-line leaf's LOAD_FAST / STORE_FAST carry
    // the frame-vable locals prologue, folded register-to-register against
    // this frame reg (see the `*_vable_via_metainterp` short-circuits).
    // `u16::MAX` for a non-portal callee keeps the strict predicate
    // byte-identical (`inline_resolvable_seeded_frame_op` declines).
    let callee_portal_frame_reg = crate::state::ensure_jitcode_index(callee_code_key as *const ())
        .filter(|&jc| crate::state::built_as_portal_at(jc))
        .map(|jc| crate::state::portal_red_regs_at(jc).0)
        .unwrap_or(u16::MAX);
    let strict_inlinable =
        callee_fast_path_inlinable(body.code, callee_descr_refs, ctx, callee_portal_frame_reg);

    // A self-recursive callee routes to the direct `CALL_ASSEMBLER` arm
    // (`try_walker_call_assembler_self_recursive`, reached when this inline
    // attempt returns `Ok(None)`) rather than the multiframe inline path.
    // Multi-parameter all-int calls fold there; other self-recursive shapes
    // decline from the fold to the plain residual call instead of aborting.
    // Detected before the multiframe gate: a forward-branch self-recursive
    // callee is `try_multiframe`-eligible, but unbounded self-recursion bottoms
    // out the multiframe inline at the depth cap.  A strict-inlinable callee is
    // a straight-line leaf (no self-recursion), so this never preempts the
    // strict path.  Gated on `PYRE_FBW_REC_CA`, matching the fold.
    if !strict_inlinable
        && std::env::var_os("PYRE_FBW_REC_CA").as_deref() != Some(std::ffi::OsStr::new("0"))
        && nparams >= 1
    {
        let sym_ptr = ctx.fbw_mode.snapshot_sym;
        let self_recursive = !sym_ptr.is_null()
            && unsafe {
                pyre_interpreter::live_code_wrapper((*(*sym_ptr).jitcode()).raw_code() as *const ())
                    as *const ()
            } as usize
                == w_code as usize;
        if self_recursive {
            // RPython `opimpl_recursive_call` / `do_recursive_call`
            // (`pyjitpl.py`) unroll within `max_unroll_recursion`,
            // then fall back to the assembler-call path.  Default-on
            // (`fbw_rec_multiframe_enabled`): a primary trace spends the
            // multiframe budget unrolling recursion below the depth bound
            // before folding the deepest call to the recursive portal
            // `CALL_ASSEMBLER`.
            let unroll = fbw_rec_multiframe_enabled()
                && !ctx.fbw_mode.carrier_resume
                && ctx.session.borrow().framestack.len() < fbw_max_multiframe_depth();
            if !unroll {
                return Ok(None);
            }
            // fall through to the multiframe gate (unroll one level)
        }
    }
    // #68: under `PYRE_FBW_INLINE_MULTIFRAME`, a forward-branch-bearing callee
    // is inlinable with a multi-frame guard snapshot (its in-callee branch
    // guard resumes through `walker_capture_multi_frame_inline_snapshot` rather
    // than collapsing to the caller boundary).  The relaxed predicate also
    // accepts a callee whose only non-strict ops are reads off its OWN seeded
    // frame register, so resolve that register up-front (the same
    // `ensure_jitcode_index` + `portal_red_regs_at` the seeding below uses).
    // A multiframe caller no longer needs to be TOP-LEVEL: a nested caller's
    // paused frame is computed from the framestack's top (the live
    // intermediate callee jitcode) by `compute_inline_caller_frame`, bounded by
    // a depth cap on the inline stack (the `n_parents == n_callees` valve in
    // the snapshot path is the real desync safety net).
    let multiframe_eligible = !strict_inlinable && fbw_inline_multiframe_enabled();
    let callee_frame_reg = if multiframe_eligible {
        crate::state::ensure_jitcode_index(callee_code_key as *const ())
            .map(|jc| crate::state::portal_red_regs_at(jc).0)
            .unwrap_or(u16::MAX)
    } else {
        u16::MAX
    };
    let inline_depth = ctx.session.borrow().framestack.len();
    let try_multiframe = multiframe_eligible
        && inline_depth < fbw_max_multiframe_depth()
        && callee_fast_path_inlinable_allowing_forward_branch(
            body.code,
            callee_descr_refs,
            ctx,
            callee_frame_reg,
        );
    if !strict_inlinable && !try_multiframe {
        // A non-self-recursive loop/branch callee that neither the strict nor
        // the multiframe fast path can serve declines to interpretation
        // (`FBW_DECLINED_KEYS`).  Self-recursive calls were already routed to
        // the `CALL_ASSEMBLER` fold or plain residual path above (`Ok(None)`).
        //
        // For method-form calls reached through the LOAD_METHOD fold, decline
        // locally instead.  The fold's first-order win is the guarded method
        // cache; making an uninlineable method body blacklist the whole outer
        // loop turns a correct specialization into a compile regression.
        if method_form {
            return Ok(None);
        }
        // Full-portal cutover: instead of poisoning the trace, fall through to
        // the CALL_ASSEMBLER fold (`try_walker_call_assembler_self_recursive`,
        // reached next in the residual-call dispatch) so a recursive callee at
        // the inline cap enters via its own (possibly tmp-callback) loop token.
        if fbw_rec_mutual_cutover_enabled() {
            return Ok(None);
        }
        return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
    }

    // Path-1 (#68): the inlined callee's compile-time-constant frame fields,
    // so a scalar `getfield_vable_r` off its own (unseeded) portal frame —
    // the `w_globals` namespace for a LOAD_GLOBAL, the promote-to-const
    // `pycode` — resolves to the constant via
    // `try_resolve_inline_callee_static_field` instead of aborting
    // `VableBoxNotSeeded`.  Mirror of the codewriter non-portal branch.
    let inline_consts = InlineCalleeConsts {
        w_globals: unsafe { pyre_interpreter::function_get_globals_obj(callable) } as usize,
        w_code: callee_code_key,
    };

    // Specialize the inlined body on this exact callable: a later
    // iteration calling a different function at this site must deopt
    // rather than run the wrong body.  The guard resumes at the caller's
    // CALL boundary (single outer Python frame — re-execute the whole
    // call on deopt), captured via `fbw_mode.inline_subwalk` for
    // the sub-walk guards below.
    if let Some((receiver, concrete_receiver, w_class, version_tag)) = exception_receiver_guard {
        walker_guard_exception_attr_slot(
            ctx,
            op.pc,
            receiver,
            concrete_receiver,
            w_class,
            version_tag,
        )?;
    }
    if let Some((arg, concrete_arg, w_type)) = arg_class_guard {
        // `GuardClass` compares the object's physical `ob_type`, not its Python
        // `W_TypeObject`.  Pin the physical type: a boxed builtin whose
        // `ob_type` the optimizer already knows would make a `GuardClass`
        // against the heap type object provably fail, discarding the loop
        // (`InvalidLoop`).  `walker_guard_class` also emits the tagged-int
        // low-bit test, needed when `arg` may arrive as a tagged int.
        let physical_type = unsafe { (*concrete_arg).ob_type } as i64;
        walker_guard_class(ctx, op.pc, arg, physical_type)?;
        // A builtin subclass / user instance shares its `ob_type` with the base
        // layout, so `ob_type` alone does not pin the class the reflected-op
        // decline (`w_type_issubtype`) was computed against.  Guard the live
        // `w_class` too so an arg of a distinct class deopts.  Singletons with a
        // null `w_class` (`Ellipsis`/`NotImplemented`) are pinned exactly by
        // `ob_type`, and guarding their null slot against `w_type` would itself
        // be provably false — so guard `w_class` only when it is populated.
        if !unsafe { (*concrete_arg).w_class }.is_null() {
            let live_w_class = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                arg,
                crate::descr::w_class_descr(),
            );
            let w_class_const = ctx.trace_ctx.const_ref(w_type as i64);
            walker_emit_fold_guard_with_snapshot(
                ctx,
                op.pc,
                OpCode::GuardValue,
                &[live_w_class, w_class_const],
            )?;
            ctx.trace_ctx
                .heap_cache_mut()
                .replace_box(live_w_class, w_class_const);
        }
    }

    let callable_expected = ctx
        .trace_ctx
        .const_ref(callable_guard_value as usize as i64);
    if !callable_guard_op.is_constant() {
        ctx.trace_ctx.record_guard(
            OpCode::GuardValue,
            &[callable_guard_op, callable_expected],
            0,
        );
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }

    let (
        mut callee_regs_r,
        mut callee_regs_i,
        mut callee_regs_f,
        mut callee_concrete_r,
        mut callee_concrete_i,
    ) = allocate_callee_register_banks(&body, ctx.trace_ctx);
    // Fast-path arg seeding: positional args land in the callee's
    // param registers with their concrete shadow (mirror of
    // `dispatch_inline_call_dr_kind`).  The canonical splice regalloc does
    // not pin local-i inputargs to identity colors, so the register the
    // body reads param i from is its per-PC pcdep color at the callee entry
    // (`pcdep_color_slots[0]`), not `r{i}`; an empty fixture map is identity.
    let entry_colors = crate::state::sub_jitcode_entry_param_colors(w_code);
    for i in 0..nparams {
        let reg = match &entry_colors {
            // Colored jitcode: seed param `i` at the register it occupies at
            // the callee entry PC.  A param dead at entry carries no entry
            // color — the body never reads it, so skip seeding rather than
            // clobber a live register.
            Some(entries) => match entries
                .iter()
                .find(|&&(b, _, slot)| b == 1 && slot as usize == i)
            {
                Some(&(_, color, _)) => color as usize,
                None => continue,
            },
            // Portal / skeleton install (empty `pcdep_color_slots`): colors
            // are slot-identity, so param `i` lives in register `i`.
            None => i,
        };
        if reg >= callee_regs_r.len() {
            return Ok(None);
        }
        callee_regs_r[reg] = callee_args[i];
        callee_concrete_r[reg] = callee_arg_concretes[i];
    }

    // #68: seed the callee's `frame` / `ec` reds that the codewriter
    // force-alives at every pc (portal_frame_reg / portal_ec_reg).  The
    // sym-less fast path seeds only the param colors above, leaving the reds
    // OpRef::NONE so an in-callee guard snapshot cannot source them
    // (`collect_callee_active_boxes` declines).  RPython seeds these reds as
    // part of `setup_call(allboxes)` for a recursive-portal inline
    // (`pyjitpl.py`, reds=['frame','ec'] `interp_jit.py`): a
    // freshly-built (virtual) callee `PyFrame` plus the caller's shared `ec`.
    // pyre's "every function is its own portal" model makes every inlined
    // callee portal-shaped, so the same seeding applies.  The frame box stays
    // virtual on the hot path (the optimizer folds the NewWithVtable away) and
    // is materialized only on guard failure; `collect_callee_active_boxes` is
    // then unchanged (it finds real boxes).
    //
    // Seeded for BOTH the forward-branch multiframe callee (`try_multiframe`)
    // AND a STRICT straight-line callee at the top inline level (`strict_seed`).
    // With the reds seeded, an in-callee guard resumes at the callee's OWN
    // coordinate through `walker_capture_multi_frame_inline_snapshot` instead of
    // collapsing to the caller boundary and re-executing the whole call — which
    // re-materializes it at a stale `valuestackdepth` (a resume `LOAD_FAST` push
    // overflows the frame, an `rd_numb` decode overruns) and re-applies a
    // committed heap side effect (visible on the wasm resume path, where a
    // guard-failure deopt is not absorbed by a compiled bridge).  A
    // `try_multiframe` callee HARD-declines the inline when a precondition below
    // fails; a strict callee instead leaves the reds `OpRef::NONE` and falls
    // back to the single-frame collapse (no paused caller frame is pushed), so
    // an un-seedable strict shape never loses its inline.  Every bail below
    // precedes any IR recording, so a strict fall-through records no dead op.
    //
    // `PYRE_FBW_LOOP_CALLEE_CA`: the seeded virtual callee frame /
    // shared ec / local count are hoisted so the sub-walk return site can
    // emit a `CALL_ASSEMBLER` into the callee loop token when the sub-walk
    // surfaces `SubLoopCalleeCallAssembler` (the callee reached its own loop
    // header).  A strict straight-line callee has no loop, so that outcome
    // never arises for it and the hoisted values are simply unused.
    let mut ca_callee_frame = OpRef::NONE;
    let mut ca_callee_ec = OpRef::NONE;
    let mut ca_nlocals = 0usize;
    // A strict straight-line callee at the top inline level is seeded the same
    // way, so its in-callee guards route through the multi-frame snapshot.  A
    // deeper strict callee (`inline_depth >= fbw_max_multiframe_depth()`) keeps the
    // single-frame collapse — a 3-frame snapshot the resume path is sound for
    // only one paused caller frame.
    let strict_seed = strict_inlinable && inline_depth < fbw_max_multiframe_depth();
    // True once the callee frame reds are actually seeded (all preconditions
    // below met).  For a strict callee this gates routing its guards through the
    // multi-frame snapshot vs. falling back to collapse.
    let mut callee_frame_seeded = false;
    if try_multiframe || strict_seed {
        'seed: {
            // Branch-A frame shape only (mirror REC_CA): no cells.
            let raw = unsafe {
                pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject
            };
            if raw.is_null() {
                if try_multiframe {
                    return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
                }
                break 'seed;
            }
            let callee_code = unsafe { &*raw };
            if pyre_interpreter::ncells(callee_code) != 0 {
                if try_multiframe {
                    return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
                }
                break 'seed;
            }
            // POP_JUMP_IF_NONE / POP_JUMP_IF_NOT_NONE lower to an `is`/`is_not`
            // identity residual call whose operands must be Ref (the codewriter
            // PopJumpIfNone arm), then a branch guard.  When the multiframe inline
            // int-specializes the tested local, the mid-body guard resume cannot
            // source that operand's Ref form from the callee register banks
            // (`collect_callee_active_boxes` would read a stale/mismatched box), so
            // the encoded liveness stream disagrees with the decoder
            // (`resume.rs decode_ref: unexpected tag`) and the caller frame is
            // corrupted. Decline to the ordinary residual call until
            // the multi-frame resume reboxes int-specialized identity operands.
            // POP_JUMP_IF_TRUE/FALSE stay inlinable: their `bool` truth folds in the
            // int bank, so no Ref rebox is needed.  A strict straight-line callee
            // has no branch at all, so this scan never fires for it.
            if (0..callee_code.instructions.len()).any(|pc| {
                matches!(
                    pyre_interpreter::decode_instruction_at(callee_code, pc),
                    Some((
                        pyre_interpreter::bytecode::Instruction::PopJumpIfNone { .. }
                            | pyre_interpreter::bytecode::Instruction::PopJumpIfNotNone { .. },
                        _
                    ))
                )
            }) {
                if try_multiframe {
                    return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
                }
                break 'seed;
            }
            let nlocals = callee_code.varnames.len();
            let frame_array_size = nlocals + callee_code.max_stackdepth as usize;

            let Some(callee_jitcode_index) =
                crate::state::ensure_jitcode_index(callee_code_key as *const ())
            else {
                if try_multiframe {
                    return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
                }
                break 'seed;
            };
            let (frame_reg, ec_reg) = crate::state::portal_red_regs_at(callee_jitcode_index as i32);
            if frame_reg == u16::MAX
                || ec_reg == u16::MAX
                || frame_reg as usize >= callee_regs_r.len()
                || ec_reg as usize >= callee_regs_r.len()
            {
                if try_multiframe {
                    return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
                }
                break 'seed;
            }

            // ec red: the shared ExecutionContext (perform_call threads the
            // caller's ec down).  Recover it off the materialized caller portal
            // frame via `GETFIELD_GC_R` rather than the seeded
            // `sym.execution_context` OpRef — the seeded OpRef rebinds to the
            // callee's own `pycode` when this compiled trace re-enters as a nested
            // bridge (see `try_walker_call_assembler_self_recursive`).  The outer
            // portal frame's `execution_context` field is the single true ec.
            let sym_ptr = ctx.fbw_mode.snapshot_sym;
            if sym_ptr.is_null() {
                if try_multiframe {
                    return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
                }
                break 'seed;
            }
            let sym = unsafe { &*sym_ptr };
            let callee_ec = ctx.trace_ctx.record_op_with_descr(
                OpCode::GetfieldGcR,
                &[sym.frame()],
                crate::descr::pyframe_execution_context_descr(),
            );

            let pycode_const = ctx.trace_ctx.const_ref(w_code as i64);
            let w_globals_obj_const = ctx.trace_ctx.const_ref(inline_consts.w_globals as i64);
            let param_boxes: Vec<OpRef> = (0..nparams).map(|i| callee_args[i]).collect();
            let callee_frame = crate::helpers::emit_new_pyframe_inline_with_params(
                ctx.trace_ctx,
                &param_boxes,
                frame_array_size,
                nlocals,
                pycode_const,
                w_globals_obj_const,
                callee_ec,
            );

            callee_regs_r[frame_reg as usize] = callee_frame;
            callee_concrete_r[frame_reg as usize] = ConcreteValue::Null;
            callee_regs_r[ec_reg as usize] = callee_ec;
            callee_concrete_r[ec_reg as usize] = ConcreteValue::Null;

            // Retain for a possible `SubLoopCalleeCallAssembler` emit.
            ca_callee_frame = callee_frame;
            ca_callee_ec = callee_ec;
            ca_nlocals = nlocals;
            callee_frame_seeded = true;
        }
    }

    // gh#467 forward-flush inputs are captured AT the CALL, after this
    // iteration's pre-CALL effects and before any callee sub-walk.  Hoisting
    // them above the paused-caller-frame gate lets its try-block decline use
    // the same Entry-carrier predicates as a zero-effect sub-walk abort.
    let unjournaled_before_subwalk = fbw_has_unjournaled_effect();
    let executed_effects_before = fbw_executed_effect_count();
    let is_top_inline = !ctx.fbw_mode.inline_subwalk;
    let abort_flush_call_jitcode_coord: Option<(u32, usize)> = if is_top_inline {
        let sym_ptr = ctx.fbw_mode.snapshot_sym;
        if sym_ptr.is_null() {
            None
        } else {
            let sym = unsafe { &*sym_ptr };
            if sym.jitcode().is_null() {
                None
            } else {
                unsafe {
                    let jc = &*sym.jitcode();
                    Some((jc.index as u32, op.pc))
                }
            }
        }
    } else {
        None
    };

    // #68: a forward-branch callee inlined under the multi-frame path needs a
    // paused caller frame on the framestack so its in-callee guards
    // snapshot both frames.  Compute it here, while the caller's live register
    // banks (`ctx.registers_*`) are still in scope — at guard-capture time the
    // walk context is the callee's. A caller frame that is not snapshot-able
    // (try-block catch marker / missing liveness) declines to interpretation.
    let parent_frame = if try_multiframe {
        match compute_inline_caller_frame(ctx, op.pc) {
            Ok(pf) => Some(pf),
            Err(InlineCallerFrameDecline::TryBlockCatchMarker) => {
                // An un-entered multiframe-inline CALL declined at its
                // try-block catch marker is re-run whole and forward, exactly
                // as if it had never been inlined (`pyjitpl.py`).  The
                // multi-frame guard snapshot itself remains declined.
                if is_top_inline
                    && !unjournaled_before_subwalk
                    && fbw_executed_effect_count() == executed_effects_before
                {
                    if let (Some((outer_jitcode_index, call_jitcode_pc)), Some(stack)) = (
                        abort_flush_call_jitcode_coord,
                        reconstructed_all_ref_call_stack(code, op, ctx),
                    ) {
                        fbw_set_abort_call_resume(outer_jitcode_index, call_jitcode_pc, stack);
                    }
                }
                return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
            }
            Err(InlineCallerFrameDecline::Unavailable) => {
                return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
            }
        }
    } else if callee_frame_seeded {
        // A strict straight-line callee seeded at the top inline level (the
        // `try_multiframe` arm above already handled the branch path).  Push the
        // paused caller frame so its in-callee guards resume through the
        // multi-frame snapshot (`walker_capture_multi_frame_inline_snapshot`) at
        // the callee's OWN coordinate, with the caller paused at the CALL return
        // point (`get_list_of_active_boxes(in_a_call=true)` parity,
        // `trace_opcode.rs`). With the callee frame red now seeded,
        // `collect_callee_active_boxes` sources the callee's live boxes and the
        // snapshot succeeds, producing the full RPython `Snapshot.frames` chain
        // (`opencoder.py create_top_snapshot`, resumed by
        // `resume.py rebuild_from_resumedata`).  This replaces the single-frame
        // collapse, whose caller-boundary re-execute both mis-sizes the resumed
        // frame (a decode / `LOAD_FAST` overrun) and re-applies the callee's
        // committed side effect on deopt.
        //
        // Best effort: `compute_inline_caller_frame` returns `Unavailable` for a caller
        // shape it cannot build yet (a CALL inside a try-block, or no result on
        // the operand stack at the return point).  Fall back to the single-frame
        // collapse there (do NOT decline the inline — that shape is served
        // correctly today), so this never removes a working inline.
        compute_inline_caller_frame(ctx, op.pc).ok()
    } else {
        // Single-frame collapse (resume at the CALL boundary, re-execute the
        // whole call on deopt): a nested strict callee
        // (`inline_depth >= fbw_max_multiframe_depth()`), an un-seedable
        // strict callee, or a callee neither seed served.  Sound for a pure
        // value-returning leaf (idempotent re-execute) and for a nested
        // straight-line callee (its pre-multiframe behavior).
        None
    };

    // CODEX1 parity: snapshot the heap-effect state before the callee
    // sub-walk.  If the prologue (callee pc 0 → its loop header) mutates the
    // heap, the `SubLoopCalleeCallAssembler` arm below would re-run the WHOLE
    // call through the residual executor to stamp `ca_result`, applying the
    // prologue's side effects a second time at trace time.  RPython's
    // `do_residual_call` runs the call exactly once (`pyjitpl.py`), so a
    // side-effecting prologue must decline the CA inline (see the arm).
    let prologue_journal_before = fbw_store_journal_len();
    // Compute fresh outer_active_boxes for the inline sub-walk when the
    // parent FBW walk carries an empty set (`dispatch_via_miframe`
    // initializes `outer_active_boxes: Vec::new()`; it is computed
    // dynamically per guard by the FBW path).  A callee guard falls
    // through to the per-opcode arm path which reads `ctx.outer_active_boxes`,
    // so an empty inherited set produces a resume snapshot with zero frame
    // boxes while the decoder expects the full liveness-derived set — the
    // same defect class as the LOAD_ATTR fold empty-boxes bug.  Mirror
    // `try_walker_list_append_inline`: read the caller's live register
    // banks from `fbw_mode.snapshot_sym` at the CALL-site py_pc.
    //
    // The snapshot header coordinate (`sub_wc.entry_py_pc` /
    // `sub_wc.outer_jitcode_index`, stamped below) MUST be the SAME coordinate
    // these boxes are collected at.  A callee guard that collapses to the
    // caller boundary stamps that coordinate as its resume `SnapshotFrame`
    // header, and the decoder (`setup_bridge_sym`) reads the liveness window at
    // that header to size and place the stored boxes
    // (`reg_indices.total_len() == frame.values.len()`).  Collecting boxes at
    // the CALL site but stamping the walk-entry header desyncs the two windows
    // (count-mismatch assert / wrong slot layout), so carry the box coordinate
    // to the header alongside the boxes.
    let (
        inline_outer_active_boxes,
        inline_outer_entry_py_pc,
        inline_outer_jc_index,
        inline_outer_resume_marker_jit_pc,
    ) = if ctx.outer_active_boxes.is_empty() {
        let sym_ptr = ctx.fbw_mode.snapshot_sym;
        if sym_ptr.is_null() {
            (
                ctx.outer_active_boxes.clone(),
                ctx.entry_py_pc,
                ctx.outer_jitcode_index,
                ctx.outer_resume_marker_jit_pc,
            )
        } else {
            let sym = unsafe { &*sym_ptr };
            if sym.jitcode().is_null() {
                (
                    ctx.outer_active_boxes.clone(),
                    ctx.entry_py_pc,
                    ctx.outer_jitcode_index,
                    ctx.outer_resume_marker_jit_pc,
                )
            } else {
                // Liveness coordinate is the CALL op's own (jitcode index,
                // py_pc) — NOT the `ctx` sentinels.  `dispatch_via_miframe`
                // initializes `ctx.outer_jitcode_index` to 0 and
                // `ctx.entry_py_pc` to the walk-entry py_pc, so for a CALL in
                // a non-root jitcode, or a CALL not at the walk-entry pc,
                // those select the wrong liveness window and the callee guard
                // snapshot encodes the wrong frame boxes.  Derive the
                // coordinate from the snapshot sym's jitcode at the CALL op's
                // pc, matching `orthodox_list_append_commit`.
                let (call_site_jc_index, call_site_marker) = unsafe {
                    let jc = &*sym.jitcode();
                    let jc_index = jc.index as u32;
                    (jc_index, jc.payload.resume_marker_for_jitcode_pc(op.pc))
                };
                let call_site_py_pc =
                    crate::state::backxlat_py_pc(call_site_jc_index as i32, op.pc as i32) as u32;
                let call_site_word = match call_site_marker {
                    Some(m) => m as i32,
                    None => majit_ir::resumedata::NO_JITCODE_PC,
                };
                let boxes = collect_outer_active_boxes(
                    sym,
                    ctx.trace_ctx,
                    ctx.registers_i,
                    ctx.registers_r,
                    ctx.registers_f,
                    call_site_jc_index,
                    call_site_py_pc,
                    None,
                    call_site_word,
                    // Keep the marker for the liveness-bank query, but key
                    // entry metadata to the raw CALL offset that produced the
                    // pre-adjustment Python coordinate.
                    op.pc as i32,
                    OuterActiveBoxesEntryTwin::Plain,
                    "call_site_capture",
                    None,
                    &[],
                );
                (
                    boxes,
                    EntryPyPc::Jit(op.pc),
                    call_site_jc_index,
                    call_site_marker,
                )
            }
        }
    } else {
        // No CALL-site coordinate is derived in these fallbacks; the outer
        // coordinate is inherited from `ctx` verbatim, so the twin is too
        // (e.g. an inline CALL inside a carrier sub-walk whose outer
        // coordinate is the paused root).
        (
            ctx.outer_active_boxes.clone(),
            ctx.entry_py_pc,
            ctx.outer_jitcode_index,
            ctx.outer_resume_marker_jit_pc,
        )
    };
    let (callee_outcome, callee_class_of_last_exc_is_const) = {
        let mut sub_wc = WalkContext {
            callee_shadow: Some(Default::default()),
            // Path-1: resolve scalar static-field reads off this callee's own
            // unseeded portal frame to its compile-time constants.
            inline_callee_consts: Some(inline_consts),
            // Guards emitted inside the callee body — both the walker's own
            // and the `_nonstandard_virtualizable` PTR_EQ promote that
            // `vable_getfield_*` records internally — resume at this CALL
            // boundary (`sub_wc.entry_py_pc` / `outer_active_boxes`, both
            // stamped at the CALL-site coordinate above), not at a callee
            // `op_pc` that has no meaning in the outer jitcode's py_pc→jitcode
            // tables.
            fbw_mode: FbwWalkMode {
                inline_subwalk: true,
                ..ctx.fbw_mode
            },
            session: ctx.session,
            registers_r: &mut callee_regs_r,
            registers_i: &mut callee_regs_i,
            registers_f: &mut callee_regs_f,
            concrete_registers_r: &mut callee_concrete_r,
            concrete_registers_i: &mut callee_concrete_i,
            descr_refs: callee_descr_refs,
            raw_descrs: RawDescrPool::PerFn(callee_perfn_descrs),
            is_authoritative_executor: ctx.is_authoritative_executor,
            store_subscr_fn_addr: ctx.store_subscr_fn_addr,
            pending_guard_snapshot_error: None,
            vstack_boxes: Vec::new(),
            vstack_depth: 0,
            vstack_cur_pypc: 0,
            vstack_valid: false,
            vstack_last_ref: OpRef::NONE,
            vstack_reorder_ceiling: u32::MAX,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
            trace_ctx: ctx.trace_ctx,
            done_with_this_frame_descr_ref: ctx.done_with_this_frame_descr_ref.clone(),
            done_with_this_frame_descr_int: ctx.done_with_this_frame_descr_int.clone(),
            done_with_this_frame_descr_float: ctx.done_with_this_frame_descr_float.clone(),
            done_with_this_frame_descr_void: ctx.done_with_this_frame_descr_void.clone(),
            exit_frame_with_exception_descr_ref: ctx.exit_frame_with_exception_descr_ref.clone(),
            is_top_level: false,
            sub_jitcode_lookup: callee_lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: inline_outer_entry_py_pc,
            outer_resume_marker_jit_pc: inline_outer_resume_marker_jit_pc,
            outer_jitcode_index: inline_outer_jc_index,
            outer_active_boxes: inline_outer_active_boxes,
        };
        // Track this callee for the lifetime of the sub-walk so nested
        // self-calls see the correct recursion depth.
        let _inline_frame = InlineFrameGuard::enter(ctx.session, callee_code_key, parent_frame);
        if let Some(frame) = ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym) {
            if frame.body_matches(&body) {
                seed_callee_vstack_mirror(&mut sub_wc, &frame);
            }
        }
        // Seed the callee's per-slot concrete-locals shadow from the param
        // boxes.  Two distinct consumers, gated differently:
        //
        // 1. Register-to-register fold (`!try_multiframe` only): a branchless
        //    leaf inlined without a materialized virtual frame folds its own
        //    `getarrayitem_vable_r` / `setarrayitem_vable_r` through the per-slot
        //    OpRef shadow (`fold_frame_reg` + `set_opref`), so the callee's first
        //    LOAD_FAST of a param folds to the arg OpRef instead of reading its
        //    unseeded frame box.  A `try_multiframe` callee HAS a real virtual
        //    frame, so this fold must stay off (its reads go through the frame).
        //
        // 2. Concrete-locals fallback (BOTH paths): the `getarrayitem_vable`
        //    read fallback and the `setarrayitem_vable` re-seed
        //    (`getarrayitem_vable_via_metainterp` / `setarrayitem_vable`) supply
        //    the local's recording-time concrete when the heapcache holds no
        //    entry.  A `try_multiframe` callee's param reads forward through the
        //    heapcache only until an in-callee may-force op runs
        //    `reset_keep_likely_virtuals` (heapcache.py:183) and clears the array
        //    cache; the post-call LOAD_FAST re-read then misses and the branch
        //    value goes non-concrete (`GotoIfNotValueNotConcrete`).  Seeding the
        //    shadow for `try_multiframe` too gives that re-read a fallback — the
        //    analog of the callee MIFrame register box RPython reads
        //    `box.getint()` off (registers survive a residual call; the heapcache
        //    does not).  STORE_FAST keeps both maps current (the
        //    `setarrayitem_vable` handler re-seeds `set_opref` + `set_concrete`
        //    on every store).
        //
        //    `set_opref` is seeded on BOTH paths (not just the fold): the read
        //    fallback re-resolves the slot's concrete through `concrete_of_opref`
        //    on this OpRef — a GC-forwarded, rooted channel — in preference to
        //    the raw `Value` copy in `concrete`, which the trace-ref walker does
        //    not visit and so dangles if a minor collection moves a nursery Ref
        //    across the may-force residual.  The fold consumer stays gated by
        //    `fold_frame_reg` (kept `!try_multiframe`), so seeding `opref` here is
        //    inert for folding on the multiframe path.
        //
        // Inert when `callee_portal_frame_reg == u16::MAX` (flip OFF / frame reg
        // unresolved).
        if callee_portal_frame_reg != u16::MAX {
            if !try_multiframe {
                sub_wc.callee_shadow.as_mut().unwrap().fold_frame_reg = callee_portal_frame_reg;
            }
            for i in 0..nparams {
                let slot = i as i64;
                let value = callee_args[i];
                let concrete = sub_wc
                    .trace_ctx
                    .concrete_of_opref(callee_args[i])
                    .unwrap_or(majit_ir::Value::Void);
                let shadow = sub_wc.callee_shadow.as_mut().unwrap();
                shadow.set_opref(slot, value);
                shadow.set_concrete(callee_portal_frame_reg, slot, concrete);
            }
        }
        // Capture a depth-1 live callee before these guards drop. This is the
        // two-frame specialization of `run_blackhole_interp_to_cancel_tracing`:
        // `_copy_data_from_miframe` preserves the callee's own position and
        // live registers instead of collapsing it onto the caller frame.
        let result = walk(body.code, 0, &mut sub_wc);
        let midbody_abort = match &result {
            Err(DispatchError::AbortPermanentMarkerReached { pc }) => {
                Some((*pc, MidBodyAbortKind::Marker))
            }
            Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc })
                if fbw_structural_abort_opcode_is_effect_free(*pc) =>
            {
                Some((*pc, MidBodyAbortKind::Structural))
            }
            _ => None,
        };
        if let Some((abort_pc, abort_kind)) = midbody_abort {
            if is_top_inline
                && !unjournaled_before_subwalk
                && fbw_executed_effect_count() != executed_effects_before
            {
                let payload = (|| {
                    let (outer_jitcode_index, call_jitcode_pc) = abort_flush_call_jitcode_coord?;
                    let callee_pjc = crate::state::pyjitcode_for_code(w_code)?;
                    let metadata = &callee_pjc.metadata;
                    let callee_py_pc = python_pc_for_jitcode_pc(metadata, abort_pc) as usize;
                    let anchor_ok = match abort_kind {
                        MidBodyAbortKind::Structural => {
                            exact_floor_segment_anchor(metadata, callee_py_pc, abort_pc)
                        }
                        MidBodyAbortKind::Marker => portal_marker_first_jit_anchor(
                            metadata,
                            metadata.built_as_portal,
                            metadata.portal_frame_reg,
                            callee_perfn_descrs,
                            body.code,
                            callee_py_pc,
                            abort_pc,
                            |op_pc| python_pc_for_jitcode_pc(metadata, op_pc) as usize,
                        ),
                    };
                    if !anchor_ok {
                        return None;
                    }
                    let raw = unsafe {
                        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                            as *const pyre_interpreter::CodeObject
                    };
                    if raw.is_null() {
                        return None;
                    }
                    let callee_code = unsafe { &*raw };
                    if pyre_interpreter::pyframe::code_flags_make_generator(callee_code.flags)
                        || !callee_code.cellvars.is_empty()
                        || !callee_code.freevars.is_empty()
                        || !unsafe { pyre_interpreter::function_get_closure(callable) }.is_null()
                    {
                        return None;
                    }
                    let depth_twin = callee_pjc.depth_for_jitcode_pc_pred(abort_pc);
                    let Some(depth) = depth_twin else {
                        return None;
                    };
                    let depth = depth as usize;
                    let nlocals = callee_code.varnames.len();
                    let pcdep_twin = callee_pjc.pcdep_for_jitcode_pc(abort_pc);
                    let Some(entries) = pcdep_twin else {
                        return None;
                    };
                    let mut live_stack = Vec::with_capacity(depth);
                    for rel in 0..depth {
                        let semantic_slot = nlocals + rel;
                        let register_value =
                            crate::state::semantic_slot_color_for_ref_slot(&entries, semantic_slot)
                                .and_then(|color| sub_wc.concrete_registers_r.get(color).copied());
                        let value = register_value.or_else(|| {
                            (metadata.built_as_portal && abort_kind == MidBodyAbortKind::Marker)
                                .then(|| {
                                    callee_vable_ref_at(
                                        sub_wc.callee_shadow.as_ref(),
                                        metadata.portal_frame_reg,
                                        semantic_slot,
                                    )
                                })
                                .flatten()
                        })?;
                        if !matches!(value, ConcreteValue::Ref(r) if !r.is_null()) {
                            return None;
                        }
                        live_stack.push(value);
                    }
                    if live_stack.len() != depth {
                        return None;
                    }
                    let lv = crate::state::liveness_for(raw);
                    let mut live_locals = vec![None; nlocals];
                    for (slot, dst) in live_locals.iter_mut().enumerate() {
                        if !lv.is_local_live(callee_py_pc, slot) {
                            continue;
                        }
                        let value = sub_wc
                            .callee_shadow
                            .as_ref()
                            .and_then(|shadow| shadow.concrete.get(&(slot as i64)).copied())
                            .filter(|entry| entry.frame_reg == metadata.portal_frame_reg)
                            .and_then(|entry| match entry.value {
                                Value::Ref(r) => Some(ConcreteValue::Ref(
                                    r.as_usize() as pyre_object::PyObjectRef
                                )),
                                Value::Int(v) => Some(ConcreteValue::Int(v)),
                                Value::Float(v) => Some(ConcreteValue::Float(v)),
                                Value::Void => None,
                            })
                            .or_else(|| callee_arg_concretes.get(slot).copied())?;
                        if matches!(value, ConcreteValue::Null | ConcreteValue::Bool(_)) {
                            return None;
                        }
                        *dst = Some(value);
                    }
                    let Some(ConcreteValue::Ref(x_arg)) = callee_arg_concretes.first().copied()
                    else {
                        return None;
                    };
                    Some(MidBodyPayload {
                        abort_kind,
                        outer_jitcode_index,
                        call_jitcode_pc,
                        call_stack_len: arg_concretes.len(),
                        callee_jitcode_index: callee_pjc.jitcode.index() as u32,
                        abort_jitcode_pc: abort_pc,
                        w_code: w_code as pyre_object::PyObjectRef,
                        w_globals: unsafe { pyre_interpreter::function_get_globals_obj(callable) },
                        x_arg,
                        live_locals,
                        live_stack,
                        return_value: pyre_object::PY_NULL,
                    })
                })();
                if let Some(payload) = payload {
                    fbw_set_midbody_abort_resume(payload);
                }
            }
        }
        let class_of_last_exc_is_const = sub_wc.fbw_mode.class_of_last_exc_is_const;
        (result, class_of_last_exc_is_const)
    };
    // RPython has one MetaInterp shared by every MIFrame.  The sub-walk uses
    // a copied FbwWalkMode only to satisfy Rust's nested borrow, so write the
    // MetaInterp-owned exception state back across the frame boundary.
    ctx.fbw_mode.class_of_last_exc_is_const = callee_class_of_last_exc_is_const;
    let (outcome, _end_pc) = match callee_outcome {
        Ok(v) => v,
        Err(e) => {
            if std::env::var("PYRE_FBW_INLINE_DIAG").is_ok() {
                eprintln!("[inline-abort] callee sub-walk err: {e:?}");
            }
            // gh#467: a supported abort fired inside this top-level inline
            // sub-walk.  If the callee executed NO concrete effect and no
            // unjournaled effect existed before the attempt, latch the outer
            // CALL boundary so the walk driver flushes the outer frame there
            // and re-executes the call FORWARD — running the callee from scratch
            // in the interpreter — instead of rolling back and replaying the
            // loop from entry, which double-applies the non-journaled pre-CALL
            // store.  Discarding a zero-executed-effect callee attempt and
            // re-running its CALL is observationally identical to upstream
            // never having inlined it: tracing aborts and `switch_to_blackhole`
            // re-runs the call (`pyjitpl.py`; gh#467).  The operand
            // stack the CALL opcode expects (`[callable, null_or_self,
            // args...]`) is re-read from the (now GC-forwarded) outer registers,
            // not the pre-sub-walk `arg_concretes`, so it is current after the
            // sub-walk's allocations.  Any doubt keeps the legacy replay — the
            // honest residual (the inner-frame rebuild is #126/#215).
            if matches!(
                e,
                DispatchError::AbortPermanentMarkerReached { .. }
                    | DispatchError::LoopBearingCalleeInlineUnsupported { .. }
            ) && is_top_inline
                && !unjournaled_before_subwalk
                && fbw_executed_effect_count() == executed_effects_before
            {
                if let Some((outer_jitcode_index, call_jitcode_pc)) = abort_flush_call_jitcode_coord
                {
                    if let Some(stack) = reconstructed_all_ref_call_stack(code, op, ctx) {
                        fbw_set_abort_call_resume(outer_jitcode_index, call_jitcode_pc, stack);
                    }
                }
            }
            return Err(e);
        }
    };

    match outcome {
        DispatchOutcome::SubReturn {
            result: Some(value),
        } => {
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
            if require_str_result
                && !matches!(
                    concrete_for_shadow,
                    ConcreteValue::Ref(obj) if !obj.is_null() && unsafe { pyre_object::is_str(obj) }
                )
            {
                // descroperation.py checks the app-level result before
                // returning from `space.str` / `space.repr`. Re-run the
                // original builtin call at the caller boundary so the
                // interpreter raises its faithful TypeError; the inlined
                // body has no committed concrete effect at this point.
                if is_top_inline
                    && !unjournaled_before_subwalk
                    && fbw_executed_effect_count() == executed_effects_before
                {
                    if let Some((outer_jitcode_index, call_jitcode_pc)) =
                        abort_flush_call_jitcode_coord
                    {
                        if let Some(stack) = reconstructed_all_ref_call_stack(code, op, ctx) {
                            fbw_set_abort_call_resume(outer_jitcode_index, call_jitcode_pc, stack);
                        }
                    }
                }
                return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
            }
            match dst_bank {
                'r' => write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?,
                'i' => write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?,
                'v' => {}
                _ => return Ok(None),
            }
            Ok(Some((DispatchOutcome::Continue, op.next_pc)))
        }
        DispatchOutcome::SubReturn { result: None } => {
            if dst_bank == 'v' {
                Ok(Some((DispatchOutcome::Continue, op.next_pc)))
            } else {
                Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
            }
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.last_exc_value = Some(exc);
                ctx.last_exc_value_concrete = exc_concrete;
                Ok(Some((DispatchOutcome::Continue, target)))
            } else {
                Ok(Some((
                    DispatchOutcome::SubRaise { exc, exc_concrete },
                    op.next_pc,
                )))
            }
        }
        DispatchOutcome::SubLoopCalleeCallAssembler { token, target_pc } => {
            // CODEX1 parity: decline the CA inline when the prologue sub-walk
            // mutated the heap (a journaled list store, or an unjournaled
            // effect newly set during the sub-walk).  Emitting the CA here
            // would re-run the whole call via the residual executor, applying
            // those side effects twice at trace time.  A side-effect-free
            // prologue (the common loop-setup-only case) still inlines.
            if fbw_store_journal_len() > prologue_journal_before
                || (!unjournaled_before_subwalk && fbw_has_unjournaled_effect())
            {
                return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
            }
            emit_walker_loop_callee_call_assembler(
                ctx,
                op,
                funcptr,
                r_args,
                call_descr,
                dst_bank,
                dst,
                ca_callee_frame,
                ca_callee_ec,
                ca_nlocals,
                token,
                target_pc,
            )
        }
        other => Ok(Some((other, op.next_pc))),
    }
}

/// Route `str(exc)` / `repr(exc)` through an app-level exception override.
/// Pyre's exact `str` type call follows `str_descr_new` → `builtin_str` →
/// `exc_user_dunder_obj`; the builtin `repr` follows `builtin_repr` →
/// `py_repr_obj`. Both paths look up the receiver dunder before builtin
/// exception formatting. This is the walker counterpart of
/// `descroperation.py`'s `space.lookup` + `get_and_call_function`: pin the
/// promoted exception class, then enter the ordinary resolved-callee inline
/// plumbing with the receiver as `self`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_exception_string_override<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if r_args.len() != 3 {
        return Ok(None);
    }
    let Some(concrete_callable) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if walker_concrete_ref_object(ctx, r_args[1]).is_some() {
        return Ok(None);
    }
    let Some(concrete_receiver) = walker_concrete_ref_object(ctx, r_args[2]) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_exception(concrete_receiver) } {
        return Ok(None);
    }

    let dunder = if std::ptr::eq(
        concrete_callable,
        pyre_interpreter::typedef::gettypeobject(&pyre_object::pyobject::STR_TYPE),
    ) {
        "__str__"
    } else if pyre_interpreter::builtins::is_builtin_repr_function(concrete_callable) {
        "__repr__"
    } else {
        return Ok(None);
    };

    let w_class = unsafe { (*concrete_receiver).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        return Ok(None);
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        return Ok(None);
    }
    let Some(method) = (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, dunder) })
    else {
        return Ok(None);
    };
    let Some(base_exception) = pyre_interpreter::builtins::lookup_exc_class("BaseException") else {
        return Ok(None);
    };
    let Some(default_method) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(base_exception, dunder) })
    else {
        return Ok(None);
    };
    if std::ptr::eq(method, default_method) {
        return Ok(None);
    }
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        return Ok(None);
    };

    let Some(body) = crate::state::sub_jitcode_body_for_code(w_code) else {
        return Ok(None);
    };
    if !exception_string_override_straight_line(body.code) {
        return Ok(None);
    }
    // A nested Python call in the override body (e.g. `return repr(self.args)`)
    // cannot be inlined on this bounded route: recording the callee's own
    // residual and its guard-resume snapshot aborts mid-trace, discarding the
    // whole loop instead of declining.  Keep such a body on the residual
    // dispatch path where the interpreter owns the nested frame.
    let Some((override_descr_refs, _, _)) = crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return Ok(None);
    };
    if exception_string_override_has_nested_call(body.code, override_descr_refs) {
        return Ok(None);
    }

    // A straight-line, effect-free override can be sampled before any IR is
    // emitted. If its observed result is not a string, decline to the original
    // builtin residual so the interpreter's result check raises TypeError and
    // the exception-handler loop remains traceable. More complex bodies are
    // not executed speculatively; their inlined result is guarded below.
    if let (Some(body), Some((callee_descr_refs, _, _))) = (
        crate::state::sub_jitcode_body_for_code(w_code),
        crate::state::sub_jitcode_descr_pool_for_code(w_code),
    ) {
        if exception_string_override_sample_safe(body.code, callee_descr_refs) {
            let sampled = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(method, &[concrete_receiver])
            };
            let sampled_is_acceptable = matches!(sampled, Ok(result)
                if !result.is_null() && unsafe { pyre_object::is_str(result) });
            if !sampled_is_acceptable {
                return Ok(None);
            }
        }
    }

    let arg_concretes = vec![
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_receiver),
    ];
    let _exception_string_inline = ExceptionStringInlineGuard::enter();
    let Some(inlined) = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        r_args[0],
        concrete_callable,
        arg_concretes,
        vec![r_args[2]],
        vec![ConcreteValue::Ref(concrete_receiver)],
        true,
        w_code,
        nparams,
        has_closure,
        Some((r_args[2], concrete_receiver, w_class, version_tag)),
        None,
        true,
        true,
    )?
    else {
        return Ok(None);
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r[dst];
        let str_type = &pyre_object::pyobject::STR_TYPE as *const _ as i64;
        let str_type_const = ctx.trace_ctx.const_int(str_type);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[result, str_type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .class_now_known(result, str_type);
    }
    Ok(Some(inlined))
}

/// Inline a plain Python `__add__` after the numeric BINARY_OP
/// specializations decline. The receiver class and its version tag pin the
/// descriptor lookup, matching `try_dispatch_binary_special`'s forward arm.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_user_binop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    op_tag: i64,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 2 {
        return Ok(None);
    }

    let Some(pyre_interpreter::bytecode::BinaryOperator::Add) =
        pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag)
    else {
        return Ok(None);
    };
    let dunder = "__add__";

    let lhs = r_args[0];
    let rhs = r_args[1];
    let Some(concrete_lhs) = walker_concrete_ref_object(ctx, lhs) else {
        return Ok(None);
    };
    let Some(concrete_rhs) = walker_concrete_ref_object(ctx, rhs) else {
        return Ok(None);
    };

    // A tagged immediate is an exact builtin `int` with C-level operator slots:
    // it has no heap `ob_type`/`w_class` to pin, and its dunder is not inlinable
    // Python code.  Decline before the concrete derefs below, which would fault
    // on the immediate (`typedef::r#type` stays the tagged-safe typing path).
    // Inert behind `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(concrete_lhs)
            || pyre_object::tagged_int::is_tagged_int(concrete_rhs))
    {
        return Ok(None);
    }

    let w_class = unsafe { (*concrete_lhs).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        return Ok(None);
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        return Ok(None);
    }

    let Some(w_typ_r) = pyre_interpreter::typedef::r#type(concrete_rhs) else {
        return Ok(None);
    };
    if !std::ptr::eq(w_class, w_typ_r)
        && unsafe { pyre_object::typeobject::w_type_issubtype(w_typ_r, w_class) }
    {
        return Ok(None);
    }

    let Some(method) = (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, dunder) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        return Ok(None);
    };
    if nparams != 2 {
        return Ok(None);
    }

    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_lhs),
        ConcreteValue::Ref(concrete_rhs),
    ];
    let method_const = ctx.trace_ctx.const_ref(method as i64);
    let Some(inlined) = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        method_const,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![lhs, rhs],
        vec![
            ConcreteValue::Ref(concrete_lhs),
            ConcreteValue::Ref(concrete_rhs),
        ],
        true,
        w_code,
        nparams,
        has_closure,
        Some((lhs, concrete_lhs, w_class, version_tag)),
        Some((rhs, concrete_rhs, w_typ_r)),
        false,
        false,
    )?
    else {
        return Ok(None);
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r[dst];
        if matches!(
            concrete_from_recorded_opref(ctx, result),
            ConcreteValue::Ref(obj)
                if std::ptr::eq(obj, pyre_object::special::w_not_implemented())
        ) {
            return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
        }
        if !result.is_constant() {
            let not_implemented = ctx
                .trace_ctx
                .const_ref(pyre_object::special::w_not_implemented() as i64);
            let is_not_implemented = ctx
                .trace_ctx
                .record_op(OpCode::PtrEq, &[result, not_implemented]);
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardFalse, &[is_not_implemented])?;
        }
    }
    Ok(Some(inlined))
}

/// Inline a plain Python rich-compare dunder after the numeric COMPARE_OP
/// specializations decline. The receiver class and its version tag pin the
/// descriptor lookup; a proper-subclass rhs declines so its reflected dunder
/// retains priority.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_user_compareop<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    op_tag: i64,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || r_args.len() != 2 {
        return Ok(None);
    }

    let Some(cmp_op) = pyre_interpreter::runtime_ops::compare_op_from_tag(op_tag) else {
        return Ok(None);
    };
    // Forward rich-compare dunder only; a proper-subclass rhs declines below so
    // its reflected dunder (__lt__/__gt__, __le__/__ge__, __eq__/__ne__ self)
    // keeps priority, matching try_compare_override's forward-first dispatch.
    let dunder = match cmp_op {
        pyre_interpreter::bytecode::ComparisonOperator::Less => "__lt__",
        pyre_interpreter::bytecode::ComparisonOperator::LessOrEqual => "__le__",
        pyre_interpreter::bytecode::ComparisonOperator::Greater => "__gt__",
        pyre_interpreter::bytecode::ComparisonOperator::GreaterOrEqual => "__ge__",
        pyre_interpreter::bytecode::ComparisonOperator::Equal => "__eq__",
        pyre_interpreter::bytecode::ComparisonOperator::NotEqual => "__ne__",
    };

    let lhs = r_args[0];
    let rhs = r_args[1];
    let Some(concrete_lhs) = walker_concrete_ref_object(ctx, lhs) else {
        return Ok(None);
    };
    let Some(concrete_rhs) = walker_concrete_ref_object(ctx, rhs) else {
        return Ok(None);
    };

    // A tagged immediate is an exact builtin `int` with C-level operator slots:
    // it has no heap `ob_type`/`w_class` to pin, and its dunder is not inlinable
    // Python code.  Decline before the concrete derefs below, which would fault
    // on the immediate (`typedef::r#type` stays the tagged-safe typing path).
    // Inert behind `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && (pyre_object::tagged_int::is_tagged_int(concrete_lhs)
            || pyre_object::tagged_int::is_tagged_int(concrete_rhs))
    {
        return Ok(None);
    }

    let w_class = unsafe { (*concrete_lhs).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        return Ok(None);
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        return Ok(None);
    }

    let Some(w_typ_r) = pyre_interpreter::typedef::r#type(concrete_rhs) else {
        return Ok(None);
    };
    if !std::ptr::eq(w_class, w_typ_r)
        && unsafe { pyre_object::typeobject::w_type_issubtype(w_typ_r, w_class) }
    {
        return Ok(None);
    }

    let Some(method) = (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, dunder) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        return Ok(None);
    };
    if nparams != 2 {
        return Ok(None);
    }

    let arg_concretes = vec![
        ConcreteValue::Ref(method),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_lhs),
        ConcreteValue::Ref(concrete_rhs),
    ];
    let method_const = ctx.trace_ctx.const_ref(method as i64);
    let Some(inlined) = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        method_const,
        r_args,
        call_descr,
        'r',
        dst,
        method,
        method_const,
        method,
        arg_concretes,
        vec![lhs, rhs],
        vec![
            ConcreteValue::Ref(concrete_lhs),
            ConcreteValue::Ref(concrete_rhs),
        ],
        true,
        w_code,
        nparams,
        has_closure,
        Some((lhs, concrete_lhs, w_class, version_tag)),
        Some((rhs, concrete_rhs, w_typ_r)),
        false,
        false,
    )?
    else {
        return Ok(None);
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r[dst];
        if matches!(
            concrete_from_recorded_opref(ctx, result),
            ConcreteValue::Ref(obj)
                if std::ptr::eq(obj, pyre_object::special::w_not_implemented())
        ) {
            return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
        }
        if !result.is_constant() {
            let not_implemented = ctx
                .trace_ctx
                .const_ref(pyre_object::special::w_not_implemented() as i64);
            let is_not_implemented = ctx
                .trace_ctx
                .record_op(OpCode::PtrEq, &[result, not_implemented]);
            walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardFalse, &[is_not_implemented])?;
        }
    }
    Ok(Some(inlined))
}

/// Allocate the callee's three symbolic register banks for a sub-walk
/// entered through any `inline_call_*` arm.
///
/// Each bank is sized to `num_regs_X + constants_X.len()`
/// (RPython `JitCode.num_regs_and_consts_X`) so callee bytecode that
/// reads the post-regs constant window (indices
/// `[num_regs_X, num_regs_and_consts_X)`) finds a populated slot.
/// Constant slots are filled via `TraceCtx::const_int` / `const_ref` /
/// `const_float`, matching RPython
/// `pyjitpl.py MIFrame.copy_constants`.
///
/// Also returns Ref- and Int-bank concrete shadows sized to match
/// `registers_r` / `registers_i`.  Constant slots seed their concrete
/// directly from the pools: `ConcreteValue::Int(v)` from
/// `body.constants_i` (so a `goto_if_not/iL` reading a constant input
/// can fold the branch) and `ConcreteValue::Ref(v)` from
/// `body.constants_r` — a Ref constant's runtime value IS the pooled
/// object pointer (kept alive by the jitcode), and the nested
/// call-inline gate (`try_walker_inline_user_call`) reads the callable
/// through this shadow when a callee body calls another function
/// through its own baked const-pool callable.
pub(crate) fn allocate_callee_register_banks(
    body: &SubJitCodeBody,
    trace_ctx: &mut TraceCtx,
) -> (
    Vec<OpRef>,
    Vec<OpRef>,
    Vec<OpRef>,
    Vec<ConcreteValue>,
    Vec<ConcreteValue>,
) {
    let total_r = body.num_regs_r + body.constants_r.len();
    let total_i = body.num_regs_i + body.constants_i.len();
    let total_f = body.num_regs_f + body.constants_f.len();
    let mut regs_r = vec![OpRef::NONE; total_r];
    let mut regs_i = vec![OpRef::NONE; total_i];
    let mut regs_f = vec![OpRef::NONE; total_f];
    let mut concrete_r = vec![ConcreteValue::Null; total_r];
    let mut concrete_i = vec![ConcreteValue::Null; total_i];
    for (i, &v) in body.constants_i.iter().enumerate() {
        regs_i[body.num_regs_i + i] = trace_ctx.const_int(v);
        concrete_i[body.num_regs_i + i] = ConcreteValue::Int(v);
    }
    for (i, &v) in body.constants_r.iter().enumerate() {
        regs_r[body.num_regs_r + i] = trace_ctx.const_ref(v);
        concrete_r[body.num_regs_r + i] =
            ConcreteValue::Ref(v as usize as pyre_object::PyObjectRef);
    }
    for (i, &v) in body.constants_f.iter().enumerate() {
        regs_f[body.num_regs_f + i] = trace_ctx.const_float(v);
    }
    (regs_r, regs_i, regs_f, concrete_r, concrete_i)
}

/// Seed a callee jitcode's register banks with positional args and walk
/// its body, returning the callee's terminal [`DispatchOutcome`]
/// (`SubReturn` / `SubRaise` / `Terminate` / `SwitchToBlackhole`).
///
/// Shared descent core of the `inline_call_*` handlers
/// ([`dispatch_inline_call_dr_kind`], `_dir`, `_dirf`) — they read the
/// callee index + arglists from the caller bytecode, then delegate the
/// bank allocation, arity check, arg seeding, sub-`WalkContext`
/// construction, and `walk()` to here. A trace-time specialization can
/// also call this directly to synthesize a descent into a charon helper
/// body (e.g. `w_list_append`), passing args it already holds rather
/// than reading them from bytecode.
///
/// `pc` is the caller-site pc, used only for arity-mismatch error
/// reporting. An empty arg slice for an unused bank passes its arity
/// check trivially. The callee runs with `is_top_level == false` and
/// inherits the caller's descr pool + sub-jitcode lookup (RPython
/// `pyjitpl.py setup_call(argboxes_i, argboxes_r, argboxes_f)`).
/// Only Ref-bank concrete shadows are seeded — matching the
/// `inline_call_*` handlers, which thread `ref_arg_concretes` but no
/// Int/Float concrete shadows across the frame boundary.
pub(crate) fn run_sub_jitcode_walk<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    pc: usize,
    sub_body: &SubJitCodeBody,
    int_args: &[OpRef],
    int_arg_concretes: &[ConcreteValue],
    ref_args: &[OpRef],
    ref_arg_concretes: &[ConcreteValue],
    float_args: &[OpRef],
) -> Result<DispatchOutcome, DispatchError> {
    let (
        mut callee_regs_r,
        mut callee_regs_i,
        mut callee_regs_f,
        mut callee_concrete_r,
        mut callee_concrete_i,
    ) = allocate_callee_register_banks(sub_body, ctx.trace_ctx);

    if int_args.len() > sub_body.num_regs_i {
        return Err(DispatchError::InlineCallIntArityMismatch {
            pc,
            provided: int_args.len(),
            callee_num_regs_i: sub_body.num_regs_i,
        });
    }
    if ref_args.len() > sub_body.num_regs_r {
        return Err(DispatchError::InlineCallArityMismatch {
            pc,
            provided: ref_args.len(),
            callee_num_regs_r: sub_body.num_regs_r,
        });
    }
    if float_args.len() > sub_body.num_regs_f {
        return Err(DispatchError::InlineCallFloatArityMismatch {
            pc,
            provided: float_args.len(),
            callee_num_regs_f: sub_body.num_regs_f,
        });
    }
    for (i, arg) in int_args.iter().enumerate() {
        callee_regs_i[i] = *arg;
    }
    for (i, arg) in ref_args.iter().enumerate() {
        callee_regs_r[i] = *arg;
    }
    for (i, arg) in float_args.iter().enumerate() {
        callee_regs_f[i] = *arg;
    }
    // Seed the callee's concrete shadows from the caller's per-arg
    // shadows (`setup_call` parity for the Int + Ref banks; the Float
    // bank has no concrete shadow companion).  A callee body folds a
    // `goto_if_not/iL` / `switch/id` over a concrete int arg, or a
    // `guard_class` over a concrete ref arg, only when its shadow is
    // seeded here.
    for (i, concrete) in int_arg_concretes.iter().enumerate() {
        callee_concrete_i[i] = *concrete;
    }
    for (i, concrete) in ref_arg_concretes.iter().enumerate() {
        callee_concrete_r[i] = *concrete;
    }

    let ((callee_outcome, _callee_end_pc), callee_class_of_last_exc_is_const) = {
        let mut sub_wc = WalkContext {
            callee_shadow: None,
            inline_callee_consts: None,
            fbw_mode: ctx.fbw_mode,
            session: ctx.session,
            registers_r: &mut callee_regs_r,
            registers_i: &mut callee_regs_i,
            registers_f: &mut callee_regs_f,
            concrete_registers_r: &mut callee_concrete_r,
            concrete_registers_i: &mut callee_concrete_i,
            descr_refs: ctx.descr_refs,
            raw_descrs: ctx.raw_descrs,
            is_authoritative_executor: ctx.is_authoritative_executor,
            trace_ctx: ctx.trace_ctx,
            done_with_this_frame_descr_ref: ctx.done_with_this_frame_descr_ref.clone(),
            done_with_this_frame_descr_int: ctx.done_with_this_frame_descr_int.clone(),
            done_with_this_frame_descr_float: ctx.done_with_this_frame_descr_float.clone(),
            done_with_this_frame_descr_void: ctx.done_with_this_frame_descr_void.clone(),
            exit_frame_with_exception_descr_ref: ctx.exit_frame_with_exception_descr_ref.clone(),
            is_top_level: false,
            sub_jitcode_lookup: ctx.sub_jitcode_lookup,
            last_exc_value: None,
            last_exc_value_concrete: ConcreteValue::Null,
            entry_py_pc: ctx.entry_py_pc,
            outer_resume_marker_jit_pc: ctx.outer_resume_marker_jit_pc,
            outer_jitcode_index: ctx.outer_jitcode_index,
            outer_active_boxes: ctx.outer_active_boxes.clone(),
            store_subscr_fn_addr: ctx.store_subscr_fn_addr,
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
        if let Some(frame) = ActiveResumeFrame::current(ctx.session, ctx.fbw_mode.snapshot_sym) {
            if frame.body_matches(sub_body) {
                seed_callee_vstack_mirror(&mut sub_wc, &frame);
            }
        }
        let (outcome, end_pc) = walk(sub_body.code, 0, &mut sub_wc)?;
        (
            (outcome, end_pc),
            sub_wc.fbw_mode.class_of_last_exc_is_const,
        )
    };
    // `MetaInterp.class_of_last_exc_is_const` is shared across the RPython
    // framestack.  Preserve the callee's final state when leaving this Rust
    // sub-context instead of manufacturing a new value at the caller catch.
    ctx.fbw_mode.class_of_last_exc_is_const = callee_class_of_last_exc_is_const;
    Ok(callee_outcome)
}

/// Operand layout `dR>X`:
///   2B descr index + 1B varlen + N×1B Ref args + 1B `>X` dst.
///
/// RPython parity: `pyjitpl.py _opimpl_inline_call*`. The
/// `_X` suffix is the callee's *return kind* — e.g. `_opimpl_inline_call_r_i`
/// dispatches an inline call whose callee body returns via
/// `int_return/i`. Walker semantics are otherwise identical to the
/// `_r_r` arm (which originally landed inline; this helper extracts the
/// shared body so kind variants can share the dispatch logic).
///
/// `dst_bank` selects where the SubReturn value lands:
/// * `'r'`: caller's `registers_r[dst]` — pairs with callee `ref_return/r`.
/// * `'i'`: caller's `registers_i[dst]` — pairs with callee `int_return/i`.
/// * `'f'`: would pair with callee `float_return/f` — not handled by
///   this helper because the codewriter doesn't emit a `dR>f` shape
///   (float return paths use the `dIRF` arglist family).
///
/// `kind_label` mirrors `dst_bank` as a static `&str` for typed-error
/// reporting (`RegisterOutOfRange::bank`).
pub(crate) fn dispatch_inline_call_dr_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    let (args, arg_width) = read_ref_var_list(code, op, 2, ctx)?;
    let arg_concretes = read_ref_var_list_concrete(code, op, 2, ctx);

    let callee_outcome =
        run_sub_jitcode_walk(ctx, op.pc, &sub_body, &[], &[], &args, &arg_concretes, &[])?;

    match callee_outcome {
        DispatchOutcome::SubReturn {
            result: Some(value),
        } => {
            if dst_bank == 'v' {
                // `inline_call_r_v/dR`
                // (`bhimpl_inline_call_r_v` `blackhole.py`)
                // expects a void-return callee. A `Some` return here is
                // a codewriter shape mismatch.
                return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
            }
            let dst = code[op.pc + 1 + 2 + arg_width] as usize;
            // inline_call_* dst writeback — `value` is the callee's
            // SubReturn OpRef.  The callee's matching concrete shadow
            // was dropped at sub-walk exit; `concrete_of_opref` still
            // sees through to `constants.get_value` for callees that
            // return a constant (e.g. `LoadConst` tail), so route via
            // the unified shadow channel.  Non-constant returns surface
            // as the sentinel `GcRef(usize::MAX)` → Null fallback.
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
            match dst_bank {
                'r' => {
                    write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                'i' => {
                    write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                _ => unreachable!(
                    "dispatch_inline_call_dr_kind dst_bank must be 'r', 'i' or 'v' (\
                     codewriter does not emit dR>f shape today)"
                ),
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        DispatchOutcome::SubReturn { result: None } => {
            if dst_bank == 'v' {
                // `inline_call_r_v/dR` expects exactly this — callee
                // exits via `void_return/`, no SubReturn writeback.
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            // Same shape contract as `_r_r`: a `_r_<X>` variant promises
            // a non-void result for the dst's `>X` slot. A void return
            // reaching here is a codewriter shape mismatch.
            Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.last_exc_value = Some(exc);
                // Thread the callee's concrete
                // exception across the frame boundary.  Without this a
                // downstream `raise/r` / `reraise/` in the caller's
                // handler would read `Null` and skip GUARD_CLASS,
                // losing the class-known pin that the callee's leg had
                // already established.
                ctx.last_exc_value_concrete = exc_concrete;
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate => Ok((DispatchOutcome::Terminate, op.next_pc)),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::CloseLoop { .. } => {
            // An inlined callee body must not close a loop — see
            // `SubWalkClosedLoop`.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::CompileTracePending { .. } => {
            // The compile_trace attempt is gated on `is_top_level`
            // (sub-walks run with `is_top_level == false`), so a callee
            // body can never surface it; fail loud like the CloseLoop
            // arm if that invariant ever breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SubLoopCalleeCallAssembler { .. } => {
            // The loop-callee CALL_ASSEMBLER request is surfaced from a
            // multi-frame inline at a `residual_call` site and consumed by
            // `try_walker_inline_user_call`; it cannot reach the `inline_call_*`
            // jitcode-op path. Fail loud (safe decline) if that invariant ever
            // breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// `inline_call_ir_<X>/dIR>X` handler shared by `dIR>i` (Int result)
/// and `dIR>r` (Ref result). Same control-flow shape as
/// [`dispatch_inline_call_dr_kind`], extended with an I-list arglist
/// preceding the R-list.
///
/// Operand layout `dIR>X`:
///   2B descr index +
///   1B I-len + N×1B int args +
///   1B R-len + M×1B ref args +
///   1B `>X` dst.
///
/// RPython parity: `pyjitpl.py _opimpl_inline_call*` —
/// kind-aware variants call `setup_call(argboxes_i, argboxes_r,
/// argboxes_f)` which distributes args into the callee's typed banks
/// (`pyjitpl.py`).
///
/// `dst_bank` selects where the SubReturn value lands: `'r'` writes to
/// `registers_r[dst]` (paired with callee `ref_return/r`), `'i'`
/// writes to `registers_i[dst]` (paired with callee `int_return/i`).
pub(crate) fn dispatch_inline_call_dir_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    // I-list at offset 2 (skip descr).
    let (int_args, int_width) = read_int_var_list(code, op, 2, ctx)?;
    let int_arg_concretes = read_int_var_list_concrete(code, op, 2, ctx);
    // R-list immediately after the I-list.
    let (ref_args, ref_width) = read_ref_var_list(code, op, 2 + int_width, ctx)?;
    let ref_arg_concretes = read_ref_var_list_concrete(code, op, 2 + int_width, ctx);

    let callee_outcome = run_sub_jitcode_walk(
        ctx,
        op.pc,
        &sub_body,
        &int_args,
        &int_arg_concretes,
        &ref_args,
        &ref_arg_concretes,
        &[],
    )?;

    match callee_outcome {
        DispatchOutcome::SubReturn {
            result: Some(value),
        } => {
            if dst_bank == 'v' {
                return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
            }
            // dst register byte sits after descr (2B) + I-list (int_width)
            // + R-list (ref_width) bytes.
            let dst = code[op.pc + 1 + 2 + int_width + ref_width] as usize;
            // See dispatch_inline_call_dr_kind: route the SubReturn
            // OpRef through the unified shadow channel so constant
            // return values propagate.
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
            match dst_bank {
                'r' => {
                    write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                'i' => {
                    write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                _ => unreachable!("dispatch_inline_call_dir_kind dst_bank must be 'r', 'i' or 'v'"),
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        DispatchOutcome::SubReturn { result: None } => {
            if dst_bank == 'v' {
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.last_exc_value = Some(exc);
                // Thread the callee's concrete
                // exception across the frame boundary.  Without this a
                // downstream `raise/r` / `reraise/` in the caller's
                // handler would read `Null` and skip GUARD_CLASS,
                // losing the class-known pin that the callee's leg had
                // already established.
                ctx.last_exc_value_concrete = exc_concrete;
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate => Ok((DispatchOutcome::Terminate, op.next_pc)),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::CloseLoop { .. } => {
            // An inlined callee body must not close a loop — see
            // `SubWalkClosedLoop`.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::CompileTracePending { .. } => {
            // The compile_trace attempt is gated on `is_top_level`
            // (sub-walks run with `is_top_level == false`), so a callee
            // body can never surface it; fail loud like the CloseLoop
            // arm if that invariant ever breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SubLoopCalleeCallAssembler { .. } => {
            // The loop-callee CALL_ASSEMBLER request is surfaced from a
            // multi-frame inline at a `residual_call` site and consumed by
            // `try_walker_inline_user_call`; it cannot reach the `inline_call_*`
            // jitcode-op path. Fail loud (safe decline) if that invariant ever
            // breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// `inline_call_irf_<X>/dIRF>X` handler shared by `dIRF>f` (Float
/// result) and `dIRF>r` (Ref result). Extends
/// [`dispatch_inline_call_dir_kind`] with an F-list arglist following
/// the R-list.
///
/// Operand layout `dIRF>X`:
///   2B descr index +
///   1B I-len + N×1B int args +
///   1B R-len + M×1B ref args +
///   1B F-len + K×1B float args +
///   1B `>X` dst.
///
/// RPython parity: same `pyjitpl.py setup_call(argboxes_i,
/// argboxes_r, argboxes_f)` distribution — all three kind banks
/// populated from the three lists.
///
/// `dst_bank` selects where the SubReturn value lands: `'f'` writes
/// `registers_f[dst]` (paired with callee `float_return/f`), `'r'`
/// writes `registers_r[dst]` (paired with callee `ref_return/r`).
pub(crate) fn dispatch_inline_call_dirf_kind<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &mut WalkContext<'_, '_, Sym>,
    dst_bank: char,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let sub_descr = read_descr(code, op, 0, ctx)?;
    let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
    let jc_descr = sub_descr
        .as_jitcode_descr()
        .ok_or(DispatchError::ExpectedJitCodeDescr {
            pc: op.pc,
            descr_index,
        })?;
    let sub_index = jc_descr.jitcode_index();
    let sub_body =
        (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
            pc: op.pc,
            jitcode_index: sub_index,
        })?;
    let (int_args, int_width) = read_int_var_list(code, op, 2, ctx)?;
    let int_arg_concretes = read_int_var_list_concrete(code, op, 2, ctx);
    let (ref_args, ref_width) = read_ref_var_list(code, op, 2 + int_width, ctx)?;
    let ref_arg_concretes = read_ref_var_list_concrete(code, op, 2 + int_width, ctx);
    let (float_args, float_width) = read_float_var_list(code, op, 2 + int_width + ref_width, ctx)?;

    let callee_outcome = run_sub_jitcode_walk(
        ctx,
        op.pc,
        &sub_body,
        &int_args,
        &int_arg_concretes,
        &ref_args,
        &ref_arg_concretes,
        &float_args,
    )?;

    match callee_outcome {
        DispatchOutcome::SubReturn {
            result: Some(value),
        } => {
            if dst_bank == 'v' {
                return Err(DispatchError::UnexpectedNonVoidSubReturn { pc: op.pc });
            }
            let dst = code[op.pc + 1 + 2 + int_width + ref_width + float_width] as usize;
            // See dispatch_inline_call_dr_kind: route the SubReturn
            // OpRef through the unified shadow channel so constant
            // return values propagate.
            let concrete_for_shadow = concrete_from_recorded_opref(ctx, value);
            match dst_bank {
                'i' => {
                    write_int_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                'r' => {
                    write_ref_reg(ctx, op.pc, dst, value, concrete_for_shadow)?;
                }
                'f' => {
                    let len = ctx.registers_f.len();
                    let slot =
                        ctx.registers_f
                            .get_mut(dst)
                            .ok_or(DispatchError::RegisterOutOfRange {
                                pc: op.pc,
                                reg: dst,
                                len,
                                bank: "f",
                            })?;
                    *slot = value;
                }
                _ => unreachable!(
                    "dispatch_inline_call_dirf_kind dst_bank must be 'i', 'r', 'f' or 'v'"
                ),
            }
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        DispatchOutcome::SubReturn { result: None } => {
            if dst_bank == 'v' {
                return Ok((DispatchOutcome::Continue, op.next_pc));
            }
            Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
        }
        DispatchOutcome::SubRaise { exc, exc_concrete } => {
            if let Some(target) = try_catch_exception_at(code, op.next_pc) {
                ctx.last_exc_value = Some(exc);
                // Thread the callee's concrete
                // exception across the frame boundary.  Without this a
                // downstream `raise/r` / `reraise/` in the caller's
                // handler would read `Null` and skip GUARD_CLASS,
                // losing the class-known pin that the callee's leg had
                // already established.
                ctx.last_exc_value_concrete = exc_concrete;
                Ok((DispatchOutcome::Continue, target))
            } else {
                Ok((DispatchOutcome::SubRaise { exc, exc_concrete }, op.next_pc))
            }
        }
        DispatchOutcome::Terminate => Ok((DispatchOutcome::Terminate, op.next_pc)),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        )),
        DispatchOutcome::CloseLoop { .. } => {
            // An inlined callee body must not close a loop — see
            // `SubWalkClosedLoop`.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::CompileTracePending { .. } => {
            // The compile_trace attempt is gated on `is_top_level`
            // (sub-walks run with `is_top_level == false`), so a callee
            // body can never surface it; fail loud like the CloseLoop
            // arm if that invariant ever breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::SubLoopCalleeCallAssembler { .. } => {
            // The loop-callee CALL_ASSEMBLER request is surfaced from a
            // multi-frame inline at a `residual_call` site and consumed by
            // `try_walker_inline_user_call`; it cannot reach the `inline_call_*`
            // jitcode-op path. Fail loud (safe decline) if that invariant ever
            // breaks.
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}
