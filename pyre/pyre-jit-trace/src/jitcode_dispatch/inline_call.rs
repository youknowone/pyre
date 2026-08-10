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

#[derive(Clone, Copy)]
struct BoundMethodInline {
    method_op: OpRef,
    function: pyre_object::PyObjectRef,
    receiver: pyre_object::PyObjectRef,
}

/// Where an element of a `defs_w` tuple lives, and therefore what the trace
/// emits to read one.  `w_tuple_new` routes EVERY arity-2 tuple through
/// `makespecialisedtuple2` (`specialisedtupleobject.py:169-179`), so a callee
/// with exactly two defaulted parameters — an extremely ordinary signature —
/// never has an array-backed `defs_w` at all.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DefaultsRepr {
    /// Array-backed `W_TupleObject`: the elements live in the `wrappeditems`
    /// block, the shape upstream's `defs_w?[*]` lowers to.
    ItemsBlock,
    /// `Cls_ii`: two inline machine ints.  `wraps[i]` is `wrapint`
    /// (`specialisedtupleobject.py:138-141`), which is what
    /// `_flat_pycall_defaults` already runs per call through
    /// `w_tuple_getitem`, so the emitted box is the same fresh box the
    /// interpreter would have made.
    PairInt,
    /// `Cls_oo`: two inline object slots, for which `wraps[i]` is the identity
    /// (`specialisedtupleobject.py:26-27`) — the field read IS the element.
    PairObject,
}

struct PositionalDefaultsInline {
    tuple: pyre_object::PyObjectRef,
    repr: DefaultsRepr,
    /// `(parameter index, tuple index, concrete value)`.
    values: Vec<(usize, usize, pyre_object::PyObjectRef)>,
}

/// `function.py:188-193,217-231` — which `defs_w` element fills each parameter
/// the call left unbound.  `defs_w` covers the LAST `len(defs_w)` parameters,
/// so parameter `p` takes `defs_w[p - (nparams - ndefaults)]`; a `missing`
/// parameter below that floor has no default and the call would raise, so the
/// whole inline declines.
///
/// `Cls_ff` is left out: its slots are inline `f64` and the walker has no
/// float-field read to pair with `wrapfloat` here, so a two-float defaults
/// tuple safely stays on the residual call path.
unsafe fn positional_defaults_for_inline(
    callable: pyre_object::PyObjectRef,
    missing: &[usize],
    nparams: usize,
) -> Option<PositionalDefaultsInline> {
    if missing.is_empty() {
        return None;
    }
    let tuple = unsafe { pyre_interpreter::function_get_defaults(callable) };
    if tuple.is_null() {
        return None;
    }
    // The layout is what `ob_type` names, and the identity guard the emitting
    // half records pins this exact object — so the type test here decides
    // which read to emit, and nothing later can invalidate it.
    let ob_type = unsafe { (*tuple).ob_type };
    let repr = if std::ptr::eq(ob_type, &pyre_object::pyobject::TUPLE_TYPE) {
        DefaultsRepr::ItemsBlock
    } else if std::ptr::eq(
        ob_type,
        &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE,
    ) {
        DefaultsRepr::PairInt
    } else if std::ptr::eq(
        ob_type,
        &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE,
    ) {
        DefaultsRepr::PairObject
    } else {
        return None;
    };
    let ndefaults = unsafe { pyre_object::w_tuple_len(tuple) };
    let first_defaulted = nparams.checked_sub(ndefaults)?;
    let mut values = Vec::with_capacity(missing.len());
    for &param_index in missing {
        let tuple_index = param_index.checked_sub(first_defaulted)?;
        let value = unsafe { pyre_object::w_tuple_getitem(tuple, tuple_index as i64) }?;
        values.push((param_index, tuple_index, value));
    }
    Some(PositionalDefaultsInline {
        tuple,
        repr,
        values,
    })
}

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
            if fbw_strict_diag_enabled() {
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
            if fbw_strict_diag_enabled() {
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
/// inline path (#68): a FORWARD `goto_if_not`
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
            if fbw_strict_diag_enabled() {
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

/// True iff `body_code` contains a `raise` op.  A callee that raises inline
/// unwinds cleanly only at the top inline level (`inline_depth == 0`, directly
/// under the real caller loop): a raise from a callee inlined BELOW another
/// inlined frame must cross the suspended intermediate frame(s), which needs the
/// cross-frame exception-unwind bridge (gh#343 / gh#467) the drain does not yet
/// reconstruct — the trace instead drops the `NestedBreakBridgeResume` bridge
/// and deopts the unwind to the blackhole.  Straight value-returning chains
/// never raise, so they still inline to the full `fbw_max_multiframe_depth`; a
/// raising callee is capped at the top level.
pub(crate) fn callee_body_contains_raise(body_code: &[u8]) -> bool {
    let mut pc = 0usize;
    while pc < body_code.len() {
        let Some(d) = crate::jitcode_runtime::decode_op_at(body_code, pc) else {
            return false;
        };
        if d.opname == "raise" {
            return true;
        }
        pc = d.next_pc;
    }
    false
}

/// True iff `body_code` carries a `jit_merge_point`, i.e. the callee owns a
/// loop header of its own.
///
/// Reaching one during an inline sub-walk surfaces
/// `SubLoopCalleeCallAssembler`, the one arm of
/// [`try_walker_inline_resolved_user_call`] that consumes the residual's
/// `funcptr` / `r_args` / `call_descr` — and it consumes them as a CALL's
/// `[callable, null_or_self, args…]` operand list.  A route that enters with a
/// callee it resolved itself (rather than off a CALL) carries a differently
/// shaped operand list, so it declines such a body up front.
pub(crate) fn callee_body_owns_loop_header(body_code: &[u8]) -> bool {
    crate::jitcode_runtime::decoded_ops(body_code).any(|op| op.opname == "jit_merge_point")
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
        return Err(DispatchError::callee_inline_unsupported(callee_op_pc));
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
    let diag = fbw_mf_diag_enabled();
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
                Err(DispatchError::callee_inline_unsupported(callee_op_pc))
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
/// at the inline recursion-bound boundary.
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
    // Authoritative walks only: the CALL_ASSEMBLER record + walk-commit
    // bookkeeping is FBW machinery; a non-authoritative context (the
    // diagnostic probe, tests) records the plain residual instead.
    if !ctx.is_authoritative_executor {
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
    // additionally admits a *mutual*-recursive callee — one whose code is
    // already on the inline framestack, i.e. a
    // genuine recursion cycle (`is_even` → `is_odd` → `is_even` at the unroll
    // cap).  It must NOT admit an arbitrary foreign call: folding a
    // non-recursive callee (e.g. a CALL_KW-bearing leaf) to CALL_ASSEMBLER
    // builds and enters a frame the callee's own loop was never traced against
    // and faults.  The emit below keys on `w_code` (callee-agnostic); the token
    // is resolved / synthesised per `callee_key` via `get_assembler_token`.
    if w_code as usize != caller_code as usize {
        let admit_mutual = ctx
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
                if p2_diag_enabled() {
                    eprintln!("[p2-ca] decline pc={} reason=synth-failed", op.pc);
                }
                return Ok(None);
            }
        };
    if p2_diag_enabled() {
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
        &[],
        0,
        nlocals + max_stack,
        nlocals,
        pycode_const,
        w_globals_obj_const,
        ec,
    );

    // do_residual_call step 1 (`pyjitpl.py`): FORCE_TOKEN +
    // SETFIELD_GC(vable_token) before the assembler call.
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);

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
            OpRef::NONE,
            op.pc,
            None,
        )?
    };
    // `pyjitpl.py:2049-2079` checks vrefs after concrete execution, records
    // CALL_ASSEMBLER, then emits GUARD_NOT_FORCED.  In particular,
    // VIRTUAL_REF_FINISH must precede the call so the call and guard remain
    // adjacent and the backend can arm the JIT frame's force descriptor.
    let ca_result = ctx.trace_ctx.call_assembler_red_only_ref_arc(
        token,
        &[callee_frame, ec],
        &[Type::Ref, Type::Ref],
    );
    if let ResidualExecOutcome::Executed(Ok(result)) = exec {
        ctx.trace_ctx.set_opref_concrete(
            ca_result,
            majit_ir::Value::Ref(majit_ir::GcRef(result as usize)),
        );
    }
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
        // #73 Slice 4: forward after-residual fallthrough coordinate; the
        // containing lookup survives only for the empty-twin class and as the
        // audit oracle.
        let legacy_resume_py_pc = || {
            let call_py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc(
                &caller_jitcode.payload.metadata,
                op.pc,
            ) as usize;
            crate::pyjitpl::semantic_fallthrough_pc(caller_code, call_py_pc) as u32
        };
        let resume_py_pc = match caller_jitcode
            .payload
            .after_residual_fallthrough_py_pc_populated()
            .then(|| {
                caller_jitcode
                    .payload
                    .after_residual_fallthrough_py_pc_for_jitcode_pc(op.pc)
            })
            .flatten()
        {
            Some(ft) => {
                if pcmap_afterresidual_audit_enabled() {
                    assert_eq!(
                        ft,
                        legacy_resume_py_pc(),
                        "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: self-recursive CA vstack fallthrough-py twin diverged at jit_pc {}",
                        op.pc
                    );
                }
                ft
            }
            None => legacy_resume_py_pc(),
        };
        let raw_depth = || {
            crate::liveness::liveness_for(caller_jitcode.payload.code_ptr)
                .depth_at_py_pc()
                .get(resume_py_pc as usize)
                .copied()
                .unwrap_or(0) as usize
        };
        let resume_depth = if caller_jitcode.payload.depth_after_residual_populated() {
            let depth = caller_jitcode
                .payload
                .depth_after_residual_for_jitcode_pc(op.pc)
                .unwrap_or(0) as usize;
            if pcmap_afterresidual_audit_enabled() {
                assert_eq!(
                    depth,
                    raw_depth(),
                    "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: self-recursive CA vstack depth twin diverged at jit_pc {} (py {resume_py_pc})",
                    op.pc
                );
            }
            depth
        } else {
            raw_depth()
        };
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
    // `pyjitpl.py:2080-2081` keeps the assembler virtualizable alive after
    // the force guard has captured its resume data.
    ctx.trace_ctx.record_op(OpCode::Keepalive, &[callee_frame]);
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
/// [`try_walker_call_assembler_self_recursive`].
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
    maybe_walker_vable_and_vrefs_before_residual_call(ctx, op.pc);

    // `direct_assembler_call` (`pyjitpl.py`) records the CALL_ASSEMBLER with
    // exactly the target jitdriver's red args — `assert len(args) ==
    // targetjitdriver_sd.num_red_args` — and the portal reds are
    // `['frame', 'ec']` (`interp_jit.py`), the frame at
    // `index_of_virtualizable`. The callee, not the caller, unpacks the
    // virtualizable: `patch_new_loop_to_load_virtualizable_fields`
    // (`compile.py`) truncates the callee loop to `inputargs[:num_red_args]`
    // and prepends a GETFIELD_GC / GETARRAYITEM_GC per field read off that
    // vable arg. So the callee loop head dereferences a real heap frame, which
    // is why forcing the still-virtual frame here — allocation plus one
    // SETARRAYITEM_GC per known element — is the upstream op sequence rather
    // than a decline.
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
    // rooting ruling out a ref-specific defect here.
    let argbox_types: Vec<Type> = vec![Type::Ref; r_args.len()];
    let allboxes = build_allboxes(funcptr, r_args, &argbox_types, call_descr.arg_types());
    let exec = try_execute_residual_call_via_executor(
        ctx,
        OpCode::CallMayForceR,
        &allboxes,
        call_descr,
        OpRef::NONE,
        op.pc,
        None,
    )?;
    // `pyjitpl.py:2049-2079` records a forced VIRTUAL_REF_FINISH before the
    // selected CALL_ASSEMBLER, followed immediately by GUARD_NOT_FORCED.
    let ca_result = ctx.trace_ctx.call_assembler_red_only_ref_arc(
        token,
        &[callee_frame, callee_ec],
        &[Type::Ref, Type::Ref],
    );
    if let ResidualExecOutcome::Executed(Ok(result)) = exec {
        ctx.trace_ctx.set_opref_concrete(
            ca_result,
            majit_ir::Value::Ref(majit_ir::GcRef(result as usize)),
        );
    }
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
    // `pyjitpl.py:2080-2081` places KEEPALIVE after GUARD_NOT_FORCED.
    ctx.trace_ctx.record_op(OpCode::Keepalive, &[callee_frame]);
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

/// #62 slice (3c): full-body-walk inline of a recognized user-function
/// `call_fn`.
///
/// Returns:
/// * `Ok(Some((outcome, next_pc)))` — the call was inlined; caller returns it.
/// * `Ok(None)` — not eligible (not a pure-Python function, has a
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
    // The Ref list is NOT at a fixed offset: the method-form `CALL` helpers
    // this leg latches for lower through the mixed `iIRd>r` shape, whose
    // leading Int list shifts it (`dispatch_residual_call_iIRd_kind` reads it
    // at `1 + i_width`).  Reading offset 1 there takes the Int list's register
    // indices into the Ref bank — refs unrelated to the call, of a length that
    // still passes the flush's depth check.
    let ref_operand_offset = ref_var_list_operand_offset(code, op)?;
    let fresh = read_ref_var_list_concrete(code, op, ref_operand_offset, ctx);
    if fresh.is_empty() {
        return None;
    }
    // Validate the CALL operand slice itself, not the complete reconstructed
    // stack.  A CALL inside WITH/FOR_ITER can retain non-null prefix operands;
    // checking `stack.first()` after prepending them lets an unresolved NULL
    // callable slip through and publishes an invalid frame at the CALL.
    if !matches!(fresh.first(), Some(ConcreteValue::Ref(r)) if !r.is_null()) {
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
    Some(stack)
}

/// Fold a keyword call's `kwnames`->parameter permutation at trace time so a
/// `call_kw` reuses the positional inline path.  The constant `kwnames` tuple
/// and the callee's static parameter names are both known at record time, so
/// the reorder is a pure trace-time permutation of the argument boxes into
/// parameter order — `_match_signature` (`@jit.unroll_safe`) unrolled and
/// folded.  Once reordered the seeding is identical to a positional call.
///
/// `r_args` layout is `[callable, self_or_null, kwnames, arg0..argN-1]`; the
/// trailing `nkw` args are the keyword values, `kwnames[j]` naming
/// `arg[n_pos + j]` where `n_pos = nargs - nkw`.  `receiver`, when present,
/// is the implicit leading argument inserted by `call_kw` for method-form
/// calls (`Arguments.prepend` / `funcrun_obj` upstream).
///
/// Returns `None` (declining to the residual call, no behavior change) for any
/// shape the plain positional seeding cannot serve: a non-constant / non-tuple
/// `kwnames`, a callee with `*args` / `**kwargs` / keyword-only parameters, an
/// argument count that does not exactly fill the positional parameters (a
/// default would be needed), a keyword naming an unknown parameter, or a
/// keyword colliding with a positionally-filled parameter ("multiple values").
///
/// # Safety
/// `w_code` must be the live code object pointer for the resolved callable.
/// Whether every local of `w_code` that a call must bind lives in
/// `registers_r[0..co_argcount]`, which is all the inline seeding fills.
///
/// `*args` and `**kwargs` own locals the seeding never packs, and a
/// keyword-only parameter owns one it never reaches — and none of the three is
/// counted by `co_argcount`, so an arity check alone lets them through.
fn fbw_callee_scope_is_positional_only(w_code: *const ()) -> bool {
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return false;
    }
    let flags = unsafe { (*raw).flags };
    !flags.contains(pyre_interpreter::CodeFlags::VARARGS)
        && !flags.contains(pyre_interpreter::CodeFlags::VARKEYWORDS)
        && unsafe { (*raw).kwonlyarg_count } == 0
}

/// The scope slot `_match_signature` writes the vararg tuple into
/// (`argument.py:222-234`): `co_argcount + co_kwonlyargcount`.  This helper
/// admits only the shape whose slot is exactly `co_argcount`, leaving
/// `**kwargs` and keyword-only locals residual.
fn fbw_callee_vararg_slot(w_code: *const ()) -> Option<usize> {
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return None;
    }
    let flags = unsafe { (*raw).flags };
    if flags.contains(pyre_interpreter::CodeFlags::VARARGS)
        && !flags.contains(pyre_interpreter::CodeFlags::VARKEYWORDS)
        && unsafe { (*raw).kwonlyarg_count } == 0
    {
        Some(unsafe { (*raw).arg_count as usize })
    } else {
        None
    }
}

unsafe fn fbw_reorder_call_kw_args(
    r_args: &[OpRef],
    arg_concretes: &[ConcreteValue],
    w_code: *const (),
    nparams: usize,
    receiver: Option<(OpRef, ConcreteValue)>,
) -> Option<(Vec<OpRef>, Vec<ConcreteValue>)> {
    if r_args.len() < 3 || arg_concretes.len() < 3 {
        return None;
    }
    let ConcreteValue::Ref(kwnames) = arg_concretes[2] else {
        return None;
    };
    if kwnames.is_null() || !unsafe { pyre_object::is_tuple(kwnames) } {
        return None;
    }
    let args = &r_args[3..];
    let arg_conc = &arg_concretes[3..];
    let nargs = args.len();
    if arg_conc.len() != nargs {
        return None;
    }
    let nkw = unsafe { pyre_object::w_tuple_len(kwnames) };
    // No positional parameter may be filled more than once, and the call may
    // not pass more than the callee takes — `*args` / `**kwargs` /
    // keyword-only slots are ruled out separately by
    // `fbw_callee_scope_is_positional_only`.  A parameter left unbound is
    // allowed through as a hole; the caller fills it from `defs_w` or declines
    // when it has no default.
    let receiver_count = usize::from(receiver.is_some());
    if nparams == 0 || nkw > nargs || nargs + receiver_count > nparams {
        return None;
    }
    let raw = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw.is_null() {
        return None;
    }
    if !fbw_callee_scope_is_positional_only(w_code) {
        return None;
    }
    let varnames = unsafe { &(*raw).varnames };
    if varnames.len() < nparams {
        return None;
    }
    let n_pos = nargs - nkw;
    let mut slot_args: Vec<Option<OpRef>> = vec![None; nparams];
    let mut slot_conc: Vec<Option<ConcreteValue>> = vec![None; nparams];
    if let Some((receiver_arg, receiver_concrete)) = receiver {
        slot_args[0] = Some(receiver_arg);
        slot_conc[0] = Some(receiver_concrete);
    }
    for k in 0..n_pos {
        let pi = receiver_count + k;
        slot_args[pi] = Some(args[k]);
        slot_conc[pi] = Some(arg_conc[k]);
    }
    for j in 0..nkw {
        let name_obj = unsafe { pyre_object::w_tuple_getitem(kwnames, j as i64) }?;
        if !unsafe { pyre_object::is_str(name_obj) } {
            return None;
        }
        let name = unsafe { pyre_object::w_str_get_wtf8(name_obj) }
            .as_str()
            .ok()?;
        let pi = varnames[..nparams]
            .iter()
            .position(|v| v.as_str() == name)?;
        // A keyword may only bind a parameter past the positional fill, and each
        // parameter at most once (else Python raises "multiple values for
        // argument").  A name in the positional-only range is not bindable by
        // keyword at all — `def f(x, /)` called as `f(x=1)` is a TypeError, so
        // binding slot 0 here would inline a call the interpreter rejects.
        if pi < receiver_count + n_pos
            || pi < unsafe { (*raw).posonlyarg_count } as usize
            || slot_args[pi].is_some()
        {
            return None;
        }
        slot_args[pi] = Some(args[n_pos + j]);
        slot_conc[pi] = Some(arg_conc[n_pos + j]);
    }
    let mut out_args = Vec::with_capacity(nparams);
    let mut out_conc = Vec::with_capacity(nparams);
    for k in 0..nparams {
        out_args.push(slot_args[k].unwrap_or(OpRef::NONE));
        out_conc.push(slot_conc[k].unwrap_or(ConcreteValue::Null));
    }
    Some((out_args, out_conc))
}

/// Unpack a `bh_call_function_ex_fn(callable, self_or_null, starargs,
/// kwargs_or_null)` star tuple into the positional argument boxes the inline
/// path seeds from, or `None` to leave the call a residual.
///
/// The elements are read out of the heap cache rather than off the tuple, so
/// this folds exactly when the star tuple is virtual at the call — the
/// `args = (...)` / `f(*args)` pair the walker just recorded, whose
/// `wrappeditems` block and per-index stores are still cached
/// (`try_walker_specialize_newtuple_object`).  A tuple that arrived from
/// anywhere else has no cached block and declines, as does any `**kwargs`
/// merge (the helper's `kwargs_or_null` is then a real mapping) and any arity
/// that is not the callee's exact parameter count.
fn fbw_unpack_call_function_ex_args<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    r_args: &[OpRef],
    arg_concretes: &[ConcreteValue],
    nparams: usize,
) -> Option<(Vec<OpRef>, Vec<ConcreteValue>)> {
    if r_args.len() < 4 || arg_concretes.len() < 4 || nparams == 0 {
        return None;
    }
    // `kwargs_or_null` is the checked `PY_NULL` sentinel for a call with no
    // `**` merge; anything else is a real mapping this fold does not bind.
    match arg_concretes[3] {
        ConcreteValue::Null => {}
        ConcreteValue::Ref(kwargs) if kwargs.is_null() || kwargs == pyre_object::PY_NULL => {}
        _ => return None,
    }
    // `wrappeditems` resolves to a structural field index (offset / size /
    // type), so an object whose slot 0 happens to match — a list's backing
    // store, say — would hit the same cache entry and be unpacked as if its
    // slots were tuple element refs.  `f(*some_list)` is ordinary Python, so
    // pin the concrete to a real tuple the way the `kwnames` path does before
    // reading the field.
    match arg_concretes[2] {
        ConcreteValue::Ref(starargs)
            if !starargs.is_null() && unsafe { pyre_object::is_tuple(starargs) } => {}
        _ => return None,
    }
    let starargs = r_args[2];
    let items_descr = crate::descr::tuple_wrappeditems_descr();
    let block = ctx
        .trace_ctx
        .heapcache_getfield_cached(starargs, items_descr.index())?;
    // The cached length pins the arity: the callee takes exactly `nparams`
    // positional parameters, and a mismatch is a runtime TypeError the inline
    // path does not model.
    let len_op = ctx.trace_ctx.heap_cache().arraylen(block)?;
    match len_op.inline_const_to_value() {
        Some(majit_ir::Value::Int(n)) if n as usize == nparams => {}
        _ => return None,
    }
    let array_descr_index = crate::state::pyobject_gcarray_descr().index();
    let mut args = Vec::with_capacity(nparams);
    let mut concretes = Vec::with_capacity(nparams);
    for index in 0..nparams {
        let elem = ctx.trace_ctx.heapcache_getarrayitem(
            block,
            OpRef::ConstInt(index as i64),
            array_descr_index,
        )?;
        // Take each concrete from the element box itself, not from the tuple,
        // so the seeded shadow is the one the symbolic argument carries.
        let concrete = walker_concrete_ref_object(ctx, elem)?;
        args.push(elem);
        concretes.push(ConcreteValue::Ref(concrete));
    }
    Some((args, concretes))
}

/// Body-shape verdicts for a callee's per-fn JitCode, decided once and kept on
/// the payload ([`crate::pyjitcode::InlineBodyFacts`]).
///
/// Every predicate behind this decodes the whole callee body, and the inline
/// recipes ask them per call site, on every trace that reaches the call — so
/// answering them here is the difference between one scan per callee and one
/// per call site. `None` when the code has no installed jitcode body or descr
/// pool, which is the same condition the callers already decline on.
fn sub_jitcode_body_facts_for_code(code: *const ()) -> Option<crate::pyjitcode::InlineBodyFacts> {
    let body = crate::state::sub_jitcode_body_for_code(code)?;
    let (descr_refs, _, _) = crate::state::sub_jitcode_descr_pool_for_code(code)?;
    // Resolved after the two lookups above so that a code object without an
    // installed body never caches a verdict computed from an absent one.
    let pjc = crate::state::pyjitcode_for_code(code)?;
    Some(
        *pjc.inline_body_facts
            .get_or_init(|| crate::pyjitcode::InlineBodyFacts {
                contains_raise: callee_body_contains_raise(body.code),
                has_abort_permanent: crate::jitcode_runtime::decoded_ops(body.code)
                    .any(|op| op.opname == "abort_permanent"),
                exc_override_straight_line: exception_string_override_straight_line(body.code),
                exc_override_sample_safe: exception_string_override_sample_safe(
                    body.code, descr_refs,
                ),
                exc_override_has_nested_call: exception_string_override_has_nested_call(
                    body.code, descr_refs,
                ),
                owns_loop_header: callee_body_owns_loop_header(body.code),
                has_exception_table: unsafe {
                    let raw = pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef)
                        as *const pyre_interpreter::CodeObject;
                    !raw.is_null() && !(&(*raw).exceptiontable).is_empty()
                },
            }),
    )
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
    // Authoritative walks only: inline sub-walks lean on FBW multi-frame
    // snapshot plumbing a non-authoritative context does not carry.
    if !ctx.is_authoritative_executor {
        return Ok(None);
    }
    // Only a genuine Python call helper is an inline target: positional
    // `call_fn` / `call_fn_N` (`PyreHelperKind::CallFn`) and keyword `call_kw_N`
    // (`PyreHelperKind::CallKw`), both tagged by the flatten lowering.  Every
    // container/builtin helper routed here carries a different tag or `None`
    // (`store_subscr_fn` -> StoreSubscr, `normalize_raise_varargs_fn` /
    // `set_current_exception` -> None).  Without this guard `d[f] = v` with a
    // 1-arg function key `f` lowers to `residual_call_r_v(store_subscr_fn, [d,
    // f, v])`, whose ref args pass the function sniff below and are mis-inlined
    // as `f(v)`, skipping the store.  Upstream never inlines a Python call at a
    // residual_call site (inlinable calls get their own inline_call jitcodes);
    // this restores that invariant for the pyre FBW inline-at-residual lever.
    let is_call_kw = pyre_helper == majit_ir::PyreHelperKind::CallKw;
    let is_call_function_ex = pyre_helper == majit_ir::PyreHelperKind::CallFunctionEx;
    if pyre_helper != majit_ir::PyreHelperKind::CallFn && !is_call_kw && !is_call_function_ex {
        return Ok(None);
    }
    if fbw_inline_diag_enabled() {
        eprintln!(
            "[inline-entry] pc={} helper={:?} nrefargs={} subwalk={}",
            op.pc,
            pyre_helper,
            r_args.len(),
            ctx.fbw_mode.inline_subwalk,
        );
    }
    // Name every bail between the entry print and callee resolution.  An
    // `[inline-entry]` with no follow-up line otherwise leaves the reason
    // unobservable, which is the whole distance between "this call did not
    // inline" and knowing why.
    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }
    if r_args.is_empty() {
        decline!("no ref args");
    }
    let mut arg_concretes = read_ref_var_list_concrete(code, op, ref_operand_offset, ctx);
    if r_args.len() < 2 {
        decline!("fewer than two ref args");
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
        decline!("callable slot carries no concrete ref");
    };
    if callable.is_null() {
        decline!("callable is null");
    }
    // The receiver slot is a checked `PY_NULL` sentinel for a plain no-receiver
    // call; its concrete shadow arrives as `Null` (`call_kw`) or `Ref(PY_NULL)`
    // (`call_fn`).  Both mean "no receiver" (`method_form = false`).
    let null_or_self = match arg_concretes[1] {
        ConcreteValue::Ref(r) => r,
        ConcreteValue::Null => pyre_object::PY_NULL,
        _ => decline!("receiver slot carries no concrete ref"),
    };
    let mut method_form = !null_or_self.is_null() && null_or_self != pyre_object::PY_NULL;
    // baseobjspace.py:1254-1259 unwraps `_Method` before the Function
    // valuestack fast path.  CALLs through a stored bound method (notably the
    // module aliases in random.py) arrive as `[Method, PY_NULL, args...]`, so
    // recover its immutable function/receiver fields and feed the same
    // method-form callee shape used by LOAD_METHOD.
    let bound_method = if !method_form && unsafe { pyre_object::is_method(callable) } {
        let function = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe { pyre_object::w_method_get_self(callable) };
        if function.is_null() || receiver.is_null() {
            return Ok(None);
        }
        method_form = true;
        Some(BoundMethodInline {
            method_op: r_args[0],
            function,
            receiver,
        })
    } else {
        None
    };
    let resolved_callable = bound_method.map_or(callable, |bound| bound.function);
    let Some((w_code, nparams, has_closure)) =
        (unsafe { resolve_inlinable_callee(resolved_callable) })
    else {
        // The callable's type is the whole answer here: `resolve_inlinable_callee`
        // takes plain `function` only, so a `builtin_function_or_method` or a
        // `method` reads as "not inlinable" for a reason no pc can convey.
        decline!(format_args!(
            "callee not inlinable (callable type {})",
            unsafe { pyre_object::type_name_of(callable) }
        ));
    };
    if fbw_inline_diag_enabled() {
        // Name the callee: a pc alone does not say which function a decline
        // cost, and one trace reaches several.
        let name = unsafe {
            let raw = pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject;
            if raw.is_null() {
                String::new()
            } else {
                (*raw).qualname.clone()
            }
        };
        eprintln!(
            "[inline-resolved] pc={} callee={name} nparams={nparams} has_closure={has_closure} method_form={method_form}",
            op.pc,
        );
    }
    let (callee_args, callee_arg_concretes) = if is_call_kw {
        // A keyword call folds its `kwnames`->parameter permutation at trace
        // time (`fbw_reorder_call_kw_args`) so the reordered param-order args
        // seed the callee exactly like a positional call.  Method form
        // prepends its receiver before that permutation, matching
        // `call_kw`'s `Arguments.prepend` / `funcrun_obj` path.
        let receiver = if let Some(bound) = bound_method {
            // Placeholder until the resolved half reads Method.w_self live.
            Some((bound.method_op, ConcreteValue::Ref(bound.receiver)))
        } else if method_form {
            Some((r_args[1], arg_concretes[1]))
        } else {
            None
        };
        let Some(reordered) = (unsafe {
            fbw_reorder_call_kw_args(r_args, &arg_concretes, w_code, nparams, receiver)
        }) else {
            return Ok(None);
        };
        reordered
    } else if is_call_function_ex {
        // `f(*args)` unpacks the star tuple into positional arguments, which is
        // a trace-time reorder whenever the tuple is the one this trace just
        // built: its element boxes are still in the heap cache, so the callee
        // seeds from them and the tuple keeps no consumer.
        //
        // The unpacked arguments come from the star tuple alone, so a
        // method-form call — whose receiver is an implicit leading argument the
        // unpack never sees — stays residual.  The callee-scope decline this
        // seeding also needs is applied once in the resolved half.
        if method_form {
            return Ok(None);
        }
        let Some(unpacked) = fbw_unpack_call_function_ex_args(ctx, r_args, &arg_concretes, nparams)
        else {
            return Ok(None);
        };
        unpacked
    } else {
        let mut callee_args = Vec::with_capacity(r_args.len().saturating_sub(1));
        let mut callee_arg_concretes = Vec::with_capacity(arg_concretes.len().saturating_sub(1));
        if let Some(bound) = bound_method {
            // Placeholder until the non-emitting eligibility checks finish;
            // the resolved half replaces it with GetfieldGcR(Method.w_self).
            callee_args.push(bound.method_op);
            callee_arg_concretes.push(ConcreteValue::Ref(bound.receiver));
        } else if method_form {
            callee_args.push(r_args[1]);
            callee_arg_concretes.push(arg_concretes[1]);
        }
        callee_args.extend_from_slice(&r_args[2..]);
        callee_arg_concretes.extend_from_slice(&arg_concretes[2..]);
        (callee_args, callee_arg_concretes)
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
        resolved_callable,
        r_args[0],
        resolved_callable,
        arg_concretes,
        callee_args,
        callee_arg_concretes,
        method_form,
        bound_method,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        true,
        false,
        None,
    )
}

/// `executioncontext.py:85-89 ExecutionContext.enter`, at an inlined call.
///
/// ```python
/// def enter(self, frame):
///     frame.f_backref = self.topframeref
///     self.topframeref = jit.virtual_ref(frame)
/// ```
///
/// Both halves run.  The recorded ops are what a guard resumes into; the
/// concrete stores are what the callee body observes while this walk records
/// it, because the walk IS the interpreter running — a `sys._getframe()` in
/// the body executes as a residual against the live `ec`, and without the
/// concrete store it would read the CALLER and commit the wrong frame.
///
/// The concrete slot holds the `JitVirtualRef`, not the frame:
/// `executioncontext::force_vref` resolves it for every reader, and a vref
/// built by `virtual_ref_during_tracing` already carries `forced = frame` with
/// `virtual_token = TOKEN_NONE` (`virtualref.py:85-92`), so the resolution is
/// exact and cannot fail.  Storing the vref rather than the frame is what lets
/// the optimizer keep the frame virtual: nothing reads the frame itself unless
/// something forces it.
///
/// Returns the vref's OpRef for the matching [`walker_ec_leave`].
fn walker_ec_enter(
    ctx: &mut TraceCtx,
    callee_frame: OpRef,
    callee_ec: OpRef,
    concrete_frame: *mut pyre_interpreter::PyFrame,
    concrete_ec: *mut pyre_interpreter::PyExecutionContext,
) -> OpRef {
    // `frame.f_backref = self.topframeref` — the caller's vref moves into the
    // callee, unforced.  `emit_new_pyframe_inline_with_params` leaves the slot
    // at its constructor default, so this is the store that links the chain.
    let concrete_caller_topframeref = unsafe { (*concrete_ec).topframeref };
    let caller_topframeref = ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[callee_ec],
        crate::descr::ec_topframeref_descr(),
    );
    ctx.set_opref_concrete(
        caller_topframeref,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_caller_topframeref as usize)),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_frame, caller_topframeref],
        crate::descr::pyframe_f_backref_descr(),
    );
    // `self.topframeref = jit.virtual_ref(frame)`.
    let (vref, concrete_vref) = ctx.opimpl_virtual_ref(callee_frame, concrete_frame as usize);
    ctx.set_opref_concrete(
        vref,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_vref as usize)),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_ec, vref],
        crate::descr::ec_topframeref_descr(),
    );
    // The recording-time shadow of the `SetfieldGc` above: `PyFrame.f_backref`
    // is a `Type::Ref` field, so the emitted store carries the generational
    // barrier and the concrete store has to carry it too.  This frame is an
    // old-gen `FrameBox` and the caller's vref can be young.
    pyre_object::gc_hook::try_gc_write_barrier(concrete_frame as *mut u8);
    majit_gc::bh_probe_note_store(
        concrete_frame as usize,
        crate::frame_layout::PYFRAME_F_BACKREF_OFFSET,
        4,
    );
    unsafe {
        (*concrete_frame).f_backref = concrete_caller_topframeref;
        (*concrete_ec).topframeref = concrete_vref as *mut pyre_interpreter::PyFrame;
    }
    vref
}

/// `executioncontext.py:91-107 ExecutionContext.leave`'s frame-chain half, at
/// the return from an inlined call.
///
/// ```python
/// frame_vref = self.topframeref
/// self.topframeref = frame.f_backref
/// if frame.escaped or got_exception:
///     f_back = frame.f_backref()
///     if f_back:
///         f_back.mark_as_escaped()
///     frame_vref()
/// jit.virtual_ref_finish(frame_vref, frame)
/// ```
///
/// The profile-hook half (`if self.profilefunc: self._trace(frame,
/// 'leaveframe', w_exitvalue)`) stays with the interpreter's own
/// [`pyre_interpreter::PyExecutionContext::leave`].  Omitting it here does not
/// lose a leave event, because `is_being_profiled` is a portal-driver GREEN
/// (`interp_jit.py:68 greens = ['next_instr', 'is_being_profiled', 'pycode']`):
/// a trace is keyed on it, so one recorded with profiling off is only ever
/// entered with profiling off, and turning profiling on selects a different
/// green key rather than reusing this trace.
///
/// The escape branch runs in both worlds.  Concretely it marks the caller and
/// forces the leaving vref; in the trace it records the force as
/// `VIRTUAL_REF_FINISH(vrefbox, virtualbox)` — upstream's "already forced
/// during tracing" form — rather than the NULL form, so `vref.forced` ends up
/// pointing at the virtual instead of staying NULL.  That is what keeps a
/// later read through a deeper escaped frame's `f_backref` from hitting
/// `InvalidVirtualRef`.
///
/// Upstream runs this from a `finally`, so the caller must too: every path out
/// of the callee level — normal return, raised exception, or a declined
/// sub-walk — has to reach it, or `virtualref_boxes` is left unbalanced and the
/// loop header trips `assert len(self.virtualref_boxes) == 0`.
pub(crate) fn walker_ec_leave(
    ctx: &mut TraceCtx,
    callee_frame: OpRef,
    callee_ec: OpRef,
    concrete_frame: *mut pyre_interpreter::PyFrame,
    concrete_ec: *mut pyre_interpreter::PyExecutionContext,
    got_exception: bool,
) {
    // `pyopcode.py:184 handle_operation_error` marks the frame finished before
    // an exception escapes into `ExecutionContext.leave`.  Ordinary returns
    // already publish `PyFrame.finish_value` at the lowered `*_return`
    // operation; this boundary supplies the exception sibling on this
    // inlined callee's own red frame.  A declined sub-walk or callee-loop
    // handoff has not finished executing and deliberately skips it.
    if got_exception {
        let flags_descr = crate::descr::pyframe_flags_descr();
        let live_flags = crate::state::opimpl_getfield_gc_i(ctx, callee_frame, flags_descr.clone());
        let finished_bit = ctx.const_int(i64::from(pyre_interpreter::PyFrame::FLAG_FRAME_FINISHED));
        let new_flags = ctx.record_op(OpCode::IntOr, &[live_flags, finished_bit]);
        ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[callee_frame, new_flags],
            flags_descr.clone(),
        );
        ctx.heapcache_setfield_cached(callee_frame, flags_descr.index(), new_flags);
        unsafe { (*concrete_frame).set_frame_finished_execution(true) };
    }
    // `self.topframeref = frame.f_backref` — no parens: the caller's vref
    // moves back unforced, so a caller frame that stayed virtual stays virtual.
    let concrete_f_backref = unsafe { (*concrete_frame).f_backref };
    let f_backref = ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[callee_frame],
        crate::descr::pyframe_f_backref_descr(),
    );
    ctx.set_opref_concrete(
        f_backref,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_f_backref as usize)),
    );
    ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[callee_ec, f_backref],
        crate::descr::ec_topframeref_descr(),
    );
    let escaped = unsafe {
        let frame_vref = (*concrete_ec).topframeref;
        (*concrete_ec).topframeref = concrete_f_backref;
        let escaped = (*concrete_frame).escaped() || got_exception;
        if escaped {
            // A frame that reached app level must keep its caller reachable
            // too, or the next `_getframe().f_back` walks into a frame the JIT
            // was still free to keep virtual.  `get_f_back` forces, which is
            // `f_back = frame.f_backref()` with the parens.
            let f_back = (*concrete_frame).get_f_back();
            if !f_back.is_null() {
                (*f_back).mark_as_escaped();
            }
            // `frame_vref()` — force the leaving frame's own vref so it
            // outlives the JIT frame.
            let _ = pyre_interpreter::executioncontext::force_vref(frame_vref);
        }
        escaped
    };
    if escaped {
        // The concrete force above is only half of `frame_vref()`: the
        // optimizer reads the trace, not the heap.  Record the force as
        // `VIRTUAL_REF_FINISH(vrefbox, virtualbox)` — the non-null second
        // operand is upstream's "this vref was forced during tracing already"
        // encoding, which `optimize_VIRTUAL_REF_FINISH` lowers to storing the
        // virtual into `vref.forced` (`virtualize.py:141-151`).
        //
        // Without it the finish below would emit the NULL form, leaving
        // `forced` NULL and `virtual_token` cleared, and a later read through
        // a deeper escaped frame's `f_backref` would hit `InvalidVirtualRef`.
        // `stop_tracking_virtualref` also replaces the vrefbox with
        // ConstPtr(NULL), so the finish that follows sees a non-vref and
        // records nothing — one finish, not two.
        //
        // Upstream reaches the same end state by a different route: it records
        // `frame_vref()` as a force and then the ordinary NULL finish, so
        // `forced` is written at runtime by `force_now` rather than by the
        // optimizer.  That route needs the `jit_force_virtual` lowering, which
        // is the one piece of this protocol pyre has not wired
        // (`jitcode_dispatch/mod.rs` item c — `_do_jit_force_virtual` is
        // tests-only, production reach 0).  Both forms are already understood
        // by `optimize_VIRTUAL_REF_FINISH`, so this uses the one that is
        // reachable; converge on the upstream spelling when the force lowering
        // lands.
        let live = ctx.virtualref_boxes_len();
        let vref_is_live = ctx
            .innermost_virtualref_vref()
            .is_some_and(|(vrefbox, _)| vrefbox.as_const_ptr().is_none_or(|vref| vref.0 != 0));
        if live >= 2 && vref_is_live {
            ctx.stop_tracking_virtualref(live - 2);
        }
    }
    // `jit.virtual_ref_finish(frame_vref, frame)`.
    ctx.opimpl_virtual_ref_finish(callee_frame);
}

/// Resolve the generated builtin-wrapper argument slice's array-item
/// descriptor. The first instruction's arraylen descriptor is deliberately
/// not interchangeable with the later getarrayitem descriptor.
pub(super) fn wrapper_args_item_descr_index(code: &[u8]) -> Option<u32> {
    // Generated gateways perform their argument extraction before entering the
    // typed body, reading the slice length before any element.  The first Ref
    // item read after the first slice-length read is therefore the
    // wrapper-argument descriptor, independent of which register colouring
    // assigns to the slice.
    let arraylen_pc = crate::jitcode_runtime::decoded_ops(code)
        .find(|decoded| decoded.key == "arraylen_gc/rd>i")
        .map(|decoded| decoded.pc)?;
    crate::jitcode_runtime::decoded_ops(code)
        .find(|decoded| decoded.pc > arraylen_pc && decoded.key == "getarrayitem_gc_r/rid>r")
        .and_then(|decoded| {
            let lo = *code.get(decoded.pc + 3)? as usize;
            let hi = *code.get(decoded.pc + 4)? as usize;
            let pool_index = lo | (hi << 8);
            crate::jitcode_runtime::all_descr_refs()
                .get(pool_index)
                .map(|descr| descr.index())
        })
}

/// `BuiltinCode.func` is an RPython PBC: the codewriter turns its finite
/// target family into an indirect call whose address is resolved back to the
/// generated target JitCode by `MetaInterpStaticData.bytecode_for_address`
/// (`pyjitpl.py:2174-2186`).  The interpreter-level `call_fn` helper hides
/// that indirect call behind `Function -> BuiltinCode -> func`, so recover
/// the same target here and enter the generated wrapper with its one red
/// `&[PyObjectRef]` argument.
///
/// The slice is represented to the translated body as a GC array.  Build that
/// array in trace IR and seed its heap-cache entries from the live CALL
/// operands; this preserves a distinct red receiver for every method call.
/// In particular, a bound Method's receiver is read from its immutable
/// `w_self` field rather than baked from the recording-time object.
pub(crate) fn try_walker_inline_builtin_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    ref_operand_offset: usize,
    r_args: &[OpRef],
    pyre_helper: majit_ir::PyreHelperKind,
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor
        // A `call_kw` residual carries its kwnames tuple in arg index 2, and
        // keywords reach a builtin as the trailing `__pyre_kw__` marker dict
        // that `split_builtin_kwargs` strips — a shape the flat
        // `&[PyObjectRef]` array built below does not construct, so leave those
        // calls to `bh_call_kw_<n>`.
        || pyre_helper != majit_ir::PyreHelperKind::CallFn
        || r_args.len() < 2
        || dst_bank != 'r'
    {
        return Ok(None);
    }

    let mut arg_concretes = read_ref_var_list_concrete(code, op, ref_operand_offset, ctx);
    for i in 0..2 {
        if matches!(arg_concretes.get(i), Some(ConcreteValue::Null)) {
            if let Some(majit_ir::Value::Ref(r)) = ctx.trace_ctx.box_value(r_args[i]) {
                if r != majit_ir::GcRef::NO_CONCRETE && r.as_usize() != 0 {
                    arg_concretes[i] = ConcreteValue::Ref(r.as_usize() as pyre_object::PyObjectRef);
                }
            }
        }
    }
    let ConcreteValue::Ref(callable_operand) = arg_concretes[0] else {
        return Ok(None);
    };
    if callable_operand.is_null() {
        return Ok(None);
    }
    let null_or_self = match arg_concretes[1] {
        ConcreteValue::Ref(value) => value,
        ConcreteValue::Null => pyre_object::PY_NULL,
        _ => return Ok(None),
    };
    let method_form = !null_or_self.is_null() && null_or_self != pyre_object::PY_NULL;
    let bound_method = !method_form && unsafe { pyre_object::is_method(callable_operand) };
    let (callable, receiver) = if bound_method {
        let function = unsafe { pyre_object::w_method_get_func(callable_operand) };
        let receiver = unsafe { pyre_object::w_method_get_self(callable_operand) };
        if function.is_null() || receiver.is_null() {
            return Ok(None);
        }
        (function, Some(receiver))
    } else {
        (callable_operand, method_form.then_some(null_or_self))
    };
    // Every decline below is silent otherwise, and they are not
    // interchangeable: `not is_function` is a class call or another
    // non-Function callable, while `no jitcode for address` names a builtin
    // whose `BuiltinCode.func` is not a member of the PBC family
    // `builtin_wrapper_indirect_graphs` builds — i.e. one registered without a
    // `__pyre_wrap_*` gateway.  The address is what identifies the missing
    // member, so print it.
    macro_rules! builtin_inline_decline {
        ($why:expr, $addr:expr) => {
            if fbw_inline_diag_enabled() {
                eprintln!(
                    "[builtin-inline-decline] pc={} why={} callable={} operand={} fnaddr={:#x}",
                    op.pc,
                    $why,
                    unsafe { pyre_object::type_name_of(callable) },
                    unsafe { pyre_object::type_name_of(callable_operand) },
                    $addr,
                );
            }
        };
    }
    if !unsafe { pyre_interpreter::is_function(callable) } {
        builtin_inline_decline!("not is_function", 0usize);
        return Ok(None);
    }
    let builtin_code =
        unsafe { pyre_interpreter::function_get_code(callable) } as pyre_object::PyObjectRef;
    if builtin_code.is_null() || !unsafe { pyre_interpreter::is_builtin_code(builtin_code) } {
        builtin_inline_decline!("not builtin_code", 0usize);
        return Ok(None);
    }
    let fnaddr = unsafe { pyre_interpreter::builtin_code_get(builtin_code) as usize };
    let Some(jitcode) = crate::state::bytecode_for_address(fnaddr) else {
        builtin_inline_decline!("no jitcode for address", fnaddr);
        return Ok(None);
    };
    let Some(body) = crate::jitcode_dispatch::sub_jitcode_body_by_index(jitcode.index()) else {
        builtin_inline_decline!("no sub jitcode body", fnaddr);
        return Ok(None);
    };
    if body.num_regs_r < 1 {
        return Ok(None);
    }
    let nested_helper_entry = if ctx.fbw_mode.inline_subwalk {
        match compute_inline_helper_call_entry_frame(ctx, op.pc) {
            Ok(frame) => Some(frame),
            Err(_) => return Ok(None),
        }
    } else {
        None
    };
    // Guards inside the generated wrapper must resume at the outer Python
    // CALL, because helper JitCodes have no blackhole entry point of their
    // own.  The full-body symbol is the authority for that caller frame's
    // liveness and resume coordinate (the same setup used by the orthodox
    // w_list_append descent below).  Resolve it before recording any guards
    // or synthetic allocations so a missing coordinate is a clean decline.
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: snapshot_sym is installed for the lifetime of the enclosing
    // full-body walk and is read-only here.
    let sym = unsafe { &*sym_ptr };
    if sym.jitcode().is_null() {
        return Ok(None);
    }
    let (call_site_py_pc, vsd_value, outer_jitcode_index, call_site_marker) = if nested_helper_entry
        .is_some()
    {
        (
            ctx.entry_py_pc(),
            0,
            ctx.outer_jitcode_index,
            ctx.outer_resume_marker_jit_pc,
        )
    } else {
        unsafe {
            let jc = &*sym.jitcode();
            let jc_index = jc.index as u32;
            let marker = jc.payload.resume_marker_for_jitcode_pc(op.pc);
            // Forward py twin first (#73 phase-3): equals the
            // containing coordinate plus trivia normalization by
            // construction; the containing lookup survives for the
            // empty-twin class, and the trivia skip below is an identity
            // on the twin path.
            let mut py = jc
                .payload
                .forward_py_pc_for_jitcode_pc(op.pc)
                .unwrap_or_else(|| {
                    crate::py_coord::note_empty_twin_fallback(
                        "builtin_call",
                        jc.index,
                        op.pc as i32,
                    );
                    crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, op.pc)
                });
            if jc.payload.code_ptr.is_null() {
                (py, sym.valuestackdepth() as i64, jc_index, marker)
            } else {
                let codeobj = &*jc.payload.code_ptr;
                py = skip_python_trivia_forward(codeobj, py as usize) as u32;
                let depth = if jc.payload.depth_trivia_populated() {
                    jc.payload.depth_trivia_for_jitcode_pc(op.pc)
                } else {
                    crate::liveness::liveness_for(jc.payload.code_ptr)
                        .depth_at_py_pc()
                        .get(py as usize)
                        .copied()
                };
                let vsd = depth
                    .map(|d| (sym.nlocals() + d as usize) as i64)
                    .unwrap_or(sym.valuestackdepth() as i64);
                (py, vsd, jc_index, marker)
            }
        }
    };
    let call_site_word = call_site_marker
        .map(|marker| marker as i32)
        .unwrap_or(majit_ir::resumedata::NO_JITCODE_PC);
    // Rewind point for the un-lowered-helper decline below.  Nothing above
    // this line records IR or touches the heap cache, so cutting back to it
    // leaves the caller's trace exactly as the ordinary residual call found it.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();
    let call_site_active = if nested_helper_entry.is_some() {
        ctx.outer_active_boxes.clone()
    } else {
        collect_outer_active_boxes(
            sym,
            ctx.trace_ctx,
            ctx.registers_i,
            ctx.registers_r,
            ctx.registers_f,
            outer_jitcode_index,
            false,
            call_site_word,
            op.pc as i32,
            OuterActiveBoxesEntryTwin::Plain,
            "builtin_wrapper_call_site",
            None,
            &[],
            None,
        )
    };

    // The generated builtin-wrapper ABI takes its `&[PyObjectRef]` argument
    // in r0 and begins by checking its length.  Resolve that instruction's
    // descriptor operand now, before switching the sub-walk to the global
    // descriptor pool below. The wrapper starts with arraylen(r0), but Charon
    // emits a distinct descriptor for slice length (header metadata) and
    // slice item access (element metadata). Heapcache array-item keys use the
    // latter, exactly like RPython `_do_getarrayitem_gc_any(arraydescr)`;
    // seeding under the arraylen descriptor makes the later getitem miss and
    // manufactures a Box without its recording-time `.value`.
    let Some(wrapper_args_descr_index) = wrapper_args_item_descr_index(body.code) else {
        return Ok(None);
    };

    let mut callable_guard_op = r_args[0];
    let mut receiver_op = method_form.then_some(r_args[1]);
    if bound_method {
        // pypy/interpreter/function.py `_Method._immutable_fields_`:
        // guard the carrier layout, read both fields live, and only promote
        // the immutable function identity used to select BuiltinCode.func.
        let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
        walker_guard_class(ctx, op.pc, r_args[0], method_type_addr)?;
        callable_guard_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_function_descr(),
        );
        let live_receiver = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            r_args[0],
            crate::descr::method_w_self_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            live_receiver,
            majit_ir::Value::Ref(majit_ir::GcRef(receiver.unwrap() as usize)),
        );
        receiver_op = Some(live_receiver);
    }
    if !callable_guard_op.is_constant() {
        // `callable_guard_op` can be the live `Method.w_function` field read
        // above.  The GuardValue pins it to `callable` at runtime; retain the
        // same recording-time Box.value so a multi-frame bridge recipe does
        // not mistake an unstamped field-read shadow for Python NULL.
        ctx.trace_ctx.set_opref_concrete(
            callable_guard_op,
            majit_ir::Value::Ref(majit_ir::GcRef(callable as usize)),
        );
        let expected = ctx.trace_ctx.const_ref(callable as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[callable_guard_op, expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        ctx.trace_ctx
            .heap_cache_mut()
            .replace_box(callable_guard_op, expected);
    }

    let mut wrapper_items = Vec::with_capacity(r_args.len().saturating_sub(1));
    let mut wrapper_item_concretes = Vec::with_capacity(arg_concretes.len().saturating_sub(1));
    if let Some(receiver_op) = receiver_op {
        wrapper_items.push(receiver_op);
        wrapper_item_concretes.push(ConcreteValue::Ref(receiver.unwrap()));
    }
    wrapper_items.extend_from_slice(&r_args[2..]);
    wrapper_item_concretes.extend_from_slice(&arg_concretes[2..]);
    for (&item, concrete) in wrapper_items.iter().zip(&wrapper_item_concretes) {
        if let ConcreteValue::Ref(value) = concrete
            && !value.is_null()
        {
            // Box.value is the recording-time shadow, not a compile-time
            // constant.  The generated wrapper's getarrayitem returns this
            // same live box; seeding it lets py_type_check choose its observed
            // arm and emit the corresponding guards while retaining `self`
            // as a red input.
            ctx.trace_ctx.try_set_opref_concrete(
                item,
                majit_ir::Value::Ref(majit_ir::GcRef(*value as usize)),
            );
        }
    }

    let array_descr = crate::state::pyobject_gcarray_descr();
    let len = ctx.trace_ctx.const_int(wrapper_items.len() as i64);
    let args_array =
        ctx.trace_ctx
            .record_op_with_descr(OpCode::NewArrayClear, &[len], array_descr.clone());
    ctx.trace_ctx
        .heap_cache_mut()
        .new_array(args_array, len, true);
    for (index, &item) in wrapper_items.iter().enumerate() {
        let index = ctx.trace_ctx.const_int(index as i64);
        ctx.trace_ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[args_array, index, item],
            array_descr.clone(),
        );
        ctx.trace_ctx
            .heapcache_setarrayitem(args_array, index, wrapper_args_descr_index, item);
    }

    if nested_helper_entry.is_none() && sym.owns_virtualizable_shadow() {
        let last_instr = call_site_py_pc as i64 - 1;
        let last_instr_op = ctx.trace_ctx.const_int(last_instr);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "last_instr",
            last_instr_op,
            Value::Int(last_instr),
        );
        let vsd_op = ctx.trace_ctx.const_int(vsd_value);
        crate::trace_opcode::mirror_vable_static_to_boxes(
            ctx.trace_ctx,
            "valuestackdepth",
            vsd_op,
            Value::Int(vsd_value),
        );
    }

    // Build-time canonical helper JitCodes use the one global Assembler
    // descriptor pool.  Temporarily give the wrapper sub-frame that pool;
    // its nested inline_call descriptors then resolve the generated child
    // JitCodes (e.g. W_Random::random -> Random::random) by global index.
    let saved_entry = ctx.entry_py_pc;
    let saved_marker = ctx.outer_resume_marker_jit_pc;
    let saved_oji = ctx.outer_jitcode_index;
    let saved_active = std::mem::take(&mut ctx.outer_active_boxes);
    let saved_descr_refs = ctx.descr_refs;
    let saved_raw_descrs = ctx.raw_descrs;
    let saved_lookup = ctx.sub_jitcode_lookup;
    let saved_fbw_mode = ctx.fbw_mode;
    let journal_before = fbw_store_journal_len();
    let unjournaled_before = fbw_has_unjournaled_effect();
    ctx.entry_py_pc = EntryPyPc::Jit(op.pc);
    ctx.outer_resume_marker_jit_pc = call_site_marker;
    ctx.outer_jitcode_index = outer_jitcode_index;
    ctx.outer_active_boxes = call_site_active;
    ctx.descr_refs = crate::jitcode_runtime::all_descr_refs();
    ctx.raw_descrs = RawDescrPool::Global;
    ctx.sub_jitcode_lookup = &GLOBAL_SUB_JITCODE_LOOKUP_FN;
    ctx.fbw_mode.inline_subwalk = true;
    ctx.fbw_mode.inline_caller_py_pc = Some(call_site_py_pc);
    // `transparent_helper_subwalk` is set by `run_sub_jitcode_walk` on the
    // sub-context it builds, so every descent into a canonical helper body
    // carries it — not just the ones entered from another sub-walk.
    let _helper_frame =
        nested_helper_entry.map(|frame| InlineFrameGuard::enter(ctx.session, 0, Some(frame)));
    let walk_result = run_sub_jitcode_walk(
        ctx,
        op.pc,
        &body,
        &[],
        &[],
        &[args_array],
        &[ConcreteValue::Null],
        &[],
    );
    ctx.fbw_mode = saved_fbw_mode;
    ctx.entry_py_pc = saved_entry;
    ctx.outer_resume_marker_jit_pc = saved_marker;
    ctx.outer_jitcode_index = saved_oji;
    ctx.outer_active_boxes = saved_active;
    ctx.descr_refs = saved_descr_refs;
    ctx.raw_descrs = saved_raw_descrs;
    ctx.sub_jitcode_lookup = saved_lookup;

    let walk_result = match walk_result {
        Ok(outcome) => outcome,
        // `try_execute_residual_call_via_executor` declines an un-lowered
        // in-body helper (a tagged symbolic fnaddr) while inlining a
        // sub-jitcode, so the descent aborts instead of baking the hash as a
        // code address.  Propagating that abort from here strands the CALL:
        // this walk is the authoritative executor and the descent declined
        // *before* running the call, so the aborted trace resumes past a
        // Python instruction whose effect never happened — `d.popleft()`
        // returns its value and leaves the element in place.  Roll the partial
        // descent back and let the ordinary residual call run, the same way
        // the orthodox `w_list_append` descent does.  A descent that already
        // applied an effect cannot be rewound this way, so it keeps the abort.
        Err(DispatchError::OrthodoxSubWalkTraceUnsupported { .. })
            if fbw_store_journal_len() == journal_before
                && fbw_has_unjournaled_effect() == unjournaled_before =>
        {
            ctx.trace_ctx.cut_trace(pre_fold_pos);
            ctx.trace_ctx.heap_cache_mut().reset();
            bool_box_truth_reset();
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    match walk_result {
        DispatchOutcome::SubReturn {
            result: Some(value),
        } => {
            let concrete = concrete_from_recorded_opref(ctx, value);
            write_ref_reg(ctx, op.pc, dst, value, concrete)?;
            Ok(Some((DispatchOutcome::Continue, op.next_pc)))
        }
        DispatchOutcome::SubReturn { result: None } => {
            Err(DispatchError::UnexpectedVoidSubReturn { pc: op.pc })
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
        DispatchOutcome::Terminate => Ok(Some((DispatchOutcome::Terminate, op.next_pc))),
        DispatchOutcome::SwitchToBlackhole {
            reason,
            raising_exception,
        } => Ok(Some((
            DispatchOutcome::SwitchToBlackhole {
                reason,
                raising_exception,
            },
            op.next_pc,
        ))),
        DispatchOutcome::CloseLoop { .. }
        | DispatchOutcome::CompileTracePending { .. }
        | DispatchOutcome::SubLoopCalleeCallAssembler { .. } => {
            Err(DispatchError::SubWalkClosedLoop { pc: op.pc })
        }
        DispatchOutcome::Continue => {
            unreachable!(
                "walk() only exits on Terminate / SubReturn / SubRaise / SwitchToBlackhole"
            )
        }
    }
}

/// Read one of the callee function's `_immutable_fields_` slots live and pin
/// the value this inline baked, replacing the box so later reads fold.
///
/// `function.py:34-42` declares `code?` / `w_func_globals?` / `closure?[*]` /
/// `defs_w?[*]`; pyre's setters do not yet force the quasi-immutable
/// invalidation, so a `GuardValue` stands in for it.
fn walker_guard_function_field<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op_pc: usize,
    callable: OpRef,
    descr: majit_ir::DescrRef,
    expected: i64,
) -> Result<(), DispatchError> {
    let field_op = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, callable, descr);
    ctx.trace_ctx.try_set_opref_concrete(
        field_op,
        majit_ir::Value::Ref(majit_ir::GcRef(expected as usize)),
    );
    let expected_op = ctx.trace_ctx.const_ref(expected);
    ctx.trace_ctx
        .record_guard(OpCode::GuardValue, &[field_op, expected_op], 0);
    walker_capture_snapshot_for_last_guard(ctx, op_pc)?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(field_op, expected_op);
    Ok(())
}

/// Shared post-resolution half of the FBW inline lever. Ordinary Python calls
/// resolve their callee from the CALL operand; builtin-dispatch specializers
/// resolve an app-level descriptor first and enter here with that function as
/// the callee while independently pinning the original builtin callable.
#[allow(clippy::too_many_arguments)]
/// Latch the outer CALL boundary as the forward-resume point for a discarded
/// inline sub-walk, so the walk driver re-executes the whole call in the
/// interpreter instead of rolling back and replaying the loop from entry.
///
/// Sound only when the attempt committed nothing observable: it must be the
/// top-level inline, no unjournaled effect may predate it, and the callee must
/// have executed no concrete effect.  Otherwise the re-execution would apply
/// twice what the sub-walk already ran.  Every caller that discards a sub-walk
/// result must go through this — returning the error without latching leaves
/// the driver replaying the loop, which repeats the callee's effects.
fn latch_abort_call_resume<Sym: WalkSym>(
    code: &[u8],
    op: &DecodedOp,
    ctx: &WalkContext<'_, '_, Sym>,
    is_top_inline: bool,
    unjournaled_before_subwalk: bool,
    executed_effects_before: usize,
    abort_flush_call_jitcode_coord: Option<(u32, usize)>,
) {
    if !is_top_inline
        || unjournaled_before_subwalk
        || fbw_executed_effect_count() != executed_effects_before
    {
        return;
    }
    let Some((outer_jitcode_index, call_jitcode_pc)) = abort_flush_call_jitcode_coord else {
        return;
    };
    if let Some(stack) = reconstructed_all_ref_call_stack(code, op, ctx) {
        fbw_set_abort_call_resume(outer_jitcode_index, call_jitcode_pc, stack);
    }
}

fn inline_caller_py_pc_from_snapshot<Sym: WalkSym>(
    ctx: &WalkContext<'_, '_, Sym>,
    call_jitcode_pc: usize,
) -> Option<u32> {
    let sym_ptr = ctx.fbw_mode.snapshot_sym;
    if sym_ptr.is_null() {
        return None;
    }
    let sym = unsafe { &*sym_ptr };
    if sym.jitcode().is_null() {
        return None;
    }
    let jc = unsafe { &*sym.jitcode() };
    Some(
        crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.payload.metadata, call_jitcode_pc)
            as u32,
    )
}

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
    mut callee_args: Vec<OpRef>,
    mut callee_arg_concretes: Vec<ConcreteValue>,
    method_form: bool,
    bound_method: Option<BoundMethodInline>,
    w_code: *const (),
    nparams: usize,
    has_closure: bool,
    exception_receiver_guard: Option<ExceptionInlineReceiverGuard>,
    arg_class_guard: Option<ArgClassGuard>,
    entry_is_call_boundary: bool,
    require_str_result: bool,
    constructor_result: Option<(OpRef, ConcreteValue)>,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    // `_compute_flatcall` (`pycode.py:256-268`) leaves `fast_natural_arity`
    // HOPELESS for a `*args` / `**kwargs` / keyword-only callee.  The general
    // `funcrun` path still traces through `_match_signature`, which writes a
    // surplus tuple for `*args` (`argument.py:222-234`); seed that one extra
    // local here while keeping `**kwargs` and keyword-only locals residual.
    let positional_only = fbw_callee_scope_is_positional_only(w_code);
    let vararg_slot = fbw_callee_vararg_slot(w_code);
    if !positional_only && vararg_slot.is_none() {
        return Ok(None);
    }
    // `Function.funccall_valuestack` fills every parameter the call left
    // unbound from `defs_w` before entering the frame
    // (`function.py:188-193,217-231`); `Arguments.parse` reaches the same frame
    // shape for a keyword call.  Mirror that shape here.  Placeholder boxes are
    // replaced by live guarded tuple-item reads after all non-emitting
    // eligibility checks.
    //
    // A positional call leaves the unbound parameters as a missing tail, but a
    // keyword one can leave a hole anywhere — `f(i, c=7)` on `def f(a, b=3,
    // c=5)` binds slots 0 and 2 and leaves 1 — so the seeding vectors carry
    // `OpRef::NONE` for a hole and the set is collected rather than assumed
    // contiguous.  Only the seeding writes that sentinel, so it cannot collide
    // with a genuinely passed argument.
    let missing: Vec<usize> = (0..nparams)
        .filter(|&i| callee_args.get(i).is_none_or(|arg| *arg == OpRef::NONE))
        .collect();
    let positional_defaults = if missing.is_empty() {
        None
    } else {
        let Some(defaults) =
            (unsafe { positional_defaults_for_inline(callable, &missing, nparams) })
        else {
            return Ok(None);
        };
        callee_args.resize(nparams, OpRef::NONE);
        callee_arg_concretes.resize(nparams, ConcreteValue::Null);
        for &(param_index, _, value) in &defaults.values {
            callee_arg_concretes[param_index] = ConcreteValue::Ref(value);
        }
        Some(defaults)
    };
    // The surplus positional arguments `_match_signature` packs into the
    // vararg.  Collected here and emitted alongside the defaults below, so
    // every remaining eligibility check still declines without having recorded
    // the tuple build; the placeholder box mirrors how a default is carried.
    let vararg_surplus = if vararg_slot.is_some() {
        // `_match_signature` (`argument.py:194-201`) prepends a receiver to
        // `args_w` — and so to the vararg tuple — when the callee has no
        // positional parameter to hold it.  A bound method reaches here with
        // `callee_args[0]` still the placeholder the resolved half replaces
        // with `GetfieldGcR(Method.w_self)`, which is emitted after the split
        // below; folding the placeholder into the tuple would put the Method
        // object where the receiver belongs.  Decline that one shape.
        if bound_method.is_some() && nparams == 0 {
            return Ok(None);
        }
        if callee_arg_concretes.len() != callee_args.len() || callee_args.len() <= nparams {
            // The empty tuple is a runtime singleton (`() is tuple([])`), so a
            // freshly allocated walker tuple would not be the object the
            // interpreter installs for a zero-surplus call.
            return Ok(None);
        }
        let surplus_ops: Vec<OpRef> = callee_args[nparams..].to_vec();
        let mut surplus_concretes = Vec::with_capacity(surplus_ops.len());
        for concrete in &callee_arg_concretes[nparams..] {
            let ConcreteValue::Ref(obj) = *concrete else {
                return Ok(None);
            };
            if obj.is_null() {
                return Ok(None);
            }
            surplus_concretes.push(obj);
        }
        // A new allocation with no heap mutation, safe during the walk, and the
        // same constructor `emit_object_tuple_inline` reproduces.
        let concrete = pyre_object::w_tuple_new_array_backed(surplus_concretes);
        if concrete.is_null() {
            return Ok(None);
        }
        callee_args.truncate(nparams);
        callee_arg_concretes.truncate(nparams);
        callee_args.push(OpRef::NONE);
        callee_arg_concretes.push(ConcreteValue::Ref(concrete));
        Some((surplus_ops, concrete))
    } else {
        None
    };
    let seeded_locals = nparams + usize::from(vararg_slot.is_some());
    // Does any incoming binding land a value the callee's register banks can
    // hold unboxed?  Only the `is`-against-None scan below consults this; see
    // its hazard-2 arm for why an int-specialized tested local is unsafe to
    // inline.  A default is pushed above as a raw `Ref` without going through
    // `ConcreteValue::from_pyobj`, so re-classify here rather than matching the
    // variant alone — `def _read_from_buffer(self, size=-1)` reaches the scan
    // with `size` as `Ref(<int object>)` and is exactly the shape that
    // miscompiled.
    let callee_binds_an_unboxed_local = callee_arg_concretes.iter().any(|c| {
        let classified = match *c {
            ConcreteValue::Ref(obj) => ConcreteValue::from_pyobj(obj),
            other => other,
        };
        matches!(
            classified,
            ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_)
        )
    });
    // Vararg/over-arity calls still use the ordinary residual path. A closure
    // is admissible when it has freevars only: the existing cell objects can
    // be threaded into this callee's own frame exactly as
    // PyFrame::finish_for_call_with_globals_obj does. A callee with cellvars
    // needs fresh cell allocation and stays residual until that constructor
    // half is ported too.
    if callee_args.len() != seeded_locals {
        return Ok(None);
    }
    let raw_callee_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_callee_code.is_null() {
        return Ok(None);
    }
    let callee_code = unsafe { &*raw_callee_code };
    let mut concrete_freevar_cells = Vec::new();
    let concrete_closure = if has_closure {
        if !callee_code.cellvars.is_empty() {
            return Ok(None);
        }
        let closure = unsafe { pyre_interpreter::function_get_closure(callable) };
        if closure.is_null()
            || !unsafe { pyre_object::is_tuple(closure) }
            || unsafe { pyre_object::w_tuple_len(closure) } != callee_code.freevars.len()
        {
            return Ok(None);
        }
        for i in 0..callee_code.freevars.len() {
            let Some(cell) = (unsafe { pyre_object::w_tuple_getitem(closure, i as i64) }) else {
                return Ok(None);
            };
            concrete_freevar_cells.push(cell);
        }
        closure
    } else {
        if !callee_code.freevars.is_empty() {
            return Ok(None);
        }
        pyre_object::PY_NULL
    };
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
    // Inlining a callee whose body carries an `abort_permanent` marker walks
    // the sub-walk straight into it.  That surfaces as
    // `TraceAction::AbortPermanent`, which stamps `DONT_TRACE_HERE` on the
    // CALLER loop's green key — so one unported opcode anywhere in a callee
    // permanently un-JITs the loop that calls it.  Upstream keeps this
    // decision static and on the callee: `codewriter/policy.py:48-84`
    // `look_inside_graph` reads whole-graph properties before tracing, and its
    // own comment (:78-79) spells out the consequence of a "no" — "the call
    // will be turned into a residual call".  Answer the same way.
    //
    // `loop_inlines_abort_permanent_callee` (`trace.rs`) already screens this
    // up front, but only for callees it can resolve statically out of globals
    // and frame slots; a bound method, a container element or a call result
    // reaches here unscreened.  At this point the callee is concrete, so the
    // screen is exact.
    //
    // Whole-body, like upstream's whole-graph test: a marker on a path this
    // trace happens not to take still costs only the inline if we decline,
    // where walking into it costs the whole loop, permanently.
    //
    // Decided once per callee and kept on its jitcode payload rather than
    // re-scanned here on every call site that reaches this recipe.  `None`
    // means no installed body or descr pool, which the pool fetch immediately
    // below declines on regardless.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if body_facts.has_abort_permanent {
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
    // EXACT int/float only.  These feed `fbw_callee_body_replay_safety`, whose
    // question is "will the walker specialize this body's BINARY_OP to a native
    // op, leaving no residual to replay?".  The walker's specialization admits
    // only exact builtin numbers (`walker_int_specialization_operands` /
    // `walker_float_specialization_operands` both require
    // `is_exact_builtin_instance`), because a numeric subclass keeps the
    // builtin layout while its Python-visible class lives in `w_class` and may
    // define its own `__add__`.  `is_int` / `is_float` are `ob_type` checks
    // that a subclass passes, so using them here claims a specialization that
    // will not happen and admits a body whose real residual a replay would
    // double.
    //
    // Preserve exactness per argument.  Method-form calls put a usually
    // nonnumeric `self` in slot 0; folding all arguments into one boolean
    // incorrectly made that erase the proof for an independent numeric `x`.
    let exact_numeric_args: Vec<ExactNumericArg> = callee_arg_concretes
        .iter()
        .map(|concrete| {
            let (plain_int, exact_float) = match concrete {
                ConcreteValue::Int(_) => (true, false),
                ConcreteValue::Float(_) => (false, true),
                ConcreteValue::Ref(obj) if !obj.is_null() => unsafe {
                    (
                        pyre_object::is_plain_int1(*obj),
                        pyre_object::is_plain_float_strict(*obj),
                    )
                },
                ConcreteValue::Bool(_) | ConcreteValue::Ref(_) | ConcreteValue::Null => {
                    (false, false)
                }
            };
            ExactNumericArg {
                numeric: plain_int || exact_float,
                plain_int,
            }
        })
        .collect();
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
    //
    // A plain ROOT bridge walk — no carrier resume, not an inline sub-walk,
    // an empty framestack, and
    // a live root portal — is uniform with a primary trace, so its second
    // virtual frame is seeded and snapshot-covered exactly as the loop's is.
    // There the decline is lifted: the call falls through to the self-recursive
    // unroll gate and multiframe seed as if walked from a primary trace.
    // True once this attempt takes the root-bridge admission for a
    // self-recursive callee.  The admitted top-level inline's
    // body sub-walk reaches its own recursive CALL as a nested residual, which
    // `fbw_abort_nested_unjournaled_residual` declines on the self-recursive
    // hazard arm — an abort storm that folds the whole guard bridge back to
    // residual.  The `CALL_ASSEMBLER` self-recursion fold already exempts that
    // decline via `SELFREC_CA_FOLD_ACTIVE`; the same exemption applies to this
    // admitted inline, whose recursive residual runs concretely at the
    // pre-execute site (executed, so no replay double-apply).
    let mut bridge_rec_root_selfrec = false;
    if ctx.trace_ctx.is_bridge_trace
        && args_all_builtin_integer
        && fbw_callee_body_has_binary_op_residual(body.code, callee_descr_refs)
    {
        // A depth-1 carrier-resume sub-walk (`drive_bridge_frame_subwalk`) drives
        // its reconstructed frame as the sub-walk root with an empty framestack,
        // so it is uniform with a plain root bridge for the multiframe seed —
        // admit it too so a rare guard-bridge continuation inlines its nested
        // int-arith calls instead of residualizing them (the gh#343 branchy-callee
        // cost).
        // The nested levels are admitted on the same terms.
        // `opimpl_recursive_call` (`pyjitpl.py:1390-1416`) makes the inline
        // decision from the portal-frame count alone — already applied above via
        // `fbw_inline_recursion_count` — and asks nothing about how deep the
        // framestack is or whether a bridge or a primary trace is walking.
        //
        // An inline sub-walk is the exception, and it is one with no upstream
        // counterpart: `perform_call` pushes onto `MetaInterp.framestack` and
        // returns to the single `interpret()` loop, so upstream is never inside
        // one walk while starting another and can never seed a multiframe
        // snapshot from a sub-walk. Seeding one here makes the wasm bridge
        // replay re-execute the sub-walk's body — wrong output from
        // `synth/ca_bridge_multiframe_resume_double_call` and
        // `synth/recursion_memo_branch`, and a `fib_recursive` timeout, all
        // while dynasm and cranelift stay green.
        let root_bridge = !ctx.fbw_mode.carrier_resume
            && !ctx.fbw_mode.inline_subwalk
            && !ctx.fbw_mode.snapshot_sym.is_null();
        // A carrier-resume sub-walk (`drive_bridge_frame_subwalk`) drives its
        // reconstructed frame(s) forward through the same metainterp the initial
        // trace uses, so its inline of a nested int-arith call is the SAME as a
        // primary trace's — admit it (any inline depth) so the rare guard-bridge
        // continuation inlines instead of residualizing.
        let subwalk_admit = ctx.fbw_mode.carrier_resume && !ctx.fbw_mode.snapshot_sym.is_null();
        let safe_root_bridge = root_bridge || subwalk_admit;
        if !safe_root_bridge {
            return Ok(None);
        }
        bridge_rec_root_selfrec = unsafe {
            let raw = pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject;
            !raw.is_null() && pyre_interpreter::code_is_self_recursive(&*raw)
        };
    }
    // A callee `fbw_abort_nested_unjournaled_residual` already named on its
    // hazard arm residualizes from here on.  The hazard is a static property of
    // the callee's `CodeObject` (loop-bearing / self-recursive), so re-inlining
    // it rebuilds the identical framestack and reaches the identical abort —
    // the enclosing loop is retired for a decline that belongs to the callee.
    // `warmstate.py:331` `disable_noninlinable_function` is the same answer:
    // the flag it sets means "do not inline calls to this function", and the
    // enclosing loop is left free to retrace.
    //
    // `bridge_rec_root_selfrec` is exempt: that admission carries its own
    // `SELFREC_CA_FOLD_ACTIVE` exemption from the hazard arm (:2696), so its
    // recursive residual is not what named the callee here.
    if !bridge_rec_root_selfrec && fbw_hazardous_inline_denied(callee_code_key) {
        return Ok(None);
    }
    // A legacy, unseeded inline sub-walk inside a FOR_ITER body resumes a guard
    // at the caller's CALL boundary, so deopt re-executes the whole callee.
    // Replaying a live-heap mutation would double it, so a Dirty body stays on
    // the residual call path.
    //
    // A body whose only unproven ops are Python-level CALL residuals is
    // admitted too: this same gate re-runs for each callee the lever resolves
    // below it, and one it cannot inline aborts before executing
    // (`fbw_abort_nested_unjournaled_residual`) and denies this callee, so a
    // deferred body commits nothing either.  Without that the whole nest
    // declines — `helper(i)` calling `add(i, 1, 2)` residualizes both calls
    // per iteration, though each body on its own is pure arithmetic.
    let mut foriter_deferred_admit = false;
    let mut foriter_dirty_bound = false;
    if fbw_foriter_inflight_active() {
        let safety = fbw_callee_body_replay_safety(
            body.code,
            &exact_numeric_args,
            body.num_regs_i,
            body.constants_i,
            body.num_regs_r,
            body.constants_r,
            callee_descr_refs,
            method_form,
        );
        let legacy_admit = match safety {
            CalleeReplaySafety::Clean => true,
            CalleeReplaySafety::DeferredCall => {
                // The deferred promise rests on the abort REWINDING to the
                // enclosing CALL and re-executing it from scratch, so the
                // entry has to be a boundary the rewind can name.  What
                // decides that is the entry opcode's stack effect rather than
                // whether it is spelled CALL: one that merely peeks its
                // operands re-executes from the stack it already had.  A
                // `BINARY_OP` or `COMPARE_OP` entry is not: the flush resumes
                // one operand short and the whole iteration's contribution is
                // dropped, silently — the subscript inline observed its index
                // operand replaced by an unrelated live Ref.  Each caller
                // states this directly in `entry_is_call_boundary`; the older
                // `arg_class_guard.is_none()` proxy stood for the same
                // property and rotted, because the `obj[key]` inline enters
                // from `BINARY_OP` while passing no `arg_class_guard`.  A
                // `Clean` body is still admitted from there — it has nothing
                // that can abort.
                //
                // The widened method-form surface — an unbound callee whose
                // body reads `self.attr` — was admitted here only once the
                // receiver was proven not to be a type object, because a type
                // receiver's read went through `type.__getattribute__` and
                // reached the deferred abort path.  That read folds now
                // ([`try_walker_specialize_load_type_name_attr`]), so a type
                // receiver reaches no residual to abort on either and the proof
                // is no longer what admits it: a classmethod body reading
                // `cls.__name__` measured 1082 ns/iter on the decline against
                // 1.6 once admitted.  A body whose attribute read does NOT fold
                // — any metaclass other than `type` — still aborts once and is
                // denied, which is what this arm's promise has always rested
                // on.
                //
                // A callee with its own exception handler has protected-region
                // state that must be restored at the callee's precise resume
                // point.  If a deferred residual later aborts after a folded
                // effect, caller-boundary replay can repeat the protected
                // entry or skip the handler cleanup.  Keep handler-bearing
                // bodies on the residual path unless this scan proves them
                // clean.
                foriter_deferred_admit = entry_is_call_boundary
                    && !body_facts.owns_loop_header
                    && !pyre_interpreter::code_has_for_iter(callee_code)
                    && !body_facts.has_exception_table
                    && !fbw_foriter_deferred_call_denied(callee_code_key);
                foriter_deferred_admit
            }
            CalleeReplaySafety::Dirty => {
                // A stored bound method has an explicit receiver and can use
                // the multi-frame red-frame path below. Keep loop-bearing and
                // recursive callees residual: either requires another loop
                // header rather than one bounded callee walk.
                foriter_dirty_bound = bound_method.is_some()
                    && !pyre_interpreter::code_has_for_iter(callee_code)
                    && !pyre_interpreter::code_is_self_recursive(callee_code);
                foriter_dirty_bound
            }
        };
        if fbw_inline_diag_enabled() {
            eprintln!(
                "[inline-foriter-gate] pc={} legacy_admit={legacy_admit} exact_numeric_args={} \
                 safety={safety:?} deferred_admit={foriter_deferred_admit}",
                op.pc,
                exact_numeric_args.iter().filter(|arg| arg.numeric).count(),
            );
        }
        // An unbound `Dirty` body is not admitted by seeding its frame.  Its
        // residual can raise, and the local `except` that catches it is a
        // callee-owned catch edge the inline path does not compile, so the
        // exception escapes the caller instead of being handled where the
        // source handles it.  Stored bound methods instead take the explicit
        // multi-frame red-frame path above.
        if !legacy_admit {
            return Ok(None);
        }
    }
    // A widened method-form body that also raises was declined here until the
    // resume-liveness filter started keeping the raise operand.  The decline
    // existed because a guard whose resume coordinate landed on the `Reraise`
    // needed ref registers the recorded path never wrote
    // (`collect_callee_active_boxes`), and that decline arrived mid-recording
    // on a non-effect-free opcode with no mid-body carrier, discarding the
    // whole enclosing loop.  With the operand retained the sub-walk records the
    // handler region with the registers the coordinate names, so the body
    // inlines like any other.
    //
    // The raise need not even execute to have been caught by it: a dead
    // `if self.i < 0: raise` in an `o.m()` callee measured 1705 ns/call against
    // 15.3 once admitted, and `self.i >= self.n` 1207 -> 15.9.  Swapping that
    // `raise` for a `return` already measured 17.7, which is what named the
    // token rather than the branch or the attribute compare.
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
    // `typeobject.py descr_call` discards `__init__`'s result and returns the
    // instance.  Hold constructors to the strict straight-line path here;
    // together with the `constructor_result.is_none()` term on `strict_seed`,
    // this keeps `__init__` out of every resume chain.  No callee frame is
    // seeded, so a guard in `__init__` resumes at the caller's CALL coordinate
    // and re-runs the instantiation, making the result discard unnecessary to
    // represent.
    if constructor_result.is_some() && !strict_inlinable {
        return Ok(None);
    }
    // A zero-param callee has no positional argument to seed, so the register
    // convention above holds vacuously and the strict path serves it like any
    // other straight-line leaf.  The residual it would otherwise fall back to
    // is not cheap: `def f0(): return 1` called from a `while` loop measured
    // 569 ns/iter against 1.16 ns for the same call with one parameter.  Only a
    // zero-param callee the strict path cannot serve declines here, so such a
    // body still takes the residual rather than the decline-to-interpretation
    // below.
    if nparams == 0 && !strict_inlinable {
        return Ok(None);
    }

    // A self-recursive callee unrolls until its own frame count reaches
    // `max_unroll_recursion`, then routes to the direct `CALL_ASSEMBLER` arm
    // (`try_walker_call_assembler_self_recursive`, reached when this inline
    // attempt returns `Ok(None)`).  That bound is the one
    // `fbw_inline_recursion_count` already applied above, matching
    // `opimpl_recursive_call` (`pyjitpl.py:1390-1416`), which counts the portal
    // frames already on the framestack and inlines while the count is below
    // `max_unroll_recursion`.  A carrier-resume sub-walk is the exception: it
    // reconstructs its frames rather than entering them, so the count is not
    // the walk's own recursion depth.
    if !strict_inlinable && nparams >= 1 && ctx.fbw_mode.carrier_resume {
        let sym_ptr = ctx.fbw_mode.snapshot_sym;
        let self_recursive = !sym_ptr.is_null()
            && unsafe {
                pyre_interpreter::live_code_wrapper((*(*sym_ptr).jitcode()).raw_code() as *const ())
                    as *const ()
            } as usize
                == w_code as usize;
        if self_recursive {
            return Ok(None);
        }
    }
    // #68: a forward-branch-bearing callee is inlinable with a multi-frame
    // guard snapshot (its in-callee branch
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
    let multiframe_eligible = !strict_inlinable;
    let callee_frame_reg = if multiframe_eligible {
        crate::state::ensure_jitcode_index(callee_code_key as *const ())
            .map(|jc| crate::state::portal_red_regs_at(jc).0)
            .unwrap_or(u16::MAX)
    } else {
        u16::MAX
    };
    let inline_depth = ctx.session.borrow().framestack.len();
    let contains_raise = body_facts.contains_raise;
    // Evaluated only behind the three cheaper terms, so the short-circuit order
    // is the same one the single condition had.  Keeping the class lets the
    // decline census say which admission would widen it.
    let branchy_handler_safety =
        if contains_raise && !strict_inlinable && body_facts.has_exception_table {
            Some(fbw_callee_body_replay_safety(
                body.code,
                &exact_numeric_args,
                body.num_regs_i,
                body.constants_i,
                body.num_regs_r,
                body.constants_r,
                callee_descr_refs,
                false,
            ))
        } else {
            None
        };
    if matches!(branchy_handler_safety, Some(s) if s != CalleeReplaySafety::Clean) {
        // A branchy callee with its own exception handler can take a structural
        // abort after an earlier effectful Python opcode. The current
        // callee-rebuild payload resumes at the Python opcode owning the abort
        // jitcode pc; it cannot yet carry a post-op stack anchor. Re-entering
        // that opcode would repeat its residual effect (PyPy instead resumes
        // the live MIFrame at its precise resumepc). Keep that callee on the
        // ordinary residual path until the generated frame snapshot can
        // represent the precise post-effect coordinate. A terminal raising
        // callee without a handler retains its after-residual live anchor.
        crate::jitcode_dispatch::census_record(
            if branchy_handler_safety == Some(CalleeReplaySafety::DeferredCall) {
                "InlineCallee::BranchyHandlerDeferredCall"
            } else {
                "InlineCallee::BranchyHandlerDirty"
            },
        );
        return Ok(None);
    }
    // A callee that raises inline needs the cross-frame bridge the carrier
    // drain builds once a guard inside the compiled chain fails.  The drain
    // walks the paused middle frames between the raising leaf and the root, so
    // one intermediate frame may sit between the loop and the raise; a middle
    // that CATCHES is still declined, which is what holds the cap here instead
    // of letting a raising chain run to `fbw_max_multiframe_depth`.  A
    // value-returning chain (no raise) inlines to the full depth either way.
    //
    // Measured against one drain-complete binary, nine interleaved reps per
    // arm: `bench/synth/exception_escape_caller_frame_tb_node` runs at 0.70x
    // of the one-level cap, `bench/synth/gc_bug_bridge_flavor_traceback_names`
    // at 1.02x and `bench/synth/selfrec_tail_exception_unwind` at 0.99x.  A
    // third level regresses — `selfrec_tail_exception_unwind` takes
    // `guard_failures` from 937 to 7408 — because the unwind then crosses two
    // suspended copies of the same frame, the shape `fbw_max_rec_unroll_depth`
    // bounds above.
    let effective_multiframe_depth = if contains_raise {
        2
    } else {
        fbw_max_multiframe_depth()
    };
    // FOR_ITER must re-execute its iterator protocol as one unit when a body
    // guard fails.  Its Clean-only admission above makes caller-boundary
    // replay safe, and keeping the callee frame unseeded makes the terminating
    // branch resume at FOR_ITER so the residual maps IndexError to exhaustion.
    let force_caller_boundary_resume =
        call_descr.get_extra_info().pyre_helper == majit_ir::PyreHelperKind::ForIterNext;
    let try_multiframe = !force_caller_boundary_resume
        && multiframe_eligible
        && inline_depth < effective_multiframe_depth
        && callee_fast_path_inlinable_allowing_forward_branch(
            body.code,
            callee_descr_refs,
            ctx,
            callee_frame_reg,
        );
    // A strict straight-line callee at the top inline level is seeded with
    // its own frame red so guards can carry a real two-frame snapshot.  A
    // callee needing fresh cellvar allocation is not seeded — the seed block
    // below breaks out to the ordinary single-frame inline for it — so exclude
    // it here too, or the preflight would decline a CALL that path still
    // serves.  Constructor inlining also stays out of the seed: `typeobject.py
    // descr_call` owns the discard of `__init__`'s result, and the flattened
    // frame shape cannot reconstruct that discard from a two-frame in-callee
    // guard pause.
    let strict_seed = !force_caller_boundary_resume
        && strict_inlinable
        && inline_depth < fbw_max_multiframe_depth()
        && callee_code.cellvars.is_empty()
        && constructor_result.is_none();
    // Preflight the caller frame BEFORE the seed below records a virtual
    // PyFrame.  A CALL covered by a try/catch marker must remain residual so
    // its post-call catch resume routes an exception; returning after frame
    // emission would strand dead seed IR and force an abort/replay after the
    // callee already consumed external state.  RPython decides whether to
    // inline the graph before `perform_call` pushes its MIFrame.
    let precomputed_parent_frame = if try_multiframe || strict_seed {
        match compute_inline_caller_frame(ctx, op.pc, !callee_code.freevars.is_empty()) {
            Ok(parent) => Some(parent),
            Err(InlineCallerFrameDecline::TryBlockCatchMarker) => return Ok(None),
            Err(InlineCallerFrameDecline::Unavailable) if try_multiframe => return Ok(None),
            Err(InlineCallerFrameDecline::Unavailable) => None,
        }
    } else {
        None
    };
    if foriter_dirty_bound && !try_multiframe {
        return Ok(None);
    }
    if !strict_inlinable && !try_multiframe && !force_caller_boundary_resume {
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
        return Ok(None);
    }

    let mut callable_guard_op = callable_guard_op;
    let mut callable_guard_value = callable_guard_value;
    if let Some(bound) = bound_method {
        // `_Method._immutable_fields_ = ['w_function', 'w_instance']`
        // (pypy/interpreter/function.py:567).  Preserve those as red field
        // reads: guard only the Method layout and underlying function, then
        // pass the live receiver field into the callee.  Baking the receiver
        // concrete would collapse bound methods with different `self` values.
        let method_type_addr = &pyre_object::function::METHOD_TYPE as *const _ as i64;
        walker_guard_class(ctx, op.pc, bound.method_op, method_type_addr)?;
        let function_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            bound.method_op,
            crate::descr::method_w_function_descr(),
        );
        let receiver_op = crate::state::opimpl_getfield_gc_r(
            ctx.trace_ctx,
            bound.method_op,
            crate::descr::method_w_self_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            receiver_op,
            majit_ir::Value::Ref(majit_ir::GcRef(bound.receiver as usize)),
        );
        callee_args[0] = receiver_op;
        callable_guard_op = function_op;
        callable_guard_value = bound.function;
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
        jitcode_index: crate::state::ensure_jitcode_index(callee_code_key as *const ())
            .map_or(-1, |index| index as i32),
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

    // Not every caller pins the callee function itself.  A specializer that
    // resolves an app-level method behind a builtin — `str(e)` reaching an
    // exception subclass's `__str__` — passes the CALL's own operand, which is
    // the `str` builtin, while `callable` is the resolved `Function`.  Reading
    // `Function.code` off that operand is a type-confused load: it returns
    // whatever sits at the same offset in a `PyCFunction`, so the guard
    // compares a value that is not `code` and fails every iteration (99480
    // failures and 497 bridges on `synth/exception_subclass_attrs`, a 31x
    // slowdown).  Guard the fields only when the pinned object really is the
    // function whose code this inline resolved.
    //
    // A trace-constant callable is excluded for a second reason: the field
    // reads would dereference a baked `ConstPtr`, and a baked constant object
    // pointer is not GC-forwarded yet (gh #108 gc-table — see the note in
    // `synth/exception_subclass_attrs.py`).  Comparing against such a constant
    // is fine, but loading through one dangles as soon as a minor collection
    // moves the object: `synth/inline_subwalk_property_mutates` — a property
    // getter that allocates on every iteration — segfaults on cranelift under
    // CI's macOS runner with the reads in place.  The callable being constant
    // means something already pinned the object, so this only gives up the
    // `f.__code__ = g.__code__` re-check on that path.
    let guards_the_callee_function = !callable_guard_op.is_constant()
        && unsafe {
            (*callable_guard_value).ob_type as *const () as usize
                == &pyre_interpreter::FUNCTION_TYPE as *const _ as usize
                && pyre_interpreter::function_get_code(callable_guard_value) as usize
                    == callee_code_key
        };

    if !guards_the_callee_function {
        // Those sites resolve the callee through their own guarded path (a
        // type version tag, a receiver class guard); all this has to pin is
        // the operand they dispatched on.
        if !callable_guard_op.is_constant() {
            let expected = ctx
                .trace_ctx
                .const_ref(callable_guard_value as usize as i64);
            ctx.trace_ctx
                .record_guard(OpCode::GuardValue, &[callable_guard_op, expected], 0);
            walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
        }
    } else {
        // `function.py:91-96 getcode()` promotes `self.code`, never `self`.
        // The three constants this inline bakes — the code object below, the
        // globals namespace in `InlineCalleeConsts`, and the freevar cells
        // baked into the callee frame — are exactly the `?`-fields
        // `_immutable_fields_ = ['code?', 'w_func_globals?', 'closure?[*]',
        // 'defs_w?[*]']` names, so guard each of them off the live function.
        // `defs_w` has its own guard below.
        //
        // Reading them live is what `f.__code__ = g.__code__` needs: pinning
        // the object pins none of its fields.  Upstream is safe there because
        // `code?` is quasi-immutable and assigning it invalidates the traces
        // that folded it; pyre has no such hook yet, so the value guard stands
        // in for the invalidation.
        //
        // Guarding the function OBJECT instead pinned its identity, which a
        // callee built by a `MAKE_FUNCTION` in the caller's own loop body can
        // never match twice: every iteration allocates a fresh function, so
        // the guard failed forever while the code object it stands for is
        // loop-invariant.
        walker_guard_function_field(
            ctx,
            op.pc,
            callable_guard_op,
            crate::descr::function_code_descr(),
            callee_code_key as i64,
        )?;
        walker_guard_function_field(
            ctx,
            op.pc,
            callable_guard_op,
            crate::descr::function_w_globals_descr(),
            inline_consts.w_globals as i64,
        )?;
        if !concrete_freevar_cells.is_empty() {
            // `closure?[*]`: the tuple itself is rebuilt by every
            // `MAKE_FUNCTION`, so guarding its identity would reintroduce the
            // per-iteration failure.  The cells are what this inline bakes and
            // what the enclosing frame keeps stable, so read the tuple live and
            // guard each element instead.  The element count needs no guard of
            // its own: `code` above pins `co_freevars`, and a closure always
            // has exactly that many cells.
            let closure_op = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                callable_guard_op,
                crate::descr::function_closure_descr(),
            );
            ctx.trace_ctx.try_set_opref_concrete(
                closure_op,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_closure as usize)),
            );
            let items_op = crate::state::opimpl_getfield_gc_r(
                ctx.trace_ctx,
                closure_op,
                crate::descr::tuple_wrappeditems_descr(),
            );
            for (i, &cell) in concrete_freevar_cells.iter().enumerate() {
                let index_op = ctx.trace_ctx.const_int(i as i64);
                let cell_op = crate::state::trace_items_block_getitem_value_pure(
                    ctx.trace_ctx,
                    items_op,
                    index_op,
                );
                ctx.trace_ctx.try_set_opref_concrete(
                    cell_op,
                    majit_ir::Value::Ref(majit_ir::GcRef(cell as usize)),
                );
                let cell_const = ctx.trace_ctx.const_ref(cell as i64);
                ctx.trace_ctx
                    .record_guard(OpCode::GuardValue, &[cell_op, cell_const], 0);
                walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
                ctx.trace_ctx
                    .heap_cache_mut()
                    .replace_box(cell_op, cell_const);
            }
        }
    }

    if let Some(defaults) = positional_defaults {
        // `defs_w?`: read the live field on every compiled iteration.
        let defaults_op = ctx.trace_ctx.record_op_with_descr(
            OpCode::GetfieldGcR,
            &[callable_guard_op],
            crate::descr::function_defs_w_descr(),
        );
        ctx.trace_ctx.try_set_opref_concrete(
            defaults_op,
            majit_ir::Value::Ref(majit_ir::GcRef(defaults.tuple as usize)),
        );

        // `defs_w?[*]`: the elements are read live below, so all trace time
        // decided is WHICH element fills which missing parameter, and
        // `positional_defaults_for_inline` derives that from `len(defs_w)`
        // alone — the length is the only thing that has to be re-checked.
        //
        // Guard the tuple's identity all the same.  Replacing this with
        // `arraylen_gc` + a length `GuardValue` was implemented and reverted:
        // it answers correctly on every defaults shape, including the
        // specialised two-int tuple, but `synth/pickle_terminal_raise_resume`
        // then segfaults deterministically (EXC_BAD_ACCESS on a null in
        // compiled code, 5/5), while keeping the added class guard and
        // restoring this identity `GuardValue` is clean 3/3.  The length guard
        // is what is unsound here; the class guard is not.
        //
        // Identity is stricter than needed and costs nothing measured.  All
        // three compilers fold an all-constant defaults list into one code
        // constant — `codegen.py:582-590 _visit_defaults` takes the
        // `_tuple_of_consts` branch, and pyre's own compiler emits the same
        // single `LOAD_CONST (None, 7)` — so even a `def` re-executed inside
        // the caller's loop hands out the same tuple every iteration.  Only a
        // non-constant default expression (`def f(a=mk())`, which emits
        // `BUILD_TUPLE`) rebuilds it; no fixture in `bench/` has that shape,
        // and `make_function_inline`, the one loop-local `def` with a default,
        // records `guard_failures=1` for its whole run.
        let tuple_expected = ctx.trace_ctx.const_ref(defaults.tuple as i64);
        ctx.trace_ctx
            .record_guard(OpCode::GuardValue, &[defaults_op, tuple_expected], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;

        // Preserve the actual Ref boxes instead of baking one anchor's
        // concrete value.  Which read that is depends on where the element
        // lives; the identity guard above already proved the layout, so no
        // arm needs a class guard of its own.
        match defaults.repr {
            DefaultsRepr::ItemsBlock => {
                let items = crate::state::opimpl_getfield_gc_r(
                    ctx.trace_ctx,
                    defaults_op,
                    crate::descr::tuple_wrappeditems_descr(),
                );
                for (param_index, tuple_index, _) in defaults.values {
                    let index = ctx.trace_ctx.const_int(tuple_index as i64);
                    callee_args[param_index] = crate::state::trace_items_block_getitem_value_pure(
                        ctx.trace_ctx,
                        items,
                        index,
                    );
                }
            }
            DefaultsRepr::PairObject => {
                for (param_index, tuple_index, _) in defaults.values {
                    let descr = if tuple_index == 0 {
                        crate::descr::specialised_tuple_oo_value0_descr()
                    } else {
                        crate::descr::specialised_tuple_oo_value1_descr()
                    };
                    callee_args[param_index] =
                        crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, defaults_op, descr);
                }
            }
            DefaultsRepr::PairInt => {
                for (param_index, tuple_index, value) in defaults.values {
                    let descr = if tuple_index == 0 {
                        crate::descr::specialised_tuple_ii_value0_descr()
                    } else {
                        crate::descr::specialised_tuple_ii_value1_descr()
                    };
                    let raw = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, defaults_op, descr);
                    let elem = unsafe { pyre_object::w_int_get_value(value) };
                    let boxed = walker_box_int(ctx, op.pc, raw, elem)?;
                    // `wrapint` emits a heap `NewWithVtable`, so the stamped
                    // concrete has to be a heap pointer too — the walk-time
                    // `w_tuple_getitem` box above may be a tagged immediate.
                    // Re-home the argument's concrete onto whatever
                    // `box_int_concrete` picked, or the seeding loop below
                    // would re-stamp the op with the other one.
                    let concrete = box_int_concrete(elem, value as i64);
                    if let majit_ir::Value::Ref(gcref) = concrete {
                        callee_arg_concretes[param_index] =
                            ConcreteValue::Ref(gcref.as_usize() as pyre_object::PyObjectRef);
                    }
                    ctx.trace_ctx.set_opref_concrete(boxed, concrete);
                    callee_args[param_index] = boxed;
                }
            }
        }
    }

    // `scope_w[co_argcount + co_kwonlyargcount] = space.newtuple(starargs_w)`
    // (`argument.py:233-234`), replacing the placeholder pushed above.
    if let Some((surplus_ops, concrete)) = vararg_surplus {
        let tuple_op = crate::helpers::emit_object_tuple_inline(ctx.trace_ctx, &surplus_ops);
        ctx.trace_ctx.set_opref_concrete(
            tuple_op,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
        );
        callee_args[nparams] = tuple_op;
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
    for i in 0..seeded_locals {
        // RPython passes the same RefFrontendOp into the callee MIFrame, so
        // the Box keeps its recording-time pointer across the frame boundary.
        // Pyre also carries a register-local concrete shadow; mirror that
        // value back onto the canonical OpRef before constructing the callee
        // register file.  Loop-sidecar locals in particular can have a live
        // register concrete even when an intervening vable load returned a
        // fresh, previously unstamped OpRef.
        if let ConcreteValue::Ref(value) = callee_arg_concretes[i]
            && !value.is_null()
        {
            ctx.trace_ctx.try_set_opref_concrete(
                callee_args[i],
                majit_ir::Value::Ref(majit_ir::GcRef(value as usize)),
            );
        }
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
    // The seeded virtual callee frame /
    // shared ec / local count are hoisted so the sub-walk return site can
    // emit a `CALL_ASSEMBLER` into the callee loop token when the sub-walk
    // surfaces `SubLoopCalleeCallAssembler` (the callee reached its own loop
    // header).  A strict straight-line callee has no loop, so that outcome
    // never arises for it and the hoisted values are simply unused.
    let mut ca_callee_frame = OpRef::NONE;
    let mut ca_callee_ec = OpRef::NONE;
    let mut ca_nlocals = 0usize;
    // The seeded callee frame's runtime object, for the `enter`/`leave`
    // bracket below — the OpRef alone cannot carry it out of the seed block.
    let mut ca_concrete_frame = std::ptr::null_mut::<pyre_interpreter::PyFrame>();
    // A strict straight-line callee at the top inline level is seeded the same
    // way, so its in-callee guards route through the multi-frame snapshot.  A
    // deeper strict callee (`inline_depth >= fbw_max_multiframe_depth()`) keeps the
    // single-frame collapse — a 3-frame snapshot the resume path is sound for
    // only one paused caller frame.
    // True once the callee frame reds are actually seeded (all preconditions
    // below met).  For a strict callee this gates routing its guards through the
    // multi-frame snapshot vs. falling back to collapse.
    let mut callee_frame_seeded = false;
    // Names the `break 'seed` arm that left `callee_frame_seeded` false, for
    // the `[fbw-census]` collapse tally at the `parent_frame` decision below.
    // Empty means the seed block was never entered or ran to completion.
    let mut seed_break_reason: &'static str = "";
    // The concrete callee frame the seed block materializes, retained so the
    // sub-walk can put it on the interpreter frame chain: the walk executes
    // the callee's residuals for real, and a residual that reads the chain
    // (`sys._getframe`, a traceback) must see the callee it is running in.
    let mut concrete_callee_frame = std::ptr::null_mut::<pyre_interpreter::PyFrame>();
    // Each precondition below answers "can the multiframe seed serve this
    // callee".  A "no" declines the INLINE — `Ok(None)`, this function's own
    // did-not-inline answer, which every caller follows to the ordinary
    // residual call.  It must not be `Err`: that is
    // `LoopBearingCalleeInlineUnsupported`, which `trace.rs` maps to
    // `TraceAction::Abort`, discarding the whole enclosing loop trace.  And
    // because the arm does not call `fbw_decline`, the same static, callee-shaped
    // precondition failed identically on every retrace, so the loop kept
    // re-tracing and re-aborting instead of settling.
    //
    // The strict path already declines gracefully here (`break 'seed`); only
    // the `try_multiframe` path aborted.  Upstream never has this state:
    // `pyjitpl.py` `do_residual_or_indirect_call` residualizes the callee it
    // cannot follow, and the recursion-budget path calls `dont_trace_here` and
    // then still falls through to `do_residual_call` — the enclosing trace
    // survives either way.  `rlib/jit.py`'s `ABORT_*` set (TOO_LONG, BRIDGE,
    // BAD_LOOP, ESCAPE, FORCE_QUASIIMMUT, SEGMENTED_TRACE) has no
    // cannot-inline-this-callee reason at all.
    //
    // The variant's own doc justifies abort-over-residual for a callee whose
    // short inner loops would compile and deopt-storm — but
    // `callee_fast_path_inlinable_allowing_forward_branch` already rejects every
    // backward `goto_if_not` and every `switch`, so a `try_multiframe` callee
    // provably has no inner loop and that rationale does not reach here.
    //
    // All of these sit before the first recorded op (the `GETFIELD_GC_R`
    // below), so returning costs nothing but the inline.
    if try_multiframe || strict_seed {
        'seed: {
            // Branch-A frame shape only (mirror REC_CA): existing freevar
            // cells are admissible, while fresh cellvar allocation is not.
            // `strict_seed` already excludes such a callee, so only the
            // multiframe path reaches this.
            if !callee_code.cellvars.is_empty() {
                return Ok(None);
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
            // Stored bound methods carry their explicit receiver and callee frame,
            // so their Ref operands remain available to the resume path.
            //
            // This precondition used to abort the enclosing trace rather than
            // decline the inline, because residualizing it let loops compile that
            // then printed traceback tuples missing their OUTERMOST frame.  That
            // node is now recorded — the two bridge handler-entry arms attach the
            // catching frame's own node — so the decline joins every other
            // precondition here and returns `Ok(None)`.
            //
            // The abort was expensive out of all proportion to the inline it was
            // protecting: a callee that walks a traceback (`while tb is not None`)
            // lowers to exactly this instruction, so any handler calling such a
            // helper aborted every retrace of the enclosing loop.  The guard whose
            // bridge the retrace was building therefore never got one and deopted
            // on every delivery.
            if bound_method.is_none() {
                let liveness = crate::liveness::liveness_for(raw_callee_code);
                let has_is_none_branch = (0..callee_code.instructions.len()).any(|pc| {
                    matches!(
                        pyre_interpreter::decode_instruction_at(callee_code, pc),
                        Some((
                            pyre_interpreter::bytecode::Instruction::PopJumpIfNone { .. }
                                | pyre_interpreter::bytecode::Instruction::PopJumpIfNotNone { .. },
                            _
                        ))
                    ) && (
                        // Hazard 1 — kept operands. A branch that leaves slots
                        // on the value stack needs its guard resume to restore
                        // them, and the inline sub-walk's mirror does not model
                        // them, so the kept Ref reads NULL and
                        // `walker_branch_guard` raises
                        // BranchGuardUnrestorableKeptStackPermanent — a
                        // permanent abort that discards the enclosing loop
                        // trace.  `stack_depth_at` is the depth BEFORE the
                        // instruction and the branch pops the tested value, so
                        // `depth > 1` is exactly "a kept slot survives".  An
                        // unreachable pc has no depth and cannot fire a guard.
                        liveness.stack_depth_at(pc).is_some_and(|depth| depth > 1)
                            // Hazard 2 — the tested operand itself.  When the
                            // multiframe inline int-specializes the tested
                            // local, the mid-body guard resume cannot source
                            // that operand's Ref form from the callee register
                            // banks (`collect_callee_active_boxes` reads a
                            // stale/mismatched box), so the encoded liveness
                            // stream disagrees with the decoder and the caller
                            // frame is corrupted.  This is independent of kept
                            // depth: a statement-level `if x is None:` reads
                            // depth 1 and still carries it.
                            || callee_binds_an_unboxed_local
                    )
                });
                if has_is_none_branch {
                    if try_multiframe {
                        return Ok(None);
                    }
                    seed_break_reason = "Collapse::IsNoneBranch";
                    break 'seed;
                }
            }
            let nlocals = callee_code.varnames.len();
            let ncells = pyre_interpreter::ncells(callee_code);
            let frame_array_size = nlocals + ncells + callee_code.max_stackdepth as usize;

            let Some(callee_jitcode_index) =
                crate::state::ensure_jitcode_index(callee_code_key as *const ())
            else {
                if try_multiframe {
                    return Ok(None);
                }
                seed_break_reason = "Collapse::NoCalleeJitcode";
                break 'seed;
            };
            let (frame_reg, ec_reg) = crate::state::portal_red_regs_at(callee_jitcode_index as i32);
            if frame_reg == u16::MAX
                || ec_reg == u16::MAX
                || frame_reg as usize >= callee_regs_r.len()
                || ec_reg as usize >= callee_regs_r.len()
            {
                if try_multiframe {
                    return Ok(None);
                }
                seed_break_reason = "Collapse::NoPortalRedRegs";
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
                    return Ok(None);
                }
                seed_break_reason = "Collapse::NoSnapshotSym";
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
            let param_boxes: Vec<OpRef> = (0..seeded_locals).map(|i| callee_args[i]).collect();
            let freevar_cells: Vec<OpRef> = concrete_freevar_cells
                .iter()
                .map(|&cell| ctx.trace_ctx.const_ref(cell as i64))
                .collect();
            let callee_frame = crate::helpers::emit_new_pyframe_inline_with_params(
                ctx.trace_ctx,
                &param_boxes,
                &freevar_cells,
                nlocals,
                frame_array_size,
                nlocals + ncells,
                pycode_const,
                w_globals_obj_const,
                callee_ec,
            );

            callee_regs_r[frame_reg as usize] = callee_frame;
            // `perform_call` creates one concrete frame per MIFrame before
            // `setup_call` installs the argument boxes (pyjitpl.py:2445-2476,
            // 1862-1874).  Mirror that recording-time object.  `setup_call`
            // installs the whole box list, so seed every local the symbolic
            // frame above got from `param_boxes` — a `*args` callee's packed
            // vararg tuple is one of them, and a frame short of it publishes
            // that name as unbound to any residual the sub-walk runs.  Root
            // each freshly boxed argument immediately: `ConcreteValue::to_pyobj`
            // can allocate, and a later argument must not collect an earlier
            // one before the frame constructor takes ownership of the slice.
            let arg_roots = pyre_object::gc_roots::push_roots();
            let arg_root_base = pyre_object::gc_roots::shadow_stack_len();
            for concrete in callee_arg_concretes.iter().take(seeded_locals).copied() {
                pyre_object::gc_roots::pin_root(concrete.to_pyobj());
            }
            let concrete_args: Vec<pyre_object::PyObjectRef> = (0..seeded_locals)
                .map(|i| pyre_object::gc_roots::shadow_stack_get(arg_root_base + i))
                .collect();
            let concrete_ec = sym.concrete_execution_context();
            // Use a GC-managed frame, not `new_boxed`: the concrete pointer is
            // stamped onto the active trace's frontend op below, and
            // `MetaInterp::walk_active_trace_refs` is then its RPython-style
            // GC root through optimization.  A scope-owned tracer snapshot
            // would be freed when this function returns while the Box value
            // still exists, leaving a dangling recording-time pointer.
            let mut frame = pyre_interpreter::pyframe::FrameBox::new(
                pyre_interpreter::pyframe::PyFrame::new_for_call_with_closure_and_globals_obj(
                    w_code,
                    &concrete_args,
                    inline_consts.w_globals as pyre_object::PyObjectRef,
                    concrete_ec,
                    concrete_closure,
                    pyre_interpreter::pyframe::FrameLocalsArrayAllocation::OldGenGc,
                ),
            );
            drop(arg_roots);
            let concrete_frame_ptr = frame.as_mut_ptr();
            concrete_callee_frame = concrete_frame_ptr;
            callee_concrete_r[frame_reg as usize] =
                ConcreteValue::Ref(concrete_frame_ptr as pyre_object::PyObjectRef);
            ctx.trace_ctx.set_opref_concrete(
                callee_frame,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete_frame_ptr as usize)),
            );
            // GC-managed FrameBox::drop intentionally relinquishes only the
            // host handle; the frontend op above keeps the frame reachable.
            drop(frame);
            callee_regs_r[ec_reg as usize] = callee_ec;
            callee_concrete_r[ec_reg as usize] = ConcreteValue::Null;

            // Retain for a possible `SubLoopCalleeCallAssembler` emit.
            ca_callee_frame = callee_frame;
            ca_callee_ec = callee_ec;
            ca_nlocals = nlocals + ncells;
            ca_concrete_frame = concrete_frame_ptr;
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
    // paused caller frame on the framestack so its in-callee guards snapshot
    // both frames.  The caller's live register banks were preflighted above,
    // before seed IR; at guard-capture time the walk context is the callee's.
    let parent_frame = if try_multiframe {
        // Declined above, before any seed IR: an un-entered multiframe-inline
        // CALL that declines at its try-block catch marker is re-run whole and
        // forward, exactly as if it had never been inlined (`pyjitpl.py`), so
        // it leaves the enclosing trace alone rather than aborting it.
        precomputed_parent_frame
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
        // shape it cannot build yet (no result on the operand stack at the
        // return point, missing liveness / resume tables).  Fall back to the
        // single-frame collapse there (do NOT decline the inline — that shape is
        // served correctly today), so this never removes a working inline.
        //
        // A `TryBlockCatchMarker` decline is different: the CALL is covered by
        // the caller's exception table AND the callee has free variables, so it
        // reads cells the caller frame owns — one of which, inside a handler, is
        // the `except E as e` binding the implicit cleanup stores `None` into
        // and then clears.  Inlining reads that cell as `None`
        // (`synth/exception_as_cell_cleanup`).  Decline so the call stays
        // residual, where the post-call catch resume
        // (`GuardCaptureScope::residual_call_catch_resume`) routes the raise.
        //
        // Both declines were taken above, ahead of the seed block, so neither
        // has to discard the enclosing loop trace to avoid stranding the
        // `GETFIELD_GC_R` + `emit_new_pyframe_inline_with_params` this arm would
        // otherwise have already recorded.
        precomputed_parent_frame
    } else {
        // Single-frame collapse (resume at the CALL boundary, re-execute the
        // whole call on deopt): a nested strict callee
        // (`inline_depth >= fbw_max_multiframe_depth()`), an un-seedable
        // strict callee, or a callee neither seed served.  Sound for a pure
        // value-returning leaf (idempotent re-execute) and for a nested
        // straight-line callee (its pre-multiframe behavior).
        None
    };
    // Name the population a collapsing CALL falls into, so the
    // `PYRE_FBW_DEBUG_ABORT` corpus can rank the remaining collapse sources
    // the same way it ranks walk declines.  A collapse is not an abort and
    // discards no trace, so this counts a resume-SHAPE choice, not a failure;
    // it is read only to decide which population to retire next.
    if fbw_debug_abort_enabled() && parent_frame.is_none() {
        census_record(if !seed_break_reason.is_empty() {
            seed_break_reason
        } else if callee_frame_seeded {
            // Seeded, but `compute_inline_caller_frame` could not build the
            // caller side (`InlineCallerFrameDecline::Unavailable`).
            "Collapse::ParentUnavailable"
        } else if inline_depth >= fbw_max_multiframe_depth() {
            "Collapse::DepthCap"
        } else if !callee_code.cellvars.is_empty() {
            "Collapse::CellVars"
        } else if constructor_result.is_some() {
            "Collapse::Constructor"
        } else {
            "Collapse::Other"
        });
    }
    let callee_frame_materialized_has_resume = callee_frame_seeded && parent_frame.is_some();

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
    // The coordinate published onto the outer portal frame's `last_instr`
    // while a residual runs inside the callee.  `op.pc` indexes the snapshot
    // sym's jitcode only while the walk is the portal's own; one level down it
    // is the INTERMEDIATE callee's offset, and mapping it through the portal's
    // pc tables stamps whatever py_pc that byte happens to land on onto a frame
    // that is still paused at the CALL the outermost sub-walk entered under —
    // which is exactly what the inherited coordinate already names.
    let inherited_caller_py_pc = if ctx.fbw_mode.inline_subwalk {
        ctx.fbw_mode.inline_caller_py_pc
    } else {
        inline_caller_py_pc_from_snapshot(ctx, op.pc)
    };
    let (
        inline_outer_active_boxes,
        inline_outer_entry_py_pc,
        inline_outer_jc_index,
        inline_outer_resume_marker_jit_pc,
        inline_caller_py_pc,
    ) = if ctx.outer_active_boxes.is_empty() {
        let sym_ptr = ctx.fbw_mode.snapshot_sym;
        if sym_ptr.is_null() {
            (
                ctx.outer_active_boxes.clone(),
                ctx.entry_py_pc,
                ctx.outer_jitcode_index,
                ctx.outer_resume_marker_jit_pc,
                inherited_caller_py_pc,
            )
        } else {
            let sym = unsafe { &*sym_ptr };
            if sym.jitcode().is_null() {
                (
                    ctx.outer_active_boxes.clone(),
                    ctx.entry_py_pc,
                    ctx.outer_jitcode_index,
                    ctx.outer_resume_marker_jit_pc,
                    inherited_caller_py_pc,
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
                let (call_site_jc_index, call_site_marker, call_site_py_pc) = unsafe {
                    let jc = &*sym.jitcode();
                    let jc_index = jc.index as u32;
                    let py_pc = crate::py_coord::containing_py_pc_for_jitcode_pc(
                        &jc.payload.metadata,
                        op.pc,
                    );
                    (
                        jc_index,
                        jc.payload.resume_marker_for_jitcode_pc(op.pc),
                        py_pc as u32,
                    )
                };
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
                    false,
                    call_site_word,
                    // Keep the marker for the liveness-bank query, but key
                    // entry metadata to the raw CALL offset that produced the
                    // pre-adjustment Python coordinate.
                    op.pc as i32,
                    OuterActiveBoxesEntryTwin::Plain,
                    "call_site_capture",
                    None,
                    &[],
                    None,
                );
                (
                    boxes,
                    EntryPyPc::Jit(op.pc),
                    call_site_jc_index,
                    call_site_marker,
                    Some(call_site_py_pc),
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
            inherited_caller_py_pc,
        )
    };
    // `executioncontext.py:88 enter` — emitted here, past every decline gate,
    // so a callee that never runs leaves no half-entered chain behind.  A
    // seeded level is the only one with a frame object to enter with; an
    // unseeded (register-resident) inline has none, which is the remaining gap
    // between this chain and upstream's, where `perform_call` builds a frame
    // for every inlined call (`pyjitpl.py:2445-2476, 1862-1874`).
    let entered_ec = callee_frame_seeded && !ca_concrete_frame.is_null() && {
        let concrete_ec = unsafe { (*ca_concrete_frame).execution_context }
            as *mut pyre_interpreter::PyExecutionContext;
        if concrete_ec.is_null() {
            false
        } else {
            walker_ec_enter(
                ctx.trace_ctx,
                ca_callee_frame,
                ca_callee_ec,
                ca_concrete_frame,
                concrete_ec,
            );
            true
        }
    };
    let (callee_outcome, callee_class_of_last_exc_is_const) = {
        let mut sub_wc = WalkContext {
            callee_shadow: Some(super::CalleeLocalsShadow {
                code_ptr: raw_callee_code,
                ..Default::default()
            }),
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
                inline_caller_py_pc,
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
            vstack_reorder_saved: None,
            vstack_handler_landing_py: None,
            live_before_jit_pc: usize::MAX,
            live_after_jit_pc: usize::MAX,
            trace_ctx: ctx.trace_ctx,
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
        let _foriter_deferred =
            ForiterDeferredInlineGuard::enter(callee_code_key, foriter_deferred_admit);
        // Name the frame this sub-walk executes concretely, so each residual
        // it runs can `enter`/`leave` it on the interpreter frame chain.
        let _inline_concrete_frame = InlineConcreteFrameGuard::enter(concrete_callee_frame);
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
            {
                let shadow = sub_wc.callee_shadow.as_mut().unwrap();
                shadow.concrete_frame = concrete_callee_frame as usize;
                shadow.frame_box = sub_wc.registers_r[callee_portal_frame_reg as usize];
            }
            if !try_multiframe {
                let shadow = sub_wc.callee_shadow.as_mut().unwrap();
                shadow.fold_frame_reg = callee_portal_frame_reg;
                // The fold's premise (`setarrayitem_vable_via_metainterp`) is
                // that it writes away from an UNSEEDED portal frame — a pure
                // SSA mirror with no heap array behind it.  The seed block
                // above may have materialized a real callee `PyFrame`, whose
                // `NewArrayClear` locals array is stored into only for the
                // parameters and freevar cells; folding away the in-callee
                // STORE_FASTs would leave every other local holding the
                // zero-fill, and a frame reachable afterwards through a
                // traceback, `f_locals` or `sys._getframe` reads them as
                // unbound.  Record that here so the store handler demotes just
                // the LOCAL region to a recorded `SETARRAYITEM_GC`, which is
                // what `_opimpl_setarrayitem_vable` does for a
                // `_nonstandard_virtualizable` (`pyjitpl.py:1120`). Recording
                // those stores also emits the promote guard in
                // `vable_getfield_*` (`pyjitpl.py:1916,2582`), whose resume
                // image must include the paused caller frame
                // (`opencoder.py:819`). If this sub-walk has no caller image,
                // keep folding: publishing only the callee frame is unsound.
                shadow.frame_materialized = callee_frame_materialized_has_resume;
            }
            for i in 0..seeded_locals {
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
            // `MIFrame.registers_r` retains cell/freevar slots across a
            // may-force call just like ordinary locals.  The heapcache entry
            // installed by `emit_new_pyframe_inline_with_params` is only a
            // forwarding optimization and is invalidated by that call; seed
            // the frame-local shadow too so a later LOAD_DEREF re-reads the
            // same live cell instead of manufacturing an unstamped
            // GetarrayitemGcR result.  This callee shape has no fresh cellvars
            // (rejected above), so existing freevars begin at `nlocals`.
            for (i, &cell) in concrete_freevar_cells.iter().enumerate() {
                let slot = (callee_code.varnames.len() + i) as i64;
                let value = sub_wc.trace_ctx.const_ref(cell as i64);
                let shadow = sub_wc.callee_shadow.as_mut().unwrap();
                shadow.set_opref(slot, value);
                shadow.set_concrete(
                    callee_portal_frame_reg,
                    slot,
                    majit_ir::Value::Ref(majit_ir::GcRef(cell as usize)),
                );
            }
        }
        // Capture a depth-1 live callee before these guards drop. This is the
        // two-frame specialization of `run_blackhole_interp_to_cancel_tracing`:
        // `_copy_data_from_miframe` preserves the callee's own position and
        // live registers instead of collapsing it onto the caller frame.
        // `newframe(jitcode, greenkey)` (pyjitpl.py:2443-2445) — the callee this
        // walk is about to inline is what upstream logs there.  The pair
        // brackets the `walk` call and nothing else: every decline is already
        // behind us and every exit below reads `result`, so the sequence
        // `find_biggest_function` pairs off cannot go out of step.  The green
        // key is the callee's function-entry key, the one
        // `disable_noninlinable_function` is applied to.
        let subwalk_jd_no = crate::state::note_inline_subwalk_start(
            crate::driver::make_green_key(raw_callee_code as *const (), 0),
            sub_wc.trace_ctx.get_trace_position(),
        );
        let result = {
            // #704 root-bridge self-recursive inline: exempt this callee body
            // sub-walk's nested recursive residual from the self-recursive
            // nested-residual decline, mirroring the native `CALL_ASSEMBLER`
            // fold's `SELFREC_CA_FOLD_ACTIVE` exemption.
            let _bridge_rec_selfrec_guard = bridge_rec_root_selfrec.then(SelfRecCaFoldGuard::enter);
            walk(body.code, 0, &mut sub_wc)
        };
        if let Some(jd_no) = subwalk_jd_no {
            crate::state::note_inline_subwalk_end(jd_no, sub_wc.trace_ctx.get_trace_position());
        }
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
            if is_top_inline && !unjournaled_before_subwalk {
                crate::trace::fbw_diag::bump(crate::trace::fbw_diag::MIDBODY_LATCH);
                if fbw_has_unjournaled_effect() {
                    crate::trace::fbw_diag::bump(
                        crate::trace::fbw_diag::MIDBODY_LATCH_NEW_UNJOURNALED,
                    );
                }
            }
            // Unlike the entry carrier, this leg resumes INSIDE the rebuilt
            // callee, so a residual the callee recorded only symbolically
            // before the abort pc lies BEHIND the resume point and the
            // discarded trace was its only carrier.  `unjournaled_before_subwalk`
            // is sampled before the sub-walk and cannot see such a mark; read
            // the flags again, as the loop-header, abort-pc and branch-guard
            // legs do.
            //
            // Attempted whether or not the callee executed anything.
            // `convert_and_run_from_pyjitpl` (`blackhole.py:1799-1821`) rebuilds
            // every framestack frame at its own pc unconditionally —
            // `run_blackhole_interp_to_cancel_tracing` ends `assert False`
            // (`pyjitpl.py:2956`) — and the caller is resumed PAST its call
            // (`blackhole.py:1653-1662`), never rewound to it.  The entry
            // carrier's rewind-to-the-CALL has no upstream counterpart, so it
            // is the fallback for a callee this one cannot rebuild, not the
            // preferred leg; `fbw_set_abort_call_resume` keeps that ordering.
            // This outer gate sits OUTSIDE the reason-producing closure below,
            // so until it was instrumented its two narrowings were the only
            // ones that could keep a callee off leg 4 without saying so — the
            // census could not tell "never fired" from "not measured".
            if fbw_debug_abort_enabled() && !(is_top_inline && !fbw_has_unjournaled_effect()) {
                eprintln!(
                    "[fbw-abort-flush] gh#467 callee-rebuild NOT LATCHED ({})",
                    if !is_top_inline {
                        "inline sub-walk (depth>=2)"
                    } else {
                        "unjournaled effect pending"
                    },
                );
            }
            if is_top_inline && !fbw_has_unjournaled_effect() {
                // Each refusal names itself so the debug log can say WHICH
                // narrowing keeps a callee off this leg — the entry carrier
                // silently absorbs every one of them.
                let payload = (|| {
                    let (outer_jitcode_index, call_jitcode_pc) =
                        abort_flush_call_jitcode_coord.ok_or("no call jitcode coord")?;
                    let callee_pjc =
                        crate::state::pyjitcode_for_code(w_code).ok_or("no callee pyjitcode")?;
                    let metadata = &callee_pjc.metadata;
                    let callee_py_pc =
                        crate::py_coord::containing_py_pc_for_jitcode_pc(metadata, abort_pc)
                            as usize;
                    // Both abort kinds sit at the head of an opcode the walker
                    // could not take, behind at most that opcode's own vable
                    // spill; the marker kind is the narrower of the two.
                    let anchor_ok = portal_vable_bookkeeping_anchor(
                        metadata,
                        metadata.built_as_portal,
                        metadata.portal_frame_reg,
                        callee_perfn_descrs,
                        body.code,
                        callee_py_pc,
                        abort_pc,
                        |op_pc| {
                            crate::py_coord::containing_py_pc_for_jitcode_pc(metadata, op_pc)
                                as usize
                        },
                    );
                    if !anchor_ok {
                        if fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] gh#467 inexact anchor at abort_pc={abort_pc} \
                                 callee_py_pc={callee_py_pc} kind={abort_kind:?} \
                                 would re-run {:?}",
                                floor_segment_ops_before(metadata, body.code, abort_pc),
                            );
                        }
                        return Err("abort pc is not an exact segment anchor");
                    }
                    let raw = unsafe {
                        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                            as *const pyre_interpreter::CodeObject
                    };
                    if raw.is_null() {
                        return Err("null callee code ptr");
                    }
                    let callee_code = unsafe { &*raw };
                    if pyre_interpreter::pyframe::code_flags_make_generator(callee_code.flags) {
                        return Err("callee is a generator");
                    }
                    if !callee_code.cellvars.is_empty() || !callee_code.freevars.is_empty() {
                        return Err("callee has cellvars/freevars");
                    }
                    if !unsafe { pyre_interpreter::function_get_closure(callable) }.is_null() {
                        return Err("callee has a closure");
                    }
                    let depth = callee_pjc
                        .depth_for_jitcode_pc_pred(abort_pc)
                        .ok_or("no stack depth for the abort pc")?
                        as usize;
                    let nlocals = callee_code.varnames.len();
                    let entries = callee_pjc
                        .pcdep_for_jitcode_pc(abort_pc)
                        .ok_or("no pcdep entries for the abort pc")?;
                    let mut live_stack = Vec::with_capacity(depth);
                    for rel in 0..depth {
                        let semantic_slot = nlocals + rel;
                        let register_value =
                            crate::state::semantic_slot_color_for_ref_slot(&entries, semantic_slot)
                                .and_then(|color| sub_wc.concrete_registers_r.get(color).copied());
                        let value = register_value
                            .or_else(|| {
                                (metadata.built_as_portal && abort_kind == MidBodyAbortKind::Marker)
                                    .then(|| {
                                        callee_vable_ref_at(
                                            sub_wc.callee_shadow.as_ref(),
                                            metadata.portal_frame_reg,
                                            semantic_slot,
                                        )
                                    })
                                    .flatten()
                            })
                            .ok_or("live stack slot has no concrete value")?;
                        if !matches!(value, ConcreteValue::Ref(r) if !r.is_null()) {
                            return Err("live stack slot is not a non-null Ref");
                        }
                        live_stack.push(value);
                    }
                    if live_stack.len() != depth {
                        return Err("live stack depth mismatch");
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
                            .or_else(|| callee_arg_concretes.get(slot).copied())
                            .ok_or("live local has no concrete value")?;
                        if matches!(value, ConcreteValue::Null | ConcreteValue::Bool(_)) {
                            return Err("live local is Null/Bool");
                        }
                        *dst = Some(value);
                    }
                    Ok(MidBodyPayload {
                        abort_kind,
                        outer_jitcode_index,
                        call_jitcode_pc,
                        call_stack_len: arg_concretes.len(),
                        callee_jitcode_index: callee_pjc.jitcode.index() as u32,
                        abort_jitcode_pc: abort_pc,
                        callee_py_pc,
                        w_code: w_code as pyre_object::PyObjectRef,
                        w_globals: unsafe { pyre_interpreter::function_get_globals_obj(callable) },
                        live_locals,
                        live_stack,
                        return_value: pyre_object::PY_NULL,
                        // Attached later by `fbw_set_abort_call_resume`, which
                        // runs in the Err arm below under the entry latch's own
                        // zero-delta gate.
                        entry_fallback: None,
                    })
                })();
                match payload {
                    Ok(payload) => fbw_set_midbody_abort_resume(payload),
                    Err(reason) => {
                        if fbw_debug_abort_enabled() {
                            eprintln!(
                                "[fbw-abort-flush] gh#467 callee-rebuild NOT LATCHED at \
                                 abort_pc={abort_pc} ({reason})"
                            );
                        }
                    }
                }
            }
        }
        let class_of_last_exc_is_const = sub_wc.fbw_mode.class_of_last_exc_is_const;
        (result, class_of_last_exc_is_const)
    };
    // `executioncontext.py:91-107 leave`, in the original's `finally`
    // position: the sub-walk block above is an expression that always
    // completes, so every callee exit — return, exception, or decline —
    // arrives here before any of the early returns below.
    if entered_ec {
        let concrete_ec = unsafe { (*ca_concrete_frame).execution_context }
            as *mut pyre_interpreter::PyExecutionContext;
        // `leave(frame, w_exitvalue, got_exception)` — the caller passes true
        // only when the frame is unwinding an exception, which for an inlined
        // callee is `SubRaise` and nothing else.  A tracing decline (`Err`) or
        // a loop transition is not an exception exit: treating it as one would
        // permanently `mark_as_escaped` the caller and force a vref that never
        // needed forcing.
        let got_exception = matches!(callee_outcome, Ok((DispatchOutcome::SubRaise { .. }, _)));
        walker_ec_leave(
            ctx.trace_ctx,
            ca_callee_frame,
            ca_callee_ec,
            ca_concrete_frame,
            concrete_ec,
            got_exception,
        );
    }
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
            //
            // The kept-stack branch-guard aborts belong to the same class: they
            // refuse to COMPILE a guard whose not-taken arm the blackhole could
            // not reconstruct, which says nothing about the sub-walk having
            // committed anything.  Without a carrier their only remaining leg
            // rewinds to the OUTER frame's entry (`trace.rs`, "legacy drop
            // kept"), re-running every effect the walk already executed —
            // `threading.Thread.start` calling `_start_joinable_thread` twice.
            if matches!(
                e,
                DispatchError::AbortPermanentMarkerReached { .. }
                    | DispatchError::LoopBearingCalleeInlineUnsupported { .. }
                    | DispatchError::BranchGuardUnrestorableKeptStackPermanent { .. }
                    | DispatchError::BranchGuardKeptStackUnsupported { .. }
            ) {
                latch_abort_call_resume(
                    code,
                    op,
                    ctx,
                    is_top_inline,
                    unjournaled_before_subwalk,
                    executed_effects_before,
                    abort_flush_call_jitcode_coord,
                );
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
                latch_abort_call_resume(
                    code,
                    op,
                    ctx,
                    is_top_inline,
                    unjournaled_before_subwalk,
                    executed_effects_before,
                    abort_flush_call_jitcode_coord,
                );
                return Err(DispatchError::callee_inline_unsupported(op.pc));
            }
            // `descr_call` discards `__init__`'s result after checking it is
            // None and returns the instance instead (`check_init_returned_none`).
            // A non-None result is a TypeError the inlined body cannot raise, so
            // give the callee back to the interpreter, which re-runs the call and
            // raises the faithful message.  Latch the CALL boundary first, like
            // the invalid-`str`/`repr`-result path above: the sub-walk already
            // executed the constructor body, so a plain abort would have the
            // interpreter replay it and repeat any effect it performed.
            let (value, concrete_for_shadow) = match constructor_result {
                Some(instance) => {
                    if !matches!(concrete_for_shadow,
                        ConcreteValue::Ref(obj) if unsafe { pyre_object::is_none(obj) })
                    {
                        latch_abort_call_resume(
                            code,
                            op,
                            ctx,
                            is_top_inline,
                            unjournaled_before_subwalk,
                            executed_effects_before,
                            abort_flush_call_jitcode_coord,
                        );
                        return Err(DispatchError::callee_inline_unsupported(op.pc));
                    }
                    instance
                }
                None => (value, concrete_for_shadow),
            };
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
                // The handler this routes to is part of the trace, so once the
                // trace runs compiled it catches the exception itself and this
                // frame never surfaces an error the interpreter's
                // `handle_exception` could record a node from.  Emit the node
                // at runtime as well as applying it for the recording pass.
                let emit_runtime =
                    !record_prepend_application_traceback(ctx, exc, exc_concrete, op.pc);
                record_inline_application_traceback(
                    ctx,
                    exc,
                    exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                );
                record_top_level_application_traceback(
                    ctx,
                    exc,
                    exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                );
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
                return Err(DispatchError::callee_inline_unsupported(op.pc));
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
/// Instantiate a user-defined class inside the trace instead of leaving `P()`
/// an opaque `bh_call_fn` residual that re-enters `type_descr_call_impl`,
/// `object.__new__` and an interpreted `__init__` frame every iteration.
///
/// This is the walker counterpart of `typeobject.py descr_call`: promote the
/// class, run `__new__`, then `__init__` with the fresh instance as `self`.
/// Only the shape where `__new__` is `object`'s is emitted — that is exactly
/// the case whose allocation `object_descr_new` performs unconditionally
/// (`w_instance_new`), so the emitted `NewWithVtable` + header/`map` stores
/// reproduce it field for field.  A class that overrides `__new__`, is
/// abstract, refuses instantiation, or carries `__del__` (whose instances
/// `w_instance_new` registers on the finalizer queue) declines to the residual.
///
/// The payoff is not the removed dispatch alone: an instance built by
/// `new_with_vtable` is a virtual, so a constructor whose result never escapes
/// the loop optimizes away entirely, as it does upstream.
#[allow(clippy::too_many_arguments)]
/// Report why the instantiation emit declined, under `PYRE_FBW_INLINE_DIAG`.
fn type_call_decline(reason: &str) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if std::env::var("PYRE_FBW_INLINE_DIAG").is_ok() {
        eprintln!("[type-call-decline] {reason}");
    }
    Ok(None)
}

pub(crate) fn try_walker_inline_type_call<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst_bank: char,
    dst: usize,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || ctx.fbw_mode.inline_subwalk || dst_bank != 'r' {
        return Ok(None);
    }
    // `[callable, null_or_self, args...]`.  A method-form call (`null_or_self`
    // populated) never names a class as its callable.
    if r_args.len() < 2 || walker_concrete_ref_object(ctx, r_args[1]).is_some() {
        return Ok(None);
    }
    let Some(w_type) = walker_concrete_ref_object(ctx, r_args[0]) else {
        return Ok(None);
    };
    if !unsafe { pyre_object::is_type(w_type) } {
        return Ok(None);
    }
    // `type(x)` is the one-argument class introspection shortcut, not an
    // instantiation (`type_call_type_x_shortcut`).
    let w_metatype = pyre_interpreter::typedef::w_type();
    if std::ptr::eq(w_type, w_metatype) {
        return type_call_decline("type(x) shortcut");
    }
    // What follows is `type.__call__`.  A metaclass that overrides `__call__`
    // runs instead of it and may return anything at all, so it stays residual.
    if !std::ptr::eq(unsafe { (*w_type).w_class }, w_metatype) {
        return type_call_decline("metaclass overrides __call__");
    }
    // A version tag of 0 is a type whose dict changes are not tracked, so the
    // `__new__` / `__init__` / `__del__` lookups below cannot be pinned.
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 {
        return type_call_decline("no version tag");
    }
    if unsafe {
        pyre_object::w_type_disallows_instantiation(w_type)
            || pyre_object::w_type_is_abstract(w_type)
            || pyre_object::typeobject::w_type_get_hasuserdel(w_type)
    } {
        return type_call_decline("not instantiable, abstract, or has __del__");
    }
    // `map` is baked as a constant, so install the terminator now if the type
    // has not been asked for one yet — a class nobody has read an attribute off
    // still carries null, and baking that would hand every traced instance a
    // map the interpreter's own instances do not have.  The terminator is
    // created once and never replaced, so the constant stays valid.
    let terminator =
        unsafe { pyre_interpreter::objspace::std::mapdict::ensure_type_terminator(w_type) };
    if terminator.is_null() {
        return type_call_decline("no terminator");
    }
    let w_object = pyre_interpreter::typedef::w_object();
    if w_object.is_null() {
        return Ok(None);
    }
    // Only `object.__new__` allocates the plain `[ob_type | w_class | map |
    // storage]` instance this emit builds; any other `__new__` picks its own
    // layout (a builtin subclass) or runs arbitrary code.
    let tp_new = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_type, "__new__") };
    let obj_new = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_object, "__new__") };
    if tp_new != obj_new {
        return type_call_decline("__new__ overridden");
    }
    let tp_init = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_type, "__init__") };
    let obj_init = unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_object, "__init__") };
    let init_override = (tp_init != obj_init).then_some(tp_init).flatten();
    // `object.__new__`/`object.__init__` both reject surplus arguments when
    // neither is overridden; leave that TypeError to the interpreter.
    if init_override.is_none() && r_args.len() != 2 {
        return type_call_decline("surplus args without __init__");
    }
    let inlinable_init = match init_override {
        Some(init) => match unsafe { resolve_inlinable_callee(init) } {
            Some(resolved) => Some((init, resolved)),
            // An overridden `__init__` that is not a plain Python function
            // (a builtin, or a callable object) has no body to walk.
            None => return type_call_decline("__init__ not inlinable"),
        },
        None => None,
    };

    let mut arg_concretes = vec![ConcreteValue::Ref(w_type), ConcreteValue::Null];
    let mut callee_arg_concretes = Vec::with_capacity(r_args.len() - 1);
    for &arg in &r_args[2..] {
        let Some(concrete) = walker_concrete_ref_object(ctx, arg) else {
            return Ok(None);
        };
        arg_concretes.push(ConcreteValue::Ref(concrete));
        callee_arg_concretes.push(ConcreteValue::Ref(concrete));
    }

    // Everything below emits.  `try_walker_inline_resolved_user_call` has
    // decline paths of its own past this point, so keep a rewind point and cut
    // back to it — the orphaned instance is unreachable and `__init__` has not
    // run, so the residual re-does the whole instantiation from scratch.
    let pre_fold_pos = ctx.trace_ctx.get_trace_position();

    // Pin the class and the type-dict version the two lookups above resolved
    // against, so redefining `__new__` / `__init__` / `__del__` deopts.
    let type_const = ctx.trace_ctx.const_ref(w_type as i64);
    walker_emit_fold_guard_with_snapshot(ctx, op.pc, OpCode::GuardValue, &[r_args[0], type_const])?;
    ctx.trace_ctx
        .heap_cache_mut()
        .replace_box(r_args[0], type_const);
    walker_pin_type_version_tag(ctx, op.pc, type_const)?;

    // The walker is the executor here, so the instance the rest of this walk
    // reads has to be a real one — the same split `trace_box_int` makes between
    // the recorded allocation and the concrete object it hands back.
    let concrete_instance = pyre_object::w_instance_new(w_type);
    let terminator_const = ctx.trace_ctx.const_int(terminator as i64);
    let instance =
        crate::helpers::emit_instance_inline(ctx.trace_ctx, type_const, terminator_const);
    // Bind the emitted allocation to the object the walker actually made, and
    // record the layout its `NewWithVtable` stamps.  Without the concrete
    // binding the instance reaches a residual as a box with no value and the
    // residual declines (`[fbw-resid-decline] box_value=None`), which costs the
    // recording iteration; without the class the receiver re-guards its own
    // freshly emitted layout.
    ctx.trace_ctx.set_opref_concrete(
        instance,
        majit_ir::Value::Ref(majit_ir::GcRef(concrete_instance as usize)),
    );
    ctx.trace_ctx.heap_cache_mut().class_now_known(
        instance,
        &pyre_object::pyobject::INSTANCE_TYPE as *const _ as i64,
    );
    if std::env::var("PYRE_FBW_INLINE_DIAG").is_ok() {
        eprintln!(
            "[type-call-inline] pc={} class={} init={}",
            op.pc,
            unsafe { pyre_object::w_type_get_name(w_type) },
            inlinable_init.is_some(),
        );
    }

    let Some((init, (w_code, nparams, has_closure))) = inlinable_init else {
        write_ref_reg(
            ctx,
            op.pc,
            dst,
            instance,
            ConcreteValue::Ref(concrete_instance),
        )?;
        return Ok(Some((DispatchOutcome::Continue, op.next_pc)));
    };

    let mut callee_args = vec![instance];
    callee_args.extend_from_slice(&r_args[2..]);
    callee_arg_concretes.insert(0, ConcreteValue::Ref(concrete_instance));
    let inlined = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        init,
        r_args[0],
        w_type,
        arg_concretes,
        callee_args,
        callee_arg_concretes,
        true,
        None,
        w_code,
        nparams,
        has_closure,
        None,
        None,
        true,
        // `__init__` bodies are `self.x = ...` stores; the sub-walk folds them
        // to slot writes on the fresh instance exactly as the property-setter
        // route folds its own.
        false,
        Some((instance, ConcreteValue::Ref(concrete_instance))),
    )?;
    if inlined.is_none() {
        ctx.trace_ctx.cut_trace(pre_fold_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
    }
    Ok(inlined)
}

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
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }
    // A nested call in the override body (e.g. `return repr(self.args)`) is
    // inlined like any other: the nested callee records its own residual and
    // guard-resume snapshot on this route without aborting, and a raise from it
    // reaches the enclosing handler unchanged.
    let Some((_override_descr_refs, _, _)) = crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return Ok(None);
    };

    // A straight-line, effect-free override can be sampled before any IR is
    // emitted. If its observed result is not a string, decline to the original
    // builtin residual so the interpreter's result check raises TypeError and
    // the exception-handler loop remains traceable. More complex bodies are
    // not executed speculatively; their inlined result is guarded below.
    if let (Some(body), Some((callee_descr_refs, _, _))) = (
        crate::state::sub_jitcode_body_for_code(w_code),
        crate::state::sub_jitcode_descr_pool_for_code(w_code),
    ) {
        if body_facts.exc_override_sample_safe {
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
        None,
        w_code,
        nparams,
        has_closure,
        Some((r_args[2], concrete_receiver, w_class, version_tag)),
        None,
        true,
        true,
        None,
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

/// Machine-int digest of a `__hash__` result: a tagged immediate or an exact
/// heap `W_IntObject`; bool/long take `hash_call_normalize`'s other arms.
fn walker_machine_int_value(obj: pyre_object::PyObjectRef) -> Option<i64> {
    if obj.is_null() {
        return None;
    }
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return Some(pyre_object::tagged_int::untag_int(obj));
    }
    unsafe {
        if pyre_object::is_int(obj) && !pyre_object::is_bool(obj) && !pyre_object::is_long(obj) {
            Some(pyre_object::w_int_get_value(obj))
        } else {
            None
        }
    }
}

/// `hash(x)` over a user instance — the hash sibling of
/// [`try_walker_inline_exception_string_override`]: pin the receiver class,
/// inline the resolved `__hash__` body in place of the opaque call residual,
/// then emit `hash_call_normalize`'s digest check and `-1 -> -2` map as int
/// ops.  Declines to the residual before any IR is emitted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_hash_builtin<Sym: WalkSym>(
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
    if !pyre_interpreter::builtins::is_builtin_hash_function(concrete_callable) {
        return Ok(None);
    }
    if !unsafe { pyre_object::is_instance(concrete_receiver) } {
        return Ok(None);
    }
    let w_type = unsafe { pyre_object::w_instance_get_type(concrete_receiver) };
    if w_type.is_null() || !unsafe { pyre_object::is_type(w_type) } {
        return Ok(None);
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) };
    if version_tag == 0 {
        return Ok(None);
    }
    let Some(method) =
        (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_type, "__hash__") })
    else {
        return Ok(None);
    };
    // `__hash__ = None` raises in the residual; a non-Python `__hash__`
    // has no body to walk.
    if unsafe { pyre_object::is_none(method) } {
        return Ok(None);
    }
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        return Ok(None);
    };
    if nparams != 1 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }
    if crate::state::sub_jitcode_body_for_code(w_code).is_none()
        || crate::state::sub_jitcode_descr_pool_for_code(w_code).is_none()
    {
        return Ok(None);
    }
    // No pre-sampling: `hash_w` calls `__hash__` exactly once, so the digest
    // is checked after the single authoritative sub-walk run instead.
    let effect_free = body_facts.exc_override_sample_safe;

    let arg_concretes = vec![
        ConcreteValue::Ref(concrete_callable),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_receiver),
    ];
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
        None,
        w_code,
        nparams,
        has_closure,
        Some((r_args[2], concrete_receiver, w_type, version_tag)),
        None,
        true,
        false,
        None,
    )?
    else {
        return Ok(None);
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r[dst];
        let concrete_result = walker_concrete_ref_object(ctx, result);
        let live = concrete_result.and_then(walker_machine_int_value);
        // The inline unbox is guard-free only against a known-class or
        // trace-built box; a live post-body guard on a side-effecting body
        // would re-run its effects on failure, so those shapes — and every
        // bool/long digest — take the fallible normalize residual instead.
        let inline_unbox = live.is_some()
            && (effect_free
                || ctx.trace_ctx.heap_cache().is_class_known(result)
                || ctx.trace_ctx.heap_cache().is_unescaped(result));
        let (norm, live_norm) = if inline_unbox {
            let live = live.unwrap();
            let int_type_addr = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
            let raw = walker_unbox_int(ctx, op.pc, result, int_type_addr)?;
            // `hash_call_normalize`'s `-1 -> -2` map as `raw - (raw == -1)`.
            let neg1 = ctx.trace_ctx.const_int(-1);
            let is_neg1 = ctx.trace_ctx.record_op(OpCode::IntEq, &[raw, neg1]);
            ctx.trace_ctx
                .set_opref_concrete(is_neg1, majit_ir::Value::Int((live == -1) as i64));
            let norm = ctx.trace_ctx.record_op(OpCode::IntSub, &[raw, is_neg1]);
            let live_norm = if live == -1 { -2 } else { live };
            ctx.trace_ctx
                .set_opref_concrete(norm, majit_ir::Value::Int(live_norm));
            (norm, live_norm)
        } else {
            let Some(concrete_result) = concrete_result else {
                return Err(DispatchError::LoopBearingCalleeInlineUnsupported { pc: op.pc });
            };
            let raw = crate::helpers::emit_trace_call_int_typed(
                ctx.trace_ctx,
                crate::helpers::jit_hash_normalize_digest as *const (),
                &[result],
                &[majit_ir::Type::Ref],
            );
            match pyre_interpreter::builtins::normalize_hash_digest(concrete_result) {
                Ok(live_norm) => {
                    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNoException, &[])?;
                    ctx.trace_ctx
                        .set_opref_concrete(raw, majit_ir::Value::Int(live_norm));
                    (raw, live_norm)
                }
                // A raising digest completes the recording the way the
                // generic raising residual does: publish the exception,
                // pin it with GuardException, surface SubRaise.
                Err(mut err) => {
                    let exc = err.to_exc_object();
                    fbw_count_executed_residual(true, true);
                    ctx.last_exc_value_concrete = ConcreteValue::Ref(exc);
                    ctx.fbw_mode.class_of_last_exc_is_const = false;
                    majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc as i64));
                    walker_record_guard_exception(ctx, op.pc);
                    let exc_box = ctx
                        .last_exc_value
                        .expect("guard_exception seeds last_exc_value");
                    return Ok(Some((
                        DispatchOutcome::SubRaise {
                            exc: exc_box,
                            exc_concrete: ConcreteValue::Ref(exc),
                        },
                        op.next_pc,
                    )));
                }
            }
        };
        let boxed = walker_box_int(ctx, op.pc, norm, live_norm)?;
        let live_ptr = pyre_object::w_int_new(live_norm) as i64;
        ctx.trace_ctx
            .set_opref_concrete(boxed, box_int_concrete(live_norm, live_ptr));
        write_residual_call_result_to_dst(ctx, op.pc, dst, 'r', boxed)?;
    }
    Ok(Some(inlined))
}

/// Inline a `property` getter read (`obj.value`) after the plain-attribute
/// mapdict fold declines because the attribute is a data descriptor.  PyPy
/// traces *through* `space.getattr` → `property.__get__` →
/// `space.call_function(fget, obj)`, inlining the pure-Python getter body; pyre
/// leaves LOAD_ATTR an opaque `getattr` `CALL_MAY_FORCE` residual whose fget
/// runs as a fresh interpreter frame every iteration.  This fold reproduces
/// PyPy's shape: pin the receiver class + version tag (so the property lookup
/// const-folds), then enter the resolved-callee inline plumbing with the
/// receiver as `self` so the getter body (commonly `return self._value`) folds
/// to a guarded slot read inside the trace.
///
/// Restricted to a straight-line, nested-call-free getter: the inline then
/// stays on the leaf sub-walk path and never reaches the loop/`CALL_ASSEMBLER`
/// route (the only consumer of this LOAD_ATTR residual's non-call `r_args`).
/// Top full-body frame only, for the resume-doubling reason
/// [`try_walker_specialize_load_bound_method_attr`] documents.  Every other
/// shape declines to the residual (SAFE — no acceleration, unchanged
/// semantics).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_property_get<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, fget)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::property_get_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(fget) }) else {
        return Ok(None);
    };
    if nparams != 1 {
        return Ok(None);
    }
    // Leaf-only getter body: a branch or a nested Python call would drive the
    // sub-walk into plumbing that consumes the non-call `r_args`; decline those
    // to the residual instead.
    let Some(body) = crate::state::sub_jitcode_body_for_code(w_code) else {
        return Ok(None);
    };
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }
    let Some((getter_descr_refs, _, _)) = crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return Ok(None);
    };
    if body_facts.exc_override_has_nested_call {
        return Ok(None);
    }

    // `[fget, <self-placeholder>, obj]`: the method-form call header the inline
    // plumbing expects (`callable`, unused self slot, then the receiver).
    let arg_concretes = vec![
        ConcreteValue::Ref(fget),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_obj),
    ];
    let fget_const = ctx.trace_ctx.const_ref(fget as i64);
    try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        fget_const,
        r_args,
        call_descr,
        'r',
        dst,
        fget,
        fget_const,
        fget,
        arg_concretes,
        vec![obj],
        vec![ConcreteValue::Ref(concrete_obj)],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((obj, concrete_obj, w_type, version_tag)),
        None,
        // LOAD_ATTR is not a CALL either, but this entry is left admitted
        // as it was: only the subscript one below has a witness.
        true,
        // Getter bodies commonly read `self._slot` — a LOAD_ATTR the method-form
        // support gate would otherwise reject; the sub-walk folds it to a slot
        // read (same allowance the exception `__str__`/`__repr__` override uses).
        false,
        None,
    )
}

/// Inline a `property` setter store (`obj.value = x`) after the plain-attribute
/// mapdict store fold declines because the attribute is a data descriptor — the
/// setter twin of [`try_walker_inline_property_get`].  Pin the receiver class +
/// version tag (so the property lookup const-folds), then inline `fset(obj, x)`
/// with the receiver as `self`; the setter body (commonly `self._value =
/// value`) folds to a guarded slot store instead of the opaque `setattr`
/// residual.  Same leaf-only / top-frame restrictions and residual fall-through
/// as the getter fold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_property_set<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    obj: OpRef,
    value: OpRef,
    w_code_ptr: usize,
    name_idx: usize,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    let Some(concrete_obj) = walker_concrete_ref_object(ctx, obj) else {
        return Ok(None);
    };
    let Some(concrete_value) = walker_concrete_ref_object(ctx, value) else {
        return Ok(None);
    };
    let Some(name) = walker_load_name_from_code(w_code_ptr, name_idx) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, fset)) = (unsafe {
        pyre_interpreter::objspace::std::mapdict::property_set_fast_path(concrete_obj, &name)
    }) else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(fset) }) else {
        return Ok(None);
    };
    if nparams != 2 {
        return Ok(None);
    }
    let Some(body) = crate::state::sub_jitcode_body_for_code(w_code) else {
        return Ok(None);
    };
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if !body_facts.exc_override_straight_line {
        return Ok(None);
    }
    let Some((setter_descr_refs, _, _)) = crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return Ok(None);
    };
    if body_facts.exc_override_has_nested_call {
        return Ok(None);
    }

    // `[fset, <self-placeholder>, obj, value]`: the method-form call header the
    // inline plumbing expects, then the two positional args (`self`, `value`).
    let arg_concretes = vec![
        ConcreteValue::Ref(fset),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_value),
    ];
    let fset_const = ctx.trace_ctx.const_ref(fset as i64);
    try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        fset_const,
        r_args,
        call_descr,
        dst_bank,
        dst,
        fset,
        fset_const,
        fset,
        arg_concretes,
        vec![obj, value],
        vec![
            ConcreteValue::Ref(concrete_obj),
            ConcreteValue::Ref(concrete_value),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((obj, concrete_obj, w_type, version_tag)),
        None,
        // STORE_ATTR, same standing as the getter above.
        true,
        false,
        None,
    )
}

/// Inline `obj[key]` into the receiver type's Python `__getitem__`.
///
/// `descroperation.py:356-381 DescrOperation.getitem` resolves `__getitem__`
/// on the receiver's type and calls it; PyPy traces through that call and
/// inlines the body.  pyre lowers BINARY_SUBSCR to one `binary_op` residual,
/// and `try_walker_specialize_subscr` only recognizes the canonical tuple and
/// exact-`list[int]` storage shapes — so a user `__getitem__` never reaches the
/// inline plumbing at all and runs as a fresh interpreter frame behind a
/// `CALL_MAY_FORCE` on every iteration.  This route closes that gap: pin the
/// receiver class + version tag so the MRO lookup const-folds, then enter the
/// resolved-callee inline with the receiver as `self`.
///
/// The gate is `getitem_fast_path`, which admits exactly the receivers whose
/// subscript the type's own MRO owns — a user instance, or a builtin sequence
/// layout that overrides `__getitem__`.  A callee owning a loop header is
/// declined for the reason [`callee_body_owns_loop_header`] documents.  Unlike
/// the property folds this admits a branching, raising body: `__getitem__`
/// raising `IndexError` at the end of a sequence is the shape worth inlining,
/// not an edge case.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_inline_subscr_getitem<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor || dst_bank != 'r' || ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    // `binary_op_fn(obj, key, tag)` — the two Ref operands are the whole
    // subscript; the operator tag rides the Int list.
    let [obj, key] = r_args else {
        return Ok(None);
    };
    let (obj, key) = (*obj, *key);
    let (Some(concrete_obj), Some(concrete_key)) = (
        walker_concrete_ref_object(ctx, obj),
        walker_concrete_ref_object(ctx, key),
    ) else {
        return Ok(None);
    };
    let Some((w_type, version_tag, w_getitem)) =
        (unsafe { pyre_interpreter::baseobjspace::getitem_fast_path(concrete_obj) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_getitem) })
    else {
        return Ok(None);
    };
    // `__getitem__(self, key)` — any other arity is a shape
    // `get_and_call_function` would reject before the body runs.
    if nparams != 2 {
        return Ok(None);
    }
    // Decided once per callee on its jitcode payload; `None` means no body or
    // descr pool, which this route declines on either way.
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if body_facts.owns_loop_header {
        return Ok(None);
    }

    // `[__getitem__, <self-placeholder>, obj, key]`: the method-form call
    // header the inline plumbing expects, then the two positional args.
    let arg_concretes = vec![
        ConcreteValue::Ref(w_getitem),
        ConcreteValue::Null,
        ConcreteValue::Ref(concrete_obj),
        ConcreteValue::Ref(concrete_key),
    ];
    let getitem_const = ctx.trace_ctx.const_ref(w_getitem as i64);
    try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        w_getitem,
        getitem_const,
        w_getitem,
        arg_concretes,
        vec![obj, key],
        vec![
            ConcreteValue::Ref(concrete_obj),
            ConcreteValue::Ref(concrete_key),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((obj, concrete_obj, w_type, version_tag)),
        None,
        // `obj[key]` enters from BINARY_OP, which the abort rewind cannot
        // name.  This is the entry the retired `arg_class_guard.is_none()`
        // proxy admitted by mistake.
        false,
        false,
        None,
    )
}

/// Inline the generic sequence cursor's `__getitem__(seq, index)` step.
///
/// The cursor update is deliberately emitted and applied only after the
/// callee returns an item.  A guard in the callee therefore resumes at this
/// FOR_ITER with the same index, while a recording-time exhaustion raise can
/// discard the inline and let the existing residual produce the NULL result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_walker_specialize_seqiter_getitem_next<Sym: WalkSym>(
    ctx: &mut WalkContext<'_, '_, Sym>,
    op: &DecodedOp,
    code: &[u8],
    funcptr: OpRef,
    r_args: &[OpRef],
    call_descr: &dyn majit_ir::descr::CallDescr,
    dst: usize,
    dst_bank: char,
) -> Result<Option<(DispatchOutcome, usize)>, DispatchError> {
    if !ctx.is_authoritative_executor {
        return Ok(None);
    }
    if dst_bank != 'r' {
        return Ok(None);
    }
    if ctx.fbw_mode.inline_subwalk {
        return Ok(None);
    }
    if r_args.len() != 1 {
        return Ok(None);
    }

    let iter_op = r_args[0];
    let Some(iter_obj) = walker_concrete_ref_object(ctx, iter_op) else {
        return Ok(None);
    };
    let (seq, index) = unsafe {
        if (*iter_obj).ob_type != &pyre_object::iterobject::SEQ_ITER_TYPE {
            return Ok(None);
        }
        let iter = iter_obj as *mut pyre_object::iterobject::W_SeqIterObject;
        ((*iter).seq, (*iter).index)
    };
    if seq.is_null() {
        return Ok(None);
    }
    if unsafe {
        pyre_object::is_list(seq)
            || pyre_object::is_tuple(seq)
            || pyre_object::is_str(seq)
            || pyre_object::bytesobject::is_bytes_like(seq)
            || pyre_object::interp_array::is_array(seq)
            || pyre_object::is_generic_alias(seq)
    } {
        return Ok(None);
    }

    let Some((w_type, version_tag, w_getitem)) =
        (unsafe { pyre_interpreter::baseobjspace::getitem_fast_path(seq) })
    else {
        return Ok(None);
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(w_getitem) })
    else {
        return Ok(None);
    };
    if nparams != 2 {
        return Ok(None);
    }
    let Some(body_facts) = sub_jitcode_body_facts_for_code(w_code) else {
        return Ok(None);
    };
    if body_facts.owns_loop_header {
        return Ok(None);
    }
    let Some(body) = crate::state::sub_jitcode_body_for_code(w_code) else {
        return Ok(None);
    };
    let Some((callee_descr_refs, _, _)) = crate::state::sub_jitcode_descr_pool_for_code(w_code)
    else {
        return Ok(None);
    };
    let exact_numeric_args = [
        ExactNumericArg::default(),
        ExactNumericArg {
            numeric: true,
            plain_int: true,
        },
    ];
    let replay_safety = fbw_callee_body_replay_safety(
        body.code,
        &exact_numeric_args,
        body.num_regs_i,
        body.constants_i,
        body.num_regs_r,
        body.constants_r,
        callee_descr_refs,
        false,
    );
    // `DeferredCall` is admitted alongside `Clean`, and the terminating `raise`
    // is what makes that necessary: `RaiseVarargs` classifies as a deferred
    // residual, so a cursor body that ends on one would otherwise never be
    // served — which is why this route passes `entry_is_call_boundary: true`
    // below.  The deferred promise holds here: a residual the lever cannot
    // inline aborts before executing and denies the callee.
    //
    // What the shared FOR_ITER gate withholds admission from is an entry
    // reached from an operator opcode, because those opcodes POP their
    // operands: a rewind that re-executes `BINARY_OP` or `COMPARE_OP` needs
    // operands the flush cannot re-materialize and resumes one short.
    // `FOR_ITER` only PEEKS, and its single operand is the iterator the walk
    // already holds, so re-executing it needs nothing the stack lost.  The
    // cursor bump below runs only after the callee returns, so the re-executed
    // step reads the same index.
    if !matches!(
        replay_safety,
        CalleeReplaySafety::Clean | CalleeReplaySafety::DeferredCall
    ) {
        return Ok(None);
    }
    if fbw_foriter_deferred_call_denied(w_code as usize) {
        return Ok(None);
    }

    let body_coord = fbw_foriter_body_from_op_pc(ctx, op.pc)
        .unwrap_or_else(|| InflightForiterBody::Py(ctx.entry_py_pc() as usize + 1));
    fbw_foriter_inflight_mark_attempt(body_coord);
    let pre_emit_pos = ctx.trace_ctx.get_trace_position();

    let seq_iter_type_addr = &pyre_object::iterobject::SEQ_ITER_TYPE as *const _ as i64;
    if !iter_op.is_constant() && !ctx.trace_ctx.heap_cache().is_class_known(iter_op) {
        let type_const = ctx.trace_ctx.const_int(seq_iter_type_addr);
        ctx.trace_ctx
            .record_guard(OpCode::GuardClass, &[iter_op, type_const], 0);
        walker_capture_snapshot_for_last_guard(ctx, op.pc)?;
    }
    ctx.trace_ctx
        .heap_cache_mut()
        .class_now_known(iter_op, seq_iter_type_addr);

    let seq_descr = crate::descr::seq_iter_seq_descr();
    let seq_op = crate::state::opimpl_getfield_gc_r(ctx.trace_ctx, iter_op, seq_descr);
    walker_emit_guard_with_snapshot(ctx, op.pc, OpCode::GuardNonnull, &[seq_op])?;
    ctx.trace_ctx
        .set_opref_concrete(seq_op, Value::Ref(majit_ir::GcRef(seq as usize)));

    let index_descr = crate::descr::seq_iter_index_descr();
    let index_op = crate::state::opimpl_getfield_gc_i(ctx.trace_ctx, iter_op, index_descr.clone());
    ctx.trace_ctx
        .set_opref_concrete(index_op, Value::Int(index));
    let boxed_index = crate::state::wrapint(ctx.trace_ctx, index_op);
    let concrete_boxed_index = pyre_object::w_int_new(index);
    ctx.trace_ctx.set_opref_concrete(
        boxed_index,
        Value::Ref(majit_ir::GcRef(concrete_boxed_index as usize)),
    );

    let _roots = pyre_object::gc_roots::push_roots();
    let iter_root = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(iter_obj);
    let boxed_index_root = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(concrete_boxed_index);
    let concrete_boxed_index = pyre_object::gc_roots::shadow_stack_get(boxed_index_root);
    let getitem_const = ctx.trace_ctx.const_ref(w_getitem as i64);

    let inline = try_walker_inline_resolved_user_call(
        ctx,
        op,
        code,
        funcptr,
        r_args,
        call_descr,
        dst_bank,
        dst,
        w_getitem,
        getitem_const,
        w_getitem,
        vec![
            ConcreteValue::Ref(w_getitem),
            ConcreteValue::Null,
            ConcreteValue::Ref(seq),
            ConcreteValue::Ref(concrete_boxed_index),
        ],
        vec![seq_op, boxed_index],
        vec![
            ConcreteValue::Ref(seq),
            ConcreteValue::Ref(concrete_boxed_index),
        ],
        true,
        None,
        w_code,
        nparams,
        has_closure,
        Some((seq_op, seq, w_type, version_tag)),
        None,
        // FOR_ITER, not a CALL, but a peeking entry the rewind can re-execute —
        // the replay-safety note above states why that is what the gate asks.
        true,
        false,
        None,
    );
    if !matches!(inline, Ok(Some((DispatchOutcome::Continue, _)))) {
        ctx.trace_ctx.cut_trace_with_snapshots(pre_emit_pos);
        ctx.trace_ctx.heap_cache_mut().reset();
        return Ok(None);
    }

    let one = ctx.trace_ctx.const_int(1);
    let next_index = ctx.trace_ctx.record_op(OpCode::IntAdd, &[index_op, one]);
    ctx.trace_ctx
        .set_opref_concrete(next_index, Value::Int(index.wrapping_add(1)));
    ctx.trace_ctx.record_op_with_descr(
        OpCode::SetfieldGc,
        &[iter_op, next_index],
        index_descr.clone(),
    );
    ctx.trace_ctx
        .heapcache_setfield_cached(iter_op, index_descr.index(), next_index);
    let iter_ptr = pyre_object::gc_roots::shadow_stack_get(iter_root)
        as *mut pyre_object::iterobject::W_SeqIterObject;
    unsafe {
        (*iter_ptr).index += 1;
    }

    let item_op = ctx.registers_r[dst];
    let Some(concrete_item) = walker_concrete_ref_object(ctx, item_op) else {
        return Err(DispatchError::callee_inline_unsupported(op.pc));
    };
    fbw_foriter_inflight_capture(concrete_item, body_coord);
    ctx.vstack_last_ref = item_op;

    Ok(Some((DispatchOutcome::Continue, op.next_pc)))
}

/// Forward dunder selected by `try_dispatch_binary_special` for a non-inplace
/// BINARY_OP. In-place operators have a distinct `__i*__` then binary fallback
/// protocol and therefore stay on the generic path until that protocol is
/// ported as a unit.
pub(super) fn user_binop_forward_dunder(
    op: pyre_interpreter::bytecode::BinaryOperator,
) -> Option<&'static str> {
    use pyre_interpreter::bytecode::BinaryOperator;

    match op {
        BinaryOperator::Add => Some("__add__"),
        BinaryOperator::And => Some("__and__"),
        BinaryOperator::FloorDivide => Some("__floordiv__"),
        BinaryOperator::Lshift => Some("__lshift__"),
        BinaryOperator::MatrixMultiply => Some("__matmul__"),
        BinaryOperator::Multiply => Some("__mul__"),
        BinaryOperator::Or => Some("__or__"),
        BinaryOperator::Power => Some("__pow__"),
        BinaryOperator::Remainder => Some("__mod__"),
        BinaryOperator::Rshift => Some("__rshift__"),
        BinaryOperator::Subtract => Some("__sub__"),
        BinaryOperator::TrueDivide => Some("__truediv__"),
        BinaryOperator::Xor => Some("__xor__"),
        BinaryOperator::Subscr
        | BinaryOperator::InplaceAdd
        | BinaryOperator::InplaceAnd
        | BinaryOperator::InplaceFloorDivide
        | BinaryOperator::InplaceLshift
        | BinaryOperator::InplaceMatrixMultiply
        | BinaryOperator::InplaceMultiply
        | BinaryOperator::InplaceOr
        | BinaryOperator::InplacePower
        | BinaryOperator::InplaceRemainder
        | BinaryOperator::InplaceRshift
        | BinaryOperator::InplaceSubtract
        | BinaryOperator::InplaceTrueDivide
        | BinaryOperator::InplaceXor => None,
    }
}

/// Inline a plain Python forward arithmetic dunder after the exact numeric
/// BINARY_OP specializations decline. The receiver class and its version tag
/// pin the descriptor lookup, matching `try_dispatch_binary_special`'s
/// forward arm. A proper-subclass rhs still declines below so reflected-method
/// priority is preserved; a traced `NotImplemented` result guards and deopts
/// to the generic dispatcher.
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

    // Name every bail, as the CallFn inline path does: a BINARY_OP that stays
    // residual costs a full interpreter dispatch per execution, and without a
    // reason line the only observable is the runtime.
    macro_rules! decline {
        ($why:expr) => {{
            if fbw_inline_diag_enabled() {
                eprintln!("[binop-inline-decline] pc={} why={}", op.pc, $why);
            }
            return Ok(None);
        }};
    }

    let Some(op_kind) = pyre_interpreter::runtime_ops::binary_op_from_tag(op_tag) else {
        decline!(format_args!("unknown binary op tag {op_tag}"));
    };
    let Some(dunder) = user_binop_forward_dunder(op_kind) else {
        decline!(format_args!("no forward dunder for {op_kind:?}"));
    };

    let lhs = r_args[0];
    let rhs = r_args[1];
    let Some(concrete_lhs) = walker_concrete_ref_object(ctx, lhs) else {
        decline!("lhs has no concrete ref");
    };
    let Some(concrete_rhs) = walker_concrete_ref_object(ctx, rhs) else {
        decline!("rhs has no concrete ref");
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
        decline!("tagged immediate operand");
    }

    let w_class = unsafe { (*concrete_lhs).w_class };
    if w_class.is_null() || !unsafe { pyre_object::is_type(w_class) } {
        decline!("lhs w_class is not a type");
    }
    let version_tag = unsafe { pyre_object::typeobject::w_type_get_version_tag(w_class) };
    if version_tag == 0 {
        decline!(format_args!("lhs class {} has no version tag", unsafe {
            pyre_object::typeobject::w_type_get_name(w_class)
        }));
    }

    let Some(w_typ_r) = pyre_interpreter::typedef::r#type(concrete_rhs) else {
        decline!("rhs has no type");
    };
    if !std::ptr::eq(w_class, w_typ_r.as_ptr())
        && unsafe { pyre_object::typeobject::w_type_issubtype(w_typ_r.as_ptr(), w_class) }
    {
        decline!("rhs is a proper subclass; its reflected dunder has priority");
    }

    let Some(method) = (unsafe { pyre_interpreter::baseobjspace::lookup_in_type(w_class, dunder) })
    else {
        decline!(format_args!("{} has no {dunder}", unsafe {
            pyre_object::typeobject::w_type_get_name(w_class)
        }));
    };
    let Some((w_code, nparams, has_closure)) = (unsafe { resolve_inlinable_callee(method) }) else {
        decline!(format_args!(
            "{}.{dunder} is not inlinable Python code",
            unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
        ));
    };
    if nparams != 2 {
        decline!(format_args!("{}.{dunder} takes {nparams} params", unsafe {
            pyre_object::typeobject::w_type_get_name(w_class)
        }));
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
        None,
        w_code,
        nparams,
        has_closure,
        Some((lhs, concrete_lhs, w_class, version_tag)),
        Some((rhs, concrete_rhs, w_typ_r.as_ptr())),
        false,
        false,
        None,
    )?
    else {
        decline!(format_args!(
            "callee inline of {}.{dunder} declined",
            unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
        ));
    };

    if matches!(inlined.0, DispatchOutcome::Continue) {
        let result = ctx.registers_r[dst];
        if matches!(
            concrete_from_recorded_opref(ctx, result),
            ConcreteValue::Ref(obj)
                if std::ptr::eq(obj, pyre_object::special::w_not_implemented())
        ) {
            if fbw_inline_diag_enabled() {
                eprintln!(
                    "[binop-inline-abort] pc={} why={}.{dunder} returned NotImplemented",
                    op.pc,
                    unsafe { pyre_object::typeobject::w_type_get_name(w_class) }
                );
            }
            return Err(DispatchError::callee_inline_unsupported(op.pc));
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
    if !std::ptr::eq(w_class, w_typ_r.as_ptr())
        && unsafe { pyre_object::typeobject::w_type_issubtype(w_typ_r.as_ptr(), w_class) }
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
        None,
        w_code,
        nparams,
        has_closure,
        Some((lhs, concrete_lhs, w_class, version_tag)),
        Some((rhs, concrete_rhs, w_typ_r.as_ptr())),
        false,
        false,
        None,
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
            return Err(DispatchError::callee_inline_unsupported(op.pc));
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
        if let ConcreteValue::Int(value) = concrete {
            // RPython `MIFrame.setup_call` passes the original BoxInt into
            // the callee, and that box retains its recording-time `.value`.
            // Pyre's register-local concrete shadow is separate from OpRef,
            // so mirror it onto the shared OpRef as well: canonical helper
            // bodies execute field/arithmetic ops through
            // `TraceCtx::box_value`, just as RPython's executor calls
            // `box.getint()`.
            ctx.trace_ctx
                .try_set_opref_concrete(int_args[i], majit_ir::Value::Int(*value));
        }
    }
    for (i, concrete) in ref_arg_concretes.iter().enumerate() {
        callee_concrete_r[i] = *concrete;
        if let ConcreteValue::Ref(value) = concrete
            && !value.is_null()
        {
            // Same `setup_call` Box.value parity for RefFrontendOp.  Without
            // this, a canonical sub-jitcode sees the pointer only in its
            // side shadow while `getfield_gc_*` asks the OpRef for the live
            // object, losing concrete length/capacity reads and aborting a
            // data-dependent branch after the helper already mutated state.
            ctx.trace_ctx.try_set_opref_concrete(
                ref_args[i],
                majit_ir::Value::Ref(majit_ir::GcRef(*value as usize)),
            );
        }
    }

    let ((callee_outcome, _callee_end_pc), callee_class_of_last_exc_is_const) = {
        let mut sub_wc = WalkContext {
            callee_shadow: None,
            inline_callee_consts: None,
            // `op_pc` in this context belongs to the callee JitCode.  Never
            // project it through the root Python JitCode's pc tables: helper
            // MIFrames are distinct in RPython, while pyre's blackhole cannot
            // enter helper jitcodes and therefore collapses their guards to
            // the carried outer Python boundary.
            fbw_mode: FbwWalkMode {
                inline_subwalk: true,
                // A canonical sub-jitcode body has no blackhole entry point of
                // its own — the fact the `op_pc` comment above states — so the
                // whole descent is transparent to the Python MIFrame stack, no
                // matter whether the caller happens to be another sub-walk.
                // Left inherited, a root-level descent walked with
                // `inline_subwalk` set and the framestack empty, which sends
                // `latch_abort_blackhole` into its multi-frame arm: that arm
                // declines on the empty framestack, so the abort latched no
                // image and fell back to entry replay — the outcome the
                // transparent-helper exclusions at the abort-coordinate claim
                // and the post-step trace-limit check exist to prevent.
                transparent_helper_subwalk: true,
                ..ctx.fbw_mode
            },
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
            vstack_reorder_saved: None,
            vstack_handler_landing_py: None,
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

    let callee_result =
        run_sub_jitcode_walk(ctx, op.pc, &sub_body, &[], &[], &args, &arg_concretes, &[]);
    let callee_outcome = callee_result?;

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
                // The handler this routes to is part of the trace, so once the
                // trace runs compiled it catches the exception itself and this
                // frame never surfaces an error the interpreter's
                // `handle_exception` could record a node from.  Emit the node
                // at runtime as well as applying it for the recording pass.
                let emit_runtime =
                    !record_prepend_application_traceback(ctx, exc, exc_concrete, op.pc);
                record_inline_application_traceback(
                    ctx,
                    exc,
                    exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                );
                record_top_level_application_traceback(
                    ctx,
                    exc,
                    exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                );
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

    // RPython `rclass.py`/`rbuiltin.py` allocation lowering: entering the
    // canonical `w_int_new` helper with one signed argument records the
    // `NEW_WITH_VTABLE + SETFIELD_GC` box directly.  The LLBC fallback body
    // spells the same operation as a zero-argument allocation residual plus
    // header/payload stores; its generic allocator has no host fnaddr (it is
    // an lltype allocation opcode, not a callable), so descending that legacy
    // spelling would abort at the symbolic funcptr.  Treat the translated
    // helper call as the allocation intrinsic at this frame boundary, exactly
    // where RPython's rtyper has already replaced `malloc(W_IntObject)`.
    let is_w_int_new = crate::jitcode_runtime::get_jitcode_ref_by_index(sub_index)
        .is_some_and(|jc| jc.name == "w_int_new" && jc.code.as_ptr() == sub_body.code.as_ptr());
    if is_w_int_new
        && dst_bank == 'r'
        && int_args.len() == 1
        && ref_args.is_empty()
        && let Some(ConcreteValue::Int(value)) = int_arg_concretes.first().copied()
    {
        let boxed_ptr = pyre_object::w_int_new(value) as i64;
        let boxed = walker_box_int(ctx, op.pc, int_args[0], value)?;
        let boxed_concrete = box_int_concrete(value, boxed_ptr);
        ctx.trace_ctx.set_opref_concrete(boxed, boxed_concrete);
        let dst = code[op.pc + 1 + 2 + int_width + ref_width] as usize;
        let ConcreteValue::Ref(boxed_shadow) = concrete_from_recorded_opref(ctx, boxed) else {
            unreachable!("box_int_concrete must produce a Ref concrete")
        };
        write_ref_reg(ctx, op.pc, dst, boxed, ConcreteValue::Ref(boxed_shadow))?;
        return Ok((DispatchOutcome::Continue, op.next_pc));
    }

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
                // The handler this routes to is part of the trace, so once the
                // trace runs compiled it catches the exception itself and this
                // frame never surfaces an error the interpreter's
                // `handle_exception` could record a node from.  Emit the node
                // at runtime as well as applying it for the recording pass.
                let emit_runtime =
                    !record_prepend_application_traceback(ctx, exc, exc_concrete, op.pc);
                record_inline_application_traceback(
                    ctx,
                    exc,
                    exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                );
                record_top_level_application_traceback(
                    ctx,
                    exc,
                    exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                );
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

    let callee_result = run_sub_jitcode_walk(
        ctx,
        op.pc,
        &sub_body,
        &int_args,
        &int_arg_concretes,
        &ref_args,
        &ref_arg_concretes,
        &float_args,
    );
    let callee_outcome = callee_result?;

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
                // The handler this routes to is part of the trace, so once the
                // trace runs compiled it catches the exception itself and this
                // frame never surfaces an error the interpreter's
                // `handle_exception` could record a node from.  Emit the node
                // at runtime as well as applying it for the recording pass.
                let emit_runtime =
                    !record_prepend_application_traceback(ctx, exc, exc_concrete, op.pc);
                record_inline_application_traceback(
                    ctx,
                    exc,
                    exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                );
                record_top_level_application_traceback(
                    ctx,
                    exc,
                    exc_concrete,
                    op.pc,
                    true,
                    emit_runtime,
                );
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
