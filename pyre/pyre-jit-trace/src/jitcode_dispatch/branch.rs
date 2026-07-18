//! Branch-target resolution, trampoline decoding, and branch-arm
//! resume liveness / kept-stack hazard analysis.
//!
//! Helpers for the `goto_if_*` guard arms: follow register-renaming
//! trampolines to the real py-boundary target, decode the side-exit
//! other-target, and decide whether a branch arm resume can restore the
//! kept operand stack.

use super::*;

/// Follow a synthetic register-renaming trampoline (`ref_copy*; goto L`,
/// emitted by `emit_trampoline_for_multi_pred_link` for multi-predecessor
/// link rewrites — `flatten.py:306-334 insert_renamings`) to the real
/// py-boundary target block.
///
/// These `epsilon_link` trampolines carry no Python pc: they sit between
/// a `goto_if_not` and the branch's target block, shuffling jitcode
/// registers so the target's inputarg colors line up.  A branch guard's
/// `other_target` can land directly on such a trampoline (the codewriter
/// makes the goto label the trampoline, not the canonical block).  The
/// blackhole resumes at the *Python* level — it reads locals from the
/// `PyFrame`, not from jitcode registers — so the register renaming is
/// irrelevant to it; the correct resume coordinate is the trampoline's
/// ultimate destination, which IS a py-boundary block (starts with a
/// `live/` marker and has an exact resume-marker entry).  Without this
/// resolution the jitcode-pc→py-pc inversion maps the trampoline offset (no
/// boundary) to the wrong Python opcode (the nearest preceding entry,
/// e.g. `RETURN_VALUE`), so the guard resumes past its real target.
///
/// Trampolines themselves can carry `live/` markers (the bare block-head
/// marker inserted after every Label), and a py-boundary block can start
/// with a renaming prefix before its own `live/` marker
/// (`live; ref_copy; live; <real op>` — the first `live` belongs to the
/// link, the second to the destination opcode).  So "starts with `live/`"
/// does NOT terminate the scan; instead the scan skips through `live` /
/// `ref_copy`, follows `goto`, and returns the LAST `live` position seen
/// before the first real op — that marker has the exact resume-marker entry
/// for the destination Python opcode.  Returning the outer block-head
/// instead resolves through the first-emission table to whatever opcode
/// the codewriter placed before the trampoline in jitcode order — an
/// unrelated coordinate (e.g. a backedge `JumpBackward`), which makes a
/// branch guard's bridge resume into the WRONG arm.  The iteration bound
/// is a safety valve against a malformed self-referential chain.
pub(crate) fn resolve_branch_target_through_trampoline(code: &[u8], target: usize) -> usize {
    let mut pc = target;
    let mut resume = target;
    for _ in 0..32 {
        let Some(op) = decode_op_at(code, pc) else {
            return resume;
        };
        match op.opname {
            "live" => {
                resume = pc;
                pc = op.next_pc;
            }
            "ref_copy" => pc = op.next_pc,
            "goto" => {
                pc = read_label(code, &op, 0);
                resume = pc;
            }
            _ => return resume,
        }
    }
    resume
}

/// #73 S3.1 (approach C): re-derive a branch guard's resume `other_target` from
/// the guard's OWN `-live-` BEFORE anchor `orgpc` (== ctx.live_before_jit_pc at
/// the goto_if_not) + the recorded flavor (GuardTrue/GuardFalse), WITHOUT the
/// runtime condbox — mirroring PyPy generate_guard(resumepc=orgpc) (pyjitpl.py:520),
/// where orgpc is arm-independent and the arm is re-derived. A BOUNDED single
/// leading `-live-` skip (NOT a permissive loop), mirroring pyjitpl.py:198's
/// `assert code[pc]==op_live`: a mispositioned orgpc (e.g. `live; ref_copy;
/// goto_if_not`, orgpc one op early) FAILS loudly rather than being walked forward.
///
/// The only conditional-branch opname reaching the walk dispatch is
/// `goto_if_not` (`goto_if_not/iL`): pyre keeps COMPARE_OP and the branch as
/// SEPARATE JitCode ops and never emits the jtransform-fused `n_<cmp>` form on
/// the trace/walker path, so the goto opname set is the single `"goto_if_not"`.
/// The `L` label operand sits at index 1 (`iL`: 1B int reg, then 2B label) —
/// the same index the `goto_if_not/iL` handler reads `target` from — re-read
/// here from the re-derived `goto` so the arm-select is genuinely reconstructed
/// from `orgpc` alone rather than reusing the capture-site op.
pub(crate) fn decode_side_other_target(
    code: &[u8],
    orgpc: usize,
    flavor_guard_true: bool,
) -> Result<usize, &'static str> {
    let live = decode_op_at(code, orgpc).ok_or("notlive")?;
    if live.opname != "live" {
        return Err("notlive");
    }
    let goto = decode_op_at(code, live.next_pc).ok_or("notgoto")?;
    if goto.opname != "goto_if_not" {
        return Err("notgoto");
    }
    let target = read_label(code, &goto, 1);
    let raw = if flavor_guard_true {
        target
    } else {
        goto.next_pc
    };
    Ok(resolve_branch_target_through_trampoline(code, raw))
}

/// #73 S4 decode: if `carried` is a tagged branch `orgpc` (negative-space
/// encoding, [`majit_ir::resumedata::encode_branch_orgpc`]), expand it to the
/// genuine not-taken-arm jitcode offset (`derived` / other_target) DIRECTLY:
/// `decode_side_other_target` picks the not-taken arm from `orgpc` + flavor and
/// that jitcode offset IS the returned value. The former py_pc round-trip
/// (`python_pc_for_jitcode_pc` -> `skip_python_trivia_forward` ->
/// historical Python-pc translation, with its `num_instrs` overshoot clamp) is
/// deleted:
/// it is byte-identical because the encode self-cert
/// (`walker_capture_snapshot_for_last_guard_impl`) tags a word ONLY when this
/// same `expand_branch_carried` yields `derived == marker`, so returning
/// `derived` reproduces exactly the marker the round-trip would have. Non-tagged
/// words (offsets `>= 0`, `NO_JITCODE_PC`) pass through unchanged, so this is a
/// no-op whenever the flip is off (`decode_branch_orgpc` returns `None`).
///
/// Returns `NO_JITCODE_PC` only if the arm cannot be decoded, so the decoder
/// declines the guard capture; the encode self-cert declines to carry any
/// tagged word whose reconstruction fails, so a genuinely-carried tagged word
/// never reaches that leg.
pub(crate) fn expand_branch_carried(payload: &crate::PyJitCode, carried: i32) -> i32 {
    match majit_ir::resumedata::decode_branch_orgpc(carried) {
        None => carried,
        Some((orgpc, flavor)) => {
            let code = payload.jitcode.code.as_slice();
            match decode_side_other_target(code, orgpc, flavor) {
                Err(_) => majit_ir::resumedata::NO_JITCODE_PC,
                Ok(derived) => derived as i32,
            }
        }
    }
}

/// Decode a not-taken branch trampoline's `ref_copy` parallel-move
/// sequence into `(dst, src)` Ref-color pairs (`#420`).  The trampoline
/// (`live; (ref_copy|int_copy|float_copy)*; [goto]; live; <op>`,
/// `flatten.rs:2145 insert_renamings`) resolves the not-taken edge's Phi:
/// each `ref_copy(dst <- src)` moves the kept value's guard-pc Ref color
/// `src` into the merge block's inputarg color `dst`.  The walk does NOT
/// execute these moves — they fire only when the branch is taken at
/// runtime — so the merge color `dst` reads stale at the guard point; the
/// live kept value sits at `src`, which the walk register file still
/// holds.  Returning the `(dst, src)` list lets the snapshot / vable
/// recovery read `registers_r[src]` for each kept slot, exact for any
/// kept-stack depth (the positional depth-1 heuristic generalized).
///
/// Returns `None` on a `*_push` / `*_pop` step — a cyclic parallel move
/// whose value transits a blackhole stack the walk has no register for;
/// the caller keeps the conservative kept-stack decline.  Each `ref_copy`
/// is `[opcode, src, dst]` (`ref_copy/r>r`: `registers_r[dst] =
/// registers_r[src]`).
pub(crate) fn decode_branch_trampoline_ref_moves(
    code: &[u8],
    tramp_start: usize,
) -> Option<Vec<(u16, u16)>> {
    let mut pc = tramp_start;
    let mut moves: Vec<(u16, u16)> = Vec::new();
    for _ in 0..64 {
        let op = decode_op_at(code, pc)?;
        match op.opname {
            "live" => pc = op.next_pc,
            "ref_copy" => {
                let src = *code.get(op.pc + 1)? as u16;
                let dst = *code.get(op.pc + 2)? as u16;
                moves.push((dst, src));
                pc = op.next_pc;
            }
            // Non-Ref-bank moves (int / float locals) never feed an
            // operand-stack slot (the operand stack is always boxed Ref);
            // step over them.
            "int_copy" | "float_copy" => pc = op.next_pc,
            "goto" => pc = read_label(code, &op, 0),
            // Cyclic parallel move: the value transits a transient stack,
            // not a register the walk can read.  Decline (conservative).
            "ref_push" | "ref_pop" | "int_push" | "int_pop" | "float_push" | "float_pop" => {
                return None;
            }
            // First real destination op terminates the move list.
            _ => return Some(moves),
        }
    }
    // Cap exhausted without reaching the first real destination op: the
    // move list is truncated, not complete — decline (conservative)
    // rather than present an incomplete recovery as the full edge.
    None
}

/// Full-body-walk operand-stack depth at a branch guard's resume target.
///
/// `target` is a jitcode pc — the `goto_if_not` `other_target` (the
/// not-taken arm a guard failure deopts into).  Maps it back to the
/// Python opcode boundary the blackhole resumes at (same coordinate
/// resolution as `walker_capture_snapshot_for_last_guard_impl`: the
/// jitcode-pc→py-pc inversion + forward trivia skip) and reads the forward
/// stack-depth
/// analysis.  A depth `> 0` means the resume target carries a live
/// operand-stack temp — the short-circuit / conditional-expression /
/// chained-comparison shape the single-frame snapshot cannot rebuild on
/// the not-taken arm (#124/#281).
///
/// Returns `None` outside a full-body walk (per-opcode / trait path,
/// where the snapshot uses the static entry coordinate and this guard
/// shape does not arise) or when the coordinate resolves past the last
/// Python opcode (a synthetic loop-close overshoot, which carries no
/// kept temp).  Callers treat `None` as "no kept temp".  `frame` is the
/// frame whose jitcode the `target` offset indexes (see
/// [`ActiveResumeFrame`]).
pub(crate) fn branch_resume_target_stack_depth(
    frame: &ActiveResumeFrame,
    target: usize,
) -> Option<u16> {
    let pjc = &frame.0;
    if pjc.code_ptr.is_null() {
        return None;
    }
    // #73 family(ii) Slice B: source the not-taken-arm depth off the genuine
    // jitcode `target` through the compile-time `depth_trivia` twin, retiring
    // the `python_pc_for_jitcode_pc` inversion + runtime
    // `skip_python_trivia_forward` + static-liveness read. The twin is built for
    // every drained real-code jitcode (codewriter.rs), and the Slice A census
    // (`PYRE_M73_ENCODE_AUDIT`) proved the empty-twin fallback is never reached
    // here (0 fallback trips / 162 programs; this reader 1181 hits, all
    // populated). The `debug_assert` re-certifies the invariant in test builds.
    debug_assert!(
        pjc.depth_trivia_populated(),
        "branch_resume_target_stack_depth on an unpopulated depth-trivia twin at target={target}"
    );
    pjc.depth_trivia_for_jitcode_pc(target)
}

/// Flat-free (#267) boxed-int kept-slot hazard: a kept operand-stack slot
/// holding a heap int outside the 1-byte immediate range `[0, 256)` is
/// reconstructed with a WRONG / NULL value on a kept-stack branch-guard resume
/// (the conditional-expression / short-circuit boxed-int crash), so its
/// presence forces the conservative decline.  This replaces the dense
/// `stack_slot_color_map` read the gate used to inspect: every kept-slot kind
/// has a per-PC source — a live Variable
/// through `pcdep_color_slots` (inspect its concrete register), a Ref constant
/// (the hoisted boxed int `pcdep_color_slots` omits) through the jitcode-pc
/// const Ref slot twin (inspect the raw value).  A kept slot in NEITHER map is
/// unrestorable / a non-Ref constant the per-PC sources cannot prove safe, so
/// it forces the decline too — strictly no less conservative than the flat
/// read.
pub(crate) fn kept_stack_has_boxed_int_hazard(
    frame: &ActiveResumeFrame,
    target: usize,
    concrete_registers_r: &[ConcreteValue],
) -> bool {
    let pjc = &frame.0;
    if pjc.code_ptr.is_null() {
        // No FBW frame layout — the trait-leg `reads_null_ref` gate already
        // declines; report no hazard so this predicate adds nothing there.
        return false;
    }
    // SAFETY: `metadata` is an immutable payload layout field kept alive by the
    // frame's `Arc<PyJitCode>`; const raws are runtime PyObject pointers
    // captured at codewrite time and read-only here.
    unsafe {
        // Depth, pcdep and consts all key on the trivia-folded twin at `target`,
        // so they cannot land on different resume coordinates. An empty twin (a
        // skeleton / fixture install) yields no depth and declines below.
        let depth_opt = pjc.depth_trivia_for_jitcode_pc(target);
        let pcdep = pjc.pcdep_trivia_for_jitcode_pc(target);
        let consts = pjc.const_ref_trivia_for_jitcode_pc(target);
        let Some(depth) = depth_opt.map(|d| d as usize) else {
            // Unknown resume depth — cannot prove safe.
            return true;
        };
        if depth == 0 {
            return false;
        }
        let stack_base = pjc.metadata.stack_base;
        // The concrete shadow unboxes exact ints, so a kept slot that holds a
        // heap int surfaces either already-unboxed as `Int(v)` or still-boxed
        // as `Ref(W_IntObject)`; a value `< 0` or `>= 256` is the unrestorable
        // boxed-int hazard in either shape.
        let raw_is_boxed_int = |p: pyre_object::PyObjectRef| {
            !p.is_null()
                && pyre_object::is_int(p)
                && !(0..256).contains(&pyre_object::w_int_get_value(p))
        };
        for s in 0..depth {
            let slot = (stack_base + s) as u16;
            // Live Variable slot: inspect its concrete register value.
            if let Some(color) = pcdep.and_then(|e| {
                e.iter()
                    .find_map(|&(b, c, sl)| (b == 1 && sl == slot).then_some(c))
            }) {
                let boxed = match concrete_registers_r.get(color as usize) {
                    Some(ConcreteValue::Int(v)) => !(0..256).contains(v),
                    Some(ConcreteValue::Ref(p)) => raw_is_boxed_int(*p),
                    _ => false,
                };
                if boxed {
                    return true;
                }
                continue;
            }
            // Ref constant slot (the hoisted boxed int): inspect the raw value.
            if let Some(raw) =
                consts.and_then(|e| e.iter().find_map(|&(sl, raw)| (sl == slot).then_some(raw)))
            {
                if raw_is_boxed_int(raw as pyre_object::PyObjectRef) {
                    return true;
                }
                continue;
            }
            // Neither in pcdep nor in const_ref_slots: this kept slot has no
            // explicit color→slot mapping at the resume py_pc.  Its value IS
            // recoverable through the virtualizable shadow (the
            // `collect_outer_active_boxes` operand-stack fallback reads the
            // shadow when pcdep yields no color, line 6602-6604), so the
            // snapshot CAN reconstruct it — but we cannot inspect its
            // concrete here because we lack the color.  Conservatively
            // report NO hazard: the slot is not a literal boxed int (the
            // hazard targets hoisted heap-int consts parked in a register
            // across the guard, which always appear in pcdep or consts),
            // and declining here is a false positive for iterator /
            // non-int kept slots.
            continue;
        }
        false
    }
}

/// The resume snapshot's live Ref register colors at a kept-stack branch
/// guard's not-taken arm, plus the jitcode `num_regs_r` (the const-window
/// boundary `n()`).  These are exactly the registers the blackhole restores
/// into `registers_r` before re-executing the arm: the snapshot live set
/// (`collect_outer_active_boxes` → `frame_liveness_reg_indices_by_bank_at`)
/// plus the const-window registers at index `>= num_regs_r` (auto-loaded
/// from `jitcode.constants_r` by `init_register_files_from_runtime_jitcode`).
/// Same `fbw_mode.snapshot_sym` contract as
/// [`branch_resume_target_stack_depth`].
pub(crate) fn branch_arm_resume_ref_liveness(
    fbw_mode: FbwWalkMode,
    target: usize,
) -> Option<(std::collections::HashSet<u16>, u16)> {
    // Conservative under an inline sub-walk: `target` indexes the innermost
    // callee's jitcode while `fbw_mode.snapshot_sym` (read below) is the
    // outer portal frame, so the outer-keyed liveness banks would be read at
    // a foreign coordinate.  `None` → the caller treats the arm as
    // unrestorable and declines, which is the current sub-walk behavior.
    if fbw_mode.inline_subwalk {
        return None;
    }
    let full_body_sym = fbw_mode.snapshot_sym;
    if full_body_sym.is_null() {
        return None;
    }
    let sym = unsafe { &*full_body_sym };
    if sym.jitcode.is_null() {
        return None;
    }
    unsafe {
        let jc = &*sym.jitcode;
        if jc.payload.code_ptr.is_null() {
            return None;
        }
        // Key the liveness store off the sym's own stamped `jitcode.index`
        // — the same per-function index the snapshot encoder
        // (`collect_outer_active_boxes`) and the resume decoder
        // (`setup_bridge_sym`, via the snapshot frame's `jitcode_index`)
        // resolve — NOT the walk context's `outer_jitcode_index`, which is
        // 0 for the second and later distinct functions in a program.  A
        // wrong index resolves `target` against the FIRST function's
        // jitcode, where it is out of range, and the silent empty default
        // reports every arm read unrestorable (a spurious permanent
        // decline for every kept-stack branch outside the first function).
        //
        // Source the branch-arm Ref-liveness banks directly from the carried
        // jitcode-pc `target` via the jitcode-pc-keyed reader, with no reverse
        // translation to a Python opcode pc and no trivia skip.
        let jitcode_index = jc.index;
        let banks =
            crate::state::frame_liveness_reg_indices_by_bank_from_pc(jitcode_index, target as i32);
        let live: std::collections::HashSet<u16> = banks.ref_.iter().map(|&c| c as u16).collect();
        let num_regs_r = jc.payload.jitcode.num_regs_r() as u16;
        Some((live, num_regs_r))
    }
}

/// Scan a kept-stack branch guard's not-taken (resume) arm for a Ref
/// register READ the blackhole cannot reconstruct on guard failure.
///
/// On deopt the blackhole rebuilds `registers_r` from the guard's resume
/// snapshot — the snapshot-live Ref colors (`live_ref`) plus the auto-loaded
/// const-window registers (index `>= num_regs_r`) — then re-executes the
/// not-taken arm's static jitcode from `arm_start`.  A *regular* Ref
/// register (`< num_regs_r`) the arm READS but that is neither in the
/// snapshot nor produced by an earlier op in the arm is left at its
/// init-zero (NULL): the blackhole then feeds NULL into the consuming op
/// (e.g. `bh_binary_op(acc, NULL)` → SIGSEGV / wrong result).
///
/// That is the boxed-int short-circuit / conditional-expression resume
/// miscompile: when the codewriter parks a heap constant (a co_consts
/// `ConstPtr` — an int outside the 1-byte immediate range `[0, 256)` — or any
/// value computed before the branch) in a regular register
/// live-ACROSS the guard rather than materializing it inside the arm, the
/// resume cannot restore it.  One-byte immediates materialize via an in-arm
/// `residual_call` (a write the blackhole re-executes), so their arms stay
/// restorable and keep compiling.
///
/// Returns `true` (→ decline → interpreter, which is correct) on any read
/// it cannot prove restorable, any op it cannot decode, and on overrun.
/// Conservative by construction: a spurious `true` only forfeits a JIT
/// optimization; a spurious `false` would compile a NULL-resuming guard.
pub(crate) fn branch_arm_reads_unrestorable_ref(
    code: &[u8],
    arm_start: usize,
    live_ref: &std::collections::HashSet<u16>,
    num_regs_r: u16,
) -> bool {
    let restorable = |reg: u16, written: &std::collections::HashSet<u16>| {
        reg >= num_regs_r || live_ref.contains(&reg) || written.contains(&reg)
    };
    let mut written: std::collections::HashSet<u16> = std::collections::HashSet::new();
    let mut pc = arm_start;
    for _ in 0..512 {
        let Some(op) = decode_op_at(code, pc) else {
            return true;
        };
        match op.opname {
            // Straight-line trivia: skip.  `catch_exception/L` is a no-op on
            // the normal fall-through path — it only diverts to its handler
            // target when an exception is in flight — so it is NOT a fresh
            // resume coordinate.  The protected body after it is still part of
            // this arm's straight-line resume region and its Ref reads must be
            // scanned; treating it as a boundary (returning "restorable") would
            // let an unrestorable read in the protected body slip through.
            "live" | "catch_exception" => {
                pc = op.next_pc;
                continue;
            }
            // Unconditional jump: follow it (the conditional-expression
            // then-arm reaches the merge through a `goto`).
            "goto" => {
                pc = read_label(code, &op, 0);
                continue;
            }
            // Any merge / loop header / further guard / terminator ends this
            // arm's straight-line resume region.  A Ref read past such a
            // boundary belongs to a different resume coordinate that gets its
            // own guard check, so nothing unrestorable was found here.
            "goto_if_not" | "jit_merge_point" | "loop_header" | "finish" | "leave_frame"
            | "rvmprof_code"
            // Return / raise terminators: the arm exits the jitcode entirely.
            // No further Ref reads to check.
            | "ref_return" | "int_return" | "void_return" | "float_return" | "raise" => {
                return false;
            }
            _ => {}
        }
        // Walk the operand bytes per `decode_op_at`'s argcode contract
        // (`jitcode_runtime.rs`), flagging an unrestorable Ref read (`r` /
        // `R`) and tracking the Ref destination write (`>r`).
        let mut cursor = op.pc + 1;
        let mut chars = op.argcodes.chars();
        let mut dst_ref: Option<u16> = None;
        while let Some(c) = chars.next() {
            match c {
                'i' | 'c' | 'f' => cursor += 1,
                'r' => {
                    let Some(&b) = code.get(cursor) else {
                        return true;
                    };
                    if !restorable(b as u16, &written) {
                        return true;
                    }
                    cursor += 1;
                }
                'L' | 'd' | 'j' => cursor += 2,
                'I' | 'F' => {
                    let Some(&len) = code.get(cursor) else {
                        return true;
                    };
                    cursor += 1 + len as usize;
                }
                'R' => {
                    let Some(&len) = code.get(cursor) else {
                        return true;
                    };
                    cursor += 1;
                    for _ in 0..len as usize {
                        let Some(&b) = code.get(cursor) else {
                            return true;
                        };
                        if !restorable(b as u16, &written) {
                            return true;
                        }
                        cursor += 1;
                    }
                }
                '>' => match chars.next() {
                    Some('r') => {
                        let Some(&b) = code.get(cursor) else {
                            return true;
                        };
                        dst_ref = Some(b as u16);
                        cursor += 1;
                    }
                    Some('i') | Some('f') => cursor += 1,
                    _ => return true,
                },
                // Pyre helper payload (`*_pyre/P`): opaque operand shape —
                // conservatively decline rather than mis-walk it.
                'P' => return true,
                _ => return true,
            }
        }
        if let Some(d) = dst_ref {
            written.insert(d);
        }
        if op.next_pc <= pc {
            return true;
        }
        pc = op.next_pc;
    }
    true
}

/// The not-taken-arm Python stack depth at a branch guard's resume target,
/// resolved leg-INDEPENDENTLY through the `MetaInterpStaticData` jitcode
/// store (`pyjitcode_for_jitcode_index`) rather than the full-body-walk-only
/// `fbw_mode.snapshot_sym`.  [`branch_resume_target_stack_depth`] returns
/// `None` in the trait leg (where the bug surfaces just as it does in the
/// full-body walk — both legs re-execute the same not-taken arm on deopt),
/// so the unrestorable-kept-stack decline needs a depth probe that works in
/// either leg.  A depth `> 0` marks the short-circuit / conditional-
/// expression / chained-comparison kept-stack shape.
pub(crate) fn branch_resume_target_stack_depth_any_leg(
    target: usize,
    jitcode_index: u32,
) -> Option<u16> {
    let pjc = crate::state::pyjitcode_for_jitcode_index(jitcode_index as i32)?;
    if pjc.code_ptr.is_null() {
        return None;
    }
    // #73 family(ii) Slice B: twin-sourced leg-independent depth, py_pc inversion
    // retired (see `branch_resume_target_stack_depth`). Slice A census: this
    // reader 1178 hits, all populated, 0 fallback trips — including at the
    // `outer_jitcode_index==0` coincidence that reads `jitcodes[0]`.
    debug_assert!(
        pjc.depth_trivia_populated(),
        "branch_resume_target_stack_depth_any_leg on an unpopulated depth-trivia twin at target={target} idx={jitcode_index}"
    );
    pjc.depth_trivia_for_jitcode_pc(target)
}
