# FILING (2026-08-24).  A caller's `f_lasti`, read from inside a callee the
# trace left RESIDUAL, reports the RESUME coordinate (`post_call_pc - 1`)
# instead of the `CALL` the caller is suspended at.
#
# READ THE FRAMING NOTE FIRST.  The first version of this filing said "only the
# FIRST residual call of a traced body is wrong".  That is a SYMPTOM, not the
# rule, and taking it as the rule sends you looking for a publication-ordering
# bug that does not by itself explain anything.  The rule is:
#
#     a read is wrong exactly when it executes while COMPILED code is the
#     caller's active executor, and right whenever it executes interpreted.
#
# The read's own force deopts the activation, so at most one such read happens
# per iteration -- which is what makes it LOOK like "the first one".
#
# WHAT WAS MEASURED (release dynasm at 82fe714868d, 20000-iteration loops).
# `guard_failures` is the discriminator: it tracks the wrong-answer count to
# within one on every variant, which is what identifies the force as the event.
#
#   variant                          reads                      loops  guard_failures
#   one call site (this file)        56 x11074 / 62 x8926           3            8927
#   two sites, same callee           A: 54 x10547 / 60 x9453        3            9453
#                                    B: 128 x20000  (all right)
#   a residual call placed FIRST     all three sites right,         5           19640
#                                    20000/20000 each
#   two sites on opposite arms       A: 82 x1074 / 88 x8926         3            8927
#   of `if i % 2`                    B: 160 x20000  (all right)
#
# Every "all right" row is a site that never executes compiled:
#   * site B of variant 2 runs after site A's force has already deopted the
#     iteration;
#   * variant 3's leading residual takes the force before any read -- and note
#     its loop DOES compile (5 loops), so "a preceding residual fixes it" is
#     false, the reads simply stopped being compiled;
#   * variant 4's B arm side-exits every odd iteration and no bridge is built
#     (`bridges_compiled=0`), so it is interpreted throughout even though it is
#     the first read of its own iteration.
#
# `post_call_pc - 1` == `CALL_idx + cache_count` identically, so an opcode with
# a different cache count does NOT discriminate the two readings.  Do not spend
# a probe on that.
#
# THE WRITER (verified by reading, four independent readers agreeing, and the
# arithmetic matching both histograms to the digit):
#
#   jitcode_dispatch/resume_snapshot.rs `let last_instr_value = py_pc as i64 - 1`
#     -- py_pc already re-pointed past the call by the `after_residual_call`
#        block (`after_residual_marker_for_jitcode_pc` / `semantic_fallthrough_pc`),
#        mirrored straight into the virtualizable shadow.  Its own comment says
#        it publishes THIS GUARD'S RESUME COORDINATE, which is correct for a
#        resume and wrong for a getter.
#   -> majit-metainterp `resume.rs` -> `virtualizable.rs`
#      `write_from_resume_data_partial`, which stamps ALL static fields
#   -> reached from `pyjitpl.rs` on the force.
#
#   CALL at instr 28 with 3 cache words -> post-call 32 -> 31 -> f_lasti 62.
#   Site A at 27 -> 30 -> 60.  Both measured values.
#
# THE ORDER (verified): in `residual_call.rs` the INLINE CALLEE's `last_instr`
# store is recorded BEFORE the CALL (through
# `maybe_walker_vable_and_vrefs_before_residual_call`), while the PORTAL
# caller's executing-pc publish sits inside
# `try_execute_residual_call_via_executor`, which is recorded AFTER the CALL op.
# So in compiled code the caller's publish runs only once the callee returns.
# Two defects on one store, and fixing either alone is inert: it is out-ordered,
# AND the force's resume-image write-back rewrites the same shadow slot anyway.
#
# REFUTED -- do not repeat these (each was investigated and killed):
#   * "the publish is merely mis-ORDERED": if the force's resume image
#     supersedes the slot, moving the publish before the CALL changes nothing.
#   * `state.rs` `virt_restore_scalars_raw` / `restore_virtualizable_from_raw`
#     as the force's stamp: that function has NO callers.
#   * "the force is one-shot per activation": the token is re-armed before
#     every may-force residual.
#   * "a non-reading residual absorbs the force": a residual that reads no frame
#     field forces nothing; the frame-chain walk does not force.
#   * `flush_active_frame_escape` as the producer: its escape pc is
#     `vstack_cur_pypc`, so it would write `call_pc - 1` -> f_lasti 54, a value
#     absent from every histogram above.
#   * the virtualizable BOXES as the reader's source: the read is a plain heap
#     read through the getset.  (Probed independently at three portal sites:
#     boxes 110/165/185 against executing pcs 115/170/197.)
#   * the walker fold `walker_frame_executing_py_pc` correctly DECLINES a
#     caller-frame-read-from-inside-a-callee receiver, so it is not in play.
#   * OptHeap eliding the publish: it keeps and hoists the store.
#
# NOT the same defect as `_pending/loop_callee_for_header_resume.py`, though it
# is the same field: there the triple `(last_instr, valuestackdepth, cells)` is
# torn at a resume; here nothing resumes, a live reader just reads the field.
# That filing lists four more refuted attempts -- read it too.
#
# HOW THIS SURFACED.  `a67442cccb0` listed `PyreHelperKind::LoadDeref` as a
# replay-safe read, which classified every freevar-reading callee
# `DeferredCall` and let it inline; `b6d9cc510cd` then denied such a callee
# from its first vable escape so the enclosing loop would stop being retired by
# `MAX_TRACE_ABORT_COUNT`.  Loops that had never compiled then compiled, and
# `bench/synth/frame_lasti_fold_callee_and_bridge` began answering its
# caller-from-callee site two ways.  Both were reverted in `82fe714868d`; the
# read below reaches the same answer without either, through a callee
# `code_has_for_iter` refuses.
import sys

N = 20000


def refused_inline():
    # A `for` anywhere in the body is what `code_has_for_iter` refuses, so this
    # callee stays residual without depending on either reverted change.
    acc = 0
    for _ in range(1):
        acc += 1
    return sys._getframe(1).f_lasti


def main():
    seen = {}
    for i in range(N):
        v = refused_inline()
        seen[v] = seen.get(v, 0) + 1
    offsets = sorted(seen)
    print("f_lasti values:", [(o, seen[o]) for o in offsets])
    if len(offsets) != 1:
        print(f"FAIL caller f_lasti diverged across the compile: {offsets}")
        return 1
    print("PASS")
    return 0


sys.exit(main())
