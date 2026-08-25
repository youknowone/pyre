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
# ===== 2026-08-25: THE WRONG VALUE IS NOT AN INSTRUCTION AT ALL =====
#
# Re-measured after the rebase onto `b9c76982bd6`; the defect is unchanged on
# both backends, and CPython names the correct answer:
#
#   pyre     f_lasti values: [(52, 11074), (58, 8926)]   FAIL
#   cpython  f_lasti values: [(52, 20000)]               PASS
#
# `dis` on `main()` places `CALL` at offset 52 and the next real instruction,
# `STORE_FAST`, at 60.  So the CALL owns 52..59 -- three cache entries at 54,
# 56, 58 -- and the wrong answer 58 is the CALL's LAST CACHE SLOT.  It is not a
# rival instruction offset; it is not an instruction boundary at all.
#
# That reframes where the fix can go.  Every refuted line above targets the
# WRITER or the record ORDER.  The reader was never examined, and CPython holds
# an invariant the reader currently breaks: `f_lasti` always names an
# instruction (`_PyInterpreterFrame_LASTI` is derived from `instr_ptr`), never a
# cache slot.  Snapping a stored `last_instr` that lands inside an instruction's
# cache region back to its owning instruction turns 58 into 52 -- the measured
# correct answer -- and touches no resume consumer, because the blackhole and
# the vable sync read the field directly, not through the getset.
#
# The getter is `pyframe.rs` `fget_f_lasti` (returns `self.last_instr` bare,
# mirroring `pyframe.py fget_f_lasti`), scaled by 2 at `typedef.rs:8222`.
#
# !! The open design question is WHICH side owns the conversion, and it is a
# real one, not a formality.  The stored value is deliberately a RESUME
# coordinate (`resume_snapshot.rs:680` and `:2624`, both `pc - 1`, with a
# comment saying a stale one makes a guard resume at the loop header).  Upstream
# PyPy has no adaptive caches, so its one convention is unambiguous; 3.14's
# adaptive bytecode is what splits it.  A reader-side snap keeps every resume
# consumer untouched but leaves two conventions in one slot; a writer-side fix
# has to satisfy the resume path first and is what the eight refutations above
# kept failing to do.  Measure before choosing -- and note the snap has NOT been
# implemented or tested, only shown to produce the right number by hand.
#
# ===== 2026-08-25: THE READER-SIDE SNAP IS REFUTED =====
#
# The section above left "which side owns the conversion" open and leaned
# reader-side.  Do not implement it.  A snap inside `fget_f_lasti` breaks two
# gated fixtures by construction, because the getset is NOT the only producer
# of a Python-visible `f_lasti`:
#
#   * COMPILED code answers the read from `try_walker_specialize_frame_lasti`,
#     which emits a trace constant `py_pc as i64 * 2` (`specialize.rs`) without
#     going near `fget_f_lasti` (`pyframe.rs`, one caller, `typedef.rs:8222`).
#     `bench/synth/frame_lasti_fold_callee_and_bridge` asserts `len(seen) == 1`
#     across BOTH channels, so snapping one of them makes the two disagree
#     wherever the raw value is already an instruction -- which is everywhere
#     except this defect.
#   * `tb_lasti` is minted from the RAW word: `pytraceback.rs` stores
#     `last_instruction * 2` verbatim.  `traceback_inlined_callee_lasti_regression`
#     (:41, :77-81) and `frame_lasti_fold_foreign_frames` (:48, :65-68) both
#     assert `f_lasti == tb_lasti`, so a snap on one side alone breaks them.
#   * upstream reads through the getset and needs it raw
#     (`pypy/interpreter/pyframe.py:675`).
#
# A SECOND OBSERVABLE BREAK, not yet exercised by any fixture: a cache-region
# `last_instr` also makes `frame.f_lineno = N` fail with "can't jump from
# unreachable code".  `mark_stacks` steps `i + cache_entries + 1`
# (`pyframe.rs:2238-2239`), so every cache index keeps `MARK_UNINITIALIZED`
# (`:2029`) and `fset_f_lineno` indexes one of them.  CPython cannot hit this:
# `TARGET(CALL)` sets `frame->instr_ptr = next_instr` BEFORE `next_instr += 4`,
# so `_PyInterpreterFrame_LASTI` names the CALL for the callee's whole lifetime
# and `frame_lineno_set` indexes `stacks[...]` with no snapping at all.  That is
# the invariant this defect violates, and it says the fix belongs at the WRITER.
#
# But the narrow writer change -- "store the CALL's own index" -- is also wrong:
# the stored value has an exact `+1` inverse applied by every resume consumer
# (`pyframe.rs:5014` `next_instr() = last_instr + 1`, `eval.rs:12971`
# `vable_ni = value + 1`), and that derived coordinate becomes `resume_pc`
# (`eval.rs:12452`), the frame's re-entry point (`call_jit.rs:3522`), the bridge
# walk's `start_pc`/`lasti_pc` (`trace.rs:1160-1183`) and the green/decline key
# (`trace.rs:3620-3623`).  Storing `CALL` would make all of those resume at the
# CALL's FIRST cache word.  It moves the defect onto the resume path.
#
# So the honest scope: of 33 field-read sites, only three want the resume
# coordinate (`PyFrame::next_instr()`, `resume_execute_frame`, and the opaque
# save/restore pairs); every other reader wants the executing instruction.  A
# real fix makes `next_instr()` cache-aware and moves the whole `pc - 1` writer
# family (`state.rs:5728/5744/5929/6249/6442` plus the two `resume_snapshot.rs`
# sites) and the interpreter's own `set_last_instr_from_next_instr(opcode_pc + 1)`
# round trip (`pyre-interpreter/src/eval.rs:2338`) together.  There is also no
# backward helper to build on: `skip_python_trivia_forward` (`diag.rs:254`),
# `semantic_fallthrough_pc` (`pyjitpl.rs:19`) and all three `skip_caches` copies
# walk FORWARD, and `decode_instruction_at` backs up over `ExtendedArg` only.
#
# That is an epic, not a fix.  Leave the defect filed.
#
# NOT the same defect as `_pending/loop_callee_for_header_resume.py`, though it
# is the same field: there the triple `(last_instr, valuestackdepth, cells)` is
# torn at a resume; here nothing resumes, a live reader just reads the field.
# That filing lists four more refuted attempts -- read it too.
#
# HOW THIS SURFACED.  `a67442cccb0` listed `RuntimeHelperKind::LoadDeref` as a
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
