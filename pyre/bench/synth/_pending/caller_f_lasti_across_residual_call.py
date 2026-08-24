# FILING (2026-08-24).  A caller's `f_lasti`, read from inside a callee the
# trace left RESIDUAL, reports the resume coordinate instead of the CALL the
# caller is suspended at -- but only for the FIRST residual call in the traced
# body.  Interpreted iterations answer correctly, so a fixture that collects a
# SET across a run sees two elements.
#
# Reduced from `bench/synth/frame_lasti_fold_callee_and_bridge` (added by
# `251ebba8b9c`).  That fixture reaches the shape only when its `caller_view`
# is refused inline; on this tree it is not, so the fixture passes and this
# defect is not gated anywhere.
#
# WHAT WAS MEASURED (dynasm, release, 2026-08-24)
#
#   * one call site, callee refused inline: `f_lasti` reads 56 for the first
#     ~2100 iterations and 62 from the compile on.  56 is the `CALL`; 62 is the
#     last cache word of that same `CALL`, i.e. `next_instr - 1`.
#   * two call sites in one body: only the FIRST diverges.  The second reports
#     its own `CALL` offset on all 20000 iterations.
#   * a residual call placed BEFORE the reads: all three sites then report
#     their own offsets, 20000/20000.  So the coordinate is correct from the
#     second residual of the body onward, and the defect is what the first one
#     reads.
#
# `next_instr - 1` is the resume convention (`state.rs`
# `flush_walk_end_state_to_frame`, `flush_walk_end_state_after_outer_call`);
# the executing convention is what a getter owes (`specialize.rs`
# `try_walker_specialize_frame_lasti`, and `LiveLastInstrGuard::enter_frame`).
# `residual_call.rs` already publishes `vstack_cur_pypc` before a residual for
# exactly this reason -- "so a force inside the callee reports the executing
# line rather than the one the last resume point left behind" -- so the
# question is not whether a publication exists but why the body's first
# residual does not get one that survives.
#
# NOT the same defect as `_pending/loop_callee_for_header_resume.py`, though it
# is the same field: there the triple `(last_instr, valuestackdepth, cells)` is
# torn at a resume; here nothing resumes, a live reader just reads the field.
#
# HOW THIS WAS REACHED.  `a67442cccb0` listed `PyreHelperKind::LoadDeref` as a
# replay-safe read, which classified every freevar-reading callee
# `DeferredCall` and let it inline; `b6d9cc510cd` then denied such a callee
# from its first vable escape so the enclosing loop would stop being retired by
# `MAX_TRACE_ABORT_COUNT`.  Loops that had never compiled then compiled, and
# `frame_lasti_fold_callee_and_bridge` began answering its caller-from-callee
# site two ways.  Both were reverted; the read below reaches the same answer
# without either, through a callee `code_has_for_iter` refuses.
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
