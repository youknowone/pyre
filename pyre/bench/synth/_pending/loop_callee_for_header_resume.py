# FILING (2026-08-23). Read this before attempting a fix -- four attempts have
# already been made and refuted by measurement, and the diagnosis they produced
# is not the one the first three assumed.
#
# THE INVARIANT IS A TRIPLE, not a pair:
#     (last_instr, valuestackdepth, the stack cells [base .. base+depth))
# Attempt 4 is the proof rather than merely a failure: pairing the depth at the
# publishing writer makes `ForIter` read slot base+0, but nothing ever wrote the
# iterator INTO that cell, so it still reads an int -- which is exactly the
# TypeError below. Supplying two words without the cells cannot work.
#
# The four refuted attempts, with what discriminated each:
#   1. `adopt_blackhole_crn` restores `last_instr` alone where its sibling
#      `apply_blackhole_crn_handoff` restores both via `correct_resume_vsd`.
#      Real and main-owned -- but an audit placed inside it NEVER FIRES here.
#   2. `LiveLastInstrGuard` saves only `last_instr`. Saving/restoring the whole
#      `FrameScalars` pair did not fix it: the guard is FAITHFUL, and the pair is
#      already inconsistent when `enter_frame` captures it.
#   3. The CALL_ASSEMBLER pin in `emit_walker_loop_callee_call_assembler`. Real
#      and live on main (11 emits on `str_search_index_bounds`) -- ZERO events
#      on this reproducer.
#   4. Pairing both words at the publish site: RED, build provenance verified.
#      `maybe_publish_inline_callee_last_instr_concrete` records this in its own
#      doc.
#
# The plurality of coordinates is NOT the defect. A virtualizable's stored fields
# are stale by design while the JIT executes -- upstream elides per-opcode
# `last_instr` stores and reconstructs at exits -- so the walk-local coordinate
# governs during a walk, the resume/blackhole image at exits, and the concrete
# frame only outside the JIT domain.
#
# THE NEXT EXPERIMENT IS CONSUMER-SIDE, not another writer audit: instrument
# every interpreter (re)entry on this frame, logging `last_instr`, `vsd`, AND THE
# TYPES OF THE CELLS [base .. base+analysis_depth), to catch a re-entry that
# never passed a reconstruction writing all three.
#
# Do NOT resurrect the "resumed inside a Cache slot" reading: `pc` in
# `report_stack_underflow` is `next_instr()`, so `last_instr=66 pc=67` means
# resume pc 66, an ordinary opcode boundary. That was an inference, never an
# observation.
#
# Census: 128-case sweep = 32 fail / 96 pass, all 32 in the `rec_for` family.
# Red on four dynasm binaries incl. one built at origin/main, and on cranelift --
# so it is backend-independent and main-owned. Clean under PYRE_NO_JIT=1.
# One-shot audit line:
#   inline_callee_last_instr: next_instr=18 op=ForIter
#     analysis_depth=Some(1) frame_depth=Some(0) (vsd=3 base=3)
#
# OPEN DEFECT reproducer, kept out of the runner glob until it passes.
#
# A loop-bearing callee resumed at its own `for` header with the operand-stack
# height left over from the walk instead of the height that header executes
# against.  `rec` has three locals (n, acc, ch) and its loop header is the
# FOR_ITER at py_pc 18, whose static operand-stack depth is 1 (the iterator).
# The frame reaches the header advertising `valuestackdepth == stack_base`, so
# its TOS is the last LOCAL and the resumed FOR_ITER calls `next()` on it:
#
#     TypeError: 'int' object is not an iterator
#
#     python3 pyre/bench/synth/_pending/loop_callee_for_header_resume.py   # PASS
#     target/release/pyre-dynasm pyre/bench/synth/_pending/loop_callee_for_header_resume.py
#
# Deterministic on dynasm; `PYRE_NO_JIT=1` passes, and the failure survives
# `PYRE_TRACE_EAGERNESS=1000000` (bridging off), `PYRE_WALKABORT_OFF=1` and
# `PYRE_FORITER_CALL_BODY=1`.  The two loops in `main` are both needed at these
# trip counts; a single loop needs ~1200 trials to reach it.
#
# Traced writer (`PYRE_FBW_DEBUG_ABORT` census against an instrumented build):
# `maybe_publish_inline_callee_last_instr_concrete`
# (pyre-jit-trace/src/jitcode_dispatch/residual_call.rs) publishes the EXECUTING
# `last_instr = callee_py_pc` onto the inline callee's concrete frame and writes
# no `valuestackdepth` beside it; `LiveLastInstrGuard` then captures that pair
# as its saved state and restores it verbatim, so the pc reads back as a resume
# coordinate with the walk's mid-flight depth.

DATA = tuple(range(40))
STEP = sum(DATA)


def rec(n):
    if n <= 0:
        return 0
    acc = 0
    for ch in DATA:
        acc = acc + ch
    return acc + rec(n - 1)


def expected(n):
    return STEP * n if n > 0 else 0


def main():
    total = 0
    bad = 0
    for trial in range(50):
        n = trial % 50
        for value in [rec(i) for i in range(n)]:
            total += value
    for trial in range(60):
        n = trial % 20
        for i in range(n):
            value = rec(i)
            if value != expected(i):
                bad += 1
                if bad < 4:
                    print("MISMATCH i=%d got=%r want=%r" % (i, value, expected(i)))
            total += value
    if bad:
        print("FAIL %d wrong rec() results" % bad)
        raise SystemExit(1)
    if total != 17955600:
        print("FAIL total=%d want=17955600" % total)
        raise SystemExit(1)
    print("PASS loop-callee for-header resume total=%d" % total)


main()
