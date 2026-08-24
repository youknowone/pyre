# pyre-check: selfcheck
# Self-checking guard for `f_trace_opcodes` armed on a frame that is ALREADY
# running compiled code.  Sibling of `settrace_f_trace_armed_mid_loop`; see
# that fixture for why the `debugdata` read in `dispatch_bytecode`'s
# `jit.we_are_jitted()` arm is what keeps the events coming, and for why the
# tail has to be long.
#
# The opcode probe is the denser half of the pair: `run_trace_func`
# (executioncontext.py) fires an `opcode` event for EVERY bytecode rather than
# once per source line, so it reads the same defect through a different rule
# and at roughly ten times the event rate.
#
# Absolute counts are not portable -- they are a property of the bytecode a
# given implementation compiles this loop to, and cpython 3.14.6 and pypy3
# 7.3.22 disagree on it for the same source -- so the invariant is stated
# against THIS BINARY'S OWN interpreted answer.  Three runs:
#
#   * two short runs, both below `threshold` and therefore never compiled,
#     whose tails differ by exactly one iteration.  One traced iteration `B`
#     and the loop-exit block `E` are recovered from the pair by algebra, and
#     so is `P`, the remainder of the iteration the setter interrupted.
#   * one long run, long past `threshold`, whose stream after the first loop
#     header must be `B * (HOT_TAIL - 1) + E` -- EXACTLY, event for event and
#     `f_lasti` for `f_lasti`, not merely as dense.
#
# Two separate assertions, because the fix establishes two different things.
# They share all three runs, which is why they are one file.
#
# 1. THE TAIL, exact.  From the first loop header after arming onward the
#    compiled stream is identical to the interpreted one.  This is the gate
#    worth having: the pre-fix behaviour is that `settrace` forces every frame,
#    the loop deopts once, reports that single iteration, and then runs
#    compiled and silent for the rest of the tail however long it is.
#
#    One normalisation, and only one: consecutive duplicate loop-header events
#    are collapsed.  The rule `run_trace_func` fires on is
#    `frame.last_instr < d.instr_prev_plus_one`, so an implementation whose
#    compiled back edge lands below that fires the header twice in a row --
#    pypy3 does (3793 header events over a 2000-iteration tail against
#    cpython's and pyre's 2000), cpython does not, and neither is wrong.
#    Nothing else is normalised and nothing is given any tolerance.
#
# 2. THE ARMING ITERATION, pinned.  The setter interrupts an iteration, and an
#    implementation that leaves compiled code at the forcing call resumes the
#    rest of that iteration through its guard-failure path rather than through
#    the eval loop, so the opcodes between the setter and the next merge point
#    report nothing.  Measured on pyre when this was written: five events,
#    `f_lasti` 78, 80, 82, 84, 86 -- against zero on `PYRE_JIT=0`, on cpython
#    and on pypy3.  That is a real and separate gap, so it is asserted rather
#    than absorbed: whatever the hot run does report in that window must be a
#    suffix of what the interpreted run reports (nothing spurious, nothing
#    reordered), and at most `ARMING_GAP_MAX` events may be missing.  The bound
#    cannot grow silently; a fix that closes the gap to zero still passes.
#    It is bounded by a single iteration at any tail length, where the defect
#    assertion 1 guards against loses the entire tail.
import sys

COLD = 40  # below `threshold` (1039) -- these two runs are never compiled
HOT = 20000  # past it many times over before ARM
COLD_TAIL = 6
HOT_TAIL = 2000
ARMING_GAP_MAX = 5  # measured on pyre; 0 on PYRE_JIT=0, cpython and pypy3

events = []


def tracer(frame, event, arg):
    events.append((event, frame.f_lasti))
    return tracer


def arm():
    frame = sys._getframe(1)
    frame.f_trace = tracer
    frame.f_trace_lines = False
    frame.f_trace_opcodes = True
    sys.settrace(tracer)


def hot(n, arm_at):              # +0
    acc = 0                      # +1
    for i in range(n):           # +2
        acc += i                 # +3
        if i == arm_at:          # +4
            arm()                # +5
    sys.settrace(None)           # +6
    return acc                   # +7


def run(n, tail):
    del events[:]
    hot(n, n - tail)
    return list(events)


def collapse(stream, header):
    """Fold runs of consecutive loop-header events down to one."""
    out = []
    for event in stream:
        if event[1] == header and out and out[-1][1] == header:
            continue
        out.append(event)
    return out


def main():
    failures = []
    # Both control runs first, while the loop is still cold.
    cold_a = run(COLD, COLD_TAIL)
    cold_b = run(COLD + 1, COLD_TAIL + 1)
    hot_events = run(HOT, HOT_TAIL)

    block_len = len(cold_b) - len(cold_a)
    if block_len < 3:
        print(f"FAIL cold: one iteration is {block_len!r} opcode events, want at least 3")
        return 1

    # `cold_a` is `P + B * (COLD_TAIL - 1) + E` and `cold_b` is the same with one
    # more `B`, so the arming remainder `P` is the shortest prefix for which
    # dropping one block from `cold_b` leaves exactly `cold_a`.
    arming = None
    for i in range(len(cold_a) + 1):
        if cold_b[i + block_len:] == cold_a[i:]:
            arming = i
            break
    if arming is None:
        print("FAIL cold: the two control runs do not differ by exactly one iteration")
        return 1

    cold_prefix = cold_a[:arming]
    block = cold_b[arming:arming + block_len]
    exit_block = cold_a[arming + block_len * (COLD_TAIL - 1):]
    header = block[0][1]
    if cold_a != cold_prefix + block * (COLD_TAIL - 1) + exit_block:
        print("FAIL cold: the interpreted tail is not a repetition of one iteration")
        return 1

    split = next(
        (i for i, event in enumerate(hot_events) if event[1] == header), len(hot_events)
    )
    hot_prefix, hot_suffix = hot_events[:split], hot_events[split:]

    # 1. The tail, exact.
    want = block * (HOT_TAIL - 1) + exit_block
    got_n, want_n = collapse(hot_suffix, header), collapse(want, header)
    if got_n != want_n:
        where = next(
            (i for i, (g, w) in enumerate(zip(got_n, want_n)) if g != w),
            min(len(got_n), len(want_n)),
        )
        failures.append(
            f"hot: the stream after the first loop header is not the interpreted "
            f"one repeated: got {len(got_n)} events, want {len(want_n)} "
            f"({len(block)}/iteration over {HOT_TAIL - 1} iterations + "
            f"{len(exit_block)} at the exit); first difference at index {where}: "
            f"{got_n[where:where + 3]!r} vs {want_n[where:where + 3]!r}"
        )

    # 2. The arming iteration, pinned.
    gap = len(cold_prefix) - len(hot_prefix)
    if gap < 0 or hot_prefix != cold_prefix[gap:]:
        failures.append(
            f"hot: the arming-iteration window is not a suffix of the interpreted "
            f"one: got {[l for _, l in hot_prefix]!r}, interpreted "
            f"{[l for _, l in cold_prefix]!r}"
        )
    elif gap > ARMING_GAP_MAX:
        failures.append(
            f"hot: {gap} events lost in the arming iteration "
            f"(f_lasti {[l for _, l in cold_prefix[:gap]]!r}), want at most "
            f"{ARMING_GAP_MAX}"
        )

    if not all(event == "opcode" for event, _ in hot_events + cold_a):
        failures.append(
            f"non-opcode events present: "
            f"{sorted({e for e, _ in hot_events + cold_a})!r}"
        )

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print(
        f"PASS f_trace_opcodes armed mid-loop "
        f"(tail exact: {len(got_n)} events; arming gap {gap}/{ARMING_GAP_MAX})"
    )
    return 0


sys.exit(main())
