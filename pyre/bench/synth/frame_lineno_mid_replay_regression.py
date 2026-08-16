# pyre-check: selfcheck
# Self-checking regression guard for the coordinate a frame reports WHILE it is
# still running. The synthetic suite discovers it as a self-checking fixture.
#
# `dispatch_bytecode` (pyopcode.py) stamps `last_instr` before every opcode, so
# a running frame answers `f_lineno`, `f_lasti` and any traceback taken off it
# for the instruction it is on.  Compiled code does not run that store, and the
# blackhole replaying a frame syncs only `valuestackdepth`, so the coordinate
# reaches the frame only if the replay publishes it at the `-live-` marker.
# Without that publish a frame the function-entry portal compiled still carries
# the `-1` initialization sentinel, which `offset2lineno` answers with the code
# object's first line -- the `def` line, i.e. offset 0.
#
# Splitting each survey across several calls is what makes it a test: the loop
# compiles part-way through, so a set over the rounds holds the interpreted
# answer and the replayed one together and a divergence appears as a SECOND
# element rather than a shifted single value.
#
# It asserts rather than diffing against pypy3 so the recursive case can pin an
# exact per-level tuple, which an output diff cannot express.
#
# It is also the guard for a store-width defect, which is why it is worth
# keeping separate: `PyFrame.valuestackdepth` and `PyFrame.last_instr` are
# adjacent machine words, so a store that takes its width from a descr
# declaring a fixed 8 bytes runs past the first field and deposits the zero
# high half onto the second.  On a 64-bit target the two are the same number
# and nothing happens; on wasm32 the words are 4 bytes and `plain` below
# reported `[0, 4]` against `[4]` everywhere else, from the first compiled call
# onward.  The publish itself was never at fault -- it stores through
# `*mut isize` and read back intact; the clobber came afterwards, from the
# blackhole replaying a `setfield_vable_i` at the neighbouring offset.
import sys

N = 4000


def caller_offset():
    """The caller's coordinate, read from a callee while the caller is still
    running.  Nothing has left the caller's frame yet, so neither exit publish
    has fired and the answer can only come from the frame being kept current."""
    frame = sys._getframe(1)
    return frame.f_lineno - frame.f_code.co_firstlineno


def raises_out(i):
    raise KeyError(i)


def plain(n):                    # +0
    k = 0                        # +1
    while k < n:                 # +2
        k += 1                   # +3
    return caller_offset()       # +4


def mid_replay_getframe(n):      # +0
    tb = None
    k = 0
    while k < n:
        try:
            raise ValueError(k)
        except ValueError as e:
            tb = e.__traceback__
        k += 1
    return caller_offset()       # +9


def mid_replay_handler(n):       # +0
    tb = None
    k = 0
    while k < n:
        try:
            raise ValueError(k)
        except ValueError as e:
            tb = e.__traceback__
        k += 1
    try:
        raises_out(k)            # +10
    except KeyError as e:
        t = e.__traceback__
        base = t.tb_frame.f_code.co_firstlineno
        return (t.tb_lineno - base, t.tb_frame.f_lineno - base)   # +14


def recursive_mid_replay(n, depth):
    """Direct recursion, every level with its own hot loop, so every level is
    replayed and every level shares ONE code object with its caller -- the shape
    where a per-level frame mix-up survives a code-object check.  A level
    answering for another one shows up as a shifted offset."""
    tb = None
    k = 0
    while k < n:
        try:
            raise ValueError(k)
        except ValueError as e:
            tb = e.__traceback__
        k += 1
    if depth > 0:
        inner = recursive_mid_replay(n, depth - 1)
    else:
        inner = ()
    return ((caller_offset(), caller_offset()),) + inner      # +17


def main():
    rounds = 8
    each = N // rounds
    failures = []

    def check(label, got, want):
        if got != want:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    check("plain", sorted({plain(each) for _ in range(rounds)}), [4])
    check(
        "getframe",
        sorted({mid_replay_getframe(each) for _ in range(rounds)}),
        [9],
    )
    check(
        "handler",
        sorted({mid_replay_handler(each) for _ in range(rounds)}),
        [(10, 14)],
    )
    check(
        "recursive",
        recursive_mid_replay(N // 2, 3),
        ((17, 17), (17, 17), (17, 17), (17, 17)),
    )

    if failures:
        for f in failures:
            print("FAIL", f)
        return 1
    print("PASS mid-replay coordinates")
    return 0


sys.exit(main())
