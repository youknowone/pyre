# Self-checking regression guard for the coordinate a frame reports WHILE it is
# still running (registered via check.py run_selfcheck, NOT the synthetic suite).
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
# Why this is not a synthetic bench: the wasm backend does not satisfy the
# invariant today, and check.py's synthetic suite has no per-backend scoping.
# Measured on this HEAD -- `plain` below reports `[0, 4]` on wasm against `[4]`
# on pypy3, CPython, dynasm and cranelift, with the divergence appearing from
# the first COMPILED call onward and persisting.  Instrumenting both ends showed
# the marker hook writing the right coordinate into the right frame at the right
# offset and reading it back intact, and the interpreter then reading 0 from
# that same address -- so a wasm-side writer clears it between the publish and
# the residual `sys._getframe`.  Neither `restore_resume_state_from` nor
# `set_last_instr_from_next_instr` is that writer (probed: neither ever targets
# the caller frame), and a blackhole `setfield_vable_i` cannot be, because the
# wasm blackhole builder carries no cpu and that handler would panic.  The
# invariant is asserted here for the native backends while that is open; the
# post-return coordinate, which wasm does satisfy, stays in the synthetic bench.
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
