# CPython-suite gap: `test.test_sys_settrace` is SKIP in
# `pyre/cpython_tests/baseline.json` ("implementation detail"), so the one suite
# module that walks a tracer's whole line-event stream never runs here.  Nothing
# else in the tree counts line events, and a spurious one is invisible to every
# test that only asks whether a callback fired at all -- the tracer still gets
# its `call`, still gets every real line, and still gets its `return`.
#
# parity-tests reason: pyre's bytecode carries 3.14's `RESUME`, which opens
# every function body stamped with the `def` line.  pypy's 3.11 bytecode has no
# such instruction -- its pc 0 IS the first body statement -- so pypy cannot
# have this bug and offers no reading on it.  CPython 3.14 does have `RESUME`
# and refuses to instrument it: `initialize_lines` (`Python/instrumentation.c`)
# excludes `END_ASYNC_FOR`, `END_FOR`, `END_SEND`, `RESUME` and `POP_ITER` from
# `INSTRUMENTED_LINE`.  Without that exclusion pyre reported one extra `line`
# event at the `def` line on every call: 104 events where CPython 3.14 and
# pypy3 both report 103.
#
# The generator half is the other side of the same instruction.  A resumed
# generator re-enters at a `RESUME` carrying the `yield` line, so instrumenting
# it would report a line event for a line the frame is only passing back
# through.  Both references agree on the whole resumed stream, so it is pinned
# verbatim.
#
# The `f_lineno` assertion at the end is the guard rail on the fix.  A
# not-yet-started frame answers `f_lineno` with its `def` line through
# `offset2lineno(code, -1)`, and that is the correct answer for the getset
# (`pyframe.py fget_f_lineno`) even though it is the wrong one for a line
# event.  Suppressing the event by changing that shared helper instead would
# break this, and `bench/synth/frame_inlined_callee_own_image_regression.py`
# with it.
import sys


def record(thunk):
    """Every event a tracer sees while `thunk` runs, as (event, name, lineno)."""
    events = []

    def tracer(frame, event, arg):
        events.append((event, frame.f_code.co_name, frame.f_lineno))
        return tracer

    sys.settrace(tracer)
    try:
        thunk()
    finally:
        sys.settrace(None)
    return events


def offsets(events, func):
    """`events` for `func`'s frames, with linenos relative to its `def`.

    Relative so the expectations survive an edit above them.
    """
    first = func.__code__.co_firstlineno
    name = func.__code__.co_name
    return [
        (event, lineno - first) for event, seen, lineno in events if seen == name
    ]


def work(n):                                              # +0
    total = 0                                             # +1
    for i in range(n):                                    # +2
        total += i                                        # +3
    return total                                          # +4


def counting(n):                                          # +0
    total = 0                                             # +1
    for i in range(n):                                    # +2
        total += i                                        # +3
        yield total                                       # +4
    return total                                          # +5


def entering_a_function_emits_no_line_event_for_the_def_line():
    got = offsets(record(lambda: work(2)), work)
    assert got == [
        ("call", 0),
        ("line", 1),
        ("line", 2),
        ("line", 3),
        ("line", 2),
        ("line", 3),
        ("line", 2),
        ("line", 4),
        ("return", 4),
    ], got


def resuming_a_generator_emits_no_line_event_for_the_yield_line():
    def drain():
        for _ in counting(3):
            pass

    got = offsets(record(drain), counting)
    assert got == [
        ("call", 0),
        ("line", 1),
        ("line", 2),
        ("line", 3),
        ("line", 4),
        ("return", 4),
        ("call", 4),
        ("line", 2),
        ("line", 3),
        ("line", 4),
        ("return", 4),
        ("call", 4),
        ("line", 2),
        ("line", 3),
        ("line", 4),
        ("return", 4),
        ("call", 4),
        ("line", 2),
        ("line", 5),
        ("return", 5),
    ], got


def an_unstarted_generator_still_reports_its_def_line():
    unstarted = counting(1)
    assert unstarted.gi_frame.f_lineno == counting.__code__.co_firstlineno, (
        unstarted.gi_frame.f_lineno
    )
    unstarted.close()


entering_a_function_emits_no_line_event_for_the_def_line()
resuming_a_generator_emits_no_line_event_for_the_yield_line()
an_unstarted_generator_still_reports_its_def_line()
print("OK")
