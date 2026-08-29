# pyre-check: gate=1
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
# The same function also excludes everything ahead of the first `RESUME`
# (`i < code->_co_firsttraceable`), which is the rest of the frame prologue:
# `COPY_FREE_VARS`, `MAKE_CELL`, and a generator's `RETURN_GENERATOR` /
# `POP_TOP` pair.  That pair carries the `def` line too, so a traced generator
# needs the prologue rule as well as the opcode one -- excluding `RESUME` alone
# only moved the spurious event one instruction earlier, which is what the
# generator case below caught.
#
# The generator half is also the other side of the `RESUME` rule.  A resumed
# generator re-enters at a `RESUME` carrying the `yield` line, so instrumenting
# it would report a line event for a line the frame is only passing back
# through.  Both references agree on the whole resumed stream, so it is pinned
# verbatim.
#
# IF YOU ARE COMPARING TOTAL EVENT COUNTS AGAINST CPYTHON, READ THIS FIRST.
# A total taken over a long traced loop wanders, and not because of event
# emission.  pyre's collection is deferred where CPython's is refcounted, so the
# `_ModuleLock` weakref callback `cb` (`_get_module_lock`, `lib-python/3/
# importlib/_bootstrap.py`) runs inside the traced region here while CPython had
# already collected those locks at import time, long before the tracer existed.
# Each such callback is a Python frame the tracer sees, so it contributes its
# own `call` + lines + `return`, and HOW MANY fire depends on the collection
# schedule -- two runs of one binary disagree.  Measured on
# `work(n): total = 0; for i in range(n): total += i; return total` traced
# whole: CPython 400005 at n=200000, which is exactly 2n+5, and pyre 400061 in
# the same shape, the difference being 8 `cb` frames at 7 events each.
#
# The instrument that does work: run the same probe at a scale where no `cb`
# fires -- n=50 and n=5000 both do -- and the total is exactly 2n+5, equal to
# CPython, with no excess and no deficit.  That is what shows a change to event
# emission removed exactly the events it meant to and no others; a large-n total
# cannot show it.  (pypy3 reads 598732 on the same probe, so a bare
# three-way total comparison says far less than it looks like it does.)
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


def entering_a_closure_emits_no_line_event_for_the_def_line():
    """The prologue ahead of `RESUME` is `COPY_FREE_VARS` / `MAKE_CELL` here.

    Both are stamped with no line at all rather than the `def` line, so this
    would pass even without the prologue rule -- it is here because the rule is
    what makes that a fact about the prologue rather than a fact about how one
    compiler happened to stamp two opcodes.
    """
    cell = 7

    def reads_the_cell(n):                                # +0
        total = cell                                      # +1
        return total + n                                  # +2

    got = offsets(record(lambda: reads_the_cell(1)), reads_the_cell)
    assert got == [
        ("call", 0),
        ("line", 1),
        ("line", 2),
        ("return", 2),
    ], got


def an_unstarted_generator_still_reports_its_def_line():
    unstarted = counting(1)
    assert unstarted.gi_frame.f_lineno == counting.__code__.co_firstlineno, (
        unstarted.gi_frame.f_lineno
    )
    unstarted.close()


entering_a_function_emits_no_line_event_for_the_def_line()
entering_a_closure_emits_no_line_event_for_the_def_line()
resuming_a_generator_emits_no_line_event_for_the_yield_line()
an_unstarted_generator_still_reports_its_def_line()
print("OK")
