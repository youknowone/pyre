# pyre-check: gate=1
# CPython-suite gap: the suite reads `f_lasti` only off frames that have run an
# instruction -- `test.test_sys_settrace`, which is the module that would see a
# frame at its `call` event, is SKIP in `pyre/cpython_tests/baseline.json`
# ("implementation detail").  A wrong value on a not-yet-started frame is also
# the kind of defect that survives every consumer that does `f_lasti // 2`,
# because the quotient of a negative sentinel still looks negative.
#
# parity-tests reason: pyre stores `last_instr` in instruction units and
# CPython's `f_lasti` is a byte offset, so the getset scales by two.  A negative
# `last_instr` is the not-yet-started sentinel rather than a coordinate, and
# scaling it produced `-2`, a value NEITHER reference can report.  Both spell
# the sentinel exactly `-1`: `PyFrame_GetLasti` (`Objects/frameobject.c`) is
# `lasti < 0 ? -1 : lasti * sizeof(_Py_CODEUNIT)`, `frame_lasti_get_impl`
# returns `-1` for the same case, and `pyframe.py fget_f_lasti` hands
# `last_instr` back unscaled.
#
# What is asserted is therefore the invariant both references satisfy -- a
# negative `f_lasti` is exactly `-1` -- rather than a literal value.  The two
# disagree on WHETHER a given frame is negative, because a CPython frame starts
# past the `RESUME` it counts as already executed while a pypy/pyre frame starts
# at `-1`; that difference is the frame model and is not what this pins.
import sys


def unstarted_generator_frame_lasti():
    def counting():
        yield 1

    generator = counting()
    try:
        return generator.gi_frame.f_lasti
    finally:
        generator.close()


def call_event_frame_lasti():
    seen = []

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "probe":
            seen.append(frame.f_lasti)
        return tracer

    def probe():
        return 1

    sys.settrace(tracer)
    try:
        probe()
    finally:
        sys.settrace(None)
    assert len(seen) == 1, seen
    return seen[0]


for label, lasti in (
    ("unstarted generator", unstarted_generator_frame_lasti()),
    ("call event", call_event_frame_lasti()),
):
    assert lasti >= 0 or lasti == -1, (label, lasti)

print("OK")
