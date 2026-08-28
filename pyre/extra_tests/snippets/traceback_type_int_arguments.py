# pyre-check: gate=1
# CPython-suite gap: `test_traceback`'s `TestTracebackType` builds nodes with
# plain in-range integers, so nothing in the suite reaches either edge of the
# `int` converter its signature declares -- neither an object that is only
# index-able, nor a value the C `int` cannot hold.
#
# parity-tests reason: `tb_lasti` and `tb_lineno` are declared `int`, so the
# converter Argument Clinic emits is `PyLong_AsInt`.  That reduces through
# `__index__`, which is why the rejection reads `'str' object cannot be
# interpreted as an integer` rather than a signature-shaped message, and it
# raises `OverflowError` rather than storing a value the slot cannot hold.  A
# runtime that tests `int` directly and stores a machine word gets all three
# answers wrong, and the stored value then reaches the traceback's line
# resolution as a byte offset no code object can have.
#
# PyPy 7.3.20 has no `TracebackType` constructor at all, so it fails here for
# the reason `traceback_lineno_sentinel.py` gives.

import types


def frame_of_a_raise():
    try:
        raise ValueError("probe")
    except ValueError as exc:
        return exc.__traceback__.tb_frame


FRAME = frame_of_a_raise()


class Indexable:
    def __index__(self):
        return 7


node = types.TracebackType(None, FRAME, Indexable(), Indexable())
print("index argument:", node.tb_lasti, node.tb_lineno)
# `True` is an `int` subclass and converts to 1 / 0.
node = types.TracebackType(None, FRAME, True, False)
print("bool argument:", node.tb_lasti, node.tb_lineno)
# The largest and smallest values the slot does hold.
node = types.TracebackType(None, FRAME, 2**31 - 1, -(2**31))
print("at the edge:", node.tb_lasti, node.tb_lineno)

for argument in (2**31, -(2**31) - 1):
    for position in (0, 1):
        arguments = [1, 1]
        arguments[position] = argument
        try:
            types.TracebackType(None, FRAME, *arguments)
        except BaseException as exc:
            print("out of range:", type(exc).__name__, exc)
        else:
            print("out of range: accepted")

for argument in ("x", 1.0, None):
    try:
        types.TracebackType(None, FRAME, argument, 1)
    except BaseException as exc:
        print("not an integer:", type(exc).__name__, exc)
    else:
        print("not an integer: accepted")

print("OK")
