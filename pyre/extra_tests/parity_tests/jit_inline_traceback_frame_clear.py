"""Completed JIT frames are clearable after return or exception unwind."""

import sys


def returns_frame():
    return sys._getframe()


for _ in range(3_000):
    frame = returns_frame()
    assert frame.f_code is returns_frame.__code__
    frame.clear()


def raises():
    raise ValueError("expected")


for _ in range(3_000):
    try:
        raises()
    except ValueError as exc:
        traceback = exc.__traceback__
        while traceback.tb_next is not None:
            traceback = traceback.tb_next
        assert traceback.tb_frame.f_code is raises.__code__
        traceback.tb_frame.clear()
    else:
        raise AssertionError("raises() returned")
