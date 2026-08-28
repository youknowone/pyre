# CPython-suite gap: range tests do not compare cold/hot zero-step exceptions.
# parity-tests reason: this guards pyre's JIT range raising specialization.

"""Guarded raising arm for an exact ``range(..., step=0)`` call."""

import sys


ROUNDS = 6000
if sys.implementation.name in ("pyre", "pypy"):
    MESSAGE = "step argument must not be zero"
else:
    MESSAGE = "range() arg 3 must not be zero"


def signature(exc):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    return (
        type(exc) is ValueError,
        str(exc),
        exc.args,
        exc.__suppress_context__,
        exc.__context__,
        tuple(frames),
    )


def same_frame(rounds, stop):
    found = None
    for _ in range(rounds):
        try:
            range(0, stop, 0)
        except ValueError as exc:
            found = signature(exc)
    return found


def range_leaf(stop, step):
    return range(0, stop, step)


def one_frame(rounds):
    found = None
    for _ in range(rounds):
        try:
            range_leaf(3, 0)
        except ValueError as exc:
            found = signature(exc)
    return found


def middle_1(stop, step):
    return range_leaf(stop, step)


def middle_2(stop, step):
    return middle_1(stop, step)


def three_frames(rounds):
    found = None
    for _ in range(rounds):
        try:
            middle_2(3, 0)
        except ValueError as exc:
            found = signature(exc)
    return found


def flip_step():
    raised = 0
    completed = 0
    checksum = 0
    for i in range(ROUNDS):
        step = 0 if i < 2000 or i >= 4000 else 2
        try:
            value = range(0, 9, step)
            checksum += len(value)
            completed += 1
        except ValueError:
            raised += 1
    return raised, completed, checksum


def check_signature(cold, hot, expected_names):
    assert cold == hot, (cold, hot)
    exact_type, message, args, suppress, context, frames = hot
    assert exact_type
    assert message == MESSAGE, message
    assert args == (MESSAGE,), args
    assert suppress is False, suppress
    assert context is None, context
    assert tuple(name for name, _ in frames) == expected_names, frames
    assert all(lineno > 0 for _, lineno in frames), frames


check_signature(same_frame(1, 3), same_frame(ROUNDS, 3), ("same_frame",))
check_signature(same_frame(1, 0), same_frame(ROUNDS, 0), ("same_frame",))
check_signature(one_frame(1), one_frame(ROUNDS), ("one_frame", "range_leaf"))
check_signature(
    three_frames(1),
    three_frames(ROUNDS),
    ("three_frames", "middle_2", "middle_1", "range_leaf"),
)

phase = flip_step()
assert phase == (4000, 2000, 10000), phase

# Successful negative and machine-word-boundary steps remain ordinary ranges.
assert list(range(3, -4, -2)) == [3, 1, -1, -3]
assert list(range(sys.maxsize, sys.maxsize - 3, -1)) == [
    sys.maxsize,
    sys.maxsize - 1,
    sys.maxsize - 2,
]
assert list(range(-sys.maxsize - 1, sys.maxsize, sys.maxsize)) == [
    -sys.maxsize - 1,
    -1,
    sys.maxsize - 1,
]

print("OK")
