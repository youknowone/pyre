"""Guarded raising arm for exact numeric true-division by zero."""

import sys


ROUNDS = 6000
if sys.implementation.name == "pypy":
    MESSAGE = "float division by zero"
else:
    MESSAGE = "division by zero"


def signature(exc):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    return (
        type(exc) is ZeroDivisionError,
        str(exc),
        exc.args,
        exc.__suppress_context__,
        exc.__context__,
        tuple(frames),
    )


def same_frame(rounds, divisor):
    found = None
    for i in range(rounds):
        try:
            1.5 * i / divisor
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def div_leaf(value, divisor):
    return value / divisor


def one_frame(rounds):
    found = None
    for i in range(rounds):
        try:
            div_leaf(i + 0.5, 0.0)
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def middle_1(value, divisor):
    return div_leaf(value, divisor)


def middle_2(value, divisor):
    return middle_1(value, divisor)


def three_frames(rounds):
    found = None
    for i in range(rounds):
        try:
            middle_2(i + 0.5, 0.0)
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def flip_divisor():
    raised = 0
    completed = 0
    total = 0.0
    for i in range(ROUNDS):
        divisor = 0.0 if i < 2000 or i >= 4000 else 1.5
        try:
            total += (i + 0.5) / divisor
            completed += 1
        except ZeroDivisionError:
            raised += 1
    return raised, completed, round(total, 4)


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


check_signature(same_frame(1, 0.0), same_frame(ROUNDS, 0.0), ("same_frame",))
check_signature(same_frame(1, -0.0), same_frame(ROUNDS, -0.0), ("same_frame",))
check_signature(one_frame(1), one_frame(ROUNDS), ("one_frame", "div_leaf"))
check_signature(
    three_frames(1),
    three_frames(ROUNDS),
    ("three_frames", "middle_2", "middle_1", "div_leaf"),
)

phase = flip_divisor()
assert phase[:2] == (4000, 2000), phase

print("OK")
