"""Guarded raising arms for exact-bigint shifts by a negative exact int."""

import sys


ROUNDS = 6000
BIG = (1 << 100) + 123
MESSAGE = "negative shift count"


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


def apply(value, count, use_left):
    if use_left:
        return value << count
    return value >> count


def same_frame(rounds, use_left, count):
    found = None
    for i in range(rounds):
        try:
            if use_left:
                (BIG + i) << count
            else:
                (BIG + i) >> count
        except ValueError as exc:
            found = signature(exc)
    return found


def one_frame(rounds, use_left):
    found = None
    for i in range(rounds):
        try:
            apply(BIG + i, -1, use_left)
        except ValueError as exc:
            found = signature(exc)
    return found


def middle_1(value, count, use_left):
    return apply(value, count, use_left)


def middle_2(value, count, use_left):
    return middle_1(value, count, use_left)


def three_frames(rounds, use_left):
    found = None
    for i in range(rounds):
        try:
            middle_2(BIG + i, -1, use_left)
        except ValueError as exc:
            found = signature(exc)
    return found


def flip_count(use_left):
    raised = 0
    completed = 0
    checksum = 0
    for i in range(ROUNDS):
        count = -1 if i < 2000 or i >= 4000 else 2
        try:
            checksum += apply(BIG + i, count, use_left)
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


for use_left in (True, False):
    check_signature(
        same_frame(1, use_left, -1),
        same_frame(ROUNDS, use_left, -1),
        ("same_frame",),
    )
    check_signature(
        same_frame(1, use_left, -sys.maxsize - 1),
        same_frame(ROUNDS, use_left, -sys.maxsize - 1),
        ("same_frame",),
    )
    check_signature(
        one_frame(1, use_left),
        one_frame(ROUNDS, use_left),
        ("one_frame", "apply"),
    )
    check_signature(
        three_frames(1, use_left),
        three_frames(ROUNDS, use_left),
        ("three_frames", "middle_2", "middle_1", "apply"),
    )
    phase = flip_count(use_left)
    assert phase[:2] == (4000, 2000), phase

print("OK")
