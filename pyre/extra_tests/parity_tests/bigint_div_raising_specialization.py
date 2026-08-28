# CPython-suite gap: bigint division tests do not compare cold/hot raising traces.
# parity-tests reason: this guards pyre's JIT exact-bigint raising specialization.

"""Guarded raising arms for exact-bigint ``//`` and ``%`` by exact zero."""

import sys


ROUNDS = 6000
BIG = (1 << 100) + 123
if sys.implementation.name == "pypy":
    FLOOR_MESSAGE = "integer division or modulo by zero"
    MOD_MESSAGE = "long division or modulo by zero"
else:
    FLOOR_MESSAGE = "division by zero"
    MOD_MESSAGE = "division by zero"


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


def apply(value, divisor, use_mod):
    if use_mod:
        return value % divisor
    return value // divisor


def same_frame(rounds, use_mod):
    found = None
    for i in range(rounds):
        try:
            if use_mod:
                (BIG + i) % 0
            else:
                (BIG + i) // 0
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def one_frame(rounds, use_mod):
    found = None
    for i in range(rounds):
        try:
            apply(BIG + i, 0, use_mod)
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def middle_1(value, divisor, use_mod):
    return apply(value, divisor, use_mod)


def middle_2(value, divisor, use_mod):
    return middle_1(value, divisor, use_mod)


def three_frames(rounds, use_mod):
    found = None
    for i in range(rounds):
        try:
            middle_2(BIG + i, 0, use_mod)
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def flip_divisor(use_mod):
    raised = 0
    completed = 0
    checksum = 0
    for i in range(ROUNDS):
        divisor = 0 if i < 2000 or i >= 4000 else 7
        try:
            checksum += apply(BIG + i, divisor, use_mod)
            completed += 1
        except ZeroDivisionError:
            raised += 1
    return raised, completed, checksum


def check_signature(cold, hot, message, expected_names):
    assert cold == hot, (cold, hot)
    exact_type, actual_message, args, suppress, context, frames = hot
    assert exact_type
    assert actual_message == message, actual_message
    assert args == (message,), args
    assert suppress is False, suppress
    assert context is None, context
    assert tuple(name for name, _ in frames) == expected_names, frames
    assert all(lineno > 0 for _, lineno in frames), frames


for use_mod, message in ((False, FLOOR_MESSAGE), (True, MOD_MESSAGE)):
    check_signature(
        same_frame(1, use_mod),
        same_frame(ROUNDS, use_mod),
        message,
        ("same_frame",),
    )
    check_signature(
        one_frame(1, use_mod),
        one_frame(ROUNDS, use_mod),
        message,
        ("one_frame", "apply"),
    )
    check_signature(
        three_frames(1, use_mod),
        three_frames(ROUNDS, use_mod),
        message,
        ("three_frames", "middle_2", "middle_1", "apply"),
    )
    phase = flip_divisor(use_mod)
    assert phase[:2] == (4000, 2000), phase

print("OK")
