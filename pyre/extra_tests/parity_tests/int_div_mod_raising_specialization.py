"""Guarded raising arms for exact-int ``//`` and ``%``.

Each traceback check compares a cold execution with the same source site after
its loop is hot, so frame names and line numbers cover both interpreted and
compiled paths without baking source coordinates into the fixture.  The phase
tests reuse one operator site with nonzero, zero, then nonzero divisors to
exercise both directions across the divisor guard and its bridge.
"""

ROUNDS = 6000
MESSAGE = "division by zero"


def signature(exc):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    return type(exc) is ZeroDivisionError, str(exc), exc.args, tuple(frames)


def same_floor(rounds):
    found = None
    zero = 0
    for i in range(rounds):
        try:
            i // zero
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def same_mod(rounds):
    found = None
    zero = 0
    for i in range(rounds):
        try:
            i % zero
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def floor_leaf(value, divisor):
    return value // divisor


def mod_leaf(value, divisor):
    return value % divisor


def one_frame(rounds, leaf):
    found = None
    for i in range(rounds):
        try:
            leaf(i, 0)
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def floor_middle_1(value, divisor):
    return floor_leaf(value, divisor)


def floor_middle_2(value, divisor):
    return floor_middle_1(value, divisor)


def mod_middle_1(value, divisor):
    return mod_leaf(value, divisor)


def mod_middle_2(value, divisor):
    return mod_middle_1(value, divisor)


def three_frames(rounds, leaf):
    found = None
    for i in range(rounds):
        try:
            leaf(i, 0)
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def flip_floor():
    raised = 0
    completed = 0
    checksum = 0
    for i in range(ROUNDS):
        divisor = 7 if i < 2000 or i >= 4000 else 0
        try:
            checksum += i // divisor
            completed += 1
        except ZeroDivisionError:
            raised += 1
    return raised, completed, checksum


def flip_mod():
    raised = 0
    completed = 0
    checksum = 0
    for i in range(ROUNDS):
        divisor = 7 if i < 2000 or i >= 4000 else 0
        try:
            checksum += i % divisor
            completed += 1
        except ZeroDivisionError:
            raised += 1
    return raised, completed, checksum


def check_signature(cold, hot, expected_names):
    assert cold == hot, (cold, hot)
    exact_type, message, args, frames = hot
    assert exact_type
    assert message == MESSAGE, message
    assert args == (MESSAGE,), args
    assert tuple(name for name, _ in frames) == expected_names, frames
    assert all(lineno > 0 for _, lineno in frames), frames


cold_same_floor = same_floor(1)
cold_same_mod = same_mod(1)
cold_one_floor = one_frame(1, floor_leaf)
cold_one_mod = one_frame(1, mod_leaf)
cold_three_floor = three_frames(1, floor_middle_2)
cold_three_mod = three_frames(1, mod_middle_2)

check_signature(cold_same_floor, same_floor(ROUNDS), ("same_floor",))
check_signature(cold_same_mod, same_mod(ROUNDS), ("same_mod",))
check_signature(cold_one_floor, one_frame(ROUNDS, floor_leaf), ("one_frame", "floor_leaf"))
check_signature(cold_one_mod, one_frame(ROUNDS, mod_leaf), ("one_frame", "mod_leaf"))
check_signature(
    cold_three_floor,
    three_frames(ROUNDS, floor_middle_2),
    ("three_frames", "floor_middle_2", "floor_middle_1", "floor_leaf"),
)
check_signature(
    cold_three_mod,
    three_frames(ROUNDS, mod_middle_2),
    ("three_frames", "mod_middle_2", "mod_middle_1", "mod_leaf"),
)

floor_phase = flip_floor()
mod_phase = flip_mod()
assert floor_phase[:2] == (2000, 4000), floor_phase
assert mod_phase[:2] == (2000, 4000), mod_phase
print("OK")
