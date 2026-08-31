# CPython-suite gap: builtin exception tests omit hot in-frame context chains.
# parity-tests reason: this guards pyre's JIT builtin raising path when the
# raise and its handler share one frame, so no callee frame is unwound.

"""Compiled builtin raises retain context both in-frame and through a leaf."""

ROUNDS = 20000


def count_subscript():
    outer = KeyError("outer")
    lost = 0
    try:
        raise outer
    except KeyError:
        for _ in range(ROUNDS):
            try:
                [][0]
            except IndexError as exc:
                if exc.__context__ is not outer:
                    lost += 1
    return lost


def count_floordiv(zero):
    outer = KeyError("outer")
    lost = 0
    try:
        raise outer
    except KeyError:
        for i in range(ROUNDS):
            try:
                i // zero
            except ZeroDivisionError as exc:
                if exc.__context__ is not outer:
                    lost += 1
    return lost


def count_attribute():
    outer = KeyError("outer")
    obj = object()
    lost = 0
    try:
        raise outer
    except KeyError:
        for _ in range(ROUNDS):
            try:
                obj.missing
            except AttributeError as exc:
                if exc.__context__ is not outer:
                    lost += 1
    return lost


def float_leaf():
    return 1.0 / 0.0


def int_leaf():
    return 7 // 0


def count_leaf(leaf):
    outer = KeyError("outer")
    lost = 0
    try:
        raise outer
    except KeyError:
        for _ in range(ROUNDS):
            try:
                leaf()
            except ZeroDivisionError as exc:
                lost += exc.__context__ is not outer
    return lost


subscript = count_subscript()
floordiv = count_floordiv(0)
attribute = count_attribute()
leaf = count_leaf(float_leaf) + count_leaf(int_leaf)

assert subscript == 0, f"subscript lost {subscript}/{ROUNDS}"
assert floordiv == 0, f"floordiv lost {floordiv}/{ROUNDS}"
assert attribute == 0, f"attribute lost {attribute}/{ROUNDS}"
assert leaf == 0, f"leaf calls lost {leaf}/{ROUNDS * 2}"

print("OK")
