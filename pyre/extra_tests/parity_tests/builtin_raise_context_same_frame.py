# CPython-suite gap: builtin exception tests omit hot in-frame context chains.
# parity-tests reason: this guards pyre's JIT builtin raising path when the
# raise and its handler share one frame, so no callee frame is unwound.

"""A compiled builtin raise retains the active exception context.

`builtin_raise_context_specialization` raises inside a called leaf; here the
raising op and its handler share the loop frame, which is the shape where the
compiled arm delivers the exception with a traceback entry already stamped.

The counts are compared only after the loop: reading `__context__` inside an
`assert` keeps the loop from compiling at all (`loops_compiled=0`), which would
leave the compiled path untested.
"""

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


subscript = count_subscript()
floordiv = count_floordiv(0)
attribute = count_attribute()

assert subscript == 0, f"subscript lost {subscript}/{ROUNDS}"
assert floordiv == 0, f"floordiv lost {floordiv}/{ROUNDS}"
assert attribute == 0, f"attribute lost {attribute}/{ROUNDS}"

print("OK")
