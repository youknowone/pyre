# CPython-suite gap: builtin exception tests omit hot specialized context chains.
# parity-tests reason: this guards pyre's JIT builtin raising/resume path.

"""Compiled builtin raising arms retain the active exception context."""

ROUNDS = 10000


def float_leaf():
    return 1.0 / 0.0


def int_leaf():
    return 7 // 0


def check_context(leaf):
    outer = KeyError("outer")
    try:
        raise outer
    except KeyError:
        for _ in range(ROUNDS):
            try:
                leaf()
            except ZeroDivisionError as exc:
                assert exc.__context__ is outer, (exc.__context__, outer)


check_context(float_leaf)
check_context(int_leaf)

print("OK")
