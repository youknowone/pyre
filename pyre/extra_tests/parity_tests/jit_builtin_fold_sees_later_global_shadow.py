# CPython-suite gap: no suite test compiles a builtin name then inserts a
# same-name global that must shadow the baked builtin.
# parity-tests reason: LOAD_GLOBAL builtins fallback must pin the module-dict
# version? so a later insert fails GUARD_NOT_INVALIDATED.

"""A compiled builtin fold sees a later same-name global insert."""

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


def hot(x):
    total = 0
    i = 0
    while i < 200:
        total += len(x)
        i += 1
    return total


def shadow():
    globals()["len"] = lambda x: -1


xs = [0, 1, 2]
assert hot(xs) == 600, hot(xs)
shadow()
assert hot(xs) == -200, hot(xs)
print("OK")
