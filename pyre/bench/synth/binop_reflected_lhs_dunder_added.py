# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=go
# A BINARY_OP whose left operand is an int subclass and whose right operand
# is a user instance.  The subclass inherits int's builtin forward slot, so
# the protocol runs `R.__radd__` until a forward dunder is assigned onto the
# lhs class.  That assignment replaces the inherited slot and must deopt the
# compiled skip; a bare `int` on the left cannot reach this shape because
# the builtin type is immutable.
#
# Deterministic, terminating, prints PASS or a FAIL naming the site.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 4000


class L(int):
    pass


class R:
    def __radd__(self, o):
        return "reflected"


def go(n, a, b):
    out = None
    for _ in range(n):
        out = a + b
    return out


def main():
    a = L(3)
    b = R()
    warm = go(N, a, b)
    if warm != "reflected":
        print(f"FAIL warm-up gave {warm}")
        return 1
    L.__add__ = lambda self, o: "forward"
    after = go(N, a, b)
    if after != "forward":
        print(f"FAIL after mutation gave {after}")
        return 1
    print("PASS lhs subclass dunder added after compile")
    return 0


import sys

sys.exit(main())
