# pyre-check: max-pypy-ratio=30
# Property getter/setter exceptions and __getattr__ hook exceptions
# propagate out of attribute access instead of being swallowed. Only the
# exception type is printed so the line matches across CPython/PyPy/Pyre.
# The raising getter must stay inside the compiled loop: leaving its fget as an
# opaque residual measured at least 31-37x PyPy in CI.  Keep the hot loop long
# enough that PyPy clears check.py's timing floor and the ratio ceiling is
# enforced rather than displayed as an informational lower bound.

N = 20000000


def show(label, fn):
    try:
        fn()
        print(label, "NO-RAISE")
    except Exception as e:
        print(label, type(e).__name__)


class PropRaises:
    @property
    def g(self):
        raise ValueError("getter")

    @g.setter
    def g(self, v):
        raise KeyError("setter")


class HookRaises:
    def __getattr__(self, name):
        raise RuntimeError("hook")


def main():
    p = PropRaises()
    h = HookRaises()

    # Hot loop: the JIT-compiled getattr path must surface the getter's
    # ValueError every iteration rather than swallowing it (which would
    # have returned a null/None-ish value and left `caught` at 0).
    caught = 0
    i = 0
    while i < N:
        try:
            p.g
        except ValueError:
            caught = caught + 1
        i = i + 1
    print("getter_caught", caught)

    # A getter exception propagates from a plain attribute read.
    show("getter", lambda: p.g)

    # A setter exception propagates from a plain attribute store rather
    # than being swallowed into a silent success.
    def set_g():
        p.g = 1

    show("setter", set_g)

    # A __getattr__ hook that raises a non-AttributeError propagates that
    # exception rather than being masked as AttributeError.
    show("hook", lambda: h.missing)


main()
