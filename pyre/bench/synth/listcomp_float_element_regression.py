# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=warm
# A list comprehension over `range` whose element is a float, run until the
# JIT compiles the loop.
#
# `jtransform.py` `_rewrite_equality` folds `int_eq(x, 0)` into the unary
# `int_is_zero(x)`, so the interpreter's own jitcode carries that opname
# wherever it tests an integer against zero -- 1232 bodies.  `pyjitpl.py`
# generates the integer unary opimpls from one list, `int_is_true` /
# `int_is_zero` / `int_neg` / `int_invert`; the walker's table here had
# every entry of that list except `int_is_zero`, and a key the table does
# not name is `DispatchError::UnsupportedOpname` -- the trace aborts part
# way through the body it was walking rather than declining up front.
#
# The observable is this loop dying with
# `TypeError: 'float' object is not an iterator`: FOR_ITER reads the
# element where the iterator belongs.  Only a float element reaches it --
# `x`, `x + 1` and `x * 2` all pass, so a fixture that produced ints would
# have gone green against the same defect.  `PYRE_NO_JIT=1` passes too.
#
# Every expectation below is the value CPython 3.14 and PyPy both produce.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

failures = []

SIZES = list(range(30)) + [2000]


def listcomp_mul(n):
    return [x * 1.0 for x in range(n)]


def listcomp_const_lhs(n):
    return [1.0 * x for x in range(n)]


def listcomp_truediv(n):
    return [x / 1 for x in range(n)]


def listcomp_call(n):
    return [float(x) for x in range(n)]


def check(label, fn):
    for n in SIZES:
        try:
            got = fn(n)
        except TypeError as exc:
            failures.append(f"{label} n={n}: raised TypeError: {exc}")
            return
        if len(got) != n:
            failures.append(f"{label} n={n}: len {len(got)} != {n}")
            return
        if n and got[-1] != float(n - 1):
            failures.append(f"{label} n={n}: last {got[-1]!r} != {float(n - 1)!r}")
            return


def warm(rounds):
    """The hot loop the JIT compiles; each round re-enters the window."""
    total = 0.0
    for _ in range(rounds):
        total += sum([x * 1.0 for x in range(64)])
    return total


def main():
    check("listcomp-mul", listcomp_mul)
    check("listcomp-const-lhs", listcomp_const_lhs)
    check("listcomp-truediv", listcomp_truediv)
    check("listcomp-call", listcomp_call)

    expected = 400 * 2016.0
    try:
        warmed = warm(400)
    except TypeError as exc:
        failures.append(f"warm loop: raised TypeError: {exc}")
    else:
        if warmed != expected:
            failures.append(f"warm loop: {warmed!r} != {expected!r}")

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS listcomp float element")
    return 0


raise SystemExit(main())
