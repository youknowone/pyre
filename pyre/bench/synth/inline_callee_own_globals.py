# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=drive
# A function executing a code object under a globals dictionary distinct from
# the portal's must keep that dictionary when its body is inlined.  This is the
# observable form of PyPy's one-red-frame-per-MIFrame rule: LOAD_GLOBAL reads
# the callee frame's `w_globals`, never the outer portal frame's.
import types

OUTER_VALUE = 7


def template(i):
    return OUTER_VALUE + i


callee_globals = {
    "OUTER_VALUE": 1000,
    "__builtins__": __builtins__,
}
callee = types.FunctionType(template.__code__, callee_globals, "callee")


def drive(n):
    total = 0
    i = 0
    while i < n:
        total += callee(i)
        i += 1
    return total


def main():
    n = 5000
    got = drive(n)
    want = n * 1000 + n * (n - 1) // 2
    if got != want:
        print("FAIL own globals", got, want)
        return 1
    print("PASS inline callee own globals", got)
    return 0


raise SystemExit(main())
