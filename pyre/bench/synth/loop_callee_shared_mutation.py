# pyre-check: max-pypy-ratio=4.0
# The fixture compiles two loops, and at this trip count pypy's own execution
# is several times the startup-subtraction floor on every runner, so the ratio
# measures generated code rather than two interpreters' startup.
#
# The ceiling is not twice the slowest reading, the convention the other
# fixtures use.  The two backends read far enough apart on this shape that a
# ceiling twice the slower one derives a floor above what the faster one reads,
# so it is fitted between them instead: it clears the slowest reading by about
# a quarter, and the floor derived under it sits under the fastest by about a
# sixth.
# Regression oracle for the #14 inline-frame heap-store double-commit via the
# loop-bearing-callee path. A loop-bearing callee that mutates a caller-owned
# heap object inside its own loop double-commits the mutation when the outer
# loop is JIT-compiled: the callee's traced iteration is committed concretely
# during recording AND re-applied at the trace->compile boundary.
#
# Expected: len(acc) == 2 * N. Under the bug the JIT printed 2*N + 3 (a constant
# over-count, independent of N, present only once N crosses the compile
# threshold), on both backends, which share the trace/resume layer.
N = 2400000


def fill(out):
    j = 0
    while j < 2:
        out.append(j)
        j += 1


def main():
    acc = []
    i = 0
    while i < N:
        fill(acc)
        i += 1
    print(len(acc))


main()
