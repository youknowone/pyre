# pyre-check: max-pypy-ratio=18
# The fixture predates the ratio-gate convention and carried no ceiling. It
# compiles two loops and, at this trip count, pypy's execution clears the
# startup-subtraction floor, so a ratio here is a measurement of generated
# code rather than of two interpreters' startup. The ceiling is twice the
# slowest of the three backends observed (8.6x on wasm).
# Regression oracle for the #14 inline-frame heap-store double-commit via the
# loop-bearing-callee path. A loop-bearing callee that mutates a caller-owned
# heap object inside its own loop double-commits the mutation when the outer
# loop is JIT-compiled: the callee's traced iteration is committed concretely
# during recording AND re-applied at the trace->compile boundary.
#
# Expected: len(acc) == 2 * N. Under the bug the JIT printed 2*N + 3 (a constant
# over-count, independent of N, present only once N crosses the compile
# threshold), on both backends, which share the trace/resume layer.
N = 1200000


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
