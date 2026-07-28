# Regression oracle for the #14 inline-frame heap-store double-commit via the
# loop-bearing-callee path. A loop-bearing callee that mutates a caller-owned
# heap object inside its own loop double-commits the mutation when the outer
# loop is JIT-compiled: the callee's traced iteration is committed concretely
# during recording AND re-applied at the trace->compile boundary.
#
# Expected: len(acc) == 2 * N. Under the bug the JIT printed 2*N + 3 (a constant
# over-count, independent of N, present only once N crosses the compile
# threshold), on both backends, which share the trace/resume layer.
N = 20000


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
