# pyre-check: max-pypy-ratio=2.6
# Run 33300212586, dynasm and cranelift over all three hosts: 0.6-1.9x.  Twice
# the slowest would be 3.8, but `PERF_GATE_FLOOR_DIVISOR` derives the floor
# from this same number and 3.8/6 sits above the 0.6x windows reads, so the
# ceiling is placed between the two bounds instead: 1.37x above the slowest
# reading, and its derived 0.43x floor 1.39x below the fastest.
# pyre-check: skip-cpython
# cpython 1.08s vs pyre 0.32s (3.4x), and it is not gated on — only pypy is.

# Overflow-crossing int multiply on a JIT-hot path. The inner loop is traced
# while `scale` is small (a*a stays in machine-int range, so the recorded
# GUARD_NO_OVERFLOW passes), then a large `scale` makes a*a overflow a 64-bit
# int and it must promote to a big int. A backend that drops the overflow check
# silently wraps the product instead of promoting, giving a wrong answer.
def hot(scale, n):
    acc = 0
    i = 0
    while i < n:
        a = scale + (i & 1)      # loop-variant: cannot fold to a constant
        acc = acc + a * a
        i = i + 1
    return acc


def main():
    warm = 0
    for _ in range(120):
        warm = warm + hot(3, 20000)      # a in {3,4}; a*a tiny, never overflows
    # Big scale: a ~ 5e9, a*a = 2.5e19 overflows int64 (and uint64) -> big int.
    # Sized so the promoting loop, not the warm-up, dominates the measurement:
    # at 20000 the whole run was pypy exec 0.01s against a 0.01s pypy startup
    # and a 0.08s dynasm one, so the ratio was startup noise rather than the
    # promotion this fixture is named for.
    print(hot(5000000000, 10000000))
    print(warm)


main()
