# pyre-check: max-pypy-ratio=30

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
    print(hot(5000000000, 20000))
    print(warm)


main()
