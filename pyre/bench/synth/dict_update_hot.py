# pyre-check: max-pypy-ratio=15
# The ceiling sits between the two measured states: folded this runs 11.4x
# pypy, and with `builtin_dict_get` suppressed about 20.2x. That 1.8x gap is
# the whole `builtin_dict_get` effect, so the margin either side is only
# ~1.3x -- loosen this ceiling rather than delete the loop if a slower host
# proves it flaky.
# N/ITERS are sized to prove the opcode compiles and to drive the compiled
# loop thousands of times, not to race pypy.
# A hot exact `dict.get` loop rides along: without the `builtin_dict_get` fold
# it measures 2.2x on its own (0.686s -> 1.504s).
N = 300
ITERS = 500


def run(n, extra):
    # `{"k": i, **extra}` compiles to BUILD_MAP + DICT_UPDATE in a
    # while-loop body.  Before DICT_UPDATE was lowered, its abort_permanent
    # marker declined the whole loop.
    total = 0
    i = 0
    while i < n:
        d = {"k": i, **extra}
        total += d["k"] + d["x"]
        i += 1
    return total


def main():
    extra = {"x": 7}
    total = 0
    for _ in range(ITERS):
        total += run(N, extra)
    print(total)


main()


GETD = {"a": 1, "b": 2, "c": 3}


def hot_dict_get(n):
    """Hot exact `dict.get`, the `builtin_dict_get` fold."""
    s = 0
    i = 0
    while i < n:
        s += GETD.get("a", 0)
        i += 1
    return s


print(hot_dict_get(20000000))
