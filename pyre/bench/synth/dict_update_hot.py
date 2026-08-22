# pyre-check: max-pypy-ratio=20
# The ceiling sits between the two measured states. Re-measured 2026-08-22 on
# darwin-arm64, user CPU less startup, median of 3: pypy 0.169s, folded 1.996s
# (11.8x), and with `builtin_dict_get` suppressed 4.596s (27.2x). The fold is
# worth 2.3x, wider than the 1.8x first recorded here, so suppressing it stays
# far outside the ceiling.
# The ceiling moved 15 -> 20 because the ubuntu-24.04 runner measures the
# FOLDED state at 14.6x, 15.5x and 15.6x across three runs -- it straddled the
# old ceiling and flipped the gate run to run while the fold was working. 20
# clears the worst folded reading by 1.25x and stays 1.36x under the suppressed
# state, which restores the margin the old ceiling had against the host it was
# first measured on.
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
