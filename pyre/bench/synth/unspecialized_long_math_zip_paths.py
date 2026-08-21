# pyre-check: max-pypy-ratio=6
# The throughput gate for eight hand-written trace-time folds that no other
# fixture makes fire: `zip` over two tuples (positional and `strict=True`), a
# long/int comparison, a two-bigint true-divide, `bigint ** int`,
# `bigint << int` / `bigint >> int`, and `math.frexp` / `math.ldexp`.
#
# The ceiling is what does the gating: measured 2.59x pypy with the eight folds
# in place and 12.4x with all eight suppressed, so 6 sits about 2.2x from
# either state. A jit-stats baseline cannot see a throughput change -- that is
# how these eight were once retired as dead -- so this ceiling is the only
# thing here that can. Loosen it rather than shrink the loops if a slower host
# proves it flaky.
#
# The legs are sized per leg, not uniformly, because sensitivity and cost run
# opposite here: measured at a common 1M iterations, the folds worth the most
# sit on the cheapest legs (frexp 23.5x, ldexp 11x, long/int compare 8.5x, each
# ~4ms) while the dearest legs are the least sensitive (zip 1.05s at 1.1x,
# zip_strict 1.77s at 1.4x). Uniform sizing lets the insensitive legs drown the
# rest and the whole fixture moves only 1.3x when all eight folds are
# suppressed. Sized so each leg contributes comparable folded time, the same
# suppression moves it several-fold, which is what the ceiling can gate.
# Every leg is also large enough that pypy's own execution clears the
# measurement floor; below that the ratio gate divides by the floor and reads
# startup instead of these loops.
#
# The frexp and ldexp legs are held small for a second reason: they are the two
# shapes wasm is worst at relative to dynasm, ~37x each against 1.3-4.4x for
# every other leg, so sizing them by sensitivity alone puts the fixture through
# the wasm/dynasm ceiling (measured 9.0x against a gate of 4x). The long/int
# compare carries their share instead -- 8.5x sensitivity at 3.8x on wasm is
# the best of both here.
#
# Every result is reduced to an integer checksum: the float legs go through
# `int(round(x * 1e6))` so the comparison cannot turn into a float-formatting
# difference between the three backends.

import math

N_ZIP = 50000
N_ZIP_STRICT = 30000
N_CMP = 40000000
N_TRUEDIV = 700000
N_POW = 125000
N_SHIFT = 600000
N_FREXP = 1000000
N_LDEXP = 1000000


def f_zip(n):
    t1 = (1, 2, 3, 4, 5, 6, 7, 8)
    t2 = (10, 20, 30, 40, 50, 60, 70, 80)
    s = 0
    for _ in range(n):
        for a, b in zip(t1, t2):
            s += a * b
    return s


def f_zip_strict(n):
    t1 = (1, 2, 3, 4, 5, 6, 7, 8)
    t2 = (10, 20, 30, 40, 50, 60, 70, 80)
    s = 0
    i = 0
    while i < n:
        for a, b in zip(t1, t2, strict=True):
            s += a + b
        i += 1
    return s


def f_cmp_long_int(n):
    big = 1 << 200
    s = 0
    i = 0
    while i < n:
        if big > i:
            s += 1
        if i < big:
            s += 2
        i += 1
    return s


def f_truediv_long(n):
    big = 1 << 200
    s = 0.0
    i = 1
    while i <= n:
        s += big / (big + i)
        i += 1
    return s


def f_pow_long_int(n):
    base = 1 << 70
    s = 0
    i = 0
    while i < n:
        s += ((base + i) ** 2) % 1000003
        i += 1
    return s


def f_shift_long_int(n):
    big = 1 << 200
    s = 0
    i = 0
    while i < n:
        s += ((big + i) >> 3) % 1000003
        s += ((big + i) << 3) % 1000003
        i += 1
    return s


def f_frexp(n):
    s = 0.0
    i = 1
    while i <= n:
        m, e = math.frexp(i * 1.5)
        s += m + e
        i += 1
    return s


def f_ldexp(n):
    s = 0.0
    i = 1
    while i <= n:
        s += math.ldexp(0.5, i % 8)
        i += 1
    return s


print("zip       ", f_zip(N_ZIP))
print("zip_strict", f_zip_strict(N_ZIP_STRICT))
print("cmp_long  ", f_cmp_long_int(N_CMP))
print("truediv   ", int(round(f_truediv_long(N_TRUEDIV) * 1e6)))
print("pow       ", f_pow_long_int(N_POW))
print("shift     ", f_shift_long_int(N_SHIFT))
print("frexp     ", int(round(f_frexp(N_FREXP) * 1e6)))
print("ldexp     ", int(round(f_ldexp(N_LDEXP) * 1e6)))
