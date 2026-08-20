# The eight hand-written trace-time folds that no other fixture ever made fire
# were retired, so these shapes now reach the generic residual: `zip` over two
# tuples (positional and `strict=True`), a long/int comparison, a two-bigint
# true-divide, `bigint ** int`, `bigint << int` / `bigint >> int`, and
# `math.frexp` / `math.ldexp`. Nothing else in the corpus records a fire for
# any of them, so without this fixture the generic path they fall back to has
# no gate at all.
#
# Every result is reduced to an integer checksum: the float legs go through
# `int(round(x * 1e6))` so the comparison cannot turn into a float-formatting
# difference between the three backends.

import math

N = 2000


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


print("zip       ", f_zip(N))
print("zip_strict", f_zip_strict(N))
print("cmp_long  ", f_cmp_long_int(N))
print("truediv   ", int(round(f_truediv_long(N) * 1e6)))
print("pow       ", f_pow_long_int(N))
print("shift     ", f_shift_long_int(N))
print("frexp     ", int(round(f_frexp(N) * 1e6)))
print("ldexp     ", int(round(f_ldexp(N) * 1e6)))
