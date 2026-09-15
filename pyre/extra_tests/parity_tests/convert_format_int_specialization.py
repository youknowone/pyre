# CPython-suite gap: f-string CONVERT_VALUE / FORMAT_WITH_SPEC on int stay
# residual in a compiled loop unless the walker records descr_str / descr_format.
# parity-tests reason: pins the ll_int2dec + pad split across a loop long
# enough to compile, including identity of f"{s!s}" and the multi-digit
# `str(i) is str(i)` False that a fused wrap would break.

"""Compiled CONVERT_VALUE / FORMAT_WITH_SPEC on exact int and exact str."""

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


def convert_loop(n):
    acc = 0
    i = 0
    while i < n:
        s = f"{i!r}-{i!s}-{i!a}"
        acc = acc + len(s)
        i = i + 1
    return acc


def spec_loop(n):
    acc = 0
    i = 0
    while i < n:
        s = f"{i:05d}"
        acc = acc + len(s)
        i = i + 1
    return acc


def simple_loop(n):
    acc = 0
    i = 0
    while i < n:
        s = f"{i}"
        acc = acc + len(s)
        i = i + 1
    return acc


N = 30000
assert convert_loop(N) == simple_loop(N) * 3 + N * 2
assert spec_loop(N) == N * 5

# `descr_str` of an exact str is identity (`unicodeobject.py`).
s = "abc"
for _ in range(N):
    assert f"{s}" is s
    assert f"{s!s}" is s
    assert f"{s!r}" == "'abc'"
    assert f"{s!a}" == "'abc'"

# Residual wrap: two `str(i)` sites allocate two wrappers.
# `is_w` of `_len() > 1` compares `_utf8` storage.
i = 12
assert str(i) == "12"
assert str(i) is not str(i)

# Sign-interior pad stays residual but must still be correct.
assert f"{-42:05d}" == "-0042"
assert f"{42:5d}" == "   42"
assert f"{42:d}" == "42"
assert f"{True}" == "True"
assert f"{True:05d}" == "00001"

# `format(1, "x") == "1"` must not select ll_int2dec.  A compiled loop
# that then sees 10 has to print hex, not decimal.
def hex_loop(n):
    acc = ""
    i = 1
    while i <= n:
        acc = format(i, "x")
        i = i + 1
    return acc


assert format(1, "x") == "1"
assert hex_loop(10) == "a"
assert hex_loop(16) == "10"

# `format(-1, "+d") == "-1"` must not select unpadded `str(i)`.
# A compiled loop that then sees `1` has to print `+1`.
def plus_d_loop(lo, hi):
    acc = []
    i = lo
    while i <= hi:
        acc.append(format(i, "+d"))
        i = i + 1
    return acc


assert plus_d_loop(-2, 2) == ["-2", "-1", "+0", "+1", "+2"]
assert plus_d_loop(1, 3) == ["+1", "+2", "+3"]

# `format(12345, "3d") == "12345"` must not select unpadded `str(i)`.
# A compiled loop that then sees `1` has to print `"  1"`.
def width_d_loop(values):
    acc = []
    i = 0
    while i < len(values):
        acc.append(format(values[i], "3d"))
        i = i + 1
    return acc


wide_then_short = [12345] * 80 + [1]
assert width_d_loop(wide_then_short)[-1] == "  1"
assert format(12345, "3d") == "12345"


class WideFmt(int):
    def __format__(self, spec):
        return "X"


assert format(WideFmt(1 << 100), "") == "X"

print("OK")
