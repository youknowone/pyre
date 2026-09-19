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


# A nested `{spec}` is a box, not a constant.  An empty spec takes the
# FORMAT_SIMPLE rendering, and a loop compiled on it must leave for the
# residual when a later iteration hands in a width.
def nested_spec_loop(values, specs):
    acc = []
    i = 0
    while i < len(values):
        acc.append(f"{values[i]:{specs[i]}}")
        i = i + 1
    return acc


empty_then_wide = [""] * 300 + [">5"]
assert nested_spec_loop([7] * 301, empty_then_wide)[-1] == "    7"
assert nested_spec_loop(["ab"] * 301, empty_then_wide)[-1] == "   ab"
assert nested_spec_loop([7] * 301, empty_then_wide)[0] == "7"


# A pad is recorded from one value: the sign and the digit count that
# produced it both have to be pinned, for every fill / align / sign spelling.
def padded_loop(values, which):
    acc = []
    i = 0
    while i < len(values):
        x = values[i]
        if which == 0:
            acc.append(f"{x:+5d}")
        elif which == 1:
            acc.append(f"{x:5d}")
        elif which == 2:
            acc.append(f"{x:<5d}|")
        elif which == 3:
            acc.append(f"{x:05d}")
        else:
            acc.append(f"{x: d}")
        i = i + 1
    return acc[-3:]


assert padded_loop([-1] * 300 + [1, 12, 0], 0) == ["   +1", "  +12", "   +0"]
assert padded_loop([1] * 300 + [12, -1, 1234567], 1) == ["   12", "   -1", "1234567"]
assert padded_loop([123] * 300 + [-5, 7, 0], 2) == ["-5   |", "7    |", "0    |"]
assert padded_loop([-1] * 300 + [1, 12, -123456], 3) == ["00001", "00012", "-123456"]
assert padded_loop([-1] * 300 + [1, 12, 0], 4) == [" 1", " 12", " 0"]

print("OK")
