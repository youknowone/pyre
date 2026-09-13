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

print("OK")
