# CPython-suite gap: BINARY_SLICE / constant str slices stay residual.
# parity-tests reason: pins unicodeobject.py _unicode_sliced as an
# elidable cut across a compiled loop, including a wide payload.

"""Compiled str[start:stop] on exact str plus exact-int bounds."""

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


def slice_loop(n):
    s = "abcdefghij一二"
    acc = 0
    i = 0
    while i < n:
        acc = acc + len(s[1:4]) + len(s[i % 5 : i % 5 + 3])
        i = i + 1
    return acc


N = 30000
got = slice_loop(N)
s = "abcdefghij一二"
expect = 0
i = 0
while i < N:
    expect = expect + len(s[1:4]) + len(s[i % 5 : i % 5 + 3])
    i = i + 1
assert got == expect, (got, expect)
assert s[1:4] == "bcd"
assert s[-2:] == "一二"

print("OK")
