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
# Closed form: a second loop spelling the same expression would be compiled
# by the same fold and agree with it whatever it computed.
assert slice_loop(N) == 6 * N
s = "abcdefghij一二"
assert s[1:4] == "bcd"
assert s[-2:] == "一二"


# The cut itself, not only its length: wide code points, an empty window
# (`stop < start`), a `None` bound and a negative one.
def slice_contents(n):
    s = "abcdefghij一二"
    last = None
    empties = 0
    i = 0
    while i < n:
        k = i % 5
        last = (s[k : k + 3], s[9:], s[:-10], s[-3:-1])
        if s[k + 3 : k] == "":
            empties = empties + 1
        i = i + 1
    return last, empties


last, empties = slice_contents(3000)
assert last == ("efg", "j一二", "ab", "j一"), last
assert empties == 3000, empties

print("OK")
