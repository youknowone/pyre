# CPython-suite gap: find/rfind/count stay residual in a compiled loop.
# parity-tests reason: pins unicodeobject.py _unwrap_and_search as an
# elidable call across a loop long enough to compile, including a wide
# payload and a negative end bound.

"""Compiled str.find / rfind / count on exact strs."""

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


def search_loop(n):
    s = "一二三四" * 64 + "zq"
    acc = 0
    i = 0
    while i < n:
        acc += s.find("zq") + s.rfind("一") + s.count("二", 7, -7)
        i = i + 1
    return acc


N = 30000
got = search_loop(N)
s = "一二三四" * 64 + "zq"
expect = N * (s.find("zq") + s.rfind("一") + s.count("二", 7, -7))
assert got == expect, (got, expect)

# Empty needle and a miss stay correct after compile.
t = "abcabc"
assert t.find("") == 0
assert t.rfind("z") == -1
assert t.count("a", 1, 5) == 1

print("OK")
