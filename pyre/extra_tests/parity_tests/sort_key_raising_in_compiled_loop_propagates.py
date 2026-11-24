# CPython-suite gap: test_sort raises from a key function only while the
# sortkey loop is still interpreted. The compiled jd4 path must still
# deliver the same exception and leave the list as list_sort_impl does.
# parity-tests reason: a key that raises after the sortkey loop is hot
# (N > threshold) must propagate that exception, restore the input order,
# and leave a later sort able to run.

"""A key function raising in a compiled sortkey loop propagates."""

N = 3000
RAISE_AT = 2500

calls = 0


class Boom(Exception):
    pass


def key(v):
    global calls
    calls += 1
    if calls == RAISE_AT:
        raise Boom("nth key")
    return v


xs = list(range(N))
saved = list(xs)
try:
    xs.sort(key=key)
except Boom as exc:
    assert type(exc) is Boom
    assert str(exc) == "nth key"
    assert exc.args == ("nth key",)
else:
    raise AssertionError("expected Boom")

assert xs == saved, xs[:8]
xs.sort()
assert xs == list(range(N))
print("OK")
