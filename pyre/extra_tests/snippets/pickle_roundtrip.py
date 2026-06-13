# Consolidated pickle round-trip verification for the iterator reducers.
# Non-divergent iterators reconstruct to the exact CPython 3.14 shape;
# map/filter/zip/set/reversed-of-sequence materialise to list_iterator
# (a documented divergence) but still round-trip their remaining items.
#
# Each object is round-tripped once at the default protocol (plus the big
# int at protocol 0, which exercises the text LONG opcode).  The exact
# reduce shapes are pinned by pickle_{seqiter,rangeiter,dictiter,enumerate}.
import pickle


def _double(x):
    return x * 2


def _is_even(x):
    return x % 2 == 0


def rt(obj):
    return pickle.loads(pickle.dumps(obj))


# --- sequence iterators (list / tuple / str / bytes) ---
for seq in ([1, 2, 3, 4], (10, 20, 30), "abcd", b"wxyz"):
    it = iter(seq)
    next(it)
    assert list(rt(it)) == list(seq)[1:], seq

# --- range iterator (small) ---
it = iter(range(10))
for _ in range(3):
    next(it)
assert list(rt(it)) == list(range(3, 10))

# --- long-range iterator + the proto-0 text LONG path for big ints ---
big = 10 ** 40
it = iter(range(big))
next(it)
next(it)
r = rt(it)
assert next(r) == 2 and next(r) == 3
assert pickle.loads(pickle.dumps(big, 0)) == big
assert pickle.loads(pickle.dumps(-big, 0)) == -big

# --- dict view iterators ---
d = {"a": 1, "b": 2, "c": 3}
ki = iter(d.keys()); next(ki)
assert list(rt(ki)) == ["b", "c"]
vi = iter(d.values()); next(vi)
assert list(rt(vi)) == [2, 3]
ii = iter(d.items()); next(ii)
assert list(rt(ii)) == [("b", 2), ("c", 3)]

# --- enumerate (default start, list source) ---
e = enumerate([10, 20, 30])
next(e)
assert list(rt(e)) == [(1, 20), (2, 30)]

# --- enumerate (custom start, non-list source) ---
e = enumerate(iter([10, 20, 30]), start=5)
next(e)
assert list(rt(e)) == [(6, 20), (7, 30)]

# --- divergent: set iterator materialises but round-trips its members ---
s = {1, 2, 3, 4}
si = iter(s)
first = next(si)
assert sorted(rt(si)) == sorted(x for x in s if x != first)

# --- divergent: map / filter / zip / reversed materialise to list_iterator ---
m = map(_double, [1, 2, 3]); next(m)
assert list(rt(m)) == [4, 6]
f = filter(_is_even, [1, 2, 3, 4]); next(f)
assert list(rt(f)) == [4]
z = zip([1, 2, 3], "abc"); next(z)
assert list(rt(z)) == [(2, "b"), (3, "c")]
rv = reversed([1, 2, 3]); next(rv)
assert list(rt(rv)) == [2, 1]

# --- reversed(range) stays a range iterator ---
rr = reversed(range(5)); next(rr)
assert list(rt(rr)) == [3, 2, 1, 0]

# --- bare builtin functions pickle by reference (__module__ == builtins) ---
assert rt(iter) is iter and rt(range) is range and rt(len) is len

print("pickle_roundtrip OK")
