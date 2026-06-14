# Direct test of the interp-level `_pickle` accelerator (increments 2-3):
# memo back-references plus the tuple / list / dict containers, including
# APPENDS / SETITEMS batching, nesting, shared references and recursion.
import io
import _pickle


def dumps(obj, proto):
    buf = io.BytesIO()
    _pickle.Pickler(buf, proto).dump(obj)
    return buf.getvalue()


def loads(data):
    return _pickle.Unpickler(io.BytesIO(data)).load()


def roundtrip(obj, proto):
    got = loads(dumps(obj, proto))
    assert got == obj, (proto, repr(obj), repr(got))
    assert type(got) is type(obj), (proto, repr(obj), type(got), type(obj))
    return got


CASES = [
    (), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5),
    [], [1, 2, 3], [1, [2, 3], 4],
    {}, {1: 2}, {1: 2, 3: 4}, {"a": 1, "b": [1, 2, 3]},
    [(1, 2), (3, 4)], {1: {2: {3: 4}}},
    [None, True, False, 3.5, b"xy", "uv"],
    list(range(2500)),                 # APPENDS batching (> _BATCHSIZE)
    {i: i * i for i in range(2500)},   # SETITEMS batching
]

for proto in range(2, 6):
    for obj in CASES:
        roundtrip(obj, proto)

# Shared reference: the same inner list is reachable twice by identity, so
# the second reference must be a memo GET that the unpickler restores to the
# same object.
inner = [1, 2]
outer = [inner, inner]
got = loads(dumps(outer, 4))
assert got == outer
assert got[0] is got[1], "shared reference identity not preserved"

# Recursive list: the memo lets the unpickler close the cycle.
rec = []
rec.append(rec)
got = loads(dumps(rec, 4))
assert got[0] is got, "recursive list not reconstructed"

# Wire-format locks — byte-identical to CPython 3.14.
assert dumps((1, 2, 3), 4) == b"\x80\x04\x95\x09\x00\x00\x00\x00\x00\x00\x00K\x01K\x02K\x03\x87\x94.", dumps((1, 2, 3), 4)
assert dumps([1, 2, 3], 2) == b"\x80\x02]q\x00(K\x01K\x02K\x03e.", dumps([1, 2, 3], 2)
assert dumps({1: 2}, 2) == b"\x80\x02}q\x00K\x01K\x02s.", dumps({1: 2}, 2)

print("_pickle_containers OK")
