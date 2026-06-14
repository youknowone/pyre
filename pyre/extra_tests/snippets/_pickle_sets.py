# Direct test of the interp-level `_pickle` accelerator (increment 4):
# set / frozenset (protocol >= 4) and bytearray (protocol >= 5).
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


for proto in (4, 5):
    for obj in [
        set(), {1}, {1, 2, 3}, {1, 2, 3, "x", "y"},
        frozenset(), frozenset({1, 2, 3}), frozenset({(1, 2), (3, 4)}),
        set(range(2500)),                  # ADDITEMS batching (> _BATCHSIZE)
        [{1, 2}, {3, 4}], {"s": {1, 2, 3}},
    ]:
        roundtrip(obj, proto)

# bytearray is ordered, so its protocol-5 form is byte-identical to CPython.
for obj in [bytearray(b""), bytearray(b"hello\x00\xff"), bytearray(range(256))]:
    roundtrip(obj, 5)

assert dumps(bytearray(b"hi"), 5) == (
    b"\x80\x05\x95\x0d\x00\x00\x00\x00\x00\x00\x00\x96\x02\x00\x00\x00\x00\x00\x00\x00hi\x94."
), dumps(bytearray(b"hi"), 5)

print("_pickle_sets OK")
