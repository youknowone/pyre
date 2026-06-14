# Direct test of the interp-level `_pickle` accelerator (increment 1):
# protocol 2-5 atoms via Pickler/Unpickler over io.BytesIO.
import io
import _pickle


def roundtrip(obj, proto):
    buf = io.BytesIO()
    _pickle.Pickler(buf, proto).dump(obj)
    buf.seek(0)
    return _pickle.Unpickler(buf).load()


CASES = [
    None, True, False,
    0, 1, -1, 255, 256, 65535, 65536,
    2 ** 31 - 1, -(2 ** 31), 2 ** 31, 2 ** 63, 2 ** 100, -(2 ** 100),
    3.14, -0.0, 1e308,
    "", "a", "hello", "유니코드",
    b"", b"abc", b"\x00\xff" * 10,
]

for proto in range(2, 6):
    for obj in CASES:
        got = roundtrip(obj, proto)
        assert got == obj, (proto, repr(obj), repr(got))
        assert type(got) is type(obj), (proto, repr(obj), type(got), type(obj))

# Wire-format lock — byte-identical to CPython 3.14.
buf = io.BytesIO()
_pickle.Pickler(buf, 4).dump(None)
assert buf.getvalue() == b"\x80\x04N.", buf.getvalue()

buf = io.BytesIO()
_pickle.Pickler(buf, 4).dump("a")
assert buf.getvalue() == b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00\x8c\x01a\x94.", buf.getvalue()

buf = io.BytesIO()
_pickle.Pickler(buf, 2).dump(1)
assert buf.getvalue() == b"\x80\x02K\x01.", buf.getvalue()

print("_pickle_atoms OK")
