# Direct test of the interp-level `_pickle` accelerator (increment 6):
# protocol 0 / 1 legacy text opcodes (INT / LONG / FLOAT / UNICODE plus the
# MARK-based LIST / DICT / TUPLE and text PUT / GET), persistent_id, and the
# `_compat_pickle` fix_imports name mapping at protocol < 3.
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
    assert type(got) is type(obj), (proto, type(obj), type(got))
    return got


# Protocol 0 (text) and 1 (binary) atoms + containers round-trip.
for proto in (0, 1):
    for obj in [
        None, True, False,
        0, 1, -1, 255, 256, -256, 2 ** 31, -(2 ** 31), 2 ** 70, -(2 ** 70),
        3.14, -2.5, 0.0, 1e300,
        "", "abc", "héllo", "ünïcödé\n\ttab", "with\\backslash", "null\0byte",
        [], [1, 2, 3], ["a", "b"],
        (), (1,), (1, 2, 3), (1, 2, 3, 4),
        {}, {"k": "v", "n": 1}, [[1, 2], [3, 4]], {"d": {"e": 1}},
    ]:
        roundtrip(obj, proto)

# Protocol 0 wire is byte-identical to CPython 3.14 for these.
assert dumps(1, 0) == b"I1\n.", dumps(1, 0)
assert dumps(2 ** 70, 0) == b"L1180591620717411303424L\n.", dumps(2 ** 70, 0)
assert dumps(2.5, 0) == b"F2.5\n.", dumps(2.5, 0)
assert dumps("ab", 0) == b"Vab\np0\n.", dumps("ab", 0)
assert dumps([1, 2], 0) == b"(lp0\nI1\naI2\na.", dumps([1, 2], 0)
assert dumps((1, 2), 0) == b"(I1\nI2\ntp0\n.", dumps((1, 2), 0)
assert dumps({"a": 1}, 0) == b"(dp0\nVa\np1\nI1\ns.", dumps({"a": 1}, 0)

# bytes / bytearray at protocol < 3 reduce through `_codecs.encode`; the global
# reference resolves the encode function's `__module__` (`_codecs`), so the wire
# is byte-identical to CPython 3.14.
assert dumps(b"abc", 0) == b"c_codecs\nencode\np0\n(Vabc\np1\nVlatin1\np2\ntp3\nRp4\n.", dumps(b"abc", 0)
assert dumps(b"abc", 1) == b"c_codecs\nencode\nq\x00(X\x03\x00\x00\x00abcq\x01X\x06\x00\x00\x00latin1q\x02tq\x03Rq\x04.", dumps(b"abc", 1)
assert dumps(b"abc", 2) == b"\x80\x02c_codecs\nencode\nq\x00X\x03\x00\x00\x00abcq\x01X\x06\x00\x00\x00latin1q\x02\x86q\x03Rq\x04.", dumps(b"abc", 2)
assert dumps(bytearray(b"abc"), 0) == b"c__builtin__\nbytearray\np0\n(c_codecs\nencode\np1\n(Vabc\np2\nVlatin1\np3\ntp4\nRp5\ntp6\nRp7\n.", dumps(bytearray(b"abc"), 0)
assert dumps(bytearray(b"abc"), 2) == b"\x80\x02c__builtin__\nbytearray\nq\x00c_codecs\nencode\nq\x01X\x03\x00\x00\x00abcq\x02X\x06\x00\x00\x00latin1q\x03\x86q\x04Rq\x05\x85q\x06Rq\x07.", dumps(bytearray(b"abc"), 2)

# fix_imports: at protocol < 3, range maps to the Python 2 __builtin__.xrange
# global; protocol 3 keeps the canonical name. Both round-trip.
assert dumps(range(5), 2) == (
    b"\x80\x02c__builtin__\nxrange\nq\x00K\x00K\x05K\x01\x87q\x01Rq\x02."
), dumps(range(5), 2)
for proto in range(0, 6):
    assert roundtrip(range(2, 10, 3), proto) == range(2, 10, 3)

# Loading the Python 2 names directly also resolves through fix_imports.
assert loads(b"\x80\x02c__builtin__\nxrange\nK\x00K\x03K\x01\x87R.") == range(3)


# persistent_id / persistent_load via subclassing.
_registry = {"shared": ["one", "two"]}


class IdPickler(_pickle.Pickler):
    def persistent_id(self, obj):
        for key, value in _registry.items():
            if value is obj:
                return key
        return None


class IdUnpickler(_pickle.Unpickler):
    def persistent_load(self, pid):
        return _registry[pid]


for proto in (0, 2, 5):
    buf = io.BytesIO()
    shared = _registry["shared"]
    IdPickler(buf, proto).dump([shared, shared, "tail"])
    got = IdUnpickler(io.BytesIO(buf.getvalue())).load()
    assert got[0] is shared, (proto, "persistent_load object identity")
    assert got[0] is got[1], (proto, "shared persistent reference")
    assert got[2] == "tail", proto

print("_pickle_legacy OK")
