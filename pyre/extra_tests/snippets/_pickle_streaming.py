# The Pickler streams to the file's write() as the pickle is produced rather
# than buffering the whole thing: a committed frame (protocol >= 4) and a large
# bytes/str/bytearray payload each go out as they are produced, while an
# unframed pickle (protocol < 4) is written in one call at the end. The wire
# bytes and the per-call split match _pickle.dumps / CPython 3.14 exactly.
import _pickle


class Recorder:
    """A minimal writable file recording each write() separately."""

    def __init__(self):
        self.calls = []

    def write(self, b):
        self.calls.append(bytes(b))
        return len(b)


def writes_for(obj, proto):
    rec = Recorder()
    _pickle.Pickler(rec, proto).dump(obj)
    return rec.calls


def check(obj, proto, expected_writes):
    calls = writes_for(obj, proto)
    ref = _pickle.dumps(obj, proto)
    # Streaming never changes the wire bytes.
    assert b"".join(calls) == ref, (proto, len(b"".join(calls)), len(ref))
    # The number of write() calls matches.
    assert len(calls) == expected_writes, (proto, len(calls), expected_writes)
    # And it round-trips.
    assert _pickle.loads(b"".join(calls)) == obj, proto
    return calls


# Small object: a single write at every protocol.
check([1, 2, 3], 0, 1)
check([1, 2, 3], 2, 1)
check([1, 2, 3], 5, 1)

# Many small objects overflow one frame (protocol >= 4): the first frame is
# committed mid-dump, the rest at the end — two writes. Unframed: one write.
big_list = list(range(40000))
two = check(big_list, 5, 2)
# Each committed write carries a FRAME opcode (0x95) at protocol 5.
assert two[0][2] == 0x95, two[0][:3]  # after PROTO 5 (2 bytes)
check(big_list, 2, 1)

# A large payload is written directly (header with the pending bytes, then the
# payload, then the trailer): three writes. Holds for bytes, str, bytearray.
payload = b"z" * (200 * 1024)
calls = check(payload, 5, 3)
assert calls[1] == payload, (len(calls[1]), len(payload))
check("u" * (200 * 1024), 5, 3)
check(bytearray(b"a" * (200 * 1024)), 5, 3)
# Unframed large bytes also splits header / payload / trailer into three.
check(payload, 2, 3)


# Module-level dump streams to the file; dumps returns the bytes unchanged.
rec = Recorder()
_pickle.dump(big_list, rec, 5)
assert len(rec.calls) == 2
assert b"".join(rec.calls) == _pickle.dumps(big_list, 5)


# GC safety: a write() that allocates heavily runs at each streaming point
# while the save tree holds live object pointers across the boundary.
class GreedyFile:
    def __init__(self):
        self.parts = []

    def write(self, b):
        _ = [object() for _ in range(500)]
        _ = "".join(str(x) for x in range(40))
        self.parts.append(bytes(b))
        return len(b)


nested = []
for i in range(2000):
    nested.append({"k%d" % i: [i, b"q" * 60, "t" * 50, (i, i + 1)]})
nested.append(b"B" * (80 * 1024))
nested.append("U" * (80 * 1024))
nested.append(bytearray(b"A" * (80 * 1024)))
for proto in (0, 2, 4, 5):
    gf = GreedyFile()
    _pickle.Pickler(gf, proto).dump(nested)
    streamed = b"".join(gf.parts)
    assert streamed == _pickle.dumps(nested, proto), proto
    assert _pickle.loads(streamed) == nested, proto

print("_pickle_streaming OK")
