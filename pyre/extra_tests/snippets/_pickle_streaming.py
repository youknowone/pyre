# The Pickler streams to the file's write() as the pickle is produced. The
# CPython suite (test_framing_*, test_framed_write_sizes_with_delayed_writer)
# only checks that framing happens at all (len(chunks) > 1). What we pin here:
#   - the EXACT number of write() calls and their byte split, matching CPython
#     3.14 (small obj -> 1 write; 40000 ints at proto 5 -> 2 writes; a 200KB
#     payload -> 3 writes split as [header, payload, trailer]),
#   - GC safety: a write() that allocates heavily forces the moving GC to
#     relocate objects at each streaming point while the save tree holds live
#     pointers across the boundary (CPython has no moving GC, so its suite
#     cannot probe this).
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
    # Streaming never changes the wire bytes.
    assert b"".join(calls) == _pickle.dumps(obj, proto), (proto, len(calls))
    # The number of write() calls matches.
    assert len(calls) == expected_writes, (proto, len(calls), expected_writes)
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

# Module-level dump streams to the file the same way.
rec = Recorder()
_pickle.dump(big_list, rec, 5)
assert len(rec.calls) == 2


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

print("_pickle_streaming OK")
