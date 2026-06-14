# Direct test of the interp-level `_pickle` accelerator (increment 7,
# framing): the multi-frame framer flushes a frame once it reaches the
# 64 KiB target, and large bytes / str / bytearray payloads are written
# outside any frame (write_large_bytes). Output is byte-identical to
# CPython 3.14 (cross-checked via the FNV-1a checksum printed below).
import io
import _pickle


def dumps(obj, proto):
    buf = io.BytesIO()
    _pickle.Pickler(buf, proto).dump(obj)
    return buf.getvalue()


def loads(data):
    return _pickle.Unpickler(io.BytesIO(data)).load()


def cksum(b):
    h = 14695981039346656037
    for x in b:
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def report(tag, data):
    print(tag, len(data), data.count(b"\x95"), cksum(data))


# A pickle large enough to cross the 64 KiB frame target produces more than
# one frame (FRAME opcode = 0x95).
big_list = list(range(40000))
data = dumps(big_list, 5)
assert loads(data) == big_list
assert data.count(b"\x95") >= 2, data.count(b"\x95")
report("list5", data)

# Large bytes (>= 64 KiB) are written outside any frame; the surrounding
# stream still frames the PROTO/STOP scaffolding.
big_bytes = bytes(range(256)) * 400
data = dumps(big_bytes, 5)
assert loads(data) == big_bytes
report("bytes5", data)

# Large str (>= 64 KiB) likewise.
big_str = "abcd" * 25000
data = dumps(big_str, 5)
assert loads(data) == big_str
report("str5", data)

# Large bytearray (proto 5 raw form) likewise.
big_ba = bytearray(bytes(range(256)) * 400)
data = dumps(big_ba, 5)
assert loads(data) == big_ba
report("bytearray5", data)

# Protocol 4 frames the same way (default proto 5 vs explicit 4 differ only
# in the bytearray raw opcode availability).
data = dumps(big_list, 4)
assert loads(data) == big_list
assert data.count(b"\x95") >= 2
report("list4", data)

# A small pickle is a single frame (or none below FRAME_SIZE_MIN).
data = dumps([1, 2, 3], 5)
assert loads(data) == [1, 2, 3]
assert data.count(b"\x95") == 1
report("small5", data)

print("_pickle_framing OK")
