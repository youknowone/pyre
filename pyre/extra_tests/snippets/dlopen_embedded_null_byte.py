# pyre-check: platforms=darwin,linux
# pyre-check: requires-modules=_ctypes,_cffi_backend
"""dlopen callers reject a name with an embedded NUL as ValueError."""
import ctypes

try:
    ctypes.CDLL("foo\0bar")
except ValueError as exc:
    assert "embedded null" in str(exc), exc
else:
    raise SystemExit("ctypes.CDLL accepted an embedded null")

try:
    import _cffi_backend
except ImportError:
    raise SystemExit(0)

try:
    _cffi_backend.FFI().dlopen("foo\0bar")
except ValueError as exc:
    assert "embedded null" in str(exc), exc
else:
    raise SystemExit("_cffi_backend.FFI.dlopen accepted an embedded null")

# `func.py from_buffer` / `W_CDataFromBuffer`: the cdata owns a `buffer_w`
# lease for its lifetime; `release` / `__exit__` / the finalizer drop it.
import array
import gc

B = _cffi_backend
BCharA = B.new_array_type(B.new_pointer_type(B.new_primitive_type("char")), None)

ba = bytearray(b"abc")
c = B.from_buffer(BCharA, ba)
assert len(c) == 3
try:
    ba.append(100)
except BufferError as exc:
    assert "Existing exports of data" in str(exc), exc
else:
    raise SystemExit("from_buffer cdata did not pin the bytearray export")
assert c[0] == b"a"

assert B.from_buffer(BCharA, memoryview(bytearray(b"xyz")))[1] == b"y"
assert B.from_buffer(BCharA, b"qr")[0] == b"q"
try:
    B.from_buffer(BCharA, "s")
except TypeError as exc:
    assert "unicode object" in str(exc), exc
else:
    raise SystemExit("from_buffer accepted a unicode object")
try:
    B.from_buffer(BCharA, memoryview(b"abcd")[::2])
except BufferError as exc:
    assert "not C-contiguous" in str(exc), exc
else:
    raise SystemExit("from_buffer accepted a strided memoryview")
assert len(B.from_buffer(BCharA, array.array("i", [1, 2]))) == 8

m = memoryview(bytearray(b"zz"))
c_mv = B.from_buffer(BCharA, m)
m.release()
assert "memoryview" in repr(c_mv)

ba_rel = bytearray(b"abc")
c_rel = B.from_buffer(BCharA, ba_rel)
B.release(c_rel)
ba_rel.append(100)
assert bytes(ba_rel) == b"abcd"

ba_with = bytearray(b"abc")
c_with = B.from_buffer(BCharA, ba_with)
with c_with:
    pass
ba_with.append(100)
assert bytes(ba_with) == b"abcd"

ba_gc = bytearray(b"abc")
c_gc = B.from_buffer(BCharA, ba_gc)
del c_gc
gc.collect()
gc.collect()
ba_gc.append(100)
assert bytes(ba_gc) == b"abcd"

try:
    B.from_buffer(BCharA, b"ab", True)
except BufferError as exc:
    assert str(exc) == "Object is not writable.", exc
else:
    raise SystemExit("from_buffer(bytes, require_writable=True) succeeded")

class CountBuf:
    acquires = 0
    releases = 0

    def __init__(self, data):
        self.data = bytearray(data)

    def __buffer__(self, flags):
        type(self).acquires += 1
        return memoryview(self.data)

    def __release_buffer__(self, view):
        type(self).releases += 1

CountBuf.acquires = 0
CountBuf.releases = 0
cb = CountBuf(b"xy")
c_pep = B.from_buffer(BCharA, cb)
assert CountBuf.acquires == 1, CountBuf.acquires
assert CountBuf.releases == 0, CountBuf.releases
B.release(c_pep)
assert CountBuf.acquires == 1, CountBuf.acquires
assert CountBuf.releases == 0, CountBuf.releases
del c_pep
gc.collect()
assert CountBuf.acquires == 1, CountBuf.acquires
assert CountBuf.releases == 0, CountBuf.releases

# `W_MemoryView.buffer_w` / `NonOwningReleaseView`: after `m.release()` the
# cdata still holds the backing, so reads and writes through `c` succeed.
m_hold = memoryview(bytearray(b"abcdefgh"))
c_hold = B.from_buffer(BCharA, m_hold)
m_hold.release()
del m_hold
for _ in range(32):
    _ = bytearray(4096)
    gc.collect()
assert c_hold[0] == b"a", c_hold[0]
assert c_hold[7] == b"h", c_hold[7]
c_hold[0] = b"Z"
assert c_hold[0] == b"Z", c_hold[0]
assert c_hold[1] == b"b", c_hold[1]

# `func.py get_raw_address_of_string` / `RawBytesCache.wdict`: two calls on
# the same live bytes object return the same pointer.
intptr = B.new_primitive_type("intptr_t")

def _buf_addr(cdata):
    return int(B.cast(intptr, cdata))

b_same = bytes(range(8))
c_same_1 = B.from_buffer(BCharA, b_same)
c_same_2 = B.from_buffer(BCharA, b_same)
assert _buf_addr(c_same_1) == _buf_addr(c_same_2)

# Distinct equal bytes objects are distinct keys (`ll_keyeq` is identity).
b_eq_1 = bytes([101, 113, 117, 97, 108, 45, 99, 111, 110, 116, 101, 110, 116])
b_eq_2 = bytes([101, 113, 117, 97, 108, 45, 99, 111, 110, 116, 101, 110, 116])
assert b_eq_1 == b_eq_2 and b_eq_1 is not b_eq_2
c_eq_1 = B.from_buffer(BCharA, b_eq_1)
c_eq_2 = B.from_buffer(BCharA, b_eq_2)
assert _buf_addr(c_eq_1) != _buf_addr(c_eq_2)

# Dead keys drop their `RawBytes` on collect; a still-live key keeps a
# consistent pointer across `gc.collect()`.
kept_bytes = None
kept_addr = None
for i in range(32):
    x = bytes([i, i, i, i, i, i, i, i])
    c = B.from_buffer(BCharA, x)
    p = _buf_addr(c)
    assert _buf_addr(B.from_buffer(BCharA, x)) == p
    kept_bytes, kept_addr = x, p
    del c
    gc.collect()
assert _buf_addr(B.from_buffer(BCharA, kept_bytes)) == kept_addr
