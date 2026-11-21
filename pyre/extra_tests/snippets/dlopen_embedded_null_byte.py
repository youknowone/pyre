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
assert CountBuf.releases == 1, CountBuf.releases
