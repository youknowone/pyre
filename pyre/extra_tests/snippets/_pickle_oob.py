# Increment 7b: protocol-5 out-of-band buffers. A `PickleBuffer` wrapping a
# bytes-like object serializes in-band by default (BINBYTES for a read-only
# buffer, BYTEARRAY8 for a mutable one, both memoized) and out-of-band when a
# `buffer_callback` returns a false value (NEXT_BUFFER, plus READONLY_BUFFER
# for a read-only buffer). The wire format is byte-identical to CPython 3.14.
import io

# pyre exposes PickleBuffer through `__pypy__` and the accelerator as
# `_pickle`; avoid `import pickle` here so the test does not depend on the
# bundled stdlib (re/_sre) being pinned. CPython has neither `__pypy__` nor a
# stdlib-free `_pickle.PickleBuffer`, so fall back to the public `pickle`.
try:
    from __pypy__ import PickleBuffer
    import _pickle as P
except ImportError:
    import pickle as P
    PickleBuffer = P.PickleBuffer

Pickler, Unpickler = P.Pickler, P.Unpickler


def dumps(obj, proto, **kw):
    b = io.BytesIO()
    Pickler(b, proto, **kw).dump(obj)
    return b.getvalue()


def loads(data, **kw):
    return Unpickler(io.BytesIO(data), **kw).load()


def as_bytes(x):
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    return x.tobytes()  # memoryview


# In-band read-only buffer round-trips through BINBYTES (no NEXT_BUFFER).
d = dumps(PickleBuffer(b"hello readonly"), 5)
assert 0x97 not in d
assert as_bytes(loads(d)) == b"hello readonly"

# In-band mutable buffer round-trips through BYTEARRAY8.
d = dumps(PickleBuffer(bytearray(b"hello mutable")), 5)
assert 0x97 not in d
r = loads(d)
assert isinstance(r, bytearray) and as_bytes(r) == b"hello mutable"

# Out-of-band read-only: callback returns a false value -> NEXT_BUFFER and
# READONLY_BUFFER, callback invoked exactly once, data not in the stream.
collected = []
orig = b"out of band readonly data"
d = dumps(PickleBuffer(orig), 5, buffer_callback=lambda b: collected.append(b) or False)
assert 0x97 in d and 0x98 in d
assert len(collected) == 1
assert as_bytes(loads(d, buffers=[orig])) == orig

# Out-of-band mutable: NEXT_BUFFER without READONLY_BUFFER.
collected = []
orig_m = bytearray(b"out of band mutable data")
d = dumps(PickleBuffer(orig_m), 5, buffer_callback=lambda b: collected.append(b) or False)
assert 0x97 in d and 0x98 not in d
assert as_bytes(loads(d, buffers=[orig_m])) == bytes(orig_m)

# A callback returning a true value keeps the buffer in-band.
collected = []
d = dumps(PickleBuffer(b"inband via callback"), 5, buffer_callback=lambda b: collected.append(b) or True)
assert 0x97 not in d
assert len(collected) == 1
assert as_bytes(loads(d)) == b"inband via callback"

# Shared identity: a PickleBuffer saved in-band twice memoizes (GET back-ref).
pb = PickleBuffer(b"shared")
g = loads(dumps([pb, pb], 5))
assert as_bytes(g[0]) == b"shared" and as_bytes(g[1]) == b"shared"
assert g[0] is g[1]

# PickleBuffer requires protocol >= 5.
try:
    dumps(PickleBuffer(b"x"), 4)
    raise AssertionError("expected error for protocol < 5")
except Exception as e:
    assert "protocol >= 5" in str(e), str(e)

# buffer_callback requires protocol >= 5.
try:
    dumps(b"x", 4, buffer_callback=lambda b: None)
    raise AssertionError("expected ValueError for buffer_callback < proto 5")
except ValueError as e:
    assert "protocol >= 5" in str(e), str(e)

# An out-of-band stream cannot be loaded without a *buffers* argument.
collected = []
d = dumps(PickleBuffer(b"needs buffers"), 5, buffer_callback=lambda b: collected.append(b) or False)
try:
    loads(d)
    raise AssertionError("expected error for missing buffers")
except Exception as e:
    assert "out-of-band" in str(e), str(e)

# raw() exposes a memoryview onto the wrapped buffer; release() invalidates it.
pb = PickleBuffer(b"raw bytes")
assert as_bytes(pb.raw()) == b"raw bytes"
pb.release()
try:
    pb.raw()
    raise AssertionError("expected error after release")
except (ValueError, Exception):
    pass

print("_pickle_oob OK")
