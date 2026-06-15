# Increment 8 (THE FLIP): the top-level `pickle` module resolves
# Pickler / Unpickler / dump / dumps / load / loads and the three error
# classes from the interp-level `_pickle` accelerator. This exercises the
# public `pickle` API end-to-end and cross-validates against CPython 3.14.
import io

try:
    import pickle
except (ImportError, AssertionError):
    # `import pickle` pulls in `re`, whose `_sre.MAGIC` is asserted against
    # the stdlib version. When PYRE_STDLIB is not pinned to the bundled 3.14
    # stdlib (e.g. the bare extra_tests runner falling back to the host's
    # older stdlib), that assertion fails before pickle loads. That is the
    # same coupling as synth/sre_pattern_methods, not a pickle defect, so
    # skip here; run with PYRE_STDLIB=lib-python/3 for the full check.
    print("_pickle_module SKIP (stdlib re/_sre MAGIC mismatch)")
    raise SystemExit(0)


# All nine names come from the accelerator.
for name in ("Pickler", "Unpickler", "dump", "dumps", "load", "loads"):
    assert hasattr(pickle, name), name
assert pickle.Pickler.__module__ == "_pickle", pickle.Pickler.__module__
assert pickle.Unpickler.__module__ == "_pickle", pickle.Unpickler.__module__

# The exception hierarchy: PicklingError / UnpicklingError subclass PickleError.
assert issubclass(pickle.PicklingError, pickle.PickleError)
assert issubclass(pickle.UnpicklingError, pickle.PickleError)
assert issubclass(pickle.PickleError, Exception)

# HIGHEST_PROTOCOL is 5 since 3.8; DEFAULT_PROTOCOL is 4 (<=3.13) or 5
# (3.14). pyre and the 3.14 target use 5; the cross-check runner may run an
# older host, so accept either rather than pinning the host version.
assert pickle.HIGHEST_PROTOCOL == 5, pickle.HIGHEST_PROTOCOL
assert pickle.DEFAULT_PROTOCOL in (4, 5), pickle.DEFAULT_PROTOCOL


def roundtrip(obj, proto):
    got = pickle.loads(pickle.dumps(obj, proto))
    assert got == obj, (proto, repr(obj), repr(got))
    assert type(got) is type(obj), (proto, type(obj), type(got))
    return got


samples = [
    None, True, False, 0, 1, -1, 255, 2 ** 70, -(2 ** 70),
    3.14, -0.0, 1e300,
    "", "abc", "ünïcödé", b"", b"\x00\xff\x80",
    [], [1, "two", 3.0], (), (1, 2, 3),
    {}, {"k": "v", 1: [2, 3]},
    bytearray(b"\x00\x01\x02"),
]
for proto in range(0, 6):
    for obj in samples:
        roundtrip(obj, proto)

# dump / load through a file object (io.BytesIO).
buf = io.BytesIO()
pickle.dump({"x": [1, 2], "y": (3, 4)}, buf, 5)
buf.seek(0)
assert pickle.load(buf) == {"x": [1, 2], "y": (3, 4)}

# Default protocol round-trips.
assert pickle.loads(pickle.dumps(["default", 1, 2])) == ["default", 1, 2]

# Shared-reference identity survives the public API.
shared = [1, 2, 3]
g = pickle.loads(pickle.dumps([shared, shared], 5))
assert g[0] is g[1]

# Protocol-0 wire is byte-identical to CPython 3.14 via the public API.
assert pickle.dumps(1, 0) == b"I1\n.", pickle.dumps(1, 0)
assert pickle.dumps([1, 2], 0) == b"(lp0\nI1\naI2\na.", pickle.dumps([1, 2], 0)

# An invalid opcode raises UnpicklingError.
try:
    pickle.loads(b"\xff.")
    raise AssertionError("expected UnpicklingError")
except pickle.UnpicklingError:
    pass

# A buffer_callback that is never invoked (no PickleBuffer values present)
# leaves pickling unchanged, exactly as in CPython.
calls = []
assert pickle.loads(pickle.dumps(b"data", 5, buffer_callback=calls.append)) == b"data"
assert calls == []

# Protocol-5 out-of-band buffers via the public API: PickleBuffer resolves
# from the accelerator and round-trips through buffer_callback / buffers.
assert pickle._HAVE_PICKLE_BUFFER
assert pickle.PickleBuffer is not None
bufs = []
pb_data = pickle.dumps(pickle.PickleBuffer(b"oob payload"), 5,
                       buffer_callback=lambda b: bufs.append(b) or False)
assert len(bufs) == 1
restored = pickle.loads(pb_data, buffers=[b"oob payload"])
restored = restored if isinstance(restored, (bytes, bytearray)) else restored.tobytes()
assert restored == b"oob payload", restored

print("_pickle_module OK")
