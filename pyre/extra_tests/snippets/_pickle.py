# Consolidated direct tests of the interp-level `_pickle` accelerator and the
# public `pickle` facade it backs. Each section below was previously a separate
# `_pickle_*` snippet; they share the `dumps` / `loads` / `roundtrip` helpers
# and run sequentially. Behaviors are pinned to CPython 3.14.
#
# Sections that need the high-level `pickle` module (and `copyreg`) are guarded
# by `HAVE_PICKLE`: importing `pickle` pulls in `re`, whose `_sre.MAGIC` is
# asserted against the stdlib version, so an unpinned stdlib (the bare
# extra_tests runner falling back to the host's older stdlib) skips them rather
# than failing — the same coupling as synth/sre_pattern_methods. Run with
# PYRE_STDLIB=lib-python/3 for the full set.
import io
import _pickle

try:
    import copyreg
    import pickle

    HAVE_PICKLE = True
except (ImportError, AssertionError):
    HAVE_PICKLE = False


def dumps(obj, proto, **kw):
    buf = io.BytesIO()
    _pickle.Pickler(buf, proto, **kw).dump(obj)
    return buf.getvalue()


def loads(data, **kw):
    return _pickle.Unpickler(io.BytesIO(data), **kw).load()


def roundtrip(obj, proto):
    got = loads(dumps(obj, proto))
    assert got == obj, (proto, repr(obj), repr(got))
    assert type(got) is type(obj), (proto, repr(obj), type(got), type(obj))
    return got


# ════════════════════════ atoms (protocol 2-5) ════════════════════════
# None / bool / int (incl. bignum) / float / str / bytes round-trip.
ATOM_CASES = [
    None, True, False,
    0, 1, -1, 255, 256, 65535, 65536,
    2 ** 31 - 1, -(2 ** 31), 2 ** 31, 2 ** 63, 2 ** 100, -(2 ** 100),
    3.14, -0.0, 1e308,
    "", "a", "hello", "유니코드",
    b"", b"abc", b"\x00\xff" * 10,
]
for proto in range(2, 6):
    for obj in ATOM_CASES:
        roundtrip(obj, proto)

# Wire-format lock — byte-identical to CPython 3.14.
assert dumps(None, 4) == b"\x80\x04N.", dumps(None, 4)
assert dumps("a", 4) == b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00\x8c\x01a\x94.", dumps("a", 4)
assert dumps(1, 2) == b"\x80\x02K\x01.", dumps(1, 2)


# ════════════════════ containers (tuple / list / dict) ════════════════════
# Memo back-references plus APPENDS / SETITEMS batching, nesting, shared
# references and recursion.
CONTAINER_CASES = [
    (), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5),
    [], [1, 2, 3], [1, [2, 3], 4],
    {}, {1: 2}, {1: 2, 3: 4}, {"a": 1, "b": [1, 2, 3]},
    [(1, 2), (3, 4)], {1: {2: {3: 4}}},
    [None, True, False, 3.5, b"xy", "uv"],
    list(range(2500)),                 # APPENDS batching (> _BATCHSIZE)
    {i: i * i for i in range(2500)},   # SETITEMS batching
]
for proto in range(2, 6):
    for obj in CONTAINER_CASES:
        roundtrip(obj, proto)

# Shared reference: the same inner list is reachable twice by identity, so the
# second reference must be a memo GET the unpickler restores to one object.
inner = [1, 2]
outer = [inner, inner]
got = loads(dumps(outer, 4))
assert got == outer
assert got[0] is got[1], "shared reference identity not preserved"

# Recursive list: the memo lets the unpickler close the cycle.
rec = []
rec.append(rec)
got = loads(dumps(rec, 4))
assert got[0] is got, "recursive list not reconstructed"

# Wire-format locks — byte-identical to CPython 3.14.
assert dumps((1, 2, 3), 4) == b"\x80\x04\x95\x09\x00\x00\x00\x00\x00\x00\x00K\x01K\x02K\x03\x87\x94.", dumps((1, 2, 3), 4)
assert dumps([1, 2, 3], 2) == b"\x80\x02]q\x00(K\x01K\x02K\x03e.", dumps([1, 2, 3], 2)
assert dumps({1: 2}, 2) == b"\x80\x02}q\x00K\x01K\x02s.", dumps({1: 2}, 2)


# ════════════════════════ set / frozenset / bytearray ════════════════════════
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


# ════════════════════════ reused-Pickler memo persistence ════════════════════════
# A reused Pickler keeps its memo across dump() calls (until clear_memo), so a
# second dump of the same object emits a back-reference; the two unpickled
# objects share identity. The Unpickler memo persists symmetrically.
for proto in range(2, 6):
    shared = ["x", "y"]
    buf = io.BytesIO()
    p = _pickle.Pickler(buf, proto)
    p.dump(shared)
    len1 = buf.tell()
    p.dump(shared)  # second dump should reference the memoized list
    len2 = buf.tell() - len1
    # The back-referenced second pickle is much smaller than the first.
    assert len2 < len1, (proto, len1, len2)

    # Load both from the same stream; identity is preserved across dumps.
    buf.seek(0)
    up = _pickle.Unpickler(buf)
    a = up.load()
    b = up.load()
    assert a is b, (proto, "cross-dump identity lost")
    assert a == ["x", "y"], (proto, a)

    # clear_memo() makes the next dump independent (full copy again).
    buf2 = io.BytesIO()
    p2 = _pickle.Pickler(buf2, proto)
    p2.dump(shared)
    n1 = buf2.tell()
    p2.clear_memo()
    p2.dump(shared)
    n2 = buf2.tell() - n1
    # After clear_memo the object is serialized in full again.
    assert n2 == n1, (proto, n1, n2)


# ════════════════════════ reduce protocol (arbitrary objects) ════════════════════════
# __dict__, __slots__, __getstate__/__setstate__, __reduce__, classes by
# reference, shared-instance identity, and the protocol < 3 codecs.encode bytes
# reduce. Instance pickles are not byte-identical (qualname interning differs),
# so this asserts roundtrip + type, not the wire.
class Plain:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, o):
        return type(self) is type(o) and self.__dict__ == o.__dict__


class Slotted:
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b

    def __eq__(self, o):
        return type(self) is type(o) and self.a == o.a and self.b == o.b


class Mixed:
    __slots__ = ("s", "__dict__")

    def __init__(self, s=0, d=0):
        self.s = s
        self.d = d

    def __eq__(self, o):
        return type(self) is type(o) and self.s == o.s and self.d == o.d


class WithState:
    def __init__(self, v=0):
        self.v = v
        self.cache = None

    def __getstate__(self):
        return {"v": self.v}

    def __setstate__(self, st):
        self.v = st["v"]
        self.cache = "rebuilt"

    def __eq__(self, o):
        return type(self) is type(o) and self.v == o.v


class WithReduce:
    def __init__(self, n):
        self.n = n

    def __reduce__(self):
        return (WithReduce, (self.n,))

    def __eq__(self, o):
        return type(self) is type(o) and self.n == o.n


# __getnewargs_ex__ with keyword args: protocol >= 4 emits NEWOBJ_EX with a
# non-empty kwargs dict; protocols 2/3 encode the constructor as
# partial(cls.__new__, cls, *args, **kwargs). Either way the unpickler must end
# up calling cls.__new__(cls, *a, **kw). A class-level sink records what __new__
# received (the __dict__ state then overwrites the instance attrs, so the sink
# is the only witness of kwargs).
class NewArgsEx:
    seen = []

    def __new__(cls, *args, **kwargs):
        cls.seen.append((args, dict(kwargs)))
        return super().__new__(cls)

    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b

    def __getnewargs_ex__(self):
        return ((self.a,), {"b": self.b})

    def __eq__(self, o):
        return type(self) is type(o) and self.a == o.a and self.b == o.b


for proto in range(2, 6):
    roundtrip(Plain(1, 2), proto)
    roundtrip(Slotted(3, 4), proto)
    roundtrip(Mixed(5, 6), proto)
    roundtrip(WithReduce(9), proto)

    # __setstate__ side-effect runs on load.
    g = roundtrip(WithState(7), proto)
    assert g.cache == "rebuilt", (proto, g.cache)

    # classes pickle by reference.
    assert loads(dumps(Plain, proto)) is Plain, proto

    # nested containers + shared-instance identity.
    pt = Plain(3, 4)
    g = roundtrip([pt, pt, {"k": Slotted(5, 6)}], proto)
    assert g[0] is g[1], (proto, "shared identity lost")

# protocol 2 routes bytes through the codecs.encode(s, 'latin1') reduce.
assert loads(dumps(b"\x00\xff\x80", 2)) == b"\x00\xff\x80"

# range reduces to range(start, stop, step) and roundtrips at all protos.
for proto in range(2, 6):
    assert roundtrip(range(2, 10, 3), proto) == range(2, 10, 3)

for proto in range(2, 6):
    NewArgsEx.seen.clear()
    roundtrip(NewArgsEx(1, 2), proto)
    # The load-time __new__ got the keyword arg from __getnewargs_ex__.
    assert ((1,), {"b": 2}) in NewArgsEx.seen, (proto, NewArgsEx.seen)


# ════════════════════════ legacy text opcodes (protocol 0 / 1) ════════════════════════
# INT / LONG / FLOAT / UNICODE plus the MARK-based LIST / DICT / TUPLE and text
# PUT / GET, persistent_id, and the _compat_pickle fix_imports name mapping.
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

# bytes / bytearray at protocol < 3 reduce through _codecs.encode; the global
# reference resolves the encode function's __module__ (_codecs), so the wire is
# byte-identical to CPython 3.14.
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
_legacy_registry = {"shared": ["one", "two"]}


class IdPickler(_pickle.Pickler):
    def persistent_id(self, obj):
        for key, value in _legacy_registry.items():
            if value is obj:
                return key
        return None


class IdUnpickler(_pickle.Unpickler):
    def persistent_load(self, pid):
        return _legacy_registry[pid]


for proto in (0, 2, 5):
    buf = io.BytesIO()
    shared = _legacy_registry["shared"]
    IdPickler(buf, proto).dump([shared, shared, "tail"])
    got = IdUnpickler(io.BytesIO(buf.getvalue())).load()
    assert got[0] is shared, (proto, "persistent_load object identity")
    assert got[0] is got[1], (proto, "shared persistent reference")
    assert got[2] == "tail", proto


# ════════════════════════ framing (multi-frame + large payloads) ════════════════════════
# The framer flushes a frame once it reaches the 64 KiB target, and large
# bytes / str / bytearray payloads are written outside any frame
# (write_large_bytes). Output is byte-identical to CPython 3.14 (cross-checked
# via the FNV-1a checksum printed below).
def cksum(b):
    h = 14695981039346656037
    for x in b:
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def report(tag, data):
    print(tag, len(data), data.count(b"\x95"), cksum(data))


# A pickle large enough to cross the 64 KiB frame target produces more than one
# frame (FRAME opcode = 0x95).
big_list = list(range(40000))
data = dumps(big_list, 5)
assert loads(data) == big_list
assert data.count(b"\x95") >= 2, data.count(b"\x95")
report("list5", data)

# Large bytes (>= 64 KiB) are written outside any frame; the surrounding stream
# still frames the PROTO/STOP scaffolding.
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

# Protocol 4 frames the same way.
data = dumps(big_list, 4)
assert loads(data) == big_list
assert data.count(b"\x95") >= 2
report("list4", data)

# A small pickle is a single frame (or none below FRAME_SIZE_MIN).
data = dumps([1, 2, 3], 5)
assert loads(data) == [1, 2, 3]
assert data.count(b"\x95") == 1
report("small5", data)


# ════════════════════════ protocol-5 out-of-band buffers ════════════════════════
# A PickleBuffer serializes in-band by default (BINBYTES read-only,
# BYTEARRAY8 mutable, both memoized) and out-of-band when a buffer_callback
# returns a false value (NEXT_BUFFER, plus READONLY_BUFFER for a read-only
# buffer). Wire format is byte-identical to CPython 3.14. Wrapped in a function
# so its PickleBuffer/Pickler selection stays isolated from the shared helpers.
def _run_oob():
    # pyre exposes PickleBuffer through __pypy__ and the accelerator as
    # _pickle; avoid `import pickle` so the test does not depend on the bundled
    # stdlib (re/_sre) being pinned. CPython has neither, so fall back to the
    # public pickle.
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
    # READONLY_BUFFER, callback invoked once, data not in the stream.
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

    # Shared identity: a PickleBuffer saved in-band twice memoizes (GET).
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

    # raw() exposes a memoryview onto the wrapped buffer; release() invalidates.
    pb = PickleBuffer(b"raw bytes")
    assert as_bytes(pb.raw()) == b"raw bytes"
    pb.release()
    try:
        pb.raw()
        raise AssertionError("expected error after release")
    except ValueError:
        pass


_run_oob()


# ════════════════════════ fix_imports gating ════════════════════════
# fix_imports gates the _compat_pickle name remap that protocols < 3 apply
# between Python 2 and Python 3 global names. It defaults to True and is a no-op
# at protocol >= 3.
assert dumps(len, 2) == b"\x80\x02c__builtin__\nlen\nq\x00."
assert dumps(len, 2, fix_imports=True) == b"\x80\x02c__builtin__\nlen\nq\x00."
assert dumps(len, 2, fix_imports=False) == b"\x80\x02cbuiltins\nlen\nq\x00."
# protocol 3 writes the Python 3 name verbatim regardless of fix_imports.
assert dumps(len, 3) == b"\x80\x03cbuiltins\nlen\nq\x00."
assert dumps(len, 3, fix_imports=False) == b"\x80\x03cbuiltins\nlen\nq\x00."

# Module-level _pickle.dumps mirrors the class behavior.
assert _pickle.dumps(len, 2) == b"\x80\x02c__builtin__\nlen\nq\x00."
assert _pickle.dumps(len, 2, fix_imports=False) == b"\x80\x02cbuiltins\nlen\nq\x00."

# load: fix_imports gates the forward (py2 -> py3) remap.
py2_stream = dumps(len, 2, fix_imports=True)   # __builtin__\nlen
py3_stream = dumps(len, 2, fix_imports=False)  # builtins\nlen
assert loads(py2_stream, fix_imports=True) is len
assert loads(py2_stream) is len  # default True
assert loads(py3_stream, fix_imports=False) is len
assert _pickle.loads(py2_stream) is len
assert _pickle.loads(py3_stream, fix_imports=False) is len

# A Python 2 module name with fix_imports=False is resolved literally, so the
# nonexistent __builtin__ module fails to import.
try:
    loads(py2_stream, fix_imports=False)
    raise AssertionError("expected an import failure")
except ImportError:
    pass


# ════════════════════════ streaming to the file (write() calls) ════════════════════════
# The Pickler streams to the file's write() as the pickle is produced. We pin
# the exact number of write() calls and their byte split (CPython's suite only
# checks len(chunks) > 1), plus GC safety: a write() that allocates heavily
# forces the moving GC to relocate objects at each streaming point.
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
    assert len(calls) == expected_writes, (proto, len(calls), expected_writes)
    return calls


# Small object: a single write at every protocol.
check([1, 2, 3], 0, 1)
check([1, 2, 3], 2, 1)
check([1, 2, 3], 5, 1)

# Many small objects overflow one frame (protocol >= 4): the first frame is
# committed mid-dump, the rest at the end — two writes. Unframed: one write.
stream_big = list(range(40000))
two = check(stream_big, 5, 2)
# Each committed write carries a FRAME opcode (0x95) at protocol 5.
assert two[0][2] == 0x95, two[0][:3]  # after PROTO 5 (2 bytes)
check(stream_big, 2, 1)

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
_pickle.dump(stream_big, rec, 5)
assert len(rec.calls) == 2


# GC safety: a write() that allocates heavily runs at each streaming point while
# the save tree holds live object pointers across the boundary.
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


# ════════════════════════ memo proxy (Pickler.memo / Unpickler.memo) ════════════════════════
# memo exposes a fresh proxy on every access. pickletester covers priming /
# clearing; what it does NOT pin and we keep: a fresh proxy per access, the
# exact setter TypeError/ValueError messages and that deletion raises, and that
# assigning a plain dict to Unpickler.memo yields an EMPTY memo.
def err(fn):
    try:
        fn()
        return None
    except Exception as e:  # noqa: BLE001
        return e


# Pickler.memo proxy
buf = io.BytesIO()
p = _pickle.Pickler(buf, 5)
p.dump(["a", "b"])

assert type(p.memo).__name__ == "PicklerMemoProxy", type(p.memo).__name__
# A fresh proxy is handed back on each access.
assert p.memo is not p.memo

# clear() empties the pickler's memo.
p.memo.clear()
assert p.memo.copy() == {}

# Wrong type / bad value shape / deletion are rejected.
e = err(lambda: setattr(p, "memo", [1, 2]))
assert isinstance(e, TypeError) and "PicklerMemoProxy object or dict" in str(e), e
e = err(lambda: setattr(p, "memo", {1: 2}))
assert isinstance(e, TypeError) and "2-item tuples" in str(e), e
# The memo is a position-indexed list, so a negative slot index is rejected
# rather than panicking on the out-of-range cast.
e = err(lambda: setattr(p, "memo", {0: (-1, "x")}))
assert isinstance(e, ValueError) and "non-negative" in str(e), e
e = err(lambda: delattr(p, "memo"))
assert isinstance(e, TypeError) and "deletion is not supported" in str(e), e

# Unpickler.memo proxy
memo_data = dumps(["x", "y", ["x"]], 5)
u = _pickle.Unpickler(io.BytesIO(memo_data))
u.load()

assert type(u.memo).__name__ == "UnpicklerMemoProxy", type(u.memo).__name__
assert u.memo is not u.memo

u.memo.clear()
assert u.memo.copy() == {}

# Assigning a plain dict validates the keys but yields an EMPTY memo.
u3 = _pickle.Unpickler(io.BytesIO(memo_data))
u3.memo = {0: "a", 1: "b"}
assert u3.memo.copy() == {}

e = err(lambda: setattr(u3, "memo", [1, 2]))
assert isinstance(e, TypeError) and "UnpicklerMemoProxy object or dict" in str(e), e
e = err(lambda: setattr(u3, "memo", {"k": 1}))
assert isinstance(e, TypeError) and "memo key must be integers" in str(e), e
e = err(lambda: setattr(u3, "memo", {-1: 1}))
assert isinstance(e, ValueError) and "positive integers" in str(e), e
e = err(lambda: delattr(u3, "memo"))
assert isinstance(e, TypeError) and "deletion is not supported" in str(e), e


# ════════════════════════ lazy builtin iterators ════════════════════════
# reversed / filter / map / zip pickle through the reduce protocol; the
# captured sub-iterators carry their positions, so a partially consumed
# iterator resumes where it left off.
assert list(loads(dumps(reversed([1, 2, 3]), 2))) == [3, 2, 1]
assert list(loads(dumps(filter(None, [0, 1, 2, 0, 3]), 2))) == [1, 2, 3]
assert list(loads(dumps(map(str, [1, 2, 3]), 2))) == ["1", "2", "3"]
assert list(loads(dumps(zip([1, 2], [3, 4]), 2))) == [(1, 3), (2, 4)]
# strict= survives the round-trip via __setstate__.
zr = loads(dumps(zip([1, 2], [3], strict=True), 2))
try:
    list(zr)
    raise AssertionError("strict flag lost on unpickle")
except ValueError:
    pass
# partial consumption resumes at the right spot.
_it = map(str, [1, 2, 3, 4])
next(_it)
assert list(loads(dumps(_it, 2))) == ["2", "3", "4"]


# ════════════════════════ high-level pickle facade (HAVE_PICKLE) ════════════════════════
# The remaining sections exercise the public `pickle` module, copyreg extension
# codes, reducer_override / dispatch_table / fast mode / find_class, and the
# persistent_id instance hook — all of which need `import pickle`.
if HAVE_PICKLE:
    # ── THE FLIP: public pickle resolves names from the accelerator ──
    # All public names come from the accelerator.
    for name in ("Pickler", "Unpickler", "dump", "dumps", "load", "loads"):
        assert hasattr(pickle, name), name
    assert pickle.Pickler.__module__ == "_pickle", pickle.Pickler.__module__
    assert pickle.Unpickler.__module__ == "_pickle", pickle.Unpickler.__module__

    # PicklingError / UnpicklingError subclass PickleError.
    assert issubclass(pickle.PicklingError, pickle.PickleError)
    assert issubclass(pickle.UnpicklingError, pickle.PickleError)
    assert issubclass(pickle.PickleError, Exception)

    assert pickle.HIGHEST_PROTOCOL == 5, pickle.HIGHEST_PROTOCOL
    assert pickle.DEFAULT_PROTOCOL in (4, 5), pickle.DEFAULT_PROTOCOL

    def pk_roundtrip(obj, proto):
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
            pk_roundtrip(obj, proto)

    # dump / load through a file object.
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

    # A buffer_callback that is never invoked leaves pickling unchanged.
    calls = []
    assert pickle.loads(pickle.dumps(b"data", 5, buffer_callback=calls.append)) == b"data"
    assert calls == []

    # Protocol-5 out-of-band buffers via the public API.
    assert pickle._HAVE_PICKLE_BUFFER
    assert pickle.PickleBuffer is not None
    bufs = []
    pb_data = pickle.dumps(pickle.PickleBuffer(b"oob payload"), 5,
                           buffer_callback=lambda b: bufs.append(b) or False)
    assert len(bufs) == 1
    restored = pickle.loads(pb_data, buffers=[b"oob payload"])
    restored = restored if isinstance(restored, (bytes, bytearray)) else restored.tobytes()
    assert restored == b"oob payload", restored

    # ── follow-ups: legacy strings, DUP, LONG4, EXT codes, hooks ──
    # legacy STRING / BINSTRING / SHORT_BINSTRING (Python 2 wire), decoded with
    # the unpickler's encoding (default "ASCII").
    assert loads(b"S'abc'\n.") == "abc"
    assert loads(b'S"abc"\n.') == "abc"
    assert loads(b"S'abc'\n.", encoding="bytes") == b"abc"
    assert loads(b"S'a\\nb'\n.") == "a\nb"
    assert loads(b"S'\\x41'\n.") == "A"
    assert loads(b"U\x03abc.") == "abc"
    assert loads(b"U\x03abc.", encoding="bytes") == b"abc"
    assert loads(b"T\x03\x00\x00\x00abc.") == "abc"
    try:
        loads(b"Sabc\n.")
        raise AssertionError("unquoted STRING accepted")
    except pickle.UnpicklingError:
        pass

    # DUP duplicates the top of stack (same object).
    g = loads(b"(]2t.")
    assert g == ([], [])
    assert g[0] is g[1]

    # LONG4 negative byte count is rejected.
    try:
        loads(b"\x8b\xff\xff\xff\xff.")
        raise AssertionError("negative LONG4 accepted")
    except pickle.UnpicklingError as e:
        assert "negative byte count" in str(e), e

    # copyreg extension codes (EXT1 / EXT2 / EXT4).
    class Ext1:
        pass

    class Ext2:
        pass

    class Ext4:
        pass

    mod = __name__
    for cls, code in ((Ext1, 0xF0), (Ext2, 0x1234), (Ext4, 0x12345)):
        copyreg.add_extension(mod, cls.__name__, code)
    try:
        for cls in (Ext1, Ext2, Ext4):
            assert loads(dumps(cls, 2)) is cls, cls.__name__
            assert loads(dumps(cls, 4)) is cls, cls.__name__
    finally:
        for cls, code in ((Ext1, 0xF0), (Ext2, 0x1234), (Ext4, 0x12345)):
            copyreg.remove_extension(mod, cls.__name__, code)

    # reducer_override on a Pickler subclass.
    class Wrapped:
        def __init__(self, v):
            self.v = v

    def rebuild_wrapped(v):
        return Wrapped(v)

    class OverridePickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, Wrapped):
                # +100 marks that the override (not the default reduce) ran.
                return (rebuild_wrapped, (obj.v + 100,))
            return NotImplemented

    buf = io.BytesIO()
    OverridePickler(buf, 2).dump(Wrapped(5))
    assert pickle.loads(buf.getvalue()).v == 105

    # dispatch_table (per-pickler reduce override).
    class Boxed:
        def __init__(self, x):
            self.x = x

    def reduce_boxed(obj):
        return (Boxed, (obj.x,))

    buf = io.BytesIO()
    p = pickle.Pickler(buf, 2)
    # Unset by default: reading it raises AttributeError (T_OBJECT_EX member).
    try:
        p.dispatch_table
        raise AssertionError("dispatch_table readable when unset")
    except AttributeError:
        pass
    p.dispatch_table = {Boxed: reduce_boxed}
    p.dump(Boxed(7))
    assert pickle.loads(buf.getvalue()).x == 7

    # dispatch_table may be any mapping, consulted via __getitem__ (a missing
    # type surfaces as KeyError == no entry).
    class MappingDT:
        def __init__(self, d):
            self._d = d

        def __getitem__(self, key):
            return self._d[key]

    buf = io.BytesIO()
    p = pickle.Pickler(buf, 2)
    p.dispatch_table = MappingDT({Boxed: reduce_boxed})
    p.dump(Boxed(9))
    assert pickle.loads(buf.getvalue()).x == 9

    # fast mode disables the memo (no shared identity).
    shared = [1, 2, 3]
    buf = io.BytesIO()
    pf = pickle.Pickler(buf, 2)
    assert pf.fast == 0
    pf.fast = 1
    assert pf.fast == 1
    pf.dump([shared, shared])
    fast_g = pickle.loads(buf.getvalue())
    assert fast_g == [[1, 2, 3], [1, 2, 3]]
    assert fast_g[0] is not fast_g[1]
    buf = io.BytesIO()
    pickle.Pickler(buf, 2).dump([shared, shared])
    slow_g = pickle.loads(buf.getvalue())
    assert slow_g[0] is slow_g[1]

    # Unpickler.find_class override + super().
    class GuardedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("builtins", "eval"):
                raise pickle.UnpicklingError("eval blocked")
            return super().find_class(module, name)

    try:
        GuardedUnpickler(io.BytesIO(pickle.dumps(eval, 4))).load()
        raise AssertionError("eval was not blocked")
    except pickle.UnpicklingError as e:
        assert "eval blocked" in str(e), e
    assert GuardedUnpickler(io.BytesIO(pickle.dumps(len, 4))).load() is len

    # dump-time verification of global resolution: a function-local class
    # cannot be referenced by a dotted path.
    def make_local():
        class Local:
            pass

        return Local

    try:
        dumps(make_local(), 2)
        raise AssertionError("local class pickled")
    except pickle.PicklingError:
        pass

    # An object whose name resolves to a different object is rejected.
    class Shadow:
        pass

    _real_shadow = Shadow
    globals()["Shadow"] = "not the class"
    try:
        dumps(_real_shadow, 2)
        raise AssertionError("shadowed class pickled")
    except pickle.PicklingError as e:
        assert "not the same object" in str(e) or "not found" in str(e), e
    finally:
        globals()["Shadow"] = _real_shadow

    # ── persistent_id / persistent_load: the external-object hook ──
    class Ref:
        def __init__(self, name):
            self.name = name

    pid_registry = {"alpha": Ref("alpha"), "beta": Ref("beta")}
    alpha = pid_registry["alpha"]

    def pid(obj):
        if isinstance(obj, Ref):
            return obj.name
        return None

    def pload(key):
        return pid_registry[key]

    # Instance-attr hooks are settable and round-trip by reference.
    buf = io.BytesIO()
    pp = pickle.Pickler(buf, 2)
    pp.persistent_id = pid
    assert pp.persistent_id is pid
    pp.dump([alpha, 42])
    uu = pickle.Unpickler(io.BytesIO(buf.getvalue()))
    uu.persistent_load = pload
    got = uu.load()
    assert got[0] is alpha
    assert got[1] == 42

    # A persistent reference with no persistent_load is rejected.
    buf = io.BytesIO()
    pp = pickle.Pickler(buf, 2)
    pp.persistent_id = pid
    pp.dump([alpha])
    try:
        pickle.Unpickler(io.BytesIO(buf.getvalue())).load()
        raise AssertionError("expected UnpicklingError without persistent_load")
    except pickle.UnpicklingError as e:
        assert "persistent_load" in str(e), e

    # Delete resets the instance hook: alpha is then pickled by value.
    buf = io.BytesIO()
    p2 = pickle.Pickler(buf, 2)
    p2.persistent_id = pid
    del p2.persistent_id
    p2.dump([alpha])
    back = pickle.loads(buf.getvalue())
    assert isinstance(back[0], Ref) and back[0].name == "alpha"


print("_pickle OK")
