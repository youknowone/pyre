# CPython-suite gap: no CPython test stores attributes, slots, weakrefs or __del__ on zlib/mmap/_random/_lsprof/_cffi_backend subclass instances.
# parity-tests reason: typedef.py _getusercls mapdict storage on those module layouts; output matches pypy3.

"""Subclass instances of the module types that accept subclassing.

`typedef.py` `_getusercls` allocates them with mapdict storage. Exact
instances keep the base layout and refuse arbitrary attributes.
"""

import gc
import mmap
import weakref
import zlib
import _lsprof
import _random

try:
    import _cffi_backend
except ImportError:
    _cffi_backend = None

COMPRESS_TYPE = type(zlib.compressobj())
DECOMPRESS_TYPE = type(zlib.decompressobj())


def exc_name(fn):
    try:
        fn()
    except Exception as e:
        return type(e).__name__
    else:
        return "ok"


def compress_ok(x):
    return zlib.decompress(x.compress(b"hello") + x.flush()) == b"hello"


def decompress_ok(x):
    return x.decompress(zlib.compress(b"hello")) == b"hello"


def mmap_ok(x):
    x.write(b"abcd")
    x.seek(0)
    return x.read(4) == b"abcd"


def random_ok(x):
    return x.random() == _random.Random(1).random()


def profiler_ok(x):
    x.enable()
    x.disable()
    return "endis"


def ffi_ok(x):
    return x.sizeof("int")


def make_compress(cls):
    if cls is COMPRESS_TYPE:
        return zlib.compressobj()
    return cls()


def make_decompress(cls):
    if cls is DECOMPRESS_TYPE:
        return zlib.decompressobj()
    return cls()


def check(label, base, make, method):
    exact = make(base)
    print("exact-set", label, exc_name(lambda: setattr(exact, "attr", 1)))

    try:
        class S(base):
            pass
    except TypeError as e:
        print("plain", label, "not-base", type(e).__name__)
        print("slots", label, "not-base")
        print("del", label, "not-base")
        return

    x = make(S)
    setattr(x, "attr", 1)
    got = x.attr
    reflected = x.__dict__["attr"]
    del x.attr
    missing = exc_name(lambda: x.attr)
    gone = "attr" in x.__dict__
    alive = weakref.ref(x)() is x
    print(
        "plain",
        label,
        type(x) is S,
        isinstance(x, base),
        got,
        reflected,
        missing,
        gone,
        alive,
        method(x),
    )

    class Sl(base):
        __slots__ = ("a",)

    s = make(Sl)
    setattr(s, "a", 2)
    slot = s.a

    def slots_del():
        del s.a

    print(
        "slots",
        label,
        type(s) is Sl,
        isinstance(s, base),
        slot,
        exc_name(slots_del),
        exc_name(lambda: s.a),
        exc_name(lambda: setattr(s, "z", 1)),
        exc_name(lambda: s.__dict__),
        method(s),
    )

    seen = []

    class D(base):
        def __del__(self):
            seen.append(label)

    def spawn():
        obj = make(D)
        return weakref.ref(obj)

    ref = spawn()
    gc.collect()
    print("del", label, ref() is None, seen)


check("compress", COMPRESS_TYPE, make_compress, compress_ok)
check("decompress", DECOMPRESS_TYPE, make_decompress, decompress_ok)
check("mmap", mmap.mmap, lambda cls: cls(-1, 16), mmap_ok)
check("random", _random.Random, lambda cls: cls(1), random_ok)
check("profiler", _lsprof.Profiler, lambda cls: cls(), profiler_ok)
if _cffi_backend is not None:
    check("ffi", _cffi_backend.FFI, lambda cls: cls(), ffi_ok)
print("OK")
