# CPython-suite gap: no CPython test stores attributes, slots, weakrefs or
# __del__ on BytesIO/StringIO/Buffered*/TextIOWrapper subclass instances.
# parity-tests reason: typedef.py _getusercls mapdict storage on those layouts.

"""Subclass instances of the typed _io payloads allocate through _getusercls.

A plain subclass and a `__slots__` subclass keep `type(x) is S`, attribute
storage, and a working base operation. `__del__` runs after `gc.collect()`.
"""

import gc
import io
import weakref


def exc_name(fn):
    try:
        fn()
    except Exception as e:
        return type(e).__name__
    else:
        return "ok"


def plain(label, base, make, operate):
    class S(base):
        pass

    x = make(S)
    object.__setattr__(x, "attr", 1)
    got = object.__getattribute__(x, "attr")
    reflected = object.__getattribute__(x, "__dict__")["attr"]
    object.__delattr__(x, "attr")
    missing = exc_name(lambda: object.__getattribute__(x, "attr"))
    gone = "attr" in object.__getattribute__(x, "__dict__")
    print(
        "plain",
        label,
        type(x) is S,
        isinstance(x, base),
        got,
        reflected,
        missing,
        gone,
        weakref.ref(x)() is x,
        operate(x),
    )


def slots(label, base, make):
    class S(base):
        __slots__ = ("a",)

    x = make(S)
    object.__setattr__(x, "a", 2)
    got = object.__getattribute__(x, "a")
    object.__delattr__(x, "a")
    missing = exc_name(lambda: object.__getattribute__(x, "a"))
    extra = exc_name(lambda: object.__setattr__(x, "z", 1))
    dic = exc_name(lambda: object.__getattribute__(x, "__dict__"))
    print(
        "slots",
        label,
        type(x) is S,
        isinstance(x, base),
        got,
        missing,
        extra,
        dic,
    )


def finalizer(label, base, make):
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


def make_bytesio(cls):
    return cls(b"ab")


def operate_bytesio(x):
    return x.getvalue() == b"ab"


def make_stringio(cls):
    return cls("ab")


def operate_stringio(x):
    return x.getvalue() == "ab"


def make_reader(cls):
    return cls(io.BytesIO(b"ab"))


def operate_reader(x):
    return x.read() == b"ab"


def make_writer(cls):
    raw = io.BytesIO()
    x = cls(raw)
    x.write(b"ab")
    x.flush()
    return x


def operate_writer(x):
    x.flush()
    return True


def make_random(cls):
    return cls(io.BytesIO(b"ab"))


def operate_random(x):
    return x.read() == b"ab"


def make_rwpair(cls):
    return cls(io.BytesIO(b"ab"), io.BytesIO())


def operate_rwpair(x):
    return x.read() == b"ab"


def make_text(cls):
    return cls(io.BytesIO(b"ab"), encoding="utf-8")


def operate_text(x):
    return x.read() == "ab"


for label, base, make, operate in [
    ("BytesIO", io.BytesIO, make_bytesio, operate_bytesio),
    ("StringIO", io.StringIO, make_stringio, operate_stringio),
    ("BufferedReader", io.BufferedReader, make_reader, operate_reader),
    ("BufferedWriter", io.BufferedWriter, make_writer, operate_writer),
    ("BufferedRandom", io.BufferedRandom, make_random, operate_random),
    ("BufferedRWPair", io.BufferedRWPair, make_rwpair, operate_rwpair),
    ("TextIOWrapper", io.TextIOWrapper, make_text, operate_text),
]:
    plain(label, base, make, operate)
    slots(label, base, make)
    finalizer(label, base, make)

print("OK")
