# Direct test of the interp-level `_pickle` accelerator (increment 5):
# the reduce protocol — arbitrary objects (`__dict__`, `__slots__`,
# `__getstate__` / `__setstate__`, `__reduce__`), classes by reference,
# shared-instance identity, and the protocol < 3 `codecs.encode` bytes
# reduce. Instance pickles are not byte-identical to CPython (string
# interning of qualnames differs), so this asserts roundtrip + type, not
# the wire.
import io
import _pickle


def dumps(obj, proto):
    buf = io.BytesIO()
    _pickle.Pickler(buf, proto).dump(obj)
    return buf.getvalue()


def loads(data):
    return _pickle.Unpickler(io.BytesIO(data)).load()


def roundtrip(obj, proto):
    got = loads(dumps(obj, proto))
    assert got == obj, (proto, repr(obj), repr(got))
    assert type(got) is type(obj), (proto, type(obj), type(got))
    return got


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

# protocol 2 routes bytes through the `codecs.encode(s, 'latin1')` reduce.
assert loads(dumps(b"\x00\xff\x80", 2)) == b"\x00\xff\x80"

# range reduces to `range(start, stop, step)` and roundtrips at all protos.
for proto in range(2, 6):
    assert roundtrip(range(2, 10, 3), proto) == range(2, 10, 3)


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
    NewArgsEx.seen.clear()
    roundtrip(NewArgsEx(1, 2), proto)
    # The load-time __new__ got the keyword arg from __getnewargs_ex__.
    assert ((1,), {"b": 2}) in NewArgsEx.seen, (proto, NewArgsEx.seen)

print("_pickle_objects OK")
