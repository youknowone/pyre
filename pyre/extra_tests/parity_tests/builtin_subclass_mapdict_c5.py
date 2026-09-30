# CPython-suite gap: subclass instances of deque, Struct, Pickler, Unpickler,
# GenericAlias and a big int still stored __dict__ beside the builtin typeptr.
# parity-tests reason: typedef.py _getusercls mapdict storage on those layouts.

"""Subclass instances allocate through objspace.py allocate_instance.

A plain subclass and a `__slots__` subclass keep `type(x) is S`, attribute
storage, and a working base operation. `__del__` runs after `gc.collect()`.
"""
import gc
import io
import weakref
import _pickle
import _struct
from collections import deque
from types import GenericAlias


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


def make_deque(cls):
    return cls()


def operate_deque(x):
    x.append(1)
    x.append(2)
    return (x.pop(), x.popleft(), len(x))


def make_struct(cls):
    return cls("h")


def operate_struct(x):
    return x.unpack(x.pack(7)) == (7,)


_BUFFERS = []


def make_pickler(cls):
    buf = io.BytesIO()
    _BUFFERS.append(buf)
    return cls(buf)


def operate_pickler(x):
    buf = _BUFFERS[-1]
    x.dump([1, "a"])
    buf.seek(0)
    return _pickle.Unpickler(buf).load() == [1, "a"]


def make_unpickler(cls):
    buf = io.BytesIO()
    _pickle.Pickler(buf).dump((2, "b"))
    buf.seek(0)
    _BUFFERS.append(buf)
    return cls(buf)


def operate_unpickler(x):
    return x.load() == (2, "b")


def make_alias(cls):
    return cls(list, (int,))


def operate_alias(x):
    return (x.__origin__ is list, x.__args__ == (int,))


BIG = 10**30


def make_int(cls):
    return cls(BIG)


def operate_int(x):
    return (x + 1 - 1 == BIG, int(x) == BIG)


plain("deque", deque, make_deque, operate_deque)
slots("deque", deque, make_deque)
finalizer("deque", deque, make_deque)

plain("struct", _struct.Struct, make_struct, operate_struct)
slots("struct", _struct.Struct, make_struct)
finalizer("struct", _struct.Struct, make_struct)

plain("pickler", _pickle.Pickler, make_pickler, operate_pickler)
slots("pickler", _pickle.Pickler, make_pickler)
finalizer("pickler", _pickle.Pickler, make_pickler)

plain("unpickler", _pickle.Unpickler, make_unpickler, operate_unpickler)
slots("unpickler", _pickle.Unpickler, make_unpickler)
finalizer("unpickler", _pickle.Unpickler, make_unpickler)

plain("alias", GenericAlias, make_alias, operate_alias)
slots("alias", GenericAlias, make_alias)
finalizer("alias", GenericAlias, make_alias)

plain("int", int, make_int, operate_int)
slots("int", int, make_int)
finalizer("int", int, make_int)


class P(_pickle.Pickler):
    pass


class U(_pickle.Unpickler):
    pass


round_buf = io.BytesIO()
P(round_buf).dump((2, "b"))
round_buf.seek(0)
print("round", U(round_buf).load() == (2, "b"))

print("OK")
