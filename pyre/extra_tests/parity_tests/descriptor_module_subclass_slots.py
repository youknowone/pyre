"""User subclasses of staticmethod, classmethod and module.

typedef.py `_getusercls` gives each a MapdictStorageMixin. The base typedef
is hasdict, so an ordinary attribute stays in the typed `__dict__` and a
`__slots__` name does not.
"""

import gc
import types


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
    x.b = 3
    got = x.b
    in_dict = "b" in x.__dict__
    del x.b
    missing = exc_name(lambda: x.b)
    gone = "b" in x.__dict__
    print(
        "plain",
        label,
        type(x) is S,
        isinstance(x, base),
        got,
        in_dict,
        missing,
        gone,
        operate(x),
    )


def slots(label, base, make, operate):
    class S(base):
        __slots__ = ("a",)

    x = make(S)
    x.a = 2
    got = x.a
    in_dict = "a" in x.__dict__
    del x.a
    missing = exc_name(lambda: x.a)
    gone = "a" in x.__dict__
    print(
        "slots",
        label,
        type(x) is S,
        isinstance(x, base),
        got,
        in_dict,
        missing,
        gone,
        operate(x),
    )


def finalizer(label, base, make):
    seen = []

    class D(base):
        def __del__(self):
            seen.append(label)

    def spawn():
        make(D)

    spawn()
    gc.collect()
    print("del", label, seen)


def bind_len(x):
    class C:
        f = x

    return C.f("ab")


def make_sm(cls):
    return cls(len)


def make_cm(cls):
    return cls(lambda cls, s: len(s))


def make_mod(cls):
    return cls("modname")


def operate_mod(x):
    return x.__name__


for label, base, make, operate in (
    ("staticmethod", staticmethod, make_sm, bind_len),
    ("classmethod", classmethod, make_cm, bind_len),
    ("module", types.ModuleType, make_mod, operate_mod),
):
    plain(label, base, make, operate)
    slots(label, base, make, operate)
    finalizer(label, base, make)

print("OK")
