# CPython-suite gap: no test stores attributes, slots, weakrefs or __del__ on enumerate/map/filter/zip/reversed/super/property subclasses.
# parity-tests reason: this targets the typedef.py _getusercls mapdict storage on those iterator and descriptor layouts.

"""Subclass instances of enumerate/map/filter/zip/reversed/super/property.

`typedef.py` `_getusercls` allocates them with mapdict storage: attributes,
`__dict__`, slots, weakrefs and `__del__` live on that layout.
"""
import gc
import weakref


def exc_name(fn):
    try:
        fn()
    except Exception as e:
        return type(e).__name__
    else:
        return "ok"


def plain(label, base, make, iterate):
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
    if iterate is None:
        produced, same = "skip", "skip"
    else:
        produced = list(iterate(S))
        same = produced == list(iterate(base))
    print(
        "plain",
        label,
        type(x).__name__,
        isinstance(x, base),
        got,
        reflected,
        missing,
        gone,
        alive,
        produced,
        same,
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
        type(x).__name__,
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


class A(object):
    pass


class B(A):
    pass


b = B()


def make_enum(cls):
    return cls("ab")


def make_map(cls):
    return cls(lambda z: z + 1, [1, 2])


def make_filter(cls):
    return cls(None, [0, 1, 2])


def make_zip(cls):
    return cls([1, 2], [3, 4])


def make_rev(cls):
    return cls((1, 2, 3))


def make_super(cls):
    return cls(B, b)


def make_prop(cls):
    return cls()


cases = [
    ("enumerate", enumerate, make_enum, make_enum),
    ("map", map, make_map, make_map),
    ("filter", filter, make_filter, make_filter),
    ("zip", zip, make_zip, make_zip),
    ("reversed", reversed, make_rev, make_rev),
    ("super", super, make_super, None),
    ("property", property, make_prop, None),
]
for label, base, make, iterate in cases:
    plain(label, base, make, iterate)
    slots(label, base, make)
    finalizer(label, base, make)
