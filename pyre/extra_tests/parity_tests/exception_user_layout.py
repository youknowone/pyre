# CPython-suite gap: no CPython test checks that a slotted exception
# subclass keeps the slot out of `__dict__` while still raising, pickling,
# copying, and finalizing.
# parity-tests reason: typedef.py _getusercls / objspace.py allocate_instance
# on W_BaseException and the extra-field realbases.

"""Exception instances follow allocate_instance.

An exact realbase stays on its interp class. Every other builtin exception
and every app-level subclass uses that realbase's `_getusercls` layout.
`__dict__` stays the typed dict; `__slots__` live on the map.
"""

import copy
import gc
import pickle
import weakref


CASES = (
    ("Exception", Exception, ("m",), {}),
    ("ValueError", ValueError, ("m",), {}),
    ("KeyError", KeyError, ("k",), {}),
    ("OSError", OSError, (1, "x"), {}),
    ("PermissionError", PermissionError, (1, "x"), {}),
    ("ImportError", ImportError, ("m",), {"name": "mod"}),
    ("StopIteration", StopIteration, (5,), {}),
    ("UnicodeDecodeError", UnicodeDecodeError, ("utf-8", b"\xff", 0, 1, "bad"), {}),
    ("SystemExit", SystemExit, (2,), {}),
    # KeyboardInterrupt is not an Exception, so this stays a BaseExceptionGroup
    # (an all-Exception payload would promote to ExceptionGroup).
    ("BaseExceptionGroup", BaseExceptionGroup, ("g", [KeyboardInterrupt()]), {}),
)


def publish(cls):
    cls.__module__ = __name__
    globals()[cls.__name__] = cls
    return cls


def construct(cls, args, kwargs):
    return cls(*args, **kwargs)


def extra(e):
    out = []
    if isinstance(e, OSError):
        out.append(("errno", e.errno, e.strerror, e.filename))
    if isinstance(e, ImportError):
        out.append(("name", e.name))
    if isinstance(e, StopIteration):
        out.append(("value", e.value))
    if isinstance(e, SystemExit):
        out.append(("code", e.code))
    if isinstance(e, BaseExceptionGroup):
        out.append(
            (
                "group",
                e.message,
                tuple(type(x).__name__ for x in e.exceptions),
            )
        )
    if isinstance(e, UnicodeDecodeError):
        out.append(("unicode", e.encoding, e.start, e.end, e.reason))
    return tuple(out)


def caught(cls, args, kwargs, handler):
    try:
        raise construct(cls, args, kwargs)
    except handler as e:
        return (type(e).__name__, e.__traceback__ is not None)


def exercise(tag, base, args, kwargs):
    plain = publish(type("P_" + tag, (base,), {}))
    slots = publish(type("S_" + tag, (base,), {"__slots__": ("a",)}))
    p = construct(plain, args, kwargs)
    p.x = 1
    p.add_note("n")
    s = construct(slots, args, kwargs)
    s.a = 1
    s.x = 2
    s.add_note("n")
    print(
        "plain",
        tag,
        type(p).__name__,
        isinstance(p, base),
        p.args,
        str(p),
        repr(p),
        sorted(p.__dict__),
        p.__notes__,
        extra(p),
    )
    print(
        "slots",
        tag,
        type(s).__name__,
        isinstance(s, base),
        s.args,
        str(s),
        repr(s),
        sorted(s.__dict__),
        "a" in s.__dict__,
        s.a,
        s.__notes__,
        extra(s),
    )
    print(
        "catch",
        tag,
        caught(plain, args, kwargs, plain),
        caught(plain, args, kwargs, base),
        caught(slots, args, kwargs, slots),
        caught(slots, args, kwargs, base),
    )
    try:
        try:
            raise ValueError("ctx")
        except ValueError:
            raise construct(plain, args, kwargs) from ValueError("cause")
    except plain as e:
        print(
            "chain",
            tag,
            None if e.__cause__ is None else type(e.__cause__).__name__,
            e.__context__ is not None,
            e.__traceback__ is not None,
            e.__suppress_context__,
        )
    try:
        try:
            raise ValueError("ctx")
        except ValueError:
            raise construct(plain, args, kwargs) from None
    except plain as e:
        print(
            "fromnone",
            tag,
            e.__cause__ is None,
            e.__context__ is not None,
            e.__suppress_context__,
        )
    fresh = construct(plain, args, kwargs)
    fresh.x = 3
    back = pickle.loads(pickle.dumps(fresh))
    print(
        "pickle",
        tag,
        type(back) is plain,
        back.args,
        sorted(back.__dict__),
        back.x,
        extra(back),
    )
    slotted = construct(slots, args, kwargs)
    slotted.a = 9
    slotted.x = 4
    back = pickle.loads(pickle.dumps(slotted))
    print(
        "spickle",
        tag,
        type(back) is slots,
        getattr(back, "a", None),
        back.x,
        sorted(back.__dict__),
    )
    copied = copy.copy(fresh)
    print(
        "copy",
        tag,
        type(copied) is plain,
        copied.args,
        sorted(copied.__dict__),
        copied is fresh,
        copied.x,
    )
    print("weak", tag, weakref.ref(p)() is p)


def finalizer(label, base, args, kwargs):
    seen = []

    class D(base):
        def __del__(self):
            seen.append(label)

    def spawn():
        return weakref.ref(D(*args, **kwargs))

    ref = spawn()
    for _ in range(8):
        gc.collect()
        if ref() is None:
            break
    print("del", label, ref() is None, seen)


for spec in CASES:
    exercise(*spec)

finalizer("Exception", Exception, ("m",), {})
finalizer("OSError", OSError, (1, "x"), {})
finalizer("BaseExceptionGroup", BaseExceptionGroup, ("g", [KeyboardInterrupt()]), {})
print("OK")
