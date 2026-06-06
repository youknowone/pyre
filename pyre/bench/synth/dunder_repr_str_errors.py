# Exceptions raised by `__repr__`/`__str__` overrides propagate out of
# `repr()`/`str()`/`format`/`%`/f-strings instead of being swallowed,
# including builtin-leaf subclasses and through container recursion.
# Only the exception *type* is printed so the line matches across
# CPython/PyPy (the non-string TypeError message text differs between them).


def show(label, fn):
    try:
        fn()
        print(label, "NO-RAISE")
    except Exception as e:
        print(label, type(e).__name__)


class RaisesRepr:
    def __repr__(self):
        raise ValueError("r")


class RaisesStr:
    def __str__(self):
        raise KeyError("s")


class MyInt(int):
    def __repr__(self):
        raise RuntimeError("mi")


class NonStrRepr:
    def __repr__(self):
        return 42


show("repr-raise", lambda: repr(RaisesRepr()))
show("str-raise", lambda: str(RaisesStr()))
show("leaf-repr-raise", lambda: repr(MyInt(7)))
show("nonstr-repr", lambda: repr(NonStrRepr()))
show("list-elem", lambda: repr([RaisesRepr()]))
show("dict-key", lambda: repr({RaisesRepr(): 1}))
show("tuple-elem", lambda: str((RaisesStr(),)))
show("format-r", lambda: "{!r}".format(RaisesRepr()))
show("percent-s", lambda: "%s" % RaisesStr())
show("fstring", lambda: f"{RaisesRepr()!r}")

# Normal formatting is unaffected.
print("normal", repr([1, 2]), str({3: 4}), repr((1,)), ascii("x"))
