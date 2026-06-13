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


class NonStrTupleRepr(tuple):
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
show("tuple-sub-nonstr-repr", lambda: repr(NonStrTupleRepr((1, 2))))

# f-string `!a` escapes non-ASCII like ascii(), not like repr().
s = "café"
print("fstring-ascii", f"{s!a}", ascii(s))


# `complex(str)` and `format(value, spec)` read the string's stored value
# directly; a `str` subclass `__str__` is not consulted (so a raising one
# does not leak out of these paths).
class StrSubRaisingStr(str):
    def __str__(self):
        raise ValueError("boom")


print("complex-strsub", complex(StrSubRaisingStr("1")) == 1.0)
print("format-strsub", format(12, StrSubRaisingStr("04d")), format(255, StrSubRaisingStr("x")))
show("format-nonstr-spec", lambda: format(12, 34))

# Normal formatting is unaffected.
print("normal", repr([1, 2]), str({3: 4}), repr((1,)), ascii("x"))
