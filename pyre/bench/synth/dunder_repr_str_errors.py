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
show("tuple-elem", lambda: str((RaisesRepr(),)))
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

# `type.__format__` (int/float/str/bool) and `object.__format__` read the
# spec storage directly too.
print("int-format-strsub", (12).__format__(StrSubRaisingStr("04d")))
print("str-format-strsub", "hi".__format__(StrSubRaisingStr(">5")))
print("object-format-strsub", object().__format__(StrSubRaisingStr("")) != "")
show("int-format-nonstr", lambda: (12).__format__(34))


# `format()` / f-strings dispatch a `__format__` override, including one on
# a builtin subclass (which `is_instance` alone would miss).
class FmtInt(int):
    def __format__(self, spec):
        return "fi:" + spec


print("subclass-format-override", format(FmtInt(5), ">3"), f"{FmtInt(5):^4}")


# `__format__` is a type-level special method: an instance-dict attribute does
# not shadow it (the instance below still formats via `object.__format__`), and
# a non-function descriptor override (e.g. `staticmethod`) on a builtin subclass
# is dispatched through the descriptor protocol rather than formatting the
# underlying value.
class PlainObj:
    pass


_inst = PlainObj()
_inst.__format__ = lambda spec: "INST"
print("inst-dict-format-ignored", format(_inst, "") == str(_inst))


class StaticFmt(int):
    __format__ = staticmethod(lambda spec: "sf:" + spec)


print("staticmethod-format-override", format(StaticFmt(7), ">3"))

# print(sep=, end=): None selects the default, a str subclass is rendered
# through str() (Py_PRINT_RAW), and a non-str raises TypeError. (A str
# subclass with a __str__ override is intentionally not exercised here —
# 3.14 calls __str__ while PyPy reads storage, so the check.py oracles
# diverge; PlainSep has no override, so all agree.)
class PlainSep(str):
    pass


print("a", "b", sep=PlainSep("-"))
print("pend-none", end=None)
show("print-end-int", lambda: print("x", end=5))
show("print-sep-int", lambda: print("x", sep=5, end="\n"))

# Normal formatting is unaffected.
print("normal", repr([1, 2]), str({3: 4}), repr((1,)), ascii("x"))
