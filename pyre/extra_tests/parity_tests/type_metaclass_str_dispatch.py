"""Parity test: `str(cls)` dispatches the metaclass `__str__`.

A class object is an instance of its metaclass, so `str(cls)` resolves
`__str__` on that metaclass exactly as `repr(cls)` resolves `__repr__`.
`descroperation.py:900-925` generates the two from one source template:

    for targetname, specialname in [
        ('str', '__str__'),
        ('repr', '__repr__')]:

        source = ...
            def %(targetname)s(space, w_obj):
                w_impl = space.lookup(w_obj, %(specialname)r)
                if w_impl is None:
                    raise oefmt(space.w_TypeError, ...)
                w_result = space.get_and_call_function(w_impl, w_obj)
                if space.isinstance_w(w_result, space.w_unicode):
                    return w_result
                raise oefmt(space.w_TypeError,
                            "%(specialname)s returned non-string (type "
                            "'%%T')", w_result)

`space.lookup(w_obj, name)` walks `type(w_obj)`, which for a class object
is the metaclass, so both members of the pair honour a metaclass override.
`type` itself defines no `__str__`, so a metaclass without an override
resolves to `object`'s and the native `<class '...'>` text stands.

Pinned contract:
  1. No override anywhere -> the native `<class '...'>` text.
  2. A metaclass `__str__` wins for str(), format() and f-strings.
  3. A metaclass `__str__` wins over that metaclass's own `__repr__`.
  4. An inherited metaclass `__str__` applies, and so does a subclass of
     the class carrying it.
  5. A raising override propagates; a non-string return is a TypeError.
  6. The result object is returned as-is, so a str subclass keeps its type.
  7. Instances of the class are unaffected (the metaclass is not their type).
  8. Adding and deleting the override re-dispatches, including after the
     lookup has been warmed by a hot loop.
"""


# (1) No override: the native representation.
class Plain:
    pass


assert str(Plain) == repr(Plain) == f"<class '{__name__}.Plain'>", str(Plain)
for builtin in (int, str, type, object, ValueError):
    assert str(builtin) == repr(builtin), builtin


# (2) A metaclass __str__ wins for every stringification form.
class MetaStr(type):
    def __str__(cls):
        return "meta-str"


class WithMetaStr(metaclass=MetaStr):
    pass


assert str(WithMetaStr) == "meta-str", str(WithMetaStr)
assert "{}".format(WithMetaStr) == "meta-str"
assert f"{WithMetaStr}" == "meta-str"
assert repr(WithMetaStr) == f"<class '{__name__}.WithMetaStr'>", repr(WithMetaStr)


# (3) __str__ wins over the same metaclass's __repr__.
class MetaBoth(type):
    def __str__(cls):
        return "both-str"

    def __repr__(cls):
        return "both-repr"


class WithMetaBoth(metaclass=MetaBoth):
    pass


assert str(WithMetaBoth) == "both-str", str(WithMetaBoth)
assert repr(WithMetaBoth) == "both-repr", repr(WithMetaBoth)

# A metaclass with only __repr__ leaves str() falling back to it.
class MetaReprOnly(type):
    def __repr__(cls):
        return "repr-only"


class WithMetaReprOnly(metaclass=MetaReprOnly):
    pass


assert str(WithMetaReprOnly) == "repr-only", str(WithMetaReprOnly)


# (4) Inheritance, on the metaclass side and on the class side.
class MetaBase(type):
    def __str__(cls):
        return "inherited-str"


class MetaDerived(MetaBase):
    pass


class WithInheritedMeta(metaclass=MetaDerived):
    pass


assert str(WithInheritedMeta) == "inherited-str", str(WithInheritedMeta)


class SubOfWithMetaStr(WithMetaStr):
    pass


assert str(SubOfWithMetaStr) == "meta-str", str(SubOfWithMetaStr)


# (5) A raising override propagates; a non-string return is a TypeError.
class MetaRaise(type):
    def __str__(cls):
        raise RuntimeError("boom")


class WithMetaRaise(metaclass=MetaRaise):
    pass


try:
    str(WithMetaRaise)
except RuntimeError:
    pass
else:
    raise AssertionError("a raising metaclass __str__ must propagate")


class MetaNonStr(type):
    def __str__(cls):
        return 7


class WithMetaNonStr(metaclass=MetaNonStr):
    pass


try:
    str(WithMetaNonStr)
except TypeError:
    pass
else:
    raise AssertionError("a non-string metaclass __str__ must raise TypeError")


# (6) The result object is returned as-is.
class StrSubclass(str):
    pass


class MetaSubResult(type):
    def __str__(cls):
        return StrSubclass("sub-result")


class WithMetaSubResult(metaclass=MetaSubResult):
    pass


result = str(WithMetaSubResult)
assert result == "sub-result", result
assert type(result) is StrSubclass, type(result)


# (7) Instances are unaffected: their type is the class, not the metaclass.
assert str(WithMetaStr()).startswith(f"<{__name__}.WithMetaStr object"), str(WithMetaStr())


# (8) Adding and deleting the override re-dispatches, warmed cache included.
class MetaMut(type):
    pass


class WithMetaMut(metaclass=MetaMut):
    pass


native = f"<class '{__name__}.WithMetaMut'>"
assert str(WithMetaMut) == native, str(WithMetaMut)

total = 0
for _ in range(3000):
    total += len(str(WithMetaMut))
assert total == 3000 * len(native)

MetaMut.__str__ = lambda cls: "added-later"
assert str(WithMetaMut) == "added-later", str(WithMetaMut)

del MetaMut.__str__
assert str(WithMetaMut) == native, str(WithMetaMut)

print("OK")
