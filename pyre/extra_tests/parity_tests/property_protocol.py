"""PyPy structural parity and Python 3.14 semantics for ``property``.

PyPy ``module/__builtin__/descriptor.py:W_Property`` supplies the native
accessor/doc fields, split ``__new__``/``__init__``, descriptor operations and
copying decorators.  Python 3.14 takes precedence for ``__name__`` fallback,
``__set_name__``, rich missing-accessor messages, subclass doc placement and
the exact public type-dictionary surface.
"""


EXPECTED_SURFACE = {
    "__delete__",
    "__doc__",
    "__get__",
    "__init__",
    "__isabstractmethod__",
    "__name__",
    "__new__",
    "__set__",
    "__set_name__",
    "deleter",
    "fdel",
    "fget",
    "fset",
    "getter",
    "setter",
}

assert set(property.__dict__) == EXPECTED_SURFACE
assert property.__doc__.startswith("Property attribute.\n")
assert property.__doc__.endswith("        del self._x")


def get_value(self):
    "getter doc"
    return self._value


def set_value(self, value):
    self._value = value


def del_value(self):
    del self._value


class Managed:
    value = property(get_value, set_value, del_value)


obj = Managed()
obj.value = 12
assert obj.value == 12
del obj.value
assert not hasattr(obj, "_value")

p = Managed.__dict__["value"]
assert p.fget is get_value
assert p.fset is set_value
assert p.fdel is del_value
assert p.__doc__ == "getter doc"
assert p.__name__ == "value"
assert p.__get__(None, Managed) is p

# Python 3.14 property_name: an explicit name wins, deletion clears only that
# slot and reveals the getter's __name__ fallback again.
plain = property(get_value)
assert plain.__name__ == "get_value"
plain.__name__ = "explicit"
assert plain.__name__ == "explicit"
del plain.__name__
assert plain.__name__ == "get_value"
del plain.__name__
assert plain.__name__ == "get_value"

unnamed = property()
assert not hasattr(unnamed, "__name__")
del unnamed.__name__
assert not hasattr(unnamed, "__name__")

plain.__doc__ = "replacement"
assert plain.__doc__ == "replacement"
del plain.__doc__
assert plain.__doc__ is None

# Python 3.14 property_copy treats None as an omitted replacement and keeps
# the old accessor.  A getter-derived doc is re-derived on each copy.
for method in ("getter", "setter", "deleter"):
    copy = getattr(property(get_value), method)(None)
    assert copy.fget is get_value
    assert copy.fset is None
    assert copy.fdel is None
    assert copy.__doc__ == "getter doc"

named = property(get_value, doc="explicit doc")
named.__set_name__(Managed, "renamed")
for copy in (named.getter(get_value), named.setter(set_value), named.deleter(del_value)):
    assert type(copy) is property
    assert copy.__name__ == "renamed"
    assert copy.__doc__ == "explicit doc"

# __new__ only allocates.  __init__ performs all field setup and can safely be
# called again, clearing the name and prior accessor/doc state first.
raw = property.__new__(property, object(), ignored=True)
assert raw.fget is None and raw.fset is None and raw.fdel is None
property.__init__(raw, fget=get_value, fset=set_value, doc="raw doc")
raw.__set_name__(Managed, "raw")
assert raw.fget is get_value and raw.fset is set_value
assert raw.__doc__ == "raw doc" and raw.__name__ == "raw"
property.__init__(raw)
assert raw.fget is None and raw.fset is None and raw.fdel is None
assert raw.__doc__ is None and not hasattr(raw, "__name__")


class PropertySubclass(property):
    pass


sub = PropertySubclass(get_value)
assert type(sub) is PropertySubclass
assert sub.__dict__ == {"__doc__": "getter doc"}
assert sub.__doc__ == "getter doc"
sub_copy = sub.setter(set_value)
assert type(sub_copy) is PropertySubclass
assert sub_copy.__dict__ == {"__doc__": "getter doc"}
sub.extra = 17
assert sub.extra == 17
assert sub.__dict__["extra"] == 17

property.__init__(sub)
assert sub.__dict__ == {"__doc__": None, "extra": 17}
assert sub.__doc__ is None

try:
    plain.fget = None
except AttributeError:
    pass
else:
    assert False, "property.fget must be read-only"


class AbstractCallable:
    __isabstractmethod__ = True

    def __call__(self, *args):
        return None


abstract = AbstractCallable()
assert property(abstract).__isabstractmethod__ is True
assert property(None, abstract).__isabstractmethod__ is True
assert property(None, None, abstract).__isabstractmethod__ is True
assert property().__isabstractmethod__ is False


class ReadOnly:
    item = property()


class RaisingDescriptor:
    def __get__(self, instance, owner):
        raise AttributeError("descriptor miss")


class PropertyWithGetattr(property):
    missing = RaisingDescriptor()

    def __getattr__(self, name):
        return f"fallback:{name}"


assert PropertyWithGetattr().missing == "fallback:missing"


try:
    ReadOnly().item
except AttributeError as exc:
    assert str(exc) == "property 'item' of 'ReadOnly' object has no getter"
else:
    assert False, "missing getter must raise"

try:
    ReadOnly().item = 1
except AttributeError as exc:
    assert str(exc) == "property 'item' of 'ReadOnly' object has no setter"
else:
    assert False, "missing setter must raise"

try:
    del ReadOnly().item
except AttributeError as exc:
    assert str(exc) == "property 'item' of 'ReadOnly' object has no deleter"
else:
    assert False, "missing deleter must raise"

for action in (
    lambda: property(1, 2, 3, 4, 5),
    lambda: property(foo=1),
    lambda: property(1, fget=2),
):
    try:
        action()
    except TypeError:
        pass
    else:
        assert False, "invalid property constructor arguments must raise"

try:
    property.__delete__(property())
except TypeError:
    pass
else:
    raise AssertionError("property.__delete__ accepted a missing target")

print("OK")
