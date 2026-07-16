"""object TypeDef parity with PyPy, using CPython 3.14 for newer surface."""


expected = {
    "__class__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__getstate__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__le__",
    "__lt__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
}
assert expected <= set(object.__dict__)

left = object()
right = object()
for name in ("__lt__", "__le__", "__gt__", "__ge__"):
    assert getattr(object, name)(left, right) is NotImplemented

assert object.__sizeof__(object()) == 16


class Plain:
    pass


class Slotted:
    __slots__ = ("first", "second")


assert object.__sizeof__(Plain()) == 16
assert object.__sizeof__(Slotted()) == 32

p = Plain()
p.answer = 42
names = object.__dir__(p)
assert isinstance(names, list)
assert {"answer", "__class__", "__sizeof__"} <= set(names)
assert dir(p) == sorted(dir(p))


class A:
    pass


class B:
    pass


a = A()
assert a.__class__ is A
a.__class__ = B
assert a.__class__ is B
assert type(a) is B

descriptor = object.__dict__["__class__"]
assert type(descriptor).__name__ == "getset_descriptor"
assert descriptor.__get__(a, B) is B

try:
    a.__class__ = 1
except TypeError:
    pass
else:
    raise AssertionError("non-type __class__ assignment must fail")

assert object.__doc__.startswith("The base class of the class hierarchy.")
print("object surface: ok")
