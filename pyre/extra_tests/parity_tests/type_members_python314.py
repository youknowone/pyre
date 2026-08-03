"""CPython 3.14 member-descriptor surface for values backed by PyPy type state."""


for name in ("__flags__", "__base__"):
    member = type.__dict__[name]
    assert type(member).__name__ == "member_descriptor"
    assert member.__objclass__ is type
    assert member.__name__ == name
    assert member.__doc__ is None


class A:
    pass


class B(A):
    pass


assert object.__base__ is None
assert type.__base__ is object
assert A.__base__ is object
assert B.__base__ is A
assert isinstance(A.__flags__, int)
assert A.__flags__ & (1 << 9)  # Py_TPFLAGS_HEAPTYPE

for cls in (A, B):
    for name in ("__flags__", "__base__"):
        original = getattr(cls, name)
        try:
            setattr(cls, name, None)
        except AttributeError as error:
            assert str(error) == "readonly attribute"
        else:
            raise AssertionError(f"{cls.__name__}.{name} must be read-only")
        try:
            delattr(cls, name)
        except AttributeError as error:
            assert str(error) == "readonly attribute"
        else:
            raise AssertionError(f"{cls.__name__}.{name} must not be deletable")
        current = getattr(cls, name)
        if name == "__base__":
            assert current is original
        else:
            assert current == original

print("OK")
