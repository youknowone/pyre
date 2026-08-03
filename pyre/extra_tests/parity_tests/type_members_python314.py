"""CPython 3.14 member-descriptor surface for values backed by PyPy type state."""


member_names = (
    "__basicsize__",
    "__itemsize__",
    "__flags__",
    "__weakrefoffset__",
    "__base__",
    "__dictoffset__",
)
for name in member_names:
    member = type.__dict__[name]
    assert type(member).__name__ == "member_descriptor"
    assert member.__objclass__ is type
    assert member.__name__ == name
    assert member.__doc__ is None


class A:
    pass


class B(A):
    pass


class Slotted:
    __slots__ = ("value",)


class NoStorage:
    __slots__ = ()


class IntSubclass(int):
    pass


class BytesSubclass(bytes):
    pass


class StrSubclass(str):
    pass


class ListSubclass(list):
    pass


class SlottedList(list):
    __slots__ = ("value",)


assert object.__base__ is None
assert type.__base__ is object
assert A.__base__ is object
assert B.__base__ is A
assert isinstance(A.__flags__, int)
assert A.__flags__ & (1 << 9)  # Py_TPFLAGS_HEAPTYPE

expected_layouts = {
    object: (16, 0, 0, 0),
    type: (936, 40, 368, 264),
    int: (24, 4, 0, 0),
    bool: (24, 4, 0, 0),
    float: (24, 0, 0, 0),
    complex: (32, 0, 0, 0),
    str: (64, 0, 0, 0),
    bytes: (33, 1, 0, 0),
    bytearray: (56, 0, 0, 0),
    list: (40, 0, 0, 0),
    tuple: (32, 8, 0, 0),
    dict: (48, 0, 0, 0),
    set: (200, 0, 192, 0),
    frozenset: (200, 0, 192, 0),
    memoryview: (144, 8, 136, 0),
    range: (48, 0, 0, 0),
    slice: (40, 0, 0, 0),
    super: (40, 0, 0, 0),
    enumerate: (56, 0, 0, 0),
    filter: (32, 0, 0, 0),
    map: (40, 0, 0, 0),
    reversed: (32, 0, 0, 0),
    zip: (48, 0, 0, 0),
    staticmethod: (32, 0, 0, 24),
    classmethod: (32, 0, 0, 24),
    property: (64, 0, 0, 0),
    type(None): (16, 0, 0, 0),
    type(NotImplemented): (16, 0, 0, 0),
    type(Ellipsis): (16, 0, 0, 0),
    BaseException: (72, 0, 0, 16),
    AttributeError: (88, 0, 0, 16),
    BaseExceptionGroup: (88, 0, 0, 16),
    ExceptionGroup: (88, 0, -32, 16),
    SyntaxError: (144, 0, 0, 16),
    OSError: (112, 0, 0, 16),
    A: (16, 0, -32, -1),
    B: (16, 0, -32, -1),
    Slotted: (24, 0, 0, 0),
    NoStorage: (16, 0, 0, 0),
    IntSubclass: (24, 4, 0, -1),
    BytesSubclass: (33, 1, 0, -1),
    StrSubclass: (64, 0, -32, -1),
    ListSubclass: (40, 0, -32, -1),
    SlottedList: (48, 0, 0, 0),
}
for cls, expected in expected_layouts.items():
    actual = (
        cls.__basicsize__,
        cls.__itemsize__,
        cls.__weakrefoffset__,
        cls.__dictoffset__,
    )
    assert actual == expected, (cls, actual, expected)

assert "__basicsize__" not in int.__dict__
assert "__itemsize__" not in int.__dict__

for cls in (A, B):
    for name in member_names:
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
