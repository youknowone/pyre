# pyre-check: platforms=linux
# The table holds absolute C-struct sizes, so it is one ABI's: the reference
# itself reports `OSError.__basicsize__` 120 on Windows (the extra `winerror`
# member) and `BaseExceptionGroup.__basicsize__` 88 on macOS arm64.
"""CPython 3.14 member-descriptor surface for values backed by PyPy type state."""

import sys
import types


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
    BaseExceptionGroup: (96, 0, 0, 16),
    ExceptionGroup: (96, 0, -32, 16),
    SyntaxError: (144, 0, 0, 16),
    # An `OSError` holds a `winerror` where the platform has Windows error
    # codes, one pointer more than the layout everywhere else.
    OSError: (120, 0, 0, 16) if sys.platform == "win32" else (112, 0, 0, 16),
    ImportWarning: (72, 0, 0, 16),
    SyntaxWarning: (72, 0, 0, 16),
    types.FunctionType: (152, 0, 96, 88),
    types.ModuleType: (56, 0, 40, 16),
    types.MethodType: (48, 0, 32, 0),
    types.BuiltinFunctionType: (56, 0, 40, 0),
    types.CodeType: (208, 2, 144, 0),
    types.GeneratorType: (152, 8, 16, 0),
    types.CoroutineType: (152, 8, 16, 0),
    types.AsyncGeneratorType: (152, 8, 16, 0),
    types.FrameType: (152, 8, 0, 0),
    types.TracebackType: (40, 0, 0, 0),
    types.CellType: (24, 0, 0, 0),
    types.MappingProxyType: (24, 0, 0, 0),
    types.SimpleNamespace: (24, 0, 0, 16),
    types.GenericAlias: (64, 0, 40, 0),
    types.UnionType: (56, 0, 48, 0),
    types.WrapperDescriptorType: (56, 0, 0, 0),
    types.MethodDescriptorType: (56, 0, 0, 0),
    types.ClassMethodDescriptorType: (56, 0, 0, 0),
    types.GetSetDescriptorType: (48, 0, 0, 0),
    types.MemberDescriptorType: (48, 0, 0, 0),
    types.MethodWrapperType: (32, 0, 0, 0),
    type(iter(b"")): (32, 0, 0, 0),
    type(iter(bytearray())): (32, 0, 0, 0),
    type(iter([])): (32, 0, 0, 0),
    type(reversed([])): (32, 0, 0, 0),
    type(iter(())): (32, 0, 0, 0),
    type(iter("")): (32, 0, 0, 0),
    type(iter(iter, None)): (32, 0, 0, 0),
    type(iter(range(1))): (40, 0, 0, 0),
    type(iter(range(1 << 70))): (40, 0, 0, 0),
    type(iter(set())): (48, 0, 0, 0),
    type(iter({})): (56, 0, 0, 0),
    type(iter({}.values())): (56, 0, 0, 0),
    type(iter({}.items())): (56, 0, 0, 0),
    type(reversed({}.keys())): (56, 0, 0, 0),
    type(reversed({}.values())): (56, 0, 0, 0),
    type(reversed({}.items())): (56, 0, 0, 0),
    type({}.keys()): (24, 0, 0, 0),
    type({}.values()): (24, 0, 0, 0),
    type({}.items()): (24, 0, 0, 0),
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

expected_flags = {
    object: 0x1502,
    type: 0x80805D02,
    int: 0x1401502,
    bool: 0x1401102,
    float: 0x401502,
    complex: 0x1502,
    str: 0x10401502,
    bytes: 0x8401502,
    bytearray: 0x401502,
    list: 0x2405522,
    tuple: 0x4405522,
    dict: 0x20405542,
    set: 0x405502,
    frozenset: 0x405502,
    memoryview: 0x5122,
    range: 0x1122,
    slice: 0x5102,
    super: 0x5502,
    enumerate: 0x5502,
    filter: 0x5502,
    map: 0x5502,
    reversed: 0x5502,
    zip: 0x5502,
    staticmethod: 0x5502,
    classmethod: 0x5502,
    property: 0x5502,
    type(None): 0x1102,
    type(NotImplemented): 0x1102,
    type(Ellipsis): 0x1102,
    BaseException: 0x40005502,
    AttributeError: 0x40005502,
    BaseExceptionGroup: 0x40005502,
    ExceptionGroup: 0x40005608,
    SyntaxError: 0x40005502,
    OSError: 0x40005502,
    A: 0x561C,
    B: 0x561C,
    Slotted: 0x5600,
    NoStorage: 0x5600,
    IntSubclass: 0x1405610,
    BytesSubclass: 0x8405610,
    StrSubclass: 0x1040561C,
    ListSubclass: 0x240563C,
    SlottedList: 0x2405620,
}
for cls, expected in expected_flags.items():
    assert cls.__flags__ == expected, (cls, hex(cls.__flags__), hex(expected))

runtime_type_flags = {
    types.FunctionType: 0x25902,
    types.BuiltinFunctionType: 0x5982,
    types.MethodType: 0x5902,
    types.MethodDescriptorType: 0x25982,
    types.WrapperDescriptorType: 0x25182,
    types.MethodWrapperType: 0x5182,
    types.ClassMethodDescriptorType: 0x5182,
    types.GeneratorType: 0x5182,
    types.CoroutineType: 0x5182,
    types.AsyncGeneratorType: 0x5182,
    types.CodeType: 0x1102,
    types.FrameType: 0x5182,
    types.TracebackType: 0x5102,
    types.CellType: 0x5102,
    types.ModuleType: 0x5502,
    types.MappingProxyType: 0x5142,
}
for cls, expected in runtime_type_flags.items():
    assert cls.__flags__ == expected, (cls, hex(cls.__flags__), hex(expected))

# ExceptionGroup is the heap-type exception-group leaf in CPython 3.14;
# BaseExceptionGroup remains an immutable static type.
ExceptionGroup._pyre_type_flags_probe = 1
assert ExceptionGroup._pyre_type_flags_probe == 1
del ExceptionGroup._pyre_type_flags_probe

for non_base in (slice, types.MappingProxyType):
    try:
        type("BadSubclass", (non_base,), {})
    except TypeError:
        pass
    else:
        raise AssertionError(f"{non_base.__name__} must not be subclassable")

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
