import types


expected = {
    "__delete__",
    "__doc__",
    "__get__",
    "__name__",
    "__objclass__",
    "__qualname__",
    "__repr__",
    "__set__",
}
assert set(types.GetSetDescriptorType.__dict__) == expected

code_descriptor = types.FunctionType.__code__
assert type(code_descriptor) is types.GetSetDescriptorType
assert code_descriptor.__name__ == "__code__"
assert code_descriptor.__qualname__ == "function.__code__"
assert code_descriptor.__objclass__ is types.FunctionType
assert code_descriptor.__doc__ is None
assert repr(code_descriptor) == "<attribute '__code__' of 'function' objects>"
assert types.GetSetDescriptorType.__repr__(code_descriptor) == repr(code_descriptor)

try:
    types.GetSetDescriptorType.__repr__(1)
except TypeError as exc:
    assert str(exc) == (
        "descriptor '__repr__' requires a 'getset_descriptor' object "
        "but received a 'int'"
    )
else:
    raise AssertionError("getset_descriptor.__repr__ accepted an int")


def first():
    return 1


def second():
    return 2


assert code_descriptor.__get__(first, types.FunctionType) is first.__code__
first.__code__ = second.__code__
assert first() == 2


class Outer:
    class Inner:
        pass


dict_descriptor = Outer.Inner.__dict__["__dict__"]
assert type(dict_descriptor) is types.GetSetDescriptorType
assert dict_descriptor.__name__ == "__dict__"
assert dict_descriptor.__qualname__ == "Outer.Inner.__dict__"
assert dict_descriptor.__objclass__ is Outer.Inner
assert dict_descriptor.__doc__ == "dictionary for instance variables"
assert repr(dict_descriptor) == "<attribute '__dict__' of 'Inner' objects>"

weakref_descriptor = Outer.Inner.__dict__["__weakref__"]
assert type(weakref_descriptor) is types.GetSetDescriptorType
assert weakref_descriptor.__name__ == "__weakref__"
assert weakref_descriptor.__qualname__ == "Outer.Inner.__weakref__"
assert weakref_descriptor.__objclass__ is Outer.Inner
assert weakref_descriptor.__doc__ == "list of weak references to the object"
assert repr(weakref_descriptor) == "<attribute '__weakref__' of 'Inner' objects>"

metadata_descriptor = types.GetSetDescriptorType.__dict__["__doc__"]
assert type(metadata_descriptor) is types.GetSetDescriptorType
assert metadata_descriptor.__qualname__ == "getset_descriptor.__doc__"
assert metadata_descriptor.__objclass__ is types.GetSetDescriptorType
assert repr(metadata_descriptor) == (
    "<attribute '__doc__' of 'getset_descriptor' objects>"
)

mapping_descriptor = type({}.keys()).mapping
assert type(mapping_descriptor) is types.GetSetDescriptorType
assert mapping_descriptor.__name__ == "mapping"
assert mapping_descriptor.__qualname__ == "dict_keys.mapping"
assert mapping_descriptor.__objclass__ is type({}.keys())
assert mapping_descriptor.__doc__ == "dictionary that this view refers to"
assert repr(mapping_descriptor) == "<attribute 'mapping' of 'dict_keys' objects>"

print("OK")
