import builtins
import types


MEMBERS = {
    "__closure__",
    "__doc__",
    "__globals__",
    "__module__",
    "__builtins__",
}
GETSETS = {
    "__code__",
    "__defaults__",
    "__kwdefaults__",
    "__annotations__",
    "__annotate__",
    "__dict__",
    "__name__",
    "__qualname__",
    "__type_params__",
}

function_dict = types.FunctionType.__dict__
for name in MEMBERS:
    assert type(function_dict[name]) is types.MemberDescriptorType
for name in GETSETS:
    assert type(function_dict[name]) is types.GetSetDescriptorType

assert set(types.MemberDescriptorType.__dict__) == {
    "__delete__",
    "__doc__",
    "__get__",
    "__name__",
    "__objclass__",
    "__qualname__",
    "__reduce__",
    "__repr__",
    "__set__",
}

for name in MEMBERS:
    descriptor = function_dict[name]
    assert descriptor.__name__ == name
    assert descriptor.__qualname__ == f"function.{name}"
    assert descriptor.__objclass__ is types.FunctionType
    assert descriptor.__doc__ is None
    assert repr(descriptor) == f"<member '{name}' of 'function' objects>"
    reduced = descriptor.__reduce__()
    assert reduced == (getattr, (types.FunctionType, name))
    assert reduced[0] is getattr
    try:
        descriptor.__get__(object(), object)
    except TypeError:
        pass
    else:
        raise AssertionError(f"function member {name} accepted a foreign receiver")

for method_name in ("__repr__", "__reduce__"):
    try:
        getattr(types.MemberDescriptorType, method_name)(1)
    except TypeError:
        pass
    else:
        raise AssertionError(f"member descriptor {method_name} accepted an int")


def make_closure(value):
    def inner():
        "inner doc"
        return value

    return inner


function = make_closure(42)
assert function_dict["__closure__"].__get__(function, types.FunctionType) is function.__closure__
assert function_dict["__doc__"].__get__(function, types.FunctionType) == "inner doc"
assert function_dict["__globals__"].__get__(function, types.FunctionType) is globals()
assert function_dict["__module__"].__get__(function, types.FunctionType) == __name__
assert function_dict["__builtins__"].__get__(function, types.FunctionType) is builtins.__dict__
assert function_dict["__globals__"].__get__(None, types.FunctionType) is function_dict["__globals__"]

custom_globals = {"__builtins__": {"marker": 1}}
custom = types.FunctionType((lambda: None).__code__, custom_globals)
selected_builtins = custom.__builtins__
assert selected_builtins is custom_globals["__builtins__"]
custom_globals["__builtins__"] = {"marker": 2}
assert custom.__builtins__ is selected_builtins
del custom_globals["__builtins__"]
assert custom.__builtins__ is selected_builtins

function_dict["__doc__"].__set__(function, "changed")
assert function.__doc__ == "changed"
function_dict["__doc__"].__delete__(function)
assert function.__doc__ is None

function_dict["__module__"].__set__(function, "changed_module")
assert function.__module__ == "changed_module"
function_dict["__module__"].__delete__(function)
assert function.__module__ is None

for name in ("__closure__", "__globals__", "__builtins__"):
    descriptor = function_dict[name]
    try:
        descriptor.__set__(function, None)
    except AttributeError as exc:
        assert str(exc) == "readonly attribute"
    else:
        raise AssertionError(f"{name} accepted assignment")
    try:
        descriptor.__delete__(function)
    except AttributeError as exc:
        assert str(exc) == "readonly attribute"
    else:
        raise AssertionError(f"{name} accepted deletion")


class Outer:
    class Inner:
        __slots__ = ("slot",)


slot_descriptor = Outer.Inner.slot
assert type(slot_descriptor) is types.MemberDescriptorType
assert slot_descriptor.__name__ == "slot"
assert slot_descriptor.__qualname__ == "Outer.Inner.slot"
assert slot_descriptor.__objclass__ is Outer.Inner
assert slot_descriptor.__doc__ is None
assert repr(slot_descriptor) == "<member 'slot' of 'Inner' objects>"
assert slot_descriptor.__reduce__() == (getattr, (Outer.Inner, "slot"))

instance = Outer.Inner()
slot_descriptor.__set__(instance, 7)
assert slot_descriptor.__get__(instance, Outer.Inner) == 7
slot_descriptor.__delete__(instance)
try:
    slot_descriptor.__get__(instance, Outer.Inner)
except AttributeError:
    pass
else:
    raise AssertionError("deleted slot remained readable")

print("OK")
