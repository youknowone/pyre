import types


expected = {
    "__annotate__",
    "__annotations__",
    "__dict__",
    "__dir__",
    "__doc__",
    "__getattribute__",
    "__init__",
    "__new__",
    "__repr__",
}
assert set(types.ModuleType.__dict__) == expected
assert types.ModuleType.__doc__ == (
    "Create a module object.\n\n"
    "The name must be a string; the optional doc argument can have any type."
)

module = types.ModuleType("sample", "documentation")
assert module.__dict__ == {
    "__name__": "sample",
    "__doc__": "documentation",
    "__package__": None,
    "__loader__": None,
    "__spec__": None,
}
assert repr(module) == "<module 'sample'>"
module.__file__ = "/tmp/sample.py"
assert repr(module) == "<module 'sample' from '/tmp/sample.py'>"
del module.__name__
assert repr(module) == "<module '?' from '/tmp/sample.py'>"

module = types.ModuleType(name="sample", doc="documentation")
assert module.__name__ == "sample"
assert module.__doc__ == "documentation"
assert types.ModuleType.__getattribute__(module, "__name__") == "sample"

for args, kwargs, message in [
    ((), {}, "module() missing required argument 'name' (pos 1)"),
    (("x", "d", 1), {}, "module() takes at most 2 arguments (3 given)"),
    ((1,), {}, "module() argument 'name' must be str, not int"),
    (("x",), {"name": "y"}, "argument for module() given by name ('name') and position (1)"),
    (("x",), {"bad": 1}, "module() got an unexpected keyword argument 'bad'"),
]:
    try:
        types.ModuleType(*args, **kwargs)
    except TypeError as exc:
        assert str(exc) == message
    else:
        raise AssertionError((args, kwargs))

dict_descriptor = types.ModuleType.__dict__["__dict__"]
assert type(dict_descriptor) is types.MemberDescriptorType
assert dict_descriptor.__name__ == "__dict__"
assert dict_descriptor.__qualname__ == "module.__dict__"
assert dict_descriptor.__objclass__ is types.ModuleType
assert repr(dict_descriptor) == "<member '__dict__' of 'module' objects>"
try:
    module.__dict__ = {}
except AttributeError as exc:
    assert str(exc) == "readonly attribute"
else:
    raise AssertionError("module.__dict__ accepted assignment")

for name in ("__annotations__", "__annotate__"):
    descriptor = types.ModuleType.__dict__[name]
    for operation in ("get", "set", "delete"):
        try:
            if operation == "get":
                descriptor.__get__(1, int)
            elif operation == "set":
                descriptor.__set__(1, None)
            else:
                descriptor.__delete__(1)
        except TypeError as exc:
            assert str(exc) == (
                f"descriptor '{name}' for 'module' objects "
                "doesn't apply to a 'int' object"
            )
        else:
            raise AssertionError((name, operation))

module = types.ModuleType("sample")
module.z = 1
module.a = 2
assert module.__dir__()[-2:] == ["z", "a"]
module.__dir__ = lambda: ("z", "a")
assert types.ModuleType.__dir__(module) == ("z", "a")
assert dir(module) == ["a", "z"]

module = types.ModuleType("sample")
assert "__annotations__" not in module.__dict__
annotations = module.__annotations__
assert annotations == {}
assert module.__annotations__ is annotations
assert module.__dict__["__annotations__"] is annotations

module.__annotations__ = 1
assert module.__annotations__ == 1
del module.__annotations__
assert "__annotations__" not in module.__dict__
try:
    del module.__annotations__
except AttributeError as exc:
    assert str(exc) == "__annotations__"
else:
    raise AssertionError("missing module.__annotations__ deletion succeeded")

module = types.ModuleType("sample")
assert module.__annotate__ is None
assert module.__dict__["__annotate__"] is None
try:
    module.__annotate__ = 1
except TypeError as exc:
    assert str(exc) == "__annotate__ must be callable or None"
else:
    raise AssertionError("module.__annotate__ accepted a non-callable")

calls = []


def annotate(format):
    calls.append(format)
    return {"answer": format}


module.__annotations__ = {"old": True}
module.__annotate__ = annotate
assert "__annotations__" not in module.__dict__
assert module.__annotations__ == {"answer": 1}
assert calls == [1]
assert module.__annotations__ == {"answer": 1}
assert calls == [1]

module.__annotations__ = {"eager": True}
assert "__annotate__" not in module.__dict__
assert module.__annotations__ == {"eager": True}
try:
    del module.__annotate__
except TypeError as exc:
    assert str(exc) == "cannot delete __annotate__ attribute"
else:
    raise AssertionError("module.__annotate__ deletion succeeded")

module = types.ModuleType("sample")
module.__annotate__ = lambda format: 1
try:
    module.__annotations__
except TypeError as exc:
    assert str(exc) == "__annotate__ returned non-dict of type 'int'"
else:
    raise AssertionError("non-dict annotations result was accepted")

print("OK")
