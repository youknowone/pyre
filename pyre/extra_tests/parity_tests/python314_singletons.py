"""Python 3.14 singleton-type differences from PyPy's 3.11 sources."""

try:
    bool(NotImplemented)
except TypeError as exc:
    assert str(exc) == "NotImplemented should not be used in a boolean context"
else:
    raise AssertionError("Python 3.14 makes bool(NotImplemented) an error")

for singleton, constructor_error in (
    (Ellipsis, "EllipsisType takes no arguments"),
    (NotImplemented, "NotImplementedType takes no arguments"),
    (None, "NoneType takes no arguments"),
):
    singleton_type = type(singleton)
    assert singleton_type() is singleton
    try:
        singleton_type(1)
    except TypeError as exc:
        assert str(exc) == constructor_error
    else:
        raise AssertionError(f"{singleton_type.__name__} accepted an argument")

    for name in singleton_type.__dict__:
        if name in {"__doc__", "__new__"}:
            continue
        method = getattr(singleton_type, name)
        call_args = (42, 0) if name in {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"} else (42,)
        try:
            method(*call_args)
        except TypeError:
            pass
        else:
            raise AssertionError(f"{singleton_type.__name__}.{name} accepted a foreign receiver")

none_type = type(None)
assert none_type() is None
try:
    none_type(None)
except TypeError as exc:
    assert str(exc) == "NoneType takes no arguments"
else:
    raise AssertionError("NoneType accepted a constructor argument")
assert None.__eq__(None) is True
assert None.__eq__(0) is NotImplemented
assert None.__ne__(None) is False
assert None.__lt__(0) is NotImplemented
assert isinstance(None.__hash__(), int)

try:
    class BadNone(none_type):
        pass
except TypeError:
    pass
else:
    raise AssertionError("NoneType must not be an acceptable base class")

print("OK")
