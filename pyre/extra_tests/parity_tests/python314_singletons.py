"""Python 3.14 singleton-type differences from PyPy's 3.11 sources."""

try:
    bool(NotImplemented)
except TypeError as exc:
    assert str(exc) == "NotImplemented should not be used in a boolean context"
else:
    raise AssertionError("Python 3.14 makes bool(NotImplemented) an error")

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
