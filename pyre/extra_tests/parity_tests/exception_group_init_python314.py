"""BaseExceptionGroup owns and applies its Python 3.14 initializer."""


assert "__init__" in BaseExceptionGroup.__dict__
assert "__init__" not in ExceptionGroup.__dict__

leaf = KeyboardInterrupt("leaf")
group = BaseExceptionGroup("message", [leaf])

assert group.args == ("message", [leaf])
assert group.message == "message"
assert group.exceptions == (leaf,)

assert BaseExceptionGroup.__init__(group, "replacement", 42) is None
assert group.args == ("replacement", 42)
assert group.message == "message"
assert group.exceptions == (leaf,)

try:
    BaseExceptionGroup.__init__(group, message="replacement")
except TypeError as exc:
    assert str(exc) == "BaseExceptionGroup() takes no keyword arguments"
else:
    raise AssertionError("BaseExceptionGroup.__init__ accepted a keyword")
