"""CPython 3.14 ContextVar constructor and default-value parity."""

from contextvars import ContextVar


var = ContextVar("name")
assert var.name == "name"
try:
    var.name = "changed"
except AttributeError:
    pass
else:
    raise AssertionError("ContextVar.name must be read-only")

defaulted = ContextVar("defaulted", default=42)
assert defaulted.get() == 42
assert defaulted.get(7) == 7


class UnhashableStr(str):
    __hash__ = None


try:
    ContextVar(UnhashableStr("bad"))
except TypeError as exc:
    assert "unhashable type" in str(exc)
else:
    raise AssertionError("an unhashable str subclass must be rejected")
