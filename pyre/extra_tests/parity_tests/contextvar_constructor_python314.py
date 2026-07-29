"""CPython 3.14 ContextVar constructor and default-value parity."""

from contextvars import ContextVar

try:
    from __pypy__ import get_contextvar_context, set_contextvar_context
except ImportError:
    get_contextvar_context = set_contextvar_context = None


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

if get_contextvar_context is not None:
    previous = get_contextvar_context()
    marker = object()
    set_contextvar_context(marker)
    assert get_contextvar_context() is marker
    set_contextvar_context(previous)
