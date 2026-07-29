"""CPython 3.14 Context and ContextVar constructor/default-value parity."""

from contextvars import Context, ContextVar, copy_context

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

empty = Context()
assert len(empty) == 0
assert list(empty) == []
assert list(empty.keys()) == []
assert list(empty.values()) == []
assert list(empty.items()) == []
assert empty.get(var) is None
assert var not in empty
assert empty.copy() == empty
assert empty.copy() is not empty
assert empty.run(lambda value, *, add: value + add, 40, add=2) == 42
assert copy_context() == Context()

try:
    empty.run(lambda: empty.run(lambda: None))
except RuntimeError:
    pass
else:
    raise AssertionError("an entered Context must reject re-entry")


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

print("OK")
