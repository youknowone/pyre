# pyre-check: gate=1
# `_codecs._unregister_error` drops a handler `register_error` installed.
# The eight handlers the codec loops reach by name cannot be dropped, because
# those lookups would then have no answer.

import codecs
from _codecs import _unregister_error

BUILTIN = (
    "strict", "ignore", "replace", "xmlcharrefreplace", "backslashreplace",
    "namereplace", "surrogateescape", "surrogatepass",
)

for name in BUILTIN:
    try:
        _unregister_error(name)
    except ValueError as error:
        assert str(error) == f"cannot un-register built-in error handler '{name}'", str(error)
    else:
        raise AssertionError(f"{name} was un-registered")
    # It still answers after the refusal.
    assert codecs.lookup_error(name) is not None

def handler(exc):
    return ("", exc.end)

codecs.register_error("pyre.registry.custom", handler)
assert codecs.lookup_error("pyre.registry.custom") is handler

# Removing it reports that something was there; removing it again does not.
assert _unregister_error("pyre.registry.custom") is True
assert _unregister_error("pyre.registry.custom") is False
assert _unregister_error("pyre.registry.never-registered") is False

try:
    codecs.lookup_error("pyre.registry.custom")
except LookupError as error:
    assert "pyre.registry.custom" in str(error), str(error)
else:
    raise AssertionError("the handler answered after being un-registered")

try:
    _unregister_error(1)
except TypeError as error:
    assert str(error) == "_unregister_error() argument must be str, not int", str(error)
else:
    raise AssertionError("a non-str name was accepted")
