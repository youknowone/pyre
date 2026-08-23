# CPython-suite gap: the suite's exception-repr coverage is in test_exceptions,
# which only builds classes whose name has no module component, so nothing in
# it distinguishes the registered name from the one `__name__` answers.
# parity-tests reason: a module exception is registered under a dotted name and
# pyre's `W_BaseException.descr_repr` read that name whole, so `repr` spelled
# the module twice as often as `str(type(e))` does.

# `pytest` prints `repr(excinfo.value)` in the one-line failure summary, and a
# doctest that expects `error(...)` sees `struct.error(...)` instead.

import binascii
import struct
import zlib

for module, attribute in (
    (struct, "error"),
    (zlib, "error"),
    (binascii, "Error"),
    (binascii, "Incomplete"),
):
    cls = getattr(module, attribute)
    # `__name__` never carried the module component; `repr` is the channel that
    # disagreed with it.
    assert cls.__name__ == attribute, (cls.__name__, attribute)
    assert cls.__module__ == module.__name__, (cls.__module__, module.__name__)
    assert str(cls) == "<class '%s.%s'>" % (module.__name__, attribute), str(cls)

    instance = cls(19, "boom")
    assert repr(instance) == "%s(19, 'boom')" % attribute, repr(instance)

    # One argument takes no trailing comma, and none at all takes empty parens.
    assert repr(cls("boom")) == "%s('boom')" % attribute, repr(cls("boom"))
    assert repr(cls()) == "%s()" % attribute, repr(cls())

# A class defined in Python keeps the name it was written with.
class Plain(Exception):
    pass


assert repr(Plain(1)) == "Plain(1)", repr(Plain(1))

print("OK")
