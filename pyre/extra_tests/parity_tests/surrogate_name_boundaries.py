"""A lone surrogate in an ordinary name is reported, never fatal.

`'\\udc80'` is an ordinary `str` literal — no filesystem, no `os`, no encoding
boundary is involved — so every place that reads a name back has to cope with
one.  pyre stores `str` as WTF-8 and the `&str` accessor cannot represent a lone
surrogate, so each of these was a process abort rather than an exception.

Covers the boundaries where the answer is exactly CPython's; the remaining
members of this family (a module or function `__name__`, an AST identifier)
still abort and are tracked separately.
"""

SURROGATE = "\udc80"


# `__slots__` entries are identifiers, and a lone surrogate is not one, so it
# takes the same rejection as a name starting with a digit.
try:

    class WithBadSlot:
        __slots__ = (SURROGATE,)

except TypeError as exc:
    assert str(exc) == "__slots__ must be identifiers", ascii(str(exc))
else:
    raise AssertionError("__slots__ accepted a non-identifier")

# A digit-leading name is the same rejection, so the surrogate arm is not a
# special case bolted on beside it.
try:

    class WithDigitSlot:
        __slots__ = ("1a",)

except TypeError as exc:
    assert str(exc) == "__slots__ must be identifiers", ascii(str(exc))
else:
    raise AssertionError("__slots__ accepted a digit-leading name")


# A property error names the owning type's `__qualname__`, so the message
# formatter reads a name the program chose.
class WithProperty:
    prop = property()


WithProperty.__qualname__ = SURROGATE

try:
    WithProperty().prop
except AttributeError as exc:
    expected = "property 'prop' of %r object has no getter" % SURROGATE
    assert str(exc) == expected, (ascii(str(exc)), ascii(expected))
else:
    raise AssertionError("a property with no getter returned a value")

print("OK")
