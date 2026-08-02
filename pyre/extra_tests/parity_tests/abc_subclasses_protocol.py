"""ABCMeta subclass checks observe the public ``__subclasses__`` protocol.

PyPy: ``pypy/module/_abc/app_abc.py:156-158``::

    for scls in cls.__subclasses__():
        if issubclass(subclass, scls):
            return True

Calling and iterating the override is observable: invalid call signatures,
non-iterable results, invalid members, and user exceptions must propagate.
"""

from abc import ABCMeta


def expect_type_error(func):
    try:
        func()
    except TypeError:
        return
    raise AssertionError("TypeError not raised")


for override in (None, lambda arg: [], lambda: 42, lambda: [42]):

    class InvalidSubclasses(metaclass=ABCMeta):
        __subclasses__ = override

    expect_type_error(lambda: issubclass(int, InvalidSubclasses))


class MarkerError(Exception):
    pass


def raise_marker():
    raise MarkerError("from __subclasses__")


class RaisingSubclasses(metaclass=ABCMeta):
    __subclasses__ = raise_marker


try:
    issubclass(int, RaisingSubclasses)
except MarkerError as exc:
    assert str(exc) == "from __subclasses__"
else:
    raise AssertionError("MarkerError not raised")


class Base(metaclass=ABCMeta):
    pass


class Child(Base):
    pass


class Grandchild(Child):
    pass


assert issubclass(Grandchild, Base)
print("OK")
