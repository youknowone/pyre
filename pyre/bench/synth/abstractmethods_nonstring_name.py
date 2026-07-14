# `_py_abc.ABCMeta.__new__` validates every inherited abstract-method name
# through `getattr(cls, name, None)`: a non-string name raises TypeError.
import abc


class Base:
    __abstractmethods__ = [1]


try:
    class Derived(Base, metaclass=abc.ABCMeta):
        pass
except TypeError as exc:
    print(type(exc).__name__, exc)
else:
    raise AssertionError("ABCMeta accepted a non-string abstract-method name")
