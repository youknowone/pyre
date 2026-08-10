# CPython-suite gap: type-cache tests omit surrogate lookup through a tag-zero MRO.
# parity-tests reason: this targets PyPy/pyre method-cache representation details.

"""A class-like non-type MRO entry keeps surrogate lookup on the tag-zero path."""

import sys


if sys.implementation.name != "cpython":
    NAME = "\udc82tag_zero"

    class ClassicLike:
        __bases__ = ()

    classic_like = ClassicLike()

    class Meta(type):
        def mro(cls):
            return [cls, classic_like, object]

    class Subject(metaclass=Meta):
        pass

    setattr(Subject, NAME, 41)
    for _ in range(3000):
        assert getattr(Subject, NAME) == 41

    setattr(Subject, NAME, 42)
    assert getattr(Subject, NAME) == 42

print("OK")
