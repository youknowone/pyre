# CPython-suite gap: type-cache tests omit hot surrogate names after __bases__ edits.
# parity-tests reason: this targets PyPy/pyre method-cache and JIT invalidation.

"""A hot surrogate-name lookup follows a reassigned ``__bases__`` MRO."""

N = 3000
NAME = "\udc81cache_bases"


class Base:
    pass


class Other:
    pass


class Sub(Base):
    pass


setattr(Base, NAME, "base")
setattr(Other, NAME, "other")
for _ in range(N):
    assert getattr(Sub, NAME) == "base"

Sub.__bases__ = (Other,)
assert getattr(Sub, NAME) == "other"

print("OK")
