# CPython-suite gap: type-cache tests omit hot surrogate-name store/delete cycles.
# parity-tests reason: this targets PyPy/pyre method-cache invalidation under JIT.

"""A hot surrogate-name type lookup observes store and delete invalidation."""

N = 3000
NAME = "\udc80cache_invalidation"


class Base:
    pass


class Sub(Base):
    pass


setattr(Base, NAME, 1)
for _ in range(N):
    assert getattr(Sub, NAME) == 1

setattr(Base, NAME, 2)
assert getattr(Sub, NAME) == 2

delattr(Base, NAME)
try:
    getattr(Sub, NAME)
except AttributeError:
    pass
else:
    raise AssertionError("deleted surrogate attribute remained cached")

print("OK")
