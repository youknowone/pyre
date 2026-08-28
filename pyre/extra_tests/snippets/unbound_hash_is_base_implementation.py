# pyre-check: gate=1
# CPython-suite gap: test_tuple and test_set never call `tuple.__hash__` or
# `frozenset.__hash__` on an instance of a subclass that overrides `__hash__`.
# parity-tests reason: both descriptors dispatched through `hash()`, which looks
# the dunder up on the instance's type, so the override ran in place of the base
# implementation the unbound call names.

# The memoising idiom below is what makes this reachable in the wild: a
# subclass keeps the base hash under a default argument and calls it, so an
# implementation that re-dispatches recurses until the stack runs out instead of
# returning the structural hash.


class LoudTuple(tuple):
    def __hash__(self):
        raise AssertionError("tuple.__hash__ must not consult the subclass")


class LoudFrozenset(frozenset):
    def __hash__(self):
        raise AssertionError("frozenset.__hash__ must not consult the subclass")


assert tuple.__hash__(LoudTuple((1, 2))) == hash((1, 2))
assert frozenset.__hash__(LoudFrozenset([1, 2])) == hash(frozenset([1, 2]))


class MemoisingKey(tuple):
    def __hash__(self, base=tuple.__hash__):
        try:
            return self._hash
        except AttributeError:
            pass
        object.__setattr__(self, "_hash", base(self))
        return self._hash


key = MemoisingKey((1, 2, 3))
assert hash(key) == hash((1, 2, 3))
assert hash(key) == hash(key)

# The bound form still honours the override, which is the difference the two
# spellings exist to express.
assert type(key).__hash__(key) == hash((1, 2, 3))
try:
    hash(LoudTuple((1, 2)))
except AssertionError:
    pass
else:
    raise SystemExit("hash() must reach the subclass override")

print("OK")
