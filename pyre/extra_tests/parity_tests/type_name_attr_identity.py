# pyre-check: pypy-diverges: pypy3 copies a str subclass assigned to type.__name__
# CPython-suite gap: mutable-type-name tests omit str-subclass identity under hot reads.
# parity-tests reason: this guards pyre's JIT fold of the metatype name slot.

"""A hot type-name read preserves the object stored by CPython's setter."""

N = 30000


class Name(str):
    def __new__(cls, value, note):
        result = str.__new__(cls, value)
        result.note = note
        return result


class Heap:
    pass


tagged = Name("Renamed", "first")
Heap.__name__ = tagged
for _ in range(N):
    seen = Heap.__name__
assert seen is tagged and seen.note == "first"

# The same compiled read must observe a later slot replacement.
again = Name("Again", "second")
Heap.__name__ = again
for _ in range(N):
    seen = Heap.__name__
assert seen is again and seen.note == "second"

print("OK")
