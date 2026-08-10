# CPython-suite gap: mutable-type-name tests omit str-subclass identity under hot reads.
# parity-tests reason: this guards pyre's JIT fold of the metatype name slot.

"""Reading `Cls.__name__` hands back the object the rename was given.

`type.__name__` is a getset on the metatype, and its setter stores the value
it was handed rather than a copy of the text: a `str` subclass assigned to a
heap type's `__name__` comes back as that same object, so a caller that put
an annotated string there can read its annotation off again.  PyPy answers
with an exact `str` here; this is the CPython behaviour.

The reads run hot enough to compile, because the JIT folds this attribute to
the name slot and a fold that baked the name instead of reading it would keep
answering with the name the trace was built against.
"""

N = 30000


class Name(str):
    def __new__(cls, value, note):
        self = str.__new__(cls, value)
        self.note = note
        return self


class Heap:
    pass


original = Heap.__name__
assert original == "Heap", original
assert type(original) is str, type(original)
# The object is stable across reads, not rebuilt per access.
assert Heap.__name__ is Heap.__name__

tagged = Name("Renamed", "from-parity-test")
Heap.__name__ = tagged
assert Heap.__name__ is tagged, Heap.__name__
assert type(Heap.__name__) is Name, type(Heap.__name__)
assert Heap.__name__.note == "from-parity-test"

# Hot enough to compile, and the identity has to survive that.
seen = None
for _ in range(N):
    seen = Heap.__name__
assert seen is tagged, seen

# A rename after the loop is compiled is seen by it.
again = Name("Third", "second-rename")
Heap.__name__ = again
for _ in range(N):
    seen = Heap.__name__
assert seen is again, seen
assert seen.note == "second-rename"

# `__qualname__` is a separate slot and the renames did not touch it.
assert Heap.__qualname__ == "Heap", Heap.__qualname__

# The class dict never gains a `__name__` entry from any of this -- the getset
# on the metatype owns the name, so the class's own mapping stays untouched.
assert "__name__" not in Heap.__dict__

# An instance reads its class's dict, not the metatype getset, so a class-level
# `__name__` entry is what an instance sees while the class keeps its own name.
class Shadowed:
    __name__ = "entry"


for _ in range(N):
    pair = (Shadowed.__name__, Shadowed().__name__)
assert pair == ("Shadowed", "entry"), pair

# Immutable types refuse the rename and keep answering with their own name.
for immutable in (int, str, type):
    before = immutable.__name__
    try:
        immutable.__name__ = "nope"
    except TypeError:
        pass
    else:
        raise AssertionError("renamed the immutable type %r" % (immutable,))
    assert immutable.__name__ == before, immutable.__name__

print("OK")
