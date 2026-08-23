# pyre-check: gate=1
"""`gc.get_referents` reports an object's type when that type is a heap type.

`subtype_traverse` visits it — "for a heaptype, the instances count as
references to the type" — and a class reports its metaclass on the same rule.
A statically allocated type is not reported, which is why an ordinary class,
whose metaclass is `type`, does not name it.

Order is not asserted here: `subtype_traverse` puts the type after an
instance's managed-dict values but ahead of a tuple subclass's items.
"""

import collections
import gc


class Plain:
    def __init__(self):
        self.a = "A"


class Meta(type):
    pass


class WithMeta(metaclass=Meta):
    pass


class Slotted:
    __slots__ = ("x",)


inst = Plain()
referents = gc.get_referents(inst)
assert Plain in referents, referents
assert "A" in referents, referents

slotted = Slotted()
slotted.x = "X"
assert Slotted in gc.get_referents(slotted), gc.get_referents(slotted)

# A class names its metaclass only when that metaclass is a heap type.
assert Meta in gc.get_referents(WithMeta), gc.get_referents(WithMeta)
assert type not in gc.get_referents(Plain), gc.get_referents(Plain)

# Exactly once, including for a layout that already carries the edge.
NT = collections.namedtuple("NT", "a")
assert gc.get_referents(NT(1)).count(NT) == 1, gc.get_referents(NT(1))
assert gc.get_referents(inst).count(Plain) == 1, gc.get_referents(inst)

# A non-GC object has no referents at all, so none names a type either.
assert gc.get_referents(1) == []
assert gc.get_referents("s") == []
assert gc.get_referents(1.5) == []
