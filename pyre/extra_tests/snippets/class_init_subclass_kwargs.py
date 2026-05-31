# __init_subclass__ receives the keyword arguments from the class
# definition (`class C(Base, key=value)`).  Self-contained (no
# testutils import) so it runs on every backend.


class Base:
    def __init_subclass__(cls, *, tag=None, **kw):
        cls.tag = tag
        cls.extra = kw


class Child(Base, tag="x", more=9):
    pass


assert Child.tag == "x"
assert Child.extra == {"more": 9}


# A subclass with no keyword arguments still runs __init_subclass__
# (defaults apply, no extras collected).
class Plain(Base):
    pass


assert Plain.tag is None
assert Plain.extra == {}


# Keyword arguments resolve against the inherited __init_subclass__ down
# an inheritance chain.
class Grand(Child, tag="y"):
    pass


assert Grand.tag == "y"


# A keyword argument the signature cannot accept raises TypeError.
class Strict:
    def __init_subclass__(cls):
        pass


raised = False
try:
    class _Bad(Strict, unexpected=1):
        pass
except TypeError:
    raised = True
assert raised, "unexpected __init_subclass__ keyword should raise TypeError"


# __init_subclass__ keyword forwarding coexists with __set_name__.
class Marker:
    def __set_name__(self, owner, name):
        self.name = name


class WithMarker:
    slot = Marker()

    def __init_subclass__(cls, **kw):
        cls.seen = kw


class HasMarker(WithMarker, mode="fast"):
    pass


assert HasMarker.slot.name == "slot"
assert HasMarker.seen == {"mode": "fast"}

print("class_init_subclass_kwargs: OK")
