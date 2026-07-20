class X:
    pass


class Y:
    pass


class A(X, Y):
    pass


assert (A, X, Y, object) == A.__mro__


class B(X, Y):
    pass


assert (B, X, Y, object) == B.__mro__


class C(A, B):
    pass


assert (C, A, B, X, Y, object) == C.__mro__

assert type.__mro__ == (type, object)


# typeobject.py:1585-1630 compute_mro — a metaclass override supplies the
# installed MRO. Extra ancestors are registered for invalidation, but are not
# reported as real direct subclasses.
class CustomAncestor:
    marker = 1


class RealBase:
    pass


class Meta(type):
    def mro(cls):
        return [cls, CustomAncestor, RealBase, object]


class CustomMro(RealBase, metaclass=Meta):
    pass


assert CustomMro.__mro__ == (CustomMro, CustomAncestor, RealBase, object)
assert CustomMro.marker == 1
assert CustomMro not in CustomAncestor.__subclasses__()
assert CustomMro in RealBase.__subclasses__()
