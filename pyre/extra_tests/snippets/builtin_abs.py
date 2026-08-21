assert abs(-3) == 3
assert abs(7) == 7
assert abs(-3.21) == 3.21
assert abs(6.25) == 6.25


# `abs()` reaches `__abs__` through the type (operation.py:14 -> `space.abs`),
# so a subtype that replaced the builtin one is dispatched to.
class AbsInt(int):
    def __abs__(self):
        return "custom int"


class AbsFloat(float):
    def __abs__(self):
        return "custom float"


class AbsComplex(complex):
    def __abs__(self):
        return "custom complex"


assert abs(AbsInt(-5)) == "custom int"
assert abs(AbsFloat(-1.5)) == "custom float"
assert abs(AbsComplex(3 + 4j)) == "custom complex"


# A subtype that inherits the builtin one keeps the structural answer, and the
# exact type that goes with it.
class PlainInt(int):
    pass


assert abs(PlainInt(-5)) == 5
assert type(abs(PlainInt(-5))) is int
assert abs(True) == 1 and type(abs(True)) is int


# `__abs__ = None` is a replacement too: the lookup finds it and the call fails.
class NoAbs(int):
    __abs__ = None


try:
    abs(NoAbs(-5))
except TypeError:
    pass
else:
    raise AssertionError("abs() answered for a subtype that unset __abs__")


# The builtin slot stays structural, so an override delegating back to it
# terminates instead of re-entering the lookup that reached it.
class DelegatesToInt(int):
    def __abs__(self):
        return int.__abs__(self)


assert abs(DelegatesToInt(-5)) == 5
assert int.__abs__(AbsInt(-5)) == 5
