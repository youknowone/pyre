"""An unhashable builtin subclass is named by its own type, and `round()` is not.

The concrete kind decides that a `list` / `set` / `bytearray` / dict view is
unhashable, but the refusal reports `space.type(w_obj)`, which for a subclass is
the subclass.  Reporting the builtin base instead makes the message name a type
the caller never wrote.

`intobject.py:167,174 descr_round` answers both the absent-ndigits and the
non-negative-ndigits case with `self.int(space)`, so rounding an `int` subclass
— `bool` included — yields the base type, matching the float arm, which already
leaves `float` behind on the single-argument path.
"""


def message(fn):
    try:
        fn()
    except TypeError as exc:
        return str(exc)
    raise AssertionError("expected TypeError")


class UList(list):
    pass


class UDict(dict):
    pass


class USet(set):
    pass


class UByteArray(bytearray):
    pass


for cls in (UList, UDict, USet, UByteArray):
    assert message(lambda cls=cls: hash(cls())) == f"unhashable type: '{cls.__name__}'", cls
    # The container paths report the same name, whatever wrapper they add.
    assert cls.__name__ in message(lambda cls=cls: {cls(): 1}), cls
    assert cls.__name__ in message(lambda cls=cls: {cls()}), cls
    assert "'list'" not in message(lambda cls=cls: hash(cls())), cls

# A dict view keeps naming its own type — there is no subclass to confuse it
# with, and the values view stays hashable.
assert message(lambda: hash({}.keys())) == "unhashable type: 'dict_keys'"
assert message(lambda: hash({}.items())) == "unhashable type: 'dict_items'"
assert isinstance(hash({}.values()), int)

# A subclass that supplies `__hash__` is hashable, and the override is what runs.
class HashableList(list):
    def __hash__(self):
        return 7


assert hash(HashableList()) == 7
assert {HashableList(): 1}[HashableList()] == 1


class Sub(int):
    pass


class SubFloat(float):
    pass


assert type(round(Sub(5))) is int
assert type(round(Sub(5), 1)) is int
assert type(round(Sub(5), -1)) is int
assert type(round(True)) is int
assert type(round(False)) is int
assert round(Sub(5)) == 5
assert round(True) == 1
assert round(Sub(15), -1) == 20

# The float arm already normalizes on the single-argument path and keeps the
# base float when ndigits is supplied.
assert type(round(SubFloat(1.5))) is int
assert type(round(SubFloat(1.55), 1)) is float
assert round(SubFloat(1.5)) == 2

# A subtype that replaced `__round__` is dispatched to instead of rounded
# structurally, with or without ndigits, on both numeric bases.
class RoundsItself(int):
    def __round__(self, ndigits=None):
        return "rounded"


class FloatRoundsItself(float):
    def __round__(self, ndigits=None):
        return ("rounded", ndigits)


assert round(RoundsItself(1)) == "rounded"
assert round(RoundsItself(1), 2) == "rounded"
assert RoundsItself(1).__round__() == "rounded"
assert round(FloatRoundsItself(1.5)) == ("rounded", None)
assert round(FloatRoundsItself(1.5), 2) == ("rounded", 2)


# `__round__ = None` is looked up and called like any other value, so the
# refusal is the call's, not a "does not define __round__" report.
class NoRound(int):
    __round__ = None


assert message(lambda: round(NoRound(1))) == "'NoneType' object is not callable"
assert message(lambda: round("x")) == "type str doesn't define __round__ method"


# An ordinary object still reaches its own `__round__`.
class OnlyRound:
    def __round__(self, ndigits=None):
        return "obj"


assert round(OnlyRound()) == "obj"

print("OK")
