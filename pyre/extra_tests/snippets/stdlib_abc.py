import _abc
import abc

from testutils import assert_raises


class CustomInterface(abc.ABC):
    @abc.abstractmethod
    def a(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return NotImplemented


with assert_raises(TypeError):
    CustomInterface()


class Concrete:
    def a(self):
        pass


CustomInterface.register(Concrete)


class SubConcrete(Concrete):
    pass


assert issubclass(Concrete, CustomInterface)
assert issubclass(SubConcrete, CustomInterface)
assert not issubclass(tuple, CustomInterface)

assert isinstance(Concrete(), CustomInterface)
assert isinstance(SubConcrete(), CustomInterface)
assert not isinstance((), CustomInterface)


# `__abc_tpflags__` in a class body is consumed by `_abc_init`, not by class
# creation: only a class going through `ABCMeta` may take the structural-match
# marker, or `case [...]` would start accepting a plain object.
def matches_sequence_pattern(value):
    match value:
        case [*_]:
            return True
        case _:
            return False


def matches_mapping_pattern(value):
    match value:
        case {}:
            return True
        case _:
            return False


def abc_with_tpflags(flags):
    return abc.ABCMeta("Tagged", (), {"__abc_tpflags__": flags})


PY_TPFLAGS_SEQUENCE = 1 << 5
PY_TPFLAGS_MAPPING = 1 << 6


class PlainWithTpflags:
    __abc_tpflags__ = PY_TPFLAGS_SEQUENCE


assert not matches_sequence_pattern(PlainWithTpflags())
assert PlainWithTpflags.__abc_tpflags__ == PY_TPFLAGS_SEQUENCE

assert matches_sequence_pattern(abc_with_tpflags(PY_TPFLAGS_SEQUENCE)())
assert matches_mapping_pattern(abc_with_tpflags(PY_TPFLAGS_MAPPING)())
# The value is masked, so a bit outside the two collection flags is ignored
# rather than rejected.
assert matches_sequence_pattern(abc_with_tpflags(PY_TPFLAGS_SEQUENCE | 1)())
assert not matches_sequence_pattern(abc_with_tpflags(0)())

# Whatever it holds, the attribute is consumed -- a leftover would be inherited
# by every subclass and read again.
assert "__abc_tpflags__" not in abc_with_tpflags(PY_TPFLAGS_SEQUENCE).__dict__
assert "__abc_tpflags__" not in abc_with_tpflags("not an int").__dict__
assert not matches_sequence_pattern(abc_with_tpflags("not an int")())

# A value past the machine word is an error, not a silently skipped one, and a
# collection bit inside such a value does not survive the conversion.
with assert_raises(OverflowError):
    abc_with_tpflags(1 << 100)
with assert_raises(OverflowError):
    abc_with_tpflags((1 << 130) | PY_TPFLAGS_SEQUENCE)
with assert_raises(OverflowError):
    abc_with_tpflags(-(1 << 130))

# `-1` carries both collection bits.
with assert_raises(TypeError):
    abc_with_tpflags(-1)
with assert_raises(TypeError):
    abc_with_tpflags(PY_TPFLAGS_SEQUENCE | PY_TPFLAGS_MAPPING)

# `PyLong_CheckExact`: a bool and an `int` subclass are consumed and ignored.
assert not matches_sequence_pattern(abc_with_tpflags(True)())


class TpflagsInt(int):
    pass


assert not matches_sequence_pattern(abc_with_tpflags(TpflagsInt(PY_TPFLAGS_SEQUENCE))())

# The attribute is taken out of the type dict itself, so a rejected value is
# consumed all the same and a metaclass `__delattr__` never sees it.
deleted = []


class WatchingMeta(abc.ABCMeta):
    def __delattr__(cls, name):
        deleted.append(name)
        super().__delattr__(name)


rejected = WatchingMeta("Rejected", (), {})
rejected.__abc_tpflags__ = PY_TPFLAGS_SEQUENCE | PY_TPFLAGS_MAPPING
with assert_raises(TypeError):
    _abc._abc_init(rejected)
assert "__abc_tpflags__" not in rejected.__dict__
assert deleted == []

WatchingMeta("Tagged", (), {"__abc_tpflags__": PY_TPFLAGS_SEQUENCE})
assert deleted == []

# Registering under a marked ABC hands the marker to the registered class and
# its descendants, but never to an immutable type: a `str` matching `case [...]`
# is the one thing a sequence pattern must not accept.
assert not matches_sequence_pattern("ab")
assert not matches_sequence_pattern(b"ab")


class Marked(abc.ABCMeta("SeqBase", (), {"__abc_tpflags__": PY_TPFLAGS_SEQUENCE})):
    pass


class Unrelated:
    pass


class UnrelatedChild(Unrelated):
    pass


Marked.register(Unrelated)
assert matches_sequence_pattern(Unrelated())
assert matches_sequence_pattern(UnrelatedChild())
Marked.register(str)
assert not matches_sequence_pattern("ab")
