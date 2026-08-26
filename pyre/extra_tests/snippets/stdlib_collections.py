from collections import defaultdict, deque
from testutils import assert_raises


# Python 3.14's defaultdict.__missing__ preserves a value installed by a
# re-entrant factory call instead of overwriting it with the outer result.
defaultdict_key = "conflict"
defaultdict_calls = 0


def reentrant_default_factory():
    global defaultdict_calls
    defaultdict_calls += 1
    call = defaultdict_calls
    if call == 1:
        reentrant_defaultdict[defaultdict_key]
    return call


reentrant_defaultdict = defaultdict(reentrant_default_factory)
assert reentrant_defaultdict[defaultdict_key] == 2
assert defaultdict_calls == 2


class DefaultDictSetDefaultOverride(defaultdict):
    def setdefault(self, *args):
        raise AssertionError("defaultdict.__missing__ called overridden setdefault")


setdefault_override = DefaultDictSetDefaultOverride(lambda: 3)
assert setdefault_override["key"] == 3


# PyPy app_defaultdict.defaultdict.__init__ writes default_factory through
# the class slot descriptor, bypassing a subclass __setattr__ override.
class DefaultDictSetAttrOverride(defaultdict):
    def __setattr__(self, name, value):
        raise AssertionError("defaultdict initialization called __setattr__")


setattr_override = DefaultDictSetAttrOverride(int)
assert setattr_override["missing"] == 0
setattr_override["updated"] += 3
assert setattr_override["updated"] == 3

d = deque([0, 1, 2])

d.append(1)
d.appendleft(3)

assert d == deque([3, 0, 1, 2, 1])

assert d <= deque([4])

assert d.copy() is not d

d = deque([1, 2, 3], 5)

d.extend([4, 5, 6])

assert d == deque([2, 3, 4, 5, 6]), d

d.remove(4)

assert d == deque([2, 3, 5, 6])

d.clear()

assert d == deque()

assert d == deque([], 4)

assert deque([1, 2, 3]) * 2 == deque([1, 2, 3, 1, 2, 3])

assert deque([1, 2, 3], 4) * 2 == deque([3, 1, 2, 3])


class DequeRepeatIndex:
    def __index__(self):
        # The receiver and count must stay rooted across arbitrary Python code.
        import gc

        gc.collect()
        return 2


class DequeRepeatReflected:
    def __rmul__(self, other):
        return ("reflected", other)


class DequeRepeatOverride(deque):
    def __mul__(self, other):
        return ("override", other)

    def __rmul__(self, other):
        return ("reflected override", other)


class SlottedDeque(deque):
    __slots__ = ("slot_value", "__dict__")


repeat_source = deque([1, 2])
assert repeat_source * DequeRepeatIndex() == deque([1, 2, 1, 2])
assert DequeRepeatIndex() * repeat_source == deque([1, 2, 1, 2])
assert repeat_source * DequeRepeatReflected() == ("reflected", repeat_source)
repeat_override = DequeRepeatOverride([1])
assert repeat_override * 3 == ("override", 3)
assert 3 * repeat_override == ("reflected override", 3)
assert_raises(TypeError, deque.__mul__, repeat_source, object())
assert_raises(OverflowError, lambda: repeat_source * (10**100))

slotted_deque = SlottedDeque([1])
slotted_deque.slot_value = 2
slotted_deque.dict_value = 3
assert slotted_deque.__dict__ == {"dict_value": 3}
assert slotted_deque.__getstate__() == (
    {"dict_value": 3},
    {"slot_value": 2},
)
del slotted_deque.slot_value
assert_raises(AttributeError, getattr, slotted_deque, "slot_value")

repeat_big = deque([0])
repeat_big *= 2**8
assert_raises(MemoryError, lambda: repeat_big * (2**56))
assert_raises(MemoryError, lambda: (2**56) * repeat_big)


def repeat_big_in_place():
    value = repeat_big.copy()
    value *= 2**56


assert_raises(MemoryError, repeat_big_in_place)

# Optional constructor args, including the `maxlen` keyword form.
assert deque(maxlen=5).maxlen == 5
assert deque().maxlen is None
assert deque(maxlen=2) == deque([], 2)
assert deque([1, 2, 3], maxlen=2) == deque([2, 3], 2)
assert deque(maxlen=None) == deque()

assert deque(maxlen=3) == deque()

assert deque([1, 2, 3, 4], maxlen=2) == deque([3, 4])

assert len(deque([1, 2, 3, 4])) == 4

assert d >= d
assert not (d > d)
assert d <= d
assert not (d < d)
assert d == d
assert not (d != d)


# Test that calling an evil __repr__ can't hang deque
class BadRepr:
    def __repr__(self):
        self.d.pop()
        return ""


b = BadRepr()
d = deque([1, b, 2])
b.d = d
repr(d)


# Dict subclasses keep their mapping payload independently of an instance
# __dict__.  defaultdict follows PyPy's slotted layout and exposes the
# default_factory member descriptor on the class (dataclasses.asdict relies on
# that class-level probe).
from collections import defaultdict


defaults = defaultdict(list)
defaults["items"].append(12)
assert defaults["items"] == [12]
assert hasattr(defaultdict, "default_factory")
assert not hasattr(defaults, "__dict__")
assert defaults.copy().default_factory is list


class SlottedDict(dict):
    __slots__ = ("marker",)


slotted = SlottedDict()
slotted.marker = 3
slotted["key"] = "value"
assert slotted == {"key": "value"}
assert slotted.marker == 3
assert not hasattr(slotted, "__dict__")
