from collections import defaultdict, deque


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
