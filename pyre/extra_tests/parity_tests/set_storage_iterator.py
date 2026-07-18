"""Set storage and iterator parity.

Structural source: ``pypy/objspace/std/setobject.py`` (W_BaseSetObject,
ObjectSetStrategy, IteratorImplementation, W_SetIterObject).  Python 3.14 is
the oracle for the concrete iterator name and RuntimeError spelling.
"""


class Counted:
    calls = 0

    def __hash__(self):
        type(self).calls += 1
        return 17


x = Counted()
s = {x}
Counted.calls = 0
assert s.copy() == s
assert Counted.calls == 0

t = {x}
Counted.calls = 0
assert t.pop() is x
assert Counted.calls == 0

t = {x}
Counted.calls = 0
t.clear()
assert Counted.calls == 0

f = frozenset({1})
assert f.copy() is f

x = Counted()
f = frozenset([x])
Counted.calls = 0
hash(f)
hash(f)
assert Counted.calls == 0


class FS(frozenset):
    pass


assert type(FS({1}).copy()) is frozenset
assert repr(FS()) == "FS()"
assert repr(FS({1})) == "FS({1})"


class S(set):
    pass


assert repr(S()) == "S()"
assert repr(S({1})) == "S({1})"


class Left(set):
    def __or__(self, other):
        return "left-or"


class Right(Left):
    def __ror__(self, other):
        return "right-ror"


assert Left({1}) | {2} == "left-or"
assert {1} | Right({2}) == "right-ror"
assert Left({1}) | Right({2}) == "right-ror"
assert set.__eq__({1}, [1]) is NotImplemented
assert set.__ne__({1}, [1]) is NotImplemented
assert set.__lt__({1}, [1]) is NotImplemented
assert set.__le__({1}, [1]) is NotImplemented
assert set.__gt__({1}, [1]) is NotImplemented
assert set.__ge__({1}, [1]) is NotImplemented
assert set.__rsub__({2}, {1, 2}) == {1}

it = iter({1, 2})
assert type(it).__name__ == "set_iterator"
assert iter(it) is it
assert it.__length_hint__() == 2
next(it)
assert it.__length_hint__() == 1

s = {1, 2}
it = iter(s)
s.add(3)
for _ in range(2):
    try:
        next(it)
    except RuntimeError as exc:
        assert str(exc) == "Set changed size during iteration"
    else:
        raise AssertionError("set size mutation did not invalidate iterator")
assert it.__length_hint__() == 0

# GET_ITER must use the same live set iterator as iter(s), not a snapshot
# sequence iterator.  CPython test_set.test_changingSizeWhileIterating and
# PyPy W_SetIterObject both require the next FOR_ITER step to see the growth.
s = {1, 2, 3}
try:
    for value in s:
        s.update([4])
except RuntimeError as exc:
    assert str(exc) == "Set changed size during iteration"
else:
    raise AssertionError("set for-loop did not observe a size mutation")

print("OK")
