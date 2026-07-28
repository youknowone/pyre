import copy
import gc
import itertools
import weakref


assert isinstance(itertools._tee, type)
assert isinstance(itertools._tee_dataobject, type)
assert "__copy__" in itertools._tee.__dict__
assert "__reduce__" not in itertools._tee.__dict__
assert "__setstate__" not in itertools._tee.__dict__

a, b = itertools.tee([1, 2, 3])
assert type(a) is itertools._tee
assert next(a) == 1
assert next(a) == 2
assert next(b) == 1
assert next(a) == 3
assert next(b) == 2
assert next(b) == 3

weak_target, _ = itertools.tee(range(1))
weak_proxy = weakref.proxy(weak_target)
assert weak_proxy.__class__ is itertools._tee
del weak_target
gc.collect()
try:
    weak_proxy.__class__
except ReferenceError:
    pass
else:
    raise AssertionError("_tee weak reference remained live")

# Python 3.14 returns without even requesting an iterator when n == 0.
class ExplodingIter:
    def __iter__(self):
        raise AssertionError("tee(iterable, 0) called iter()")


assert itertools.tee(ExplodingIter(), 0) == ()


class IndexOnly:
    def __index__(self):
        return 3


copies = itertools.tee([10], IndexOnly())
assert len(copies) == 3
assert [list(item) for item in copies] == [[10], [10], [10]]

try:
    itertools.tee([], -1)
except ValueError as exc:
    assert str(exc) == "n must be >= 0"
else:
    raise AssertionError("negative tee count accepted")

for call in (
    lambda: itertools.tee(iterable=[]),
    lambda: itertools.tee([], n=2),
):
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError("tee accepted a keyword argument")

# __copy__ starts at the current cursor and shares the future source.
original = itertools._tee(iter([1, 2, 3]))
assert next(original) == 1
cloned = copy.copy(original)
assert next(original) == 2
assert next(cloned) == 2
assert next(cloned) == 3
assert next(original) == 3

# Python 3.14 clones every result when tee() receives an existing _tee.
source = itertools._tee(iter([4, 5]))
left, right = itertools.tee(source)
assert left is not source
assert right is not source
assert list(left) == [4, 5]
assert list(right) == [4, 5]

# PyPy W_TeeIterable.running rejects a recursive pull of the same empty node.
class Reentrant:
    def __iter__(self):
        return self

    def __next__(self):
        return next(self.other)


reentrant = Reentrant()
first, reentrant.other = itertools.tee(reentrant)
try:
    next(first)
except RuntimeError as exc:
    assert str(exc) == "cannot re-enter the tee iterator"
else:
    raise AssertionError("recursive tee pull was accepted")

# A lagging copy keeps both the source iterator and cached objects alive.
owned = [object(), object()]
fast, slow = itertools.tee(owned)
first_owned = next(fast)
second_owned = next(fast)
del owned, fast
gc.collect()
assert next(slow) is first_owned
assert next(slow) is second_owned

# Python 3.14 exposes the pickle reconstruction type even though the public
# types no longer expose pickle methods.
assert type(itertools._tee_dataobject(iter(()), [], None)) is itertools._tee_dataobject
try:
    itertools._tee_dataobject(iter(()), (), None)
except TypeError:
    pass
else:
    raise AssertionError("_tee_dataobject accepted a non-list cache")
try:
    itertools._tee_dataobject(iter(()), [None] * 58, None)
except ValueError as exc:
    assert str(exc) == "Invalid arguments"
else:
    raise AssertionError("_tee_dataobject accepted an oversized cache")

for internal_type in (itertools._tee, itertools._tee_dataobject):
    try:
        class InvalidSubtype(internal_type):
            pass
    except TypeError:
        pass
    else:
        raise AssertionError(f"{internal_type.__name__} accepted subclassing")

print("OK")
