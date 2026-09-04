# CPython-suite gap: iterator-protocol tests never hot-loop a `for` over a user
# instance whose `__iter__` is rebound, blanked, or returns a different object
# on a later iteration than the one the trace recorded.
# parity-tests reason: pins the arms `instance_iter` must NOT take, and the one
# it takes only until the type changes under it, without adding their guard
# traffic to the synth gate.

"""`iter`'s instance arm calls `__iter__` and then requires the result to carry
`__next__` on its type.  The inline decides that check against the receiver's
guarded class, so every way of making the answer something else must be
observed."""


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 30000


class Counter:
    def __init__(self, limit):
        self.limit = limit
        self.j = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.j >= self.limit:
            raise StopIteration
        v = self.j
        self.j += 1
        return v


# `__iter__` hands back a fresh object, not the receiver: the identity the
# inline's check rests on does not hold, so the residual runs the real
# `iter_check_is_iterator`.
class Fresh:
    def __init__(self, limit):
        self.limit = limit

    def __iter__(self):
        return Counter(self.limit)


def fresh_result():
    src = Fresh(3)
    total = 0
    for _ in range(N):
        for x in src:
            total += x
    return total


assert fresh_result() == 3 * N


# descroperation.py:339-341 — an explicit `__iter__ = None` marks the type
# non-iterable even though the lookup succeeds.
class Blanked:
    __iter__ = None

    def __next__(self):
        raise StopIteration


def blanked():
    obj = Blanked()
    hits = 0
    for _ in range(N):
        try:
            for _x in obj:
                pass
        except TypeError:
            hits += 1
    return hits


assert blanked() == N


# `__iter__` returns the receiver but the receiver has no `__next__`:
# `iter_check_is_iterator` raises, and the message names the returned object's
# type rather than the one `for` was written over.
class NotAnIterator:
    def __iter__(self):
        return self


def not_an_iterator():
    obj = NotAnIterator()
    hits = 0
    for _ in range(N):
        try:
            for _x in obj:
                pass
        except TypeError as err:
            if "non-iterator" in str(err):
                hits += 1
    return hits


assert not_an_iterator() == N


# The fold pins the receiver type's version tag, so replacing `__iter__`
# mid-loop must switch away from the recorded body on the very next iteration.
class Rebound:
    def __init__(self):
        self.j = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.j >= 1:
            raise StopIteration
        self.j += 1
        return "recorded"


def rebound():
    obj = Rebound()
    seen = None
    for i in range(N):
        obj.j = 0
        for x in obj:
            seen = x
        if i == N // 2:
            Rebound.__iter__ = lambda self: iter(["replaced"])
    return seen


assert rebound() == "replaced"


# The same rebind in the other direction: a type whose `__iter__` starts out
# returning a fresh iterator and becomes `return self` mid-loop, so the fold
# has to start firing on a location that recorded the residual.
class Widened:
    def __init__(self):
        self.j = 0

    def __iter__(self):
        return iter(("first",))

    def __next__(self):
        if self.j >= 1:
            raise StopIteration
        self.j += 1
        return "second"


def widened():
    obj = Widened()
    seen = None
    for i in range(N):
        obj.j = 0
        for x in obj:
            seen = x
        if i == N // 2:
            Widened.__iter__ = lambda self: self
    return seen


assert widened() == "second"


# A subclass that inherits `__iter__` and overrides `__next__`: the version tag
# the fold pins is the instance's own type's, and the base's answer must not be
# reused for it.
class Doubling(Counter):
    def __next__(self):
        if self.j >= self.limit:
            raise StopIteration
        v = self.j
        self.j += 1
        return v * 2


def subclass_next():
    total = 0
    for _ in range(N):
        for x in Doubling(3):
            total += x
    return total


assert subclass_next() == 6 * N

print("PASS")
