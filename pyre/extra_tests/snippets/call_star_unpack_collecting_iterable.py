"""``f(*iterable)`` when producing each item runs a collection.

The starred iterable of a ``CALL_FUNCTION_EX`` is unpacked by the call itself,
through one of two drains: a generator runs its suspended frame per item, and
anything else runs the iterator's ``__next__``.  Both keep producing while the
items already pulled are held only by the drain, so an item has to survive
every turn that follows it.
"""

import gc


def f(*args):
    return args


def gen(n):
    for i in range(n):
        gc.collect()
        yield [i]


class CollectingIterator:
    """Not a generator, so the ``next()`` drain runs instead of ``unpack_into``."""

    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        i = self.i
        self.i += 1
        gc.collect()
        return [i]


class CollectingIterable:
    """``__iter__`` itself collects, before a single item is produced."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        gc.collect()
        return iter([[i] for i in range(self.n)])


for n in range(5):
    expected = [[i] for i in range(n)]
    for source in (gen(n), CollectingIterator(n), CollectingIterable(n)):
        got = f(*source)
        assert len(got) == n, (n, len(got))
        assert [list(x) for x in got] == expected, (n, got)
        # Each turn produced a fresh list; two slots holding one address means
        # an item was dropped and its storage handed to a later one.
        assert len({id(x) for x in got}) == n, (n, [id(x) for x in got])

# The unpacked set also has to survive the callee's frame setup and whatever
# collects after the call returns.
r = f(*gen(4))
gc.collect()
assert [list(x) for x in r] == [[0], [1], [2], [3]], r

# A `*` unpack alongside keywords reaches the same drain.
def g(*args, **kwargs):
    return args, sorted(kwargs.items())


args, kwargs = g(*gen(3), x=1, **{"y": 2})
assert [list(a) for a in args] == [[0], [1], [2]], args
assert kwargs == [("x", 1), ("y", 2)], kwargs

# An exhausted or empty producer still returns the empty unpack.
assert f(*gen(0)) == ()
assert f(*CollectingIterator(0)) == ()

# A generator already drained by an earlier unpack yields nothing the second
# time; `unpack_into` returns without appending.
once = gen(2)
assert [list(x) for x in f(*once)] == [[0], [1]]
assert f(*once) == ()
