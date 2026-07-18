from testutils import assert_raises

assert (1, 2) == (1, 2)

x = (1, 2)
assert x[0] == 1

y = (1,)
assert y[0] == 1

assert x + y == (1, 2, 1)

assert x * 3 == (1, 2, 1, 2, 1, 2)
assert 3 * x == (1, 2, 1, 2, 1, 2)
assert x * 0 == ()
assert x * -1 == ()  # integers less than zero treated as 0

assert y < x, "tuple __lt__ failed"
assert x > y, "tuple __gt__ failed"


b = (1, 2, 3)
assert b.index(2) == 1

recursive_list = []
recursive = (recursive_list,)
recursive_list.append(recursive)
assert repr(recursive) == "([(...)],)"

assert (None, "", 1).index(1) == 2
assert 1 in (None, "", 1)


class Foo(object):
    def __eq__(self, x):
        return False


foo = Foo()
assert (foo,) == (foo,)

a = (1, 2, 3)
a += (1,)
assert a == (1, 2, 3, 1)

b = (55, *a)
assert b == (55, 1, 2, 3, 1)

assert () is ()  # noqa

a = ()
b = ()
assert a is b

assert (1,).__ne__((2,))
assert not (1,).__ne__((1,))

# tuple gt, ge, lt, le
assert_raises(TypeError, lambda: (0, ()) < (0, 0))
assert_raises(TypeError, lambda: (0, ()) <= (0, 0))
assert_raises(TypeError, lambda: (0, ()) > (0, 0))
assert_raises(TypeError, lambda: (0, ()) >= (0, 0))

assert_raises(TypeError, lambda: (0, 0) < (0, ()))
assert_raises(TypeError, lambda: (0, 0) <= (0, ()))
assert_raises(TypeError, lambda: (0, 0) > (0, ()))
assert_raises(TypeError, lambda: (0, 0) >= (0, ()))

assert (0, 0) < (1, -1)
assert (0, 0) < (0, 0, 1)
assert (0, 0) < (0, 0, -1)
assert (0, 0) <= (0, 0, -1)
assert not (0, 0, 0, 0) <= (0, -1)

assert (0, 0) > (-1, 1)
assert (0, 0) >= (-1, 1)
assert (0, 0, 0) >= (-1, 1)

assert (0, 0) <= (0, 1)
assert (0, 0) <= (0, 0)
assert (0, 0) <= (0, 0)
assert not (0, 0) > (0, 0)
assert not (0, 0) < (0, 0)

assert not (float("nan"), float("nan")) <= (float("nan"), 1)
assert not (float("nan"), float("nan")) <= (float("nan"), float("nan"))
assert not (float("nan"), float("nan")) >= (float("nan"), float("nan"))
assert not (float("nan"), float("nan")) < (float("nan"), float("nan"))
assert not (float("nan"), float("nan")) > (float("nan"), float("nan"))

assert (float("inf"), float("inf")) >= (float("inf"), 1)
assert (float("inf"), float("inf")) <= (float("inf"), float("inf"))
assert (float("inf"), float("inf")) >= (float("inf"), float("inf"))
assert not (float("inf"), float("inf")) < (float("inf"), float("inf"))
assert not (float("inf"), float("inf")) > (float("inf"), float("inf"))

exact_tuple = tuple(range(3))
assert exact_tuple * 1 is exact_tuple


class TupleSubclass(tuple):
    pass


tuple_subclass = TupleSubclass((1, 2))
assert tuple_subclass * 1 == (1, 2)
assert type(tuple_subclass * 1) is tuple


class TupleContainsError(Exception):
    pass


class RaisingTupleItem:
    def __eq__(self, other):
        raise TupleContainsError


assert_raises(TupleContainsError, (RaisingTupleItem(), 1).__contains__, 1)


import gc

finalized_tuple_subclasses = []


class FinalizedTupleSubclass(tuple):
    def __del__(self):
        finalized_tuple_subclasses.append(True)


finalized_tuple = FinalizedTupleSubclass()
del finalized_tuple
gc.collect()
assert finalized_tuple_subclasses == [True]
