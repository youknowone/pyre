from testutils import assert_raises

# test lists
assert 3 in [1, 2, 3]
assert 3 not in [1, 2]

assert not (3 in [1, 2])
assert not (3 not in [1, 2, 3])

# test strings
assert "foo" in "foobar"
assert "whatever" not in "foobar"

# test bytes
assert b"foo" in b"foobar"
assert b"whatever" not in b"foobar"
assert b"1" < b"2"
assert b"1" <= b"2"
assert b"5" <= b"5"
assert b"4" > b"2"
assert not b"1" >= b"2"
assert b"10" >= b"10"
assert_raises(TypeError, lambda: bytes() > 2)

# test tuple
assert 1 in (1, 2)
assert 3 not in (1, 2)

# test set
assert 1 in set([1, 2])
assert 3 not in set([1, 2])

# test dicts
assert "a" in {"a": 0, "b": 0}
assert "c" not in {"a": 0, "b": 0}
assert 1 in {1: 5, 7: 12}
assert 5 not in {9: 10, 50: 100}
assert True in {True: 5}
assert False not in {True: 5}

# test iter
assert 3 in iter([1, 2, 3])
assert 3 not in iter([1, 2])

# test sequence
assert 1 in range(0, 2)
assert 3 not in range(0, 2)


# test __contains__ in user objects
class MyNotContainingClass:
    pass


assert_raises(TypeError, lambda: 1 in MyNotContainingClass())


class MyContainingClass:
    def __init__(self, value):
        self.value = value

    def __contains__(self, something):
        return something == self.value


assert 2 in MyContainingClass(2)
assert 1 not in MyContainingClass(2)


# PyPy descroperation.py sequence_contains obtains iter(container) before
# scanning.  An explicit __iter__ therefore wins over any __getitem__ path.
class IterOnly:
    def __iter__(self):
        return iter((1, 2, 3))

    def __getitem__(self, index):
        raise AssertionError("membership bypassed __iter__")


assert 2 in IterOnly()
assert 4 not in IterOnly()


# Special methods are resolved on the type, never from the instance dict.
instance_only = MyNotContainingClass()
instance_only.__contains__ = lambda value: True
assert_raises(TypeError, lambda: 1 in instance_only)


# Python 3.14 membership diagnostics and exception propagation while
# acquiring the fallback iterator.
class RaisingIterTypeError:
    def __iter__(self):
        raise TypeError("custom iterator error")


try:
    1 in RaisingIterTypeError()
except TypeError as exc:
    assert str(exc) == (
        "argument of type 'RaisingIterTypeError' is not a container or iterable"
    )
else:
    raise AssertionError("membership accepted a raising iterator")


class RaisingIterValueError:
    def __iter__(self):
        raise ValueError("custom iterator error")


assert_raises(ValueError, lambda: 1 in RaisingIterValueError())


class BlockContains(IterOnly):
    __contains__ = None


try:
    1 in BlockContains()
except TypeError as exc:
    assert str(exc) == "'BlockContains' object is not a container"
else:
    raise AssertionError("__contains__ = None did not block iteration fallback")
