from testutils import assert_raises


class Foo(object):
    pass


Foo.__repr__ = Foo.__str__

foo = Foo()
# Since the default __str__ implementation calls __repr__ and __repr__ is
# actually __str__, str(foo) should raise a RecursionError.
assert_raises(RecursionError, str, foo)


# A comparison override implemented natively re-enters the comparison operator
# without pushing a Python frame, so the frame-count limit never sees the
# cycle: binding `_operator.eq` as a bound method makes `c == c` call
# `_operator.eq(c, c)`, which is the same operator again.  The native stack
# check is what has to answer.
import types
import _operator


class Cmp(object):
    pass


cmp = Cmp()
Cmp.__eq__ = types.MethodType(_operator.eq, cmp)
assert_raises(RecursionError, lambda: cmp == cmp)
