import gc

from _weakref import getweakrefcount, getweakrefs, proxy, ref

from testutils import assert_raises


class X:
    pass


a = X()
b = ref(a)

assert callable(b)
assert b() is a

# Test __callback__ property
assert b.__callback__ is None, (
    "weakref without callback should have __callback__ == None"
)

callback = lambda r: None
c = ref(a, callback)
assert c.__callback__ is callback, "weakref with callback should return the callback"
assert getweakrefcount(a) == 2
assert set(getweakrefs(a)) == {b, c}

# Test __callback__ is read-only
try:
    c.__callback__ = lambda r: None
    assert False, "Setting __callback__ should raise AttributeError"
except AttributeError:
    pass

# Test __callback__ after referent deletion
x = X()
seen = []
cb1 = lambda r: seen.append((1, r()))
cb2 = lambda r: seen.append((2, r()))
w1 = ref(x, cb1)
w2 = ref(x, cb2)
assert w1.__callback__ is cb1
assert w2.__callback__ is cb2
del x
gc.collect()
assert seen == [(2, None), (1, None)]
assert w1.__callback__ is None
assert w2.__callback__ is None


class G:
    def __init__(self, h):
        self.h = h


g = G(5)
p = proxy(g)

assert p.h == 5

del g
gc.collect()

assert_raises(ReferenceError, lambda: p.h)
