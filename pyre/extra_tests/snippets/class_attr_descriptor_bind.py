# pyre-check: gate=1
# The guards `load_type_attr` and `load_bound_method_attr` depend on, each
# revoked while the loop that folded it is already compiled.


class Base:
    def val(self, x):
        return x + 1

    @staticmethod
    def sm(x):
        return x + 1


class Child(Base):
    pass


# A plain function read off a class is returned unbound; an instance binds it.
assert Base.val is Base.__dict__["val"]
assert Child.val is Base.__dict__["val"]
o = Child()
g = o.val
assert type(g).__name__ == "method", type(g).__name__
assert g.__self__ is o
assert g.__func__ is Base.__dict__["val"]
assert o.val is not o.val, "each read builds its own Method"

# `__get__(None, None)` stays invalid: no owner means no class access.
try:
    Base.__dict__["val"].__get__(None, None)
    raise AssertionError("expected TypeError")
except TypeError:
    pass


# An instance attribute of the same name shadows the type lookup.  The map
# guard has to observe the store that installs it.
def shadow_mid_loop(n):
    p = Child()
    out = []
    for i in range(n):
        out.append(p.val)
        if i == n - 2:
            p.val = "SHADOW"
    return out


shadowed = shadow_mid_loop(5000)
assert type(shadowed[0]).__name__ == "method"
assert shadowed[-1] == "SHADOW", shadowed[-1]


# Rebinding the method on the type bumps the version tag.
class RBase:
    def val(self, x):
        return x + 1


class RChild(RBase):
    pass


def rebind_mid_loop(n):
    p = RChild()
    out = []
    for i in range(n):
        out.append(p.val(0))
        if i == n - 2:
            RBase.val = lambda self, x: "REBOUND"
    return out


rebound = rebind_mid_loop(5000)
assert rebound[0] == 1
assert rebound[-1] == "REBOUND", rebound[-1]


# Re-initialising a `staticmethod` in place moves `w_function` without
# touching the class dict, the wrapper's address, or any version tag.  That is
# what the quasi-immutable slot pin is for.
class SHolder:
    pass


wrapper = staticmethod(lambda x: "first")
SHolder.s = wrapper


def reinit_mid_loop(n):
    out = []
    for i in range(n):
        out.append(SHolder.s(1))
        if i == n - 2:
            wrapper.__init__(lambda x: "second")
    return out


reinit = reinit_mid_loop(5000)
assert reinit[0] == "first"
assert reinit[-1] == "second", reinit[-1]


# A `staticmethod` subclass may override `__get__`, which the exact-type test
# leaves to the general lookup.
class MySM(staticmethod):
    def __get__(self, obj, objtype=None):
        return "overridden"


class WithSub:
    m = MySM(lambda: 1)


def subclass_get(n):
    seen = None
    for _ in range(n):
        seen = WithSub.m
    return seen


assert subclass_get(5000) == "overridden"
assert WithSub().m == "overridden"


# A devolved receiver keeps one map across a shadowing store, so the fold
# declines it and the residual answers.
def devolved(n):
    p = Child()
    p.__dict__["z"] = 1
    seen = None
    for _ in range(n):
        seen = p.val
    return seen


assert devolved(5000).__func__ is Base.__dict__["val"]


# An exception receiver pins its still-unallocated `w_dict` slot instead.
def exception_receiver(n):
    e = ValueError("x")
    seen = None
    for _ in range(n):
        seen = e.args
    return seen


assert exception_receiver(5000) == ("x",)


# Two receiver classes at one site: the class guard has to separate them.
class Other:
    def val(self, x):
        return x + 10


def alternating(n):
    a, b = Child(), Other()
    total = 0
    for i in range(n):
        r = a if i & 1 else b
        total += r.val(0)
    return total


assert alternating(5000) == 5000 // 2 * 1 + 5000 // 2 * 10


# A bound method that outlives its loop: the virtual has to materialise.
def escaping(n):
    p = Child()
    keep = None
    for _ in range(n):
        keep = p.val
    return keep


kept = escaping(5000)
assert kept(5) == 6
assert kept.__func__ is Base.__dict__["val"]
