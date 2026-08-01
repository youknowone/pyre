"""Regression: __call__ resolved as a class attribute must honour the descriptor
protocol.  Only a plain function or method descriptor binds the receiver as an
implicit self; a callable instance or an already-bound method is called without
one.

A non-binding builtin as __call__ is deliberately left out: CPython 3.14 calls
it without a host self (``UsesAbs()(-7) == 7``) while PyPy prepends one and
raises TypeError, so no single expected output passes on every backend.
"""


# callable instance as __call__ — no implicit host self
class Callee:
    def __call__(self, *a):
        return a


class UsesInstance:
    __call__ = Callee()


assert UsesInstance()(1, 2) == (1, 2)


# a real descriptor as __call__ — its __get__ resolves to a different callable,
# which is then called without a host self
class ResolvingDescriptor:
    def __get__(self, obj, owner):
        return lambda x: ("resolved", type(obj).__name__, x)


class UsesDescriptor:
    __call__ = ResolvingDescriptor()


assert UsesDescriptor()(6) == ("resolved", "UsesDescriptor", 6)


# already-bound method as __call__ — bound to its own receiver, no host self
class Holder:
    def m(self, x):
        return ("m", x)


_h = Holder()


class UsesBound:
    __call__ = _h.m


assert UsesBound()(9) == ("m", 9)


# plain function as __call__ — DOES bind the receiver as self
class UsesFunc:
    def __call__(self, x):
        return ("self", type(self).__name__, x)


assert UsesFunc()(4) == ("self", "UsesFunc", 4)


# The keyword-call path must honour the same protocol as the positional one.
class CalleeKw:
    def __call__(self, *a, **k):
        return (a, k)


class UsesInstanceKw:
    __call__ = CalleeKw()


assert UsesInstanceKw()(1, x=2) == ((1,), {"x": 2})


class ResolvingDescriptorKw:
    def __get__(self, obj, owner):
        return lambda a, b=0: ("resolved", a, b)


class UsesDescriptorKw:
    __call__ = ResolvingDescriptorKw()


assert UsesDescriptorKw()(1, b=2) == ("resolved", 1, 2)


class HolderKw:
    def m(self, a, b=0):
        return ("m", a, b)


_hk = HolderKw()


class UsesBoundKw:
    __call__ = _hk.m


assert UsesBoundKw()(1, b=2) == ("m", 1, 2)


class UsesFuncKw:
    def __call__(self, a, b=0):
        return (type(self).__name__, a, b)


assert UsesFuncKw()(1, b=2) == ("UsesFuncKw", 1, 2)

print("OK")
