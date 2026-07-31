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

print("OK")
