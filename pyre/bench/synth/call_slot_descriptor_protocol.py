"""Regression: __call__ resolved as a class attribute honours the descriptor
protocol — only a plain function binds the receiver as self; a non-binding
builtin, a callable instance, or an already-bound method is called without an
implicit self.  typing.NewType uses `__call__ = _typing._idfunc` (a non-binding
builtin), so NewType(...)(x) must return x, not the NewType instance.
"""
import _typing
import typing

NON_BINDING = ("builtin_function_or_method", "builtin_function")
assert type(_typing._idfunc).__name__ in NON_BINDING, type(_typing._idfunc).__name__

UserId = typing.NewType("UserId", int)
assert UserId(5) == 5, UserId(5)
marker = ["marker"]
assert UserId(marker) is marker

class UsesAbs:
    __call__ = abs
assert UsesAbs()(-7) == 7

class UsesLen:
    __call__ = len
assert UsesLen()([1, 2, 3]) == 3

class Callee:
    def __call__(self, *a):
        return a
class UsesInstance:
    __call__ = Callee()
assert UsesInstance()(1, 2) == (1, 2)

class Holder:
    def m(self, x):
        return ("m", x)
_h = Holder()
class UsesBound:
    __call__ = _h.m
assert UsesBound()(9) == ("m", 9)

class UsesFunc:
    def __call__(self, x):
        return ("self", type(self).__name__, x)
assert UsesFunc()(4) == ("self", "UsesFunc", 4)

print("OK")
