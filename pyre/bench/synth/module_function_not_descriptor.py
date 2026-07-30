# A function directly in a builtin module is a non-descriptor builtin function
# (no `__get__`), so storing it on a user class and reading it through an
# instance returns the SAME function object — it does not synthesize a bound
# method or inject `self`.  Mirrors mixedmodule.py:_load_lazily; guards
# demote_module_function_to_builtin at both call sites:
#   - builtins.rs new_builtin_module_dict   (the `builtins` module: len)
#   - importing.rs load_builtin_module      (every other builtin module:
#       sys.exc_info from a non-macro register_module; math.sqrt; time.sleep)
# A function that carries an instance attribute (populated `w_func_dict`) must
# NOT be demoted (`builtin_function_or_method` has no instance `__dict__`, so
# the retag would hide the attribute): `itertools.chain` keeps its
# `from_iterable` alternate constructor, whose loss broke `import unittest.mock`
# -> asyncio -> dataclasses.  A plain user `def` is the binding control.
#
# The exact builtin type name differs across interpreters (CPython/pyre 3.14
# `builtin_function_or_method` vs PyPy `builtin_function`), so this asserts the
# non-binding BEHAVIOUR, which is invariant.  Output verified on CPython 3.14,
# PyPy, and pyre.
import sys
import math
import time
import itertools

BUILTIN_FN_NAMES = ("builtin_function_or_method", "builtin_function")


def check_non_binding(f):
    tname = type(f).__name__
    assert tname in BUILTIN_FN_NAMES, tname
    assert not hasattr(type(f), "__get__"), tname

    class C:
        m = f

    inst = C()
    assert C.m is f            # class access does not transform a non-descriptor
    assert inst.m is f         # instance access does NOT bind (the key property)
    assert getattr(inst.m, "__self__", None) is not inst


def main():
    check_non_binding(len)           # builtins module dict path
    check_non_binding(sys.exc_info)  # non-macro register_module path
    check_non_binding(math.sqrt)     # macro-module inline function
    check_non_binding(time.sleep)    # non-macro register_module path

    class C:
        info = sys.exc_info

    assert C().info() == (None, None, None)      # no implicit self injected

    # `itertools.chain` carries `from_iterable` as an attribute, so it is left
    # binding rather than demoted — the attribute must stay reachable.
    assert list(itertools.chain.from_iterable([[1, 2], [3], []])) == [1, 2, 3]
    assert list(itertools.chain.from_iterable(["ab", "c"])) == ["a", "b", "c"]

    # a plain user `def` IS a descriptor and still binds through an instance
    class D:
        def g(self):
            return type(self).__name__

    d = D()
    assert hasattr(type(D.g), "__get__")
    assert d.g is not D.g            # instance access binds -> a distinct object
    assert d.g.__self__ is d
    assert d.g() == "D"

    print("OK")


main()
