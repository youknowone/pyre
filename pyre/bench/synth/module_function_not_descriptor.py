# A function directly in a builtin module is a non-descriptor
# `builtin_function_or_method` (no `__get__`), so storing it on a user class
# and reading it through an instance does not synthesize a bound method and
# does not inject `self`.  Mirrors mixedmodule.py:_load_lazily; guards the
# demote_module_function_to_builtin pass at both call sites:
#   - builtins.rs new_builtin_module_dict   (the `builtins` module: len)
#   - importing.rs load_builtin_module      (every other builtin module:
#       sys.exc_info from a non-macro register_module; math.sqrt; time.sleep)
# A plain user `def` on a class is the control: it MUST still bind.
#
# A function that carries an instance attribute (populated `w_func_dict`) must
# NOT be demoted: `builtin_function_or_method` has no instance `__dict__`, so
# the retag would hide the attribute.  `itertools.chain` is a function-shaped
# constructor whose `from_iterable` alternate constructor is attached as such
# an attribute; demoting it dropped `chain.from_iterable`, which broke
# `import unittest.mock` -> asyncio -> dataclasses.  It stays binding here.
# Output verified against CPython 3.14.
import sys
import math
import time
import itertools


def check_non_binding(f, expected_name):
    assert type(f).__name__ == "builtin_function_or_method", type(f).__name__
    assert not hasattr(type(f), "__get__"), expected_name
    assert f.__name__ == expected_name, f.__name__

    class C:
        m = f

    inst = C()
    assert C.m is f
    assert inst.m is f
    assert type(inst.m).__name__ == "builtin_function_or_method"
    assert getattr(inst.m, "__self__", None) is not inst


def main():
    check_non_binding(len, "len")                # builtins module dict path
    check_non_binding(sys.exc_info, "exc_info")  # non-macro register_module path
    check_non_binding(math.sqrt, "sqrt")         # macro-module inline function
    check_non_binding(time.sleep, "sleep")       # non-macro register_module path

    class C:
        info = sys.exc_info

    assert C().info() == (None, None, None)      # no implicit self injected

    # A function carrying an attribute is left binding so the attribute
    # survives — `itertools.chain.from_iterable` must remain reachable.
    assert list(itertools.chain.from_iterable([[1, 2], [3], []])) == [1, 2, 3]
    assert list(itertools.chain.from_iterable(["ab", "c"])) == ["a", "b", "c"]

    class D:
        def g(self):
            return type(self).__name__

    d = D()
    assert type(D.g).__name__ == "function"
    assert hasattr(type(D.g), "__get__")
    assert type(d.g).__name__ == "method"
    assert d.g.__self__ is d
    assert d.g() == "D"

    print("OK")


main()
