# CPython-suite gap: `test_context` builds contexts, runs them and copies
# them; it never asks the class what it is.  `test_pickle` pickles values, not
# the runtime's own types.  So nothing there reads `Context.__module__`, and a
# wrong answer costs nothing until someone prints the class, prints an
# instance, or pickles either.
#
# The subject is a class two runtimes build two different ways.  It is a C
# static type on CPython (`PyContext_Type`, `tp_name` `"_contextvars.Context"`)
# and an ordinary Python class on PyPy (`lib_pypy/_contextvars.py`), and pyre
# writes it in Python while projecting CPython's public type flags onto it --
# `Context.__flags__` reports no heap bit and the class refuses attribute
# assignment, both of which this suite's callers can see.  A runtime with two
# separate notions of "this is a class" can answer the flag question one way
# and the where-is-`__module__`-stored question the other, and then the class
# reports a module it was never defined in.
#
# parity-tests reason: `__module__` is not read on its own -- three things
# derive from it, and all three are what a user sees rather than what a test
# asserts.  `repr` of the class, `repr` of an instance, and the reference
# `pickle` writes and resolves.  The pickle arm is the one that raises; the
# other two silently print a name that does not identify anything.
#
# The flags themselves are deliberately not pinned here: CPython reports no
# heap bit and PyPy reports one, because PyPy really does define the class in
# Python.  What both agree on is everything below.
import pickle
import sys

import _contextvars
import contextvars


def the_class_reports_the_module_that_defines_it():
    assert _contextvars.Context.__module__ == '_contextvars', (
        _contextvars.Context.__module__)
    # The other half: the name stays unqualified.  A runtime that repaired the
    # module by qualifying the type's name instead would move this too.
    assert _contextvars.Context.__name__ == 'Context', _contextvars.Context.__name__
    assert _contextvars.Context.__qualname__ == 'Context', (
        _contextvars.Context.__qualname__)


def both_reprs_name_the_module():
    assert repr(_contextvars.Context) == "<class '_contextvars.Context'>", (
        repr(_contextvars.Context))
    instance = repr(contextvars.copy_context())
    assert instance.startswith('<_contextvars.Context object at 0x'), instance


def the_class_pickles_by_reference():
    # `pickle` writes the class as its module plus its qualified name and
    # resolves it back through `sys.modules`, so a module the class was not
    # defined in is a `PicklingError` rather than a cosmetic slip.
    restored = pickle.loads(pickle.dumps(_contextvars.Context))
    assert restored is _contextvars.Context, restored
    module = sys.modules[_contextvars.Context.__module__]
    assert getattr(module, 'Context') is _contextvars.Context


def the_module_it_reports_is_the_one_it_is_reached_through():
    # `contextvars` re-exports the accelerator's class rather than defining
    # its own, so the owner the class claims has to be the accelerator.
    assert contextvars.Context is _contextvars.Context


the_class_reports_the_module_that_defines_it()
both_reprs_name_the_module()
the_class_pickles_by_reference()
the_module_it_reports_is_the_one_it_is_reached_through()
print('OK')
