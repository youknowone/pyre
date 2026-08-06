# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# `_py_abc.ABCMeta.__new__` validates every inherited abstract-method name
# through `getattr(cls, name, None)`: a non-string name raises TypeError.
import abc


class Base:
    __abstractmethods__ = [1]


try:
    class Derived(Base, metaclass=abc.ABCMeta):
        pass
except TypeError as exc:
    print(type(exc).__name__, exc)
else:
    raise AssertionError("ABCMeta accepted a non-string abstract-method name")
