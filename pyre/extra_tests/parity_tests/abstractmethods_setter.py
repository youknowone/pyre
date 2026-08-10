# CPython-suite gap: no abstract-method setter test collects during truth testing.
# parity-tests reason: this specifically guards pyre/PyPy moving-GC callback roots.

"""`type.__abstractmethods__` survives a `__bool__` that collects.

The setter truth-tests the value before storing it, and that test runs the
value's own `__bool__`, so both the receiver and the value have to outlive it.
A class whose marker is not a container reports the iteration's own TypeError
when instantiated, the way `objectobject.py:17-21 _abstract_method_error`
reaches the attribute through `sorted(...)`.
"""

import gc


class Churn:
    """Truthy, and allocates heavily while answering."""

    def __init__(self):
        self.junk = []

    def __bool__(self):
        self.junk = [[object() for _ in range(48)] for _ in range(32)]
        gc.collect()
        return True


for step in range(300):
    cls = type("C%d" % step, (), {})
    marker = frozenset({"m%d" % step})
    cls.__abstractmethods__ = marker
    assert cls.__abstractmethods__ == marker, (step, cls.__abstractmethods__)
    assert cls.__name__ == "C%d" % step, step
    try:
        cls()
    except TypeError as error:
        assert "m%d" % step in str(error), error
    else:
        raise AssertionError("an abstract class must not instantiate")

for step in range(120):
    cls = type("D%d" % step, (), {})
    churn = Churn()
    cls.__abstractmethods__ = churn
    assert cls.__abstractmethods__ is churn, step
    assert cls.__name__ == "D%d" % step, step
    try:
        cls()
    except TypeError as error:
        assert str(error) == "'Churn' object is not iterable", error
    else:
        raise AssertionError("a non-container marker still blocks instantiation")

deletable = type("E", (), {})
deletable.__abstractmethods__ = frozenset({"x"})
del deletable.__abstractmethods__
assert deletable() is not None

print("OK")
