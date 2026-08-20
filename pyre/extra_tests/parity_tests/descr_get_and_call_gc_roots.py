# CPython-suite gap: no test binds a binary special to a descriptor whose
# __get__ collects.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""A special method bound through a descriptor must survive its own ``__get__``.

``get_and_call_function`` receives the raw class attribute and the operand
list.  When that attribute is a descriptor rather than a plain function, the
``__get__`` dispatch is application-level Python and can run a collection; the
receiver and the argument slice — the latter living on the Rust stack, where no
root reaches it — must therefore be read back afterwards rather than reused.
"""

import gc


def _churn():
    # Enough allocation to fill a nursery and force a moving collection while
    # the caller's operand addresses are the only copies in existence.
    garbage = [[index] for index in range(20000)]
    assert len(garbage) == 20000
    gc.collect()


class BoundThroughDescriptor:
    """A non-function class attribute, so the descriptor arm of `get` runs."""

    def __init__(self, impl):
        self.impl = impl

    def __get__(self, obj, objtype=None):
        _churn()
        impl = self.impl
        return lambda *args: impl(obj, *args)


def _take_rhs(self, other):
    return other


def _take_rhs_inplace(self, other):
    return other


class Operand:
    __add__ = BoundThroughDescriptor(_take_rhs)
    __iadd__ = BoundThroughDescriptor(_take_rhs_inplace)


for round_index in range(20):
    # A `list` right-hand side is a kind whose header a minor collection
    # relocates, so a stale operand address is observable rather than benign.
    rhs = [[value] for value in range(round_index + 1)]
    expected = [[value] for value in range(round_index + 1)]

    got = Operand() + rhs
    assert got is rhs, (round_index, "binary special lost its operand")
    assert got == expected, (round_index, got)

    accumulator = Operand()
    accumulator += rhs
    assert accumulator is rhs, (round_index, "in-place special lost its operand")
    assert accumulator == expected, (round_index, accumulator)

print("OK")
