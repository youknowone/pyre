# pyre-check: max-pypy-ratio=40
# A `BINARY_OP` dunder whose body builds an instance of its own class and
# reads a field back off it.
#
# `synth/binop_dunder_leaf_inline` covers the body that only reads its two
# operands.  This one is `return Pair(self.x + o).x`, which three separate
# refusals used to stack against:
#
#   * the entry refused any body making a nested Python call at all, on a
#     recorded `phase2 snapshot remap cache miss` that no longer reproduces;
#   * `try_walker_inline_type_call` refused every instantiation appearing
#     inside an inline sub-walk, so `Pair(...)` stayed a residual; and
#   * the rewind region then refused that residual's `__init__` slot write as
#     an unjournaled commit it could not undo.
#
# The last one is the interesting bar: the write goes to an object the region
# itself allocated, so the record-time cut leaves it unreachable and there is
# nothing to undo.  A write to anything else is still refused.
#
# The shape is what every value type in the pure-Python stdlib is made of --
# `date.__add__` returns a `date`, `Decimal.__add__` returns a `Decimal` -- so
# losing this puts a residual construction plus an interpreted `__init__` frame
# back inside every arithmetic operator those modules define.
#
# Deterministic, terminating, prints an int checksum; jit == nojit.
M = 1000000007
N = 1600000


class Pair:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __add__(self, o):
        return Pair(self.x + o).x

    def __mul__(self, o):
        return Pair(self.x * o + 1).x


def fast(n):
    p = Pair(0)
    acc = 0
    for i in range(n):
        p.x = i % 9173
        acc = (acc + (p + i) + (p * 3)) % M
    return acc


print(fast(N))
