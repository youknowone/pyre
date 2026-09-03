# pyre-check: max-pypy-ratio=40
# A `BINARY_OP` / `COMPARE_OP` dunder that carries a defaulted parameter past
# the two the operands bind.
#
# `synth/binop_dunder_leaf_inline` covers the two-parameter shape.  This one is
# the same admitted body with `def __add__(self, o, context=None)`, which the
# entry used to refuse outright on its arity: the operands bind `self` and `o`
# and nothing bound `context`, so the route declined before it ever looked at
# the body.  The tail is not unbindable -- `funccall_valuestack` fills every
# parameter a call leaves unbound from `defs_w`, and the resolved descent this
# entry already delegates to seeds exactly that from `__defaults__` -- so the
# arity test was standing in for a decision the descent makes itself.
#
# The shape is not incidental.  `_pydecimal` writes every arithmetic operator
# as `(self, other, context=None)`, so the refusal covered `Decimal.__add__`,
# `__mul__`, `__sub__` and the rest with no other cause needed.
#
# The ceiling is what gates the admission: the same body without the defaulted
# parameter is the admitted side of `binop_dunder_leaf_inline`, so losing this
# puts a residual interpreter frame back on every `+`, `*` and `<` below.
#
# Deterministic, terminating, prints an int checksum; jit == nojit.
M = 1000000007
N = 14400000


class Ctx:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __add__(self, o, context=None):
        return self.x + o

    def __mul__(self, o, context=None):
        return self.x * o + 1

    def __lt__(self, o, context=None):
        return self.x < o


def fast(n):
    p = Ctx(0)
    acc = 0
    hits = 0
    for i in range(n):
        p.x = i % 9173
        acc = (acc + (p + i) + (p * 3)) % M
        if p < 4096:
            hits += 1
    return acc, hits


a, h = fast(N)
print(a, h)
