# pyre-check: max-pypy-ratio=40
# A `BINARY_OP` / `COMPARE_OP` whose dunder is a call-free Python body.
#
# The walker used to refuse every one of these.  The refusal was not about the
# operand or the class: it was that a `NotImplemented` result has to go back to
# the full binary protocol, and with no rewind at this entry the walk discarded
# it with a bare `Err`, which leaves the driver replaying the loop with the
# body's effects already applied.  So admission asked for a whole-body `Clean`
# verdict -- and `LoadAttr` and `BinaryOp` are both deferred helpers, which
# makes `return self.x + o` `DeferredCall` and refuses the simplest dunder
# there is.  The residual that stood in its place runs a whole interpreter
# frame per iteration.
#
# What the loops below cover is the shape that IS admitted now: a body that
# cannot name `NotImplemented` and calls nothing.  The second half of that is
# what `Fraction.__gt__` fails -- its whole body is
# `return a._richcmp(b, operator.gt)` -- and admitting it put five aborts
# through the unstaged-reason fallback on `synth/inline_freevar_after_mayforce`,
# each one dropping its iteration's contribution, which is the fixture that
# guards that half.  `slow` below is the same delegating shape and must stay
# residual; it is here so the two sit side by side, not because it gates.
#
# The ceiling is what gates the admission.  Losing it puts `fast`'s `+`, `*`
# and `<` back on a residual frame apiece: one binary read 0.21s admitted
# against 3.95s declined, which is 20x pypy against roughly 500x, so the
# ceiling sits an order of magnitude below the declined reading with 2x of
# headroom above the admitted one.  No jit-stats band is needed to read it.
#
# pypy's own execution stays under `FLOOR_GATE_MIN_BASELINE_S` however large N
# gets -- it folds this loop to about 0.7ns an iteration -- so the run carries
# the `?` that says the floor gate declined the baseline as too small to judge.
# The ceiling is still applied, and the ceiling is the whole instrument here.
#
# Deterministic, terminating, prints an int checksum; jit == nojit.
import operator

M = 1000000007
# Sized so pypy's own execution clears `FLOOR_GATE_MIN_BASELINE_S`: below it
# the floor gate declines the baseline as too small to judge and the ratio
# reports startup rather than these loops.
N = 1600000


class Leaf:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __add__(self, o):
        return self.x + o

    def __mul__(self, o):
        return self.x * o + 1

    def __lt__(self, o):
        return self.x < o


class Delegating:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def _cmp(self, o, op):
        return op(self.x, o)

    def __lt__(self, o):
        return self._cmp(o, operator.lt)


def fast(n):
    """The admitted shape: every dunder body reads a slot and calls nothing."""
    p = Leaf(0)
    acc = 0
    hits = 0
    for i in range(n):
        p.x = i % 9173
        acc = (acc + (p + i) + (p * 3)) % M
        if p < 4096:
            hits += 1
    return acc, hits


def slow(n):
    """The delegating shape, which stays residual."""
    q = Delegating(0)
    hits = 0
    for i in range(n):
        q.x = i % 9173
        if q < 4096:
            hits += 1
    return hits


a, h = fast(N)
print(a, h, slow(20000))
