# pyre-check: max-pypy-ratio=3
# The ceiling gates cranelift as well as dynasm, and `perf_gate_floor` derives
# a floor from it as ceiling/6, so both ends of the reading spread pick it. Run
# 33384229844 reads 0.7x (macos dynasm), 1.4x (macos cranelift), 1.6x and 2.3x
# (ubuntu) on the four pairs where pypy's baseline was measurable -- wasm is
# ungated and windows read a clamped baseline. 4x derived a floor of 0.67x,
# 5% under the fast end and inside that reading's own rounding; 3x clears the
# slow end by 30% and the fast end by 29%.
# pyre-check: skip-cpython
# cpython >5s (it already timed out) vs pyre 0.20s, and it is not gated on — only pypy is.
# STORE_ATTR that ADDS an attribute not yet in the instance's map: the
# `map -> PlainAttribute` transition plus the grow-by-one storage rewrite
# (mapdict.py:942-959 `_set_mapdict_increase_storage1`).  The values are
# strings so every slot stays boxed, which is the shape the JIT emits inline;
# an unboxed slot keeps the general residual.
#
# The hot loop holds the two folded shapes — the first attribute of a fresh
# instance (empty storage block) and a second attribute (the live slot is
# copied into the wider block) — so the ratio measures the fold itself.  The
# shapes the fold declines or that force materialization are exercised once
# each in `edges()`, for agreement with the interpreter rather than for speed.
#
# `N` is sized so the measured time clears the ratio's noise floor: check.py
# subtracts each interpreter's empty-program startup, and below roughly ten
# million iterations what is left on the pypy side is startup jitter rather
# than the loop.  `total` accumulates a fixed amount per iteration, so it
# scales exactly with `N`; dividing by the ratio to `OUTPUT_N` keeps stdout
# stable across resizings, which is what every interpreter's output is
# compared against.
OUTPUT_N = 200000
N = 119172400


class Fresh:
    pass


class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Either:
    pass


def edges():
    out = []
    # `_reorder_and_add` (mapdict.py:204-258): both orders off one class, so
    # the second instance finds the attribute above its own map.
    for first in (0, 1):
        obj = Either()
        if first:
            obj.p = "p"
            obj.q = "qq"
        else:
            obj.q = "qq"
            obj.p = "p"
        out.append(sorted(vars(obj).items()))
    # An instance the loop keeps has to materialize instead of staying virtual.
    kept = []
    for i in range(4):
        obj = Pair("a" * i, "b")
        kept.append(obj)
    out.append([(o.a, o.b) for o in kept])
    # Add, delete, re-add: the second add starts from the popped-back map.
    obj = Fresh()
    obj.x = "1"
    del obj.x
    obj.x = "2"
    out.append((obj.x, sorted(vars(obj).items())))
    return out


def main():
    total = 0
    i = 0
    while i < N:
        # first attribute: storage grows 0 -> 1
        one = Fresh()
        one.x = "x"
        total = total + len(one.x)

        # second attribute: the existing slot is copied into the wider block
        two = Pair("ab", "cde")
        total = total + len(two.a) + len(two.b)

        i = i + 1
    print(total // (N // OUTPUT_N), edges())


main()
