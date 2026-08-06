# pyre-check: max-pypy-ratio=25
# Changing an unboxed int or float slot to the other type creates a boxed map;
# the promoted-map guard must deopt before the old raw-storage read or write is
# reused for the new representation (mapdict.py:577-584, 600-619, 905-916).
#
# Only `run_float` covers that on a 32-bit target. `ALLOW_UNBOXING_INTS` is
# `LONG_BIT == 64` (mapdict.py:30), so `run_int`'s slot is never unboxed there
# and the read never folds — which is why the wasm baseline differs from the
# native two: with no unboxed fold the trace pins no `Terminator.allow_unboxing`
# quasi-immutable, so demoting the map invalidates nothing, the compiled loop
# stays runnable past the type change, and its guard is bridged instead of the
# loop being retraced. Measured: `run_float` alone is identical on all three
# backends; `run_int` alone reads 1 loop / 1 bridge / 202 guard failures under
# the guest against 2 / 0 / 2 on both native backends.
class IntSlot:
    def __init__(self):
        self.x = 0


class FloatSlot:
    def __init__(self):
        self.x = 0.0


def run_int(n):
    obj = IntSlot()
    total = 0
    i = 0
    while i < n:
        obj.x = obj.x + 1
        if i == n // 2:
            obj.x = 1.5
        total += int(obj.x)
        i += 1
    return total, obj.x


def run_float(n):
    obj = FloatSlot()
    total = 0.0
    i = 0
    while i < n:
        obj.x = obj.x + 1.0
        if i == n // 2:
            obj.x = 7
        total += float(obj.x)
        i += 1
    return total, obj.x


print(run_int(100000))
print(run_float(100000))
