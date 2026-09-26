# pyre-check: max-pypy-ratio=11.8
# Measured on darwin/arm64 (user time, best of 3): dynasm 0.14s, cranelift
# 0.23s, pypy 0.03s, i.e. 4.7x and 7.7x. The ceiling is about 1.5x the
# slower backend's ratio. pypy's 0.03s is near the timer resolution and sits
# under FLOOR_GATE_MIN_BASELINE_S, so the floor gate does not judge this
# fixture.
# The class entry is an ObjectMutableCell before the loop: the first store
# builds the cell (`typeobject.py write_cell`) and the hot `__next__` has to
# promote `ObjectMutableCell.w_value` instead of declining `next_fast_path`.
N = 3557424


class It:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i


def _n(self):
    if self.i >= self.n:
        raise StopIteration
    self.i += 1
    return self.i


# Rebind once so the class entry becomes a cell before the loop compiles.
It.__next__ = _n

total = 0
# 20 steps per iterator, matching cr_next.py's inner length.
steps = 20
for _r in range(N // steps):
    for x in It(steps):
        total += x
print(total)
