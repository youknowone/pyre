# Guard-failure blackhole resume across a LOAD_ATTR (issue #143 guard-12
# class).  The rare branch fails its guard until the bridge threshold, and
# each failure resumes the rest of the iteration through `c.v` — the
# jitcode position that used to carry `abort_permanent` (which crashed the
# blackhole with an out-of-bounds register read) and is now lowered to the
# `load_attr_fn` residual call by the canonical splice.
class C:
    def __init__(self):
        self.v = 3


def run(c, n):
    total = 0
    i = 0
    while i < n:
        if i % 37 == 36:
            k = 2
        else:
            k = 1
        total += c.v * k
        i += 1
    return total


print(run(C(), 300000))
