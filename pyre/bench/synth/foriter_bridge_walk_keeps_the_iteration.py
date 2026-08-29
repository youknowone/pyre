# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=loop_list,loop_tuple
# Three spellings of one loop over four values.  They differ only in the
# expression the `for` iterates, so the three totals must agree.
#
# `trace_limit` is sized so one traced iteration overflows it and the walk
# aborts with the iterator already advanced.  A bridge/retrace recording that
# does not commit restores the cursor it advanced eagerly, but only the range
# and zip FOR_ITER specializations journalled theirs: the list and tuple
# iterators reach the generic `for_iter_next` residual, so nothing was recorded
# and the epilogue restored nothing.  The consumed item was then handed to
# nobody -- neither the forward adopt nor the in-flight delivery runs on that
# leg -- and the loop ran one iteration short.
#
# `range` is the control: its fold journals its own advance, so it was correct
# throughout.
try:
    import pypyjit

    pypyjit.set_param("trace_limit=300")
    pypyjit.set_param("threshold=20")
except ImportError:
    pass

N = 4000


def loop_range(a, b, c, d, e):
    t = 0
    for _ in range(4):
        t += a * 3 + b - c
        t += (a + b) * (c - d) + e
        t ^= (t >> 3) + (a | b) + (c & d)
        t += a * b + c * d + e * 7
        t -= (a - b) * (c + d) - e
        t += (t & 0xFF) * 3 + a + b + c + d + e
    return t


def run_range():
    s = 0
    for i in range(N):
        s += loop_range(i, i + 1, i + 2, i + 3, i + 4)
    return s

def loop_list(a, b, c, d, e):
    t = 0
    for _ in [0, 1, 2, 3]:
        t += a * 3 + b - c
        t += (a + b) * (c - d) + e
        t ^= (t >> 3) + (a | b) + (c & d)
        t += a * b + c * d + e * 7
        t -= (a - b) * (c + d) - e
        t += (t & 0xFF) * 3 + a + b + c + d + e
    return t


def run_list():
    s = 0
    for i in range(N):
        s += loop_list(i, i + 1, i + 2, i + 3, i + 4)
    return s

def loop_tuple(a, b, c, d, e):
    t = 0
    for _ in (0, 1, 2, 3):
        t += a * 3 + b - c
        t += (a + b) * (c - d) + e
        t ^= (t >> 3) + (a | b) + (c & d)
        t += a * b + c * d + e * 7
        t -= (a - b) * (c + d) - e
        t += (t & 0xFF) * 3 + a + b + c + d + e
    return t


def run_tuple():
    s = 0
    for i in range(N):
        s += loop_tuple(i, i + 1, i + 2, i + 3, i + 4)
    return s

totals = {
    "range": run_range(),
    "list": run_list(),
    "tuple": run_tuple(),
}
expected = totals["range"]
for name, got in totals.items():
    assert got == expected, f"{name} totalled {got}, expected {expected}"
print("PASS")
