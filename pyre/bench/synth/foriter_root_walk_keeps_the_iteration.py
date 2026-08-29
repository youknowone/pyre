# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=loop_list,loop_tuple
# The same three-spelling identity as its `bridge_walk` sibling, but the
# iterator is built by the CALLER, so `GET_ITER` inside the loop's own frame
# returns it unchanged and a resume from the function entry re-reads the very
# cursor the walk advanced.
#
# The walk consumes an item, overflows `trace_limit`, and aborts.  The in-flight
# delivery can push the item back only when the live frame is parked at the
# FOR_ITER header; parked at the function entry it refuses, and that refusal is
# where the walk's journalled cursor is restored instead -- the resume then
# re-consumes the same item rather than reading the next one.
#
# `range` is the control: it is passed as a range OBJECT, so `GET_ITER` builds a
# fresh iterator inside the loop's frame and the resume rebuilds it, which was
# correct however the item was accounted for.
try:
    import pypyjit

    pypyjit.set_param("trace_limit=300")
    pypyjit.set_param("threshold=20")
except ImportError:
    pass

N = 4000


def loop_range(a, b, c, d, e, it):
    t = 0
    for _ in it:
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
        s += loop_range(i, i + 1, i + 2, i + 3, i + 4, range(4))
    return s

def loop_list(a, b, c, d, e, it):
    t = 0
    for _ in it:
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
        s += loop_list(i, i + 1, i + 2, i + 3, i + 4, iter([0, 1, 2, 3]))
    return s

def loop_tuple(a, b, c, d, e, it):
    t = 0
    for _ in it:
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
        s += loop_tuple(i, i + 1, i + 2, i + 3, i + 4, iter((0, 1, 2, 3)))
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
