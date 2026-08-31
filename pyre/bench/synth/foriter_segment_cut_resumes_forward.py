# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=loop_genexp,prefix_next
# A merge point reached with `force_finish_trace` armed and the recording past
# 0.8x `trace_limit` cuts the trace there and blackholes back to the
# interpreter rather than jumping into the segment it just compiled, "because
# we are at a really arbitrary place here".  Single-pass tracing has already RUN
# every iteration that segment records, so resuming from the walk's entry
# instead executes them a second time.
#
# Both loops take the iterator as a CALLER-built argument, so `GET_ITER` inside
# the loop frame returns it unchanged and the second execution reads a cursor
# the walk already moved:
#   * `loop_genexp` iterates a generator, which has no cursor to put back at
#     all -- the FOR_ITER journal cannot cover it, only the forward resume can.
#   * `prefix_next` consumes one item BEFORE the loop through a residual the
#     journal never sees; a header rewind restores only the FOR_ITER's own
#     advance, leaving the replayed `next` one item further along.
#
# `loop_range_object` is the control: a range OBJECT is rebuilt by the frame's
# own `GET_ITER`, so replaying that frame's entry was correct however the walked
# iterations were accounted for.
try:
    import pypyjit

    import sys

    # The 0.8x window is a raw-op-count property and the wasm build records
    # more raw ops per iteration than the native backends, so the two need
    # different limits: the sweep put the native window around 220-260 and
    # the wasm window at 355-360.
    pypyjit.set_param(
        "trace_limit=358" if sys.platform == "wasi" else "trace_limit=240"
    )
    pypyjit.set_param("threshold=20")
except ImportError:
    pass

N = 4000


def loop_range_object(a, b, c, d, e, it):
    t = 0
    for _ in it:
        t += a * 3 + b - c
        t += (a + b) * (c - d) + e
        t ^= (t >> 3) + (a | b) + (c & d)
        t += a * b + c * d + e * 7
        t -= (a - b) * (c + d) - e
        t += (t & 0xFF) * 3 + a + b + c + d + e
    return t


def run_range_object():
    s = 0
    for i in range(N):
        s += loop_range_object(i, i + 1, i + 2, i + 3, i + 4, range(4))
    return s


def loop_genexp(a, b, c, d, e, it):
    t = 0
    for _ in it:
        t += a * 3 + b - c
        t += (a + b) * (c - d) + e
        t ^= (t >> 3) + (a | b) + (c & d)
        t += a * b + c * d + e * 7
        t -= (a - b) * (c + d) - e
        t += (t & 0xFF) * 3 + a + b + c + d + e
    return t


def run_genexp():
    s = 0
    for i in range(N):
        s += loop_genexp(i, i + 1, i + 2, i + 3, i + 4, (x for x in range(4)))
    return s


def prefix_next(a, b, c, d, e, it):
    t = next(it)
    for _ in it:
        t += a * 3 + b - c
        t += (a + b) * (c - d) + e
        t ^= (t >> 3) + (a | b) + (c & d)
        t += a * b + c * d + e * 7
        t -= (a - b) * (c + d) - e
        t += (t & 0xFF) * 3 + a + b + c + d + e
    return t


def run_prefix_next():
    s = 0
    for i in range(N):
        s += prefix_next(i, i + 1, i + 2, i + 3, i + 4, iter([0, 1, 2, 3, 4]))
    return s


totals = {
    "range_object": run_range_object(),
    "genexp": run_genexp(),
    "prefix_next": run_prefix_next(),
}
expected = totals["range_object"]
for name, got in totals.items():
    assert got == expected, f"{name} totalled {got}, expected {expected}"
print("PASS")
