# pyre-check: max-pypy-ratio=20
# A set comprehension whose element call consumes the very heap it is draining,
# so a body execution replayed or dropped by the tracer changes the heap, not
# just the result.
#
# The set is the weaker of the two checks here: it absorbs a duplicated pop
# silently, so `len(heap) == 0` is what carries the detection — a replayed body
# leaves the heap short, a dropped one leaves it long.
#
# `for_iter_body_is_jit_safe_at` admits SET_ADD with no condition on what else
# the body does, so a call-bearing set comprehension is traced. The LIST_APPEND
# spelling of the same loop is declined by that gate's CALL scan, which is why
# only the set spelling is pinned here.
import heapq

N = 300
SIZE = 128


def lcg(seed, n):
    # Deterministic data, so the fixture's output is a constant rather than a
    # property of the host's hash seed or clock.
    out = []
    x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x % 100000)
    return out


total = 0
for round_ in range(N):
    data = lcg(round_ + 1, SIZE)
    heap = list(data)
    heapq.heapify(heap)
    size = len(heap)
    # `range(size)` is fixed before the body runs, so the body executes exactly
    # `size` times unless the tracer replays one.
    drained = {heapq.heappop(heap) for _ in range(size)}
    assert len(heap) == 0, 'heap not drained: %d left' % len(heap)
    assert drained == set(data), 'set comprehension lost or invented an item'
    total += len(drained)

print(total)
