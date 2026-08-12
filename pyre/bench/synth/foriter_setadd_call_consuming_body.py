# pyre-check: max-pypy-ratio=20
# The SET_ADD spelling of `foriter_listappend_call_consuming_body`. `heappop`
# consumes the very heap the comprehension is draining, so a body execution
# replayed or dropped by the tracer changes the heap, not just the result.
#
# The set is the weaker of the two checks here: it absorbs a duplicated pop
# silently, so `len(heap) == 0` is what carries the detection in this spelling
# — a replayed body leaves the heap short, a dropped one leaves it long.
#
# SET_ADD is admitted by the FOR_ITER body allow-list on the same footing as
# LIST_APPEND. Keeping both spellings lets a divergence between them show up as
# a fixture disagreement rather than as a silent difference in which
# accumulator the gate happens to cover.
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
