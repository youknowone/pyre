# pyre-check: max-pypy-ratio=20
# An input-CONSUMING call inside a comprehension: `heappop` mutates the very
# list the comprehension is draining, so one body execution too many raises
# IndexError and one too few leaves an element behind. Both directions fail
# loudly, in the same assert.
#
# The shape is the one the FOR_ITER gate declined until the call-bearing
# LIST_APPEND admission: a mid-body abort with a committed body effect reaches
# `fbw_foriter_inflight_take`, whose refusal arm drops the consumed item.
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
    sorted_out = [heapq.heappop(heap) for _ in range(size)]
    assert len(heap) == 0, 'heap not drained: %d left' % len(heap)
    assert sorted_out == sorted(data), 'comprehension did not sort its input'
    total += len(sorted_out)

print(total)
