# CPython-suite gap: test_heapq's test_heapsort builds its data with a
# comprehension and then walks it in a second loop, and the failure surfaces as
# an IndexError out of heappop several statements later, which reads as a heapq
# defect rather than as a loop iteration whose tail ran against the next
# iteration's counters.
# parity-tests reason: a walk that does not commit hands its region back to a
# replay that re-executes it FROM THE WALK ENTRY, and the module namespace
# bindings the walk already wrote were the one part of the frame no journal
# restored — so the replayed statements ahead of the store read the walk's
# values while everything else still held the pre-walk ones.

"""A declined walk replays a module loop's tail against pre-walk bindings."""


def plus_one(i):
    return i + 1


# Two loops in one module frame. Tracing the second one runs a bridge off the
# tail of one iteration into the head of the next: the bridge consumes `trial`,
# binds `size` from it, enters the comprehension, and closes there on a merge
# point whose end-flush cannot resolve the mid-expression operand stack. The
# walk is handed to the replay, which re-runs the tail it already walked; before
# the replay reaches `size = trial % 50` it reads the `size` the walk stored,
# against the `data` the previous iteration built.
rows = []
for trial in range(2000):
    size = trial % 50
    data = [plus_one(i) for i in range(size)]
    total = 0
    for item in data:
        total += item
    rows.append((trial, size, len(data), total))

mixed = [row for row in rows if row[1] != row[2]]
print("iterations:", len(rows))
print("mixed-state rows:", mixed[:4])
assert not mixed, mixed
print("OK")
