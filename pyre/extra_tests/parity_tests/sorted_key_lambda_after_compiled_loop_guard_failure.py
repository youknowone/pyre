# CPython-suite gap: test_sort / test.test_heapq nsmallest keyed sort.
# After a compiled sortkey loop, leftover execute_frame of the key must
# restore the recursion slots from frame-locals; a shadow-stack pin left
# above the blackhole arg-list span panics ListRootGuard depth=end+1.
# parity-tests reason: two keyed sorts of 200 tuples; the second hits
# compiled sortkey blackhole resume of the key lambda.

"""A keyed sort after a compiled sortkey loop returns the sorted list."""

data = [(i % 17, i) for i in range(200)]
xs = sorted(data, key=lambda x: x[0])
assert xs == sorted(data, key=lambda x: x[0])
print("OK")
