from operator import itemgetter


# Warm the first arm of itemgetter.__call__, then enter its inlined list
# comprehension through the other branch.  A guard abort at FOR_ITER must
# resume with both the MIFrame header stack and the item already consumed by
# the authoritative walk; otherwise the first comprehension element is lost.
single = itemgetter(0)
value = ("B", -260)
for _ in range(20_000):
    single(value)

multiple = itemgetter(1, 0)
for _ in range(20_000):
    assert multiple(value) == (-260, "B")

print("OK")
