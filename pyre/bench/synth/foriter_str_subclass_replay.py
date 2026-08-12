# The FOR_ITER inline replay exemption for `str(value)` applies only to exact
# immutable builtin scalars.  A str subclass can override `__str__` and mutate
# visible state.  The trailing opaque `id` call makes the first inline walk
# abort; treating the subclass as exact replays that override once (N + 1).

N = 5000
hits = [0]


class S(str):
    def __str__(self):
        hits[0] += 1
        return "x"


def helper(value):
    result = str(value)
    id(value)
    return len(result)


value = S("abc")
total = 0
for _ in range(N):
    total += helper(value)

print(hits[0], total)
