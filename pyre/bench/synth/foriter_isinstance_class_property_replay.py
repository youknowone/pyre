# `isinstance(obj, T)` first checks the object's real MRO and, on a miss,
# reads `obj.__class__`.  Inheriting `object.__getattribute__` does not make
# that read pure: a class can still override `__class__` with a property.
#
# `helper` is admitted into the surrounding FOR_ITER body.  The trailing
# opaque `id` call makes the first inline sub-walk abort and replay `helper`.
# If the `isinstance` call is incorrectly marked replay-safe, its property
# effect is not recorded and that one replay invokes it twice (N + 1 hits).

N = 5000
hits = [0]


class C:
    def __init__(self):
        self.x = 1

    @property
    def __class__(self):
        hits[0] += 1
        return str


def helper(obj):
    # Memoize that C inherits object.__getattribute__; this was the incomplete
    # proof used by the buggy replay-safe predicate.
    obj.x
    result = isinstance(obj, int)
    id(obj)
    return result


obj = C()
total = 0
for _ in range(N):
    total += helper(obj)

print(hits[0], total)
