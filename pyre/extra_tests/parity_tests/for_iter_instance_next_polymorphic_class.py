# CPython-suite gap: the suite does not alternate two user iterator classes at
# one hot FOR_ITER site.
# parity-tests reason: user instances share one physical layout, so pinning the
# layout alone would let a second class run the first class's __next__.


class Ascending:
    def __init__(self, limit):
        self.index = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        return self.index


class Descending:
    def __init__(self, limit):
        self.index = limit
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.index <= 0:
            raise StopIteration
        self.index -= 1
        return self.index


def collect(iterator):
    seen = []
    for value in iterator:
        seen.append(value)
    return seen


limit = 1600
ascending = list(range(1, limit + 1))
descending = list(range(limit - 1, -1, -1))

# Warm the site on one class alone so the loop compiles against it, then keep
# feeding both through the same FOR_ITER.
for _ in range(8):
    assert collect(Ascending(limit)) == ascending

for _ in range(8):
    assert collect(Descending(limit)) == descending
    assert collect(Ascending(limit)) == ascending

# A third class whose __next__ returns a different type must not inherit either
# body: a stale method would return ints here.
class Tagging:
    def __init__(self, limit):
        self.index = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.limit:
            raise StopIteration
        self.index += 1
        return "t%d" % self.index


for _ in range(8):
    tagged = collect(Tagging(4))
    assert tagged == ["t1", "t2", "t3", "t4"], tagged

print("OK")
