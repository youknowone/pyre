a = list(map(str, [1, 2, 3]))
assert a == ["1", "2", "3"]


b = list(map(lambda x, y: x + y, [1, 2, 4], [3, 5]))
assert b == [4, 7]

# `map` is a lazy iterator (`W_Map`).
assert type(map(lambda x: x, [])).__name__ == "map"
m = map(str, [1])
assert iter(m) is m

# Laziness: the function runs only on demand.
calls = []


def doubler(x):
    calls.append(x)
    return x * 2


m = map(doubler, [1, 2, 3])
assert calls == []
assert next(m) == 2
assert calls == [1]

# __reduce__: (map, (func, *iterators)); the iterators carry their position.
m = map(str, [1, 2, 3])
recon, args = m.__reduce__()
assert args[0] is str
assert list(args[1]) == [1, 2, 3]
m = map(str, [1, 2, 3])
next(m)
assert list(m.__reduce__()[1][1]) == [2, 3]

# strict= (CPython 3.14) raises on a length mismatch.
assert list(map(lambda a, b: (a, b), [1, 2], [3, 4], strict=True)) == [(1, 3), (2, 4)]
try:
    list(map(lambda a, b: 0, [1, 2, 3], [4, 5], strict=True))
    raise AssertionError("no strict error")
except ValueError as e:
    assert str(e) == "map() argument 2 is shorter than argument 1", e


# test infinite iterator
class Counter(object):
    counter = 0

    def __next__(self):
        self.counter += 1
        return self.counter

    def __iter__(self):
        return self


it = map(lambda x: x + 1, Counter())
assert next(it) == 2
assert next(it) == 3


def mapping(x):
    if x == 0:
        raise StopIteration()
    return x


assert list(map(mapping, [1, 2, 0, 4, 5])) == [1, 2]
