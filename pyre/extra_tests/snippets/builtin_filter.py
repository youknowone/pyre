assert list(filter(lambda x: (x % 2) == 0, [0, 1, 2])) == [0, 2]

# None implies identity
assert list(filter(None, [0, 1, 2])) == [1, 2]

# `filter` is a lazy iterator (`W_Filter`).
assert type(filter(None, [])).__name__ == "filter"
f = filter(None, [1])
assert iter(f) is f

# Laziness: the predicate runs only on demand and only up to the first match.
calls = []


def even(x):
    calls.append(x)
    return x % 2 == 0


f = filter(even, [1, 2, 3, 4])
assert calls == []
assert next(f) == 2
assert calls == [1, 2]

# __reduce__ shape: (filter, (predicate_or_None, iterator)); the captured
# iterator carries its position so pickle/copy resume correctly.
f = filter(None, [1, 2, 3])
recon, args = f.__reduce__()
assert args[0] is None, args
assert list(args[1]) == [1, 2, 3]
assert filter(abs, [1]).__reduce__()[1][0] is abs


# test infinite iterator
class Counter(object):
    counter = 0

    def __next__(self):
        self.counter += 1
        return self.counter

    def __iter__(self):
        return self


it = filter(lambda x: (x % 2) == 0, Counter())
assert next(it) == 2
assert next(it) == 4


def predicate(x):
    if x == 0:
        raise StopIteration()
    return True


filtered = list(filter(predicate, [1, 2, 0, 4, 5]))
assert filtered == [1, 2], filtered
