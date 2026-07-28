import itertools


assert list(itertools.accumulate(range(5))) == [0, 1, 3, 6, 10]
assert list(itertools.accumulate(iterable=range(4))) == [0, 1, 3, 6]
assert list(itertools.accumulate([2, 3, 4], lambda a, b: a * b)) == [2, 6, 24]
assert list(itertools.accumulate([2, 3], initial=10)) == [10, 12, 15]

try:
    itertools.accumulate(range(3), None, 10)
except TypeError:
    pass
else:
    raise AssertionError("accumulate accepted initial as a positional argument")

try:
    itertools.accumulate(range(3), unknown=10)
except TypeError:
    pass
else:
    raise AssertionError("accumulate accepted an unknown keyword")

print("OK")
