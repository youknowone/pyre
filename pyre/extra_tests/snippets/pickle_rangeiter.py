assert iter(range(3)).__reduce__() == (iter, (range(0, 3),), None)
assert iter(range(3)).__length_hint__() == 3
it = iter(range(10))
next(it)
next(it)
next(it)
assert it.__reduce__() == (iter, (range(3, 10),), None), it.__reduce__()
assert it.__length_hint__() == 7
big = 10 ** 30
assert iter(range(big)).__reduce__() == (iter, (range(0, big),), None)
assert iter(range(big)).__length_hint__() == big
assert reversed(range(3)).__reduce__() == (iter, (range(2, -1, -1),), None)
import pickle
it = iter(range(5))
next(it)
assert list(pickle.loads(pickle.dumps(it))) == [1, 2, 3, 4]
print("pickle_rangeiter OK")
