e = enumerate([10, 20, 30])
r = e.__reduce__()
assert r[0] is enumerate and r[1][1] == 0 and list(r[1][0]) == [10, 20, 30], r
e = enumerate([10, 20, 30])
next(e)
assert e.__reduce__()[1][1] == 1
import pickle
e = enumerate([10, 20, 30])
next(e)
assert list(pickle.loads(pickle.dumps(e))) == [(1, 20), (2, 30)]
e = enumerate([10, 20], start=5)
assert e.__reduce__()[1][1] == 5
print("pickle_enumerate OK")
