d = {"a": 1, "b": 2, "c": 3}
assert iter(d.keys()).__reduce__() == (iter, (["a", "b", "c"],)), iter(d.keys()).__reduce__()
assert iter(d.keys()).__length_hint__() == 3
ki = iter(d.keys())
next(ki)
assert ki.__reduce__() == (iter, (["b", "c"],))
assert iter(d.values()).__reduce__() == (iter, ([1, 2, 3],))
assert iter(d.items()).__reduce__() == (iter, ([("a", 1), ("b", 2), ("c", 3)],))
import pickle
ki = iter(d.keys())
next(ki)
assert list(pickle.loads(pickle.dumps(ki))) == ["b", "c"]
print("pickle_dictiter OK")
