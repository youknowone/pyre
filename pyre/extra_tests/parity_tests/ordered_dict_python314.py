"""Python 3.14 public OrderedDict contracts layered over PyPy app_odict."""

from collections import OrderedDict
import pickle


od = OrderedDict.fromkeys(iterable="abc", value=4)
assert list(od.items()) == [("a", 4), ("b", 4), ("c", 4)]
assert od.setdefault("d", default=5) == 5
assert od.pop(key="missing", default=6) == 6
assert repr(od) == "OrderedDict({'a': 4, 'b': 4, 'c': 4, 'd': 5})"

recursive = OrderedDict.fromkeys("a")
recursive["self"] = recursive
assert repr(recursive) == "OrderedDict({'a': None, 'self': ...})"

for view in (od.keys(), od.values(), od.items()):
    iterator = iter(view)
    next(iterator)
    expected = list(iterator)
    iterator = iter(view)
    next(iterator)
    assert list(pickle.loads(pickle.dumps(iterator))) == expected

print("OK")
