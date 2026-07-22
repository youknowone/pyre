import copy
import gc
import operator
import weakref


class Value:
    pass


a, b, c, d = [Value() for _ in range(4)]
original = weakref.WeakValueDictionary({a: b, c: d})
cloned = copy.copy(original)
del c, d
gc.collect()
gc.collect()
gc.collect()
assert len(original) == 1
assert len(cloned) == 1


left = []
left.append(left)
right = copy.deepcopy(left)
assert right is not left
assert right[0] is right
for comparison in (
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
):
    try:
        comparison(left, right)
    except RecursionError:
        pass
    else:
        raise AssertionError("recursive list comparison did not stop")


print("stdlib copy ok")
